/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"

namespace cutlass::fmha::collective {

using namespace cute;

// SM120 FMHA epilogue: convert float accumulators to output type, store via SMEM to GMEM.
//
// This is simpler than SM100 because SM120 keeps all accumulators in registers
// (no TMEM step). The flow is:
//   1. Scale float accumulators by (1/row_sum * scale_output)
//   2. Convert float -> BF16 in registers
//   3. Store BF16 to SMEM (smem_o, which unions with smem_v)
//   4. Copy SMEM -> GMEM
//
template <class ElementOut_, class ElementAccum_, class TileShapePV_>
struct SM120FmhaFwdEpilogue {
  using ElementOut = ElementOut_;
  using ElementAccum = ElementAccum_;
  using TileShapePV = TileShapePV_;

  // Output layout: (Q, D, ((H_R, H_KV), CUMULATIVE_Q))
  using ShapeO = cute::Shape<int32_t, int32_t,
                              cute::Shape<cute::Shape<int32_t, int32_t>, int32_t>>;
  using StrideO = cute::Shape<int32_t, _1,
                               cute::Shape<cute::Shape<int32_t, int32_t>, int32_t>>;
  using LayoutO = cute::Layout<ShapeO, StrideO>;

  using ShapeLSE = cute::Shape<int32_t, cute::Shape<int32_t, int32_t>>;
  using StrideLSE = cute::Shape<int32_t, cute::Shape<_1, int32_t>>;
  using LayoutLSE = cute::Layout<ShapeLSE, StrideLSE>;

  // SMEM layout for O (same as kernel traits SmemLayoutO)
  using SmemLayoutO = decltype(tile_to_shape(
      composition(Swizzle<3, 3, 3>{}, Layout<Shape<_16, _32>, Stride<_32, _1>>{}),
      select<0, 1>(TileShapePV_{})));

  struct TensorStorage {
    cute::array_aligned<ElementOut, cute::cosize_v<SmemLayoutO>> smem_o;
    using SmemLayoutO = SM120FmhaFwdEpilogue::SmemLayoutO;
  };

  struct Arguments {
    ElementOut* ptr_O;
    LayoutO layout_O;
    float* ptr_LSE;
    LayoutLSE layout_LSE;
    int32_t max_qo_len;
  };

  struct Params {
    ElementOut* ptr_O;
    LayoutO layout_O;
    float* ptr_LSE;
    LayoutLSE layout_LSE;
    int32_t max_qo_len;
  };

  template <class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& problem_shape,
                                        Arguments const& args, void* workspace) {
    return Params{args.ptr_O, args.layout_O, args.ptr_LSE, args.layout_LSE, args.max_qo_len};
  }

  Params params;

  SM120FmhaFwdEpilogue() = default;
  SM120FmhaFwdEpilogue(Params const& params) : params(params) {}

  // Store the output tile from registers through SMEM to GMEM.
  //
  // tOrO: float accumulator fragment (CTA_Q x HEAD_DIM_VO), already scaled
  // row_sum, row_max: per-row softmax statistics for LSE computation
  template <class BlkCoord, class ProblemShape, class ParamsProblemShape,
            class FragO, class SmemO>
  CUTLASS_DEVICE void store(
      BlkCoord const& blk_coord,
      ProblemShape const& problem_shape,
      ParamsProblemShape const& params_problem_shape,
      FragO const& tOrO,  // float fragment [CTA_Q, HEAD_DIM_VO]
      SmemO& smem_o,      // SMEM buffer for O
      float const* row_sum,
      float const* row_max,
      float scale_output,
      float scale_softmax_log2,
      int thread_idx,
      int num_threads) {
    int qo_tile_idx = get<0>(blk_coord);
    int qo_head_idx = get<2, 0>(blk_coord);
    int batch_idx = get<2, 1>(blk_coord);
    int qo_len = get<0>(problem_shape);
    int segment_offset = get<0>(params_problem_shape).segment_offsets[batch_idx];

    int cta_q = get<0>(TileShapePV{});
    int head_dim_vo = get<1>(TileShapePV{});

    // Write LSE if requested
    if (params.ptr_LSE != nullptr) {
      Tensor gLSE = make_tensor(make_gmem_ptr(params.ptr_LSE), params.layout_LSE);
      // Each thread writes one LSE value for its row
      int row_idx = thread_idx;
      if (row_idx < cta_q) {
        int global_row = qo_tile_idx * cta_q + row_idx;
        if (global_row < qo_len) {
          float lse = __log2f(row_sum[row_idx]) + scale_softmax_log2 * row_max[row_idx];
          gLSE(segment_offset + global_row, qo_head_idx) = lse;
        }
      }
    }

    // Convert float -> ElementOut and store to SMEM, then SMEM -> GMEM
    Tensor sO = make_tensor(make_smem_ptr(smem_o.data()), SmemLayoutO{});
    Tensor gO = make_tensor(make_gmem_ptr(params.ptr_O), params.layout_O);

    // Each thread converts and stores its portion
    int elems_per_row = head_dim_vo;
    int total_elems = cta_q * elems_per_row;

    for (int idx = thread_idx; idx < total_elems; idx += num_threads) {
      int row = idx / elems_per_row;
      int col = idx % elems_per_row;
      int global_row = qo_tile_idx * cta_q + row;

      if (global_row < qo_len) {
        float val = tOrO(idx) * scale_output;
        // Convert float -> output type
        ElementOut out_val;
        if constexpr (std::is_same_v<ElementOut, nv_bfloat16>) {
          out_val = __float2bfloat16(val);
        } else if constexpr (std::is_same_v<ElementOut, half>) {
          out_val = __float2half(val);
        } else {
          out_val = static_cast<ElementOut>(val);
        }

        // Store to global memory directly
        int o_offset = (segment_offset + global_row) * get<0>(params.layout_O.stride()) +
                       col * get<1>(params.layout_O.stride()) +
                       qo_head_idx;
        params.ptr_O[o_offset] = out_val;
      }
    }
  }
};

}  // namespace cutlass::fmha::collective
