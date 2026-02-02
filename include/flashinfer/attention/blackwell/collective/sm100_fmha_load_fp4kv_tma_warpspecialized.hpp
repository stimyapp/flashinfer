/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "../../../cutlass_utils.cuh"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "fmha_common.hpp"
#include "fmha_fusion.hpp"

namespace cutlass::fmha::collective {

using namespace cute;

// E2M1 has 8 representable magnitudes: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
// The lookup table maps a 4-bit nibble (with sign bit) to float.
// Nibble encoding: bit3=sign, bits[2:0]=magnitude index
__device__ __forceinline__ float e2m1_nibble_to_float(uint8_t nibble) {
  constexpr float lut[16] = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
      -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
  return lut[nibble & 0xF];
}

// Convert FP8 E4M3 byte to float. Uses __nv_fp8_e4m3 type available in CUDA 12+.
__device__ __forceinline__ float fp8e4m3_to_float(uint8_t fp8_byte) {
  __nv_fp8_e4m3 val;
  memcpy(&val, &fp8_byte, 1);
  return static_cast<float>(val);
}

// Convert float to FP8 E4M3 byte with saturation.
__device__ __forceinline__ uint8_t float_to_fp8e4m3(float val) {
  __nv_fp8_e4m3 fp8 = static_cast<__nv_fp8_e4m3>(val);
  uint8_t result;
  memcpy(&result, &fp8, 1);
  return result;
}

// Load handler for E2M1 (FP4) KV cache with TMA + SMEM dequant.
// Loads E2M1 packed KV from HBM via TMA, dequantizes to FP8 E4M3 in SMEM,
// then signals the MMA pipeline.
//
// Template parameters:
//   Element: FP8 E4M3 (the MMA element type after dequant)
//   CollectiveMmaQK/PV: standard FP8 MMA collectives
//   SmemLayoutQ/K/V: SMEM layouts for FP8 data
//   TensorStorage: storage struct with smem_q, smem_k, smem_v, smem_k_sf, smem_v_sf
//   PipelineQ: TMA pipeline for Q (unchanged)
//   PipelineK/V: Async pipelines for K/V (software barrier after dequant)
//   Mask/TileShape: mask and tile configuration
template <class Element, class CollectiveMmaQK, class CollectiveMmaPV, class SmemLayoutQ,
          class SmemLayoutK, class SmemLayoutV, class TensorStorage, class PipelineQ,
          class PipelineK, class PipelineV, class Mask, class TileShape, int HeadDim_>
struct Sm100FmhaLoadFP4KVTmaWarpspecialized {
  using TileShapeQK = typename CollectiveMmaQK::TileShape;
  using TileShapePV = typename CollectiveMmaPV::TileShape;

  static constexpr int HeadDim = HeadDim_;
  static constexpr int SfSize = HeadDim / 16;  // one scale per 16 elements

  using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
  using GmemTiledCopyKV = cute::SM90_TMA_LOAD;
  static constexpr uint32_t NumStagesQ = PipelineQ::Stages;

  // Q layout: standard (N, D, (H_R, H_G))
  using ShapeT = cute::Shape<int32_t, int32_t, cute::Shape<int32_t, int32_t>>;
  using StrideQ = cute::Shape<int32_t, _1, cute::Shape<int32_t, int32_t>>;

  // K/V E2M1 layout: (N, D/2, (H_R, H_G)) - packed pairs in uint8
  using ShapeKV_E2M1 = cute::Shape<int32_t, int32_t, cute::Shape<int32_t, int32_t>>;
  using StrideK_E2M1 = cute::Shape<int32_t, _1, cute::Shape<_0, int32_t>>;
  using StrideV_E2M1 = cute::Shape<_1, int32_t, cute::Shape<_0, int32_t>>;

  // Scale factor layout: (N, D/16, (H_R, H_G))
  using ShapeSF = cute::Shape<int32_t, int32_t, cute::Shape<int32_t, int32_t>>;
  using StrideSF = cute::Shape<int32_t, _1, cute::Shape<_0, int32_t>>;

  // Standard FP8 layouts for Q (used for TMA of Q which is FP8)
  using StrideK_FP8 = cute::Shape<int32_t, _1, cute::Shape<_0, int32_t>>;
  using StrideV_FP8 = cute::Shape<_1, int32_t, cute::Shape<_0, int32_t>>;

  using LayoutQ = cute::Layout<ShapeT, StrideQ>;
  using LayoutK = cute::Layout<ShapeT, StrideK_FP8>;  // exposed as FP8 layout for mainloop compat
  using LayoutV = cute::Layout<ShapeT, StrideV_FP8>;

  struct Arguments {
    const Element* ptr_Q;
    LayoutQ layout_Q;
    // FP8 layout pointers (for compatibility with mainloop stride types)
    // but the actual data is E2M1 packed at ptr_K_e2m1 / ptr_V_e2m1
    LayoutK layout_K;
    LayoutV layout_V;
    // E2M1 packed data pointers
    const uint8_t* ptr_K_e2m1;  // [total_kv, num_kv_heads, head_dim/2]
    const uint8_t* ptr_V_e2m1;  // [total_kv, num_kv_heads, head_dim/2]
    // Scale factor pointers (FP8 E4M3)
    const uint8_t* ptr_K_sf;    // [total_kv, num_kv_heads, head_dim/16]
    const uint8_t* ptr_V_sf;    // [total_kv, num_kv_heads, head_dim/16]
    // Strides for E2M1 data
    int k_e2m1_stride_n;
    int k_e2m1_stride_h;
    int v_e2m1_stride_n;
    int v_e2m1_stride_h;
    // Strides for scale factors
    int k_sf_stride_n;
    int k_sf_stride_h;
    int v_sf_stride_n;
    int v_sf_stride_h;
  };

  // TMA descriptors for Q only (K/V loaded manually due to E2M1 format)
  using ClusterLayout_VMNK =
      decltype(tiled_divide(make_layout(Shape<_1, _1, _1>{}),
                            make_tile(typename CollectiveMmaQK::TiledMma::AtomThrID{})));
  using TMA_Q = typename CollectiveMmaQK::Params::TMA_A;

  struct Params {
    TMA_Q tma_load_Q;
    LayoutQ layout_Q;
    LayoutK layout_K;
    LayoutV layout_V;
    // E2M1 data pointers and strides
    const uint8_t* ptr_K_e2m1;
    const uint8_t* ptr_V_e2m1;
    const uint8_t* ptr_K_sf;
    const uint8_t* ptr_V_sf;
    int k_e2m1_stride_n;
    int k_e2m1_stride_h;
    int v_e2m1_stride_n;
    int v_e2m1_stride_h;
    int k_sf_stride_n;
    int k_sf_stride_h;
    int v_sf_stride_n;
    int v_sf_stride_h;
  };

  template <class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args,
                                        void* workspace) {
    static_assert(is_variable_length_v<tuple_element_t<0, ProblemShape>>);
    static_assert(is_variable_length_v<tuple_element_t<1, ProblemShape>>);
    auto ptr_Q = args.ptr_Q;
    LayoutQ layout_Q = args.layout_Q;

    auto mQ = make_tensor(make_gmem_ptr(ptr_Q), layout_Q);

    auto cluster_layout_vmnk =
        tiled_divide(make_layout(Shape<_1, _1, _1>{}),
                     make_tile(typename CollectiveMmaQK::TiledMma::AtomThrID{}));
    TMA_Q tma_load_Q = make_tma_atom_A_sm100<Element>(
        GmemTiledCopyQ{}, mQ, SmemLayoutQ{}(_, _, _, _0{}), TileShapeQK{},
        typename CollectiveMmaQK::TiledMma{}, cluster_layout_vmnk);

    return Params{tma_load_Q, layout_Q, args.layout_K, args.layout_V,
                  args.ptr_K_e2m1, args.ptr_V_e2m1, args.ptr_K_sf, args.ptr_V_sf,
                  args.k_e2m1_stride_n, args.k_e2m1_stride_h,
                  args.v_e2m1_stride_n, args.v_e2m1_stride_h,
                  args.k_sf_stride_n, args.k_sf_stride_h,
                  args.v_sf_stride_n, args.v_sf_stride_h};
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
  }

  // Helper: load E2M1 K tile from global memory, dequant to FP8 in SMEM
  CUTLASS_DEVICE void load_and_dequant_k(
      Params const& params,
      uint8_t* smem_k_ptr,          // destination: FP8 data in SMEM
      int kv_offset,                 // sequence offset for this batch
      int k_tile_idx,                // which K tile to load
      int head_idx,                  // KV head index
      int kv_len,                    // sequence length
      int tile_kv_size) {            // number of KV positions per tile
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int tile_start = k_tile_idx * tile_kv_size;
    int head_dim_qk = HeadDim;

    // Each row in the tile: load E2M1 packed data + SF, then dequant
    // Distribute rows across warps
    for (int row = warp_id; row < tile_kv_size; row += (blockDim.x / 32)) {
      int seq_idx = tile_start + row;
      if (seq_idx >= kv_len) {
        // Zero-fill out-of-bounds rows
        for (int d = lane_id; d < head_dim_qk; d += 32) {
          smem_k_ptr[row * head_dim_qk + d] = 0;
        }
        continue;
      }

      int global_seq = kv_offset + seq_idx;

      // Pointer to this row's E2M1 packed data
      const uint8_t* row_e2m1 = params.ptr_K_e2m1 +
                                  global_seq * params.k_e2m1_stride_n +
                                  head_idx * params.k_e2m1_stride_h;
      // Pointer to this row's scale factors
      const uint8_t* row_sf = params.ptr_K_sf +
                                global_seq * params.k_sf_stride_n +
                                head_idx * params.k_sf_stride_h;

      // Load E2M1 packed bytes to registers and dequant
      for (int d = lane_id; d < head_dim_qk; d += 32) {
        uint8_t packed = row_e2m1[d / 2];
        uint8_t nibble = (d & 1) ? (packed >> 4) : (packed & 0xF);
        uint8_t sf_byte = row_sf[d / 16];
        float e2m1_val = e2m1_nibble_to_float(nibble);
        float sf_f32 = fp8e4m3_to_float(sf_byte);
        float result = e2m1_val * sf_f32;
        smem_k_ptr[row * head_dim_qk + d] = float_to_fp8e4m3(result);
      }
    }
  }

  // Helper: load E2M1 V tile from global memory, dequant to FP8 in SMEM
  // V is transposed: layout is (D, N, H) instead of (N, D, H)
  CUTLASS_DEVICE void load_and_dequant_v(
      Params const& params,
      uint8_t* smem_v_ptr,
      int kv_offset,
      int v_tile_idx,
      int head_idx,
      int kv_len,
      int tile_kv_size) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int tile_start = v_tile_idx * tile_kv_size;
    int head_dim_vo = HeadDim;

    // V in SMEM is (D, N) layout — columns are sequence positions
    for (int row = warp_id; row < tile_kv_size; row += (blockDim.x / 32)) {
      int seq_idx = tile_start + row;
      if (seq_idx >= kv_len) {
        for (int d = lane_id; d < head_dim_vo; d += 32) {
          // V SMEM is stride-1 on D dimension, stride-head_dim_vo on N dimension
          smem_v_ptr[d + row * head_dim_vo] = 0;
        }
        continue;
      }

      int global_seq = kv_offset + seq_idx;

      const uint8_t* row_e2m1 = params.ptr_V_e2m1 +
                                  global_seq * params.v_e2m1_stride_n +
                                  head_idx * params.v_e2m1_stride_h;
      const uint8_t* row_sf = params.ptr_V_sf +
                                global_seq * params.v_sf_stride_n +
                                head_idx * params.v_sf_stride_h;

      for (int d = lane_id; d < head_dim_vo; d += 32) {
        uint8_t packed = row_e2m1[d / 2];
        uint8_t nibble = (d & 1) ? (packed >> 4) : (packed & 0xF);
        uint8_t sf_byte = row_sf[d / 16];
        float e2m1_val = e2m1_nibble_to_float(nibble);
        float sf_f32 = fp8e4m3_to_float(sf_byte);
        float result = e2m1_val * sf_f32;
        // V SMEM: (D, N) layout
        smem_v_ptr[d + row * head_dim_vo] = float_to_fp8e4m3(result);
      }
    }
  }

  template <class BlkCoord, class ProblemShape, class ParamsProblemShape>
  CUTLASS_DEVICE void load(BlkCoord const& blk_coord, ProblemShape const& problem_shape,
                           Params const& params, ParamsProblemShape const& params_problem_shape,
                           TensorStorage& storage, PipelineQ& pipeline_q,
                           typename PipelineQ::PipelineState& pipeline_q_producer_state,
                           PipelineK& pipeline_k,
                           typename PipelineK::PipelineState& pipeline_k_producer_state,
                           PipelineV& pipeline_v,
                           typename PipelineV::PipelineState& pipeline_v_producer_state) {
    int qo_tile_idx = get<0>(blk_coord);
    int qo_head_idx = get<2, 0>(blk_coord);
    int batch_idx = get<2, 1>(blk_coord);
    int qo_len = get<0>(problem_shape);
    int kv_len = get<1>(problem_shape);
    int qo_segment_offset = get<0>(params_problem_shape).segment_offsets[batch_idx];
    int kv_segment_offset = get<1>(params_problem_shape).segment_offsets[batch_idx];

    int mask_tile_count = Mask{}.get_trip_count(blk_coord, TileShape{}, problem_shape);
    // problem_shape[3] = ((h_r, num_kv_heads), batch_size)
    int num_qo_heads_per_kv_head = get<0>(get<0>(get<3>(problem_shape)));
    int kv_head_idx = qo_head_idx / num_qo_heads_per_kv_head;

    // Tile sizes from TileShapeQK
    int tile_kv_size = get<1>(TileShapeQK{});  // typically 128

    using X = Underscore;

    // TMA setup for Q (standard FP8 path)
    Tensor mQ = params.tma_load_Q.get_tma_tensor(params.layout_Q.shape());
    ThrMMA mma_qk = typename CollectiveMmaQK::TiledMma{}.get_slice(0);
    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});

    auto gQ = get_local_tile_tensor(mQ, select<0, 2>(TileShapeQK{}), qo_head_idx, qo_segment_offset,
                                    qo_len);
    Tensor tSgQ_qdl = mma_qk.partition_A(gQ);
    auto [tQgQ, tQsQ] = tma_partition(params.tma_load_Q, _0{}, Layout<_1>{}, group_modes<0, 3>(sQ),
                                      group_modes<0, 3>(tSgQ_qdl));

    uint32_t lane_predicate = cute::elect_one_sync();

    // Q1: TMA load
    int q0_index = 2 * get<0>(blk_coord);
    int q1_index = 2 * get<0>(blk_coord) + 1;
    pipeline_q.producer_acquire(pipeline_q_producer_state);
    if (lane_predicate) {
      auto tma_barrier = pipeline_q.producer_get_barrier(pipeline_q_producer_state);
      copy(params.tma_load_Q.with(*tma_barrier, 0), tQgQ(_, q0_index),
           tQsQ(_, pipeline_q_producer_state.index()));
    }
    ++pipeline_q_producer_state;

    // K1: manual load + dequant
    int k_index = 0;
    {
      pipeline_k.producer_acquire(pipeline_k_producer_state);
      int smem_k_stage = pipeline_k_producer_state.index();
      uint8_t* smem_k_base = reinterpret_cast<uint8_t*>(storage.smem_k.data());
      int smem_k_stage_bytes = cute::cosize_v<SmemLayoutK> / PipelineK::Stages * sizeof(Element);
      uint8_t* smem_k_ptr = smem_k_base + smem_k_stage * smem_k_stage_bytes;

      load_and_dequant_k(params, smem_k_ptr,
                          kv_segment_offset, k_index, kv_head_idx, kv_len, tile_kv_size);
      __syncthreads();
      pipeline_k.producer_commit(pipeline_k_producer_state);
      ++pipeline_k_producer_state;
      k_index += 1;
    }

    // Q2: TMA load
    pipeline_q.producer_acquire(pipeline_q_producer_state);
    if (lane_predicate) {
      auto tma_barrier = pipeline_q.producer_get_barrier(pipeline_q_producer_state);
      copy(params.tma_load_Q.with(*tma_barrier, 0), tQgQ(_, q1_index),
           tQsQ(_, pipeline_q_producer_state.index()));
    }
    ++pipeline_q_producer_state;

    // V1: manual load + dequant
    int v_index = 0;
    {
      pipeline_v.producer_acquire(pipeline_v_producer_state);
      int smem_v_stage = pipeline_v_producer_state.index();
      uint8_t* smem_v_base = reinterpret_cast<uint8_t*>(storage.smem_v.data());
      int smem_v_stage_bytes = cute::cosize_v<SmemLayoutV> / PipelineV::Stages * sizeof(Element);
      uint8_t* smem_v_ptr = smem_v_base + smem_v_stage * smem_v_stage_bytes;

      load_and_dequant_v(params, smem_v_ptr,
                          kv_segment_offset, v_index, kv_head_idx, kv_len, tile_kv_size);
      __syncthreads();
      pipeline_v.producer_commit(pipeline_v_producer_state);
      ++pipeline_v_producer_state;
      v_index += 1;
    }

    // Loop: remaining K/V tiles
    mask_tile_count -= 1;
    for (; mask_tile_count > 0; mask_tile_count -= 1) {
      // Ki
      {
        pipeline_k.producer_acquire(pipeline_k_producer_state);
        int smem_k_stage = pipeline_k_producer_state.index();
        uint8_t* smem_k_base = reinterpret_cast<uint8_t*>(storage.smem_k.data());
        int smem_k_stage_bytes = cute::cosize_v<SmemLayoutK> / PipelineK::Stages * sizeof(Element);
        uint8_t* smem_k_ptr = smem_k_base + smem_k_stage * smem_k_stage_bytes;

        load_and_dequant_k(params, smem_k_ptr,
                            kv_segment_offset, k_index, kv_head_idx, kv_len, tile_kv_size);
        __syncthreads();
        pipeline_k.producer_commit(pipeline_k_producer_state);
        ++pipeline_k_producer_state;
        k_index += 1;
      }

      // Vi
      {
        pipeline_v.producer_acquire(pipeline_v_producer_state);
        int smem_v_stage = pipeline_v_producer_state.index();
        uint8_t* smem_v_base = reinterpret_cast<uint8_t*>(storage.smem_v.data());
        int smem_v_stage_bytes = cute::cosize_v<SmemLayoutV> / PipelineV::Stages * sizeof(Element);
        uint8_t* smem_v_ptr = smem_v_base + smem_v_stage * smem_v_stage_bytes;

        load_and_dequant_v(params, smem_v_ptr,
                            kv_segment_offset, v_index, kv_head_idx, kv_len, tile_kv_size);
        __syncthreads();
        pipeline_v.producer_commit(pipeline_v_producer_state);
        ++pipeline_v_producer_state;
        v_index += 1;
      }
    }
  }
};

}  // namespace cutlass::fmha::collective
