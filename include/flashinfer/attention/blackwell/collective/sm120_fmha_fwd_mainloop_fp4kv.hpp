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

#include "../../../cutlass_utils.cuh"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "fmha_common.hpp"
#include "fmha_fusion.hpp"
#include "sm100_fmha_load_fp4kv_tma_warpspecialized.hpp"

namespace cutlass::fmha::collective {

using namespace cute;

// SM120 FMHA forward mainloop with E2M1 (FP4) KV cache.
//
// Architecture: All 8 warps (256 threads) iterate the tile loop together.
// Producer warps (0-1) load Q/K/V via E2M1 dequant; consumer warps (2-7)
// perform MMA and softmax. Coordination uses __syncthreads (no PipelineAsync).
// Single-buffered SMEM: K and V each use one stage, loaded per-tile.
//
// IMPORTANT: SMEM tensors for MMA partition/copy must use uint8_t element type
// (matching the MMA atom's ValTypeA/B = uint8_t). Using float_e4m3_t as the
// element type causes implicit float_e4m3_t → float → uint8_t conversion
// during copy, which truncates small FP8 values to zero.
//
template <class Ktraits, class ActiveMask>
struct SM120FmhaFwdMainloopFP4KV {
  using DTypeQ = typename Ktraits::DTypeQ;
  using DTypeKV = typename Ktraits::DTypeKV;
  using DTypeO = typename Ktraits::DTypeO;
  using IdType = typename Ktraits::IdType;

  using TileShape_QKD = typename Ktraits::TileShape_QKD;
  using TileShape_PDV = typename Ktraits::TileShape_PDV;
  static constexpr int CTA_Q = get<0>(TileShape_QKD{});
  static constexpr int CTA_KV = get<1>(TileShape_QKD{});
  static constexpr int HEAD_DIM_QK = Ktraits::HEAD_DIM_QK;
  static constexpr int HEAD_DIM_VO = Ktraits::HEAD_DIM_VO;

  static constexpr int NUM_STAGES = Ktraits::NUM_STAGES;
  static constexpr int NUM_THREADS = Ktraits::NUM_THREADS;
  static constexpr int NUM_PRODUCER_THREADS = Ktraits::NUM_PRODUCER_THREADS;
  static constexpr int NUM_CONSUMER_WARPS = Ktraits::NUM_CONSUMER_WARPS;

  using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
  using SmemLayoutK = typename Ktraits::SmemLayoutK;
  using SmemLayoutV = typename Ktraits::SmemLayoutV;
  using SmemLayoutVt = typename Ktraits::SmemLayoutVt;

  using TiledMmaQK = typename Ktraits::TiledMmaQK;
  using TiledMmaPV = typename Ktraits::TiledMmaPV;

  using SharedStorage = typename Ktraits::SharedStorage;

  static constexpr int SfSize = HEAD_DIM_QK / 16;  // one scale per 16 elements

  // Layout types matching SM100 FP4KV loader
  using ShapeT = cute::Shape<int32_t, int32_t, cute::Shape<int32_t, int32_t>>;
  using StrideQ = cute::Shape<int32_t, _1, cute::Shape<int32_t, int32_t>>;
  using StrideK = cute::Shape<int32_t, _1, cute::Shape<_0, int32_t>>;
  using StrideV = cute::Shape<_1, int32_t, cute::Shape<_0, int32_t>>;

  using LayoutQ = cute::Layout<ShapeT, StrideQ>;
  using LayoutK = cute::Layout<ShapeT, StrideK>;
  using LayoutV = cute::Layout<ShapeT, StrideV>;

  // TileShape for mask computation (2D: Q x KV)
  using TileShape = decltype(select<0, 1>(TileShape_QKD{}));

  struct Arguments {
    const DTypeQ* ptr_Q;
    LayoutQ layout_Q;
    LayoutK layout_K;
    LayoutV layout_V;
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

    float scale_softmax;
    float scale_q = 1.0f;
    float scale_k = 1.0f;
    float scale_v = 1.0f;
    float inv_scale_o = 1.0f;
  };

  struct Params {
    const DTypeQ* ptr_Q;
    LayoutQ layout_Q;
    LayoutK layout_K;
    LayoutV layout_V;
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

    float scale_softmax;
    float scale_softmax_log2;
    float scale_output;
  };

  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& problem_shape,
                                        Arguments const& args, void* workspace) {
    float scale_softmax = args.scale_q * args.scale_k * args.scale_softmax;
    float log2_e = static_cast<float>(std::log2(std::exp(1.0)));
    return Params{
        args.ptr_Q, args.layout_Q, args.layout_K, args.layout_V,
        args.ptr_K_e2m1, args.ptr_V_e2m1, args.ptr_K_sf, args.ptr_V_sf,
        args.k_e2m1_stride_n, args.k_e2m1_stride_h,
        args.v_e2m1_stride_n, args.v_e2m1_stride_h,
        args.k_sf_stride_n, args.k_sf_stride_h,
        args.v_sf_stride_n, args.v_sf_stride_h,
        scale_softmax,
        scale_softmax * log2_e,
        args.scale_v * args.inv_scale_o};
  }

  // ──── Producer: E2M1 dequant load ──────────────────────────────────────────

  CUTLASS_DEVICE static void load_q(
      const DTypeQ* ptr_Q, LayoutQ const& layout_Q,
      uint8_t* smem_q_ptr,
      int qo_segment_offset, int qo_tile_idx, int qo_head_idx,
      int qo_len, int head_dim_qk,
      int thread_idx, int num_producer_threads) {
    int tile_start = qo_tile_idx * CTA_Q;
    const uint8_t* q_base = reinterpret_cast<const uint8_t*>(ptr_Q);
    int q_stride_n = get<0>(layout_Q.stride());
    int q_stride_h = get<0>(get<2>(layout_Q.stride()));
    int total_elements = CTA_Q * head_dim_qk;

    for (int elem = thread_idx; elem < total_elements; elem += num_producer_threads) {
      int row = elem / head_dim_qk;
      int col = elem % head_dim_qk;
      int seq_idx = tile_start + row;
      if (seq_idx < qo_len) {
        int global_seq = qo_segment_offset + seq_idx;
        int src_offset = global_seq * q_stride_n + qo_head_idx * q_stride_h + col;
        smem_q_ptr[row * head_dim_qk + col] = q_base[src_offset];
      } else {
        smem_q_ptr[row * head_dim_qk + col] = 0;
      }
    }
  }

  CUTLASS_DEVICE static void load_and_dequant_k(
      Params const& params,
      uint8_t* smem_k_ptr,
      int kv_segment_offset, int k_tile_idx, int kv_head_idx,
      int kv_len, int thread_idx, int num_producer_threads) {
    int tile_start = k_tile_idx * CTA_KV;
    int total_elements = CTA_KV * HEAD_DIM_QK;

    for (int elem = thread_idx; elem < total_elements; elem += num_producer_threads) {
      int row = elem / HEAD_DIM_QK;
      int d = elem % HEAD_DIM_QK;

      int seq_idx = tile_start + row;
      if (seq_idx >= kv_len) {
        smem_k_ptr[row * HEAD_DIM_QK + d] = 0;
        continue;
      }

      int global_seq = kv_segment_offset + seq_idx;
      const uint8_t* row_e2m1 = params.ptr_K_e2m1 +
                                  global_seq * params.k_e2m1_stride_n +
                                  kv_head_idx * params.k_e2m1_stride_h;
      const uint8_t* row_sf = params.ptr_K_sf +
                                global_seq * params.k_sf_stride_n +
                                kv_head_idx * params.k_sf_stride_h;

      uint8_t packed = row_e2m1[d / 2];
      uint8_t nibble = (d & 1) ? (packed >> 4) : (packed & 0xF);
      uint8_t sf_byte = row_sf[d / 16];
      float e2m1_val = e2m1_nibble_to_float(nibble);
      float sf_f32 = fp8e4m3_to_float(sf_byte);
      float result = e2m1_val * sf_f32;
      smem_k_ptr[row * HEAD_DIM_QK + d] = float_to_fp8e4m3(result);
    }
  }

  CUTLASS_DEVICE static void load_and_dequant_v(
      Params const& params,
      uint8_t* smem_v_ptr,
      int kv_segment_offset, int v_tile_idx, int kv_head_idx,
      int kv_len, int thread_idx, int num_producer_threads) {
    int tile_start = v_tile_idx * CTA_KV;
    int total_elements = CTA_KV * HEAD_DIM_VO;

    for (int elem = thread_idx; elem < total_elements; elem += num_producer_threads) {
      int row = elem / HEAD_DIM_VO;
      int d = elem % HEAD_DIM_VO;

      int seq_idx = tile_start + row;
      if (seq_idx >= kv_len) {
        smem_v_ptr[d + row * HEAD_DIM_VO] = 0;
        continue;
      }

      int global_seq = kv_segment_offset + seq_idx;
      const uint8_t* row_e2m1 = params.ptr_V_e2m1 +
                                  global_seq * params.v_e2m1_stride_n +
                                  kv_head_idx * params.v_e2m1_stride_h;
      const uint8_t* row_sf = params.ptr_V_sf +
                                global_seq * params.v_sf_stride_n +
                                kv_head_idx * params.v_sf_stride_h;

      uint8_t packed = row_e2m1[d / 2];
      uint8_t nibble = (d & 1) ? (packed >> 4) : (packed & 0xF);
      uint8_t sf_byte = row_sf[d / 16];
      float e2m1_val = e2m1_nibble_to_float(nibble);
      float sf_f32 = fp8e4m3_to_float(sf_byte);
      float result = e2m1_val * sf_f32;
      smem_v_ptr[d + row * HEAD_DIM_VO] = float_to_fp8e4m3(result);
    }
  }

  // ──── Consumer: MMA + Online Softmax ───────────────────────────────────────

  static constexpr int NUM_CONSUMER_THREADS = NUM_THREADS - NUM_PRODUCER_THREADS;
  static constexpr int NUM_MMA_THREADS = Ktraits::NUM_MMA_THREADS;

  using SmemLayoutP = typename Ktraits::SmemLayoutP;

  struct MaxOp {
    __device__ __forceinline__ float operator()(float x, float y) const { return fmaxf(x, y); }
  };
  struct SumOp {
    __device__ __forceinline__ float operator()(float x, float y) const { return x + y; }
  };

  // Reduce across 4 threads that share the same M (row) coordinate in the
  // SM120 MMA accumulator. For mma.m16n8k32, each group of 4 consecutive
  // threads (T%4) holds different N columns for the same pair of M rows.
  // XOR-1 and XOR-2 cover all 4 threads within a group.
  template <typename T, typename Op>
  CUTLASS_DEVICE static T allreduce_row(T x, Op op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 2));
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
  }

  // Convert the flat accumulator layout into a (rows, cols) view where
  // "rows" corresponds to M (query) positions and "cols" to N (KV/output dim).
  //
  // For SM120 mma.m16n8k32, each thread's C fragment has atom shape (2, 2):
  //   get<0,0>: size 2, stride 1 — N_atom (consecutive N positions)
  //   get<0,1>: size 2, stride 2 — M_atom (M positions 8 apart in the tile)
  // The full tiled fragment: ((N_atom=2, M_atom=2), M_rep, N_rep)
  //
  // We group: rows = (M_atom, M_rep), cols = (N_atom, N_rep)
  template <typename Layout>
  CUTLASS_DEVICE static auto sm120_convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
    static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = acc_layout;
    return make_layout(
        make_layout(get<0, 1>(l), get<1>(l)),    // (M_atom=2, M_rep) = rows
        make_layout(get<0, 0>(l), get<2>(l)));   // (N_atom=2, N_rep) = cols
  }

  // ── Raw byte copy from SMEM partition to register fragment ──
  // Required because SMEM uses float_e4m3_t type while MMA register fragments
  // use uint8_t (ValTypeA/B). Direct assignment float_e4m3_t → uint8_t goes
  // through float, truncating small FP8 values to 0. memcpy preserves the raw
  // FP8 bit pattern.
  template <typename SrcTensor, typename DstTensor>
  CUTLASS_DEVICE static void copy_fp8_smem_to_reg(SrcTensor const& src, DstTensor& dst) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(dst); ++i) {
      uint8_t b;
      memcpy(&b, &src(i), 1);
      memcpy(&dst(i), &b, 1);
    }
  }

  // ──── Combined kernel entry ────────────────────────────────────────────────

  template <class BlkCoord, class ProblemShape, class ParamsProblemShape>
  CUTLASS_DEVICE void run(
      BlkCoord const& blk_coord,
      ProblemShape const& problem_shape,
      ParamsProblemShape const& params_problem_shape,
      Params const& params,
      SharedStorage& storage,
      float* lse_ptr,
      typename Ktraits::DTypeO* o_ptr,
      int o_stride_n,
      int o_stride_h) {
    int thread_idx = threadIdx.x;
    bool is_producer = (thread_idx < NUM_PRODUCER_THREADS);
    int consumer_idx = thread_idx - NUM_PRODUCER_THREADS;
    bool is_mma_thread = (!is_producer) && (consumer_idx < NUM_MMA_THREADS);

    int qo_tile_idx = get<0>(blk_coord);
    int qo_head_idx = get<2, 0>(blk_coord);
    int batch_idx = get<2, 1>(blk_coord);
    int qo_len = get<0>(problem_shape);
    int kv_len = get<1>(problem_shape);

    int num_qo_heads = (o_stride_h > 0) ? (o_stride_n / o_stride_h) : 1;

    int mask_tile_count = ActiveMask{}.get_trip_count(blk_coord, TileShape{}, problem_shape);

    int qo_segment_offset = get<0>(params_problem_shape).segment_offsets[batch_idx];
    int kv_segment_offset = get<1>(params_problem_shape).segment_offsets[batch_idx];
    int num_qo_heads_per_kv_head = get<0>(get<0>(get<3>(problem_shape)));
    int kv_head_idx = qo_head_idx / num_qo_heads_per_kv_head;

    float scale_softmax = params.scale_softmax;

    // ════════════════════════════════════════════════════════════════════════
    // Phase 1: Load Q into SMEM, setup MMA fragments
    // ════════════════════════════════════════════════════════════════════════

    {
      uint8_t* smem_q_base = reinterpret_cast<uint8_t*>(storage.smem_q.data());
      load_q(params.ptr_Q, params.layout_Q, smem_q_base,
             qo_segment_offset, qo_tile_idx, qo_head_idx,
             qo_len, HEAD_DIM_QK, thread_idx, NUM_THREADS);
    }
    __syncthreads();

    // SMEM tensors for MMA: use the underlying float_e4m3_t pointer for CuTe
    // layout computation, but the actual SMEM→register copy uses raw byte
    // memcpy via copy_fp8_smem_to_reg to avoid type conversion issues.
    TiledMmaQK tiled_mma_qk;
    TiledMmaPV tiled_mma_pv;

    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK_all = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
    Tensor sVt_all = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutVt{});

    auto tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_QKD{}));
    auto tOrO = partition_fragment_C(tiled_mma_pv, select<0, 1>(TileShape_PDV{}));

    Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_QKD{}));
    Tensor cO = cute::make_identity_tensor(select<0, 1>(TileShape_PDV{}));

    auto scores_rowcol_layout = sm120_convert_layout_acc_rowcol(tSrS.layout());
    constexpr int NUM_FRAG_ROWS = decltype(size<0>(scores_rowcol_layout))::value;

    float row_max_arr[NUM_FRAG_ROWS];
    float row_sum_arr[NUM_FRAG_ROWS];
    float scores_scale_arr[NUM_FRAG_ROWS];

    int slice_idx = is_mma_thread ? consumer_idx : 0;
    auto thr_mma_qk = tiled_mma_qk.get_slice(slice_idx);
    auto thr_mma_pv = tiled_mma_pv.get_slice(slice_idx);

    Tensor tScS = thr_mma_qk.partition_C(cS);
    Tensor tOcO = thr_mma_pv.partition_C(cO);

    // Load Q from SMEM to register fragment
    Tensor tSsQ = thr_mma_qk.partition_A(sQ);
    Tensor tSrQ = thr_mma_qk.partition_fragment_A(sQ);
    if (is_mma_thread) {
      copy_fp8_smem_to_reg(tSsQ, tSrQ);

      clear(tOrO);
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < NUM_FRAG_ROWS; ++mi) {
        row_max_arr[mi] = -INFINITY;
        row_sum_arr[mi] = 0.0f;
        scores_scale_arr[mi] = 0.0f;
      }
    }

    // ════════════════════════════════════════════════════════════════════════
    // Phase 2: Main KV tile loop
    // ════════════════════════════════════════════════════════════════════════
#pragma unroll 1
    for (int tile = 0; tile < mask_tile_count; ++tile) {
      int kv_tile_idx = mask_tile_count - 1 - tile;

      // Step 1a: All threads load K tile into SMEM (dequant E2M1→FP8)
      {
        uint8_t* smem_k_base = reinterpret_cast<uint8_t*>(storage.smem_k.data());
        load_and_dequant_k(params, smem_k_base,
                           kv_segment_offset, kv_tile_idx, kv_head_idx, kv_len,
                           thread_idx, NUM_THREADS);
      }
      __syncthreads();

      // Step 1b: Consumers: QK GEMM -> S = Q @ K^T
      if (is_mma_thread) {
        Tensor sK_stage = sK_all(_, _, 0);
        Tensor tSsK = thr_mma_qk.partition_B(sK_stage);
        Tensor tSrK = thr_mma_qk.partition_fragment_B(sK_stage);
        copy_fp8_smem_to_reg(tSsK, tSrK);

        clear(tSrS);
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
          cute::gemm(tiled_mma_qk, tSrQ(_, _, k_block), tSrK(_, _, k_block), tSrS);
        }

        // Step 2: Scale + mask
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tSrS); ++i) {
          tSrS(i) *= scale_softmax;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tSrS); ++i) {
          int qo_idx = get<0>(tScS(i)) + qo_tile_idx * CTA_Q;
          int kv_idx = get<1>(tScS(i)) + kv_tile_idx * CTA_KV;
          if (kv_idx >= kv_len || qo_idx >= qo_len) {
            tSrS(i) = -INFINITY;
          }
          if constexpr (std::is_same_v<ActiveMask, CausalMask>) {
            int offset_q = kv_len - qo_len;
            if ((qo_idx + offset_q) < kv_idx) {
              tSrS(i) = -INFINITY;
            }
          }
        }

        // Step 3: Online softmax
        Tensor scores = make_tensor(tSrS.data(),
                                    sm120_convert_layout_acc_rowcol(tSrS.layout()));

        // 3a. Row-wise max
        float prev_max[NUM_FRAG_ROWS];
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < NUM_FRAG_ROWS; ++mi) {
          prev_max[mi] = row_max_arr[mi];
        }
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size<0>(scores); ++mi) {
          float local_max = scores(mi, 0);
          CUTLASS_PRAGMA_UNROLL
          for (int ni = 1; ni < size<1>(scores); ++ni) {
            local_max = fmaxf(local_max, scores(mi, ni));
          }
          row_max_arr[mi] = fmaxf(prev_max[mi], local_max);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < NUM_FRAG_ROWS; ++mi) {
          row_max_arr[mi] = allreduce_row(row_max_arr[mi], MaxOp{});
        }

        // 3b. Rescale factor.  Guard against -INF - (-INF) = NaN which happens
        // when all logits in a row were -INFINITY in every KV tile so far
        // (e.g. causal mask with reverse tile iteration).
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < NUM_FRAG_ROWS; ++mi) {
          scores_scale_arr[mi] = (prev_max[mi] == -INFINITY)
                                    ? 0.0f
                                    : expf(prev_max[mi] - row_max_arr[mi]);
        }

        // 3c. Rescale O and row_sum from previous tiles
        if (tile > 0) {
          Tensor acc_o_rowcol = make_tensor(tOrO.data(),
                                            sm120_convert_layout_acc_rowcol(tOrO.layout()));
          CUTLASS_PRAGMA_UNROLL
          for (int mi = 0; mi < NUM_FRAG_ROWS; ++mi) {
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
              acc_o_rowcol(mi, ni) *= scores_scale_arr[mi];
            }
            row_sum_arr[mi] *= scores_scale_arr[mi];
          }
        }

        // 3d. P = exp(S - max), accumulate row_sum.
        // When rmax == -INFINITY (all logits masked), exp(-INF - (-INF)) = NaN.
        // Set P = 0 in that case to prevent NaN propagation.
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size<0>(scores); ++mi) {
          float rmax = row_max_arr[mi];
          bool all_masked = (rmax == -INFINITY);
          CUTLASS_PRAGMA_UNROLL
          for (int ni = 0; ni < size<1>(scores); ++ni) {
            scores(mi, ni) = all_masked ? 0.0f : expf(scores(mi, ni) - rmax);
          }
        }
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size<0>(scores); ++mi) {
          CUTLASS_PRAGMA_UNROLL
          for (int ni = 0; ni < size<1>(scores); ++ni) {
            row_sum_arr[mi] += scores(mi, ni);
          }
        }

        // Step 4: Convert P to FP8, write to SMEM (reuse smem_q space)
        {
          uint8_t* smem_p_ptr = reinterpret_cast<uint8_t*>(storage.smem_q.data());
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tSrS); ++i) {
            int m_coord = get<0>(tScS(i));
            int n_coord = get<1>(tScS(i));
            smem_p_ptr[m_coord * CTA_KV + n_coord] = float_to_fp8e4m3(tSrS(i));
          }
        }
      }  // end is_mma_thread (steps 1b-4)

      __syncthreads();  // P in smem_q ready

      // Step 5a: All threads load V tile into SMEM (dequant E2M1→FP8)
      {
        uint8_t* smem_v_base = reinterpret_cast<uint8_t*>(storage.smem_v.data());
        load_and_dequant_v(params, smem_v_base,
                           kv_segment_offset, kv_tile_idx, kv_head_idx, kv_len,
                           thread_idx, NUM_THREADS);
      }
      __syncthreads();  // V fully in SMEM

      // Step 5b: Consumers: PV GEMM -> O += P @ V
      if (is_mma_thread) {
        Tensor sP_pv = make_tensor(make_smem_ptr(
            reinterpret_cast<DTypeKV*>(storage.smem_q.data())), SmemLayoutP{});
        Tensor sVt_stage = sVt_all(_, _, 0);

        Tensor tOrP_s = thr_mma_pv.partition_A(sP_pv);
        Tensor tOrP = thr_mma_pv.partition_fragment_A(sP_pv);
        copy_fp8_smem_to_reg(tOrP_s, tOrP);

        Tensor tOrV_s = thr_mma_pv.partition_B(sVt_stage);
        Tensor tOrV = thr_mma_pv.partition_fragment_B(sVt_stage);
        copy_fp8_smem_to_reg(tOrV_s, tOrV);

        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tOrP); ++k_block) {
          cute::gemm(tiled_mma_pv, tOrP(_, _, k_block), tOrV(_, _, k_block), tOrO);
        }
      }

      __syncthreads();  // V consumed, ready for next tile
    }  // end KV tile loop

    // ════════════════════════════════════════════════════════════════════════
    // Phase 3: Finalize softmax and write output
    // ════════════════════════════════════════════════════════════════════════
    if (is_mma_thread) {
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < NUM_FRAG_ROWS; ++mi) {
        row_sum_arr[mi] = allreduce_row(row_sum_arr[mi], SumOp{});
      }

      // Scale O by (scale_output / row_sum)
      {
        Tensor acc_o_rowcol = make_tensor(tOrO.data(),
                                          sm120_convert_layout_acc_rowcol(tOrO.layout()));
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < NUM_FRAG_ROWS; ++mi) {
          float inv_sum = (row_sum_arr[mi] > 0.0f)
                            ? (params.scale_output / row_sum_arr[mi])
                            : 0.0f;
          CUTLASS_PRAGMA_UNROLL
          for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
            acc_o_rowcol(mi, ni) *= inv_sum;
          }
        }
      }

      // Write O fragment to GMEM
      int segment_offset = get<0>(params_problem_shape).segment_offsets[batch_idx];

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tOrO); ++i) {
        int m_local = get<0>(tOcO(i));
        int d_local = get<1>(tOcO(i));
        int global_row = qo_tile_idx * CTA_Q + m_local;
        if (global_row < qo_len) {
          int global_seq = segment_offset + global_row;
          float val = tOrO(i);
          DTypeO out_val;
          if constexpr (std::is_same_v<DTypeO, nv_bfloat16>) {
            out_val = __float2bfloat16(val);
          } else if constexpr (std::is_same_v<DTypeO, half>) {
            out_val = __float2half(val);
          } else {
            out_val = static_cast<DTypeO>(val);
          }
          o_ptr[global_seq * o_stride_n + qo_head_idx * o_stride_h + d_local] = out_val;
        }
      }

      // Write LSE if requested.
      // For mma.m16n8, threads T%4=0 are the "first" in each row group.
      // Each such thread writes LSE for its NUM_FRAG_ROWS unique M positions.
      if (lse_ptr != nullptr) {
        int lane = consumer_idx % 32;
        if (lane % 4 == 0) {
          CUTLASS_PRAGMA_UNROLL
          for (int mi = 0; mi < NUM_FRAG_ROWS; ++mi) {
            auto rowcol_layout = sm120_convert_layout_acc_rowcol(tSrS.layout());
            int flat_idx = rowcol_layout(mi, _0{});
            int m_local = get<0>(tScS(flat_idx));
            int global_row = qo_tile_idx * CTA_Q + m_local;
            if (global_row < qo_len) {
              int global_seq = segment_offset + global_row;
              float lse_val = logf(fmaxf(row_sum_arr[mi], 1e-10f)) + row_max_arr[mi];
              lse_ptr[global_seq * num_qo_heads + qo_head_idx] = lse_val;
            }
          }
        }
      }
    }
  }
};

}  // namespace cutlass::fmha::collective
