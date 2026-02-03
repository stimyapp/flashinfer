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
#include "cute/arch/mma_sm120.hpp"
#include "cutlass/cutlass.h"
#include "fmha_common.hpp"
#include "fmha_fusion.hpp"
#include "sm100_fmha_load_fp4kv_tma_warpspecialized.hpp"

namespace cutlass::fmha::collective {

using namespace cute;

// ─── Block-scaled FP4 MMA helpers (direct PTX) ──────────────────────────────
//
// Bypasses CuTe's gemm()/zip tensor infrastructure for the QK GEMM.
// The SM120 block-scaled FP4 MMA atom SM120_16x8x64_TN_VS takes:
//   A: 4 × uint32_t (128 bits = 32 E2M1 elements; covers M=16, K=64)
//   B: 2 × uint32_t (64 bits = 16 E2M1 elements; covers N=8, K=64)
//   C/D: 4 × float (4 accumulator elements: 2 M-rows × 2 N-columns)
//   SFA: 1 × uint32_t (4 UE4M3 SF bytes; one per K/16 block in K=64)
//   SFB: 1 × uint32_t (4 UE4M3 SF bytes; one per K/16 block in K=64)
//
// Thread-to-data mapping derived from MMA_Traits SFALayout/SFBLayout:
//   SFA: M-row within atom = (lane/16)*8 + (lane%8)
//   SFB: N-row within atom = lane%8
//   Both: all 4 SF columns for the K=64 block
//
// Data register mapping from ALayout ((4,8),(8,2,2)):((128,1),(16,8,512)):
//   A registers a0-a3 are loaded sequentially from the CuTe partition_fragment_A
//   result, which gives 32 uint4_t values = 16 bytes = 4 uint32_t per atom.
//   B registers b0-b1: 16 uint4_t = 8 bytes = 2 uint32_t per atom.
namespace blockscaled_helpers {

// Load 4 contiguous UE4M3 SF bytes from compact SMEM, pack into uint32_t.
// SMEM layout: row-major (rows, SF_COUNT) with stride SF_COUNT.
CUTE_DEVICE static uint32_t load_sf_packed(const uint8_t* smem_sf, int row,
                                            int sf_stride, int k_sf_base) {
  const uint8_t* p = smem_sf + row * sf_stride + k_sf_base;
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
         (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

// Execute one SM120 block-scaled FP4 MMA atom: D = A * B + C with SFA, SFB.
// Calls the PTX instruction directly via the CUTLASS MMA operation struct.
CUTE_DEVICE static void mma_fp4_atom(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3,
    uint32_t sfa, uint32_t sfb) {
  using MMAOp = SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
      cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, cutlass::float_ue4m3_t, 16>;
  MMAOp::fma(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, sfa, sfb);
}

}  // namespace blockscaled_helpers

// SM120 FMHA forward mainloop with NVFP4 KV cache using dual MMA (SageAttention3).
//
// QK GEMM: Block-scaled FP4×FP4 (SM120_16x8x64_TN_VS, K=64)
//   - Q: BF16 → E2M1 + UE4M3 scale factors during load
//   - K: Raw E2M1 packed bytes + FP8 scale factors → loaded directly, no dequant
//   - Hardware applies per-16-element scale factors during MMA
//
// PV GEMM: FP8×FP8 (SM120_16x8x32_TN, K=32)
//   - P: softmax output, F32→FP8 conversion
//   - V: E2M1 + SF → dequanted to FP8 during load
//
// Both GEMMs share SM80_16x8_Row C layout → softmax/P handling is identical.
//
template <class Ktraits, class ActiveMask>
struct SM120FmhaFwdMainloopFP4KV {
  using DTypeQ = typename Ktraits::DTypeQ;  // BF16
  using DTypeKV = typename Ktraits::DTypeKV;
  using DTypeO = typename Ktraits::DTypeO;
  using IdType = typename Ktraits::IdType;

  using TileShape_QKD = typename Ktraits::TileShape_QKD;
  using TileShape_PDV = typename Ktraits::TileShape_PDV;
  static constexpr int CTA_Q = get<0>(TileShape_QKD{});
  static constexpr int CTA_KV = get<1>(TileShape_QKD{});
  static constexpr int HEAD_DIM_QK = Ktraits::HEAD_DIM_QK;
  static constexpr int HEAD_DIM_VO = Ktraits::HEAD_DIM_VO;
  static constexpr int HEAD_DIM_MAX = Ktraits::HEAD_DIM_MAX;

  static constexpr int NUM_THREADS = Ktraits::NUM_THREADS;
  static constexpr int NUM_PRODUCER_THREADS = Ktraits::NUM_PRODUCER_THREADS;
  static constexpr int NUM_MMA_THREADS = Ktraits::NUM_MMA_THREADS;

  static constexpr int SF_BLOCK_SIZE = Ktraits::SF_BLOCK_SIZE;
  static constexpr int SF_COUNT_QK = Ktraits::SF_COUNT_QK;
  static constexpr int SF_COUNT_VO = Ktraits::SF_COUNT_VO;

  using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
  using SmemLayoutQSF = typename Ktraits::SmemLayoutQSF;
  using SmemLayoutK = typename Ktraits::SmemLayoutK;
  using SmemLayoutKSF = typename Ktraits::SmemLayoutKSF;
  using SmemLayoutVt = typename Ktraits::SmemLayoutVt;
  using SmemLayoutP = typename Ktraits::SmemLayoutP;

  using TiledMmaQK = typename Ktraits::TiledMmaQK;
  using TiledMmaPV = typename Ktraits::TiledMmaPV;

  using SharedStorage = typename Ktraits::SharedStorage;

  // Layout types matching SM100 FP4KV loader
  using ShapeT = cute::Shape<int32_t, int32_t, cute::Shape<int32_t, int32_t>>;
  using StrideQ = cute::Shape<int32_t, _1, cute::Shape<int32_t, int32_t>>;
  using StrideK = cute::Shape<int32_t, _1, cute::Shape<_0, int32_t>>;
  using StrideV = cute::Shape<_1, int32_t, cute::Shape<_0, int32_t>>;

  using LayoutQ = cute::Layout<ShapeT, StrideQ>;
  using LayoutK = cute::Layout<ShapeT, StrideK>;
  using LayoutV = cute::Layout<ShapeT, StrideV>;

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
    float scale_softmax = args.scale_softmax;
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

  // ──── Load helpers ──────────────────────────────────────────────────────

  // E2M1 magnitudes as float LUT (indexed by 4-bit nibble, sign in bit 3).
  static __device__ __forceinline__ float e2m1_to_float(uint8_t nibble) {
    static constexpr float LUT[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return LUT[nibble];
  }

  // Dequant a packed byte (2 E2M1 nibbles) to 2 FP8 bytes using float SF.
  static __device__ __forceinline__ uint16_t dequant_packed_byte(uint8_t packed, float sf_float) {
    float lo_f = e2m1_to_float(packed & 0xF) * sf_float;
    float hi_f = e2m1_to_float(packed >> 4) * sf_float;
    uint8_t lo = float_to_fp8e4m3(lo_f);
    uint8_t hi = float_to_fp8e4m3(hi_f);
    return uint16_t(lo) | (uint16_t(hi) << 8);
  }

  // Load Q (BF16) from GMEM, quantize to E2M1 + scale factors, store in SMEM.
  // Per-16-element blocks: find max magnitude, compute UE4M3 scale, quantize.
  // Branchless E2M1 quantization: map |x| to nearest E2M1 magnitude index.
  // E2M1 magnitudes: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
  // Decision boundaries: {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0}
  CUTLASS_DEVICE static uint8_t quantize_e2m1_branchless(float abs_val) {
    // Use cascading comparisons — generates predicated selects, no branches.
    uint8_t idx = 0;
    idx += (abs_val >= 0.25f);  // >=0.25 → at least 1 (0.5)
    idx += (abs_val >= 0.75f);  // >=0.75 → at least 2 (1.0)
    idx += (abs_val >= 1.25f);  // >=1.25 → at least 3 (1.5)
    idx += (abs_val >= 1.75f);  // >=1.75 → at least 4 (2.0)
    idx += (abs_val >= 2.5f);   // >=2.5  → at least 5 (3.0)
    idx += (abs_val >= 3.5f);   // >=3.5  → at least 6 (4.0)
    idx += (abs_val >= 5.0f);   // >=5.0  → 7 (6.0)
    return idx;
  }

  CUTLASS_DEVICE static void load_q_e2m1(
      const DTypeQ* ptr_Q, LayoutQ const& layout_Q,
      uint8_t* smem_q_ptr, uint8_t* smem_q_sf_ptr,
      int qo_segment_offset, int qo_tile_idx, int qo_head_idx,
      int qo_len, int thread_idx, int num_threads) {
    int tile_start = qo_tile_idx * CTA_Q;
    int q_stride_n = get<0>(layout_Q.stride());
    int q_stride_h = get<0>(get<2>(layout_Q.stride()));

    static constexpr float E2M1_MAX = 6.0f;

    // Each thread processes one (row, sf_block) pair
    int total_blocks = CTA_Q * SF_COUNT_QK;
    for (int blk = thread_idx; blk < total_blocks; blk += num_threads) {
      int row = blk / SF_COUNT_QK;
      int sf_idx = blk % SF_COUNT_QK;
      int d_base = sf_idx * SF_BLOCK_SIZE;
      int seq_idx = tile_start + row;

      uint8_t* q_dst = smem_q_ptr + row * (HEAD_DIM_QK / 2) + d_base / 2;
      uint8_t* sf_dst = smem_q_sf_ptr + row * SF_COUNT_QK + sf_idx;

      if (seq_idx >= qo_len) {
        // Zero-fill packed bytes and scale factor
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < SF_BLOCK_SIZE / 2; ++i) {
          q_dst[i] = 0;
        }
        *sf_dst = 0;
        continue;
      }

      int global_seq = qo_segment_offset + seq_idx;
      const DTypeQ* q_row = ptr_Q + global_seq * q_stride_n + qo_head_idx * q_stride_h + d_base;

      // Load 16 BF16 values via vectorized loads and find block max
      float vals[SF_BLOCK_SIZE];
      float amax = 0.0f;
      // Load 8 BF16 values (16 bytes) at a time via uint4
      CUTLASS_PRAGMA_UNROLL
      for (int vec = 0; vec < SF_BLOCK_SIZE / 8; ++vec) {
        const uint4 data = *reinterpret_cast<const uint4*>(q_row + vec * 8);
        const __nv_bfloat162* bf16_pairs = reinterpret_cast<const __nv_bfloat162*>(&data);
        CUTLASS_PRAGMA_UNROLL
        for (int p = 0; p < 4; ++p) {
          float2 f2 = __bfloat1622float2(bf16_pairs[p]);
          vals[vec * 8 + p * 2] = f2.x;
          vals[vec * 8 + p * 2 + 1] = f2.y;
          amax = fmaxf(amax, fmaxf(fabsf(f2.x), fabsf(f2.y)));
        }
      }

      // Compute scale factor: scale = amax / E2M1_MAX, stored as UE4M3
      float scale_f = amax / E2M1_MAX;
      scale_f = fmaxf(scale_f, 1.0f / 256.0f);  // min scale to avoid denorm issues
      uint8_t sf_byte = float_to_fp8e4m3(scale_f);
      float scale_actual = fp8e4m3_to_float(sf_byte);
      float inv_scale = (scale_actual > 0.0f) ? (1.0f / scale_actual) : 0.0f;

      // Quantize and pack using branchless E2M1 quantization
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < SF_BLOCK_SIZE; i += 2) {
        float abs0 = fabsf(vals[i] * inv_scale);
        float abs1 = fabsf(vals[i+1] * inv_scale);
        uint8_t nib0 = quantize_e2m1_branchless(abs0);
        uint8_t nib1 = quantize_e2m1_branchless(abs1);
        if (vals[i] < 0.0f) nib0 |= 8;
        if (vals[i+1] < 0.0f) nib1 |= 8;
        q_dst[i / 2] = nib0 | (nib1 << 4);
      }

      *sf_dst = sf_byte;
    }
  }

  // Load K (E2M1 packed bytes + scale factors) from GMEM directly into SMEM.
  // No dequantization — block-scaled MMA applies SF in hardware.
  // Nibble-swap a uint32_t: swap high/low nibbles of each byte.
  // (a7a6a5a4 a3a2a1a0) -> (a3a2a1a0 a7a6a5a4) per byte.
  CUTLASS_DEVICE static uint32_t nibble_swap_u32(uint32_t x) {
    return ((x & 0x0F0F0F0Fu) << 4) | ((x & 0xF0F0F0F0u) >> 4);
  }

  CUTLASS_DEVICE static void load_k_raw(
      const uint8_t* ptr_e2m1, const uint8_t* ptr_sf,
      int e2m1_stride_n, int e2m1_stride_h,
      int sf_stride_n, int sf_stride_h,
      uint8_t* smem_k_ptr, uint8_t* smem_k_sf_ptr,
      int kv_segment_offset, int tile_idx, int kv_head_idx,
      int kv_len, int thread_idx, int num_threads) {
    int tile_start = tile_idx * CTA_KV;
    constexpr int PACKED_COLS = HEAD_DIM_QK / 2;  // 64 bytes per row

    // Vectorized load: read 16 bytes (uint4) at a time for K packed data.
    // Total: CTA_KV rows × PACKED_COLS bytes = CTA_KV × 64 bytes
    // With uint4 (16 bytes): CTA_KV × 4 vector loads per row
    constexpr int VECS_PER_ROW = PACKED_COLS / 16;  // 64/16 = 4
    constexpr int TOTAL_VECS = CTA_KV * VECS_PER_ROW;

    for (int vid = thread_idx; vid < TOTAL_VECS; vid += num_threads) {
      int row = vid / VECS_PER_ROW;
      int vec_col = vid % VECS_PER_ROW;
      int seq_idx = tile_start + row;
      int byte_col = vec_col * 16;

      uint4 data;
      if (seq_idx < kv_len) {
        int global_seq = kv_segment_offset + seq_idx;
        const uint8_t* src = ptr_e2m1 + global_seq * e2m1_stride_n
                            + kv_head_idx * e2m1_stride_h + byte_col;
        data = *reinterpret_cast<const uint4*>(src);
        // Nibble-swap all 4 uint32_t components in registers
        data.x = nibble_swap_u32(data.x);
        data.y = nibble_swap_u32(data.y);
        data.z = nibble_swap_u32(data.z);
        data.w = nibble_swap_u32(data.w);
      } else {
        data = make_uint4(0, 0, 0, 0);
      }
      *reinterpret_cast<uint4*>(smem_k_ptr + row * PACKED_COLS + byte_col) = data;
    }

    // Load scale factors: CTA_KV rows × SF_COUNT_QK bytes per row.
    // SF_COUNT_QK = HEAD_DIM_QK/16 = 8 bytes per row → use uint2 (8 bytes) loads.
    static_assert(SF_COUNT_QK == 8, "Expected SF_COUNT_QK == 8 for vectorized load");
    for (int row = thread_idx; row < CTA_KV; row += num_threads) {
      int seq_idx = tile_start + row;
      uint2 sf_data;
      if (seq_idx < kv_len) {
        int global_seq = kv_segment_offset + seq_idx;
        const uint8_t* src = ptr_sf + global_seq * sf_stride_n
                            + kv_head_idx * sf_stride_h;
        sf_data = *reinterpret_cast<const uint2*>(src);
      } else {
        sf_data = make_uint2(0, 0);
      }
      *reinterpret_cast<uint2*>(smem_k_sf_ptr + row * SF_COUNT_QK) = sf_data;
    }
  }

  // Load V (E2M1+SF) from GMEM and dequant to FP8 in SMEM in column-major order.
  CUTLASS_DEVICE static void load_and_dequant_v_fp8_colmajor(
      const uint8_t* ptr_e2m1, const uint8_t* ptr_sf,
      int e2m1_stride_n, int e2m1_stride_h,
      int sf_stride_n, int sf_stride_h,
      cutlass::float_e4m3_t* smem_ptr,
      int head_dim, int sf_count,
      int kv_segment_offset, int tile_idx, int kv_head_idx,
      int kv_len, int thread_idx, int num_threads) {
    int tile_start = tile_idx * CTA_KV;
    constexpr int BLOCK = 16;
    int total_blocks = CTA_KV * sf_count;

    for (int blk_idx = thread_idx; blk_idx < total_blocks; blk_idx += num_threads) {
      int row = blk_idx / sf_count;
      int sf_idx = blk_idx % sf_count;
      int d_base = sf_idx * BLOCK;
      int seq_idx = tile_start + row;

      uint8_t* smem_u8 = reinterpret_cast<uint8_t*>(smem_ptr);

      if (seq_idx >= kv_len) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < BLOCK; ++i) {
          smem_u8[(d_base + i) * CTA_KV + row] = 0;
        }
        continue;
      }

      int global_seq = kv_segment_offset + seq_idx;
      const uint8_t* row_e2m1 = ptr_e2m1 + global_seq * e2m1_stride_n +
                                  kv_head_idx * e2m1_stride_h;
      const uint8_t* row_sf = ptr_sf + global_seq * sf_stride_n +
                                kv_head_idx * sf_stride_h;

      float sf_float = fp8e4m3_to_float(row_sf[sf_idx]);
      const uint8_t* packed_ptr = row_e2m1 + d_base / 2;

      // Vectorized load: read 8 packed bytes (uint2) at once
      uint2 packed_vec = *reinterpret_cast<const uint2*>(packed_ptr);
      const uint8_t* packed_bytes = reinterpret_cast<const uint8_t*>(&packed_vec);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < BLOCK / 2; ++i) {
        uint16_t pair = dequant_packed_byte(packed_bytes[i], sf_float);
        smem_u8[(d_base + 2*i) * CTA_KV + row]     = uint8_t(pair);
        smem_u8[(d_base + 2*i + 1) * CTA_KV + row] = uint8_t(pair >> 8);
      }
    }
  }

  // ──── Utility ───────────────────────────────────────────────────────────

  struct MaxOp {
    __device__ __forceinline__ float operator()(float x, float y) const { return fmaxf(x, y); }
  };
  struct SumOp {
    __device__ __forceinline__ float operator()(float x, float y) const { return x + y; }
  };

  template <typename T, typename Op>
  CUTLASS_DEVICE static T allreduce_row(T x, Op op) {
    // SM80_16x8_Row C layout: each thread has 2 M-rows × 2 N-cols.
    // Lane 0: M={0,8} N={0,1}, Lane 1: M={0,8} N={2,3}, etc.
    // 4 lanes (bits 0-1) share each M-row pair. Butterfly reduce across 4.
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 2));
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
  }

  // ──── Combined kernel entry ────────────────────────────────────────────

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

    // Zero all SMEM to ensure deterministic behavior for boundary/padded elements.
    // Required on SM120: column-major byte writes in load_and_dequant_v_fp8_colmajor
    // for padded KV rows have SMEM visibility issues without pre-zeroing.
    {
      constexpr int smem_bytes = sizeof(SharedStorage);
      uint8_t* smem_base = reinterpret_cast<uint8_t*>(&storage);
      for (int i = thread_idx; i < smem_bytes; i += NUM_THREADS) {
        smem_base[i] = 0;
      }
    }
    __syncthreads();

    // ══════════════════════════════════════════════════════════════════════
    // Phase 1: Load Q (BF16→E2M1 + SF) into SMEM
    // ══════════════════════════════════════════════════════════════════════

    {
      load_q_e2m1(params.ptr_Q, params.layout_Q,
                  storage.smem_q.data(), storage.smem_q_sf.data(),
                  qo_segment_offset, qo_tile_idx, qo_head_idx,
                  qo_len, thread_idx, NUM_THREADS);
    }
    __syncthreads();

    // Set up tiled MMAs — PV uses CuTe gemm(), QK uses direct PTX (below)
    TiledMmaQK tiled_mma_qk;
    TiledMmaPV tiled_mma_pv;
    auto thr_mma_qk = tiled_mma_qk.get_slice(thread_idx);
    auto thr_mma_pv = tiled_mma_pv.get_slice(thread_idx);

    // ── Q SMEM tensor for data (loaded into registers via CuTe partition) ──
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<cute::uint4_t*>(storage.smem_q.data())),
                            Layout<Shape<Int<CTA_Q>, Int<HEAD_DIM_QK>>,
                                   Stride<Int<HEAD_DIM_QK>, _1>>{});

    // QK accumulator — use QK TiledMMA for shape/coordinate computation
    auto tSrS = partition_fragment_C(tiled_mma_qk, Shape<Int<CTA_Q>, Int<CTA_KV>>{});
    // PV accumulator
    auto tOrO = partition_fragment_C(tiled_mma_pv, select<0, 1>(TileShape_PDV{}));

    // Coordinate tensors for masking and output
    Tensor cS = cute::make_identity_tensor(Shape<Int<CTA_Q>, Int<CTA_KV>>{});
    Tensor cO = cute::make_identity_tensor(select<0, 1>(TileShape_PDV{}));
    Tensor tScS = thr_mma_qk.partition_C(cS);
    Tensor tOcO = thr_mma_pv.partition_C(cO);

    // Fragment row mapping (C layout is SM80_16x8_Row for both QK and PV)
    // SM80_16x8_Row partition_C gives each thread 2 M-rows × 2 N-cols:
    //   Lane 0: M={0,8}, N={0,1}. Lane 1: M={0,8}, N={2,3}.
    //   Lane 8: M={2,10}, N={0,1}. etc.
    // 4 lanes (differing by bits 0-1) share each pair of M-rows.
    constexpr int NUM_FRAG_ELEMS = decltype(size(tSrS))::value;
    constexpr int NUM_FRAG_ROWS = 2;

    int row_idx[NUM_FRAG_ELEMS];
    int unique_m[NUM_FRAG_ROWS];
    {
      int num_unique = 0;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < NUM_FRAG_ELEMS; ++i) {
        int m = get<0>(tScS(i));
        int found = -1;
        CUTLASS_PRAGMA_UNROLL
        for (int r = 0; r < num_unique; ++r) {
          if (unique_m[r] == m) { found = r; break; }
        }
        if (found >= 0) {
          row_idx[i] = found;
        } else {
          unique_m[num_unique] = m;
          row_idx[i] = num_unique;
          num_unique++;
        }
      }
    }

    float row_max_arr[NUM_FRAG_ROWS];
    float row_sum_arr[NUM_FRAG_ROWS];
    float scores_scale_arr[NUM_FRAG_ROWS];

    // ── Load Q data (E2M1 packed) into registers via CuTe ──
    // Q is used for the A operand of the block-scaled FP4 MMA.
    // partition_fragment_A creates register fragments; partition_A creates SMEM views.
    // The data layout follows ALayout from MMA_Traits, which CuTe handles correctly.
    Tensor tSrQ = thr_mma_qk.partition_fragment_A(sQ);
    {
      Tensor tSsQ = thr_mma_qk.partition_A(sQ);
      cute::copy(tSsQ, tSrQ);
    }

    // ── QK GEMM dimensions ──
    // The FP4 MMA atom is M=16, N=8, K=64. With TiledMMA having 8 warps along M:
    //   CTA_Q=128 → 128/(16*8) = 1 M-repetition (each warp handles one 16-row atom)
    //   CTA_KV=64 → 64/8 = 8 N-repetitions per atom
    //   HEAD_DIM_QK=128 → 128/64 = 2 K-blocks
    //
    // tSrQ shape: (V=32, M_rep=1, K_blocks=2) where V=32 uint4_t per atom per thread.
    // Recast to uint32_t: V becomes 32/8 = 4 (matching ARegisters = uint32_t[4]).
    constexpr int K_BLOCKS_QK = HEAD_DIM_QK / 64;  // K=64 per block-scaled MMA atom
    constexpr int N_ATOMS_QK = CTA_KV / 8;         // N=8 per atom

    // Pre-allocate PV SMEM tensors
    Tensor sP = make_tensor(make_smem_ptr(reinterpret_cast<uint8_t*>(storage.smem_q.data())),
                            SmemLayoutP{});
    auto tOsP = thr_mma_pv.partition_A(sP);
    auto tOrP = thr_mma_pv.partition_fragment_A(sP);

    clear(tOrO);
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < NUM_FRAG_ROWS; ++mi) {
      row_max_arr[mi] = -INFINITY;
      row_sum_arr[mi] = 0.0f;
      scores_scale_arr[mi] = 0.0f;
    }

    // ── Pre-compute SFA M-row for this thread (constant across tiles) ──
    // SFALayout: ((2,2,8),64):((8,0,1),16) maps (T32,V64) → (M16,K64)
    // Thread decomposition: l0=lane/16, l1=(lane/8)%2, l2=lane%8
    // Thread offset = l0*8 + l1*0 + l2*1 = (lane/16)*8 + (lane%8)
    // This gives the M-row within the 16-row MMA atom.
    int lane = thread_idx % 32;
    int warp_id = thread_idx / 32;
    int sfa_m_row = warp_id * 16 + (lane / 16) * 8 + (lane % 8);
    // SFBLayout: ((4,8),64):((0,1),8) maps (T32,V64) → (N8,K64)
    // Thread decomposition: l0=lane/8, l1=lane%8
    // Thread offset = l0*0 + l1*1 = lane%8
    // This gives the N-row within the 8-row MMA atom.
    int sfb_n_base = lane % 8;

    // ══════════════════════════════════════════════════════════════════════
    // Phase 2: Main KV tile loop
    // ══════════════════════════════════════════════════════════════════════

#pragma unroll 1
    for (int tile = 0; tile < mask_tile_count; ++tile) {
      int kv_tile_idx = mask_tile_count - 1 - tile;

      // ── Step 1: Load K + V simultaneously ──
      {
        load_k_raw(
            params.ptr_K_e2m1, params.ptr_K_sf,
            params.k_e2m1_stride_n, params.k_e2m1_stride_h,
            params.k_sf_stride_n, params.k_sf_stride_h,
            storage.smem_k.data(), storage.smem_k_sf.data(),
            kv_segment_offset, kv_tile_idx, kv_head_idx, kv_len,
            thread_idx, NUM_THREADS);
        load_and_dequant_v_fp8_colmajor(
            params.ptr_V_e2m1, params.ptr_V_sf,
            params.v_e2m1_stride_n, params.v_e2m1_stride_h,
            params.v_sf_stride_n, params.v_sf_stride_h,
            reinterpret_cast<cutlass::float_e4m3_t*>(storage.smem_v.data()),
            HEAD_DIM_VO, SF_COUNT_VO,
            kv_segment_offset, kv_tile_idx, kv_head_idx, kv_len,
            thread_idx, NUM_THREADS);
      }
      __syncthreads();  // sync: K + V data ready

      // ── Step 2: Block-scaled QK GEMM → S = Q @ K^T (FP4×FP4, K=64) ──
      {
        Tensor sK = make_tensor(
            make_smem_ptr(reinterpret_cast<cute::uint4_t*>(storage.smem_k.data())),
            Layout<Shape<Int<CTA_KV>, Int<HEAD_DIM_QK>>,
                   Stride<Int<HEAD_DIM_QK>, _1>>{});

        Tensor tSrK = thr_mma_qk.partition_fragment_B(sK);
        {
          Tensor tSsK = thr_mma_qk.partition_B(sK);
          cute::copy(tSsK, tSrK);
        }

        Tensor rQ_u32 = recast<uint32_t>(tSrQ);
        Tensor rK_u32 = recast<uint32_t>(tSrK);

        const uint8_t* q_sf_smem = storage.smem_q_sf.data();
        const uint8_t* k_sf_smem = storage.smem_k_sf.data();

        clear(tSrS);

        CUTLASS_PRAGMA_UNROLL
        for (int k_blk = 0; k_blk < K_BLOCKS_QK; ++k_blk) {
          uint32_t sfa = blockscaled_helpers::load_sf_packed(
              q_sf_smem, sfa_m_row, SF_COUNT_QK, k_blk * 4);

          CUTLASS_PRAGMA_UNROLL
          for (int n_atom = 0; n_atom < N_ATOMS_QK; ++n_atom) {
            int k_n_row = n_atom * 8 + sfb_n_base;
            uint32_t sfb = blockscaled_helpers::load_sf_packed(
                k_sf_smem, k_n_row, SF_COUNT_QK, k_blk * 4);

            uint32_t a0 = rQ_u32(0, 0, k_blk);
            uint32_t a1 = rQ_u32(1, 0, k_blk);
            uint32_t a2 = rQ_u32(2, 0, k_blk);
            uint32_t a3 = rQ_u32(3, 0, k_blk);

            uint32_t b0 = rK_u32(0, n_atom, k_blk);
            uint32_t b1 = rK_u32(1, n_atom, k_blk);

            float& s0 = tSrS(0, 0, n_atom);
            float& s1 = tSrS(1, 0, n_atom);
            float& s2 = tSrS(2, 0, n_atom);
            float& s3 = tSrS(3, 0, n_atom);

            blockscaled_helpers::mma_fp4_atom(
                s0, s1, s2, s3,
                a0, a1, a2, a3, b0, b1,
                s0, s1, s2, s3,
                sfa, sfb);
          }
        }
      }

      // ── Step 3: Scale + mask ──
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

      // ── Step 4: Online softmax ──
      {
        float prev_max[NUM_FRAG_ROWS];
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < NUM_FRAG_ROWS; ++mi) {
          prev_max[mi] = row_max_arr[mi];
        }

        float local_max[NUM_FRAG_ROWS];
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < NUM_FRAG_ROWS; ++mi) {
          local_max[mi] = -INFINITY;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < NUM_FRAG_ELEMS; ++i) {
          local_max[row_idx[i]] = fmaxf(local_max[row_idx[i]], tSrS(i));
        }
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < NUM_FRAG_ROWS; ++mi) {
          row_max_arr[mi] = fmaxf(prev_max[mi], local_max[mi]);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < NUM_FRAG_ROWS; ++mi) {
          row_max_arr[mi] = allreduce_row(row_max_arr[mi], MaxOp{});
        }

        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < NUM_FRAG_ROWS; ++mi) {
          scores_scale_arr[mi] = (prev_max[mi] == -INFINITY)
                                    ? 0.0f
                                    : expf(prev_max[mi] - row_max_arr[mi]);
        }

        if (tile > 0) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < decltype(size(tOrO))::value; ++i) {
            int m_o = get<0>(tOcO(i));
            int ri = 0;
            CUTLASS_PRAGMA_UNROLL
            for (int r = 0; r < NUM_FRAG_ROWS; ++r) {
              if (unique_m[r] == m_o) { ri = r; break; }
            }
            tOrO(i) *= scores_scale_arr[ri];
          }
          CUTLASS_PRAGMA_UNROLL
          for (int mi = 0; mi < NUM_FRAG_ROWS; ++mi) {
            row_sum_arr[mi] *= scores_scale_arr[mi];
          }
        }

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < NUM_FRAG_ELEMS; ++i) {
          int ri = row_idx[i];
          float rmax = row_max_arr[ri];
          tSrS(i) = (rmax == -INFINITY) ? 0.0f : expf(tSrS(i) - rmax);
        }

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < NUM_FRAG_ELEMS; ++i) {
          row_sum_arr[row_idx[i]] += tSrS(i);
        }
      }

      // ── Step 5: Scatter P to SMEM ──
      // All CTA_Q × CTA_KV entries are written by the scatter below
      // (8 warps × 8 N-atoms × 32 threads × 4 elements = 8192 = full coverage).
      {
        uint8_t* smem_p = reinterpret_cast<uint8_t*>(storage.smem_q.data());
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tSrS); ++i) {
          int m_coord = get<0>(tScS(i));
          int n_coord = get<1>(tScS(i));
          smem_p[m_coord * CTA_KV + n_coord] = float_to_fp8e4m3(tSrS(i));
        }
      }
      __syncthreads();  // sync: P visible for PV GEMM

      // ── Step 6: PV GEMM (FP8×FP8) ──
      cute::copy(tOsP, tOrP);

      {
        Tensor sVt = make_tensor(make_smem_ptr(reinterpret_cast<uint8_t*>(storage.smem_v.data())),
                                 SmemLayoutVt{});
        Tensor tOrV = thr_mma_pv.partition_fragment_B(sVt);
        Tensor tOsV = thr_mma_pv.partition_B(sVt);
        cute::copy(tOsV, tOrV);

        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tOrP); ++k_block) {
          cute::gemm(tiled_mma_pv, tOrP(_, _, k_block), tOrV(_, _, k_block), tOrO);
        }
      }

      __syncthreads();  // sync: tile done, safe to overwrite SMEM in next iter
    }  // end KV tile loop

    // ══════════════════════════════════════════════════════════════════════
    // Phase 3: Finalize softmax and write output
    // ══════════════════════════════════════════════════════════════════════
    {
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < NUM_FRAG_ROWS; ++mi) {
        row_sum_arr[mi] = allreduce_row(row_sum_arr[mi], SumOp{});
      }

      float inv_sum_arr[NUM_FRAG_ROWS];
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < NUM_FRAG_ROWS; ++mi) {
        inv_sum_arr[mi] = (row_sum_arr[mi] > 0.0f)
                            ? (params.scale_output / row_sum_arr[mi])
                            : 0.0f;
      }

      int segment_offset = get<0>(params_problem_shape).segment_offsets[batch_idx];

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < decltype(size(tOrO))::value; ++i) {
        int m_local = get<0>(tOcO(i));
        int d_local = get<1>(tOcO(i));
        int global_row = qo_tile_idx * CTA_Q + m_local;

        int ri = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int r = 0; r < NUM_FRAG_ROWS; ++r) {
          if (unique_m[r] == m_local) { ri = r; break; }
        }

        if (global_row < qo_len) {
          int global_seq = segment_offset + global_row;
          float val = tOrO(i) * inv_sum_arr[ri];
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

      // Write LSE — one lane per M-row group writes.
      // 4 lanes share the same 2 M-rows after allreduce; pick lane%4==0.
      if (lse_ptr != nullptr) {
        int lane = thread_idx % 32;
        if (lane % 4 == 0) {
          CUTLASS_PRAGMA_UNROLL
          for (int mi = 0; mi < NUM_FRAG_ROWS; ++mi) {
            int m_local = unique_m[mi];
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
