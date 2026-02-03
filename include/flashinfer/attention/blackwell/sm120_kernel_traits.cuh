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

#include <type_traits>

#include "../../cutlass_utils.cuh"
#include "cute/algorithm/copy.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/arch/mma_sm120.hpp"
#include "cute/atom/mma_traits_sm120.hpp"
#include "cute/arch/mma_sm80.hpp"
#include "cute/atom/mma_traits_sm80.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"

namespace flashinfer {

using namespace cute;

// SharedStorage for SM120 FMHA with NVFP4 KV cache.
//
// Uses dual MMA strategy (SageAttention3):
//   QK GEMM: Block-scaled FP4×FP4 (SM120_16x8x64_TN_VS, K=64) — no K dequant needed
//   PV GEMM: FP8×FP8 (SM120_16x8x32_TN, K=32) — V dequanted E2M1+SF→FP8
//
// Q is quantized BF16→E2M1 with per-16-element UE4M3 scale factors.
// K stays as packed E2M1 bytes with FP8 E4M3 scale factors (cast to UE4M3).
// V is dequanted E2M1+SF→FP8 during load.
// P is softmax(S) converted from F32→FP8 for PV GEMM.
//
// This gives 4x QK throughput over BF16 (FP4 K=64 vs BF16 K=16) and
// eliminates K dequantization overhead entirely.
//
// SMEM layout:
//   smem_q:   E2M1 packed [CTA_Q, HEAD_DIM_QK/2]  — Q data (BF16→E2M1 during load)
//   smem_q_sf: UE4M3 [CTA_Q, SF_COUNT_QK]         — Q scale factors
//   smem_k:    [CTA_KV, HEAD_DIM_QK/2]              — K (E2M1 packed)
//   smem_k_sf: [CTA_KV, SF_COUNT_QK]               — K scale factors (UE4M3)
//   smem_v:    [CTA_KV, HEAD_DIM_VO]               — V (FP8)
//   P overlay: FP8 [CTA_Q, CTA_KV] in smem_q region
template <int CTA_Q_, int CTA_KV_, int HEAD_DIM_QK_, int HEAD_DIM_VO_,
          int SF_COUNT_QK_, int SF_COUNT_VO_>
struct SM120SharedStorageFP4 {
  // Q: packed E2M1 [CTA_Q, HEAD_DIM_QK/2] (4-bit packed, 2 per byte)
  // Also used as P overlay: FP8 [CTA_Q, CTA_KV] after softmax
  cute::array_aligned<uint8_t, CTA_Q_ * HEAD_DIM_QK_ / 2> smem_q;
  // Q scale factors: UE4M3 [CTA_Q, SF_COUNT_QK]
  cute::array_aligned<uint8_t, CTA_Q_ * SF_COUNT_QK_> smem_q_sf;
  // K: packed E2M1 [CTA_KV, HEAD_DIM_QK/2] row-major
  cute::array_aligned<uint8_t, CTA_KV_ * HEAD_DIM_QK_ / 2> smem_k;
  // K scale factors: UE4M3 [CTA_KV, SF_COUNT_QK]
  cute::array_aligned<uint8_t, CTA_KV_ * SF_COUNT_QK_> smem_k_sf;
  // V: dequanted FP8 [HEAD_DIM_VO, CTA_KV] column-major (for PV GEMM B operand)
  cute::array_aligned<uint8_t, CTA_KV_ * HEAD_DIM_VO_> smem_v;
};

template <int HEAD_DIM_QK_, int HEAD_DIM_VO_, int CTA_Q_, int CTA_KV_,
          int NUM_STAGES_, typename DTypeQ_, typename DTypeKV_, typename DTypeO_,
          typename IdType_, typename AttentionVariant_>
struct SM120AttentionKernelTraits {
  using AttentionVariant = AttentionVariant_;

  using DTypeQ = DTypeQ_;    // BF16 (nv_bfloat16) — quantized to E2M1 in kernel
  using DTypeKV = DTypeKV_;  // float_e4m3_t (API compatibility)
  using DTypeO = DTypeO_;
  using IdType = IdType_;
  using DTypeQKAccum = float;

  static constexpr int CTA_Q = CTA_Q_;
  static_assert(CTA_Q % 16 == 0);
  static constexpr int CTA_KV = CTA_KV_;
  static constexpr int HEAD_DIM_QK = HEAD_DIM_QK_;
  static constexpr int HEAD_DIM_VO = HEAD_DIM_VO_;
  static_assert(HEAD_DIM_QK % 64 == 0, "HEAD_DIM_QK must be divisible by 64 for FP4 MMA K=64");
  static_assert(HEAD_DIM_VO % 32 == 0, "HEAD_DIM_VO must be divisible by 32 for FP8 MMA K=32");

  // All 8 warps participate in everything (load + MMA + softmax).
  static constexpr int NUM_WARPS = 8;
  static constexpr int NUM_THREADS = NUM_WARPS * cutlass::NumThreadsPerWarp;  // 256
  static constexpr int NUM_PRODUCER_WARPS = 0;
  static constexpr int NUM_CONSUMER_WARPS = NUM_WARPS;
  static constexpr int NUM_PRODUCER_THREADS = 0;

  using TileShape_QKD = Shape<Int<CTA_Q>, Int<CTA_KV>, Int<HEAD_DIM_QK>>;
  using TileShape_PDV = Shape<Int<CTA_Q>, Int<HEAD_DIM_VO>, Int<CTA_KV>>;

  static constexpr int NUM_STAGES = 1;

  // ─── Block-scaled FP4 MMA for QK GEMM ────────────────────────────────
  // SM120_16x8x64_TN_VS: M=16, N=8, K=64, E2M1×E2M1 with UE4M3 scale factors
  // Per-16-element scale factors applied in hardware — no K dequant needed.
  // 2x compute throughput over FP8 MMA (K=64 vs K=32).
  using MmaAtomQK = MMA_Atom<SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
      cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, cutlass::float_ue4m3_t, 16>>;

  // ─── FP8 MMA for PV GEMM ────────────────────────────────────────────
  // SM120_16x8x32_TN: M=16, N=8, K=32, FP8×FP8, F32 accumulator
  // V is dequanted E2M1+SF→FP8 during load; P is F32→FP8 in registers.
  using MmaAtomPV = MMA_Atom<SM120_16x8x32_TN<
      cutlass::float_e4m3_t, cutlass::float_e4m3_t, float>>;

  static constexpr int NUM_MMA_WARPS = NUM_WARPS;
  static_assert(CTA_Q % (16 * NUM_MMA_WARPS) == 0,
                "CTA_Q must be divisible by 16 * NUM_MMA_WARPS");

  using AtomLayoutMMA = Layout<Shape<Int<NUM_MMA_WARPS>, _1, _1>>;
  using TiledMmaQK = decltype(make_tiled_mma(MmaAtomQK{}, AtomLayoutMMA{}));
  using TiledMmaPV = decltype(make_tiled_mma(MmaAtomPV{}, AtomLayoutMMA{}));

  static constexpr int NUM_MMA_THREADS = size(TiledMmaQK{});
  static_assert(NUM_MMA_THREADS == NUM_MMA_WARPS * 32);

  // ─── SMEM Layouts ──────────────────────────────────────────────────────

  static constexpr int SF_BLOCK_SIZE = 16;
  static constexpr int SF_COUNT_QK = HEAD_DIM_QK / SF_BLOCK_SIZE;
  static constexpr int SF_COUNT_VO = HEAD_DIM_VO / SF_BLOCK_SIZE;

  static constexpr int HEAD_DIM_MAX = (HEAD_DIM_QK > HEAD_DIM_VO) ? HEAD_DIM_QK : HEAD_DIM_VO;

  // Q: packed E2M1, 4-bit per element → (CTA_Q, HEAD_DIM_QK/2) bytes, row-major
  // For QK GEMM A operand: need (CTA_Q, HEAD_DIM_QK) in uint4_t = (CTA_Q, HEAD_DIM_QK/2) bytes
  using SmemLayoutQ = Layout<Shape<Int<CTA_Q>, Int<HEAD_DIM_QK / 2>>,
                              Stride<Int<HEAD_DIM_QK / 2>, _1>>;

  // Q scale factors: UE4M3 (CTA_Q, SF_COUNT_QK) compact row-major.
  // SF values are loaded manually in the mainloop — no CuTe partitioning needed.
  using SmemLayoutQSF = Layout<Shape<Int<CTA_Q>, Int<SF_COUNT_QK>>,
                                Stride<Int<SF_COUNT_QK>, _1>>;

  // K packed E2M1: (CTA_KV, HEAD_DIM_QK/2) row-major — stored in smem_kv region
  using SmemLayoutK = Layout<Shape<Int<CTA_KV>, Int<HEAD_DIM_QK / 2>>,
                              Stride<Int<HEAD_DIM_QK / 2>, _1>>;

  // K scale factors: UE4M3 (CTA_KV, SF_COUNT_QK) compact row-major.
  using SmemLayoutKSF = Layout<Shape<Int<CTA_KV>, Int<SF_COUNT_QK>>,
                                Stride<Int<SF_COUNT_QK>, _1>>;

  // V: stored column-major in smem_v so K=CTA_KV is contiguous.
  // Storage layout: smem_v[dim * CTA_KV + row] (column-major).
  // PV GEMM B operand view: (N=HEAD_DIM_VO, K=CTA_KV) with K-contiguous.
  using SmemLayoutVt = Layout<Shape<Int<HEAD_DIM_VO>, Int<CTA_KV>>,
                               Stride<Int<CTA_KV>, _1>>;

  // P: FP8 (CTA_Q, CTA_KV), overlays smem_q space.
  // smem_q has CTA_Q * HEAD_DIM_QK / 2 bytes.
  // P needs CTA_Q * CTA_KV FP8 bytes.
  // Requires: CTA_KV <= HEAD_DIM_QK / 2 (true for 64 <= 128/2 = 64).
  static_assert(CTA_KV <= HEAD_DIM_QK / 2,
                "CTA_KV must be <= HEAD_DIM_QK/2 for P to fit in smem_q overlay");
  using SmemLayoutP = Layout<Shape<Int<CTA_Q>, Int<CTA_KV>>,
                              Stride<Int<CTA_KV>, _1>>;

  using SharedStorage = SM120SharedStorageFP4<
      CTA_Q, CTA_KV, HEAD_DIM_QK, HEAD_DIM_VO,
      SF_COUNT_QK, SF_COUNT_VO>;
};

}  // namespace flashinfer
