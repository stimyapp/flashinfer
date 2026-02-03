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
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"

namespace flashinfer {

using namespace cute;

// SharedStorage for SM120 FMHA: Q, K, union(V, O).
//
// No pipeline barriers are needed because the SM120 mainloop uses simple
// __syncthreads-based coordination between producer and consumer threads
// (all threads iterate the tile loop together; no PipelineAsync).
template <class DTypeQ, class DTypeKV, class DTypeOut,
          class SmemLayoutQ, class SmemLayoutK, class SmemLayoutV, class SmemLayoutO>
struct SM120SharedStorageQKVO {
  cute::array_aligned<DTypeQ, cute::cosize_v<SmemLayoutQ>> smem_q;
  cute::array_aligned<DTypeKV, cute::cosize_v<SmemLayoutK>> smem_k;
  union {
    cute::array_aligned<DTypeKV, cute::cosize_v<SmemLayoutV>> smem_v;
    cute::array_aligned<DTypeOut, cute::cosize_v<SmemLayoutO>> smem_o;
  };
};

template <int HEAD_DIM_QK_, int HEAD_DIM_VO_, int CTA_Q_, int CTA_KV_,
          int NUM_STAGES_, typename DTypeQ_, typename DTypeKV_, typename DTypeO_,
          typename IdType_, typename AttentionVariant_>
struct SM120AttentionKernelTraits {
  using AttentionVariant = AttentionVariant_;

  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = IdType_;
  using DTypeQKAccum = float;

  static constexpr int CTA_Q = CTA_Q_;
  static_assert(CTA_Q % 16 == 0);
  static constexpr int CTA_KV = CTA_KV_;
  static constexpr int HEAD_DIM_QK = HEAD_DIM_QK_;
  static constexpr int HEAD_DIM_VO = HEAD_DIM_VO_;
  static_assert(HEAD_DIM_QK % 32 == 0);
  static_assert(HEAD_DIM_VO % 32 == 0);

  // SM120 uses per-warp mma.sync (32 threads each), not warpgroup GMMA
  // 8 warps total: 2 producer (dequant), 6 consumer (MMA + softmax)
  static constexpr int NUM_WARPS = 8;
  static constexpr int NUM_THREADS = NUM_WARPS * cutlass::NumThreadsPerWarp;  // 256
  static constexpr int NUM_PRODUCER_WARPS = 2;
  static constexpr int NUM_CONSUMER_WARPS = NUM_WARPS - NUM_PRODUCER_WARPS;
  static constexpr int NUM_PRODUCER_THREADS = NUM_PRODUCER_WARPS * cutlass::NumThreadsPerWarp;

  using TileShape_QKD = Shape<Int<CTA_Q>, Int<CTA_KV>, Int<HEAD_DIM_QK>>;
  using TileShape_PDV = Shape<Int<CTA_Q>, Int<HEAD_DIM_VO>, Int<CTA_KV>>;

  // Only 1 stage is needed: the SM120 mainloop uses single-buffered SMEM
  // with __syncthreads coordination (no PipelineAsync double-buffering).
  // The NUM_STAGES_ template parameter is accepted for API compatibility
  // but always overridden to 1 to avoid wasting SMEM.
  static constexpr int NUM_STAGES = 1;

  // SM120 MMA atom: mma.sync.aligned.kind::f8f6f4.m16n8k32
  // This is register-based: A in 4x uint32, B in 2x uint32, C/D in 4x float
  using MmaAtom = MMA_Atom<SM120_16x8x32_TN<float_e4m3_t, float_e4m3_t, float>>;

  // Tile the MMA across the CTA using multiple warps in the M dimension.
  // Each MMA atom is 16x8x32 and requires one warp (32 threads).
  // We use NUM_MMA_WARPS atoms along M; CuTe's partition_fragment_C/A/B
  // automatically handle repetition when the full tile is larger.
  //
  // NUM_MMA_WARPS chosen so that CTA_Q is divisible by (16 * NUM_MMA_WARPS).
  // With 4 warps: covers 64 rows per MMA call. CTA_Q=64,128,192,256 all work.
  // The remaining (NUM_CONSUMER_WARPS - NUM_MMA_WARPS) warps are idle during
  // MMA but help with the P->SMEM write and epilogue output store.
  static constexpr int NUM_MMA_WARPS = 4;
  static_assert(CTA_Q % (16 * NUM_MMA_WARPS) == 0,
                "CTA_Q must be divisible by 16 * NUM_MMA_WARPS");

  using AtomLayoutMMA = Layout<Shape<Int<NUM_MMA_WARPS>, _1, _1>>;
  using TiledMmaQK = decltype(make_tiled_mma(MmaAtom{}, AtomLayoutMMA{}));
  using TiledMmaPV = decltype(make_tiled_mma(MmaAtom{}, AtomLayoutMMA{}));

  static constexpr int NUM_MMA_THREADS = size(TiledMmaQK{});
  static_assert(NUM_MMA_THREADS == NUM_MMA_WARPS * 32,
                "MMA thread count must equal NUM_MMA_WARPS * 32");
  static_assert(NUM_MMA_THREADS <= NUM_CONSUMER_WARPS * 32,
                "MMA threads must not exceed consumer threads");

  // SMEM layouts for FP8 data.
  //
  // IMPORTANT: Simple row-major layouts are used (no swizzle, no tiled atoms)
  // because the producer threads write to SMEM using raw byte offsets:
  //   smem_ptr[row * dim + col]
  // The consumer threads read via CuTe's partition_A/B which apply these layouts.
  // Both sides must agree on the mapping from logical (row, col) to physical
  // byte offset. Simple row-major ensures logical(row, col) = row * dim + col.
  //
  // NOTE: tile_to_shape with a non-trivial atom produces a block-interleaved
  // layout that does NOT match raw row-major writes. Hence we use plain
  // make_layout with explicit row-major strides.
  //
  // Bank conflicts are acceptable for correctness; swizzle can be added later
  // if the producer writes are also done through the CuTe SMEM layout.

  // Q: (CTA_Q, HEAD_DIM_QK) - row-major, loaded once
  using SmemLayoutQ = Layout<Shape<Int<CTA_Q>, Int<HEAD_DIM_QK>>,
                              Stride<Int<HEAD_DIM_QK>, _1>>;

  // K: (CTA_KV, HEAD_DIM_QK, 1) - row-major, single-buffered
  // The trailing dimension of 1 is kept for compatibility with CuTe tensor
  // slicing: sK_all(_, _, 0) selects the single stage.
  using SmemLayoutK = Layout<Shape<Int<CTA_KV>, Int<HEAD_DIM_QK>, Int<NUM_STAGES>>,
                              Stride<Int<HEAD_DIM_QK>, _1, Int<CTA_KV * HEAD_DIM_QK>>>;

  // V: (CTA_KV, HEAD_DIM_VO, 1) - row-major, single-buffered
  using SmemLayoutV = Layout<Shape<Int<CTA_KV>, Int<HEAD_DIM_VO>, Int<NUM_STAGES>>,
                              Stride<Int<HEAD_DIM_VO>, _1, Int<CTA_KV * HEAD_DIM_VO>>>;

  // Transposed view of V for PV GEMM: logical shape (HEAD_DIM_VO, CTA_KV, 1)
  // The PV GEMM B operand is (N=HEAD_DIM_VO, K=CTA_KV) in TN layout.
  // We swap the first two dimensions: V's (row=CTA_KV, col=HEAD_DIM_VO) becomes
  // Vt's (row=HEAD_DIM_VO, col=CTA_KV), i.e., Vt(d, n, s) = V(n, d, s).
  using SmemLayoutVt = Layout<Shape<Int<HEAD_DIM_VO>, Int<CTA_KV>, Int<NUM_STAGES>>,
                               Stride<_1, Int<HEAD_DIM_VO>, Int<CTA_KV * HEAD_DIM_VO>>>;

  // P: (CTA_Q, CTA_KV) - softmax output in SMEM, reuses smem_q space for PV GEMM.
  // Q is no longer needed after the QK GEMM, so P can safely overlay it.
  // P must fit: CTA_Q * CTA_KV <= CTA_Q * HEAD_DIM_QK, i.e., CTA_KV <= HEAD_DIM_QK.
  static_assert(CTA_KV <= HEAD_DIM_QK,
                "CTA_KV must be <= HEAD_DIM_QK for P to fit in smem_q space");
  using SmemLayoutP = Layout<Shape<Int<CTA_Q>, Int<CTA_KV>>,
                              Stride<Int<CTA_KV>, _1>>;

  // O: (CTA_Q, HEAD_DIM_VO) - output buffer in SMEM, shares space with V
  using SmemLayoutO = Layout<Shape<Int<CTA_Q>, Int<HEAD_DIM_VO>>,
                              Stride<Int<HEAD_DIM_VO>, _1>>;

  using SharedStorage = SM120SharedStorageQKVO<DTypeQ, DTypeKV, DTypeO,
                                               SmemLayoutQ, SmemLayoutK, SmemLayoutV, SmemLayoutO>;
};

}  // namespace flashinfer
