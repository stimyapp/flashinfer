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

#include <cstdint>
#include <vector>

#include "../../allocator.h"
#include "sm120_kernel_traits.cuh"
#include "collective/fmha_fusion.hpp"
#include "collective/sm120_fmha_fwd_mainloop_fp4kv.hpp"
#include "collective/sm120_fmha_fwd_epilogue.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"

namespace flashinfer {

using namespace cute;
using namespace cutlass::fmha::collective;

// SM120 FMHA FP4KV kernel: a simple global function that dispatches to the mainloop.
//
// Unlike SM100 which uses the CUTLASS device::FMHA wrapper with tcgen05/TMEM,
// SM120 uses a straightforward kernel launch since mma.sync is register-based.
template <class Mainloop, class DTypeO, class IdType>
__global__ void sm120_fmha_fp4kv_kernel(
    typename Mainloop::Params mainloop_params,
    DTypeO* __restrict__ o_ptr,
    float* __restrict__ lse_ptr,
    int o_stride_n,
    int o_stride_h,
    const IdType* __restrict__ qo_segment_offsets,
    const IdType* __restrict__ kv_segment_offsets,
    const IdType* __restrict__ work_indptr,
    const IdType* __restrict__ qo_tile_indices,
    const IdType* __restrict__ qo_head_indices,
    const IdType* __restrict__ batch_indices,
    int qo_tile_size,
    int num_work_items,
    int num_qo_heads_per_kv_head) {
  // Each CTA processes one work item
  int work_idx = blockIdx.x;
  if (work_idx >= num_work_items) return;

  int qo_tile_idx = qo_tile_indices[work_idx];
  int qo_head_idx = qo_head_indices[work_idx];
  int batch_idx = batch_indices[work_idx];

  int qo_len = qo_segment_offsets[batch_idx + 1] - qo_segment_offsets[batch_idx];
  int kv_len = kv_segment_offsets[batch_idx + 1] - kv_segment_offsets[batch_idx];

  // Construct problem shape: (qo_len, kv_len, head_dim, ((h_r, num_kv_heads), batch_size))
  // h_r = num_qo_heads_per_kv_head is needed for correct KV head mapping in GQA
  auto blk_coord = make_coord(qo_tile_idx, _0{}, make_coord(qo_head_idx, batch_idx));

  // Problem shape compatible with mask. h_r (= num_qo_heads_per_kv_head) is
  // used by the mainloop to compute: kv_head_idx = qo_head_idx / h_r.
  auto problem_shape = make_tuple(qo_len, kv_len, 0,
                                  make_tuple(make_tuple(num_qo_heads_per_kv_head, 1), 1));

  // Params problem shape (segment offsets)
  struct SegOffsets {
    const IdType* segment_offsets;
  };
  SegOffsets qo_seg{qo_segment_offsets};
  SegOffsets kv_seg{kv_segment_offsets};
  auto params_problem_shape = make_tuple(qo_seg, kv_seg);

  // Shared memory: the mainloop SharedStorage (Q, K, V/O buffers).
  // Accumulator buffers (acc_o, row_max, row_sum) live in per-thread registers.
  extern __shared__ char smem_buf[];
  auto& storage = *reinterpret_cast<typename Mainloop::SharedStorage*>(smem_buf);

  Mainloop mainloop;
  mainloop.run(blk_coord, problem_shape, params_problem_shape,
               mainloop_params, storage,
               lse_ptr, o_ptr, o_stride_n, o_stride_h);
}

// Runner for SM120 FMHA FP4KV
template <typename DTypeQ, typename DTypeOut, typename IdType, class TileShapeQK,
          class TileShapePV, class ActiveMask>
struct FwdRunnerFP4KV_SM120 {
  using Element = DTypeQ;  // FP8 E4M3

  static constexpr int CTA_Q = get<0>(TileShapeQK{});
  static constexpr int CTA_KV = get<1>(TileShapeQK{});
  static constexpr int HEAD_DIM_QK = get<2>(TileShapeQK{});
  static constexpr int HEAD_DIM_VO = get<1>(TileShapePV{});
  // NUM_STAGES is passed to Ktraits but overridden to 1 internally
  // (SM120 uses single-buffered SMEM with __syncthreads coordination).
  static constexpr int NUM_STAGES = 1;

  using Ktraits = flashinfer::SM120AttentionKernelTraits<
      HEAD_DIM_QK, HEAD_DIM_VO, CTA_Q, CTA_KV, NUM_STAGES,
      Element, Element, DTypeOut, IdType, void>;

  using Mainloop = SM120FmhaFwdMainloopFP4KV<Ktraits, ActiveMask>;

  using StrideQ = typename Mainloop::StrideQ;
  using StrideK = typename Mainloop::StrideK;
  using StrideV = typename Mainloop::StrideV;

  static cudaError_t run(void* workspace_buffer,
                         DTypeQ* q,
                         const uint8_t* k_e2m1,
                         const uint8_t* v_e2m1,
                         const uint8_t* k_sf,
                         const uint8_t* v_sf,
                         IdType* qo_segment_offsets,
                         IdType* kv_segment_offsets,
                         IdType* work_indptr,
                         IdType* qo_tile_indices,
                         IdType* qo_head_indices,
                         IdType* batch_indices,
                         DTypeOut* o,
                         float* maybe_lse,
                         int mask_mode_code,
                         double sm_scale,
                         double q_scale,
                         int num_qo_heads,
                         int num_kv_heads,
                         int head_dim_qk,
                         int head_dim_vo,
                         int q_stride_n,
                         int q_stride_h,
                         int k_e2m1_stride_n,
                         int k_e2m1_stride_h,
                         int v_e2m1_stride_n,
                         int v_e2m1_stride_h,
                         int k_sf_stride_n,
                         int k_sf_stride_h,
                         int v_sf_stride_n,
                         int v_sf_stride_h,
                         int batch_size,
                         int total_qo_len,
                         int total_kv_len,
                         int max_qo_len,
                         cudaStream_t stream) {
    int h_r = num_qo_heads / num_kv_heads;
    assert(num_qo_heads % num_kv_heads == 0);

    auto shape_Q = make_shape(total_qo_len, head_dim_qk, make_shape(h_r, num_kv_heads));
    auto stride_Q = make_stride(q_stride_n, _1{}, make_stride(q_stride_h, h_r * q_stride_h));
    auto layout_Q = make_layout(shape_Q, stride_Q);

    auto shape_K = make_shape(total_kv_len, head_dim_qk, make_shape(h_r, num_kv_heads));
    auto stride_K = make_stride(head_dim_qk, _1{}, make_stride(_0{}, num_kv_heads * head_dim_qk));
    auto layout_K = make_layout(shape_K, stride_K);

    auto shape_V = make_shape(head_dim_vo, total_kv_len, make_shape(h_r, num_kv_heads));
    auto stride_V = make_stride(_1{}, head_dim_vo, make_stride(_0{}, num_kv_heads * head_dim_vo));
    auto layout_V = make_layout(shape_V, stride_V);

    typename Mainloop::Arguments mainloop_args{
        q, layout_Q, layout_K, layout_V,
        k_e2m1, v_e2m1, k_sf, v_sf,
        k_e2m1_stride_n, k_e2m1_stride_h,
        v_e2m1_stride_n, v_e2m1_stride_h,
        k_sf_stride_n, k_sf_stride_h,
        v_sf_stride_n, v_sf_stride_h,
        static_cast<float>(sm_scale),
        static_cast<float>(q_scale),
        1.0f, 1.0f, 1.0f};

    // Convert arguments to params
    // Construct a dummy problem shape for to_underlying_arguments
    auto dummy_problem_shape = make_tuple(
        VariableLength{qo_segment_offsets}, VariableLength{kv_segment_offsets}, head_dim_qk,
        make_tuple(make_tuple(h_r, num_kv_heads), batch_size));

    auto mainloop_params = Mainloop::to_underlying_arguments(
        dummy_problem_shape, mainloop_args, workspace_buffer);

    // Compute total work items from segment offsets.
    // Each batch element produces ceil(qo_len / CTA_Q) tiles per QO head.
    // We compute this directly rather than reading from work_indptr, because
    // the plan kernel uses num_ctas as its bucket count (which differs from
    // num_qo_heads * batch_size).
    std::vector<IdType> h_qo_seg(batch_size + 1);
    cudaMemcpyAsync(h_qo_seg.data(), qo_segment_offsets,
                    (batch_size + 1) * sizeof(IdType), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int num_work_items = 0;
    for (int b = 0; b < batch_size; ++b) {
      int qo_len_b = h_qo_seg[b + 1] - h_qo_seg[b];
      num_work_items += (qo_len_b + CTA_Q - 1) / CTA_Q;
    }
    num_work_items *= num_qo_heads;

    if (num_work_items == 0) return cudaSuccess;

    // Compute shared memory size.
    // Accumulator buffers (acc_o, row_max, row_sum) now live in per-thread
    // registers, so only the mainloop SharedStorage is needed in SMEM.
    constexpr int smem_mainloop = sizeof(typename Mainloop::SharedStorage);
    constexpr int smem_total = smem_mainloop;

    // Launch kernel
    dim3 grid(num_work_items);
    dim3 block(Ktraits::NUM_THREADS);

    int o_stride_n_val = num_qo_heads * head_dim_vo;
    int o_stride_h_val = head_dim_vo;

    // Set dynamic shared memory size
    cudaFuncSetAttribute(
        sm120_fmha_fp4kv_kernel<Mainloop, DTypeOut, IdType>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_total);

    sm120_fmha_fp4kv_kernel<Mainloop, DTypeOut, IdType>
        <<<grid, block, smem_total, stream>>>(
            mainloop_params,
            o,
            maybe_lse,
            o_stride_n_val,
            o_stride_h_val,
            qo_segment_offsets,
            kv_segment_offsets,
            work_indptr,
            qo_tile_indices,
            qo_head_indices,
            batch_indices,
            CTA_Q,
            num_work_items,
            h_r);

    return cudaGetLastError();
  }
};

template <typename DTypeQ, typename DTypeOut, typename IdType, class TileShapeQK,
          class TileShapePV, class ActiveMask>
cudaError_t run_fmha_fwd_fp4kv_sm120(
    void* workspace_buffer, DTypeQ* q,
    const uint8_t* k_e2m1, const uint8_t* v_e2m1,
    const uint8_t* k_sf, const uint8_t* v_sf,
    IdType* qo_segment_offsets, IdType* kv_segment_offsets,
    IdType* work_indptr, IdType* qo_tile_indices,
    IdType* qo_head_indices, IdType* batch_indices,
    DTypeOut* o, float* maybe_lse,
    int mask_mode_code, double sm_scale, double q_scale,
    int num_qo_heads, int num_kv_heads,
    int head_dim_qk, int head_dim_vo,
    int q_stride_n, int q_stride_h,
    int k_e2m1_stride_n, int k_e2m1_stride_h,
    int v_e2m1_stride_n, int v_e2m1_stride_h,
    int k_sf_stride_n, int k_sf_stride_h,
    int v_sf_stride_n, int v_sf_stride_h,
    int batch_size, int total_qo_len, int total_kv_len,
    int max_qo_len, cudaStream_t stream) {
  return FwdRunnerFP4KV_SM120<DTypeQ, DTypeOut, IdType, TileShapeQK, TileShapePV,
                                ActiveMask>::run(
      workspace_buffer, q, k_e2m1, v_e2m1, k_sf, v_sf,
      qo_segment_offsets, kv_segment_offsets, work_indptr,
      qo_tile_indices, qo_head_indices, batch_indices,
      o, maybe_lse, mask_mode_code, sm_scale, q_scale,
      num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo,
      q_stride_n, q_stride_h,
      k_e2m1_stride_n, k_e2m1_stride_h,
      v_e2m1_stride_n, v_e2m1_stride_h,
      k_sf_stride_n, k_sf_stride_h,
      v_sf_stride_n, v_sf_stride_h,
      batch_size, total_qo_len, total_kv_len, max_qo_len, stream);
}

}  // namespace flashinfer
