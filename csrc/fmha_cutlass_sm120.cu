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
#include <flashinfer/attention/blackwell/fmha_cutlass_sm120.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/cutlass_utils.cuh>

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

#define DISPATCH_mask_mode(mask_mode, MASK_MODE, ...)   \
  [&]() -> bool {                                       \
    if (mask_mode == MaskMode::kNone) {                 \
      constexpr MaskMode MASK_MODE = MaskMode::kNone;   \
      return __VA_ARGS__();                             \
    } else if (mask_mode == MaskMode::kCausal) {        \
      constexpr MaskMode MASK_MODE = MaskMode::kCausal; \
      return __VA_ARGS__();                             \
    }                                                   \
    return false;                                       \
  }()

#define DISPATCH_head_dim_fp4kv(head_dim_qk, head_dim_vo, HEAD_DIM_QK, HEAD_DIM_VO, ...) \
  [&]() -> bool {                                                                         \
    if (head_dim_qk == 128 && head_dim_vo == 128) {                                       \
      constexpr int HEAD_DIM_QK = 128;                                                    \
      constexpr int HEAD_DIM_VO = 128;                                                    \
      return __VA_ARGS__();                                                               \
    }                                                                                     \
    return false;                                                                         \
  }()

using namespace flashinfer;
using namespace cutlass::fmha::collective;

#if defined(FLASHINFER_ENABLE_FP4_E2M1)

void FMHACutlassSM120RunFP4KV(
    ffi::TensorView workspace_buffer,
    ffi::TensorView q,           // BF16 [total_qo, num_qo_heads, head_dim_qk]
    ffi::TensorView k,           // uint8 packed E2M1 [total_kv, num_kv_heads, head_dim/2]
    ffi::TensorView v,           // uint8 packed E2M1 [total_kv, num_kv_heads, head_dim/2]
    ffi::TensorView k_sf,        // uint8 FP8 E4M3 [total_kv, num_kv_heads, head_dim/16]
    ffi::TensorView v_sf,        // uint8 FP8 E4M3 [total_kv, num_kv_heads, head_dim/16]
    ffi::TensorView qo_segment_offsets,
    ffi::TensorView kv_segment_offsets,
    ffi::TensorView work_indptr,
    ffi::TensorView qo_tile_indices,
    ffi::TensorView qo_head_indices,
    ffi::TensorView batch_indices,
    ffi::TensorView o,           // BF16 output
    Optional<ffi::TensorView> maybe_lse,
    int64_t mask_mode_code,
    double sm_scale,
    int64_t max_qo_len,
    int64_t num_work_items) {
  MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  int total_qo_len = q.size(0);
  int total_kv_len = k.size(0);
  int num_qo_heads = q.size(1);
  int num_kv_heads = k.size(1);
  int head_dim_qk = q.size(2);
  int head_dim_vo = head_dim_qk;
  int batch_size = qo_segment_offsets.size(0) - 1;
  int q_stride_n = q.stride(0);
  int q_stride_h = q.stride(1);

  int k_e2m1_stride_n = k.stride(0);
  int k_e2m1_stride_h = k.stride(1);
  int v_e2m1_stride_n = v.stride(0);
  int v_e2m1_stride_h = v.stride(1);

  int k_sf_stride_n = k_sf.stride(0);
  int k_sf_stride_h = k_sf.stride(1);
  int v_sf_stride_n = v_sf.stride(0);
  int v_sf_stride_h = v_sf.stride(1);

  ffi::CUDADeviceGuard device_guard(qo_segment_offsets.device().device_id);
  const cudaStream_t stream = get_stream(o.device());

  DISPATCH_mask_mode(mask_mode, MASK_MODE, [&] {
    return DISPATCH_head_dim_fp4kv(head_dim_qk, head_dim_vo, HEAD_DIM_QK, HEAD_DIM_VO, [&] {
      // Q is BF16 (quantized to E2M1 inside the kernel by producer warps)
      using cutlass_type_q = cutlass_dtype_t<nv_bfloat16>;
      using cutlass_type_out = cutlass_dtype_t<nv_bfloat16>;
      // SM120 tile sizes for FP8 MMA.
      // P (FP8) overlay requires CTA_KV <= HEAD_DIM_QK.
      // CTA_KV=64 balances parallelism vs tile overhead.
      using TILE_Q = cute::Int<128>;
      using TILE_KV = cute::Int<64>;
      using D_QK = cute::Int<HEAD_DIM_QK>;
      using D_VO = cute::Int<HEAD_DIM_VO>;
      using TileShapeQK = Shape<TILE_Q, TILE_KV, D_QK>;
      using TileShapePV = Shape<TILE_Q, D_VO, TILE_KV>;
      using CutlassMaskMode =
          typename std::conditional<MASK_MODE == MaskMode::kCausal, CausalMask, ResidualMask>::type;

      auto status = run_fmha_fwd_fp4kv_sm120<cutlass_type_q, cutlass_type_out, int32_t,
                                                TileShapeQK, TileShapePV, CutlassMaskMode>(
          workspace_buffer.data_ptr(),
          static_cast<cutlass_type_q*>(q.data_ptr()),
          static_cast<const uint8_t*>(k.data_ptr()),
          static_cast<const uint8_t*>(v.data_ptr()),
          static_cast<const uint8_t*>(k_sf.data_ptr()),
          static_cast<const uint8_t*>(v_sf.data_ptr()),
          static_cast<int*>(qo_segment_offsets.data_ptr()),
          static_cast<int*>(kv_segment_offsets.data_ptr()),
          static_cast<int*>(work_indptr.data_ptr()),
          static_cast<int*>(qo_tile_indices.data_ptr()),
          static_cast<int*>(qo_head_indices.data_ptr()),
          static_cast<int*>(batch_indices.data_ptr()),
          static_cast<cutlass_type_out*>(o.data_ptr()),
          maybe_lse.has_value() ? static_cast<float*>(maybe_lse.value().data_ptr()) : nullptr,
          mask_mode_code, sm_scale,
          num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo,
          q_stride_n, q_stride_h,
          k_e2m1_stride_n, k_e2m1_stride_h,
          v_e2m1_stride_n, v_e2m1_stride_h,
          k_sf_stride_n, k_sf_stride_h,
          v_sf_stride_n, v_sf_stride_h,
          batch_size, total_qo_len, total_kv_len, max_qo_len,
          static_cast<int>(num_work_items), stream);
      TVM_FFI_ICHECK_EQ(status, cudaSuccess)
          << "SM120 CUTLASS FP4KV FMHA forward pass failed: " << cudaGetErrorString(status);

      return true;
    });
  });
}

#endif  // FLASHINFER_ENABLE_FP4_E2M1
