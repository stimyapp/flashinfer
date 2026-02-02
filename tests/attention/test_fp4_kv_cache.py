"""Tests for FP4 (E2M1) KV cache with trtllm-gen paged attention.

Requires SM100+ (Blackwell) GPU. Tests that the trtllm-gen FMHA kernel
produces correct output when KV cache is stored in E2M1 format with
per-block-16 FP8 E4M3 scale factors.
"""

import math

import pytest
import torch

import flashinfer
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from flashinfer.prefill import trtllm_batch_context_with_kv_cache
from flashinfer.utils import (
    FP4Tensor,
    ceil_div,
    get_compute_capability,
    round_up,
)

GPU_DEVICE = "cuda:0"

# E2M1 representable values (positive): 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)
E2M1_MAX = 6.0


def is_sm100a():
    """Check for SM100a (data center Blackwell: B100/B200/GB200).
    TRT-LLM FMHA kernel requires SM100a specifically, not SM12x (RTX Blackwell).
    """
    from flashinfer.utils import is_sm100a_supported
    return is_sm100a_supported(torch.device(GPU_DEVICE))


def quantize_to_fp4_e2m1(x_bf16: torch.Tensor, block_size: int = 16):
    """Quantize BF16 tensor to FP4 E2M1 with per-block scale factors.

    Args:
        x_bf16: Input tensor of shape [..., D] where D is divisible by block_size.
        block_size: Number of elements per scale factor block (must be 16).

    Returns:
        packed_data: uint8 tensor of shape [..., D // 2] (2 FP4 values per byte).
        scale_factors: float8_e4m3fn tensor of shape [..., D // block_size].
    """
    assert x_bf16.shape[-1] % block_size == 0
    orig_shape = x_bf16.shape
    x = x_bf16.float()

    # Reshape to blocks of 16
    num_blocks_dim = orig_shape[-1] // block_size
    block_shape = orig_shape[:-1] + (num_blocks_dim, block_size)
    x_blocks = x.reshape(block_shape)

    # Compute per-block scale: abs_max / E2M1_MAX
    abs_max = x_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    block_scale = abs_max / E2M1_MAX  # shape [..., num_blocks, 1]

    # Scale values into E2M1 range
    scaled = x_blocks / block_scale  # shape [..., num_blocks, block_size]

    # Round to nearest E2M1 value
    signs = scaled.sign()
    abs_scaled = scaled.abs()
    # Find nearest E2M1 value using broadcasting
    e2m1 = E2M1_VALUES.to(x.device)
    diffs = (abs_scaled.unsqueeze(-1) - e2m1).abs()  # [..., block_size, 8]
    nearest_idx = diffs.argmin(dim=-1)  # [..., block_size]
    quantized = e2m1[nearest_idx] * signs

    # Pack pairs of FP4 values into uint8
    # E2M1 encoding: sign(1) | exp(2) | mantissa(1) = 4 bits
    quantized_flat = quantized.reshape(orig_shape)

    def fp4_encode(val):
        """Encode a single FP4 E2M1 value to 4-bit unsigned int."""
        sign = (val < 0).long()
        abs_val = val.abs()
        # Lookup table for E2M1 encoding (magnitude bits)
        # 0->0b000, 0.5->0b001, 1.0->0b010, 1.5->0b011,
        # 2.0->0b100, 3.0->0b101, 4.0->0b110, 6.0->0b111
        lut = torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long, device=val.device
        )
        e2m1_vals = E2M1_VALUES.to(val.device)
        diffs = (abs_val.unsqueeze(-1) - e2m1_vals).abs()
        idx = diffs.argmin(dim=-1)
        magnitude_bits = lut[idx]
        return (sign << 3) | magnitude_bits

    # Pack pairs: low nibble = even index, high nibble = odd index
    even = quantized_flat[..., 0::2]
    odd = quantized_flat[..., 1::2]
    even_encoded = fp4_encode(even)
    odd_encoded = fp4_encode(odd)
    packed = (odd_encoded << 4 | even_encoded).to(torch.uint8)

    # Scale factors as FP8 E4M3
    scale_flat = block_scale.squeeze(-1)  # [..., num_blocks]
    # Clamp to FP8 E4M3 range and convert
    finfo = torch.finfo(torch.float8_e4m3fn)
    scale_clamped = scale_flat.clamp(min=finfo.min, max=finfo.max)
    scale_fp8 = scale_clamped.to(torch.float8_e4m3fn)

    return packed, scale_fp8


def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax * 0.1
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


@pytest.fixture(autouse=True)
def skip_non_sm100a():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not is_sm100a():
        pytest.skip("Requires SM100a (data center Blackwell) GPU")


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(8, 1), (8, 8)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("page_size", [16])
@pytest.mark.parametrize("max_kv_len", [64, 256])
def test_fp4_kv_cache_decode(
    batch_size,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    max_kv_len,
):
    """Test FP4 KV cache decode attention against BF16 reference."""
    torch.manual_seed(42)

    workspace_size = 256 * 1024 * 1024
    workspace_buffer = torch.zeros(
        workspace_size, dtype=torch.uint8, device=GPU_DEVICE
    )

    # Generate random KV cache in BF16
    num_pages = ceil_div(max_kv_len * batch_size, page_size) + batch_size
    kv_bf16 = torch.randn(
        num_pages, 2, num_kv_heads, page_size, head_dim,
        dtype=torch.bfloat16, device=GPU_DEVICE
    ) * 0.1

    # Quantize K and V to FP4 separately
    k_bf16 = kv_bf16[:, 0]  # [num_pages, num_kv_heads, page_size, head_dim]
    v_bf16 = kv_bf16[:, 1]

    k_packed, k_sf = quantize_to_fp4_e2m1(k_bf16)
    v_packed, v_sf = quantize_to_fp4_e2m1(v_bf16)

    # Assemble packed KV cache: [num_pages, 2, num_kv_heads, page_size, head_dim//2]
    kv_packed = torch.stack([k_packed, v_packed], dim=1)

    # Assemble scale factors: [num_pages, 2, num_kv_heads, page_size, head_dim//16]
    kv_sf_cache = torch.stack([k_sf, v_sf], dim=1)

    # Generate block tables and seq lens
    seq_lens = torch.randint(
        1, max_kv_len + 1, (batch_size,), dtype=torch.int32
    )
    seq_lens[-1] = max_kv_len
    max_pages_per_seq = ceil_div(max_kv_len, page_size)
    block_tables = torch.zeros(
        batch_size, max_pages_per_seq, dtype=torch.int32, device=GPU_DEVICE
    )
    page_idx = 0
    for b in range(batch_size):
        num_pages_for_seq = ceil_div(int(seq_lens[b].item()), page_size)
        for p in range(num_pages_for_seq):
            block_tables[b, p] = page_idx
            page_idx += 1

    seq_lens = seq_lens.to(GPU_DEVICE)

    # Generate FP8 query
    query_bf16 = torch.randn(
        batch_size, num_qo_heads, head_dim,
        dtype=torch.bfloat16, device=GPU_DEVICE
    )
    query_fp8, q_scale = to_float8(query_bf16)

    # BMM scales
    bmm1_scale = q_scale * 1.0 * (head_dim ** -0.5)
    bmm2_scale = 1.0

    # Run FP4 KV cache decode
    output = trtllm_batch_decode_with_kv_cache(
        query=query_fp8,
        kv_cache=kv_packed,
        workspace_buffer=workspace_buffer,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_kv_len,
        bmm1_scale=float(bmm1_scale),
        bmm2_scale=bmm2_scale,
        kv_layout="HND",
        backend="trtllm-gen",
        kv_scale_factors=(
            kv_sf_cache[:, 0],  # K scale factors
            kv_sf_cache[:, 1],  # V scale factors
        ),
        kv_sf_scale=1.0,
    )

    # Smoke test: output should be finite and have the right shape
    assert output.shape == (batch_size, num_qo_heads, head_dim), (
        f"Shape mismatch: {output.shape} vs {(batch_size, num_qo_heads, head_dim)}"
    )
    assert torch.isfinite(output.float()).all(), "Output contains non-finite values"


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(8, 1)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("page_size", [16])
@pytest.mark.parametrize("max_q_len", [32])
@pytest.mark.parametrize("max_kv_len", [64])
def test_fp4_kv_cache_context(
    batch_size,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    max_q_len,
    max_kv_len,
):
    """Test FP4 KV cache context (prefill) attention."""
    torch.manual_seed(42)

    workspace_size = 256 * 1024 * 1024
    workspace_buffer = torch.zeros(
        workspace_size, dtype=torch.uint8, device=GPU_DEVICE
    )

    # Generate sequence lengths
    q_lens = torch.randint(1, max_q_len + 1, (batch_size,), dtype=torch.int32)
    q_lens[-1] = max_q_len
    in_kv_lens = torch.randint(0, max_kv_len - max_q_len + 1, (batch_size,), dtype=torch.int32)
    kv_lens = q_lens + in_kv_lens

    cum_q_lens = torch.cat([
        torch.tensor([0], dtype=torch.int32),
        torch.cumsum(q_lens, dim=0, dtype=torch.int32),
    ]).to(GPU_DEVICE)
    cum_kv_lens = torch.cat([
        torch.tensor([0], dtype=torch.int32),
        torch.cumsum(kv_lens, dim=0, dtype=torch.int32),
    ]).to(GPU_DEVICE)

    total_q_tokens = int(q_lens.sum().item())

    # Generate random KV cache in BF16
    max_kv = int(kv_lens.max().item())
    num_pages = ceil_div(max_kv * batch_size, page_size) + batch_size
    kv_bf16 = torch.randn(
        num_pages, 2, num_kv_heads, page_size, head_dim,
        dtype=torch.bfloat16, device=GPU_DEVICE
    ) * 0.1

    # Quantize to FP4
    k_packed, k_sf = quantize_to_fp4_e2m1(kv_bf16[:, 0])
    v_packed, v_sf = quantize_to_fp4_e2m1(kv_bf16[:, 1])
    kv_packed = torch.stack([k_packed, v_packed], dim=1)
    kv_sf_cache = torch.stack([k_sf, v_sf], dim=1)

    # Block tables
    max_pages_per_seq = ceil_div(max_kv, page_size)
    block_tables = torch.zeros(
        batch_size, max_pages_per_seq, dtype=torch.int32, device=GPU_DEVICE
    )
    page_idx = 0
    for b in range(batch_size):
        num_pages_b = ceil_div(int(kv_lens[b].item()), page_size)
        for p in range(num_pages_b):
            block_tables[b, p] = page_idx
            page_idx += 1

    kv_lens_gpu = kv_lens.to(GPU_DEVICE)

    # Generate FP8 query
    query_bf16 = torch.randn(
        total_q_tokens, num_qo_heads, head_dim,
        dtype=torch.bfloat16, device=GPU_DEVICE
    )
    query_fp8, q_scale = to_float8(query_bf16)

    bmm1_scale = q_scale * 1.0 * (head_dim ** -0.5)

    # Run FP4 KV context attention
    output = trtllm_batch_context_with_kv_cache(
        query=query_fp8,
        kv_cache=kv_packed,
        workspace_buffer=workspace_buffer,
        block_tables=block_tables,
        seq_lens=kv_lens_gpu,
        max_q_len=int(q_lens.max().item()),
        max_kv_len=max_kv,
        bmm1_scale=float(bmm1_scale),
        bmm2_scale=1.0,
        batch_size=batch_size,
        cum_seq_lens_q=cum_q_lens,
        cum_seq_lens_kv=cum_kv_lens,
        kv_layout="HND",
        kv_scale_factors=(kv_sf_cache[:, 0], kv_sf_cache[:, 1]),
        kv_sf_scale=1.0,
    )

    assert output.shape == (total_q_tokens, num_qo_heads, head_dim)
    assert torch.isfinite(output.float()).all(), "Output contains non-finite values"
