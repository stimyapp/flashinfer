"""Tests for CUTLASS FMHA with E2M1 (FP4) KV cache.

The E2M1 format packs two 4-bit float values per byte. Each value has
8 representable magnitudes: {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
Per-block FP8 E4M3 scale factors (one per 16 elements) recover dynamic range.

This test:
1. Generates random Q (FP8 E4M3), K/V (BF16)
2. Quantizes K/V to E2M1 + FP8 scale factors
3. Runs CUTLASS FMHA with E2M1 KV
4. Compares against BF16 reference attention
"""

import math

import pytest
import torch

import flashinfer
from flashinfer.prefill import fmha_varlen_fp4kv
from flashinfer.utils import is_sm100a_supported, is_sm110a_supported, is_sm120a_supported


# E2M1 lookup table: nibble -> float value
E2M1_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0


def quantize_to_e2m1(tensor: torch.Tensor, block_size: int = 16):
    """Quantize a BF16/FP32 tensor to E2M1 with per-block FP8 E4M3 scale factors.

    Args:
        tensor: Input tensor of shape [..., head_dim], BF16 or FP32
        block_size: Number of elements per scale factor block (default 16)

    Returns:
        packed: uint8 tensor of shape [..., head_dim/2] with E2M1 packed pairs
        sf: uint8 tensor of shape [..., head_dim/block_size] with FP8 E4M3 scale factors
    """
    orig_shape = tensor.shape
    head_dim = orig_shape[-1]
    assert head_dim % block_size == 0
    assert head_dim % 2 == 0

    flat = tensor.float().reshape(-1, head_dim)
    num_rows = flat.shape[0]
    num_blocks = head_dim // block_size

    # Compute per-block scale factors
    blocks = flat.reshape(num_rows, num_blocks, block_size)
    block_amax = blocks.abs().amax(dim=-1).clamp(min=1e-12)  # [num_rows, num_blocks]
    # Scale = amax / E2M1_MAX, stored as FP8 E4M3
    scale_float = block_amax / FLOAT4_E2M1_MAX  # [num_rows, num_blocks]

    # Clamp scale to FP8 E4M3 range
    scale_float = scale_float.clamp(max=FLOAT8_E4M3_MAX)
    # Convert to FP8 E4M3 and back to get the actual quantized scale
    scale_fp8 = scale_float.to(torch.float8_e4m3fn)
    scale_float_actual = scale_fp8.float()

    # Quantize elements to E2M1
    # For each element: value / scale -> find nearest E2M1 representable value
    scale_expanded = scale_float_actual.unsqueeze(-1).expand_as(blocks)
    normalized = blocks / scale_expanded.clamp(min=1e-12)

    # Map to nearest E2M1 value
    e2m1_magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                                    device=tensor.device)
    signs = normalized.sign()
    abs_normalized = normalized.abs()

    # Find nearest magnitude
    diffs = (abs_normalized.unsqueeze(-1) - e2m1_magnitudes.unsqueeze(0).unsqueeze(0).unsqueeze(0)).abs()
    nearest_idx = diffs.argmin(dim=-1)  # [num_rows, num_blocks, block_size]

    # Encode: lower 3 bits = magnitude index, bit 3 = sign (1 if negative)
    nibbles = nearest_idx.byte()
    neg_mask = (signs < 0).byte()
    nibbles = nibbles | (neg_mask << 3)  # 4-bit encoding

    # Reshape to [num_rows, head_dim]
    nibbles = nibbles.reshape(num_rows, head_dim)

    # Pack pairs: even index in lower nibble, odd index in upper nibble
    even = nibbles[:, 0::2]  # [num_rows, head_dim/2]
    odd = nibbles[:, 1::2]   # [num_rows, head_dim/2]
    packed = (odd << 4) | even  # [num_rows, head_dim/2]

    # Reshape back to original shape
    packed_shape = list(orig_shape)
    packed_shape[-1] = head_dim // 2
    packed = packed.reshape(packed_shape).to(torch.uint8)

    sf_shape = list(orig_shape)
    sf_shape[-1] = num_blocks
    sf = scale_fp8.reshape(sf_shape).view(torch.uint8)

    return packed, sf


def dequantize_e2m1(packed: torch.Tensor, sf: torch.Tensor, head_dim: int,
                     block_size: int = 16) -> torch.Tensor:
    """Dequantize E2M1 packed data back to float for reference comparison.

    Args:
        packed: uint8 tensor [..., head_dim/2]
        sf: uint8 tensor [..., head_dim/block_size] (FP8 E4M3 scale factors)
        head_dim: original head dimension
        block_size: elements per scale block

    Returns:
        float tensor [..., head_dim]
    """
    orig_shape = list(packed.shape)
    orig_shape[-1] = head_dim

    flat_packed = packed.reshape(-1, head_dim // 2)
    flat_sf = sf.view(torch.uint8).reshape(-1, head_dim // block_size)
    num_rows = flat_packed.shape[0]

    # Unpack nibbles
    even = (flat_packed & 0xF).long()
    odd = ((flat_packed >> 4) & 0xF).long()

    # Interleave
    nibbles = torch.zeros(num_rows, head_dim, dtype=torch.long, device=packed.device)
    nibbles[:, 0::2] = even
    nibbles[:, 1::2] = odd

    # Convert nibbles to float via LUT
    lut = torch.tensor(E2M1_LUT, device=packed.device, dtype=torch.float32)
    values = lut[nibbles]

    # Get scale factors
    sf_fp8 = flat_sf.view(torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else torch.uint8)
    if hasattr(torch, 'float8_e4m3fn'):
        sf_fp8 = flat_sf.view(torch.float8_e4m3fn)
        sf_float = sf_fp8.float()
    else:
        # Fallback: reinterpret as FP8
        sf_float = flat_sf.float()

    # Expand scale factors to match element positions
    sf_expanded = sf_float.unsqueeze(-1).expand(-1, -1, block_size).reshape(num_rows, head_dim)
    result = values * sf_expanded

    return result.reshape(orig_shape)


def attention_ref(
    batch_size,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    """Reference attention computation in FP32."""
    qo_len = q.shape[0] // batch_size
    kv_len = k.shape[0] // batch_size
    num_qo_heads = q.shape[1]
    head_dim_qk = q.shape[2]
    head_dim_vo = v.shape[2]
    logits = (
        torch.einsum(
            "bmhd,bnhd->bhmn",
            q.view(batch_size, qo_len, num_qo_heads, head_dim_qk).float(),
            k.view(batch_size, kv_len, num_qo_heads, head_dim_qk).float(),
        )
        * sm_scale
    )

    if causal:
        mask = torch.arange(kv_len - qo_len, kv_len, device=q.device).unsqueeze(
            1
        ) >= torch.arange(0, kv_len, device=q.device).unsqueeze(0)
    else:
        mask = torch.ones(qo_len, kv_len, device=q.device)

    logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
    p = torch.softmax(logits, dim=-1)
    o_ref = (
        torch.einsum(
            "bhmn,bnhd->bmhd",
            p,
            v.view(batch_size, kv_len, num_qo_heads, head_dim_vo).float(),
        )
        .contiguous()
        .view(batch_size * qo_len, num_qo_heads, head_dim_vo)
    )

    return o_ref


@pytest.mark.parametrize("batch_size", [1, 2, 9])
@pytest.mark.parametrize("qo_len", [177, 377])
@pytest.mark.parametrize("kv_len", [544, 977])
@pytest.mark.parametrize(
    "num_qo_heads,num_kv_heads",
    [
        (32, 32),
        (32, 8),
    ],
)
@pytest.mark.parametrize(
    "head_dim,sm_scale",
    [
        (128, 1.0 / math.sqrt(128)),
        (64, 1.0 / math.sqrt(64)),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_cutlass_fmha_fp4kv(
    batch_size,
    qo_len,
    kv_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    sm_scale,
    causal,
):
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")

    if (not is_sm100a_supported(torch.device("cuda"))
            and not is_sm110a_supported(torch.device("cuda"))
            and not is_sm120a_supported(torch.device("cuda"))):
        pytest.skip("Requires SM100A, SM110A, or SM120A")

    torch.manual_seed(42)

    # Generate Q in FP8 E4M3
    q_bf16 = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    q_fp8 = q_bf16.to(torch.float8_e4m3fn)

    # Generate K/V in BF16, then quantize to E2M1
    # For GQA, K/V have num_kv_heads; expand for reference computation
    k_bf16 = torch.randn(
        batch_size * kv_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    ) * 0.5  # Scale down to stay in E2M1 range
    v_bf16 = torch.randn(
        batch_size * kv_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    ) * 0.5

    # Quantize K/V to E2M1
    k_packed, k_sf = quantize_to_e2m1(k_bf16)
    v_packed, v_sf = quantize_to_e2m1(v_bf16)

    # Dequantize for reference
    k_deq = dequantize_e2m1(k_packed, k_sf, head_dim)
    v_deq = dequantize_e2m1(v_packed, v_sf, head_dim)

    # Segment offsets for ragged batching
    qo_indptr = torch.arange(0, (batch_size + 1) * qo_len, qo_len, dtype=torch.int32, device="cuda")
    kv_indptr = torch.arange(0, (batch_size + 1) * kv_len, kv_len, dtype=torch.int32, device="cuda")

    # Run CUTLASS FMHA with FP4 KV
    out, lse = fmha_varlen_fp4kv(
        q_fp8,
        k_packed,
        v_packed,
        k_sf,
        v_sf,
        qo_indptr,
        kv_indptr,
        causal=causal,
        sm_scale=sm_scale,
        return_lse=True,
    )

    assert out.dtype == torch.bfloat16, f"Expected BF16 output, got {out.dtype}"

    # Reference: expand KV heads for MHA if needed, use dequantized K/V
    h_r = num_qo_heads // num_kv_heads
    if h_r > 1:
        k_ref = k_deq.unsqueeze(2).expand(-1, -1, h_r, -1).reshape(
            batch_size * kv_len, num_qo_heads, head_dim
        )
        v_ref = v_deq.unsqueeze(2).expand(-1, -1, h_r, -1).reshape(
            batch_size * kv_len, num_qo_heads, head_dim
        )
    else:
        k_ref = k_deq
        v_ref = v_deq

    # Q for reference: convert FP8 back to float
    q_ref = q_fp8.float()

    o_ref = attention_ref(batch_size, q_ref, k_ref, v_ref, causal, sm_scale)
    o_ref_bf16 = o_ref.to(torch.bfloat16)

    # Compare with relaxed tolerance (double quantization: BF16 -> E2M1 -> FP8 -> MMA)
    torch.testing.assert_close(
        out.float(),
        o_ref_bf16.float(),
        atol=0.15,
        rtol=0.1,
    )


# ── Variable-length sequence test ─────────────────────────────────────────────


def attention_varlen_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    """Reference attention for variable-length sequences."""
    batch_size = qo_indptr.shape[0] - 1
    nnz_qo = qo_indptr[-1].item()
    num_qo_heads = q.shape[1]
    head_dim = v.shape[2]
    o = torch.empty(nnz_qo, num_qo_heads, head_dim, device=q.device, dtype=torch.float32)

    for i in range(batch_size):
        qi = q[qo_indptr[i]:qo_indptr[i + 1]]
        ki = k[kv_indptr[i]:kv_indptr[i + 1]]
        vi = v[kv_indptr[i]:kv_indptr[i + 1]]
        o_i = attention_ref(1, qi, ki, vi, causal, sm_scale)
        o[qo_indptr[i]:qo_indptr[i + 1]] = o_i

    return o


# A subset of representative variable-length patterns
VARLEN_INDPTR_PARAMS = [
    [0, 5, 20, 45],           # 3 sequences: 5, 15, 25 tokens
    [0, 1, 2, 3, 4, 5],       # 5 single-token sequences (decode-like)
    [0, 100, 200, 300],        # 3 equal-length sequences
    [0, 1, 129],               # 1-token + 128-token (tests tile boundary)
    [0, 50],                   # single sequence
]


@pytest.mark.parametrize("qo_indptr_list", VARLEN_INDPTR_PARAMS)
@pytest.mark.parametrize(
    "num_qo_heads,num_kv_heads",
    [
        (32, 32),
        (32, 8),
    ],
)
@pytest.mark.parametrize(
    "head_dim,sm_scale",
    [
        (128, 1.0 / math.sqrt(128)),
        (64, 1.0 / math.sqrt(64)),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_cutlass_fmha_fp4kv_varlen(
    qo_indptr_list,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    sm_scale,
    causal,
):
    """Test CUTLASS FMHA with E2M1 KV on variable-length sequences."""
    if (not is_sm100a_supported(torch.device("cuda"))
            and not is_sm110a_supported(torch.device("cuda"))
            and not is_sm120a_supported(torch.device("cuda"))):
        pytest.skip("Requires SM100A, SM110A, or SM120A")

    torch.manual_seed(42)

    # Use same indptr for Q/O and K/V (self-attention)
    qo_indptr = torch.tensor(qo_indptr_list, dtype=torch.int32, device="cuda")
    kv_indptr = qo_indptr
    total_tokens = qo_indptr_list[-1]

    if total_tokens == 0:
        pytest.skip("Empty sequence")

    # Generate Q in FP8 E4M3
    q_bf16 = torch.randn(
        total_tokens, num_qo_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    q_fp8 = q_bf16.to(torch.float8_e4m3fn)

    # Generate K/V in BF16, quantize to E2M1
    k_bf16 = torch.randn(
        total_tokens, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    ) * 0.5
    v_bf16 = torch.randn(
        total_tokens, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    ) * 0.5

    k_packed, k_sf = quantize_to_e2m1(k_bf16)
    v_packed, v_sf = quantize_to_e2m1(v_bf16)
    k_deq = dequantize_e2m1(k_packed, k_sf, head_dim)
    v_deq = dequantize_e2m1(v_packed, v_sf, head_dim)

    # Run CUTLASS FMHA
    out, lse = fmha_varlen_fp4kv(
        q_fp8,
        k_packed,
        v_packed,
        k_sf,
        v_sf,
        qo_indptr,
        kv_indptr,
        causal=causal,
        sm_scale=sm_scale,
        return_lse=True,
    )

    assert out.dtype == torch.bfloat16
    assert out.shape == (total_tokens, num_qo_heads, head_dim)

    # Reference: expand KV heads for MHA if needed
    h_r = num_qo_heads // num_kv_heads
    if h_r > 1:
        k_ref = k_deq.unsqueeze(2).expand(-1, -1, h_r, -1).reshape(
            total_tokens, num_qo_heads, head_dim
        )
        v_ref = v_deq.unsqueeze(2).expand(-1, -1, h_r, -1).reshape(
            total_tokens, num_qo_heads, head_dim
        )
    else:
        k_ref = k_deq
        v_ref = v_deq

    q_ref = q_fp8.float()

    o_ref = attention_varlen_ref(
        q_ref, k_ref, v_ref, qo_indptr, kv_indptr, causal, sm_scale
    )
    o_ref_bf16 = o_ref.to(torch.bfloat16)

    torch.testing.assert_close(
        out.float(),
        o_ref_bf16.float(),
        atol=0.15,
        rtol=0.1,
    )


# ── Different QO/KV length test ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "qo_indptr_list,kv_indptr_list",
    [
        ([0, 10, 20, 30], [0, 50, 100, 150]),      # QO shorter than KV
        ([0, 1, 2, 3, 4], [0, 100, 200, 300, 400]), # decode-like: 1 Q, many KV
    ],
)
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(32, 8)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [True, False])
def test_cutlass_fmha_fp4kv_qo_kv_varlen(
    qo_indptr_list,
    kv_indptr_list,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    causal,
):
    """Test CUTLASS FMHA with E2M1 KV where QO and KV have different lengths."""
    if (not is_sm100a_supported(torch.device("cuda"))
            and not is_sm110a_supported(torch.device("cuda"))
            and not is_sm120a_supported(torch.device("cuda"))):
        pytest.skip("Requires SM100A, SM110A, or SM120A")

    torch.manual_seed(42)
    sm_scale = 1.0 / math.sqrt(head_dim)

    qo_indptr = torch.tensor(qo_indptr_list, dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor(kv_indptr_list, dtype=torch.int32, device="cuda")
    total_qo = qo_indptr_list[-1]
    total_kv = kv_indptr_list[-1]

    q_bf16 = torch.randn(
        total_qo, num_qo_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    q_fp8 = q_bf16.to(torch.float8_e4m3fn)

    k_bf16 = torch.randn(
        total_kv, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    ) * 0.5
    v_bf16 = torch.randn(
        total_kv, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    ) * 0.5

    k_packed, k_sf = quantize_to_e2m1(k_bf16)
    v_packed, v_sf = quantize_to_e2m1(v_bf16)
    k_deq = dequantize_e2m1(k_packed, k_sf, head_dim)
    v_deq = dequantize_e2m1(v_packed, v_sf, head_dim)

    out, lse = fmha_varlen_fp4kv(
        q_fp8,
        k_packed,
        v_packed,
        k_sf,
        v_sf,
        qo_indptr,
        kv_indptr,
        causal=causal,
        sm_scale=sm_scale,
        return_lse=True,
    )

    assert out.dtype == torch.bfloat16
    assert out.shape == (total_qo, num_qo_heads, head_dim)

    h_r = num_qo_heads // num_kv_heads
    if h_r > 1:
        k_ref = k_deq.unsqueeze(2).expand(-1, -1, h_r, -1).reshape(
            total_kv, num_qo_heads, head_dim
        )
        v_ref = v_deq.unsqueeze(2).expand(-1, -1, h_r, -1).reshape(
            total_kv, num_qo_heads, head_dim
        )
    else:
        k_ref = k_deq
        v_ref = v_deq

    q_ref = q_fp8.float()

    o_ref = attention_varlen_ref(
        q_ref, k_ref, v_ref, qo_indptr, kv_indptr, causal, sm_scale
    )
    o_ref_bf16 = o_ref.to(torch.bfloat16)

    torch.testing.assert_close(
        out.float(),
        o_ref_bf16.float(),
        atol=0.15,
        rtol=0.1,
    )


# ── Output dtype and shape tests ─────────────────────────────────────────────


def test_cutlass_fmha_fp4kv_output_dtype():
    """Verify that the output is always BF16 regardless of input FP8 Q."""
    if (not is_sm100a_supported(torch.device("cuda"))
            and not is_sm110a_supported(torch.device("cuda"))
            and not is_sm120a_supported(torch.device("cuda"))):
        pytest.skip("Requires SM100A, SM110A, or SM120A")

    torch.manual_seed(42)
    batch_size, qo_len, kv_len = 1, 16, 32
    num_qo_heads, num_kv_heads, head_dim = 8, 8, 64

    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim,
        dtype=torch.bfloat16, device="cuda"
    ).to(torch.float8_e4m3fn)

    k_bf16 = torch.randn(
        batch_size * kv_len, num_kv_heads, head_dim,
        dtype=torch.bfloat16, device="cuda"
    ) * 0.5
    v_bf16 = torch.randn(
        batch_size * kv_len, num_kv_heads, head_dim,
        dtype=torch.bfloat16, device="cuda"
    ) * 0.5

    k_packed, k_sf = quantize_to_e2m1(k_bf16)
    v_packed, v_sf = quantize_to_e2m1(v_bf16)

    qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device="cuda")

    out, lse = fmha_varlen_fp4kv(
        q, k_packed, v_packed, k_sf, v_sf,
        qo_indptr, kv_indptr,
        causal=False,
        return_lse=True,
    )

    assert out.dtype == torch.bfloat16, f"Expected BF16, got {out.dtype}"
    assert out.shape == (batch_size * qo_len, num_qo_heads, head_dim)
    assert lse.dtype == torch.float32
    assert lse.shape == (batch_size * qo_len, num_qo_heads)
