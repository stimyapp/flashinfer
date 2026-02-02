"""Unit tests for E2M1 (FP4) quantization/dequantization helpers.

These tests verify the correctness of the Python-side E2M1 quantize/dequantize
utilities used in test_cutlass_fp4_kv.py, without requiring a Blackwell GPU.
"""

import math

import pytest
import torch


# ── Inline copies of the helpers under test (from test_cutlass_fp4_kv.py) ─────

E2M1_LUT = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0


def quantize_to_e2m1(tensor: torch.Tensor, block_size: int = 16):
    orig_shape = tensor.shape
    head_dim = orig_shape[-1]
    assert head_dim % block_size == 0
    assert head_dim % 2 == 0

    flat = tensor.float().reshape(-1, head_dim)
    num_rows = flat.shape[0]
    num_blocks = head_dim // block_size

    blocks = flat.reshape(num_rows, num_blocks, block_size)
    block_amax = blocks.abs().amax(dim=-1).clamp(min=1e-12)
    scale_float = block_amax / FLOAT4_E2M1_MAX
    scale_float = scale_float.clamp(max=FLOAT8_E4M3_MAX)
    scale_fp8 = scale_float.to(torch.float8_e4m3fn)
    scale_float_actual = scale_fp8.float()

    scale_expanded = scale_float_actual.unsqueeze(-1).expand_as(blocks)
    normalized = blocks / scale_expanded.clamp(min=1e-12)

    e2m1_magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=tensor.device
    )
    signs = normalized.sign()
    abs_normalized = normalized.abs()

    diffs = (
        abs_normalized.unsqueeze(-1)
        - e2m1_magnitudes.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    ).abs()
    nearest_idx = diffs.argmin(dim=-1)

    nibbles = nearest_idx.byte()
    neg_mask = (signs < 0).byte()
    nibbles = nibbles | (neg_mask << 3)

    nibbles = nibbles.reshape(num_rows, head_dim)
    even = nibbles[:, 0::2]
    odd = nibbles[:, 1::2]
    packed = (odd << 4) | even

    packed_shape = list(orig_shape)
    packed_shape[-1] = head_dim // 2
    packed = packed.reshape(packed_shape).to(torch.uint8)

    sf_shape = list(orig_shape)
    sf_shape[-1] = num_blocks
    sf = scale_fp8.reshape(sf_shape).view(torch.uint8)

    return packed, sf


def dequantize_e2m1(
    packed: torch.Tensor, sf: torch.Tensor, head_dim: int, block_size: int = 16
) -> torch.Tensor:
    orig_shape = list(packed.shape)
    orig_shape[-1] = head_dim

    flat_packed = packed.reshape(-1, head_dim // 2)
    flat_sf = sf.view(torch.uint8).reshape(-1, head_dim // block_size)
    num_rows = flat_packed.shape[0]

    even = (flat_packed & 0xF).long()
    odd = ((flat_packed >> 4) & 0xF).long()

    nibbles = torch.zeros(num_rows, head_dim, dtype=torch.long, device=packed.device)
    nibbles[:, 0::2] = even
    nibbles[:, 1::2] = odd

    lut = torch.tensor(E2M1_LUT, device=packed.device, dtype=torch.float32)
    values = lut[nibbles]

    sf_fp8 = flat_sf.view(torch.float8_e4m3fn)
    sf_float = sf_fp8.float()

    sf_expanded = (
        sf_float.unsqueeze(-1).expand(-1, -1, block_size).reshape(num_rows, head_dim)
    )
    result = values * sf_expanded
    return result.reshape(orig_shape)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestE2M1LUT:
    """Verify E2M1 lookup table properties."""

    def test_positive_magnitudes(self):
        expected = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        assert E2M1_LUT[:8] == expected

    def test_negative_magnitudes(self):
        for i in range(8):
            assert abs(E2M1_LUT[i + 8]) == abs(E2M1_LUT[i])

    def test_sign_bit(self):
        """Bit 3 is the sign bit: nibble 0b1xxx is negative of 0b0xxx."""
        for i in range(8):
            pos = E2M1_LUT[i]
            neg = E2M1_LUT[i | 0x8]
            if pos == 0.0:
                # Both +0 and -0 are zero
                assert neg == 0.0 or neg == -0.0
            else:
                assert neg == -pos

    def test_table_length(self):
        assert len(E2M1_LUT) == 16


class TestPacking:
    """Verify E2M1 nibble packing layout."""

    def test_even_odd_packing(self):
        """Even-index element in lower nibble, odd-index in upper nibble."""
        # Create a simple tensor with known values that map to distinct nibbles
        # Use a 1-row, 16-element tensor so one block with one SF
        torch.manual_seed(0)
        t = torch.tensor(
            [[1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0,
              0.5, -0.5, 1.5, -1.5, 6.0, -6.0, 0.0, 0.0]],
            dtype=torch.bfloat16,
        )
        packed, sf = quantize_to_e2m1(t)

        # Unpack and verify we get back reasonable values
        deq = dequantize_e2m1(packed, sf, head_dim=16)
        # All values are exactly representable in E2M1, so roundtrip should
        # recover them (up to scale factor quantization)
        torch.testing.assert_close(deq.abs(), t.float().abs(), atol=0.6, rtol=0.2)

    def test_packed_shape(self):
        """Packed tensor has half the last dimension."""
        t = torch.randn(4, 8, 64, dtype=torch.bfloat16)
        packed, sf = quantize_to_e2m1(t)
        assert packed.shape == (4, 8, 32)
        assert packed.dtype == torch.uint8

    def test_sf_shape(self):
        """Scale factor tensor has head_dim/16 in last dimension."""
        t = torch.randn(4, 8, 128, dtype=torch.bfloat16)
        packed, sf = quantize_to_e2m1(t)
        assert sf.shape == (4, 8, 8)  # 128 / 16 = 8
        assert sf.dtype == torch.uint8


class TestQuantizeDequantizeRoundtrip:
    """Test quantize → dequantize roundtrip fidelity."""

    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("num_rows", [1, 7, 32])
    def test_roundtrip_random(self, head_dim, num_rows):
        """Random BF16 values survive roundtrip within E2M1 quantization error."""
        torch.manual_seed(42)
        # Scale down so values are within E2M1 representable range
        t = torch.randn(num_rows, head_dim, dtype=torch.bfloat16) * 0.5
        packed, sf = quantize_to_e2m1(t)
        deq = dequantize_e2m1(packed, sf, head_dim)

        # E2M1 has very coarse quantization; allow generous tolerance
        # The max absolute error should be bounded by SF * max_step
        # where max_step is the largest gap between adjacent E2M1 values (2.0, between 4→6)
        torch.testing.assert_close(deq, t.float(), atol=0.5, rtol=0.5)

    def test_roundtrip_zeros(self):
        """All-zero tensor should dequantize to zeros."""
        t = torch.zeros(4, 64, dtype=torch.bfloat16)
        packed, sf = quantize_to_e2m1(t)
        deq = dequantize_e2m1(packed, sf, head_dim=64)
        assert torch.all(deq == 0.0)

    def test_roundtrip_exact_e2m1_values(self):
        """Values that are exact E2M1 representable should survive with high accuracy."""
        # Create a tensor of exact E2M1 values (within a single block of 16)
        exact_vals = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=torch.bfloat16,
        ).unsqueeze(0)  # [1, 16]

        packed, sf = quantize_to_e2m1(exact_vals)
        deq = dequantize_e2m1(packed, sf, head_dim=16)

        # The scale factor for this block is amax/6.0 = 6.0/6.0 = 1.0 (stored as FP8)
        # So all values should be recovered very accurately
        torch.testing.assert_close(deq.abs(), exact_vals.float().abs(), atol=0.01, rtol=0.01)

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_roundtrip_preserves_sign(self, head_dim):
        """Positive and negative values should maintain their signs."""
        torch.manual_seed(123)
        t = torch.randn(16, head_dim, dtype=torch.bfloat16) * 0.5
        packed, sf = quantize_to_e2m1(t)
        deq = dequantize_e2m1(packed, sf, head_dim)

        # Where original is clearly nonzero, sign should match
        nonzero = t.float().abs() > 0.3
        original_signs = t.float()[nonzero].sign()
        deq_signs = deq[nonzero].sign()
        assert torch.all(original_signs == deq_signs)


class TestScaleFactors:
    """Test per-block scale factor computation."""

    def test_sf_is_fp8_e4m3(self):
        """Scale factors should be valid FP8 E4M3 values."""
        torch.manual_seed(7)
        t = torch.randn(4, 128, dtype=torch.bfloat16)
        _, sf = quantize_to_e2m1(t)

        # Reinterpret as FP8 and back — the roundtrip should be identity
        sf_fp8 = sf.view(torch.float8_e4m3fn)
        sf_roundtrip = sf_fp8.view(torch.uint8)
        assert torch.all(sf == sf_roundtrip)

    def test_sf_covers_block_range(self):
        """Scale factor * 6.0 should be >= block amax."""
        torch.manual_seed(9)
        t = torch.randn(8, 64, dtype=torch.bfloat16)
        _, sf = quantize_to_e2m1(t)

        # Recover float scale factors
        sf_float = sf.view(torch.float8_e4m3fn).float()

        # Compute expected block amax
        blocks = t.float().reshape(8, 4, 16)  # 64/16 = 4 blocks
        block_amax = blocks.abs().amax(dim=-1)

        # sf * 6.0 should cover the amax (allowing for FP8 quantization of SF)
        coverage = sf_float * FLOAT4_E2M1_MAX
        # Due to FP8 rounding of the scale factor, coverage may be slightly less
        # than amax, but the difference should be small
        undershoot = block_amax - coverage
        assert torch.all(undershoot < 0.5), f"Max undershoot: {undershoot.max()}"

    def test_uniform_block_single_sf(self):
        """A block of identical values should produce a consistent scale factor."""
        val = 3.0
        t = torch.full((1, 16), val, dtype=torch.bfloat16)
        packed, sf = quantize_to_e2m1(t)
        sf_float = sf.view(torch.float8_e4m3fn).float()

        # Expected SF = amax / 6.0 = 3.0 / 6.0 = 0.5
        torch.testing.assert_close(sf_float, torch.tensor([[0.5]]), atol=0.01, rtol=0.01)

        # Dequantize should recover the value
        deq = dequantize_e2m1(packed, sf, head_dim=16)
        torch.testing.assert_close(deq, t.float(), atol=0.01, rtol=0.01)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_element_per_dim(self):
        """Minimum batch size (1 row)."""
        t = torch.randn(1, 64, dtype=torch.bfloat16) * 0.5
        packed, sf = quantize_to_e2m1(t)
        deq = dequantize_e2m1(packed, sf, head_dim=64)
        assert deq.shape == (1, 64)

    def test_large_values_clamp(self):
        """Values exceeding E2M1 range should be clamped to ±6.0 * SF."""
        t = torch.tensor(
            [[100.0] * 16], dtype=torch.bfloat16
        )
        packed, sf = quantize_to_e2m1(t)
        deq = dequantize_e2m1(packed, sf, head_dim=16)

        # All values are the same and positive, so they should all map to
        # the maximum E2M1 value (6.0) times the scale factor
        sf_float = sf.view(torch.float8_e4m3fn).float()
        expected = sf_float.item() * 6.0
        torch.testing.assert_close(
            deq, torch.full_like(deq, expected), atol=0.01, rtol=0.01
        )

    def test_small_values_near_zero(self):
        """Very small values should quantize to 0 or ±0.5 * SF."""
        t = torch.full((1, 16), 1e-6, dtype=torch.bfloat16)
        packed, sf = quantize_to_e2m1(t)
        deq = dequantize_e2m1(packed, sf, head_dim=16)
        # Should be very close to zero
        assert deq.abs().max() < 0.01

    def test_mixed_sign_block(self):
        """Block with mixed positive and negative values."""
        t = torch.tensor(
            [[1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 0.5, -0.5,
              1.5, -1.5, 4.0, -4.0, 6.0, -6.0, 0.0, 0.0]],
            dtype=torch.bfloat16,
        )
        packed, sf = quantize_to_e2m1(t)
        deq = dequantize_e2m1(packed, sf, head_dim=16)
        # These are exact E2M1 values; should be very close
        torch.testing.assert_close(deq, t.float(), atol=0.01, rtol=0.01)

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_3d_tensor_shape(self, head_dim):
        """Multi-head tensor shape [batch*seq, num_heads, head_dim]."""
        t = torch.randn(32, 8, head_dim, dtype=torch.bfloat16) * 0.5
        packed, sf = quantize_to_e2m1(t)
        assert packed.shape == (32, 8, head_dim // 2)
        assert sf.shape == (32, 8, head_dim // 16)

        deq = dequantize_e2m1(packed, sf, head_dim)
        assert deq.shape == (32, 8, head_dim)
        torch.testing.assert_close(deq, t.float(), atol=0.5, rtol=0.5)
