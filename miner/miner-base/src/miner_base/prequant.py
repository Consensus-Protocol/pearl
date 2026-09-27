"""The miner's prequantized input dtype: int8 values + per-block BF16 scales.

``int8 blk8 bf16s``: each row is split into contiguous blocks of 8 elements;
a block stores 8 quantized int8 values plus one shared BF16 scale, and opens
(dequantizes) to BF16 as ``int_value * scale``. The commitment hashes the two
tensors separately and combines the digests --
``blake3(commit(int_values) || commit(scales))`` -- see
``commitment.commit_planes``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

DEFAULT_BITS = 8  # int8 values
DEFAULT_BLOCK_SIZE = 8  # one scale per 1x8 block
L2_ROUNDED_BITS = 2  # low explicit BF16 mantissa bits cleared from l2 (see round_l2_to_grid)
L2_FRAME_WIDTH_CONSTANT = 27  # Wl2 = 27 - ceil(log2 k)


def _qmax(bits: int) -> int:
    """Largest stored magnitude for ``bits``-bit signed values (symmetric grid)."""
    return 2 ** (bits - 1) - 1


def _l2_frame_width(k: int) -> int:
    """Wl2 = 27 - ceil(log2 k): the precision shift of the block-integer L2 frame sum."""
    ceil_log2_k = (k - 1).bit_length()
    assert ceil_log2_k <= L2_FRAME_WIDTH_CONSTANT, f"k={k} exceeds the L2 frame width"
    return L2_FRAME_WIDTH_CONSTANT - ceil_log2_k


def _decode_bf16_scale(code: int) -> tuple[int, int]:
    """Decode a BF16 scale code into (M, 2*E*) for the integer L2 formula.

    |scale| = M * 2^(E* - 134); returns (M, 2*E*).
    """
    exp_field = (code & 0x7FFF) >> 7
    m = (code & 0x7F) + (128 if exp_field != 0 else 0)
    doubled_exp = 2 * max(exp_field, 1)
    return m, doubled_exp


def _cmp_shifted(s: int, d: int, c: int) -> int:
    """Compare s * 2^d with c, returning -1, 0, or 1."""
    if s == 0:
        return -1
    if d >= 0:
        if d >= 66:
            return 1  # s >= 1, so s*2^66 > 2^57 >= c
        lhs = s << d
    else:
        g = -d
        if g >= 71:
            return -1  # c >= 1, so c*2^71 > 2^62 > s
        lhs = s
        c = c << g
    if lhs < c:
        return -1
    elif lhs > c:
        return 1
    return 0


def _sqrt_bracket_ok(s: int, e_max: int, wl2: int, k: int, t: int, f: int) -> tuple[bool, bool]:
    """Check if (t, f) satisfies the RNE bracket conditions for the given frame sum."""
    bottom = 1 if (t == 128 and f >= 2) else 0
    b_lo = 4 * t - 2 + bottom
    b_hi = 4 * t + 2
    odd = (t & 1) == 1
    d = e_max + 4 - wl2 - 2 * f
    hi = _cmp_shifted(s, d, b_hi * b_hi * k)
    hi_ok = hi < 0 or (hi == 0 and not odd)
    if t == 0:
        return True, hi_ok
    lo = _cmp_shifted(s, d, b_lo * b_lo * k)
    return lo > 0 or (lo == 0 and not odd), hi_ok


def _rne_sqrt_hat(s: int, e_max: int, wl2: int, k: int) -> tuple[int, int]:
    """RNE_bf16(sqrt(v_hat)) where v_hat = s * 2^(e_max - 14 - wl2) / k, s > 0.

    Returns (exp, mantissa) such that the BF16 code is (exp << 7) | mantissa.
    """
    assert s > 0
    log2y = (math.log2(s) - math.log2(k) + (e_max - 14 - wl2)) / 2.0
    f = int(math.floor(log2y))
    if f < 1:
        f = 1
        t = int(round(2.0**log2y * 64.0))
        t = max(0, min(128, t))
    else:
        t = int(round(2.0 ** (log2y - f) * 128.0))
        if t >= 256:
            t = 128
            f += 1
        t = max(128, t)
    assert f <= 254, "sqrt exceeds the bf16 range"

    for _ in range(1024):
        lo_ok, hi_ok = _sqrt_bracket_ok(s, e_max, wl2, k, t, f)
        if lo_ok and hi_ok:
            if t >= 128:
                return f, t - 128
            else:
                assert f == 1
                return 0, t
        if not lo_ok:
            assert t > 0
            t -= 1
            if t == 127 and f >= 2:
                t = 255
                f -= 1
        else:
            t += 1
            if t == 256:
                assert f < 254
                t = 128
                f += 1
    raise RuntimeError(
        f"sqrt bracket search did not converge (s={s}, e_max={e_max}, wl2={wl2}, k={k})"
    )


def _round_l2_code_to_grid(l2_code: int) -> int:
    """Round a BF16 code to the nearest multiple of 2^L2_ROUNDED_BITS ulps (ties up)."""
    step = 1 << L2_ROUNDED_BITS
    return ((l2_code + (step >> 1)) & ~(step - 1)) & 0xFFFF


@dataclass(frozen=True)
class RowNorms:
    """Per-row ``(l2, linf)``, BF16 ``(n x 1)`` each, feeding
    `~.quantization.Fp8QuantScheme.row_norms`. ``l2`` is
    ``rms(X) = sqrt(sumsq / k)``, grid-rounded (:func:`round_l2_to_grid`);
    both are computed by the caller straight from an operand's
    pre-BF16-rounding values, right where its ``sumsq`` is produced (see
    :meth:`PrequantMatrix.exact_norms`)."""

    l2: torch.Tensor
    linf: torch.Tensor


@dataclass
class PrequantMatrix:
    """A prequantized 2D operand.

    ``int_values``: (n x k) int8, each in ``[-qmax, qmax]``.
    ``scales``: (n x k/block_size) BF16, one positive scale per block.
    """

    int_values: torch.Tensor
    scales: torch.Tensor
    block_size: int = DEFAULT_BLOCK_SIZE
    bits: int = DEFAULT_BITS

    def __post_init__(self) -> None:
        assert self.int_values.dim() == 2, "int_values must be 2D"
        assert self.int_values.dtype == torch.int8, (
            f"int_values must be int8, got {self.int_values.dtype}"
        )
        n, k = self.int_values.shape
        assert k % self.block_size == 0, f"k={k} must be a multiple of block_size={self.block_size}"
        assert self.scales.dtype == torch.bfloat16, f"scales must be BF16, got {self.scales.dtype}"
        assert self.scales.shape == (n, k // self.block_size), (
            f"scales shape {tuple(self.scales.shape)} != ({n}, {k // self.block_size})"
        )

    @property
    def shape(self) -> tuple[int, int]:
        """The shape (n, k) of the operand this opens to."""
        return (self.int_values.shape[0], self.int_values.shape[1])

    @property
    def qmax(self) -> int:
        """Largest stored magnitude: 127 for int8 (the grid is symmetric)."""
        return _qmax(self.bits)

    def planes(self) -> list[torch.Tensor]:
        """The tensors to commit to, in order: int_values, scales."""
        return [self.int_values, self.scales]

    def open(self) -> torch.Tensor:
        """Dequantize to the (n x k) BF16 operand: ``int_value * scale`` per block.

        The product is exact in FP32 (int8 x BF16 fits easily), so the only
        rounding is the final cast to BF16.
        """
        n, k = self.shape
        blocks = self.int_values.to(torch.float32).reshape(n, -1, self.block_size)
        opened = blocks * self.scales.to(torch.float32).unsqueeze(-1)
        return opened.reshape(n, k).to(torch.bfloat16)

    def exact_norms(self) -> RowNorms:
        """`RowNorms` computed directly from the int8 blocks + BF16 scales,
        bypassing :meth:`open`'s per-element BF16 rounding. Bit-identical to
        the ZK certificate (InputQuantStark group B + ScaleStark groups Q and G).

        ``l2`` is ``grid4(RNE_bf16(sqrt(v)))`` of the canonical mean square
        ``v = S * 2^(E_MAX - 268 - Wl2) / k``, where per block
        ``p_b = M(scale_b)^2 * sum(int_i^2)``, ``E_MAX = max 2*E*(scale_b)``
        over blocks with ``p_b != 0``, and
        ``S = sum floor(p_b * 2^Wl2 / 2^(E_MAX - 2*E*(scale_b)))``
        over the same blocks — exact integers, one rounding in the square root.

        ``linf`` is ``RNE_bf16(max(|int_i| * |scale|))``, exact in f32 before
        the cast.
        """
        n, k = self.shape
        n_blocks = k // self.block_size
        wl2 = _l2_frame_width(k)

        int_values_np = self.int_values.numpy()
        scales_bits = self.scales.view(torch.int16).numpy()
        scales_f32 = self.scales.to(torch.float32)

        l2_list = []
        linf_list = []

        for row_idx in range(n):
            row_ints = int_values_np[row_idx]
            row_scale_bits = scales_bits[row_idx]
            row_scales_f32 = scales_f32[row_idx]

            e_max = 0
            linf_val = 0.0

            for b in range(n_blocks):
                block_start = b * self.block_size
                block_end = block_start + self.block_size
                block_ints = row_ints[block_start:block_end]
                code = int(row_scale_bits[b]) & 0xFFFF

                amax = int(max(abs(int(v)) for v in block_ints))
                m, doubled_exp = _decode_bf16_scale(code)

                if amax != 0 and m != 0:
                    e_max = max(e_max, doubled_exp)

                scale_abs = abs(float(row_scales_f32[b]))
                linf_val = max(linf_val, float(amax) * scale_abs)

            s = 0
            for b in range(n_blocks):
                block_start = b * self.block_size
                block_end = block_start + self.block_size
                block_ints = row_ints[block_start:block_end]
                code = int(row_scale_bits[b]) & 0xFFFF

                m, doubled_exp = _decode_bf16_scale(code)
                block_sumsq = sum(int(v) ** 2 for v in block_ints)
                p = m * m * block_sumsq

                shift = e_max - doubled_exp
                if shift < 64:
                    s += (p << wl2) >> shift
                # else: shift saturates to 0 contribution

            if s == 0:
                l2_code = 0
            else:
                exp, mantissa = _rne_sqrt_hat(s, e_max, wl2, k)
                l2_code = (exp << 7) + mantissa
                l2_code = _round_l2_code_to_grid(l2_code)
            assert l2_code < 0x7F80, "l2 snaps into the bf16 infinity code"

            l2_list.append(l2_code)
            linf_list.append(linf_val)

        l2_tensor = torch.tensor(l2_list, dtype=torch.int16).view(torch.bfloat16).unsqueeze(1)
        linf_tensor = torch.tensor(linf_list, dtype=torch.float32).to(torch.bfloat16).unsqueeze(1)

        return RowNorms(l2=l2_tensor, linf=linf_tensor)

    @classmethod
    def encode(cls, rows: torch.Tensor) -> PrequantMatrix:
        """Quantize a (n x k) matrix into this format.

        Per block: ``scale = bf16(amax * reciprocal(qmax))``, ``int_value =
        round(x * reciprocal(scale))`` clamped to ``[-qmax, qmax]``, where
        ``reciprocal`` is the correctly-rounded FP32 reciprocal. A scale
        that rounds below ``amax / qmax`` trims the block max by at most
        half a step via the clamp. All-zero blocks get a tiny scale floor
        so the per-block reciprocal stays well-defined (and FP32-normal).

        The scale multiply is the fast form of ``bf16(amax / qmax)``: for
        every reachable amax (a BF16 magnitude at or above the floor) the
        FP32 quotient sits far enough from any BF16 rounding midpoint that
        the BF16 cast absorbs the multiply's error, so the two produce
        identical bits (``tests/test_prequant_encode.py`` checks the whole
        domain exhaustively). Device implementations may take further
        bit-preserving shortcuts on the same grounds: ``rcp.approx.f32``
        plus one Newton step reproduces the correctly-rounded reciprocal
        for every BF16-precision normal scale, and ``|x *
        reciprocal(scale)| <= qmax * (1 + 2^-9 + O(2^-24)) < qmax + 0.5``
        means a round-to-nearest-even saturating int8 convert reproduces
        ``round().clamp(-qmax, qmax)`` (no -128 is ever produced).

        Reference *suggestion*, not a protocol requirement: a miner may
        derive ``int_values``/``scales`` by any procedure, so long as the
        result is a valid `PrequantMatrix`.
        """
        block_size = cls.block_size
        bits = cls.bits
        assert rows.dim() == 2
        n, k = rows.shape
        assert k % block_size == 0, f"k={k} must be a multiple of block_size={block_size}"
        qmax = _qmax(bits)

        blocks = rows.to(torch.float32).reshape(n, -1, block_size)
        amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(2.0**-100)
        inv_qmax = torch.reciprocal(torch.tensor(float(qmax), dtype=torch.float32))
        scales = (amax * inv_qmax).to(torch.bfloat16)  # (n x n_blocks x 1)
        # Correctly-rounded FP32 reciprocal, matching the device's rcp.rn.f32;
        # the scale floor keeps it finite and normal at either extreme.
        inv_scales = torch.reciprocal(scales.to(torch.float32))
        int_values = (blocks * inv_scales).round().clamp(-qmax, qmax)
        return cls(
            int_values=int_values.reshape(n, k).to(torch.int8),
            scales=scales.squeeze(-1),
            block_size=block_size,
            bits=bits,
        )


def round_l2_to_grid(l2: torch.Tensor) -> torch.Tensor:
    """Round non-negative BF16 ``l2`` to the nearest multiple of ``2**L2_ROUNDED_BITS`` ulps (ties up).

    Non-negative IEEE-754 floats order like their bit patterns, so this is
    "add half a grid step to the int16 view, clear the low bits" (any carry
    into the mantissa/exponent falls out for free). Vs. ceiling to the same
    grid: equally robust to a 1-ulp host discrepancy, but half the mean bias
    (1.5 -> 0.5 ulp), removing ceiling's ~0.7% high bias on ``beta`` (hence
    on the injected noise-to-signal). Called wherever a row's exact ``sumsq``
    is turned into ``l2`` (see `RowNorms`), so a <=1-ulp FP32 discrepancy in
    ``sumsq`` between miner and verifier can't change the result.
    """
    step = 1 << L2_ROUNDED_BITS
    bits = l2.view(torch.int16)
    return ((bits + (step >> 1)) & -step).view(torch.bfloat16)
