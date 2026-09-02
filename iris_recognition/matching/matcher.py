"""
Iris Template Matcher — Masked Hamming Distance.

For two binary iris codes A and B:

    HD = (number of differing valid bits) / (number of valid bits)

where "valid bits" are those marked 1 in both noise masks.

Rotation compensation:
    Eye torsion causes a cyclic shift in the normalised iris strip.
    We compute HD for pixel_shift ∈ {−N, …, +N} and return the minimum.
    Shift is applied column-wise, independently within each scale block,
    to match the Log-Gabor block layout:

        Layout: [scale_0_real | scale_0_imag | scale_1_real | scale_1_imag | …]
        Each block width = norm_width columns.
        A shift of +k pixels → roll by +k columns within each block.
"""

import numpy as np
from iris_recognition.matching.hamming import masked_hamming_distance


class IrisMatcher:
    """Hamming-distance iris template matcher with rotation invariance."""

    def __init__(self, config=None):
        cfg          = (config or {}).get("matching", {})
        feat_cfg     = (config or {}).get("features", {}).get("log_gabor", {})
        self.rotation_shift = cfg.get("rotation_shift", 10)
        self.num_scales     = feat_cfg.get("num_scales", 4)
        self.min_valid_frac = cfg.get("min_valid_frac", 0.15)  # minimum fraction of valid bits

    # ------------------------------------------------------------------
    # Shift helper
    # ------------------------------------------------------------------

    def _shift_code(
        self,
        code: np.ndarray,
        mask: np.ndarray,
        pixel_shift: int,
        norm_width: int,
    ):
        """Cyclically shift code and mask within each scale block.

        Block layout assumption (from log_gabor.py):
            block s = [real_s (norm_width cols) | imag_s (norm_width cols)]

        A pixel shift of k shifts by k columns within each block.
        """
        bits_per_scale = norm_width * 2   # real + imag
        shifted_code = np.empty_like(code)
        shifted_mask = np.empty_like(mask)

        for s in range(self.num_scales):
            start = s * bits_per_scale
            end   = start + bits_per_scale
            shifted_code[:, start:end] = np.roll(code[:, start:end], pixel_shift, axis=1)
            shifted_mask[:, start:end] = np.roll(mask[:, start:end], pixel_shift, axis=1)

        return shifted_code, shifted_mask

    # ------------------------------------------------------------------
    # Hamming distance
    # ------------------------------------------------------------------

    def compute_hd(
        self,
        code_a: np.ndarray,
        mask_a: np.ndarray,
        code_b: np.ndarray,
        mask_b: np.ndarray,
    ):
        """Compute the minimum masked Hamming distance across rotation shifts.

        Delegates to the fast vectorized implementation in hamming.py.
        """
        return masked_hamming_distance(
            code_a, mask_a, code_b, mask_b,
            max_shift=self.rotation_shift,
            num_scales=self.num_scales,
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    NUM_SCALES  = 4
    NORM_WIDTH  = 512
    H           = 64
    TOTAL_BITS  = NUM_SCALES * 2 * NORM_WIDTH   # 4096

    cfg = {
        "matching":  {"rotation_shift": 10, "min_valid_frac": 0.15},
        "features":  {"log_gabor": {"num_scales": NUM_SCALES}},
    }
    matcher = IrisMatcher(cfg)

    code_a = np.random.randint(0, 2, (H, TOTAL_BITS), dtype=np.uint8)
    mask_a = np.ones((H, TOTAL_BITS), dtype=np.uint8)

    # Build code_b as a shifted + noisy version of code_a
    TRUE_SHIFT = 4
    bits_per_scale = NORM_WIDTH * 2
    code_b = np.zeros_like(code_a)
    for s in range(NUM_SCALES):
        start = s * bits_per_scale
        end   = start + bits_per_scale
        code_b[:, start:end] = np.roll(code_a[:, start:end], TRUE_SHIFT, axis=1)
    # Add 5% noise
    flip = np.random.rand(H, TOTAL_BITS) < 0.05
    code_b[flip] ^= 1
    mask_b = np.ones_like(mask_a)

    hd, best_shift = matcher.compute_hd(code_a, mask_a, code_b, mask_b)
    print(f"HD (same identity, shift={TRUE_SHIFT}): {hd:.4f}  (should be ~0.05)")
    print(f"Best shift: {best_shift}  (should be {-TRUE_SHIFT})")

    # Random codes should give HD ≈ 0.5
    code_c = np.random.randint(0, 2, (H, TOTAL_BITS), dtype=np.uint8)
    mask_c = np.ones_like(mask_a)
    hd_imp, _ = matcher.compute_hd(code_a, mask_a, code_c, mask_c)
    print(f"HD (different identities):               {hd_imp:.4f}  (should be ~0.50)")
