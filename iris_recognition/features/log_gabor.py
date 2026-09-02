"""
Log-Gabor Feature Extractor for Iris Recognition.

Implements multi-scale, phase-based iris coding based on the method described by
Daugman (2004) and the Log-Gabor implementation by Masek (2003).

For each scale s and each row r of the normalized iris strip:
  1. Apply 1-D Log-Gabor bandpass filter in the frequency domain.
  2. Extract the real and imaginary parts of the complex filter response.
  3. Quantize each part to a single bit (sign: ≥0 → 1, <0 → 0).

The final binary iris code is laid out as:
    [scale_0_real | scale_0_imag | scale_1_real | scale_1_imag | ...]

Each block has exactly `width` columns (one bit per pixel).
Total code shape: (height, width * 2 * num_scales)

This layout is consistent with the multi-scale shift used in IrisMatcher.
"""

import numpy as np


class LogGaborExtractor:
    """Multi-scale Log-Gabor phase quantization for binary iris code extraction."""

    def __init__(self, config=None):
        cfg = (config or {}).get("features", {}).get("log_gabor", {})
        self.min_wavelength = cfg.get("min_wavelength", 18)   # pixels, smallest scale
        self.num_scales     = cfg.get("num_scales", 4)        # number of filter scales
        self.mult           = cfg.get("mult", 1.8)            # inter-scale multiplier
        self.sigma_on_f     = cfg.get("sigma_on_f", 0.5)      # bandwidth parameter

    # ------------------------------------------------------------------
    # Filter construction
    # ------------------------------------------------------------------

    def _log_gabor_filter(self, n_cols: int, wavelength: float) -> np.ndarray:
        """1-D Log-Gabor filter in the frequency domain.

        Parameters
        ----------
        n_cols     : length of the signal (number of columns in normalized iris).
        wavelength : centre wavelength in pixels.

        Returns
        -------
        filter_fft : complex-valued array of shape (n_cols,).
                     Multiply with the FFT of a row then IFFT to filter.
        """
        # Normalised frequency axis (0 … 0.5, DC at index 0)
        freq = np.fft.fftfreq(n_cols).astype(np.float64)
        freq[0] = 1.0   # avoid log(0); DC will be zeroed out below

        fo = 1.0 / wavelength  # centre frequency

        # Log-Gabor envelope in frequency domain
        log_ratio = np.log(np.abs(freq) / fo)
        sigma_sq  = 2.0 * (np.log(self.sigma_on_f)) ** 2
        envelope  = np.exp(-(log_ratio ** 2) / sigma_sq)

        envelope[0] = 0.0   # remove DC component

        return envelope   # real-valued (symmetric filter → real output from IFFT)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract(
        self,
        normalized_iris: np.ndarray,
        normalized_mask: np.ndarray,
    ):
        """Extract multi-scale binary iris code from a normalized iris strip.

        Parameters
        ----------
        normalized_iris : (H, W) uint8   — rubber-sheet normalized iris texture.
        normalized_mask : (H, W) uint8   — validity mask (255 = valid, 0 = invalid).

        Returns
        -------
        iris_code  : (H, W * 2 * num_scales) uint8   — binary iris code (0 or 1).
        noise_mask : (H, W * 2 * num_scales) uint8   — validity bits (0 or 1).
        """
        h, w = normalized_iris.shape
        # Each scale contributes 2 * w bits (real + imaginary), one bit per pixel per part.
        bits_per_scale = w * 2
        total_cols = bits_per_scale * self.num_scales

        iris_code  = np.zeros((h, total_cols), dtype=np.uint8)
        noise_mask = np.zeros((h, total_cols), dtype=np.uint8)

        # Float image in [0, 1] for numerically stable filtering
        img_f = normalized_iris.astype(np.float64) / 255.0

        # Pre-compute the valid-bit row masks (one per row, same across scales)
        # validity mask: 255 valid → 1 in noise_mask
        row_valid = (normalized_mask > 0).astype(np.uint8)   # (H, W)

        for s in range(self.num_scales):
            wavelength = self.min_wavelength * (self.mult ** s)
            filt = self._log_gabor_filter(w, wavelength)    # (W,) real envelope

            col_real_start = s * bits_per_scale             # index of real bits block
            col_imag_start = col_real_start + w             # index of imag bits block

            for r in range(h):
                row_fft    = np.fft.fft(img_f[r])
                # Complex analytic signal: keep positive freqs, zero negative
                # This gives a proper analytic (complex) signal from the real filter
                analytic_fft = np.zeros(w, dtype=np.complex128)
                analytic_fft[0]     = row_fft[0] * filt[0]   # DC
                analytic_fft[1:w//2] = 2.0 * row_fft[1:w//2] * filt[1:w//2]
                if w % 2 == 0:
                    analytic_fft[w//2] = row_fft[w//2] * filt[w//2]   # Nyquist
                # Negative freqs set to zero → analytic signal
                filtered = np.fft.ifft(analytic_fft)

                real_bits = (np.real(filtered) >= 0).astype(np.uint8)
                imag_bits = (np.imag(filtered) >= 0).astype(np.uint8)

                # Store: real block then imag block for this scale
                iris_code[r, col_real_start : col_real_start + w] = real_bits
                iris_code[r, col_imag_start : col_imag_start + w] = imag_bits

                # Validity mask — same for both real and imag of this row
                iris_code_mask_row = row_valid[r]  # (W,)
                noise_mask[r, col_real_start : col_real_start + w] = iris_code_mask_row
                noise_mask[r, col_imag_start : col_imag_start + w] = iris_code_mask_row

        return iris_code, noise_mask


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    H, W = 64, 512
    dummy_iris = np.random.randint(0, 256, (H, W), dtype=np.uint8)
    dummy_mask = np.ones((H, W), dtype=np.uint8) * 255
    dummy_mask[:10, :100] = 0   # some invalid region

    extractor = LogGaborExtractor()
    code, mask = extractor.extract(dummy_iris, dummy_mask)

    n_scales    = extractor.num_scales
    bits_per_s  = W * 2
    total_bits  = bits_per_s * n_scales

    print("Log-Gabor Multi-Scale Extractor test")
    print(f"  num_scales      : {n_scales}")
    print(f"  bits_per_scale  : {bits_per_s}  (real={W} + imag={W})")
    print(f"  Iris code shape : {code.shape}   expected ({H}, {total_bits})")
    print(f"  Noise mask shape: {mask.shape}")
    print(f"  Valid bit ratio : {mask.sum() / mask.size:.4f}")

    # Sanity: code must be binary
    assert code.max() <= 1 and code.min() >= 0, "Code values out of [0,1]"
    print("  Binary check    : PASSED")
