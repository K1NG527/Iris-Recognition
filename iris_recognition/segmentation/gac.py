"""
Geodesic Active Contours (GAC) refinement module.

Used as a post-processing step to refine circular boundary estimates
produced by the Integro-Differential Operator and thresholding.

The GAC evolves an initial contour toward strong image gradients,
snapping it accurately to the pupil/limbus boundary.

Fixed: skimage >=0.19 uses 'num_iter' not 'iterations'.
"""

import numpy as np
import cv2

try:
    from skimage.segmentation import morphological_geodesic_active_contour as _skimage_gac
    from skimage.segmentation import inverse_gaussian_gradient
    import inspect
    _gac_sig = inspect.signature(_skimage_gac)
    # Detect whether the installed version uses 'num_iter' or 'iterations'
    _GAC_PARAM = "num_iter" if "num_iter" in _gac_sig.parameters else "iterations"
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    _GAC_PARAM = "num_iter"


def _call_gac(gimage, n_iter, init_level_set, smoothing, balloon, threshold):
    """Wraps skimage GAC to handle the num_iter / iterations parameter rename."""
    kwargs = {
        _GAC_PARAM: n_iter,
        "init_level_set": init_level_set,
        "smoothing": smoothing,
        "balloon": balloon,
        "threshold": threshold,
    }
    return _skimage_gac(gimage, **kwargs)


class GACRefiner:
    """Refines a circular boundary estimate using Geodesic Active Contours.

    Pipeline:
        initial circle estimate
            ↓
        gradient image (inverse Gaussian gradient)
            ↓
        initial binary level-set (filled circle)
            ↓
        GAC evolution
            ↓
        fit enclosing circle to evolved mask
            ↓
        refined (cx, cy, radius)
    """

    def __init__(self, config=None):
        self.config = config if config else {}
        gac_cfg = self.config.get("segmentation", {}).get("gac", {})
        self.enabled    = gac_cfg.get("enabled", True)
        self.iterations = gac_cfg.get("iterations", 150)
        self.smoothing  = gac_cfg.get("smoothing", 2)
        self.balloon    = gac_cfg.get("balloon", 0.5)
        self.alpha      = gac_cfg.get("alpha", 100.0)  # inv-Gaussian alpha (edge sensitivity)
        self.sigma      = gac_cfg.get("sigma", 2.0)    # inv-Gaussian sigma (Gaussian pre-blur)
        self.threshold  = gac_cfg.get("threshold", 0.69)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refine_circle(self, img_gray, cx, cy, r, is_pupil=True):
        """Refines a circular boundary using GAC.

        Parameters
        ----------
        img_gray : ndarray (H, W) uint8
            Preprocessed (denoised) grayscale image.
        cx, cy, r : float
            Initial circle centre and radius from IDO / threshold stage.
        is_pupil : bool
            True  → the initial circle is the pupil boundary (dark interior).
            False → the initial circle is the limbus boundary (iris interior).

        Returns
        -------
        (refined_cx, refined_cy, refined_r, evolved_mask)
            refined_cx/cy/r  – refined circle parameters (float).
            evolved_mask     – uint8 binary mask of the evolved region (0 / 255).

        Performance note:
            GAC is run at half-resolution to reduce cost (≈4× faster).
            Results are scaled back to original resolution.
        """
        h, w = img_gray.shape

        # Fallback mask used on any failure
        fallback_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(fallback_mask, (int(round(cx)), int(round(cy))), max(1, int(round(r))), 255, -1)

        if not SKIMAGE_AVAILABLE or not self.enabled:
            return cx, cy, r, fallback_mask

        try:
            # ── Downsample to half resolution for speed ───────────────
            scale = 0.5
            sh, sw = int(h * scale), int(w * scale)
            small  = cv2.resize(img_gray, (sw, sh), interpolation=cv2.INTER_AREA)
            scx, scy, sr = cx * scale, cy * scale, r * scale

            scx_r, scy_r, sr_r, small_mask = self._run_gac(
                small, scx, scy, sr, is_pupil, sh, sw,
                _fallback=None,   # allow internal fallback
            )

            # ── Scale results back to original resolution ──────────────
            out_cx = scx_r / scale
            out_cy = scy_r / scale
            out_r  = sr_r  / scale

            # Build full-resolution mask from the fitted circle
            out_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(out_mask, (int(round(out_cx)), int(round(out_cy))),
                       max(1, int(round(out_r))), 255, -1)

            return float(out_cx), float(out_cy), float(out_r), out_mask

        except Exception:
            return cx, cy, r, fallback_mask

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_gac(self, img_gray, cx, cy, r, is_pupil, h, w, fallback_mask=None, _fallback=None):
        """Core GAC logic (called inside a try/except from refine_circle)."""

        # 1. Build edge-stopping function.
        #    For pupil: strong response at the dark/bright transition.
        #    For limbus: strong response at the iris/sclera transition.
        #    We work on a lightly blurred image to suppress noise.
        blurred = cv2.GaussianBlur(img_gray, (5, 5), self.sigma)
        gimage = inverse_gaussian_gradient(
            blurred.astype(np.float64),
            alpha=self.alpha,
            sigma=self.sigma,
        )

        # 2. Initial level-set: filled circle at the initial estimate.
        #    Use int32 as required by skimage's GAC.
        init_ls = np.zeros((h, w), dtype=np.int32)
        r_int = max(1, int(round(r)))
        cx_int, cy_int = int(round(cx)), int(round(cy))
        cv2.circle(init_ls, (cx_int, cy_int), r_int, 1, -1)

        # 3. Choose balloon force.
        #    Pupil: allow mild expansion (balloon > 0) so the contour reaches
        #           the dark→bright edge from inside.
        #    Limbus: shrink slightly (balloon < 0) to pull from outside
        #            toward the iris→sclera edge.
        if is_pupil:
            balloon = abs(self.balloon)   # expand
        else:
            balloon = -abs(self.balloon)  # contract

        # 4. Evolve.
        evolved = _call_gac(
            gimage,
            n_iter=self.iterations,
            init_level_set=init_ls,
            smoothing=self.smoothing,
            balloon=balloon,
            threshold=self.threshold,
        )

        # 5. Convert boolean level-set → uint8 mask.
        evolved_mask = (evolved > 0).astype(np.uint8) * 255

        # 6. Fit enclosing circle to the largest contour of the evolved mask.
        contours, _ = cv2.findContours(
            evolved_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            if fallback_mask is not None:
                return cx, cy, r, fallback_mask
            # Build a simple circle fallback
            fb = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(fb, (int(round(cx)), int(round(cy))), max(1, int(round(r))), 255, -1)
            return cx, cy, r, fb

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < np.pi * 4:   # degenerate contour
            if fallback_mask is not None:
                return cx, cy, r, fallback_mask
            fb = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(fb, (int(round(cx)), int(round(cy))), max(1, int(round(r))), 255, -1)
            return cx, cy, r, fb

        # Prefer minimum enclosing circle — more stable than bounding-rect for iris
        (fit_x, fit_y), fit_r = cv2.minEnclosingCircle(largest)

        # Sanity: don't accept a wildly different radius (>50% change)
        if not (0.5 * r <= fit_r <= 1.5 * r):
            if fallback_mask is not None:
                return cx, cy, r, evolved_mask
            return cx, cy, r, evolved_mask

        return float(fit_x), float(fit_y), float(fit_r), evolved_mask


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"skimage available : {SKIMAGE_AVAILABLE}")
    print(f"GAC iter param    : '{_GAC_PARAM}'")

    # Synthetic eye image
    test_img = np.ones((480, 640), dtype=np.uint8) * 180
    cv2.circle(test_img, (320, 240), 110, 100, -1)   # iris
    cv2.circle(test_img, (320, 240), 40,  20,  -1)   # pupil
    test_img = cv2.GaussianBlur(test_img, (5, 5), 0)

    refiner = GACRefiner()

    cx_p, cy_p, r_p, mask_p = refiner.refine_circle(test_img, 318, 242, 38, is_pupil=True)
    print(f"Pupil  GAC: cx={cx_p:.1f}  cy={cy_p:.1f}  r={r_p:.1f}  (expected ~40)")

    cx_i, cy_i, r_i, mask_i = refiner.refine_circle(test_img, 320, 240, 105, is_pupil=False)
    print(f"Limbus GAC: cx={cx_i:.1f}  cy={cy_i:.1f}  r={r_i:.1f}  (expected ~110)")
