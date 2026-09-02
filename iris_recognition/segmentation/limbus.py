"""
Limbus (outer iris boundary) detector.

The limbus is the boundary between the iris and the sclera.
It is typically found at 2.5–4× the pupil radius.

Strategy:
  1. Restrict the IDO to lateral sectors (avoiding eyelid/eyelash occlusion
     at top and bottom of the iris).
  2. Gradient-based Hough-circle fallback when IDO confidence is low.
  3. Coarse-to-fine IDO search.
"""

import cv2
import numpy as np
from iris_recognition.segmentation.integro_differential import IntegroDifferentialOperator


# ---------------------------------------------------------------------------
# Sector-restricted IDO (left + right flanks only)
# ---------------------------------------------------------------------------

class SectorIDO(IntegroDifferentialOperator):
    """IDO that samples only the lateral sectors of the circle.

    This avoids the heavily occluded top (eyelid) and bottom (lower eyelid /
    eyelash shadow) regions, giving a more reliable response for the limbus.
    """

    # Sample 45° each side of horizontal (left and right flanks).
    _SECTOR_ARC = np.pi / 3   # ±60° around the horizontal axis

    def __init__(self, num_points: int = 80):
        self.num_points = num_points
        # Left flank: −60° … +60° (centred on 0)
        left = np.linspace(-self._SECTOR_ARC, self._SECTOR_ARC, num_points // 2)
        # Right flank: 120° … 240°  (centred on π)
        right = np.linspace(np.pi - self._SECTOR_ARC, np.pi + self._SECTOR_ARC, num_points // 2)
        thetas = np.concatenate([left, right])
        self.cos_t = np.cos(thetas).astype(np.float32)
        self.sin_t = np.sin(thetas).astype(np.float32)


# ---------------------------------------------------------------------------
# Limbus Detector
# ---------------------------------------------------------------------------

class LimbusDetector:
    """Detects the outer iris boundary (limbus) using:
      1. Sector-restricted Integro-Differential Operator (primary)
      2. Gradient-based Hough circle transform (fallback)
    """

    def __init__(self, config=None):
        cfg = (config or {}).get("segmentation", {}).get("limbus", {})
        self.min_radius = cfg.get("min_radius", 70)
        self.max_radius = cfg.get("max_radius", 175)
        self._ido = SectorIDO(num_points=80)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, img_gray: np.ndarray, pupil_x: float, pupil_y: float, pupil_r: float):
        """Detect the limbus.

        Parameters
        ----------
        img_gray          : (H, W) uint8 preprocessed (denoised) grayscale image.
        pupil_x/y/r       : Detected pupil parameters used as initialisation prior.

        Returns
        -------
        (iris_x, iris_y, iris_r, score)  or  None on failure.
        """
        h, w = img_gray.shape

        # Estimate expected iris radius range from the pupil radius.
        # Typical pupil-to-iris ratio is 0.25–0.50.
        r_lo = max(self.min_radius, int(pupil_r * 1.8))
        r_hi = min(self.max_radius, int(pupil_r * 4.5))
        if r_lo >= r_hi:
            r_lo = self.min_radius
            r_hi = self.max_radius

        r_mid  = (r_lo + r_hi) // 2
        r_half = (r_hi - r_lo) // 2 + 5   # half-range for IDO

        # ------ Stage 1: Coarse-to-fine IDO -----
        ido_x, ido_y, ido_r, ido_score = self._ido.search_circle(
            img_gray,
            init_x=int(pupil_x), init_y=int(pupil_y), init_r=r_mid,
            search_range_xy=8, search_range_r=r_half,
            step_xy=3,          step_r=4,
            is_pupil=False,
            fine_step_xy=1,     fine_step_r=1,
            fine_range=5,
        )

        # Validate IDO result
        ido_ok = self._validate(ido_x, ido_y, ido_r, pupil_x, pupil_y, pupil_r, h, w)

        if ido_ok:
            return int(ido_x), int(ido_y), int(ido_r), float(ido_score)

        # ------ Stage 2: Gradient Hough fallback -----
        hough_result = self._hough_fallback(img_gray, pupil_x, pupil_y, pupil_r, r_lo, r_hi)
        if hough_result is not None:
            hx, hy, hr = hough_result
            if self._validate(hx, hy, hr, pupil_x, pupil_y, pupil_r, h, w):
                return int(hx), int(hy), int(hr), 0.5   # fallback gets a neutral score

        # ------ Stage 3: Last resort — return IDO result even if suspect -----
        if ido_score > 0:
            return int(ido_x), int(ido_y), int(ido_r), float(ido_score)

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(ix, iy, ir, px, py, pr, h, w) -> bool:
        """Returns True if the iris circle is geometrically plausible."""
        # Radius must be in a sensible range relative to the image
        if not (30 <= ir <= min(h, w) // 2):
            return False
        # Iris must be larger than the pupil
        if ir <= pr * 1.3:
            return False
        # Centre must not be wildly off from the pupil centre
        dist = np.sqrt((ix - px) ** 2 + (iy - py) ** 2)
        if dist > 0.3 * ir:
            return False
        # Centre must be reasonably inside the image
        margin = ir * 0.25
        if not (margin <= ix <= w - margin and margin <= iy <= h - margin):
            return False
        return True

    def _hough_fallback(self, img_gray, px, py, pr, r_lo, r_hi):
        """Gradient-based Hough circle transform to find the limbus."""
        # Enhance the iris-sclera gradient
        blurred = cv2.GaussianBlur(img_gray, (5, 5), 1.5)
        edges   = cv2.Canny(blurred, 20, 60)

        # Restrict to an annular search region around the pupil
        search_mask = np.zeros_like(edges)
        cv2.circle(search_mask, (int(px), int(py)), int(r_hi + 10), 255, -1)
        cv2.circle(search_mask, (int(px), int(py)), int(r_lo - 10), 0,   -1)
        edges = cv2.bitwise_and(edges, search_mask)

        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(pr * 2),
            param1=50,
            param2=15,
            minRadius=int(r_lo),
            maxRadius=int(r_hi),
        )

        if circles is None:
            return None

        circles = np.round(circles[0]).astype(int)
        # Pick the circle whose centre is closest to the pupil centre
        dists = [np.sqrt((c[0] - px) ** 2 + (c[1] - py) ** 2) for c in circles]
        best  = circles[int(np.argmin(dists))]
        return int(best[0]), int(best[1]), int(best[2])


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic eye: sclera(180) → iris(100) → pupil(20)
    test_img = np.ones((480, 640), dtype=np.uint8) * 180
    cv2.circle(test_img, (320, 240), 110, 100, -1)
    cv2.circle(test_img, (320, 240), 40,  20,  -1)
    test_img = cv2.GaussianBlur(test_img, (7, 7), 2)

    det = LimbusDetector()
    res = det.detect(test_img, pupil_x=320.0, pupil_y=240.0, pupil_r=40.0)
    if res:
        print(f"Limbus: x={res[0]}  y={res[1]}  r={res[2]}  score={res[3]:.4f}"
              f"  (expected ~320,240,110)")
    else:
        print("Limbus detection FAILED on synthetic image")
