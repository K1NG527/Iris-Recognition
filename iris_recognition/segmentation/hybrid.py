"""
Hybrid Iris Segmentation Pipeline.

Orchestrates the full segmentation chain:

    RAW IMAGE
        ↓
    Preprocessing (denoise, CLAHE, reflection suppression)
        ↓
    Pupil Detection  (threshold + Hough fallback)
        ↓
    IDO Pupil Refinement  (coarse-to-fine)
        ↓
    Limbus Detection  (sector-IDO + Hough fallback)
        ↓
    GAC Refinement  (contour evolution)
        ↓
    Occlusion / Eyelid Masking
        ↓
    Final Iris Mask

Every image receives a segmentation confidence score and a status:
    GOOD    — all checks passed
    WARNING — minor issues but usable
    FAILED  — segmentation not reliable
"""

import cv2
import numpy as np

from iris_recognition.preprocessing.preprocessing import Preprocessor
from iris_recognition.segmentation.pupil import PupilDetector
from iris_recognition.segmentation.limbus import LimbusDetector
from iris_recognition.segmentation.integro_differential import IntegroDifferentialOperator
from iris_recognition.segmentation.gac import GACRefiner


class HybridSegmenter:
    """Complete iris segmentation pipeline."""

    def __init__(self, config=None):
        self.config       = config if config else {}
        self.preprocessor = Preprocessor(self.config)
        self.pupil_det    = PupilDetector(self.config)
        self.limbus_det   = LimbusDetector(self.config)
        self.ido          = IntegroDifferentialOperator(num_points=120)
        self.gac          = GACRefiner(self.config)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def segment(self, img_bgr: np.ndarray) -> dict:
        """Segment an eye image and return all relevant artefacts.

        Parameters
        ----------
        img_bgr : (H, W, 3) uint8 BGR image.

        Returns
        -------
        dict with keys:
            status, confidence, reason (on failure),
            pupil (x,y,r), iris (x,y,r),
            mask (H×W uint8), grayscale (H×W uint8), enhanced (H×W uint8).
        """
        if img_bgr is None or img_bgr.ndim != 3:
            return self._fail("INVALID_IMAGE", np.zeros((1, 1), dtype=np.uint8))

        h, w = img_bgr.shape[:2]
        fail_mask = np.zeros((h, w), dtype=np.uint8)

        # ── 1. Preprocess ─────────────────────────────────────────────
        prep = self.preprocessor.process(img_bgr)
        gray      = prep["grayscale"]
        denoised  = prep["denoised"]
        enhanced  = prep["enhanced"]
        refl_mask = prep["refl_mask"]

        # ── 2. Pupil detection ────────────────────────────────────────
        pupil_res = self.pupil_det.detect(enhanced)
        if pupil_res is None:
            pupil_res = self.pupil_det.detect(denoised)
        if pupil_res is None:
            return self._fail("NO_PUPIL_DETECTED", fail_mask, gray=gray, enhanced=enhanced)

        px0, py0, pr0, p_score0 = pupil_res

        # ── 3. IDO pupil refinement ───────────────────────────────────
        smoothed = cv2.GaussianBlur(denoised, (7, 7), 2)
        px, py, pr, p_ido = self.ido.search_circle(
            smoothed,
            init_x=int(px0), init_y=int(py0), init_r=int(pr0),
            search_range_xy=8, search_range_r=8,
            step_xy=2, step_r=2, is_pupil=True,
            fine_step_xy=1, fine_step_r=1, fine_range=3,
        )

        # If IDO shifted the centre implausibly far, revert
        if np.sqrt((px - px0) ** 2 + (py - py0) ** 2) > 20:
            px, py, pr = px0, py0, int(pr0)

        # ── 4. Limbus detection ───────────────────────────────────────
        limbus_res = self.limbus_det.detect(denoised, px, py, pr)
        if limbus_res is None:
            return self._fail("NO_LIMBUS_DETECTED", fail_mask, gray=gray, enhanced=enhanced)

        ix0, iy0, ir0, l_score = limbus_res

        # ── 5. GAC refinement ─────────────────────────────────────────
        px_g, py_g, pr_g, pupil_gmask = self.gac.refine_circle(
            denoised, px, py, pr, is_pupil=True
        )
        ix_g, iy_g, ir_g, iris_gmask  = self.gac.refine_circle(
            denoised, ix0, iy0, ir0, is_pupil=False
        )

        # Sanity-check GAC outputs; revert to pre-GAC if bogus
        if not self._gac_plausible(px_g, py_g, pr_g, px, py, pr):
            px_g, py_g, pr_g = px, py, pr
        if not self._gac_plausible(ix_g, iy_g, ir_g, ix0, iy0, ir0):
            ix_g, iy_g, ir_g = ix0, iy0, ir0

        # ── 6. Confidence & status ────────────────────────────────────
        confidence, status = self._quality(
            px_g, py_g, pr_g,
            ix_g, iy_g, ir_g,
            p_score0, l_score, h, w,
        )

        # ── 7. Iris mask ──────────────────────────────────────────────
        iris_mask = self._build_mask(
            gray, px_g, py_g, pr_g,
            ix_g, iy_g, ir_g,
            refl_mask, h, w,
        )

        return {
            "status":     status,
            "confidence": confidence,
            "pupil":      (float(px_g), float(py_g), float(pr_g)),
            "iris":       (float(ix_g), float(iy_g), float(ir_g)),
            "mask":       iris_mask,
            "grayscale":  gray,
            "enhanced":   enhanced,
        }

    # ------------------------------------------------------------------
    # Iris mask construction
    # ------------------------------------------------------------------

    def _build_mask(self, gray, px, py, pr, ix, iy, ir, refl_mask, h, w):
        """Generate the binary iris mask (valid iris texture region)."""
        # Outer iris ring
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (int(ix), int(iy)), max(1, int(ir)), 255, -1)

        # Subtract pupil
        pupil_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(pupil_mask, (int(px), int(py)), max(1, int(pr)), 255, -1)
        mask = cv2.subtract(mask, pupil_mask)

        # Subtract specular reflections
        if refl_mask is not None and cv2.countNonZero(refl_mask) > 0:
            refl_dilated = cv2.dilate(
                refl_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )
            mask = cv2.subtract(mask, refl_dilated)

        # Subtract very dark pixels inside the iris ring (eyelashes, shadows)
        # Use a dynamic threshold: below 40% of the expected iris mean intensity
        iris_region   = cv2.bitwise_and(gray, mask)
        iris_mean     = cv2.mean(iris_region, mask=mask)[0]
        dark_thresh   = max(30, int(iris_mean * 0.45))
        _, dark_noise = cv2.threshold(gray, dark_thresh, 255, cv2.THRESH_BINARY_INV)
        dark_noise    = cv2.bitwise_and(dark_noise, mask)
        # Dilate noise mask slightly to cover transition pixels
        dark_noise = cv2.dilate(
            dark_noise,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        mask = cv2.subtract(mask, dark_noise)

        # Eyelid occlusion masks
        #   Upper lid: cut the top ~25% of the iris ring radius
        #   Lower lid: cut the bottom ~15% of the iris ring radius
        eyelid_mask = np.zeros((h, w), dtype=np.uint8)
        top_cut    = int(iy - ir * 0.75)   # 75% from centre upward → cut top 25%
        bottom_cut = int(iy + ir * 0.88)   # 88% from centre downward → cut bottom 12%
        if top_cut > 0:
            cv2.rectangle(eyelid_mask, (0, 0), (w, top_cut), 255, -1)
        if bottom_cut < h:
            cv2.rectangle(eyelid_mask, (0, bottom_cut), (w, h), 255, -1)
        mask = cv2.subtract(mask, eyelid_mask)

        # Small morphological clean-up of residual noise patches
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    # ------------------------------------------------------------------
    # Quality control
    # ------------------------------------------------------------------

    def _quality(self, px, py, pr, ix, iy, ir, p_score, l_score, h, w):
        """Compute a confidence score [0,1] and status label."""
        score = 0.0

        # 1. Pupil-to-iris radius ratio (expected 0.20–0.55)
        ratio = pr / ir if ir > 0 else 0
        ratio_ok = 0.20 <= ratio <= 0.55
        if ratio_ok:
            score += 0.30
        else:
            score += max(0.0, 0.30 - abs(ratio - 0.35) * 0.6)

        # 2. Pupil–iris centre displacement (expect < 15% of iris radius)
        disp = np.sqrt((px - ix) ** 2 + (py - iy) ** 2)
        disp_ok = disp <= 0.20 * ir
        if disp_ok:
            score += 0.25
        else:
            score += max(0.0, 0.25 - (disp / max(ir, 1)) * 0.4)

        # 3. Radius bounds
        radii_ok = (self.pupil_det.min_radius <= pr <= self.pupil_det.max_radius) and \
                   (self.limbus_det.min_radius <= ir <= self.limbus_det.max_radius)
        if radii_ok:
            score += 0.20
        # else: 0

        # 4. Boundary response strengths (normalised)
        score += min(0.15, p_score / 30.0 * 0.15)
        score += min(0.10, l_score / 10.0 * 0.10)

        # Determine status
        if score >= 0.72 and ratio_ok and disp_ok and radii_ok:
            status = "GOOD"
        elif score >= 0.50 and ratio_ok:
            status = "WARNING"
        else:
            status = "FAILED"

        return float(score), status

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fail(reason, mask, gray=None, enhanced=None):
        return {
            "status":     "FAILED",
            "confidence": 0.0,
            "reason":     reason,
            "mask":       mask,
            "grayscale":  gray,
            "enhanced":   enhanced,
        }

    @staticmethod
    def _gac_plausible(cx_new, cy_new, r_new, cx_old, cy_old, r_old):
        """Accept the GAC result only if it hasn't drifted unreasonably."""
        if r_new < 5:
            return False
        centre_drift = np.sqrt((cx_new - cx_old) ** 2 + (cy_new - cy_old) ** 2)
        radius_ratio = r_new / max(r_old, 1)
        return centre_drift <= 20 and 0.65 <= radius_ratio <= 1.35


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic eye: sclera(180) → iris(100) → pupil(20)
    test = np.ones((480, 640, 3), dtype=np.uint8) * 180
    cv2.circle(test, (320, 240), 110, (100, 100, 100), -1)
    cv2.circle(test, (320, 240), 40,  (20,  20,  20),  -1)
    # Slight reflection
    test[200:206, 315:321] = 255
    test = cv2.GaussianBlur(test, (5, 5), 0)

    seg = HybridSegmenter()
    res = seg.segment(test)
    print("Hybrid Segmenter self-test")
    print(f"  Status    : {res['status']}")
    print(f"  Confidence: {res['confidence']:.4f}")
    if res["status"] != "FAILED":
        print(f"  Pupil     : {res['pupil']}  (expected ~320,240,40)")
        print(f"  Iris      : {res['iris']}   (expected ~320,240,110)")
        print(f"  Mask px   : {cv2.countNonZero(res['mask'])}")
