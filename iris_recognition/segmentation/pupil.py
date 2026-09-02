"""
Pupil detection module.

Implements a multi-stage pupil detector:

  Stage 1 — Threshold + Morphology:
      Detects dark, circular regions using adaptive thresholding,
      morphological clean-up, and contour analysis.

  Stage 2 — Hough Circle Transform (fallback):
      When Stage 1 fails (very dark images, unusual lighting),
      a Hough circle transform is used as a robust fallback.

Both stages rank candidates by a composite score:
  - circularity     (4πA / P²)
  - aspect ratio    (fitted ellipse)
  - distance from centre (iris images usually centred on the pupil)
  - mean intensity  (pupil must be dark)
"""

import cv2
import numpy as np


class PupilDetector:
    """Multi-stage pupil detector with Hough circle fallback."""

    def __init__(self, config=None):
        cfg = (config or {}).get("segmentation", {}).get("pupil", {})
        self.min_radius    = cfg.get("min_radius", 15)
        self.max_radius    = cfg.get("max_radius", 90)
        self.threshold_val = cfg.get("threshold_val", 55)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, img_gray: np.ndarray):
        """Detect the pupil in a grayscale image.

        Parameters
        ----------
        img_gray : (H, W) uint8 grayscale (CLAHE-enhanced recommended).

        Returns
        -------
        (cx, cy, radius, score)  or  None if no pupil found.
        """
        # Try primary threshold-based detector
        result = self._detect_threshold(img_gray)
        if result is not None:
            return result

        # Fallback: Hough circle transform
        return self._detect_hough(img_gray)

    # ------------------------------------------------------------------
    # Stage 1: Threshold + morphology + contour
    # ------------------------------------------------------------------

    def _detect_threshold(self, img_gray: np.ndarray):
        """Threshold-based pupil candidate extraction."""
        h, w = img_gray.shape

        # Try multiple threshold values to be robust to varying illumination
        thresholds = [self.threshold_val, self.threshold_val - 15, self.threshold_val + 15]
        all_candidates = []

        for tval in thresholds:
            tval = max(10, min(100, tval))
            _, thresh = cv2.threshold(img_gray, tval, 255, cv2.THRESH_BINARY_INV)

            # Morphological clean-up
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k)

            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_candidates.extend(self._score_contours(contours, img_gray, h, w))

        if not all_candidates:
            return None

        # Sort by descending score, take the best
        all_candidates.sort(key=lambda c: c["score"], reverse=True)
        best = all_candidates[0]

        return (best["cx"], best["cy"], best["radius"], best["score"])

    def _score_contours(self, contours, img_gray, h, w):
        """Filter and score contour candidates as pupil regions."""
        min_area = np.pi * self.min_radius ** 2
        max_area = np.pi * self.max_radius ** 2
        img_cx, img_cy = w / 2.0, h / 2.0
        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area <= area <= max_area):
                continue

            perim = cv2.arcLength(cnt, True)
            if perim < 1:
                continue

            circularity = (4.0 * np.pi * area) / (perim ** 2)
            if circularity < 0.50:   # allow slightly non-circular (eyelashes)
                continue

            # Fit ellipse or bounding circle
            if len(cnt) >= 5:
                (ex, ey), (d1, d2), _ = cv2.fitEllipse(cnt)
                dmax = max(d1, d2)
                dmin = min(d1, d2)
                aspect = dmin / dmax if dmax > 0 else 0
                est_r  = (d1 + d2) / 4.0
            else:
                (ex, ey), est_r = cv2.minEnclosingCircle(cnt)
                aspect = 1.0

            if aspect < 0.65:
                continue

            # Position sanity: not at the extreme edge
            margin = self.min_radius
            if not (margin <= ex <= w - margin and margin <= ey <= h - margin):
                continue

            # Intensity: pupil must be dark
            mask = np.zeros_like(img_gray)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_int = cv2.mean(img_gray, mask=mask)[0]
            if mean_int > 100:
                continue

            # Distance from image centre (normalised)
            dist = np.sqrt((ex - img_cx) ** 2 + (ey - img_cy) ** 2)
            norm_dist = dist / (max(w, h) / 2.0)

            # Composite score
            score = (
                0.35 * circularity
                + 0.25 * aspect
                + 0.20 * max(0.0, 1.0 - norm_dist)
                + 0.20 * (1.0 - mean_int / 255.0)
            )

            candidates.append({
                "cx": ex, "cy": ey, "radius": est_r,
                "score": score, "mean_int": mean_int,
            })

        return candidates

    # ------------------------------------------------------------------
    # Stage 2: Hough circle fallback
    # ------------------------------------------------------------------

    def _detect_hough(self, img_gray: np.ndarray):
        """Hough-transform pupil detector for difficult images."""
        h, w = img_gray.shape

        # Emphasise dark pupil region before edge detection
        # Use a strong blur to suppress iris texture
        blurred = cv2.GaussianBlur(img_gray, (9, 9), 2)

        # Dark-region mask to focus Hough on the pupil area
        _, dark_mask = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY_INV)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, k)

        search_img = cv2.bitwise_and(blurred, dark_mask)

        circles = cv2.HoughCircles(
            search_img,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(self.min_radius * 2),
            param1=40,
            param2=12,
            minRadius=int(self.min_radius),
            maxRadius=int(self.max_radius),
        )

        if circles is None:
            return None

        circles = np.round(circles[0]).astype(float)
        img_cx, img_cy = w / 2.0, h / 2.0

        best = None
        best_score = -1.0

        for cx, cy, r in circles:
            # Score by proximity to image centre + darkness inside circle
            dist = np.sqrt((cx - img_cx) ** 2 + (cy - img_cy) ** 2)
            norm_dist = dist / (max(w, h) / 2.0)

            # Sample intensity inside the circle
            tmp_mask = np.zeros_like(img_gray)
            cv2.circle(tmp_mask, (int(cx), int(cy)), int(r), 255, -1)
            mean_int = cv2.mean(img_gray, mask=tmp_mask)[0]

            if mean_int > 100:
                continue

            score = 0.5 * max(0.0, 1.0 - norm_dist) + 0.5 * (1.0 - mean_int / 255.0)
            if score > best_score:
                best_score = score
                best = (cx, cy, r, score)

        return best


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic eye image
    test_img = np.ones((480, 640), dtype=np.uint8) * 180
    cv2.circle(test_img, (320, 240), 110, 100, -1)
    cv2.circle(test_img, (320, 240), 40,  20,  -1)

    det = PupilDetector()
    res = det.detect(test_img)
    if res:
        print(f"Pupil: cx={res[0]:.1f}  cy={res[1]:.1f}  r={res[2]:.1f}"
              f"  score={res[3]:.3f}  (expected ~320,240,40)")
    else:
        print("Pupil detection FAILED on synthetic image")

    # Test Hough fallback on dark image
    dark_img = np.ones((480, 640), dtype=np.uint8) * 20
    cv2.circle(dark_img, (310, 235), 35, 5, -1)
    res2 = det._detect_hough(dark_img)
    print(f"Hough fallback: {res2}")
