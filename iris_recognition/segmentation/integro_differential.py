"""
Integro-Differential Operator (IDO) — Daugman-style circular boundary detector.

Implements the operator:
    max_{r,x0,y0} | G_sigma * d/dr [ 1/(2πr) ∮ I(x0+r·cosθ, y0+r·sinθ) dθ ] |

where G_sigma is a Gaussian smoothing operator applied along the radial direction.

Search strategy: coarse-to-fine
  Stage 1 — coarse grid: large step_xy and step_r
  Stage 2 — fine grid  : 1-pixel steps around the coarse best candidate

This avoids the O(Nx * Ny * Nr) brute-force cost while preserving accuracy.
"""

import cv2
import numpy as np


class IntegroDifferentialOperator:
    """Circular boundary detector based on the Integro-Differential Operator."""

    def __init__(self, num_points: int = 120):
        """
        Parameters
        ----------
        num_points : int
            Number of sample points on each circular contour.
            More points → smoother integral but slower.  120 is a good trade-off.
        """
        self.num_points = num_points
        thetas = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        self.cos_t = np.cos(thetas).astype(np.float32)
        self.sin_t = np.sin(thetas).astype(np.float32)

    # ------------------------------------------------------------------
    # Low-level: single contour integral
    # ------------------------------------------------------------------

    def _contour_integral(self, img: np.ndarray, cx: float, cy: float, r: float) -> float:
        """Average intensity along the circle of radius r centred at (cx, cy)."""
        h, w = img.shape
        xs = np.clip(np.round(cx + r * self.cos_t), 0, w - 1).astype(np.int32)
        ys = np.clip(np.round(cy + r * self.sin_t), 0, h - 1).astype(np.int32)
        return float(img[ys, xs].mean())

    # ------------------------------------------------------------------
    # Medium-level: radial derivative (IDO response) at a single centre
    # ------------------------------------------------------------------

    def _ido_response(
        self,
        img: np.ndarray,
        cx: float,
        cy: float,
        r_vals: np.ndarray,
        is_pupil: bool,
    ):
        """Compute IDO response for all r values at a fixed centre.

        Returns an array of responses, one per radius value.
        """
        dr = 1  # finite-difference step
        integrals     = np.array([self._contour_integral(img, cx, cy, r) for r in r_vals])
        integrals_fwd = np.array([self._contour_integral(img, cx, cy, r + dr) for r in r_vals])

        diff = integrals_fwd - integrals   # d/dr of the contour integral

        if is_pupil:
            # Pupil boundary: dark interior → bright exterior ⟹ positive derivative
            responses = diff
        else:
            # Limbus boundary: can go either way (iris darker or brighter than sclera)
            responses = np.abs(diff)

        return responses

    # ------------------------------------------------------------------
    # High-level: full coarse-to-fine search
    # ------------------------------------------------------------------

    def search_circle(
        self,
        img: np.ndarray,
        init_x: float,
        init_y: float,
        init_r: float,
        search_range_xy: int = 10,
        search_range_r: int = 10,
        step_xy: int = 2,
        step_r: int = 2,
        is_pupil: bool = True,
        fine_step_xy: int = 1,
        fine_step_r: int = 1,
        fine_range: int = 4,
    ):
        """Coarse-to-fine IDO search.

        Stage 1 — Coarse: search on a grid with (step_xy, step_r) spacing.
        Stage 2 — Fine  : search within ±fine_range pixels / radius of the
                          coarse best candidate with 1-pixel resolution.

        Parameters
        ----------
        img            : (H, W) uint8 grayscale (should be pre-smoothed).
        init_x/y/r     : Initial estimate centre and radius.
        search_range_xy: Half-width of the search box around (init_x, init_y).
        search_range_r : Half-range of radius search around init_r.
        step_xy        : Coarse spatial step (pixels).
        step_r         : Coarse radius step (pixels).
        is_pupil       : True for pupil boundary, False for limbus.
        fine_step_xy   : Fine spatial step (pixels).
        fine_step_r    : Fine radius step (pixels).
        fine_range     : Half-width of fine search around coarse best.

        Returns
        -------
        (best_x, best_y, best_r, best_response)
        """
        h, w = img.shape

        # ---- Stage 1: Coarse ----------------------------------------
        best_x, best_y, best_r, best_resp = self._grid_search(
            img, init_x, init_y, init_r,
            search_range_xy, search_range_r,
            step_xy, step_r,
            is_pupil, h, w,
        )

        # ---- Stage 2: Fine around coarse best -----------------------
        best_x, best_y, best_r, best_resp = self._grid_search(
            img, best_x, best_y, best_r,
            fine_range, fine_range,
            fine_step_xy, fine_step_r,
            is_pupil, h, w,
            prev_best=best_resp,
        )

        return int(best_x), int(best_y), int(best_r), float(best_resp)

    # ------------------------------------------------------------------
    # Internal grid search
    # ------------------------------------------------------------------

    def _grid_search(
        self, img, init_x, init_y, init_r,
        range_xy, range_r, step_xy, step_r,
        is_pupil, h, w, prev_best=-1.0,
    ):
        """Evaluates the IDO on a regular grid and returns the best (x,y,r,resp)."""
        cx_vals = np.arange(
            init_x - range_xy, init_x + range_xy + step_xy, step_xy
        )
        cy_vals = np.arange(
            init_y - range_xy, init_y + range_xy + step_xy, step_xy
        )
        r_vals = np.arange(
            init_r - range_r, init_r + range_r + step_r, step_r
        )

        # Clip to valid image region
        margin = 8
        cx_vals = cx_vals[(cx_vals >= margin) & (cx_vals <= w - margin)]
        cy_vals = cy_vals[(cy_vals >= margin) & (cy_vals <= h - margin)]
        # Enforce minimum radius — for pupil searches, don't consider tiny radii
        r_min_hard = max(6, int(init_r * 0.5))
        r_vals  = r_vals[(r_vals >= r_min_hard)]

        if len(cx_vals) == 0 or len(cy_vals) == 0 or len(r_vals) == 0:
            return init_x, init_y, init_r, prev_best

        best_x, best_y, best_r = init_x, init_y, init_r
        best_resp = prev_best

        for cx in cx_vals:
            for cy in cy_vals:
                responses = self._ido_response(img, cx, cy, r_vals, is_pupil)
                idx = int(np.argmax(responses))
                if responses[idx] > best_resp:
                    best_resp = responses[idx]
                    best_x = cx
                    best_y = cy
                    best_r = r_vals[idx]

        return best_x, best_y, best_r, best_resp


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic image: sclera(180) → iris(100) → pupil(20)
    test_img = np.ones((480, 640), dtype=np.uint8) * 180
    cv2.circle(test_img, (320, 240), 110, 100, -1)
    cv2.circle(test_img, (320, 240), 40,  20,  -1)
    test_img = cv2.GaussianBlur(test_img, (7, 7), 2)

    op = IntegroDifferentialOperator()

    # Pupil search from a slightly offset start
    x, y, r, score = op.search_circle(
        test_img, 315, 243, 35,
        search_range_xy=12, search_range_r=12,
        step_xy=3, step_r=3, is_pupil=True,
    )
    print(f"Pupil IDO  : x={x}  y={y}  r={r}  score={score:.4f}  (expected 320,240,40)")

    x2, y2, r2, s2 = op.search_circle(
        test_img, 320, 240, 100,
        search_range_xy=12, search_range_r=20,
        step_xy=3, step_r=3, is_pupil=False,
    )
    print(f"Limbus IDO : x={x2}  y={y2}  r={r2}  score={s2:.4f}  (expected 320,240,110)")
