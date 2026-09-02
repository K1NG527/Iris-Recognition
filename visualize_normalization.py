"""
Visualize the iris normalization (rubber-sheet mapping) result.

Produces 3 separate images:
  1. Segmentation overlay  — original eye with pupil/iris circles
  2. Normalized iris        — unwrapped iris strip
  3. Normalized mask        — binary mask strip

Usage:
  python visualize_normalization.py
  python visualize_normalization.py --image path/to/eye.jpg
  python visualize_normalization.py --outdir results/normalization
  python visualize_normalization.py --no-display
"""

import os
import sys
import cv2
import argparse
import yaml
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from iris_recognition.segmentation.hybrid import HybridSegmenter
from iris_recognition.normalization.rubber_sheet import RubberSheetNormalizer


def load_config(config_path="iris_recognition/configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_segmentation_image(img, pupil, iris, mask):
    """Return the eye image with segmentation circles drawn on it."""
    overlay = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    xi, yi, ri = [int(v) for v in iris]
    xp, yp, rp = [int(v) for v in pupil]

    # Limbus (iris outer boundary) – blue
    cv2.circle(overlay, (xi, yi), max(1, ri), (255, 180, 0), 2, cv2.LINE_AA)
    # Pupil boundary – green
    cv2.circle(overlay, (xp, yp), max(1, rp), (0, 255, 120), 2, cv2.LINE_AA)

    # Mask contour – red
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1, cv2.LINE_AA)

    # Label
    add_label(overlay, "Segmentation  (green=pupil, blue=iris, red=mask)")
    return overlay


def build_normalized_iris_image(norm_iris, norm_mask):
    """Return a colour visualization of the normalized iris strip.
    Masked-out (invalid) regions are tinted red.
    The image is scaled up so it's clearly visible."""
    vis = cv2.cvtColor(norm_iris, cv2.COLOR_GRAY2BGR)

    # Tint invalid pixels red
    invalid = norm_mask == 0
    tint = vis.copy()
    tint[invalid] = [0, 0, 200]  # BGR red
    vis = cv2.addWeighted(vis, 0.6, tint, 0.4, 0)

    # Scale up: 4× height so the thin strip is easy to inspect
    h, w = vis.shape[:2]
    vis = cv2.resize(vis, (w, h * 4), interpolation=cv2.INTER_LINEAR)

    add_label(vis, "Normalized Iris  (red = masked-out region)")
    return vis


def build_normalized_mask_image(norm_mask):
    """Return a BGR visualization of the normalized mask, scaled up."""
    vis = cv2.cvtColor(norm_mask, cv2.COLOR_GRAY2BGR)

    # Scale up: 4× height
    h, w = vis.shape[:2]
    vis = cv2.resize(vis, (w, h * 4), interpolation=cv2.INTER_NEAREST)

    add_label(vis, "Normalized Mask  (white = valid iris)")
    return vis


def add_label(img, text, origin=(10, 25), scale=0.7, color=(255, 255, 255),
              bg_color=(0, 0, 0)):
    """Put a labelled banner on the image (in-place)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + baseline + 4),
                  bg_color, cv2.FILLED)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize iris normalization (rubber-sheet unwrapping)")
    parser.add_argument("--image", type=str,
                        default="Dataset/000/L/S5000L00.jpg",
                        help="Path to the eye image")
    parser.add_argument("--outdir", type=str, default="results/normalization",
                        help="Output directory for the 3 PNGs")
    parser.add_argument("--no-display", action="store_true",
                        help="Do not open GUI windows; save only")
    args = parser.parse_args()

    # ── Load config & modules ─────────────────────────────────────
    config     = load_config()
    segmenter  = HybridSegmenter(config)
    normalizer = RubberSheetNormalizer(config)

    # ── Read image ────────────────────────────────────────────────
    img = cv2.imread(args.image)
    if img is None:
        print(f"ERROR: Cannot read image: {args.image}")
        sys.exit(1)
    print(f"Loaded image: {args.image}  shape={img.shape}")

    # ── Segment ───────────────────────────────────────────────────
    seg = segmenter.segment(img)
    print(f"Segmentation status: {seg['status']}  confidence: {seg['confidence']:.4f}")

    if seg["status"] == "FAILED":
        print("Segmentation failed – cannot normalise this image.")
        sys.exit(1)

    xp, yp, rp = seg["pupil"]
    xi, yi, ri = seg["iris"]
    gray       = seg["grayscale"]
    iris_mask  = seg["mask"]

    print(f"Pupil  : center=({xp:.1f}, {yp:.1f})  r={rp:.1f}")
    print(f"Iris   : center=({xi:.1f}, {yi:.1f})  r={ri:.1f}")

    # ── Normalize ─────────────────────────────────────────────────
    norm_iris, norm_mask = normalizer.normalize(
        gray, xp, yp, rp, xi, yi, ri, iris_mask
    )
    valid_ratio = np.sum(norm_mask > 0) / norm_mask.size
    print(f"Normalized iris shape: {norm_iris.shape}")
    print(f"Valid mask ratio     : {valid_ratio:.2%}")

    # ── Build the 3 images ────────────────────────────────────────
    seg_img  = build_segmentation_image(img, seg["pupil"], seg["iris"], seg["mask"])
    iris_img = build_normalized_iris_image(norm_iris, norm_mask)
    mask_img = build_normalized_mask_image(norm_mask)

    # ── Save ──────────────────────────────────────────────────────
    os.makedirs(args.outdir, exist_ok=True)

    seg_path  = os.path.join(args.outdir, "1_segmentation.png")
    iris_path = os.path.join(args.outdir, "2_normalized_iris.png")
    mask_path = os.path.join(args.outdir, "3_normalized_mask.png")

    cv2.imwrite(seg_path,  seg_img)
    cv2.imwrite(iris_path, iris_img)
    cv2.imwrite(mask_path, mask_img)

    print(f"\nSaved 3 images to: {args.outdir}/")
    print(f"  1) {seg_path}")
    print(f"  2) {iris_path}")
    print(f"  3) {mask_path}")

    # ── Display ───────────────────────────────────────────────────
    if not args.no_display:
        cv2.imshow("1 - Segmentation", seg_img)
        cv2.imshow("2 - Normalized Iris", iris_img)
        cv2.imshow("3 - Normalized Mask", mask_img)
        print("\nPress any key to close all windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

