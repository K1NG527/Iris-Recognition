"""
Visualize the complete Iris Normalization and Feature Encoding pipeline based on Daugman's Rubber-Sheet model
and class lecture notes (Slides 70, 98, 99, 103, 107).

Outputs generated:
  1. 1_segmentation.png          — Original eye with localized pupil, limbus, and eyelid contours
  2. 2_normalized_iris_clean.png  — Clean unwrapped grayscale iris texture strip (Slide 98, 107)
  3. 3_normalized_iris_overlay.png— Unwrapped iris with highlighted occlusions (red tint)
  4. 4_normalized_mask.png        — Unwrapped binary mask (white=valid iris, black=occlusion)
  5. 5_iris_code.png              — 2D Log-Gabor phase-demodulated iris code (Slide 103, 107)
  6. normalization_pipeline.png   — Multi-stage end-to-end pipeline comparison chart (Slide 70)

Usage:
  python visualize_normalization.py
  python visualize_normalization.py --image Dataset/000/L/S5000L00.jpg --outdir results/normalization
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
from iris_recognition.features.log_gabor import LogGaborExtractor


def load_config(config_path="iris_recognition/configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def add_label(img, text, origin=(8, 20), scale=0.48, color=(255, 255, 255),
              bg_color=(0, 0, 0)):
    """Put a styled label banner on the image (in-place)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    # Ensure background rectangle stays within image bounds
    x_end = min(img.shape[1] - 2, x + tw + 4)
    cv2.rectangle(img, (x - 4, y - th - 5), (x_end, y + baseline + 4),
                  bg_color, cv2.FILLED)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def build_segmentation_image(img, pupil, iris, mask, label_extra=""):
    """Eye image with localized pupillary boundary, limbus boundary, and eyelid masks."""
    overlay = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    xi, yi, ri = [int(round(v)) for v in iris]
    xp, yp, rp = [int(round(v)) for v in pupil]

    # Limbus (outer boundary) — Blue
    cv2.circle(overlay, (xi, yi), max(1, ri), (255, 180, 0), 2, cv2.LINE_AA)
    # Pupil (inner boundary) — Green
    cv2.circle(overlay, (xp, yp), max(1, rp), (0, 255, 120), 2, cv2.LINE_AA)
    # Centers
    cv2.circle(overlay, (xi, yi), 2, (255, 180, 0), -1)
    cv2.circle(overlay, (xp, yp), 2, (0, 255, 120), -1)

    # Mask contour (eyelids/eyelashes) — Red
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1, cv2.LINE_AA)

    label = "1. Segmentation (Green=Pupil, Blue=Limbus, Red=Eyelids)"
    if label_extra:
        label += f" | {label_extra}"
    add_label(overlay, label, scale=0.55)
    return overlay


def build_clean_normalized_iris(norm_iris, label_extra=""):
    """Clean grayscale normalized iris strip I(r, theta) as shown in Slide 98 and 107."""
    vis = cv2.cvtColor(norm_iris, cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]
    vis_scaled = cv2.resize(vis, (w, h * 4), interpolation=cv2.INTER_LINEAR)
    label = "2. Normalized Iris Texture I(r, theta) [Daugman Rubber Sheet]"
    if label_extra:
        label += f" | {label_extra}"
    add_label(vis_scaled, label, scale=0.48)
    return vis_scaled


def build_overlay_normalized_iris(norm_iris, norm_mask, label_extra=""):
    """Normalized iris with occluded regions highlighted in translucent red."""
    vis = cv2.cvtColor(norm_iris, cv2.COLOR_GRAY2BGR)
    invalid = norm_mask == 0
    tint = vis.copy()
    tint[invalid] = [0, 0, 200]  # BGR Red
    vis = cv2.addWeighted(vis, 0.65, tint, 0.35, 0)
    h, w = vis.shape[:2]
    vis_scaled = cv2.resize(vis, (w, h * 4), interpolation=cv2.INTER_LINEAR)
    label = "3. Normalized Iris with Noise Mask (Red = Occlusions)"
    if label_extra:
        label += f" | {label_extra}"
    add_label(vis_scaled, label, scale=0.48)
    return vis_scaled


def build_normalized_mask_image(norm_mask, label_extra=""):
    """Binary normalized mask M(r, theta) — white=valid iris, black=occluded."""
    vis = cv2.cvtColor(norm_mask, cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]
    vis_scaled = cv2.resize(vis, (w, h * 4), interpolation=cv2.INTER_NEAREST)
    label = "4. Normalized Binary Mask M(r, theta) [White=Valid, Black=Occluded]"
    if label_extra:
        label += f" | {label_extra}"
    add_label(vis_scaled, label, scale=0.48)
    return vis_scaled


def build_iris_code_image(iris_code, label_extra=""):
    """Visualizes binary IrisCode extracted via 2D Log-Gabor phase demodulation (Slide 103, 107)."""
    vis = (iris_code * 255).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]
    vis_scaled = cv2.resize(vis, (512, 256), interpolation=cv2.INTER_NEAREST)
    label = "5. Extracted 2D Iris Code (Phase Demodulation Bits)"
    if label_extra:
        label += f" | {label_extra}"
    add_label(vis_scaled, label, scale=0.48)
    return vis_scaled


def build_pipeline_summary(seg_img, clean_iris, overlay_iris, mask_img, code_img):
    """Combines all stages into a structured overview chart matching Slide 70/98/107."""
    target_w = 640
    
    def prep_panel(img, target_h):
        return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    p1 = prep_panel(seg_img, 380)
    p2 = prep_panel(clean_iris, 140)
    p3 = prep_panel(overlay_iris, 140)
    p4 = prep_panel(mask_img, 140)
    p5 = prep_panel(code_img, 140)

    sep = np.full((3, target_w, 3), (0, 200, 255), dtype=np.uint8)
    pad = np.zeros((8, target_w, 3), dtype=np.uint8)
    summary = np.vstack([p1, sep, p2, sep, p3, sep, p4, sep, p5, pad])
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Iris Normalization & Pipeline per Class Notes")
    parser.add_argument("--image", type=str,
                        default="Dataset/000/L/S5000L00.jpg",
                        help="Path to the eye image")
    parser.add_argument("--outdir", type=str, default="results/normalization",
                        help="Output directory for generated PNGs")
    parser.add_argument("--no-display", action="store_true",
                        help="Do not open GUI windows; save only")
    args = parser.parse_args()

    config     = load_config()
    segmenter  = HybridSegmenter(config)
    normalizer = RubberSheetNormalizer(config)
    extractor  = LogGaborExtractor(config)

    img = cv2.imread(args.image)
    if img is None:
        print(f"ERROR: Cannot read image: {args.image}")
        sys.exit(1)
    print(f"Loaded image: {args.image}  shape={img.shape}")

    # 1. Segmentation
    seg = segmenter.segment(img)
    print(f"Segmentation status: {seg['status']}  confidence: {seg['confidence']:.4f}")
    if seg["status"] == "FAILED":
        print("Segmentation failed.")
        sys.exit(1)

    xp, yp, rp = seg["pupil"]
    xi, yi, ri = seg["iris"]
    gray       = seg["grayscale"]
    iris_mask  = seg["mask"]

    print(f"Pupillary Boundary: center=({xp:.1f}, {yp:.1f})  r={rp:.1f}")
    print(f"Limbus Boundary   : center=({xi:.1f}, {yi:.1f})  r={ri:.1f}")

    # 2. Normalization (Daugman Rubber Sheet)
    norm_iris, norm_mask = normalizer.normalize(
        gray, xp, yp, rp, xi, yi, ri, iris_mask
    )
    valid_ratio = np.sum(norm_mask > 0) / norm_mask.size
    print(f"Normalized Iris shape : {norm_iris.shape} (r x theta)")
    print(f"Valid Iris Texture    : {valid_ratio:.2%}")

    # 3. Feature Extraction (Log-Gabor Phase Demodulation)
    iris_code, noise_mask = extractor.extract(norm_iris, norm_mask)
    print(f"IrisCode shape        : {iris_code.shape}")

    # Build image outputs
    seg_img      = build_segmentation_image(img, seg["pupil"], seg["iris"], seg["mask"])
    clean_iris   = build_clean_normalized_iris(norm_iris)
    overlay_iris = build_overlay_normalized_iris(norm_iris, norm_mask)
    mask_img     = build_normalized_mask_image(norm_mask)
    code_img     = build_iris_code_image(iris_code)
    summary_img  = build_pipeline_summary(seg_img, clean_iris, overlay_iris, mask_img, code_img)

    # Save all to output directory
    os.makedirs(args.outdir, exist_ok=True)
    p_seg     = os.path.join(args.outdir, "1_segmentation.png")
    p_clean   = os.path.join(args.outdir, "2_normalized_iris_clean.png")
    p_overlay = os.path.join(args.outdir, "3_normalized_iris_overlay.png")
    p_mask    = os.path.join(args.outdir, "4_normalized_mask.png")
    p_code    = os.path.join(args.outdir, "5_iris_code.png")
    p_summary = os.path.join(args.outdir, "normalization_pipeline.png")

    cv2.imwrite(p_seg,     seg_img)
    cv2.imwrite(p_clean,   clean_iris)
    cv2.imwrite(p_overlay, overlay_iris)
    cv2.imwrite(p_mask,    mask_img)
    cv2.imwrite(p_code,    code_img)
    cv2.imwrite(p_summary, summary_img)

    print(f"\nAll visualization outputs saved to: {args.outdir}/")
    print(f"  1) {p_seg}")
    print(f"  2) {p_clean}")
    print(f"  3) {p_overlay}")
    print(f"  4) {p_mask}")
    print(f"  5) {p_code}")
    print(f"  6) {p_summary}")

    if not args.no_display:
        cv2.imshow("Iris Normalization Pipeline", summary_img)
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
