"""
Batch normalization visualization for 10 iris images across 5 subjects (L + R eyes)
conforming to class notes on Daugman Rubber-Sheet Normalization and Gabor Wavelet Encoding.

For each subject and eye, saves:
  1_segmentation.png           — Original eye with localized pupil, limbus, and eyelid contours
  2_normalized_iris_clean.png  — Clean grayscale unwrapped iris texture strip (Slide 98, 107)
  3_normalized_iris_overlay.png— Unwrapped iris with occlusions highlighted (red tint)
  4_normalized_mask.png        — Unwrapped binary mask (white=valid iris, black=occluded)
  5_iris_code.png              — 2D Log-Gabor phase-demodulated iris code (Slide 103, 107)
  normalization_summary.png    — Multi-stage comparison chart (Slide 70, 107)

Directory hierarchy:
  results/normalization_batch/
    Subject_000/
      L/
      R/
    Subject_001/
      ...

Usage:
  python batch_normalize_visual.py
"""

import os
import sys
import cv2
import yaml
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from iris_recognition.segmentation.hybrid import HybridSegmenter
from iris_recognition.normalization.rubber_sheet import RubberSheetNormalizer
from iris_recognition.features.log_gabor import LogGaborExtractor


# ── Configuration ─────────────────────────────────────────────────
DATASET_DIR = "Dataset"
OUTPUT_DIR  = "results/normalization_batch"

# 5 subjects × 2 eyes (L + R) = 10 images
SUBJECTS = ["000", "001", "002", "003", "004"]
SIDES    = ["L", "R"]


def load_config(config_path="iris_recognition/configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def add_label(img, text, origin=(8, 20), scale=0.48,
              color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Put a styled label banner on the image (in-place)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    x_end = min(img.shape[1] - 2, x + tw + 4)
    cv2.rectangle(img, (x - 4, y - th - 5), (x_end, y + baseline + 4),
                  bg_color, cv2.FILLED)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def build_segmentation_image(img, pupil, iris, mask, label_extra=""):
    overlay = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    xi, yi, ri = [int(round(v)) for v in iris]
    xp, yp, rp = [int(round(v)) for v in pupil]

    cv2.circle(overlay, (xi, yi), max(1, ri), (255, 180, 0), 2, cv2.LINE_AA)
    cv2.circle(overlay, (xp, yp), max(1, rp), (0, 255, 120), 2, cv2.LINE_AA)
    cv2.circle(overlay, (xi, yi), 2, (255, 180, 0), -1)
    cv2.circle(overlay, (xp, yp), 2, (0, 255, 120), -1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1, cv2.LINE_AA)

    label = "1. Segmentation (Green=Pupil, Blue=Limbus, Red=Eyelids)"
    if label_extra:
        label += f" | {label_extra}"
    add_label(overlay, label, scale=0.52)
    return overlay


def build_clean_normalized_iris(norm_iris, label_extra=""):
    vis = cv2.cvtColor(norm_iris, cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]
    vis_scaled = cv2.resize(vis, (w, h * 4), interpolation=cv2.INTER_LINEAR)
    label = "2. Normalized Iris Texture I(r, theta) [Daugman Rubber Sheet]"
    if label_extra:
        label += f" | {label_extra}"
    add_label(vis_scaled, label, scale=0.48)
    return vis_scaled


def build_overlay_normalized_iris(norm_iris, norm_mask, label_extra=""):
    vis = cv2.cvtColor(norm_iris, cv2.COLOR_GRAY2BGR)
    invalid = norm_mask == 0
    tint = vis.copy()
    tint[invalid] = [0, 0, 200]
    vis = cv2.addWeighted(vis, 0.65, tint, 0.35, 0)
    h, w = vis.shape[:2]
    vis_scaled = cv2.resize(vis, (w, h * 4), interpolation=cv2.INTER_LINEAR)
    label = "3. Normalized Iris with Noise Mask (Red = Occlusions)"
    if label_extra:
        label += f" | {label_extra}"
    add_label(vis_scaled, label, scale=0.48)
    return vis_scaled


def build_normalized_mask_image(norm_mask, label_extra=""):
    vis = cv2.cvtColor(norm_mask, cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]
    vis_scaled = cv2.resize(vis, (w, h * 4), interpolation=cv2.INTER_NEAREST)
    label = "4. Normalized Binary Mask M(r, theta) [White=Valid, Black=Occluded]"
    if label_extra:
        label += f" | {label_extra}"
    add_label(vis_scaled, label, scale=0.48)
    return vis_scaled


def build_iris_code_image(iris_code, label_extra=""):
    vis = (iris_code * 255).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    vis_scaled = cv2.resize(vis, (512, 256), interpolation=cv2.INTER_NEAREST)
    label = "5. Extracted 2D Iris Code (Phase Demodulation Bits)"
    if label_extra:
        label += f" | {label_extra}"
    add_label(vis_scaled, label, scale=0.48)
    return vis_scaled


def build_pipeline_summary(seg_img, clean_iris, overlay_iris, mask_img, code_img):
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
    return np.vstack([p1, sep, p2, sep, p3, sep, p4, sep, p5, pad])


def process_one(img_path, out_dir, subject_id, side, segmenter, normalizer, extractor):
    """Process a single image and save the full visualization suite to out_dir."""
    side_name = "Left" if side == "L" else "Right"
    tag = f"Subject {subject_id} ({side_name} Eye)"

    img = cv2.imread(img_path)
    if img is None:
        print(f"  [SKIP] Cannot read: {img_path}")
        return False

    seg = segmenter.segment(img)
    if seg["status"] == "FAILED":
        print(f"  [FAIL] Segmentation failed for {tag}: {seg.get('reason','')}")
        return False

    xp, yp, rp = seg["pupil"]
    xi, yi, ri = seg["iris"]
    gray       = seg["grayscale"]
    iris_mask  = seg["mask"]

    norm_iris, norm_mask = normalizer.normalize(
        gray, xp, yp, rp, xi, yi, ri, iris_mask
    )
    iris_code, noise_mask = extractor.extract(norm_iris, norm_mask)

    os.makedirs(out_dir, exist_ok=True)

    seg_img      = build_segmentation_image(img, seg["pupil"], seg["iris"], seg["mask"], tag)
    clean_iris   = build_clean_normalized_iris(norm_iris, tag)
    overlay_iris = build_overlay_normalized_iris(norm_iris, norm_mask, tag)
    mask_img     = build_normalized_mask_image(norm_mask, tag)
    code_img     = build_iris_code_image(iris_code, tag)
    summary_img  = build_pipeline_summary(seg_img, clean_iris, overlay_iris, mask_img, code_img)

    cv2.imwrite(os.path.join(out_dir, "1_segmentation.png"),            seg_img)
    cv2.imwrite(os.path.join(out_dir, "2_normalized_iris_clean.png"),   clean_iris)
    cv2.imwrite(os.path.join(out_dir, "3_normalized_iris_overlay.png"), overlay_iris)
    cv2.imwrite(os.path.join(out_dir, "4_normalized_mask.png"),         mask_img)
    cv2.imwrite(os.path.join(out_dir, "5_iris_code.png"),               code_img)
    cv2.imwrite(os.path.join(out_dir, "normalization_summary.png"),     summary_img)

    valid_ratio = np.sum(norm_mask > 0) / norm_mask.size
    print(f"  [OK]   {tag} | Conf={seg['confidence']:.4f} | Valid Iris={valid_ratio:.1%}")
    return True


def main():
    config     = load_config()
    segmenter  = HybridSegmenter(config)
    normalizer = RubberSheetNormalizer(config)
    extractor  = LogGaborExtractor(config)

    print(f"Batch normalization visualization (Daugman Rubber-Sheet & Gabor Encoding)")
    print(f"Subjects: {SUBJECTS}")
    print(f"Output  : {OUTPUT_DIR}/\n")

    total = 0
    ok    = 0

    for subj in SUBJECTS:
        for side in SIDES:
            side_dir = os.path.join(DATASET_DIR, subj, side)
            if not os.path.isdir(side_dir):
                print(f"  [SKIP] Folder not found: {side_dir}")
                continue

            images = sorted([f for f in os.listdir(side_dir)
                             if f.lower().endswith(('.jpg', '.png', '.bmp'))])
            if not images:
                print(f"  [SKIP] No images in: {side_dir}")
                continue

            img_path = os.path.join(side_dir, images[0])
            out_dir  = os.path.join(OUTPUT_DIR, f"Subject_{subj}", side)
            total += 1

            if process_one(img_path, out_dir, subj, side, segmenter, normalizer, extractor):
                ok += 1

    print(f"\n{'='*55}")
    print(f"Done! {ok}/{total} images processed successfully.")
    print(f"Results saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
