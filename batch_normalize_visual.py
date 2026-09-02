"""
Batch normalization visualization for 10 iris images.

Picks 1 Left-eye + 1 Right-eye image from 5 different subjects and
saves the 3 visualization PNGs per image in a clean folder hierarchy:

  results/normalization_batch/
    Subject_000/
      L/
        1_segmentation.png
        2_normalized_iris.png
        3_normalized_mask.png
      R/
        1_segmentation.png
        2_normalized_iris.png
        3_normalized_mask.png
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


# ── Configuration ─────────────────────────────────────────────────
DATASET_DIR = "Dataset"
OUTPUT_DIR  = "results/normalization_batch"

# 5 subjects × 2 eyes (L + R) = 10 images
SUBJECTS = ["000", "001", "002", "003", "004"]
SIDES    = ["L", "R"]


def load_config(config_path="iris_recognition/configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def add_label(img, text, origin=(10, 25), scale=0.7,
              color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Put a labelled banner on the image (in-place)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + baseline + 4),
                  bg_color, cv2.FILLED)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def build_segmentation_image(img, pupil, iris, mask, label_extra=""):
    overlay = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    xi, yi, ri = [int(v) for v in iris]
    xp, yp, rp = [int(v) for v in pupil]
    cv2.circle(overlay, (xi, yi), max(1, ri), (255, 180, 0), 2, cv2.LINE_AA)
    cv2.circle(overlay, (xp, yp), max(1, rp), (0, 255, 120), 2, cv2.LINE_AA)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1, cv2.LINE_AA)
    label = "Segmentation  (green=pupil, blue=iris, red=mask)"
    if label_extra:
        label += f"  |  {label_extra}"
    add_label(overlay, label)
    return overlay


def build_normalized_iris_image(norm_iris, norm_mask, label_extra=""):
    vis = cv2.cvtColor(norm_iris, cv2.COLOR_GRAY2BGR)
    invalid = norm_mask == 0
    tint = vis.copy()
    tint[invalid] = [0, 0, 200]
    vis = cv2.addWeighted(vis, 0.6, tint, 0.4, 0)
    # Scale up 4× height
    h, w = vis.shape[:2]
    vis = cv2.resize(vis, (w, h * 4), interpolation=cv2.INTER_LINEAR)
    label = "Normalized Iris  (red = masked-out region)"
    if label_extra:
        label += f"  |  {label_extra}"
    add_label(vis, label)
    return vis


def build_normalized_mask_image(norm_mask, label_extra=""):
    vis = cv2.cvtColor(norm_mask, cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]
    vis = cv2.resize(vis, (w, h * 4), interpolation=cv2.INTER_NEAREST)
    label = "Normalized Mask  (white = valid iris)"
    if label_extra:
        label += f"  |  {label_extra}"
    add_label(vis, label)
    return vis


def process_one(img_path, out_dir, subject_id, side, segmenter, normalizer):
    """Process a single image and save the 3 PNGs to out_dir."""
    side_name = "Left" if side == "L" else "Right"
    tag = f"Subject {subject_id} - {side_name} Eye"

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

    os.makedirs(out_dir, exist_ok=True)

    seg_img  = build_segmentation_image(img, seg["pupil"], seg["iris"], seg["mask"], tag)
    iris_img = build_normalized_iris_image(norm_iris, norm_mask, tag)
    mask_img = build_normalized_mask_image(norm_mask, tag)

    cv2.imwrite(os.path.join(out_dir, "1_segmentation.png"),     seg_img)
    cv2.imwrite(os.path.join(out_dir, "2_normalized_iris.png"),  iris_img)
    cv2.imwrite(os.path.join(out_dir, "3_normalized_mask.png"),  mask_img)

    valid_ratio = np.sum(norm_mask > 0) / norm_mask.size
    print(f"  [OK]   {tag}  |  conf={seg['confidence']:.4f}  valid={valid_ratio:.1%}")
    return True


def main():
    config     = load_config()
    segmenter  = HybridSegmenter(config)
    normalizer = RubberSheetNormalizer(config)

    print(f"Batch normalization visualization")
    print(f"Subjects: {SUBJECTS}")
    print(f"Output  : {OUTPUT_DIR}/\n")

    total = 0
    ok    = 0

    for subj in SUBJECTS:
        for side in SIDES:
            # Pick the first image in the folder
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

            if process_one(img_path, out_dir, subj, side, segmenter, normalizer):
                ok += 1

    print(f"\n{'='*50}")
    print(f"Done!  {ok}/{total} images processed successfully.")
    print(f"Results saved in: {OUTPUT_DIR}/")
    print(f"\nFolder structure:")
    print(f"  {OUTPUT_DIR}/")
    for subj in SUBJECTS:
        print(f"    Subject_{subj}/")
        for side in SIDES:
            print(f"      {side}/")
            print(f"        1_segmentation.png")
            print(f"        2_normalized_iris.png")
            print(f"        3_normalized_mask.png")


if __name__ == "__main__":
    main()
