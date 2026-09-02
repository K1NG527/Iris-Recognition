"""
Batch processing pipeline: processes all eye images through the full
segmentation → normalization → feature-extraction chain and saves
templates + overlays to disk.

Supports:
  --split       : process only train / val / test / all
  --workers     : parallel workers (multiprocessing, Windows-safe)
  --max-images  : cap for development testing
  --force       : ignore cached .npz files and reprocess
  --no-overlays : skip saving overlay PNGs (faster)
  --resume      : (default) skip images whose .npz already exists
"""

import os
import sys
import cv2
import csv
import time
import argparse
import yaml
import traceback
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ── path bootstrap so workers can import the package ──────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from iris_recognition.segmentation.hybrid import HybridSegmenter
from iris_recognition.normalization.rubber_sheet import RubberSheetNormalizer
from iris_recognition.features.log_gabor import LogGaborExtractor


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def load_config(config_path="iris_recognition/configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _template_path(output_dir, img_rel):
    rel_dir   = os.path.dirname(img_rel)
    base_name = os.path.splitext(os.path.basename(img_rel))[0]
    return os.path.join(output_dir, "templates", rel_dir, f"{base_name}_template.npz")

def _overlay_path(output_dir, img_rel):
    rel_dir   = os.path.dirname(img_rel)
    base_name = os.path.splitext(os.path.basename(img_rel))[0]
    return os.path.join(output_dir, "overlays", rel_dir, f"{base_name}_overlay.png")


# ─────────────────────────────────────────────────────────────────────
# Per-image worker  (top-level so multiprocessing can pickle it)
# ─────────────────────────────────────────────────────────────────────

def _process_image(args):
    """
    Process one eye image end-to-end.

    args = (img_rel_path, dataset_path, output_dir, save_overlays, force)

    Returns a result dict with fields:
        image_path, status, confidence, skipped, error,
        pupil_x, pupil_y, pupil_radius, iris_x, iris_y, iris_radius
    """
    img_rel, dataset_path, output_dir, save_overlays, force = args

    tpl_path = _template_path(output_dir, img_rel)
    ovl_path = _overlay_path(output_dir, img_rel)

    # ── Resume: skip if template already exists and we are not forcing ──
    if not force and os.path.exists(tpl_path):
        try:
            d = np.load(tpl_path, allow_pickle=True)
            status     = str(d["status"])
            confidence = float(d["confidence"])
            rec = dict(image_path=img_rel, status=status,
                       confidence=confidence, skipped=True, error=None,
                       pupil_x="", pupil_y="", pupil_radius="",
                       iris_x="",  iris_y="",  iris_radius="")
            if status != "FAILED":
                try:
                    pc = d["pupil_circle"]
                    lc = d["limbus_circle"]
                    rec.update(pupil_x=f"{pc[0]:.2f}", pupil_y=f"{pc[1]:.2f}",
                               pupil_radius=f"{pc[2]:.2f}",
                               iris_x=f"{lc[0]:.2f}",  iris_y=f"{lc[1]:.2f}",
                               iris_radius=f"{lc[2]:.2f}")
                except Exception:
                    pass
            return rec
        except Exception:
            pass   # corrupted cache → fall through and reprocess

    abs_img = os.path.join(dataset_path, img_rel)
    img = cv2.imread(abs_img)
    if img is None:
        return dict(image_path=img_rel, status="FAILED", confidence=0.0,
                    skipped=False, error="UNREADABLE_IMAGE",
                    pupil_x="", pupil_y="", pupil_radius="",
                    iris_x="",  iris_y="",  iris_radius="")

    try:
        config     = load_config()
        segmenter  = HybridSegmenter(config)
        normalizer = RubberSheetNormalizer(config)
        extractor  = LogGaborExtractor(config)

        # ── Segment ───────────────────────────────────────────────────
        seg = segmenter.segment(img)
        status     = seg["status"]
        confidence = seg["confidence"]

        tpl_dir = os.path.dirname(tpl_path)
        os.makedirs(tpl_dir, exist_ok=True)

        if status == "FAILED":
            np.savez(tpl_path, status=status, confidence=confidence,
                     reason=seg.get("reason", "UNKNOWN"))
            return dict(image_path=img_rel, status=status,
                        confidence=confidence, skipped=False,
                        error=seg.get("reason", "SEG_FAILED"),
                        pupil_x="", pupil_y="", pupil_radius="",
                        iris_x="",  iris_y="",  iris_radius="")

        xp, yp, rp = seg["pupil"]
        xi, yi, ri = seg["iris"]
        iris_mask  = seg["mask"]
        gray       = seg["grayscale"]

        # ── Normalize ────────────────────────────────────────────────
        norm_iris, norm_mask = normalizer.normalize(
            gray, xp, yp, rp, xi, yi, ri, iris_mask
        )

        # ── Extract features ─────────────────────────────────────────
        iris_code, noise_mask = extractor.extract(norm_iris, norm_mask)

        # ── Save template ────────────────────────────────────────────
        np.savez(tpl_path,
                 iris_code=iris_code,
                 noise_mask=noise_mask,
                 status=status,
                 confidence=confidence,
                 pupil_circle=np.array([xp, yp, rp], dtype=np.float32),
                 limbus_circle=np.array([xi, yi, ri], dtype=np.float32))

        # ── Save overlay ─────────────────────────────────────────────
        if save_overlays:
            ovl_dir = os.path.dirname(ovl_path)
            os.makedirs(ovl_dir, exist_ok=True)
            ov = img.copy()
            # Limbus (blue), pupil (green), mask contours (red)
            cv2.circle(ov, (int(xi), int(yi)), max(1, int(ri)), (255, 0, 0), 2)
            cv2.circle(ov, (int(xp), int(yp)), max(1, int(rp)), (0, 255, 0), 2)
            ctrs, _ = cv2.findContours(
                iris_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(ov, ctrs, -1, (0, 0, 255), 1)
            cv2.imwrite(ovl_path, ov)

        return dict(image_path=img_rel, status=status,
                    confidence=confidence, skipped=False, error=None,
                    pupil_x=f"{xp:.2f}", pupil_y=f"{yp:.2f}",
                    pupil_radius=f"{rp:.2f}",
                    iris_x=f"{xi:.2f}",  iris_y=f"{yi:.2f}",
                    iris_radius=f"{ri:.2f}")

    except Exception as exc:
        tb = traceback.format_exc(limit=3)
        return dict(image_path=img_rel, status="FAILED", confidence=0.0,
                    skipped=False, error=f"EXCEPTION: {exc}",
                    pupil_x="", pupil_y="", pupil_radius="",
                    iris_x="",  iris_y="",  iris_radius="")


# ─────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────

class PipelineRunner:

    def __init__(self, config_path="iris_recognition/configs/config.yaml"):
        self.config      = load_config(config_path)
        self.dataset_dir = self.config["dataset"]["path"]
        self.output_dir  = self.config["dataset"]["output_dir"]
        self.index_path  = self.config["dataset"]["index_path"]

    def _load_index(self, split_name):
        """Return image_path list filtered by split."""
        rows = []
        with open(self.index_path, "r") as f:
            for row in csv.DictReader(f):
                if split_name == "all" or row["split"] == split_name:
                    rows.append(row["image_path"])
        return rows

    def run(
        self,
        split_name  = "all",
        workers     = 4,
        max_images  = -1,
        resume      = True,
        save_overlays = True,
    ):
        if not os.path.exists(self.index_path):
            print("ERROR: Dataset index not found. Run data/dataset_index.py first.")
            return

        img_paths = self._load_index(split_name)
        if max_images > 0:
            img_paths = img_paths[:max_images]

        total = len(img_paths)
        print(f"Pipeline  split='{split_name}'  images={total}  workers={workers}"
              f"  resume={resume}  overlays={save_overlays}")

        os.makedirs(self.output_dir, exist_ok=True)

        args_list = [
            (p, self.dataset_dir, self.output_dir, save_overlays, not resume)
            for p in img_paths
        ]

        results   = []
        t_start   = time.time()

        if workers > 1:
            # Windows-safe: use 'spawn' context (default on Windows).
            # chunksize > 1 reduces IPC overhead for many small tasks.
            chunk = max(1, total // (workers * 8))
            with Pool(processes=workers) as pool:
                for res in tqdm(
                    pool.imap_unordered(_process_image, args_list, chunksize=chunk),
                    total=total,
                    desc="Processing",
                    unit="img",
                ):
                    results.append(res)
        else:
            for args in tqdm(args_list, desc="Processing", unit="img"):
                results.append(_process_image(args))

        elapsed = time.time() - t_start

        # ── Summary ──────────────────────────────────────────────────
        skipped  = sum(1 for r in results if r["skipped"])
        good     = sum(1 for r in results if not r["skipped"] and r["status"] == "GOOD")
        warning  = sum(1 for r in results if not r["skipped"] and r["status"] == "WARNING")
        failed   = sum(1 for r in results if not r["skipped"] and r["status"] == "FAILED")
        processed = good + warning + failed

        print("\n────────────────────────────────────────")
        print(f"  Total images           : {total}")
        print(f"  Skipped (cached)       : {skipped}")
        print(f"  Newly processed        : {processed}")
        print(f"    GOOD                 : {good}")
        print(f"    WARNING              : {warning}")
        print(f"    FAILED               : {failed}")
        if processed > 0:
            fail_rate = failed / processed
            print(f"  Failure rate           : {fail_rate:.2%}")
            print(f"  Speed                  : {processed / max(elapsed,1):.1f} img/s")
        print(f"  Elapsed                : {elapsed:.1f}s")
        print("────────────────────────────────────────")

        # ── Write segmentation_results.csv ───────────────────────────
        csv_path   = os.path.join(self.output_dir, "segmentation_results.csv")
        fieldnames = [
            "image_path", "status", "confidence",
            "pupil_x", "pupil_y", "pupil_radius",
            "iris_x",  "iris_y",  "iris_radius",
            "error",
        ]

        # Merge with existing records (resume mode)
        existing = {}
        if resume and os.path.exists(csv_path):
            try:
                with open(csv_path, "r") as f:
                    for row in csv.DictReader(f):
                        existing[row["image_path"]] = row
            except Exception:
                pass

        for r in results:
            rec = {k: r.get(k, "") for k in fieldnames}
            rec["error"] = rec["error"] or ""
            existing[r["image_path"]] = rec

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for key in sorted(existing):
                writer.writerow(existing[key])

        print(f"  Results CSV            : {csv_path}")


# ─────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Required for Windows multiprocessing
    from multiprocessing import freeze_support
    freeze_support()

    parser = argparse.ArgumentParser(description="Iris Recognition Batch Pipeline")
    parser.add_argument("--split",        default="all",
                        choices=["train", "val", "test", "all"])
    parser.add_argument("--workers",      type=int, default=4)
    parser.add_argument("--max-images",   type=int, default=-1)
    parser.add_argument("--force",        action="store_true",
                        help="Reprocess even cached images")
    parser.add_argument("--no-overlays",  action="store_true",
                        help="Skip saving overlay PNGs")
    args = parser.parse_args()

    runner = PipelineRunner()
    runner.run(
        split_name   = args.split,
        workers      = args.workers,
        max_images   = args.max_images,
        resume       = not args.force,
        save_overlays= not args.no_overlays,
    )
