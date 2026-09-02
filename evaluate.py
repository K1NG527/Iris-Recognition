"""
Iris Recognition Evaluation Script.

Evaluates the recognition system using templates stored in results/templates/
with proper subject-wise biometric protocol:

  Verification:
    Genuine pairs  — two templates from the same identity (subject + eye side).
    Impostor pairs — templates from different identities.
    Metrics: FAR, FRR, EER, AUC, ROC curve.

  Identification:
    For each query template, rank the gallery by Hamming distance.
    Metrics: Rank-1, Rank-5 accuracy.

Gallery / Probe split:
    First image per identity → gallery.
    Remaining images → probe.

Subject-independent protocol is maintained because templates were already
split at the subject level during pipeline processing.
"""

import os
import sys
import csv
import time
import random
import argparse
import numpy as np
import yaml
from tqdm import tqdm

# Path bootstrap
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from iris_recognition.matching.matcher import IrisMatcher
from iris_recognition.evaluation.metrics import (
    calculate_verification_metrics,
    plot_distributions_and_roc,
)


# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

def load_config(path="iris_recognition/configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────
# Template loading
# ─────────────────────────────────────────────────────────────────────

def load_templates(templates_dir: str, split_filter=None):
    """Load all valid templates from disk.

    Returns a dict:
        { (subject_id, eye_side) : [ {iris_code, noise_mask, image_path}, ... ] }

    Parameters
    ----------
    split_filter : set of subject_ids or None
        If provided, only load templates for those subjects.
    """
    index = {}
    total_loaded = 0
    total_failed = 0

    for root, dirs, files in os.walk(templates_dir):
        dirs.sort()
        for fname in sorted(files):
            if not fname.endswith("_template.npz"):
                continue

            path = os.path.join(root, fname)
            # Expected structure: .../templates/<subject>/<eye>/<base>_template.npz
            parts = path.replace("\\", "/").split("/")
            if len(parts) < 3:
                continue
            subject_id = parts[-3]
            eye_raw    = parts[-2]
            side       = "left" if eye_raw.upper() == "L" else "right"

            if split_filter is not None and subject_id not in split_filter:
                continue

            try:
                d = np.load(path, allow_pickle=True)
                if str(d["status"]) == "FAILED":
                    total_failed += 1
                    continue
                entry = {
                    "iris_code":  d["iris_code"],
                    "noise_mask": d["noise_mask"],
                    "image_path": path,
                    "confidence": float(d["confidence"]),
                }
                key = (subject_id, side)
                index.setdefault(key, []).append(entry)
                total_loaded += 1
            except Exception:
                pass

    return index, total_loaded, total_failed


# ─────────────────────────────────────────────────────────────────────
# Gallery / Probe split
# ─────────────────────────────────────────────────────────────────────

def build_gallery_probe(index):
    """Split each identity's templates into gallery (first) and probe (rest)."""
    gallery = {}   # (subject, side) → single template dict
    probe   = []   # list of (identity, template_dict)

    for identity, templates in index.items():
        gallery[identity] = templates[0]
        for t in templates[1:]:
            probe.append((identity, t))

    return gallery, probe


# ─────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────

def run_evaluation(config, max_impostors=50000, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    output_dir    = config["dataset"]["output_dir"]
    templates_dir = os.path.join(output_dir, "templates")

    print(f"\nLoading templates from: {templates_dir}")
    index, n_loaded, n_failed = load_templates(templates_dir)

    print(f"  Valid templates : {n_loaded}")
    print(f"  Failed templates: {n_failed}")
    print(f"  Unique identities (subject+eye): {len(index)}")

    if n_loaded < 10:
        print("ERROR: Too few templates. Run the pipeline first.")
        return

    matcher = IrisMatcher(config)

    # ── Gallery / Probe split ─────────────────────────────────────────
    gallery, probe = build_gallery_probe(index)
    print(f"  Gallery entries : {len(gallery)}")
    print(f"  Probe entries   : {len(probe)}")

    # ── Genuine comparisons ───────────────────────────────────────────
    print("\nGenerating genuine comparisons…")
    genuine_scores = []
    for identity, templates in tqdm(index.items(), desc="Genuine", unit="id"):
        n = len(templates)
        for i in range(n):
            for j in range(i + 1, n):
                hd, _ = matcher.compute_hd(
                    templates[i]["iris_code"], templates[i]["noise_mask"],
                    templates[j]["iris_code"], templates[j]["noise_mask"],
                )
                genuine_scores.append(hd)

    print(f"  Genuine pairs   : {len(genuine_scores)}")

    # ── Impostor comparisons ──────────────────────────────────────────
    print("Generating impostor comparisons…")
    identity_list = list(index.keys())
    n_identities  = len(identity_list)

    impostor_scores = []
    n_attempts = 0
    max_attempts = max_impostors * 5

    with tqdm(total=max_impostors, desc="Impostors", unit="pair") as pbar:
        while len(impostor_scores) < max_impostors and n_attempts < max_attempts:
            n_attempts += 1
            id_a, id_b = random.sample(identity_list, 2)
            t_a = random.choice(index[id_a])
            t_b = random.choice(index[id_b])
            hd, _ = matcher.compute_hd(
                t_a["iris_code"], t_a["noise_mask"],
                t_b["iris_code"], t_b["noise_mask"],
            )
            impostor_scores.append(hd)
            pbar.update(1)

    genuine_scores  = np.array(genuine_scores,  dtype=np.float32)
    impostor_scores = np.array(impostor_scores, dtype=np.float32)

    print(f"  Impostor pairs  : {len(impostor_scores)}")

    # ── Verification metrics ──────────────────────────────────────────
    print("\nCalculating verification metrics…")
    metrics = calculate_verification_metrics(genuine_scores, impostor_scores)

    # ── Rank-1 / Rank-5 Identification ───────────────────────────────
    print("Calculating Rank-1 / Rank-5 identification…")
    gallery_list = [(identity, t) for identity, t in gallery.items()]

    # Cap probe size at 2000 for reasonable runtime
    eval_probe = probe
    if len(probe) > 2000:
        eval_probe = random.sample(probe, 2000)

    rank1_correct = 0
    rank5_correct = 0
    total_queries = len(eval_probe)

    t_start = time.time()
    for q_identity, q_tmpl in tqdm(eval_probe, desc="Rank-N", unit="query"):
        scores = []
        for g_identity, g_tmpl in gallery_list:
            hd, _ = matcher.compute_hd(
                q_tmpl["iris_code"],  q_tmpl["noise_mask"],
                g_tmpl["iris_code"],  g_tmpl["noise_mask"],
            )
            scores.append((hd, g_identity))
        scores.sort(key=lambda x: x[0])

        if scores[0][1] == q_identity:
            rank1_correct += 1
        if any(s[1] == q_identity for s in scores[:5]):
            rank5_correct += 1

    rank1_acc = rank1_correct / total_queries
    rank5_acc = rank5_correct / total_queries
    id_elapsed = time.time() - t_start

    # ── Print report ──────────────────────────────────────────────────
    print("\n" + "═" * 50)
    print("       IRIS RECOGNITION PERFORMANCE REPORT")
    print("═" * 50)
    print(f"  Enrolled identities           : {len(index)}")
    print(f"  Valid templates               : {n_loaded}")
    print(f"  Failed templates              : {n_failed}")
    fail_total = n_loaded + n_failed
    if fail_total > 0:
        print(f"  Segmentation failure rate     : {n_failed / fail_total:.2%}")
    print()
    print(f"  Genuine comparisons           : {len(genuine_scores)}")
    print(f"  Impostor comparisons          : {len(impostor_scores)}")
    print()
    print(f"  Avg genuine HD                : {genuine_scores.mean():.4f} ± {genuine_scores.std():.4f}")
    print(f"  Avg impostor HD               : {impostor_scores.mean():.4f} ± {impostor_scores.std():.4f}")
    print()
    print(f"  Equal Error Rate (EER)        : {metrics['eer']:.2%}")
    print(f"  EER threshold                 : {metrics['eer_threshold']:.4f}")
    print(f"  AUC                           : {metrics['auc']:.4f}")
    print()
    print(f"  Rank-1 Identification Rate    : {rank1_acc:.2%}  ({rank1_correct}/{total_queries})")
    print(f"  Rank-5 Identification Rate    : {rank5_acc:.2%}  ({rank5_correct}/{total_queries})")
    print(f"  Identification eval time      : {id_elapsed:.1f}s")
    print("═" * 50)

    # ── Save plots ────────────────────────────────────────────────────
    plot_distributions_and_roc(genuine_scores, impostor_scores, metrics, output_dir)

    # ── Append to experiments.csv ─────────────────────────────────────
    import datetime
    exp_path   = os.path.join(output_dir, "experiments.csv")
    csv_exists = os.path.exists(exp_path)
    fieldnames = [
        "date", "num_identities", "num_templates",
        "seg_failure_rate", "genuine_mean_hd", "impostor_mean_hd",
        "eer", "eer_threshold", "auc",
        "rank1_accuracy", "rank5_accuracy",
    ]
    with open(exp_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()
        writer.writerow({
            "date":             datetime.date.today().isoformat(),
            "num_identities":   len(index),
            "num_templates":    n_loaded,
            "seg_failure_rate": f"{n_failed / max(1, fail_total):.4f}",
            "genuine_mean_hd":  f"{genuine_scores.mean():.4f}",
            "impostor_mean_hd": f"{impostor_scores.mean():.4f}",
            "eer":              f"{metrics['eer']:.4f}",
            "eer_threshold":    f"{metrics['eer_threshold']:.4f}",
            "auc":              f"{metrics['auc']:.4f}",
            "rank1_accuracy":   f"{rank1_acc:.4f}",
            "rank5_accuracy":   f"{rank5_acc:.4f}",
        })
    print(f"  Experiment logged to          : {exp_path}")
    print(f"  ROC / distribution plots      : {output_dir}/performance_evaluation.png")

    return metrics, rank1_acc, rank5_acc


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Iris Recognition System")
    parser.add_argument("--max-impostors", type=int, default=50000)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    config = load_config()
    run_evaluation(config, max_impostors=args.max_impostors, seed=args.seed)
