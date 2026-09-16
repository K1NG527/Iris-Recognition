"""
1:N Deep Learning Iris Identification CLI.

Identifies a query eye image against enrolled gallery identities using IrisDeepNet
and generates a visual side-by-side verification & ranking dashboard.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from iris_recognition.normalization.rubber_sheet import RubberSheetNormalizer
from iris_recognition.segmentation.hybrid import HybridSegmenter
from iris_recognition.deep_learning.model import IrisDeepNet
from iris_recognition.deep_learning.dataset import NormalizedIrisDataset
from iris_recognition.deep_learning.matcher import DeepIrisMatcher


def get_or_compute_segmentation(img_path, segmentation_csv="results/segmentation_results.csv"):
    """Finds segmentation parameters in CSV or falls back to HybridSegmenter."""
    rel_path = img_path.replace("\\", "/").replace("Dataset/", "")
    
    if os.path.exists(segmentation_csv):
        import csv
        with open(segmentation_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r_path = row["image_path"].replace("\\", "/")
                if r_path == rel_path or rel_path.endswith(r_path):
                    if row.get("status") == "GOOD":
                        return {
                            "pupil": (float(row["pupil_x"]), float(row["pupil_y"]), float(row["pupil_radius"])),
                            "iris": (float(row["iris_x"]), float(row["iris_y"]), float(row["iris_radius"])),
                            "status": "GOOD"
                        }

    # Fallback to online segmentation
    img = cv2.imread(img_path)
    if img is None:
        return {"status": "FAILED", "reason": "Cannot read image file"}
    segmenter = HybridSegmenter()
    return segmenter.segment(img)


def perform_identification(
    query_img_path,
    model_path="results/deep_learning/iris_deep_net.pt",
    gallery_subjects=None,
    top_k=5,
    threshold=0.28,
    out_visual="results/deep_learning/identification_match.png",
    device="cpu"
):
    """
    Performs 1:N Identification for query image against enrolled gallery templates.
    """
    if not os.path.exists(query_img_path):
        raise FileNotFoundError(f"Query image not found: {query_img_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}. Run evaluate_deep_iris.py first.")

    print("\n" + "=" * 65)
    print("      DEEP LEARNING 1:N IRIS IDENTIFICATION")
    print("=" * 65)
    print(f"Query Image  : {query_img_path}")
    print(f"Model Path   : {model_path}")
    print(f"Decision Thresh : {threshold:.3f} Cosine Similarity")

    # 1. Load trained IrisDeepNet
    checkpoint = torch.load(model_path, map_location=device)
    model = IrisDeepNet(in_channels=1, embedding_dim=checkpoint.get("embedding_dim", 256))
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    matcher = DeepIrisMatcher(model=model, device=device, roll_shifts=(-8, -4, 0, 4, 8))
    normalizer = RubberSheetNormalizer()
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 16))

    # 2. Process Query Image
    print("\n[1/3] Segmenting & Unwrapping Query Image...")
    query_bgr = cv2.imread(query_img_path)
    query_gray = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2GRAY)
    seg = get_or_compute_segmentation(query_img_path)

    if seg.get("status") != "GOOD":
        print(f"ERROR: Segmentation failed for query image. Reason: {seg.get('reason')}")
        return

    xp, yp, rp = seg["pupil"]
    xi, yi, ri = seg["iris"]

    H, W = query_gray.shape
    q_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(q_mask, (int(round(xi)), int(round(yi))), int(round(ri)), 255, -1)
    cv2.circle(q_mask, (int(round(xp)), int(round(yp))), int(round(rp)), 0, -1)

    q_norm, _ = normalizer.normalize(query_gray, xp, yp, rp, xi, yi, ri, q_mask)
    q_norm_enhanced = clahe.apply(q_norm)

    q_tensor = (torch.from_numpy(q_norm_enhanced).float().unsqueeze(0).unsqueeze(0) / 127.5) - 1.0
    with torch.no_grad():
        q_embs = matcher.extract_embedding(q_tensor, multi_shift=True).numpy() # (K, D)

    # 3. Build / Load Gallery Templates
    print("\n[2/3] Enrolling Gallery Identities & Computing 1:N Match Scores...")
    # By default, load enrolled gallery subjects from test/val splits
    if gallery_subjects is None:
        with open("splits/val_subjects.txt") as f:
            gallery_subjects = [line.strip() for line in f if line.strip()][:25]

    gallery_dataset = NormalizedIrisDataset(
        dataset_dir="Dataset",
        segmentation_csv="results/segmentation_results.csv",
        subject_filter=gallery_subjects,
        max_samples_per_eye=1, # 1 template per identity in gallery
        is_train=False,
        augment=False,
        cache_in_memory=True
    )

    gallery_list = []
    gallery_embs = []

    with torch.no_grad():
        for i in range(len(gallery_dataset)):
            item = gallery_dataset[i]
            img_tensor = item["image"].unsqueeze(0) # (1, 1, 64, 512)
            g_emb = matcher.extract_embedding(img_tensor, multi_shift=False).squeeze(0).numpy()
            sample = gallery_dataset.samples[i]
            gallery_list.append(sample)
            gallery_embs.append(g_emb)

    gallery_matrix = np.stack(gallery_embs, axis=0) # (M, D)

    # Match: (K, D) x (D, M) -> (K, M) -> max over K -> (M,)
    sim_matrix = np.dot(q_embs, gallery_matrix.T)
    scores = np.max(sim_matrix, axis=0)

    ranked_indices = np.argsort(-scores) # Descending

    # 4. Display 1:N Results
    print("\n" + "-" * 65)
    print(f"{'Rank':^6} | {'Identity':^16} | {'Cosine Sim':^12} | {'Decision':^14}")
    print("-" * 65)

    top_candidates = []
    for rank in range(min(top_k, len(ranked_indices))):
        idx = ranked_indices[rank]
        item = gallery_list[idx]
        score = float(scores[idx])
        ident = item["identity"]
        decision = "VERIFIED MATCH" if score >= threshold else "IMPOSTOR"
        print(f"#{rank + 1:^5d} | {ident:^16} | {score:^12.4f} | {decision:^14}")
        top_candidates.append({
            "rank": rank + 1,
            "identity": ident,
            "score": score,
            "decision": decision,
            "item": item
        })
    print("-" * 65)

    top1 = top_candidates[0]
    print(f"\nFinal Identification Verdict: {top1['identity']} (Confidence: {top1['score']:.4f})")

    # 5. Generate Visual Side-by-Side Match Dashboard
    print(f"\n[3/3] Rendering Visual Match Dashboard to: {out_visual}...")
    os.makedirs(os.path.dirname(out_visual), exist_ok=True)
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0], width_ratios=[1, 1, 1.2])

    # A. Query Eye with segmentation overlay
    ax_q = fig.add_subplot(gs[0, 0])
    q_overlay = query_bgr.copy()
    cv2.circle(q_overlay, (int(round(xp)), int(round(yp))), int(round(rp)), (0, 0, 255), 2) # Red pupil
    cv2.circle(q_overlay, (int(round(xi)), int(round(yi))), int(round(ri)), (0, 255, 0), 2) # Green iris
    ax_q.imshow(cv2.cvtColor(q_overlay, cv2.COLOR_BGR2RGB))
    ax_q.set_title(f"Query Eye Image\n({os.path.basename(query_img_path)})", fontweight="bold", fontsize=11)
    ax_q.axis("off")

    # B. Matched Top-1 Gallery Eye with overlay
    ax_g = fig.add_subplot(gs[0, 1])
    match_item = top1["item"]
    match_bgr = cv2.imread(match_item["image_path"])
    m_xp, m_yp, m_rp = match_item["pupil"]
    m_xi, m_yi, m_ri = match_item["iris"]
    m_overlay = match_bgr.copy()
    cv2.circle(m_overlay, (int(round(m_xp)), int(round(m_yp))), int(round(m_rp)), (0, 0, 255), 2)
    cv2.circle(m_overlay, (int(round(m_xi)), int(round(m_yi))), int(round(m_ri)), (0, 255, 0), 2)
    ax_g.imshow(cv2.cvtColor(m_overlay, cv2.COLOR_BGR2RGB))
    ax_g.set_title(f"Top-1 Matched Gallery Eye\nIdentity: {top1['identity']} (Sim: {top1['score']:.4f})", fontweight="bold", fontsize=11, color="darkgreen" if top1['score'] >= threshold else "crimson")
    ax_g.axis("off")

    # C. Top-K Candidate Cosine Similarity Bar Chart
    ax_bar = fig.add_subplot(gs[0, 2])
    c_names = [c["identity"] for c in top_candidates][::-1]
    c_scores = [c["score"] for c in top_candidates][::-1]
    colors = ["#28a745" if s >= threshold else "#dc3545" for s in c_scores]

    bars = ax_bar.barh(range(len(c_names)), c_scores, color=colors, height=0.55)
    ax_bar.set_yticks(range(len(c_names)))
    ax_bar.set_yticklabels(c_names, fontsize=10, fontweight="bold")
    ax_bar.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"Decision Thresh ({threshold:.2f})")
    ax_bar.set_xlim([0.0, 1.0])
    ax_bar.set_xlabel("Spherical Cosine Similarity", fontweight="bold")
    ax_bar.set_title("1:N Identification Candidate Ranking", fontweight="bold", fontsize=11)
    ax_bar.legend(loc="lower right")
    ax_bar.grid(True, linestyle=":", alpha=0.5)

    for bar, s in zip(bars, c_scores):
        ax_bar.text(s + 0.02, bar.get_y() + bar.get_height()/2, f"{s:.3f}", va="center", fontsize=9, fontweight="bold")

    # D. Query Normalized Iris Strip
    ax_qn = fig.add_subplot(gs[1, 0])
    ax_qn.imshow(q_norm_enhanced, cmap="gray", aspect="auto")
    ax_qn.set_title("Query Normalized Iris Strip (64 x 512)", fontweight="bold", fontsize=10)
    ax_qn.set_xlabel("Theta (0 to 2pi, 512 samples)")
    ax_qn.set_ylabel("Radius r")

    # E. Top-1 Gallery Normalized Iris Strip
    ax_gn = fig.add_subplot(gs[1, 1])
    # Load match normalized strip
    m_gray = cv2.cvtColor(match_bgr, cv2.COLOR_BGR2GRAY)
    m_mask = np.zeros(m_gray.shape, dtype=np.uint8)
    cv2.circle(m_mask, (int(round(m_xi)), int(round(m_yi))), int(round(m_ri)), 255, -1)
    cv2.circle(m_mask, (int(round(m_xp)), int(round(m_yp))), int(round(m_rp)), 0, -1)
    m_norm, _ = normalizer.normalize(m_gray, m_xp, m_yp, m_rp, m_xi, m_yi, m_ri, m_mask)
    m_norm_enhanced = clahe.apply(m_norm)

    ax_gn.imshow(m_norm_enhanced, cmap="gray", aspect="auto")
    ax_gn.set_title(f"Matched Gallery Normalized Strip (64 x 512)", fontweight="bold", fontsize=10)
    ax_gn.set_xlabel("Theta (0 to 2pi, 512 samples)")
    ax_gn.set_ylabel("Radius r")

    # F. Feature Map / Differential Comparison Strip
    ax_diff = fig.add_subplot(gs[1, 2])
    diff_strip = np.abs(q_norm_enhanced.astype(float) - m_norm_enhanced.astype(float)).astype(np.uint8)
    im_diff = ax_diff.imshow(diff_strip, cmap="hot", aspect="auto")
    ax_diff.set_title("Absolute Intensity Difference Strip", fontweight="bold", fontsize=10)
    ax_diff.set_xlabel("Theta")
    ax_diff.set_ylabel("Radius r")
    fig.colorbar(im_diff, ax=ax_diff, orientation="vertical", fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_visual, dpi=150)
    plt.close()
    print(f"Visual identification result successfully generated and saved to: {out_visual}\n")


def main():
    parser = argparse.ArgumentParser(description="1:N Deep Iris Identification")
    parser.add_argument("--image", type=str, default="Dataset/000/L/S5000L01.jpg", help="Path to query image")
    parser.add_argument("--model_path", type=str, default="results/deep_learning/iris_deep_net.pt", help="Path to model checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="Number of candidate matches to show")
    parser.add_argument("--threshold", type=float, default=0.28, help="Cosine similarity match threshold")
    parser.add_argument("--output_visual", type=str, default="results/deep_learning/identification_match.png", help="Path to save visual dashboard")
    args = parser.parse_args()

    perform_identification(
        query_img_path=args.image,
        model_path=args.model_path,
        top_k=args.top_k,
        threshold=args.threshold,
        out_visual=args.output_visual
    )


if __name__ == "__main__":
    main()
