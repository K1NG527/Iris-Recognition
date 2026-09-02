import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_verification_metrics(genuine_scores, impostor_scores):
    """Calculates FAR, FRR, EER, and AUC given genuine and impostor similarity/distance scores.
    
    In our case, scores are Hamming distances (lower is better, 0.0 is perfect match).
    """
    thresholds = np.linspace(0.0, 1.0, 1000)
    far = []
    frr = []
    
    total_gen = len(genuine_scores)
    total_imp = len(impostor_scores)
    
    if total_gen == 0 or total_imp == 0:
        return {
            "thresholds": thresholds, "far": [1.0]*1000, "frr": [1.0]*1000,
            "eer": 1.0, "eer_threshold": 0.5, "auc": 0.5
        }
        
    for t in thresholds:
        # FAR: False Acceptances / Impostor Attempts
        # An impostor is accepted if their distance score <= t
        fa = np.sum(impostor_scores <= t)
        far.append(fa / total_imp)
        
        # FRR: False Rejections / Genuine Attempts
        # A genuine subject is rejected if their distance score > t
        fr = np.sum(genuine_scores > t)
        frr.append(fr / total_gen)
        
    far = np.array(far)
    frr = np.array(frr)
    
    # EER is where FAR = FRR
    # Find the threshold where the difference between FAR and FRR is minimized
    diff = np.abs(far - frr)
    idx = np.argmin(diff)
    
    eer = (far[idx] + frr[idx]) / 2.0
    eer_threshold = thresholds[idx]
    
    # Calculate Area Under Curve (AUC) using trapezoidal integration of ROC
    # ROC is TAR (1 - FRR) vs FAR
    # We sort by FAR to integrate
    sort_idx = np.argsort(far)
    far_sorted = far[sort_idx]
    tar_sorted = (1.0 - frr)[sort_idx]
    
    auc = np.trapz(tar_sorted, far_sorted)
    # Since FAR is sorted descending (far = 1.0 at t=1.0 down to 0.0 at t=0.0),
    # we take absolute value
    auc = abs(auc)
    
    return {
        "thresholds": thresholds,
        "far": far,
        "frr": frr,
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "auc": float(auc)
    }

def plot_distributions_and_roc(genuine_scores, impostor_scores, metrics, output_dir):
    """Generates and saves the score distribution and ROC plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Plot Score Distribution
    ax1.hist(genuine_scores, bins=30, alpha=0.6, color='green', label='Genuine Scores', edgecolor='black', density=True)
    ax1.hist(impostor_scores, bins=50, alpha=0.6, color='red', label='Impostor Scores', edgecolor='black', density=True)
    
    # Draw EER threshold line
    ax1.axvline(metrics["eer_threshold"], color='blue', linestyle='--', linewidth=2, 
                label=f'EER Threshold ({metrics["eer_threshold"]:.3f})')
    
    ax1.set_xlabel("Hamming Distance")
    ax1.set_ylabel("Probability Density")
    ax1.set_title("Genuine vs Impostor Distance Distributions")
    ax1.legend(loc='upper right')
    ax1.grid(True)
    
    # 2. Plot ROC Curve (TAR vs FAR)
    tar = 1.0 - metrics["frr"]
    ax2.plot(metrics["far"], tar, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {metrics["auc"]:.4f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    
    # Mark EER point
    ax2.plot([metrics["eer"]], [1.0 - metrics["eer"]], marker='o', markersize=8, color="blue", 
             label=f'EER Point ({metrics["eer"]:.2%})')
    
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Acceptance Rate (FAR)')
    ax2.set_ylabel('True Acceptance Rate (TAR)')
    ax2.set_title('Receiver Operating Characteristic (ROC)')
    ax2.legend(loc="lower right")
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "performance_evaluation.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Performance plots successfully saved to: {plot_path}")
