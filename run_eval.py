"""
Evaluation runner with optimized pair sampling.
Writes progress to eval_progress.log for monitoring.
"""
import os, sys, time, random, csv, datetime
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import yaml

LOG = 'eval_progress.log'

def log(msg):
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

# ── Config ────────────────────────────────────────────────────────────
with open('iris_recognition/configs/config.yaml') as f:
    config = yaml.safe_load(f)

from iris_recognition.matching.hamming import masked_hamming_distance
from iris_recognition.evaluation.metrics import (
    calculate_verification_metrics,
    plot_distributions_and_roc,
)

MAX_SHIFT    = config['matching']['rotation_shift']
NUM_SCALES   = config['features']['log_gabor']['num_scales']
output_dir   = 'results'
templates_dir = os.path.join(output_dir, 'templates')

# ── Load templates ────────────────────────────────────────────────────
log("Loading templates...")
index = {}
n_loaded = n_failed = 0

for root, dirs, files in os.walk(templates_dir):
    dirs.sort()
    for fname in sorted(files):
        if not fname.endswith('_template.npz'):
            continue
        path = os.path.join(root, fname)
        parts = path.replace('\\', '/').split('/')
        if len(parts) < 3:
            continue
        subject_id = parts[-3]
        side = 'left' if parts[-2].upper() == 'L' else 'right'
        try:
            d = np.load(path, allow_pickle=True)
            if str(d['status']) == 'FAILED':
                n_failed += 1
                continue
            index.setdefault((subject_id, side), []).append({
                'code': d['iris_code'].astype(bool),
                'mask': d['noise_mask'].astype(bool),
                'conf': float(d['confidence']),
            })
            n_loaded += 1
        except Exception:
            n_failed += 1

log(f"Valid: {n_loaded}  Failed: {n_failed}  Identities: {len(index)}")

# ── Gallery / Probe ───────────────────────────────────────────────────
gallery = {}
probe   = []
for identity, tmpls in index.items():
    gallery[identity] = tmpls[0]
    for t in tmpls[1:]:
        probe.append((identity, t))

log(f"Gallery: {len(gallery)}  Probe: {len(probe)}")

def hd(ta, tb):
    return masked_hamming_distance(
        ta['code'].view(np.uint8), ta['mask'].view(np.uint8),
        tb['code'].view(np.uint8), tb['mask'].view(np.uint8),
        max_shift=MAX_SHIFT, num_scales=NUM_SCALES,
    )[0]

# ── Genuine comparisons (~74k pairs) ─────────────────────────────────
log("Computing genuine comparisons (all intra-identity pairs)...")
genuine_scores = []
t0 = time.time()
identities = list(index.items())
for idx_i, (identity, tmpls) in enumerate(identities):
    n = len(tmpls)
    for i in range(n):
        for j in range(i + 1, n):
            genuine_scores.append(hd(tmpls[i], tmpls[j]))
    if (idx_i + 1) % 200 == 0:
        elapsed = time.time() - t0
        rate = len(genuine_scores) / elapsed
        log(f"  Genuine {idx_i+1}/{len(identities)} identities  {len(genuine_scores)} pairs  {rate:.0f} pairs/s")

log(f"Genuine pairs: {len(genuine_scores)}  Mean HD: {np.mean(genuine_scores):.4f}  ({time.time()-t0:.1f}s)")

# ── Impostor comparisons (20k pairs) ─────────────────────────────────
log("Computing impostor comparisons (20,000 pairs)...")
random.seed(42)
identity_list   = list(index.keys())
impostor_scores = []
target = 20000
t0 = time.time()
attempts = 0

while len(impostor_scores) < target and attempts < target * 5:
    attempts += 1
    id_a, id_b = random.sample(identity_list, 2)
    t_a = random.choice(index[id_a])
    t_b = random.choice(index[id_b])
    impostor_scores.append(hd(t_a, t_b))
    if len(impostor_scores) % 2000 == 0:
        elapsed = time.time() - t0
        rate = len(impostor_scores) / elapsed
        log(f"  Impostor {len(impostor_scores)}/{target}  {rate:.0f} pairs/s")

log(f"Impostor pairs: {len(impostor_scores)}  Mean HD: {np.mean(impostor_scores):.4f}  ({time.time()-t0:.1f}s)")

genuine_scores  = np.array(genuine_scores,  dtype=np.float32)
impostor_scores = np.array(impostor_scores, dtype=np.float32)

# ── Verification metrics ──────────────────────────────────────────────
log("Calculating verification metrics...")
metrics = calculate_verification_metrics(genuine_scores, impostor_scores)
log(f"EER: {metrics['eer']:.4f}  AUC: {metrics['auc']:.4f}")

# ── Rank-1 / Rank-5 (500 queries × full gallery) ─────────────────────
log("Calculating Rank-1/5 identification (500 probe queries vs full gallery)...")
gallery_list = list(gallery.items())
random.seed(42)
eval_probe = random.sample(probe, min(500, len(probe)))
rank1 = rank5 = 0
t0 = time.time()

for qi, (q_id, q_t) in enumerate(eval_probe):
    scores = []
    for g_id, g_t in gallery_list:
        d_val, _ = masked_hamming_distance(
            q_t['code'].view(np.uint8), q_t['mask'].view(np.uint8),
            g_t['code'].view(np.uint8), g_t['mask'].view(np.uint8),
            max_shift=MAX_SHIFT, num_scales=NUM_SCALES,
        )
        scores.append((d_val, g_id))
    scores.sort(key=lambda x: x[0])
    if scores[0][1] == q_id:
        rank1 += 1
    if any(s[1] == q_id for s in scores[:5]):
        rank5 += 1
    if (qi + 1) % 50 == 0:
        elapsed = time.time() - t0
        rate = (qi + 1) / elapsed
        log(f"  Rank-N progress: {qi+1}/{len(eval_probe)}  r1={rank1/(qi+1):.3f}  {rate:.2f} q/s")

rank1_acc = rank1 / len(eval_probe)
rank5_acc = rank5 / len(eval_probe)
log(f"Rank-1: {rank1_acc:.4f} ({rank1}/{len(eval_probe)})  Rank-5: {rank5_acc:.4f}  ({time.time()-t0:.1f}s)")

# ── Final report ──────────────────────────────────────────────────────
sep = "=" * 52
log(sep)
log("       IRIS RECOGNITION PERFORMANCE REPORT")
log(sep)
fail_total = n_loaded + n_failed
log(f"  Enrolled identities  : {len(index)}")
log(f"  Valid templates      : {n_loaded}")
log(f"  Failed templates     : {n_failed}")
log(f"  Seg failure rate     : {n_failed/max(1,fail_total):.2%}")
log("")
log(f"  Genuine pairs        : {len(genuine_scores)}")
log(f"  Impostor pairs       : {len(impostor_scores)}")
log(f"  Avg genuine HD       : {genuine_scores.mean():.4f} +/- {genuine_scores.std():.4f}")
log(f"  Avg impostor HD      : {impostor_scores.mean():.4f} +/- {impostor_scores.std():.4f}")
log("")
log(f"  EER                  : {metrics['eer']:.2%}")
log(f"  EER threshold        : {metrics['eer_threshold']:.4f}")
log(f"  AUC                  : {metrics['auc']:.4f}")
log("")
log(f"  Rank-1 Accuracy      : {rank1_acc:.2%}  ({rank1}/{len(eval_probe)})")
log(f"  Rank-5 Accuracy      : {rank5_acc:.2%}  ({rank5}/{len(eval_probe)})")
log(sep)

# ── Plots ─────────────────────────────────────────────────────────────
plot_distributions_and_roc(genuine_scores, impostor_scores, metrics, output_dir)

# ── experiments.csv ───────────────────────────────────────────────────
exp_path   = os.path.join(output_dir, 'experiments.csv')
fieldnames = ['date','num_identities','num_templates','seg_failure_rate',
              'genuine_mean_hd','impostor_mean_hd','eer','eer_threshold',
              'auc','rank1_accuracy','rank5_accuracy']
csv_exists = os.path.exists(exp_path)
with open(exp_path, 'a', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if not csv_exists:
        w.writeheader()
    w.writerow({
        'date':             datetime.date.today().isoformat(),
        'num_identities':   len(index),
        'num_templates':    n_loaded,
        'seg_failure_rate': f"{n_failed/max(1,fail_total):.4f}",
        'genuine_mean_hd':  f"{genuine_scores.mean():.4f}",
        'impostor_mean_hd': f"{impostor_scores.mean():.4f}",
        'eer':              f"{metrics['eer']:.4f}",
        'eer_threshold':    f"{metrics['eer_threshold']:.4f}",
        'auc':              f"{metrics['auc']:.4f}",
        'rank1_accuracy':   f"{rank1_acc:.4f}",
        'rank5_accuracy':   f"{rank5_acc:.4f}",
    })

log(f"Experiment logged: {exp_path}")
log("EVALUATION COMPLETE")
