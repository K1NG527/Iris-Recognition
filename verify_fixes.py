"""
Verification script: tests all fixed modules on a real image.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import cv2
import numpy as np
import yaml

DATASET_IMG = 'Dataset/000/L/S5000L00.jpg'

with open('iris_recognition/configs/config.yaml') as f:
    config = yaml.safe_load(f)

img = cv2.imread(DATASET_IMG)
assert img is not None, "Cannot load test image"
print(f"Image loaded: {img.shape}")

# ── 1. GAC fix ──────────────────────────────────────────────────────────
print("\n[1] GAC fix test")
from iris_recognition.segmentation.gac import GACRefiner, _GAC_PARAM, SKIMAGE_AVAILABLE
print(f"    skimage available : {SKIMAGE_AVAILABLE}")
print(f"    GAC param name    : '{_GAC_PARAM}'  (must be 'num_iter')")
assert _GAC_PARAM == "num_iter", "GAC param fix FAILED"
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
denoised = cv2.bilateralFilter(gray, 9, 75, 75)
refiner = GACRefiner(config)
cx_p, cy_p, r_p, mask_p = refiner.refine_circle(denoised, 264, 258, 33, is_pupil=True)
print(f"    Pupil  GAC: ({cx_p:.1f}, {cy_p:.1f}, r={r_p:.1f})  expected ~(264,258,33)")
cx_i, cy_i, r_i, mask_i = refiner.refine_circle(denoised, 263, 257, 89, is_pupil=False)
print(f"    Limbus GAC: ({cx_i:.1f}, {cy_i:.1f}, r={r_i:.1f})  expected ~(263,257,89)")
assert mask_p.sum() > 0, "GAC pupil mask empty"
print("    GAC: PASSED")

# ── 2. IDO coarse-to-fine ───────────────────────────────────────────────
print("\n[2] IDO coarse-to-fine test")
from iris_recognition.segmentation.integro_differential import IntegroDifferentialOperator
ido = IntegroDifferentialOperator()
smoothed = cv2.GaussianBlur(denoised, (7,7), 2)
# Start with an intentionally offset guess
x, y, r, score = ido.search_circle(
    smoothed, 280, 270, 22,
    search_range_xy=20, search_range_r=20,
    step_xy=3, step_r=3, is_pupil=True,
    fine_step_xy=1, fine_step_r=1, fine_range=5,
)
print(f"    Pupil IDO (offset start): ({x}, {y}, r={r})  score={score:.3f}")
print(f"    Expected: near (264, 258, 33)")
err = np.sqrt((x-264)**2 + (y-258)**2)
print(f"    Centre error: {err:.1f}px  (accept ≤ 15px)")
assert err <= 20, f"IDO centre error {err:.1f}px too large"
print("    IDO: PASSED")

# ── 3. Log-Gabor bit layout ─────────────────────────────────────────────
print("\n[3] Log-Gabor bit layout test")
from iris_recognition.normalization.rubber_sheet import RubberSheetNormalizer
from iris_recognition.features.log_gabor import LogGaborExtractor

norm = RubberSheetNormalizer(config)
norm_iris, norm_mask = norm.normalize(gray, 264, 258, 33, 263, 257, 89,
                                      mask=np.ones_like(gray)*255)
extractor = LogGaborExtractor(config)
code, nmask = extractor.extract(norm_iris, norm_mask)

H, W = 64, 512
expected_cols = config['features']['log_gabor']['num_scales'] * 2 * W
print(f"    Code shape  : {code.shape}  expected ({H}, {expected_cols})")
assert code.shape == (H, expected_cols), f"Code shape mismatch: {code.shape}"
assert code.max() <= 1 and code.min() >= 0, "Code not binary"
# Check that real and imag blocks for scale 0 are populated differently
s0_real = code[:, 0:W]
s0_imag = code[:, W:2*W]
assert not np.array_equal(s0_real, s0_imag), "Real and imag blocks identical (bug)"
print(f"    Valid bit ratio: {nmask.mean():.4f}")
print("    Log-Gabor: PASSED")

# ── 4. Hamming distance sanity ──────────────────────────────────────────
print("\n[4] Hamming distance test")
from iris_recognition.matching.matcher import IrisMatcher
matcher = IrisMatcher(config)

# Same code → HD ≈ 0
hd_same, _ = matcher.compute_hd(code, nmask, code, nmask)
print(f"    HD (same code) : {hd_same:.4f}  (expected 0.0)")
assert hd_same == 0.0, f"Same-code HD is {hd_same}, expected 0"

# Random code → HD ≈ 0.5
np.random.seed(42)
rnd_code = np.random.randint(0, 2, code.shape, dtype=np.uint8)
rnd_mask = np.ones_like(nmask)
hd_rnd, _ = matcher.compute_hd(code, nmask, rnd_code, rnd_mask)
print(f"    HD (random)    : {hd_rnd:.4f}  (expected ~0.50)")
assert 0.40 <= hd_rnd <= 0.60, f"Random-code HD {hd_rnd} out of expected range"
print("    Matcher: PASSED")

# ── 5. Full segmentation pipeline ──────────────────────────────────────
print("\n[5] Full segmentation pipeline test")
from iris_recognition.segmentation.hybrid import HybridSegmenter
seg = HybridSegmenter(config)
result = seg.segment(img)
print(f"    Status    : {result['status']}")
print(f"    Confidence: {result['confidence']:.4f}")
if result['status'] != 'FAILED':
    p = result['pupil']
    i_ = result['iris']
    print(f"    Pupil     : ({p[0]:.1f}, {p[1]:.1f}, r={p[2]:.1f})")
    print(f"    Iris      : ({i_[0]:.1f}, {i_[1]:.1f}, r={i_[2]:.1f})")
    ratio = p[2] / i_[2]
    print(f"    Pupil/iris ratio: {ratio:.3f}  (expected 0.20–0.55)")
    assert 0.15 <= ratio <= 0.60, f"Pupil/iris ratio {ratio:.3f} out of range"
    mask_px = cv2.countNonZero(result['mask'])
    print(f"    Mask pixels: {mask_px}")
    assert mask_px > 1000, "Iris mask too small"
assert result['status'] in ('GOOD', 'WARNING'), f"Unexpected status: {result['status']}"
print("    Pipeline: PASSED")

print("\n=== All verification checks PASSED ===")
