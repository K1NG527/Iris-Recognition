"""
Check if GAC is actually evolving the contour or just returning the initial estimate.
The current gac.py has a bug: it uses 'iterations' keyword but skimage's API uses 'num_iter'.
Let's verify.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import cv2
import numpy as np
from skimage.segmentation import morphological_geodesic_active_contour as skimage_gac
from skimage.segmentation import inverse_gaussian_gradient

img = cv2.imread('Dataset/000/L/S5000L00.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
denoised = cv2.bilateralFilter(gray, 9, 75, 75)
h, w = denoised.shape

# Test with correct num_iter parameter
blurred = cv2.GaussianBlur(denoised, (5, 5), 0)
gimage = inverse_gaussian_gradient(blurred, alpha=5.0, sigma=2.0)

# Initial contour - circle slightly off
init_ls = np.zeros((h, w), dtype=np.int32)
cv2.circle(init_ls, (260, 255), 30, 1, -1)  # slightly off from true (264, 258, 33)

print("Testing GAC with num_iter=100...")
try:
    evolved = skimage_gac(
        gimage,
        num_iter=100,
        init_level_set=init_ls,
        smoothing=1,
        balloon=1.0,
        threshold=0.3
    )
    evolved_mask = (evolved > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(evolved_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        (fx, fy), fr = cv2.minEnclosingCircle(largest)
        print(f"GAC evolved pupil: cx={fx:.1f}, cy={fy:.1f}, r={fr:.1f}")
    print("num_iter=100 works!")
except Exception as e:
    print(f"num_iter parameter error: {e}")

# Test with wrong 'iterations' parameter (the bug in current code)
print("\nTesting GAC with iterations=100 (incorrect param name)...")
try:
    evolved2 = skimage_gac(
        gimage,
        iterations=100,  # Wrong name
        init_level_set=init_ls,
        smoothing=1,
        balloon=1.0,
        threshold=0.3
    )
    print("iterations=100 also works (no error)")
except TypeError as e:
    print(f"BUG CONFIRMED: {e}")
