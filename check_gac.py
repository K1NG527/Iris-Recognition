import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import inspect
from skimage.segmentation import morphological_geodesic_active_contour as gac
from skimage.segmentation import inverse_gaussian_gradient
print("GAC signature:", inspect.signature(gac))
print("inv_gauss_grad signature:", inspect.signature(inverse_gaussian_gradient))

# Test GAC refinement on a synthetic iris
import cv2
import numpy as np

# Make a synthetic eye image with a clear pupil
img = np.ones((480, 640), dtype=np.uint8) * 150
cv2.circle(img, (320, 240), 40, 20, -1)   # pupil (dark)
cv2.circle(img, (320, 240), 100, 100, -1) # iris - not in the image (sclera fills)
# Actually: background=sclera(180), iris(100), pupil(20)
img2 = np.ones((480, 640), dtype=np.uint8) * 180
cv2.circle(img2, (320, 240), 100, 100, -1)
cv2.circle(img2, (320, 240), 40, 20, -1)
img2 = cv2.GaussianBlur(img2, (5, 5), 0)

from iris_recognition.segmentation.gac import GACRefiner
refiner = GACRefiner()
print("GAC enabled:", refiner.enabled)
cx, cy, r, mask = refiner.refine_circle(img2, 320, 240, 40, is_pupil=True)
print(f"Pupil GAC: cx={cx:.1f}, cy={cy:.1f}, r={r:.1f}")
print("Evolved mask sum:", mask.sum() // 255)

cx2, cy2, r2, mask2 = refiner.refine_circle(img2, 320, 240, 100, is_pupil=False)
print(f"Iris GAC: cx={cx2:.1f}, cy={cy2:.1f}, r={r2:.1f}")
print("Evolved iris mask sum:", mask2.sum() // 255)
