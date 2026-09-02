"""Profile IDO at different resolutions."""
import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import cv2, numpy as np
from iris_recognition.segmentation.integro_differential import IntegroDifferentialOperator

img = cv2.imread('Dataset/000/L/S5000L00.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
denoised = cv2.bilateralFilter(gray, 9, 75, 75)
smoothed = cv2.GaussianBlur(denoised, (7,7), 2)

# Half-res IDO
h, w = smoothed.shape
small = cv2.resize(smoothed, (w//2, h//2))

ido = IntegroDifferentialOperator()

t = time.time()
for _ in range(5):
    x, y, r, _ = ido.search_circle(smoothed, 264, 258, 33,
        search_range_xy=8, search_range_r=8, step_xy=2, step_r=2, is_pupil=True,
        fine_step_xy=1, fine_step_r=1, fine_range=3)
print(f"IDO full-res  : {(time.time()-t)*200:.0f}ms  result=({x},{y},{r})")

t = time.time()
for _ in range(5):
    x2, y2, r2, _ = ido.search_circle(small, 132, 129, 17,
        search_range_xy=8, search_range_r=8, step_xy=2, step_r=2, is_pupil=True,
        fine_step_xy=1, fine_step_r=1, fine_range=3)
print(f"IDO half-res  : {(time.time()-t)*200:.0f}ms  result=({x2*2},{y2*2},{r2*2}) (scaled back)")

# Limbus IDO
from iris_recognition.segmentation.limbus import LimbusDetector
ld = LimbusDetector()

t = time.time()
for _ in range(5):
    lres = ld.detect(denoised, 264, 258, 33)
print(f"Limbus full   : {(time.time()-t)*200:.0f}ms  result={lres[:3] if lres else None}")

t = time.time()
small_d = cv2.resize(denoised, (w//2, h//2))
for _ in range(5):
    lres2 = ld.detect(small_d, 132, 129, 17)
print(f"Limbus half   : {(time.time()-t)*200:.0f}ms  result={tuple(v*2 for v in lres2[:3]) if lres2 else None}")
