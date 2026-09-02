import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from iris_recognition.segmentation.gac import GACRefiner
import cv2, numpy as np

# Real iris image test
img = cv2.imread('Dataset/000/L/S5000L00.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
denoised = cv2.bilateralFilter(gray, 9, 75, 75)

refiner = GACRefiner()
print("GAC enabled:", refiner.enabled)
print("GAC iterations:", refiner.iterations)

# Known pupil: (264, 258, 33)
cx, cy, r, mask = refiner.refine_circle(denoised, 264, 258, 33, is_pupil=True)
print(f'Pupil GAC: cx={cx:.1f}, cy={cy:.1f}, r={r:.1f}')
print('Pupil mask pixels:', np.sum(mask > 0))

# Known iris: (263, 257, 89)
cx2, cy2, r2, mask2 = refiner.refine_circle(denoised, 263, 257, 89, is_pupil=False)
print(f'Iris GAC: cx={cx2:.1f}, cy={cy2:.1f}, r={r2:.1f}')
print('Iris mask pixels:', np.sum(mask2 > 0))
