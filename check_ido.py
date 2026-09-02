import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import cv2, numpy as np
from iris_recognition.segmentation.integro_differential import IntegroDifferentialOperator

img = cv2.imread('Dataset/000/L/S5000L00.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
smoothed = cv2.GaussianBlur(gray, (7,7), 2)

ido = IntegroDifferentialOperator()
# Known pupil ~(264, 258, 33)
x, y, r, score = ido.search_circle(smoothed, 264, 258, 33, search_range_xy=8, search_range_r=8, step_xy=1, step_r=1, is_pupil=True)
print(f'IDO pupil: x={x}, y={y}, r={r}, score={score:.4f}')

# Test with offset initial guess
x2, y2, r2, score2 = ido.search_circle(smoothed, 280, 270, 25, search_range_xy=20, search_range_r=15, step_xy=2, step_r=2, is_pupil=True)
print(f'IDO pupil (offset start): x={x2}, y={y2}, r={r2}, score={score2:.4f}')

# Test limbus IDO
from iris_recognition.segmentation.limbus import LimbusDetector
ldet = LimbusDetector()
lx, ly, lr, lscore = ldet.detect(smoothed, 264, 258, 33)
print(f'Limbus IDO: x={lx}, y={ly}, r={lr}, score={lscore:.4f}')
print(f'Expected iris approx (263, 257, 89)')

# Check if the IDO search is finding a good response
# by plotting response over radii
print("\nRadius scan at known center (263, 257):")
for test_r in range(75, 115, 5):
    _, _, best_r, resp = ido.search_circle(smoothed, 263, 257, test_r, 
                                            search_range_xy=0, search_range_r=0, 
                                            step_xy=1, step_r=1, is_pupil=False)
    print(f"  r={test_r}: resp={resp:.4f}")
