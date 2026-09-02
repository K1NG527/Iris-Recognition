"""Profile timing of each pipeline stage."""
import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import cv2, numpy as np, yaml

with open('iris_recognition/configs/config.yaml') as f:
    config = yaml.safe_load(f)

img = cv2.imread('Dataset/000/L/S5000L00.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

from iris_recognition.preprocessing.preprocessing import Preprocessor
t = time.time()
prep = Preprocessor(config).process(img)
print(f"Preprocessing:   {(time.time()-t)*1000:.0f}ms")
denoised = prep['denoised']
enhanced = prep['enhanced']

from iris_recognition.segmentation.pupil import PupilDetector
t = time.time()
pd = PupilDetector(config)
r_pupil = pd.detect(enhanced)
print(f"Pupil detection: {(time.time()-t)*1000:.0f}ms  result={r_pupil[:3] if r_pupil else None}")

from iris_recognition.segmentation.integro_differential import IntegroDifferentialOperator
t = time.time()
ido = IntegroDifferentialOperator()
smoothed = cv2.GaussianBlur(denoised, (7,7), 2)
px, py, pr, _ = ido.search_circle(smoothed, int(r_pupil[0]), int(r_pupil[1]), int(r_pupil[2]),
    search_range_xy=12, search_range_r=12, step_xy=2, step_r=2, is_pupil=True,
    fine_step_xy=1, fine_step_r=1, fine_range=4)
print(f"IDO pupil:       {(time.time()-t)*1000:.0f}ms  result=({px},{py},{pr})")

from iris_recognition.segmentation.limbus import LimbusDetector
t = time.time()
ld = LimbusDetector(config)
lres = ld.detect(denoised, px, py, pr)
print(f"Limbus detection:{(time.time()-t)*1000:.0f}ms  result={lres[:3] if lres else None}")

from iris_recognition.segmentation.gac import GACRefiner
t = time.time()
gac = GACRefiner(config)
gac.iterations = 150
r = gac.refine_circle(denoised, px, py, pr, is_pupil=True)
print(f"GAC pupil (150): {(time.time()-t)*1000:.0f}ms")

t = time.time()
gac.iterations = 50
r = gac.refine_circle(denoised, px, py, pr, is_pupil=True)
print(f"GAC pupil (50):  {(time.time()-t)*1000:.0f}ms")

t = time.time()
gac.iterations = 20
r = gac.refine_circle(denoised, px, py, pr, is_pupil=True)
print(f"GAC pupil (20):  {(time.time()-t)*1000:.0f}ms")

# Test on half-resolution image
h, w = denoised.shape
small = cv2.resize(denoised, (w//2, h//2))
t = time.time()
gac.iterations = 50
r = gac.refine_circle(small, px//2, py//2, pr//2, is_pupil=True)
print(f"GAC pupil half-res (50): {(time.time()-t)*1000:.0f}ms")
