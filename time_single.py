"""Time one full image through the pipeline."""
import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import cv2, yaml
from iris_recognition.segmentation.hybrid import HybridSegmenter
from iris_recognition.normalization.rubber_sheet import RubberSheetNormalizer
from iris_recognition.features.log_gabor import LogGaborExtractor

with open('iris_recognition/configs/config.yaml') as f:
    config = yaml.safe_load(f)

img = cv2.imread('Dataset/000/L/S5000L00.jpg')
seg = HybridSegmenter(config)
nrm = RubberSheetNormalizer(config)
ext = LogGaborExtractor(config)

t = time.time()
res = seg.segment(img)
print(f"Segment  : {(time.time()-t)*1000:.0f}ms  status={res['status']}")

t = time.time()
ni, nm = nrm.normalize(res['grayscale'], *res['pupil'], *res['iris'], res['mask'])
print(f"Normalize: {(time.time()-t)*1000:.0f}ms")

t = time.time()
c, m = ext.extract(ni, nm)
print(f"Extract  : {(time.time()-t)*1000:.0f}ms")

# Also time 5 images consecutively (amortised init cost)
t = time.time()
for _ in range(5):
    res = seg.segment(img)
    ni, nm = nrm.normalize(res['grayscale'], *res['pupil'], *res['iris'], res['mask'])
    c, m = ext.extract(ni, nm)
avg = (time.time()-t)*1000/5
print(f"\nAvg over 5 images: {avg:.0f}ms/img  => est 20k images: {avg*20000/1000/60:.0f} min @ 1 worker")
print(f"With 4 workers: ~{avg*20000/1000/60/4:.0f} min")
