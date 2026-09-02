import os, sys, cv2, numpy as np
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import yaml

with open('iris_recognition/configs/config.yaml') as f:
    config = yaml.safe_load(f)

from iris_recognition.segmentation.hybrid import HybridSegmenter
from iris_recognition.normalization.rubber_sheet import RubberSheetNormalizer
from iris_recognition.features.log_gabor import LogGaborExtractor

img = cv2.imread('Dataset/000/L/S5000L00.jpg')
print('Image shape:', img.shape)
print('Image dtype:', img.dtype)

segmenter = HybridSegmenter(config)
result = segmenter.segment(img)
print('Segmentation status:', result['status'])
print('Segmentation confidence:', result['confidence'])
if result['status'] != 'FAILED':
    print('Pupil:', result['pupil'])
    print('Iris:', result['iris'])
    mask = result['mask']
    print('Mask non-zero pixels:', cv2.countNonZero(mask), 'of', mask.shape[0]*mask.shape[1])
    
    normalizer = RubberSheetNormalizer(config)
    gray = result['grayscale']
    xp, yp, rp = result['pupil']
    xi, yi, ri = result['iris']
    norm_iris, norm_mask = normalizer.normalize(gray, xp, yp, rp, xi, yi, ri, mask)
    print('Normalized iris shape:', norm_iris.shape)
    valid_ratio = np.sum(norm_mask>0) / norm_mask.size
    print('Valid mask ratio (normalized):', valid_ratio)
    
    extractor = LogGaborExtractor(config)
    code, noise_mask = extractor.extract(norm_iris, norm_mask)
    print('Iris code shape:', code.shape)
    valid_code_ratio = np.sum(noise_mask) / noise_mask.size
    print('Valid bits ratio in iris code:', valid_code_ratio)
