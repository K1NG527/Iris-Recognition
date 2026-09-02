import os
import cv2
import argparse
import yaml
import numpy as np

from iris_recognition.preprocessing.preprocessing import Preprocessor
from iris_recognition.segmentation.hybrid import HybridSegmenter
from iris_recognition.normalization.rubber_sheet import RubberSheetNormalizer
from iris_recognition.features.log_gabor import LogGaborExtractor
from iris_recognition.matching.matcher import IrisMatcher

def load_config(config_path="iris_recognition/configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_gallery(gallery_dir):
    """Loads all enrolled templates from the templates directory."""
    gallery = []
    if not os.path.exists(gallery_dir):
        return gallery
        
    for root, dirs, files in os.walk(gallery_dir):
        for f in files:
            if f.endswith("_template.npz"):
                path = os.path.join(root, f)
                try:
                    data = np.load(path)
                    # Check if template is valid (not failed segmentation)
                    if str(data["status"]) != "FAILED":
                        # Determine subject and side from folder names/file names
                        parts = path.replace("\\", "/").split("/")
                        # Structure: .../templates/<subject_id>/<L or R>/<base_name>_template.npz
                        sub_id = parts[-3]
                        side = "left" if parts[-2] == "L" else "right"
                        
                        gallery.append({
                            "subject_id": sub_id,
                            "eye_side": side,
                            "iris_code": data["iris_code"],
                            "noise_mask": data["noise_mask"],
                            "file_path": path
                        })
                except Exception:
                    pass
    return gallery

def recognize_image(img_path, gallery_dir, threshold=0.38):
    """Performs 1-to-N identification for a query eye image."""
    config = load_config()
    
    # 1. Read Image
    if not os.path.exists(img_path):
        print(f"Error: Query image {img_path} not found.")
        return
        
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to read image {img_path}.")
        return
        
    # 2. Extract Template
    segmenter = HybridSegmenter(config)
    normalizer = RubberSheetNormalizer(config)
    extractor = LogGaborExtractor(config)
    
    seg_res = segmenter.segment(img)
    if seg_res["status"] == "FAILED":
        print("\nIris Recognition Result")
        print("-----------------------")
        print("Identity: UNKNOWN")
        print(f"Confidence: {seg_res['confidence']:.4f}")
        print("Segmentation Status: FAILED")
        print(f"Reason: {seg_res.get('reason', 'LOW_SEGMENTATION_CONFIDENCE')}")
        return
        
    xp, yp, rp = seg_res["pupil"]
    xi, yi, ri = seg_res["iris"]
    iris_mask = seg_res["mask"]
    enhanced = seg_res["enhanced"]
    gray = seg_res["grayscale"]
    
    norm_iris, norm_mask = normalizer.normalize(gray, xp, yp, rp, xi, yi, ri, iris_mask)
    query_code, query_mask = extractor.extract(norm_iris, norm_mask)
    
    # 3. Load Gallery
    gallery = load_gallery(gallery_dir)
    if not gallery:
        print(f"Error: No valid enrolled templates found in gallery directory {gallery_dir}.")
        print("Please run the pipeline to enroll subjects first.")
        return
        
    print(f"Matching query against {len(gallery)} enrolled templates...")
    
    # 4. Compare Templates
    matcher = IrisMatcher(config)
    
    best_match = None
    min_hd = 1.0
    best_side = ""
    
    for candidate in gallery:
        hd, shift = matcher.compute_hd(query_code, query_mask, candidate["iris_code"], candidate["noise_mask"])
        if hd < min_hd:
            min_hd = hd
            best_match = candidate
            best_side = candidate["eye_side"]
            
    print("\nIris Recognition Result")
    print("-----------------------")
    
    if min_hd <= threshold and best_match is not None:
        print(f"Predicted Identity: Subject_{best_match['subject_id']}")
        print(f"Eye: {best_side.capitalize()}")
        print(f"Match Score (Hamming Distance): {min_hd:.4f}")
        print(f"Confidence: {seg_res['confidence']:.4f}")
        print("Segmentation Status:", seg_res["status"])
    else:
        print("Identity: UNKNOWN")
        print(f"Reason: NO_MATCH_FOUND (Best Hamming Distance was {min_hd:.4f}, threshold={threshold})")
        if best_match is not None:
            print(f"Closest Match: Subject_{best_match['subject_id']} ({best_side}) with score {min_hd:.4f}")
        print(f"Confidence: {seg_res['confidence']:.4f}")
        print("Segmentation Status:", seg_res["status"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Iris Recognition System")
    parser.add_argument("--image", type=str, required=True, help="Path to the eye image to recognize")
    parser.add_argument("--gallery", type=str, default="results/templates", help="Path to enrolled templates directory")
    parser.add_argument("--threshold", type=float, default=0.38, help="Hamming distance acceptance threshold")
    
    args = parser.parse_args()
    recognize_image(args.image, args.gallery, args.threshold)
