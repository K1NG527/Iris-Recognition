"""
PyTorch Dataset for Normalized Iris Biometric Recognition.
Loads eye images, applies Daugman's Rubber-Sheet Normalization,
CLAHE contrast enhancement, and iris-specific domain augmentations.
"""

import os
import csv
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from ..normalization.rubber_sheet import RubberSheetNormalizer


class NormalizedIrisDataset(Dataset):
    """
    Dataset that provides normalized iris strips for deep learning.
    
    Each identity is defined by (subject_id, eye_side) since left and right
    eyes of the same individual possess completely independent iris texture patterns.
    """
    def __init__(
        self,
        dataset_dir="Dataset",
        segmentation_csv="results/segmentation_results.csv",
        subject_filter=None,
        max_samples_per_eye=None,
        is_train=True,
        augment=True,
        cache_in_memory=True,
        norm_width=512,
        norm_height=64
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.is_train = is_train
        self.augment = augment and is_train
        self.cache_in_memory = cache_in_memory
        
        self.normalizer = RubberSheetNormalizer({"normalization": {"width": norm_width, "height": norm_height}})
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 16))

        self.samples = []
        self.identity_to_label = {}
        self.label_to_identity = {}

        # 1. Parse segmentation CSV
        if not os.path.exists(segmentation_csv):
            raise FileNotFoundError(f"Segmentation CSV not found: {segmentation_csv}")

        # Set of allowed subjects if filtered
        allowed_subjects = set(subject_filter) if subject_filter is not None else None

        with open(segmentation_csv, "r") as f:
            reader = csv.DictReader(f)
            # Group per identity to enforce max_samples_per_eye if desired
            id_counts = {}

            for row in reader:
                if row.get("status", "") != "GOOD":
                    continue

                rel_path = row["image_path"].replace("\\", "/")
                parts = rel_path.split("/")
                if len(parts) < 3:
                    continue
                subj = parts[0]
                side = parts[1].upper() # 'L' or 'R'
                
                if allowed_subjects is not None and subj not in allowed_subjects:
                    continue

                identity_key = f"{subj}_{side}"
                count = id_counts.get(identity_key, 0)
                if max_samples_per_eye is not None and count >= max_samples_per_eye:
                    continue

                abs_img_path = os.path.join(self.dataset_dir, rel_path)
                if not os.path.exists(abs_img_path):
                    continue

                try:
                    pupil_x = float(row["pupil_x"])
                    pupil_y = float(row["pupil_y"])
                    pupil_r = float(row["pupil_radius"])
                    iris_x = float(row["iris_x"])
                    iris_y = float(row["iris_y"])
                    iris_r = float(row["iris_radius"])
                except (ValueError, KeyError):
                    continue

                if identity_key not in self.identity_to_label:
                    lbl = len(self.identity_to_label)
                    self.identity_to_label[identity_key] = lbl
                    self.label_to_identity[lbl] = identity_key

                sample = {
                    "image_path": abs_img_path,
                    "rel_path": rel_path,
                    "identity": identity_key,
                    "label": self.identity_to_label[identity_key],
                    "pupil": (pupil_x, pupil_y, pupil_r),
                    "iris": (iris_x, iris_y, iris_r),
                }
                self.samples.append(sample)
                id_counts[identity_key] = count + 1

        self.num_classes = len(self.identity_to_label)
        self.cached_norms = {}

        if self.cache_in_memory:
            self._preload_cache()

    def _preload_cache(self):
        """Pre-normalizes all iris images once into memory for rapid CPU training."""
        for idx, sample in enumerate(self.samples):
            norm_strip = self._load_and_normalize(sample)
            self.cached_norms[idx] = norm_strip

    def _load_and_normalize(self, sample):
        """Loads raw image, applies rubber-sheet unwrapping and CLAHE contrast enhancement."""
        img = cv2.imread(sample["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback zero image
            return np.zeros((self.normalizer.height, self.normalizer.width), dtype=np.uint8)

        H, W = img.shape
        xp, yp, rp = sample["pupil"]
        xi, yi, ri = sample["iris"]

        # Approximate iris mask
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(mask, (int(round(xi)), int(round(yi))), int(round(ri)), 255, -1)
        cv2.circle(mask, (int(round(xp)), int(round(yp))), int(round(rp)), 0, -1)

        norm_iris, _ = self.normalizer.normalize(img, xp, yp, rp, xi, yi, ri, mask)
        
        # Biometric CLAHE enhancement
        norm_iris = self.clahe.apply(norm_iris)
        return norm_iris

    def _apply_augmentation(self, norm_strip):
        """
        Data augmentation tailored for iris biometrics:
        1. Circular horizontal roll (simulating head tilt / torsional eye rotation +-12 px ~ +-8.4 deg)
        2. Slight gamma / brightness perturbation
        3. Random rectangular noise occlusions (simulating partial eyelid/eyelash artifacts)
        """
        augmented = norm_strip.copy()

        # 1. Circular roll along theta (horizontal axis)
        if random.random() < 0.7:
            shift = random.randint(-14, 14)
            augmented = np.roll(augmented, shift, axis=1)

        # 2. Photometric jitter
        if random.random() < 0.5:
            gamma = random.uniform(0.85, 1.15)
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
            augmented = cv2.LUT(augmented, table)

        # 3. Small occlusion patches
        if random.random() < 0.3:
            h, w = augmented.shape
            occ_h = random.randint(4, 12)
            occ_w = random.randint(20, 60)
            y0 = random.randint(0, h - occ_h)
            x0 = random.randint(0, w - occ_w)
            augmented[y0:y0 + occ_h, x0:x0 + occ_w] = random.randint(0, 50)

        return augmented

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self.cache_in_memory and idx in self.cached_norms:
            norm_strip = self.cached_norms[idx]
        else:
            norm_strip = self._load_and_normalize(sample)

        if self.augment:
            norm_strip = self._apply_augmentation(norm_strip)

        # Convert to float tensor normalized to [-1, 1]
        tensor = torch.from_numpy(norm_strip).float().unsqueeze(0) # (1, H, W)
        tensor = (tensor / 127.5) - 1.0

        return {
            "image": tensor,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "identity": sample["identity"],
            "image_path": sample["image_path"],
            "rel_path": sample["rel_path"],
        }
