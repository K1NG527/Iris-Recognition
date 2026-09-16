# 👁️ Iris Recognition System

A high-performance, modular Python implementation of an **End-to-End Iris Recognition System** based on **Daugman's Rubber-Sheet Model**, **Log-Gabor Feature Extraction**, and **Bit-Masked Hamming Distance Matching**.

---

## 🌟 Key Features

* **Segmentation & Boundary Detection**:
  * **Integro-Differential Operator (IDO)** for rapid pupil and limbus boundary estimation.
  * **Morphological Geodesic Active Contours (GAC)** for sub-pixel boundary refinement.
  * Eyelid and eyelash occlusion masking for noise mitigation.
* **Iris Normalization**:
  * Daugman's Rubber-Sheet Model mapping non-concentric circular iris boundaries into fixed rectangular polar coordinates ($64 \times 512$).
  * Preserves spatial frequency features regardless of pupil dilation/constriction.
* **Feature Extraction**:
  * Multi-scale 2D Log-Gabor filters generating compact binary iris codes.
  * Bit-mask generation indicating valid iris texture vs. occluded regions.
* **Matching & Recognition**:
  * Bit-masked Hamming distance computation with rotation alignment ($\pm 10$ angular shifts).
  * $1:1$ Verification and $1:N$ Identification capabilities.
* **Evaluation & Visualization Tools**:
  * Automated ROC/EER evaluation tools.
  * Visualization utilities for inspecting normalization and segmentation overlays.

---

## 📁 Project Structure

```
IRIS/
├── iris_recognition/             # Core Package
│   ├── configs/                  # Configuration files (config.yaml)
│   ├── data/                     # Indexing and data loaders
│   ├── evaluation/               # Metric computation (ROC, EER, Rank-N)
│   ├── features/                 # Log-Gabor feature extractor
│   ├── matching/                 # Hamming distance matcher with shift compensation
│   ├── normalization/            # Daugman's rubber-sheet normalizer
│   ├── preprocessing/            # Image enhancement and bilateral filtering
│   ├── segmentation/             # IDO and GAC segmenters
│   └── pipeline.py               # Batch processing pipeline runner
├── dataset_index.csv             # Dataset metadata index
├── splits/                       # Train / Val / Test subject split splits
│   ├── train_subjects.txt
│   ├── val_subjects.txt
│   └── test_subjects.txt
├── visualize_normalization.py   # Single-image normalization visualization tool
├── batch_normalize_visual.py     # Multi-image/subject visualization runner
├── recognize.py                  # 1-to-N query recognition CLI
├── evaluate.py                   # System performance evaluator
└── verify_fixes.py               # Verification & debugging suite
```

---

## 🚀 Getting Started

### Prerequisites

Install required Python dependencies:

```bash
pip install numpy opencv-python pyyaml scikit-image tqdm matplotlib scipy
```

---

## 💻 Usage Guide

### 1. Visualizing Iris Normalization

Generate visualization outputs (segmentation overlay, normalized iris strip, normalized mask strip) for a single image:

```bash
python visualize_normalization.py --image path/to/eye.jpg --outdir results/normalization
```

Run batch normalization on sample subjects ($5\text{ subjects} \times 2\text{ eyes}$):

```bash
python batch_normalize_visual.py
```

Outputs will be saved in structured folders:
`results/normalization_batch/Subject_<ID>/<L|R>/`

---

### 2. Processing Dataset Pipeline

To run batch segmentation, normalization, and feature extraction across the dataset:

```bash
python iris_recognition/pipeline.py --workers 4
```

---

### 3. Iris Recognition ($1:N$ Identification)

To match a query eye image against enrolled gallery templates:

```bash
python recognize.py --image path/to/query_eye.jpg --gallery results/templates
```

---

### 4. Evaluating System Performance (Classical Daugman Baseline)

Compute verification metrics (FAR, FRR, EER, AUC) and identification rank metrics (Rank-1, Rank-5):

```bash
python evaluate.py
```

---

### 5. Deep Learning Iris Recognition & Extrapolation Suite

Train and evaluate the deep neural network (`IrisDeepNet`) on normalized iris strips, verify normalization accuracy, evaluate recognition on a fast representative subset, and mathematically project accuracy for the full 20,000-image dataset:

```bash
# Run normalization verification, subset training & testing, and whole-dataset extrapolation:
python evaluate_deep_iris.py --epochs 10 --num_train_subjects 30 --num_test_subjects 15
```

Outputs:
- Normalization fidelity and invariance verification metrics.
- 1:1 Verification (EER, AUC, Decidability $d'$) and 1:N Identification (Rank-1, Rank-5, Rank-10) on unseen test subset.
- Extreme Value Theory (EVT) extrapolation for the full 2,000-identity dataset.
- High-resolution 6-panel diagnostic dashboard saved to `results/deep_learning/evaluation_results.png`.

---

## ⚙️ Configuration

System parameters can be adjusted in [`iris_recognition/configs/config.yaml`](iris_recognition/configs/config.yaml):
* **Normalization resolution**: `width: 512`, `height: 64`
* **Log-Gabor scales**: `num_scales: 4`, `min_wavelength: 18`
* **Matching shift**: `rotation_shift: 10`

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
