# Breast Histopathology Image Classification using CNN

## Step-by-Step Breakdown from a Computer Vision Perspective

**Goal:** Binary classification of breast tissue microscopy images as **Benign (Non-IDC)** or **Malignant (IDC — Invasive Ductal Carcinoma)** using deep learning.

---

### Step 0 — Environment Setup
- Detects the best compute device (Apple MPS GPU / CUDA / CPU)
- Points to the Kaggle dataset cached locally (~277,524 patches of 50x50 px images from 162 patients)

### Step 1 — Data Preparation
- **CV concept: Dataset organization for supervised learning**
- The raw dataset is nested per-patient (`patient_id/0/` and `patient_id/1/`). This step **flattens** all images into two top-level folders (`class_0`, `class_1`) so PyTorch's `ImageFolder` can load them with automatic labeling.

### Step 2 — Library Imports
- Loads PyTorch, Torchvision, NumPy, Matplotlib, and scikit-learn.
- Sets random seeds for **reproducibility**.

### Step 3 — Data Augmentation & Normalization
- **CV concept: Preprocessing pipeline**
  - **(a) Resize** — Upscales 50x50 patches to **224x224** (required input size for ResNet-18)
  - **(b) Random Horizontal Flip** — A **data augmentation** technique (50% chance) that artificially increases training diversity, helping the model generalize and reducing overfitting
  - **(c) Normalize** — Scales pixel values using **ImageNet mean/std** `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`, which is required when using pretrained ImageNet models
- Splits the dataset: **70% train / 15% validation / 15% test** using random indices

### Step 4 — DataLoaders
- Creates batched data pipelines with **batch size = 32** and **10 epochs**
- Training loader shuffles data; validation/test loaders do not

### Step 4b — Visualize Samples
- **CV concept: Sanity checking**
- Displays sample images with labels after reversing normalization (tensor to displayable image) to verify the pipeline is correct

### Step 5 — Model: Pretrained ResNet-18 (Transfer Learning)
- **CV concept: Transfer learning**
  - Loads **ResNet-18** pretrained on ImageNet (1.2M images, 1000 classes) — it already knows low-level features (edges, textures) and mid-level features (shapes, patterns)
  - **Replaces the final FC layer**: `Linear(512 -> 1000)` becomes `Linear(512 -> 2)` for binary classification
  - Fine-tunes the entire network on the breast histopathology data

### Step 5b — Training Loop (10 Epochs)
- **CV concept: Supervised training with backpropagation**
  - **Training phase**: Forward pass -> CrossEntropyLoss -> Backpropagation -> Adam optimizer weight update
  - **Validation phase**: Evaluates on held-out data with `torch.no_grad()` (no gradient computation) to monitor overfitting
  - Tracks loss and accuracy per epoch for both train and validation sets

### Step 5c — Training Curves
- Plots **loss curves** and **accuracy curves** (train vs. validation) to visually detect overfitting (diverging curves) or underfitting

### Step 6 — Inference on Unseen Test Images
- **CV concept: Model inference / visual evaluation**
- Runs the trained model on test images it has never seen
- Displays predictions directly on images with **green** (correct) / **red** (incorrect) labels

### Step 7 — Quantitative Evaluation
- **CV concept: Classification metrics**
  - **Accuracy**: 90.41% on 41,630 test images
  - **Precision**: Of predicted malignant cases, 79% were truly malignant
  - **Recall**: Of actual malignant cases, 90% were detected (critical for medical diagnosis — minimizing missed cancers)
  - **F1-Score**: Harmonic mean of precision and recall
  - **Confusion Matrix**: Visual breakdown of true positives, true negatives, false positives, false negatives

### Step 8 — Save Model
- Saves trained weights to `breast_cancer_resnet18.pth` for later inference or deployment (e.g., a Streamlit web app)

---

## Summary of CV Concepts Used

| Concept | Where Applied |
|---|---|
| Image classification (binary) | Entire pipeline |
| Transfer learning | ResNet-18 pretrained on ImageNet |
| Data augmentation | Random horizontal flip |
| Image normalization | ImageNet mean/std scaling |
| CNNs (Convolutional Neural Networks) | ResNet-18 architecture |
| Train/Val/Test split | 70/15/15 split |
| Overfitting detection | Training vs. validation curves |
| Confusion matrix & metrics | Precision, Recall, F1, Accuracy |

---

## Final Result

A fine-tuned ResNet-18 that classifies breast tissue as benign or malignant with **90.41% accuracy**, with particularly strong recall (90%) for detecting cancer — the more critical class in a medical context.
