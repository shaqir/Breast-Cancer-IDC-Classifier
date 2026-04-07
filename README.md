# Breast Cancer IDC Classifier

A deep learning project that classifies breast histopathology images as **Non-IDC (Benign)** or **IDC (Malignant)** using a fine-tuned **ResNet-18** model. Includes a **Streamlit** web app for real-time inference.

- **Dataset:** [Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) (277,524 patches from 162 patients)
- **Model:** ResNet-18 pretrained on ImageNet, fine-tuned for binary classification
- **Test Accuracy:** 90.41%

---

## Project Structure

```
├── CV_final.ipynb              # Full training & evaluation notebook
├── app.py                      # Streamlit inference app
├── CV_Notebook_Breakdown.md    # Documentation/breakdown of the notebook
├── requirements.txt            # Python dependencies
├── Final-Project/
│   └── breast_cancer_resnet18.pth  # Trained model weights (45 MB)
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/shaqir/Breast-Cancer-IDC-Classifier.git
cd Breast-Cancer-IDC-Classifier
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App (no dataset needed)

```bash
streamlit run app.py
```

- The trained model is already included in the repo
- Upload any breast histopathology image to get a prediction
- Works out of the box after cloning

### 4. Download the Dataset (for retraining only)

If you want to retrain the model, you need the dataset from Kaggle (~3.1 GB):

```bash
pip install kagglehub
```

Set up Kaggle credentials:

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) and click **Create New Token** (downloads `kaggle.json`)
2. Place it in the right location:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

The notebook will automatically download the dataset using:

```python
import kagglehub
path = kagglehub.dataset_download("paultimothymooney/breast-histopathology-images")
```

### 5. Run the Training Notebook (optional)

```bash
jupyter notebook CV_final.ipynb
```

- Requires the dataset from Step 4
- Run all cells in order
- Step 1 in the notebook reorganizes the raw data into `class_0` / `class_1` folders

---

## Quick Reference

| What you want to do | Steps needed |
|---|---|
| **Just run the app** | Steps 1, 2, 3 |
| **Retrain the model** | Steps 1, 2, 4, 5 |
