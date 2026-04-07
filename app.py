import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os

# ---- Page Config ----
st.set_page_config(
    page_title="Breast Cancer IDC Classifier",
    page_icon="🔬",
    layout="centered"
)

# ---- Constants ----
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = {0: "Non-IDC (Benign)", 1: "IDC (Malignant)"}
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Final-Project", "breast_cancer_resnet18.pth")


@st.cache_resource
def load_model():
    """Load the trained ResNet-18 model (cached so it only loads once)."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    return model, device


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Apply the same transforms used during training/validation."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


def predict(model, device, image: Image.Image):
    """Run inference and return class label + confidence."""
    tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    predicted_class = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class].item() * 100

    return predicted_class, confidence, probabilities


# ---- UI ----
st.title("Breast Histopathology – IDC Classifier")
st.markdown(
    "Upload a breast tissue microscopy image to classify it as "
    "**Non-IDC (Benign)** or **IDC (Malignant)**."
)
st.markdown(
    "This app uses a **ResNet-18** model fine-tuned on the "
    "[Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) "
    "dataset with **90.41% test accuracy**."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload a histopathology image (PNG or JPG)",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    model, device = load_model()
    predicted_class, confidence, probs = predict(model, device, image)

    label = CLASS_NAMES[predicted_class]
    is_malignant = predicted_class == 1

    with col2:
        st.markdown("### Prediction")

        if is_malignant:
            st.error(f"**{label}**")
        else:
            st.success(f"**{label}**")

        st.metric("Confidence", f"{confidence:.1f}%")

        st.markdown("### Class Probabilities")
        st.progress(probs[0].item(), text=f"Benign: {probs[0].item()*100:.1f}%")
        st.progress(probs[1].item(), text=f"Malignant: {probs[1].item()*100:.1f}%")

    st.divider()
    st.caption(
        "Model: ResNet-18 (pretrained on ImageNet, fine-tuned on 277k histopathology images) | "
        "Device: " + str(device)
    )
