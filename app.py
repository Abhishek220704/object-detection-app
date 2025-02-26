import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ✅ Load the YOLO model
@st.cache_resource()
def load_model():
    model = YOLO("model.pt")  # Ensure model.pt is in the same directory
    return model

model = load_model()

# ✅ Streamlit UI
st.title("🚀 Object Detection with YOLOv5")
st.write("Upload an image and the model will detect objects!")

# ✅ File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ✅ Run YOLO on image
    results = model(image)

    # ✅ Display detected objects
    st.write("### Detection Results:")
    for result in results:
        for box in result.boxes:
            st.write(f"Detected: **{model.names[int(box.cls)]}** (Confidence: {box.conf[0]:.2f})")

    # ✅ Show image with detections
    results[0].show()
    st.image(results[0].plot(), caption="Detected Objects", use_column_width=True)
