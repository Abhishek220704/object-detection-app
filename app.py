import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image

# Load the ONNX model
MODEL_PATH = "best.onnx"  # Ensure this file is in your GitHub repo
session = ort.InferenceSession(MODEL_PATH)

st.title("🚀 YOLOv5 Object Detection App")
st.write("Upload an image and detect objects using your trained YOLOv5 model!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to an OpenCV image
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_resized = cv2.resize(img_np, (640, 640))  # Resize to model input

    # Prepare input tensor
    img_input = img_resized.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))  # Change from HWC to CHW
    img_input = np.expand_dims(img_input, axis=0)  # Add batch dimension

    # Run inference
    outputs = session.run(None, {"images": img_input})

    # Display image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("✅ Objects detected!")

    # Process results (you may need to modify this based on your model output format)
    st.write(outputs)  # Debugging output

st.write("💡 Model deployed using Streamlit & ONNX.")
