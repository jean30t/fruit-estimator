import streamlit as st
from ultralytics import YOLO
import os
import urllib.request
from PIL import Image
import tempfile

st.set_page_config(page_title="Fruit Estimator", layout="centered")
st.title("Fruit Estimator using YOLOv8")

# Automatically download YOLOv8n model if not already present
model_path = "yolov8n.pt"
if not os.path.exists(model_path):
    st.info("Downloading YOLOv8n model...")
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    urllib.request.urlretrieve(url, model_path)
    st.success("Model downloaded!")

# Load YOLO model
model = YOLO(model_path)

# Upload image
uploaded_file = st.file_uploader("Upload an image of fruit", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    # Run prediction
    st.info("Detecting fruits...")
    results = model(image_path)

    # Display results
    res_plotted = results[0].plot()  # Returns a numpy array with bounding boxes
    st.image(res_plotted, caption="Detected Fruits", use_column_width=True)

    # Optionally list classes detected
    boxes = results[0].boxes
    class_names = results[0].names
    detected = [class_names[int(cls)] for cls in boxes.cls]

    if detected:
        st.success(f"Detected objects: {', '.join(detected)}")
    else:
        st.warning("No fruits detected.")
