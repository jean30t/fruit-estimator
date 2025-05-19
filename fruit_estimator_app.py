import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
from PIL import Image

model = YOLO("yolov8n.pt")
FRUIT_CLASSES = [46, 47, 49]  # banana, apple, orange

st.title("Fruit Box Weight Estimator")

box_length = st.number_input("Box Length (cm)", min_value=1.0)
box_width = st.number_input("Box Width (cm)", min_value=1.0)
box_height = st.number_input("Box Height (cm)", min_value=1.0)
avg_fruit_weight = st.number_input("Average Fruit Weight (kg)", min_value=0.01)
avg_fruit_diameter = st.number_input("Average Fruit Diameter (cm)", min_value=1.0)

uploaded_file = st.file_uploader("Upload Fruit Box Image", type=["jpg", "jpeg", "png"])

if uploaded_file and all([box_length, box_width, box_height, avg_fruit_weight, avg_fruit_diameter]):
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        image.save(temp_file.name)
        results = model(temp_file.name)
        detections = results[0]
        fruit_count = sum(1 for d in detections.boxes.cls if int(d) in FRUIT_CLASSES)

        layers = max(1, box_height // avg_fruit_diameter)
        estimated_total_fruits = fruit_count * layers
        total_weight = estimated_total_fruits * avg_fruit_weight

        st.success("Estimation Complete:")
        st.write(f"**Detected Fruits (top layer):** {fruit_count}")
        st.write(f"**Estimated Layers:** {int(layers)}")
        st.write(f"**Estimated Total Fruits:** {int(estimated_total_fruits)}")
        st.write(f"**Estimated Stock Weight:** {total_weight:.2f} kg")

        st.image(results[0].plot(), caption="Detected Fruits", use_column_width=True)
