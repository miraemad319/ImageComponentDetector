import streamlit as st
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch

process = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')

def main():
    st.title("Image Component Detector")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button("Analyse Image"):
            analyze_image(image)

def analyze_image(image):
    inputs = process(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values']

    outputs = model(pixel_values = pixel_values)

    target_sizes = torch.tensor([image.size[::-1]])
    results_list = process.post_process_object_detection(outputs, target_sizes=target_sizes)

    threshold = 0.9

    for result in results_list:
        labels = result["labels"]
        scores = result["scores"]

        for label, score in zip(labels, scores):
            if score.item() > threshold:
                detected_component = model.config.id2label[label.item()]
                confidence_score = score.item()
                st.write(f"Detected Component: {detected_component} (Score: {confidence_score})")

if __name__ == "__main__":
    main()


