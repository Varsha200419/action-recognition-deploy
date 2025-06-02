import streamlit as st
import torch
import numpy as np
import os
import cv2
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForVideoClassification

# ---------------------
# Download model if not already present
# ---------------------
def download_model_if_needed(url, save_path):
    if not os.path.exists(save_path):
        st.info("Downloading fine-tuned model from Google Drive...")
        response = requests.get(url)
        with open(save_path, 'wb') as f:
            f.write(response.content)
        st.success("Model download complete!")

# ---------------------
# Load model and processor
# ---------------------
def load_model(model_path="timesformer_model.pth"):
    url = "https://drive.google.com/uc?export=download&id=1yegsjiRVRtXpLfaIpisNPSX6B931sbTG"
    download_model_if_needed(url, model_path)

    processor = AutoProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    model = AutoModelForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")
    model.classifier = torch.nn.Linear(model.config.hidden_size, 25)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return processor, model

# ---------------------
# Extract video frames
# ---------------------
def extract_frames(video_path, num_frames=8, frame_rate=32):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampled_indices = list(range(0, total_frames, frame_rate))[:num_frames]
    frames = []
    for idx in sampled_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    cap.release()
    return frames

# ---------------------
# Class index to label mapping
# ---------------------
class_to_activity = {
    0: "Brush Hair", 1: "Cartwheel", 2: "Catch", 3: "Chew", 4: "Climb", 5: "Climb Stairs", 6: "Draw Sword",
    7: "Eat", 8: "Fencing", 9: "Flic Flac", 10: "Golf", 11: "Handstand", 12: "Kiss", 13: "Pick", 14: "Pour",
    15: "Pullup", 16: "Pushup", 17: "Ride Bike", 18: "Shoot Bow", 19: "Shoot Gun", 20: "Situp",
    21: "Smile", 22: "Smoke", 23: "Throw", 24: "Wave"
}

# ---------------------
# Streamlit Interface
# ---------------------
st.set_page_config(layout="wide", page_title="Action Recognition")
st.title("🎬 Human Action Recognition App")

st.write("""
Upload a video clip and this app will classify the action using a fine-tuned **TimeSformer** model.
The model will be automatically downloaded from Google Drive if not available locally.
""")

st.sidebar.title("Upload Your Video 🎥")
uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# Run prediction
if uploaded_file is not None:
    with st.spinner("Loading model..."):
        processor, model = load_model()

    # Save uploaded file temporarily
    temp_video_path = "temp_uploaded_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    frames = extract_frames(temp_video_path)

    if frames:
        inputs = processor(images=frames, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_class_idx = torch.argmax(logits, dim=-1).item()
            confidence = torch.softmax(logits, dim=-1)[0, predicted_class_idx].item()

        predicted_activity = class_to_activity.get(predicted_class_idx, "Unknown Activity")

        # Display
        col1, col2 = st.columns([2, 1])
        with col1:
            st.video(temp_video_path)
        with col2:
            st.markdown("### 🏷️ Predicted Action")
            st.markdown(f"**{predicted_activity}**")
            st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
    else:
        st.error("Couldn't extract enough frames from the video.")
