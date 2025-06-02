import streamlit as st
import torch
import numpy as np
import os
import cv2
import gdown
from PIL import Image
from transformers import AutoProcessor, AutoModelForVideoClassification

# ---------------------
# Download model from Google Drive using gdown
# ---------------------
def download_model_if_needed(save_path):
    if not os.path.exists(save_path):
        st.info("Downloading model from Google Drive...")
        file_id = "1yegsjiRVRtXpLfaIpisNPSX6B931sbTG"  # Your shared Drive file ID
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, save_path, quiet=False)
        st.success("Model downloaded successfully!")

# ---------------------
# Load processor and model
# ---------------------
def load_model(model_path="timesformer_model.pth"):
    download_model_if_needed(model_path)

    processor = AutoProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    model = AutoModelForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")
    model.classifier = torch.nn.Linear(model.config.hidden_size, 25)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return processor, model

# ---------------------
# Extract 8 frames from video
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
# Streamlit App UI
# ---------------------
st.set_page_config(layout="wide", page_title="Action Recognition App")
st.title("🎬 Human Action Recognition")

st.write("""
Upload a short video clip, and this app will classify the action using a fine-tuned **TimeSformer** model.
The model is automatically downloaded from Google Drive if not already present.
""")

st.sidebar.title("📁 Upload Your Video")
uploaded_file = st.sidebar.file_uploader("Supported formats: MP4, AVI, MOV", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with st.spinner("Loading model..."):
        processor, model = load_model()

    # Save uploaded video to temp file
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    # Extract frames
    frames = extract_frames(video_path)

    if frames:
        inputs = processor(images=frames, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_class_idx = torch.argmax(logits, dim=-1).item()
            confidence = torch.softmax(logits, dim=-1)[0, predicted_class_idx].item()

        predicted_activity = class_to_activity.get(predicted_class_idx, "Unknown Activity")

        # Display results
        col1, col2 = st.columns([2, 1])
        with col1:
            st.video(video_path)
        with col2:
            st.write("### 🏷️ Predicted Action")
            st.markdown(f"**{predicted_activity}**")
            st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
    else:
        st.error("Unable to extract enough frames from the video.")

