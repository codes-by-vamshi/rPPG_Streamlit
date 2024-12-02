import streamlit as st
import cv2
import tempfile
import os
from chunk_video import split_video
import numpy as np

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent  # rPPG_Project
sys.path.append(str(project_root))
import testing

# Directory to save intermediate files
CHUNKS_DIR = "chunked_videos"
os.makedirs(CHUNKS_DIR, exist_ok=True)

preproc_data = []

def main():
    global preproc_data
    st.title("Heart Rate Estimation using rPPG")
    
    # Upload or record video
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    video_path = None

    if uploaded_file:
        # Save the uploaded video to a temporary location
        temp_dir = os.path.join(tempfile.gettempdir(), uploaded_file.name.split('.')[0])
        os.makedirs(temp_dir, exist_ok=True)
        temp_dir = os.path.join(temp_dir, uploaded_file.name.split('.')[0])
        os.makedirs(temp_dir, exist_ok=True)
        temp_video_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        video_path = temp_video_path
        st.success("Uploaded video saved for processing.")

    # Proceed if we have a video
    if video_path:
        # Step 1: Chunk the video
        st.write("Chunking the video into 10-second intervals...")
        preprocessed_chunk_data = split_video(video_path, CHUNKS_DIR, fps=30, chunk_frames=300)
        preproc_data = preprocessed_chunk_data
        st.success(f"Video chunked into {len(preproc_data)} parts.")

        # Step 2: Run inference directly
        st.write("Running inference on chunked videos...")
        hr = testing.testing(CHUNKS_DIR, preproc_data)
        st.success("Inference completed.")
        st.write(f"Predicted Heart Rate: {np.floor(hr)}")
    
if __name__ == "__main__":
    main()