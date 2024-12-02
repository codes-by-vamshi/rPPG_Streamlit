import cv2
import os
import tempfile
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent  # rPPG_Project
sys.path.append(str(project_root))
import preprocessing

def split_video(input_file, output_prefix, fps=30, chunk_frames=300):
    temp_dir = os.path.join(tempfile.gettempdir(), os.path.basename(input_file).split('.')[0])
    preprocessed_main_data = preprocessing.main(temp_dir, cache_path="NOT_USED_HERE", save_data=False)
    preprocessed_data = []
    for i in range(5):
        start_frame = 150 + i * 50
        end_frame = start_frame + chunk_frames
        preprocessed_data.append(preprocessed_main_data[0][start_frame:end_frame,:,:,:])
    return preprocessed_data