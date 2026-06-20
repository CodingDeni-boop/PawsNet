import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
import os
import math
import time
from collections import defaultdict
from Model import CNNTransformerClassifier
from VideoRotator import VideoRotator
from Dataset import VideoSequenceDatasetNoLabels
from model_methods import predict_per_video
from post_processing import apply_gap_fill, apply_min_duration_filter, apply_supported_rearing_merge

if __name__ == "__main__":

    ROTATE = False
    MODEL_PATH = "model_saves/The_PawsNet.pth" # I called the model which I trained on all videos (62 train, 11 validation, 0 test) The_PawsNet.pth
    RAW_VIDEOS_DIR = "./to_predict/raw_videos"
    TRACKING_DIR = "./to_predict/tracking"
    ROTATED_VIDEOS_DIR = "./to_predict/rotated_videos"
    PREDICTIONS_DIR      = "./predictions_no_true_labels"
    SEQUENCE_LENGTH = 30
    EVAL_STRIDE     = 5     # For fast debugging put this to 25
    IMG_SIZE        = (76, 142)
    BATCH_SIZE      = 32
    DROPOUT         = 0.3
    COLUMN_NAMES = {0 : "background", 
                    1 : "supportedrear", 
                    2 : "unsupportedrear", 
                    3 : "grooming", 
                    4 : "digging"}
    VIDEO_EXTENSIONS = [".mp4", ".avi"]

    behaviors = list(COLUMN_NAMES.values())

    if ROTATE:
        for video_name in os.listdir(RAW_VIDEOS_DIR):
            if not (video_name.endswith(".mp4") or video_name.endswith(".avi")):
                print(f"{video_name} is not a valid video file and wasn't rotated")
                continue            ## If not .mp4 or .avi file, it is not rotated
            
            name, extension = os.path.splitext(video_name)

            video_path = os.path.join(RAW_VIDEOS_DIR, video_name)
            
            tracking_name = name + ".csv"
            tracking_path = os.path.join(TRACKING_DIR, tracking_name)

            output_name = name + ".mp4"
            output_path = os.path.join(ROTATED_VIDEOS_DIR, output_name)

            rotator = VideoRotator(video_path = video_path, output_path = output_path, out_width = 76, out_height = 142)
            rotator.follow(tracking_path, "mouse_top.mouse_top_0.bodycentre","mouse_top.mouse_top_0.neck")

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = CNNTransformerClassifier(
        cnn_feature_dim=checkpoint['cnn_feature_dim'],
        d_model=checkpoint['d_model'],
        nhead=checkpoint['nhead'],
        num_layers=checkpoint['num_layers'],
        num_classes=checkpoint['num_classes'],
        dim_feedforward=checkpoint['dim_feedforward'],
        dropout=DROPOUT
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded.")

    # Auto-discover all video files in the folder
    video_ids = []
    for video_name in os.listdir(ROTATED_VIDEOS_DIR):
        if not video_name.endswith(".mp4"):
            print(f"{video_name} is not a valid video file and wasn't rotated")
            continue            ## If not .mp4 or .avi file, it is not rotated
        video_ids.append(video_name[:-4])

    print(f"Found {len(video_ids)} video(s): {[v for v in video_ids]}")

    # Build dataset & loader
    dataset = VideoSequenceDatasetNoLabels(
        ROTATED_VIDEOS_DIR, video_ids,
        SEQUENCE_LENGTH, EVAL_STRIDE, IMG_SIZE
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Predict
    per_video = predict_per_video(model, loader, device)

    # Save one CSV per video
    for video_id, (frame_indices, preds) in per_video.items():
        preds = apply_gap_fill(preds)
        preds = apply_min_duration_filter(preds)
        one_hot = np.zeros((len(preds), len(behaviors)), dtype=int)
        for i, cls_idx in enumerate(preds):
            one_hot[i, cls_idx] = 1

        df = pd.DataFrame(one_hot, columns=behaviors, index=frame_indices)
        df.index.name = "frame"

        out_path = os.path.join(PREDICTIONS_DIR, f"{video_id}.csv")
        df.to_csv(out_path)
        print(f"Saved: {out_path}  ({len(df)} frames)")

    print(f"\nDone. Predictions saved to: {PREDICTIONS_DIR}")

