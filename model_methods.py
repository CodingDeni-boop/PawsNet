import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
import os
import math
import time
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from post_processing import apply_gap_fill, apply_min_duration_filter, apply_supported_rearing_merge
from collections import defaultdict

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch with per-frame predictions.
    CHANGE: scheduler removed from here — now stepped per epoch in the main loop.
    """
    model.train()
    total_loss = 0

    for batch_X, batch_y in tqdm(dataloader, desc="Training"):
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)  # (batch, seq_len)

        optimizer.zero_grad()
        outputs = model(batch_X)  # (batch, seq_len, num_classes)

        # Reshape for loss computation
        batch_size, seq_len, num_classes = outputs.shape
        outputs_flat = outputs.view(batch_size * seq_len, num_classes)
        labels_flat  = batch_y.view(batch_size * seq_len)

        loss = criterion(outputs_flat, labels_flat)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, return_per_video=False, background_class=0):
    """
    Evaluate model with per-frame predictions.

    Arguments:
        use_consensus: If True, uses majority voting across overlapping sequences for each unique frame.
                       If False, treats all predictions independently.
    """
    model.eval()

    frame_predictions = defaultdict(list)
    frame_labels      = {}

    with torch.no_grad():
        # CHANGE: track global sequence index explicitly to avoid batch-size assumption
        seq_idx_offset = 0
        for batch_X, batch_y in tqdm(dataloader, desc="Evaluating"):
            batch_X   = batch_X.to(device)
            outputs   = model(batch_X)
            probs     = torch.softmax(outputs, dim=2)

            probs_np  = probs.cpu().numpy()
            labels_np = batch_y.numpy()

            actual_batch_size = batch_X.shape[0]
            for b in range(actual_batch_size):
                seq_idx = seq_idx_offset + b
                if seq_idx >= len(dataloader.dataset):
                    continue

                video_id, start_frame = dataloader.dataset.sequence_info[seq_idx]

                for frame_offset in range(dataloader.dataset.sequence_length):
                    frame_idx = start_frame + frame_offset
                    key       = (video_id, frame_idx)

                    frame_predictions[key].append(probs_np[b, frame_offset])
                    frame_labels[key] = labels_np[b, frame_offset]

            seq_idx_offset += actual_batch_size

    # Consensus voting
    consensus_preds  = []
    consensus_labels = []
    per_video_data   = defaultdict(lambda: {'preds': [], 'labels': []})

    for key in sorted(frame_predictions.keys()):
        video_id, frame_idx = key
        preds          = frame_predictions[key]
        consensus_pred = np.argmax(np.sum(preds, axis=0))

        consensus_preds.append(consensus_pred)
        consensus_labels.append(frame_labels[key])

        per_video_data[video_id]['preds'].append(consensus_pred)
        per_video_data[video_id]['labels'].append(frame_labels[key])

    # Apply postprocessing per video
    for video_id in per_video_data:
        filtered = apply_min_duration_filter(per_video_data[video_id]['preds'], background_class=background_class)
        filtered = apply_gap_fill(filtered, background_class=background_class)
        per_video_data[video_id]['preds'] = filtered.tolist()

    # Reconstruct flat arrays from filtered per-video videos
    video_frame_counters = defaultdict(int)
    consensus_preds = []
    for key in sorted(frame_predictions.keys()):
        video_id, _ = key
        i = video_frame_counters[video_id]
        consensus_preds.append(per_video_data[video_id]['preds'][i])
        video_frame_counters[video_id] += 1

    if return_per_video:
        return np.array(consensus_preds), np.array(consensus_labels), dict(per_video_data)
    return np.array(consensus_preds), np.array(consensus_labels)


def predict_per_video(model, dataloader, device):
    """
    Run inference and return per-video frame predictions using consensus voting.
    Returns dict: {video_id: (sorted_frame_indices, np.array of predicted class indices)}
    """
    model.eval()

    frame_predictions = defaultdict(list)

    with torch.no_grad():
        seq_idx_offset = 0
        for batch_X in tqdm(dataloader, desc="Predicting"):
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=2).cpu().numpy()

            actual_batch_size = batch_X.shape[0]
            for b in range(actual_batch_size):
                seq_idx = seq_idx_offset + b
                if seq_idx >= len(dataloader.dataset):
                    continue
                video_id, start_frame = dataloader.dataset.sequence_info[seq_idx]
                for frame_offset in range(dataloader.dataset.sequence_length):
                    key = (video_id, start_frame + frame_offset)
                    frame_predictions[key].append(probs[b, frame_offset])

            seq_idx_offset += actual_batch_size

    # Consensus vote per frame
    per_video_preds = defaultdict(dict)
    for (video_id, frame_idx), preds in frame_predictions.items():
        per_video_preds[video_id][frame_idx] = np.argmax(np.sum(preds, axis=0))

    result = {}
    for video_id, frame_dict in per_video_preds.items():
        sorted_frames = sorted(frame_dict.keys())
        preds = np.array([frame_dict[f] for f in sorted_frames])
        result[video_id] = (sorted_frames, preds)

    return result