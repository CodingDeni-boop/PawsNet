import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
from tqdm import tqdm
import pandas as pd

class VideoSequenceDataset(Dataset):
    """
    Lazy-loading dataset for video sequences (loads frames on-demand to save memory).
    Returns per-frame labels for behavior classification.
    Augmentation applied during training.
    """
    def __init__(self, video_folder, label_folder, video_ids, sequence_length=30,
                 stride=10, img_size=(76, 142), behavior_names=["background", "supportedrear", "unsupportedrear", "grooming", "digging"], 
                 augment=False):
        """
        Args:
            video_folder: Path to folder containing .mp4 files
            label_folder: Path to folder containing label CSV files
            video_ids: List of video IDs (filenames without extension)
            sequence_length: Number of frames per sequence
            stride: Step size between sequences
            img_size: (width, height) - original video dimensions
            behavior_names: Ordered list of behavior class names
            augment: If True, apply random augmentations (only for training)
        """
        self.video_folder = video_folder
        self.label_folder = label_folder
        self.sequence_length = sequence_length
        self.stride = stride
        self.img_size = img_size
        self.augment = augment

        self.sequence_info = []
        self.labels = []
        self.label_cache = {}
        self.behavior_names = behavior_names

        print(f"Indexing sequences from {len(video_ids)} videos...")
        self._index_sequences(video_ids)

    def _index_sequences(self, video_ids):
        """Create index of sequences without loading video frames"""
        for video_id in tqdm(video_ids, desc="Indexing videos"):
            video_path = os.path.join(self.video_folder, f"{video_id}.mp4")
            label_path = os.path.join(self.label_folder, f"{video_id}.csv")

            if not os.path.exists(video_path):
                print(f"Warning: Video not found: {video_path}")
                continue
            if not os.path.exists(label_path):
                print(f"Warning: Labels not found: {label_path}")
                continue

            try:
                labels_df = pd.read_csv(label_path)
                video_labels = labels_df[self.behavior_names].values.argmax(axis=1)
                self.label_cache[video_id] = video_labels
            except Exception as e:
                print(f"Error loading labels for {video_id}: {e}")
                continue
            

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Cannot open video: {video_path}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            for start_idx in range(0, min(total_frames, len(video_labels)) - self.sequence_length + 1, self.stride):
                first_frame_label = video_labels[start_idx]
                self.sequence_info.append((video_id, start_idx))
                self.labels.append(int(first_frame_label))

        print(f"Indexed {len(self.sequence_info)} sequences (per-frame labels)")

    def __len__(self):
        return len(self.sequence_info)

    def __getitem__(self, idx):
        """Load video frames and per-frame labels on-demand"""
        video_id, start_frame = self.sequence_info[idx]
        video_path = os.path.join(self.video_folder, f"{video_id}.mp4")

        video_labels = self.label_cache[video_id]
        sequence_labels = video_labels[start_frame:start_frame + self.sequence_length]

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for _ in range(self.sequence_length):
            ret, frame = cap.read()
            if not ret:
                frames.append(np.zeros(self.img_size[::-1], dtype=np.uint8))
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, self.img_size)  # img_size is (width, height)
                frames.append(gray)

        cap.release()

        frames = np.array(frames, dtype=np.float32)

        # Data augmentation (training only).
        # 25% of sequences pass through completely unaugmented.
        # Each remaining augmentation fires independently at 50%.
        if self.augment and np.random.random() > 0.25:
            # Horizontal flip
            if np.random.random() > 0.5:
                frames = frames[:, :, ::-1].copy()

            # Per-frame brightness/contrast jitter
            if np.random.random() > 0.5:
                contrast   = np.random.uniform(0.8, 1.2, size=(len(frames), 1, 1)).astype(np.float32)
                brightness = np.random.uniform(-20.0, 20.0, size=(len(frames), 1, 1)).astype(np.float32)
                frames = np.clip(frames * contrast + brightness, 0, 255)

            # Additive Gaussian noise
            if np.random.random() > 0.5:
                frames = np.clip(frames + np.random.normal(0, 8, frames.shape).astype(np.float32), 0, 255)

        # Normalize to [-0.5, 0.5] (inverted, matching VideoDataSet/TCNN convention)
        frames = -(frames / 255.0 - 0.5)

        # Add channel dimension: (seq_len, H, W) -> (seq_len, 1, H, W)
        frames = frames[:, np.newaxis, :, :]

        return torch.FloatTensor(frames), torch.LongTensor(sequence_labels)


class VideoSequenceDatasetNoLabels(Dataset):
    """Sequences from videos with no ground-truth labels."""

    def __init__(self, video_folder, video_ids, sequence_length=30,
                 stride=5, img_size=(76, 142)):
        self.video_folder = video_folder
        self.sequence_length = sequence_length
        self.stride = stride
        self.img_size = img_size
        self.sequence_info = []   # list of (video_id, ext, start_frame)

        print(f"Indexing sequences from {len(video_ids)} videos...")
        for video_id in tqdm(video_ids, desc="Indexing videos"):
            video_path = os.path.join(video_folder, f"{video_id}.mp4")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Cannot open video: {video_path}")
                continue
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            for start_idx in range(0, total_frames - sequence_length + 1, stride):
                self.sequence_info.append((video_id, start_idx))

        print(f"Indexed {len(self.sequence_info)} sequences")

    def __len__(self):
        return len(self.sequence_info)

    def __getitem__(self, idx):
        video_id, start_frame = self.sequence_info[idx]
        video_path = os.path.join(self.video_folder, f"{video_id}.mp4")

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(self.sequence_length):
            ret, frame = cap.read()
            if not ret:
                frames.append(np.zeros(self.img_size[::-1], dtype=np.uint8))
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, self.img_size)
                frames.append(gray)
        cap.release()

        frames = np.array(frames, dtype=np.float32)
        frames = -(frames / 255.0 - 0.5)                                                    #normalization !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        frames = frames[:, np.newaxis, :, :]
        return torch.FloatTensor(frames)
