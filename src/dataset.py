"""
Dataset module for Speech Emotion Recognition
Handles RAVDESS and other emotion speech datasets
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path
import hashlib

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EMOTIONS,
    EMOTION_LABELS,
    EMOTION_TO_IDX,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.features import load_audio, extract_mel_spectrogram, add_noise, pitch_shift


class RAVDESSDataset(Dataset):
    """
    PyTorch Dataset for RAVDESS emotional speech.
    
    RAVDESS file naming convention:
    Modality-Vocal channel-Emotion-Intensity-Statement-Repetition-Actor.wav
    Example: 03-01-01-01-01-01-01.wav
    
    Emotion codes:
    01 = neutral, 02 = calm, 03 = happy, 04 = sad
    05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
    """
    
    def __init__(self, data_dir: str, split: str = "train", 
                 augment: bool = False, transform=None):
        """
        Initialize RAVDESS dataset.
        
        Args:
            data_dir: Path to RAVDESS data directory
            split: Dataset split ("train", "val", "test")
            augment: Whether to apply data augmentation
            transform: Optional transforms to apply
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment and split == "train"
        self.transform = transform
        
        # Collect audio files and labels
        all_samples = self._collect_samples()
        self.samples = self._apply_split(all_samples)
        
    def _collect_samples(self) -> List[Tuple[str, int]]:
        """Collect all audio files and their emotion labels."""
        samples = []
        
        if not self.data_dir.exists():
            print(f"Warning: Data directory {self.data_dir} does not exist.")
            return samples
            
        # RAVDESS structure: data_dir/Actor_XX/*.wav
        for actor_dir in self.data_dir.iterdir():
            if not actor_dir.is_dir():
                continue
                
            for audio_file in actor_dir.glob("*.wav"):
                # Parse filename for emotion
                parts = audio_file.stem.split("-")
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    emotion_name = EMOTIONS.get(emotion_code, None)
                    
                    if emotion_name:
                        # Map calm to neutral, skip disgust
                        if emotion_name == "calm":
                            emotion_name = "neutral"
                        elif emotion_name == "disgust":
                            continue  # Skip disgust - not in our labels
                        
                        if emotion_name in EMOTION_LABELS:
                            label = EMOTION_TO_IDX[emotion_name]
                            samples.append((str(audio_file), label))
        
        return samples

    def _apply_split(self, samples: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """Apply deterministic split to avoid train/val/test leakage."""
        if self.split not in {"train", "val", "test"}:
            return samples

        actor_ids = sorted({Path(file_path).parent.name for file_path, _ in samples})
        if len(actor_ids) >= 3:
            train_end = max(1, int(len(actor_ids) * TRAIN_RATIO))
            val_count = max(1, int(len(actor_ids) * VAL_RATIO))
            val_end = min(len(actor_ids) - 1, train_end + val_count)

            train_actors = set(actor_ids[:train_end])
            val_actors = set(actor_ids[train_end:val_end])
            test_actors = set(actor_ids[val_end:])

            if not test_actors:
                last_actor = actor_ids[-1]
                test_actors = {last_actor}
                train_actors.discard(last_actor)

            if self.split == "train":
                keep_actors = train_actors
            elif self.split == "val":
                keep_actors = val_actors
            else:
                keep_actors = test_actors

            split_samples = [
                sample
                for sample in samples
                if Path(sample[0]).parent.name in keep_actors
            ]

            if split_samples:
                # Store for class weight computation
                self.indices = list(range(len(split_samples)))
                self.labels = [label for _, label in split_samples]
                return split_samples

        # Fallback split when actor folders are insufficient.
        split_samples = []
        for file_path, label in samples:
            bucket = int(hashlib.md5(file_path.encode("utf-8")).hexdigest(), 16) % 10
            if self.split == "train" and bucket < 8:
                split_samples.append((file_path, label))
            elif self.split == "val" and bucket == 8:
                split_samples.append((file_path, label))
            elif self.split == "test" and bucket == 9:
                split_samples.append((file_path, label))
        
        # Store for class weight computation
        self.indices = list(range(len(split_samples)))
        self.labels = [label for _, label in split_samples]
        return split_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (mel_spectrogram tensor, emotion label)
        """
        file_path, label = self.samples[idx]
        
        # Load and extract features
        signal = load_audio(file_path)
        
        # Apply augmentation with higher probability
        if self.augment:
            # Add noise more frequently
            if np.random.random() < 0.6:
                signal = add_noise(signal, noise_factor=np.random.uniform(0.002, 0.01))
            # Pitch shift occasionally
            if np.random.random() < 0.4:
                signal = pitch_shift(signal, n_steps=np.random.uniform(-2, 2))
            # Time stretch occasionally
            if np.random.random() < 0.3:
                import librosa
                rate = np.random.uniform(0.9, 1.1)
                signal = librosa.effects.time_stretch(signal, rate=rate)
        
        # Extract mel-spectrogram
        mel_spec = extract_mel_spectrogram(signal)
        
        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        # Convert to tensor
        mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)  # (1, n_mels, time)
        
        if self.transform:
            mel_tensor = self.transform(mel_tensor)
        
        return mel_tensor, label


class EmotionDataset(Dataset):
    """
    Generic emotion dataset that works with pre-extracted features.
    """
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Initialize with pre-extracted features.
        
        Args:
            features: Array of shape (n_samples, ...)
            labels: Array of shape (n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.features[idx], self.labels[idx]


def create_dataloaders(data_dir: str, batch_size: int = 64, 
                       num_workers: int = 4, augment: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        augment: Whether to use augmentation for training data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = RAVDESSDataset(data_dir, split="train", augment=augment)
    val_dataset = RAVDESSDataset(data_dir, split="val", augment=False)
    test_dataset = RAVDESSDataset(data_dir, split="test", augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                             shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Dataset Module for Speech Emotion Recognition")
    print("=" * 50)
    print(f"Emotion Labels: {EMOTION_LABELS}")
    print(f"Number of classes: {len(EMOTION_LABELS)}")
