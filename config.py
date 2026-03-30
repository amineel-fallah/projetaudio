"""
Configuration for Speech Emotion Recognition Project
"""

# Audio parameters
SAMPLE_RATE = 16000
DURATION = 4  # seconds
N_MFCC = 13
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

# Model parameters
BATCH_SIZE = 32
LEARNING_RATE = 5e-4  # Increased from 1e-4 for faster convergence
EPOCHS = 50  # Increased for better accuracy

# Emotion labels (RAVDESS encoding)
EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Simplified emotion mapping (6 core emotions)
EMOTION_LABELS = ["neutral", "happy", "sad", "angry", "fearful", "surprised"]
EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTION_LABELS)}
NUM_CLASSES = len(EMOTION_LABELS)

# Paths
DATA_DIR = "data"
RAVDESS_DIR = "data/ravdess"
MODEL_DIR = "models"
LOGS_DIR = "logs"

# Train/Val/Test split
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
