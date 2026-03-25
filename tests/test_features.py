"""
Unit tests for feature extraction module
"""

import pytest
import numpy as np
import sys
import os
import tempfile
import soundfile as sf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SAMPLE_RATE, DURATION, N_MFCC, N_MELS


class TestFeatureExtraction:
    """Test suite for feature extraction."""
    
    def test_load_audio_shape(self):
        """Test that loaded audio has correct shape."""
        from src.features import load_audio

        # Create a temporary stereo audio file with a different sample rate
        raw_sr = 8000
        duration = 1.0
        t = np.linspace(0, duration, int(raw_sr * duration), endpoint=False)
        mono = 0.2 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
        stereo = np.stack([mono, mono], axis=1)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name

        try:
            sf.write(temp_path, stereo, raw_sr)
            loaded = load_audio(temp_path)

            assert loaded.ndim == 1
            assert len(loaded) == SAMPLE_RATE * DURATION
            assert loaded.dtype == np.float32
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_mfcc_extraction(self):
        """Test MFCC feature extraction."""
        from src.features import extract_mfcc
        
        # Create dummy signal
        signal = np.random.randn(SAMPLE_RATE * DURATION).astype(np.float32)
        
        mfcc = extract_mfcc(signal)
        
        # Check shape: (3 * n_mfcc, time_frames)
        assert mfcc.shape[0] == 3 * N_MFCC
        assert mfcc.shape[1] > 0
    
    def test_mel_spectrogram_extraction(self):
        """Test mel-spectrogram extraction."""
        from src.features import extract_mel_spectrogram
        
        # Create dummy signal
        signal = np.random.randn(SAMPLE_RATE * DURATION).astype(np.float32)
        
        mel_spec = extract_mel_spectrogram(signal)
        
        # Check shape: (n_mels, time_frames)
        assert mel_spec.shape[0] == N_MELS
        assert mel_spec.shape[1] > 0
    
    def test_augmentation_noise(self):
        """Test noise augmentation."""
        from src.features import add_noise
        
        signal = np.zeros(1000, dtype=np.float32)
        augmented = add_noise(signal, noise_factor=0.01)
        
        # Original is zeros, augmented should have non-zero values
        assert not np.allclose(augmented, signal)
    
    def test_augmentation_pitch_shift(self):
        """Test pitch shift augmentation."""
        from src.features import pitch_shift
        
        # Create a simple sine wave
        t = np.linspace(0, 1, SAMPLE_RATE)
        signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        shifted = pitch_shift(signal, n_steps=2)
        
        # Should have same length
        assert len(shifted) == len(signal)


class TestModel:
    """Test suite for model architecture."""
    
    def test_cnn_lstm_forward(self):
        """Test CNN-LSTM forward pass."""
        import torch
        from src.model import CNNLSTM
        
        model = CNNLSTM(num_classes=6)
        
        # Batch of 2, 1 channel, 128 mel bands, 128 time frames
        dummy_input = torch.randn(2, 1, 128, 128)
        
        output = model(dummy_input)
        
        assert output.shape == (2, 6)
    
    def test_simple_cnn_forward(self):
        """Test Simple CNN forward pass."""
        import torch
        from src.model import SimpleCNN
        
        model = SimpleCNN(num_classes=6)
        
        dummy_input = torch.randn(2, 1, 128, 128)
        
        output = model(dummy_input)
        
        assert output.shape == (2, 6)
    
    def test_model_factory(self):
        """Test model factory function."""
        from src.model import get_model
        
        model1 = get_model("cnn_lstm", num_classes=6)
        model2 = get_model("simple_cnn", num_classes=6)
        
        assert model1 is not None
        assert model2 is not None


class TestDataset:
    """Test suite for dataset module."""
    
    def test_emotion_labels(self):
        """Test emotion label configuration."""
        from config import EMOTION_LABELS, EMOTION_TO_IDX
        
        assert len(EMOTION_LABELS) == 6
        assert all(emotion in EMOTION_TO_IDX for emotion in EMOTION_LABELS)
    
    def test_ravdess_filename_parsing(self):
        """Test RAVDESS filename parsing logic."""
        from config import EMOTIONS
        
        # Test filename: 03-01-05-01-01-01-12.wav
        # Emotion code is 05 = angry
        filename = "03-01-05-01-01-01-12"
        parts = filename.split("-")
        emotion_code = parts[2]
        
        assert emotion_code == "05"
        assert EMOTIONS[emotion_code] == "angry"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
