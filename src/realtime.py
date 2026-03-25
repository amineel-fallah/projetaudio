"""
Real-time inference module for Speech Emotion Recognition
Handles microphone input and live prediction
"""

import numpy as np
import torch
import sounddevice as sd
import queue
import threading
from typing import Callable, Optional
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SAMPLE_RATE, EMOTION_LABELS
from src.model import get_model
from src.features import extract_mel_spectrogram
from src.utils import get_device


class RealTimeEmotionRecognizer:
    """
    Real-time emotion recognition from microphone input.
    
    Uses a sliding window approach to continuously analyze audio
    and predict emotions.
    """
    
    def __init__(self, model_path: str = None,
                 sample_rate: int = SAMPLE_RATE,
                 window_duration: float = 3.0,
                 hop_duration: float = 0.5,
                 callback: Optional[Callable] = None):
        """
        Initialize real-time recognizer.
        
        Args:
            model_path: Path to trained model checkpoint
            sample_rate: Audio sample rate
            window_duration: Analysis window duration in seconds
            hop_duration: Hop between consecutive windows
            callback: Function to call with predictions
        """
        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.hop_duration = hop_duration
        self.callback = callback
        
        # Buffer for audio samples
        self.window_samples = int(window_duration * sample_rate)
        self.hop_samples = int(hop_duration * sample_rate)
        self.audio_buffer = np.zeros(self.window_samples)
        
        # Audio queue
        self.audio_queue = queue.Queue()
        
        # Load model
        self.device = get_device()
        self.model = self._load_model(model_path)
        
        # State
        self.running = False
        self.current_emotion = "neutral"
        self.confidence = 0.0
        
    def _load_model(self, model_path: str = None) -> torch.nn.Module:
        """Load emotion recognition model."""
        model = get_model("cnn_lstm", num_classes=len(EMOTION_LABELS))
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        else:
            print("Using untrained model")
            
        model = model.to(self.device)
        model.eval()
        return model
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input stream."""
        if status:
            print(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())
    
    def _process_audio(self):
        """Process audio from queue."""
        samples_since_prediction = 0
        
        while self.running:
            try:
                # Get audio chunk from queue
                chunk = self.audio_queue.get(timeout=0.1)
                chunk = chunk.flatten()
                
                # Update buffer (sliding window)
                self.audio_buffer = np.roll(self.audio_buffer, -len(chunk))
                self.audio_buffer[-len(chunk):] = chunk
                
                samples_since_prediction += len(chunk)
                
                # Predict every hop_samples
                if samples_since_prediction >= self.hop_samples:
                    self._predict()
                    samples_since_prediction = 0
                    
            except queue.Empty:
                continue
    
    def _predict(self):
        """Run emotion prediction on current buffer."""
        # Extract features
        mel_spec = extract_mel_spectrogram(self.audio_buffer)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
        
        # Get top prediction
        pred_idx = np.argmax(probs)
        self.current_emotion = EMOTION_LABELS[pred_idx]
        self.confidence = probs[pred_idx]
        
        # Call callback if set
        if self.callback:
            self.callback({
                'emotion': self.current_emotion,
                'confidence': self.confidence,
                'probabilities': {EMOTION_LABELS[i]: probs[i] for i in range(len(EMOTION_LABELS))}
            })
    
    def start(self):
        """Start real-time recognition."""
        self.running = True
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_audio)
        self.process_thread.start()
        
        # Start audio stream
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self._audio_callback,
            blocksize=int(self.sample_rate * 0.1)  # 100ms blocks
        )
        self.stream.start()
        
        print("Real-time recognition started")
        print(f"Window: {self.window_duration}s, Hop: {self.hop_duration}s")
    
    def stop(self):
        """Stop real-time recognition."""
        self.running = False
        
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
            
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=1.0)
            
        print("Real-time recognition stopped")
    
    def get_current_state(self) -> dict:
        """Get current emotion state."""
        return {
            'emotion': self.current_emotion,
            'confidence': self.confidence
        }


def print_emotion(result: dict):
    """Simple callback to print emotion results."""
    emotion = result['emotion']
    confidence = result['confidence']
    
    # Simple emoji mapping
    emoji_map = {
        'neutral': '😐',
        'happy': '😊',
        'sad': '😢',
        'angry': '😠',
        'fearful': '😨',
        'surprised': '😲'
    }
    
    emoji = emoji_map.get(emotion, '❓')
    bar = '█' * int(confidence * 20)
    
    print(f"\r{emoji} {emotion:10s} [{bar:20s}] {confidence:.1%}", end='', flush=True)


def main():
    """Demo real-time emotion recognition."""
    print("=" * 50)
    print("Real-Time Emotion Recognition Demo")
    print("=" * 50)
    print("\nPress Ctrl+C to stop\n")
    
    # Create recognizer with print callback
    recognizer = RealTimeEmotionRecognizer(
        model_path="models/best_model.pt",
        callback=print_emotion
    )
    
    try:
        recognizer.start()
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopping...")
        recognizer.stop()


if __name__ == "__main__":
    main()
