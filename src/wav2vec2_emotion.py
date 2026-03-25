"""
Wav2Vec2 Pre-finetuned for Speech Emotion Recognition (EXPRESS VERSION)
Uses existing emotion-finetuned model from HuggingFace for instant results
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from typing import Union, Dict, Tuple
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EMOTION_LABELS, SAMPLE_RATE


class Wav2Vec2EmotionClassifier:
    """
    Wav2Vec2 emotion classifier using pre-finetuned model.
    
    Uses 'ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition'
    which is already fine-tuned on multiple emotion datasets.
    """
    
    def __init__(self, 
                 model_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                 device: str = None):
        """
        Initialize Wav2Vec2 emotion classifier.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to use (cuda/cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Loading Wav2Vec2 model on {self.device}...")
        
        try:
            # Load pre-trained emotion model
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            ).to(self.device)
            
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            
            self.model.eval()
            
            # Model emotion labels (might differ from our 6 classes)
            self.model_labels = self.model.config.id2label
            
            print(f"✅ Model loaded! Labels: {list(self.model_labels.values())}")
            
            # Create mapping to our 6 emotions
            self._create_label_mapping()
            
        except Exception as e:
            print(f"⚠️ Could not load pre-finetuned model: {e}")
            print("💡 Falling back to base Wav2Vec2 with custom head...")
            self._load_custom_model()
    
    def _create_label_mapping(self):
        """Map model's labels to our 6 emotion classes."""
        # Common mappings
        self.label_map = {
            'angry': 'angry',
            'disgust': 'angry',  # map disgust to angry
            'fear': 'fearful',
            'fearful': 'fearful',
            'happy': 'happy',
            'joy': 'happy',
            'neutral': 'neutral',
            'calm': 'neutral',
            'sad': 'sad',
            'surprise': 'surprised',
            'surprised': 'surprised'
        }
    
    def _load_custom_model(self):
        """Fallback: Load base Wav2Vec2 with custom head."""
        from transformers import Wav2Vec2Model
        
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(self.device)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        
        # Add custom classifier
        hidden_size = self.model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, len(EMOTION_LABELS))
        ).to(self.device)
        
        self.model_labels = {i: label for i, label in enumerate(EMOTION_LABELS)}
        self.custom_head = True
    
    def preprocess_audio(self, audio: Union[np.ndarray, torch.Tensor], 
                        sample_rate: int = SAMPLE_RATE) -> Dict:
        """
        Preprocess audio for Wav2Vec2.
        
        Args:
            audio: Audio waveform (1D array)
            sample_rate: Sample rate in Hz
            
        Returns:
            Dictionary with input_values
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        # Ensure 1D
        if audio.ndim > 1:
            audio = audio.squeeze()
        
        # Normalize
        audio = audio / (np.abs(audio).max() + 1e-8)
        
        # Extract features
        inputs = self.feature_extractor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    @torch.no_grad()
    def predict(self, audio: Union[np.ndarray, torch.Tensor], 
                sample_rate: int = SAMPLE_RATE,
                return_all: bool = False) -> Union[str, Tuple[str, Dict]]:
        """
        Predict emotion from audio.
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate
            return_all: Return probabilities for all emotions
            
        Returns:
            Predicted emotion label (and optionally all probabilities)
        """
        self.model.eval()
        
        # Preprocess
        inputs = self.preprocess_audio(audio, sample_rate)
        
        # Forward pass
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)[0]
        
        # Map to our 6 emotions
        emotion_probs = self._map_to_our_emotions(probs)
        
        # Get top emotion
        top_emotion = max(emotion_probs.items(), key=lambda x: x[1])[0]
        
        if return_all:
            return top_emotion, emotion_probs
        return top_emotion
    
    def _map_to_our_emotions(self, probs: torch.Tensor) -> Dict[str, float]:
        """Map model probabilities to our 6 emotion classes."""
        emotion_scores = {emotion: 0.0 for emotion in EMOTION_LABELS}
        
        # Convert to numpy if needed
        if isinstance(probs, torch.Tensor):
            probs_np = probs.cpu().numpy()
        else:
            probs_np = probs
        
        # Map each model emotion to our emotion classes
        for idx, prob in enumerate(probs_np):
            model_label = self.model_labels.get(idx, '').lower()
            
            # Direct mapping based on exact label match
            our_emotion = self.label_map.get(model_label, None)
            
            if our_emotion and our_emotion in emotion_scores:
                emotion_scores[our_emotion] += float(prob)
        
        # No normalization needed - probabilities already sum to 1 from softmax
        # (multiple model emotions can map to same output emotion, that's fine)
        
        return emotion_scores
    
    def predict_batch(self, audios: list, 
                     sample_rate: int = SAMPLE_RATE) -> list:
        """
        Predict emotions for multiple audio files.
        
        Args:
            audios: List of audio waveforms
            sample_rate: Sample rate
            
        Returns:
            List of (emotion, probabilities) tuples
        """
        results = []
        for audio in audios:
            emotion, probs = self.predict(audio, sample_rate, return_all=True)
            results.append((emotion, probs))
        return results


# Singleton instance for reuse
_wav2vec2_instance = None

def get_wav2vec2_classifier(force_reload: bool = False) -> Wav2Vec2EmotionClassifier:
    """
    Get or create Wav2Vec2 classifier instance (singleton pattern).
    
    Args:
        force_reload: Force reload the model
        
    Returns:
        Wav2Vec2EmotionClassifier instance
    """
    global _wav2vec2_instance
    
    if _wav2vec2_instance is None or force_reload:
        _wav2vec2_instance = Wav2Vec2EmotionClassifier()
    
    return _wav2vec2_instance


if __name__ == "__main__":
    print("=" * 60)
    print("  Wav2Vec2 Speech Emotion Recognition - EXPRESS")
    print("=" * 60)
    
    # Test model loading
    classifier = get_wav2vec2_classifier()
    
    print(f"\n✅ Model ready on {classifier.device}")
    print(f"📊 Target emotions: {EMOTION_LABELS}")
    
    # Test with dummy audio
    print("\n🧪 Testing with random audio...")
    dummy_audio = np.random.randn(16000 * 3)  # 3 seconds
    
    emotion, probs = classifier.predict(dummy_audio, return_all=True)
    
    print(f"\n🎭 Predicted emotion: {emotion}")
    print("\n📈 Probability distribution:")
    for emo, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 40)
        print(f"  {emo:10s} {prob:.2%} {bar}")
