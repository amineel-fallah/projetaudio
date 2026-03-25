"""
Ensemble System for Speech Emotion Recognition
Combines CNN-LSTM and Wav2Vec2 models for robust predictions
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Union
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EMOTION_LABELS


class EmotionEnsemble:
    """
    Ensemble classifier combining multiple models.
    
    Uses weighted voting based on model confidence and performance.
    """
    
    def __init__(self, models: Dict[str, any], weights: Dict[str, float] = None):
        """
        Initialize ensemble.
        
        Args:
            models: Dictionary of {model_name: model_instance}
            weights: Dictionary of {model_name: weight} for voting
        """
        self.models = models
        
        # Default weights (can be tuned based on validation performance)
        self.weights = weights or {
            'wav2vec2': 0.6,  # Higher weight for Wav2Vec2 (better performance)
            'cnn_lstm': 0.4   # Lower weight for CNN-LSTM
        }
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        print(f"🔗 Ensemble initialized with weights: {self.weights}")
    
    def predict(self, audio: Union[np.ndarray, torch.Tensor], 
                sample_rate: int = 16000,
                return_details: bool = False) -> Union[str, Tuple[str, Dict]]:
        """
        Predict emotion using ensemble.
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate
            return_details: Return detailed predictions from each model
            
        Returns:
            Predicted emotion (and optionally details)
        """
        predictions = {}
        all_probs = {}
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            try:
                if model_name == 'wav2vec2':
                    emotion, probs = model.predict(audio, sample_rate, return_all=True)
                else:
                    # CNN-LSTM or other models
                    probs = self._get_cnn_lstm_probs(model, audio, sample_rate)
                    emotion = max(probs.items(), key=lambda x: x[1])[0]
                
                predictions[model_name] = emotion
                all_probs[model_name] = probs
                
            except Exception as e:
                print(f"⚠️ Model {model_name} failed: {e}")
                # Use uniform distribution as fallback
                uniform = 1.0 / len(EMOTION_LABELS)
                all_probs[model_name] = {emo: uniform for emo in EMOTION_LABELS}
        
        # Weighted voting
        ensemble_probs = self._weighted_vote(all_probs)
        
        # Get final prediction
        final_emotion = max(ensemble_probs.items(), key=lambda x: x[1])[0]
        
        if return_details:
            details = {
                'predictions': predictions,
                'probabilities': all_probs,
                'ensemble': ensemble_probs,
                'weights': self.weights
            }
            return final_emotion, details
        
        return final_emotion
    
    def _weighted_vote(self, all_probs: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Combine predictions using weighted voting.
        
        Args:
            all_probs: {model_name: {emotion: probability}}
            
        Returns:
            Combined probability distribution
        """
        ensemble_probs = {emotion: 0.0 for emotion in EMOTION_LABELS}
        
        for model_name, probs in all_probs.items():
            weight = self.weights.get(model_name, 0.0)
            
            for emotion, prob in probs.items():
                if emotion in ensemble_probs:
                    ensemble_probs[emotion] += weight * prob
        
        # Normalize
        total = sum(ensemble_probs.values())
        if total > 0:
            ensemble_probs = {k: v/total for k, v in ensemble_probs.items()}
        
        return ensemble_probs
    
    def _get_cnn_lstm_probs(self, model, audio: np.ndarray, 
                           sample_rate: int) -> Dict[str, float]:
        """Get probabilities from CNN-LSTM model."""
        from src.features import extract_mel_spectrogram
        
        # Extract features
        mel_spec = extract_mel_spectrogram(audio, sample_rate)
        
        # Convert to tensor
        if not isinstance(mel_spec, torch.Tensor):
            mel_spec = torch.FloatTensor(mel_spec)
        
        # Add batch and channel dimensions
        mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)
        
        # Get probabilities
        with torch.no_grad():
            model.eval()
            probs = model.predict_proba(mel_spec)[0]
        
        # Convert to dictionary
        return {emotion: float(probs[i]) for i, emotion in enumerate(EMOTION_LABELS)}
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update model weights.
        
        Args:
            new_weights: New weights dictionary
        """
        self.weights = new_weights
        
        # Normalize
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
        print(f"✅ Weights updated: {self.weights}")
    
    def calibrate_weights(self, validation_data: List[Tuple],
                         metric: str = 'f1') -> Dict[str, float]:
        """
        Automatically calibrate weights based on validation performance.
        
        Args:
            validation_data: List of (audio, true_label) tuples
            metric: Metric to optimize ('accuracy', 'f1')
            
        Returns:
            Optimized weights
        """
        from sklearn.metrics import f1_score, accuracy_score
        
        # Evaluate each model
        scores = {}
        for model_name in self.models.keys():
            predictions = []
            true_labels = []
            
            for audio, true_label in validation_data:
                # Temporarily use only this model
                temp_weights = {m: 0.0 for m in self.models.keys()}
                temp_weights[model_name] = 1.0
                
                old_weights = self.weights
                self.weights = temp_weights
                
                pred = self.predict(audio[0], audio[1])
                predictions.append(pred)
                true_labels.append(true_label)
                
                self.weights = old_weights
            
            # Calculate score
            if metric == 'f1':
                score = f1_score(true_labels, predictions, average='macro')
            else:
                score = accuracy_score(true_labels, predictions)
            
            scores[model_name] = score
            print(f"  {model_name}: {metric}={score:.3f}")
        
        # Set weights proportional to scores
        total_score = sum(scores.values())
        new_weights = {k: v/total_score for k, v in scores.items()}
        
        self.update_weights(new_weights)
        return new_weights


def create_ensemble(use_wav2vec2: bool = True, 
                   use_cnn_lstm: bool = True) -> EmotionEnsemble:
    """
    Create emotion ensemble with specified models.
    
    Args:
        use_wav2vec2: Include Wav2Vec2 model
        use_cnn_lstm: Include CNN-LSTM model
        
    Returns:
        EmotionEnsemble instance
    """
    models = {}
    
    if use_wav2vec2:
        print("📥 Loading Wav2Vec2...")
        from src.wav2vec2_emotion import get_wav2vec2_classifier
        models['wav2vec2'] = get_wav2vec2_classifier()
    
    if use_cnn_lstm:
        print("📥 Loading CNN-LSTM...")
        from src.model import get_model
        import os
        
        model_path = "models/best_model.pt"
        if os.path.exists(model_path):
            model = get_model('cnn_lstm', num_classes=len(EMOTION_LABELS))
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            models['cnn_lstm'] = model
        else:
            print("⚠️ CNN-LSTM model not found, using only Wav2Vec2")
    
    if not models:
        raise ValueError("No models available for ensemble!")
    
    return EmotionEnsemble(models)


if __name__ == "__main__":
    print("=" * 60)
    print("  Emotion Ensemble System")
    print("=" * 60)
    
    # Create ensemble
    ensemble = create_ensemble(use_wav2vec2=True, use_cnn_lstm=False)
    
    # Test
    print("\n🧪 Testing ensemble...")
    dummy_audio = np.random.randn(16000 * 3)
    
    emotion, details = ensemble.predict(dummy_audio, return_details=True)
    
    print(f"\n🎭 Final prediction: {emotion}")
    print(f"\n📊 Ensemble probabilities:")
    for emo, prob in sorted(details['ensemble'].items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 40)
        print(f"  {emo:10s} {prob:.2%} {bar}")
