"""
Wav2Vec2 Fine-tuning for Speech Emotion Recognition
Alternative approach using pre-trained transformer model
"""

import os
import sys
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2Processor
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EMOTION_LABELS


class Wav2Vec2ForEmotionRecognition(nn.Module):
    """
    Wav2Vec2 model fine-tuned for emotion recognition.
    
    Uses the pre-trained Wav2Vec2 base model from Hugging Face
    and adds an emotion classification head.
    """
    
    def __init__(self, num_classes: int = 6, 
                 model_name: str = "facebook/wav2vec2-base",
                 freeze_feature_extractor: bool = True,
                 dropout: float = 0.3):
        """
        Initialize Wav2Vec2 emotion model.
        
        Args:
            num_classes: Number of emotion classes
            model_name: Hugging Face model name
            freeze_feature_extractor: Whether to freeze CNN feature extractor
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Load pre-trained Wav2Vec2
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        
        # Freeze feature extractor (CNN layers)
        if freeze_feature_extractor:
            for param in self.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False
        
        # Get hidden size from config
        hidden_size = self.wav2vec2.config.hidden_size  # Usually 768
        
        # Emotion classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Pooling strategy
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, input_values: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_values: Raw audio waveform (batch, sequence_length)
            attention_mask: Optional attention mask
            
        Returns:
            Logits (batch, num_classes)
        """
        # Get Wav2Vec2 outputs
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask
        )
        
        # Get hidden states: (batch, seq_len, hidden_size)
        hidden_states = outputs.last_hidden_state
        
        # Pool over time dimension: (batch, hidden_size, seq_len) -> (batch, hidden_size, 1)
        pooled = self.pooling(hidden_states.transpose(1, 2)).squeeze(-1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits
    
    def predict(self, input_values: torch.Tensor) -> torch.Tensor:
        """Get predicted class."""
        logits = self.forward(input_values)
        return torch.argmax(logits, dim=1)
    
    def predict_proba(self, input_values: torch.Tensor) -> torch.Tensor:
        """Get probability distribution."""
        logits = self.forward(input_values)
        return torch.softmax(logits, dim=1)


class Wav2Vec2EmotionTrainer:
    """
    Trainer for Wav2Vec2 emotion recognition model.
    """
    
    def __init__(self, model: Wav2Vec2ForEmotionRecognition,
                 processor: Wav2Vec2Processor,
                 device: str = None,
                 learning_rate: float = 1e-5):
        """
        Initialize trainer.
        
        Args:
            model: Wav2Vec2 emotion model
            processor: Wav2Vec2 processor for audio preprocessing
            device: Device to use
            learning_rate: Learning rate (lower for fine-tuning)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.processor = processor
        
        # Optimizer with different learning rates for different parts
        # Lower LR for pre-trained layers, higher for classifier
        self.optimizer = torch.optim.AdamW([
            {'params': model.wav2vec2.parameters(), 'lr': learning_rate},
            {'params': model.classifier.parameters(), 'lr': learning_rate * 10}
        ])
        
        self.criterion = nn.CrossEntropyLoss()
        
    def preprocess_audio(self, audio: torch.Tensor, 
                         sample_rate: int = 16000) -> torch.Tensor:
        """
        Preprocess audio for Wav2Vec2.
        
        Args:
            audio: Raw audio tensor
            sample_rate: Audio sample rate
            
        Returns:
            Preprocessed input values
        """
        # Wav2Vec2 expects 16kHz audio
        inputs = self.processor(
            audio.numpy() if isinstance(audio, torch.Tensor) else audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
        return inputs.input_values.to(self.device)
    
    def train_step(self, audio: torch.Tensor, 
                   labels: torch.Tensor) -> float:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        input_values = self.preprocess_audio(audio)
        labels = labels.to(self.device)
        
        logits = self.model(input_values)
        loss = self.criterion(logits, labels)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


def create_wav2vec2_model(num_classes: int = 6, 
                          pretrained: bool = True) -> Wav2Vec2ForEmotionRecognition:
    """
    Factory function to create Wav2Vec2 emotion model.
    
    Args:
        num_classes: Number of emotion classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        Wav2Vec2 emotion model
    """
    model_name = "facebook/wav2vec2-base" if pretrained else None
    
    if pretrained:
        return Wav2Vec2ForEmotionRecognition(
            num_classes=num_classes,
            model_name=model_name,
            freeze_feature_extractor=True
        )
    else:
        # Initialize with random weights (not recommended)
        config = Wav2Vec2Config()
        model = Wav2Vec2ForEmotionRecognition.__new__(Wav2Vec2ForEmotionRecognition)
        model.wav2vec2 = Wav2Vec2Model(config)
        return model


if __name__ == "__main__":
    print("Wav2Vec2 Emotion Recognition Module")
    print("=" * 40)
    print(f"Number of emotion classes: {len(EMOTION_LABELS)}")
    print(f"Emotions: {EMOTION_LABELS}")
    print("\nNote: This module requires downloading the Wav2Vec2 model")
    print("from Hugging Face (~360MB)")
    print("\nUsage:")
    print("  model = create_wav2vec2_model(num_classes=6)")
    print("  logits = model(audio_tensor)")
