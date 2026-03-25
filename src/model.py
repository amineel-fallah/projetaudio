"""
CNN-LSTM Model for Speech Emotion Recognition
CNN extracts spectral patterns, LSTM captures temporal dynamics
Attention pooling over all LSTM steps (better than last-step only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Soft-attention over time steps – learns which frames matter most.
    Replaces the naive last-step readout.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch, time, hidden)
        weights = torch.softmax(self.attn(lstm_out), dim=1)   # (batch, time, 1)
        context = (weights * lstm_out).sum(dim=1)              # (batch, hidden)
        return context


class CNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM model for emotion recognition.

    Architecture:
    - CNN layers extract local spectral patterns from mel-spectrograms
    - LSTM layers capture temporal dynamics across time frames
    - Attention pooling over all LSTM outputs (better than last-step)
    - Fully connected layers for classification
    """

    def __init__(self, num_classes: int = 6, input_channels: int = 1,
                 hidden_size: int = 256, num_lstm_layers: int = 2,
                 dropout: float = 0.4):
        super(CNNLSTM, self).__init__()

        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # CNN layers for spectral feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3)
        )

        # After 4× MaxPool2d(2,2): 128 mel → 8 freq bins
        lstm_input_size = 256 * 8

        # Bidirectional LSTM for temporal dynamics
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        # Attention over all LSTM time steps
        self.attention = AttentionPooling(hidden_size * 2)  # *2 bidirectional

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, 1, n_mels, time_frames)

        Returns:
            Logits (batch, num_classes)
        """
        # CNN feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Reshape for LSTM: (batch, time, features)
        batch_size = x.size(0)
        x = x.permute(0, 3, 1, 2)            # (batch, time, channels, freq)
        x = x.reshape(batch_size, x.size(1), -1)  # (batch, time, features)

        # LSTM temporal modeling
        lstm_out, _ = self.lstm(x)           # (batch, time, hidden*2)

        # Attention-weighted pooling over all time steps
        x = self.attention(lstm_out)          # (batch, hidden*2)

        # Classification
        logits = self.fc(x)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability distribution over classes."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class."""
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)


class SimpleCNN(nn.Module):
    """
    Simple CNN model for baseline comparison.
    """

    def __init__(self, num_classes: int = 6, dropout: float = 0.3):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(model_type: str = "cnn_lstm", num_classes: int = 6, **kwargs) -> nn.Module:
    """
    Factory function to get model by type.

    Args:
        model_type: "cnn_lstm" or "simple_cnn"
        num_classes: Number of emotion classes

    Returns:
        PyTorch model
    """
    if model_type == "cnn_lstm":
        return CNNLSTM(num_classes=num_classes, **kwargs)
    elif model_type == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model architecture
    print("Testing CNN-LSTM Model with Attention Pooling")
    print("=" * 40)

    model = CNNLSTM(num_classes=6)

    # Create dummy input: (batch=4, channels=1, n_mels=128, time_frames=128)
    dummy_input = torch.randn(4, 1, 128, 128)

    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
