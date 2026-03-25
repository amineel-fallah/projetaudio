"""
Advanced CNN-LSTM Model for Speech Emotion Recognition
Incorporates state-of-the-art techniques:
- Squeeze-and-Excitation blocks
- Multi-head attention
- Residual connections
- Advanced regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block - adaptively recalibrates channel-wise features.
    Boosts important channels, suppresses less useful ones.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: global spatial pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: channel-wise scaling
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for LSTM outputs.
    Learns to focus on different aspects of temporal dynamics.
    """
    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        # x: (batch, time, hidden)
        batch_size, seq_len, _ = x.size()
        
        # Linear projections in batch
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Final linear
        out = self.out(context)
        
        # Average over time
        return out.mean(dim=1)


class ResidualBlock(nn.Module):
    """
    Residual convolutional block with identity skip connection.
    Helps with gradient flow and enables deeper networks.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = F.gelu(out)
        return out


class AdvancedCNNLSTM(nn.Module):
    """
    State-of-the-art CNN-LSTM model with:
    - Squeeze-and-Excitation blocks for channel attention
    - Residual connections for better gradient flow
    - Multi-head attention for temporal modeling
    - Advanced regularization (dropout, batch norm)
    - GELU activation for better gradients
    """

    def __init__(self, num_classes: int = 6, input_channels: int = 1,
                 hidden_size: int = 256, num_lstm_layers: int = 2,
                 dropout: float = 0.4, num_attention_heads: int = 4):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # Stage 1: Initial feature extraction with SE
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1)
        )
        self.se1 = SqueezeExcitation(64)
        
        # Stage 2: Deeper features with residual
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        self.res2 = ResidualBlock(128)
        self.se2 = SqueezeExcitation(128)
        
        # Stage 3: High-level features
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        self.res3 = ResidualBlock(256)
        self.se3 = SqueezeExcitation(256)
        
        # Stage 4: Final CNN stage
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3)
        )
        self.se4 = SqueezeExcitation(512)

        # After 4× MaxPool2d(2,2): 128 mel → 8 freq bins
        # Flattened feature size per time step
        self.feature_size = 512 * 8

        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Multi-head attention over LSTM outputs
        self.attention = MultiHeadAttention(hidden_size * 2, num_heads=num_attention_heads)
        
        # Classification head with residual
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn_fc = nn.BatchNorm1d(hidden_size)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, freq, time) mel-spectrogram
        Returns:
            logits: (batch, num_classes)
        """
        batch_size = x.size(0)

        # CNN feature extraction with SE blocks
        x = self.conv1(x)
        x = self.se1(x)
        
        x = self.conv2(x)
        x = self.res2(x)
        x = self.se2(x)
        
        x = self.conv3(x)
        x = self.res3(x)
        x = self.se3(x)
        
        x = self.conv4(x)
        x = self.se4(x)

        # Reshape for LSTM: (batch, time, features)
        # x is (batch, channels=512, freq=8, time)
        x = x.permute(0, 3, 1, 2)  # → (batch, time, 512, 8)
        x = x.reshape(batch_size, x.size(1), -1)  # → (batch, time, 512*8)

        # LSTM temporal modeling
        lstm_out, _ = self.lstm(x)  # → (batch, time, hidden*2)

        # Multi-head attention pooling
        context = self.attention(lstm_out)  # → (batch, hidden*2)

        # Classification with residual
        out = F.gelu(self.bn_fc(self.fc1(context)))
        out = self.dropout_fc(out)
        
        # Second layer
        out = F.gelu(self.fc2(out))
        out = self.dropout_fc(out)
        
        # Final logits
        logits = self.fc3(out)

        return logits


def get_advanced_model(num_classes: int = 6, **kwargs):
    """Factory function to create the advanced model."""
    return AdvancedCNNLSTM(num_classes=num_classes, **kwargs)
