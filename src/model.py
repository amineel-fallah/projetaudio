"""
CNN-LSTM Model for Speech Emotion Recognition
CNN extracts spectral patterns, LSTM captures temporal dynamics
Enhanced version with better emotion differentiation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance and hard examples.
    Helps the model focus on hard-to-classify emotions like neutral, happy, sad.
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Class weights (tensor of shape [num_classes])
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for better feature combination."""
    
    def __init__(self, hidden_size: int, num_heads: int = 4):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        return self.out(context).mean(dim=1), attn


class AttentionLayer(nn.Module):
    """Attention mechanism for LSTM outputs."""
    
    def __init__(self, hidden_size: int):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # attention_weights: (batch, seq_len, 1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        # context_vector: (batch, hidden_size)
        return context_vector, attention_weights


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        
    def forward(self, x):
        batch, channels, h, w = x.shape
        y = x.view(batch, channels, -1).mean(dim=2)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(batch, channels, 1, 1)
        return x * y


class CNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM model for emotion recognition.
    
    Architecture:
    - CNN layers with SE blocks extract local spectral patterns from mel-spectrograms
    - LSTM layers capture temporal dynamics across time frames
    - Multi-head attention for better feature aggregation
    - Fully connected layers for classification
    """
    
    def __init__(self, num_classes: int = 6, input_channels: int = 1,
                 hidden_size: int = 128, num_lstm_layers: int = 2,
                 dropout: float = 0.3, use_attention: bool = True):
        """
        Initialize CNN-LSTM model.
        
        Args:
            num_classes: Number of emotion classes
            input_channels: Number of input channels (1 for mono audio)
            hidden_size: LSTM hidden state size
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout probability
            use_attention: Whether to use attention mechanism
        """
        super(CNNLSTM, self).__init__()
        
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.use_attention = use_attention
        
        # CNN layers for spectral feature extraction with SE blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.5)
        )
        self.se1 = SEBlock(32)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.5)
        )
        self.se2 = SEBlock(64)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.7)
        )
        self.se3 = SEBlock(128)
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.7)
        )
        self.se4 = SEBlock(256)
        
        # LSTM for temporal dynamics
        self.lstm = nn.LSTM(
            input_size=256 * 8,  # channels * frequency bins after pooling
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Attention mechanism - use multi-head for better feature extraction
        if use_attention:
            self.attention = MultiHeadAttention(hidden_size * 2, num_heads=4)
        
        # Fully connected layers with residual connection
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, 1, n_mels, time_frames)
            
        Returns:
            Logits (batch, num_classes)
        """
        # CNN feature extraction with SE blocks
        x = self.conv1(x)
        x = self.se1(x)
        x = self.conv2(x)
        x = self.se2(x)
        x = self.conv3(x)
        x = self.se3(x)
        x = self.conv4(x)
        x = self.se4(x)
        
        # Reshape for LSTM: (batch, time, features)
        batch_size = x.size(0)
        x = x.permute(0, 3, 1, 2)  # (batch, time, channels, freq)
        x = x.reshape(batch_size, x.size(1), -1)  # (batch, time, features)
        
        # LSTM temporal modeling
        lstm_out, _ = self.lstm(x)
        
        # Apply attention or use last hidden state
        if self.use_attention:
            x, _ = self.attention(lstm_out)
        else:
            x = lstm_out[:, -1, :]
        
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
    print("Testing CNN-LSTM Model")
    print("=" * 40)
    
    model = CNNLSTM(num_classes=6)
    
    # Create dummy input: (batch=4, channels=1, n_mels=128, time_frames=128)
    dummy_input = torch.randn(4, 1, 128, 128)
    
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
