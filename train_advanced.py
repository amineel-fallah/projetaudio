"""
Advanced training script with all modern techniques:
- SpecAugment
- Mixup
- Label Smoothing
- Cosine Annealing LR
- Advanced CNN-LSTM model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

from src.model_advanced import AdvancedCNNLSTM
from src.dataset import EmotionDataset
from src.augmentation import SpecAugment
from config import *


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss - prevents overconfident predictions.
    Instead of hard targets [0,0,1,0,0,0], uses soft targets [ε,ε,1-5ε,ε,ε,ε]
    """
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        """
        Args:
            pred: (batch, num_classes) logits
            target: (batch,) class indices
        """
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            # Create soft targets
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


def mixup_data(x, y, alpha=0.2):
    """
    Mixup augmentation: creates virtual training examples.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def apply_spec_augment(spectrograms):
    """Apply SpecAugment to batch of spectrograms."""
    batch_size = spectrograms.size(0)
    augmented = torch.zeros_like(spectrograms)
    
    for i in range(batch_size):
        # Extract spectrogram (channels, freq, time)
        spec = spectrograms[i, 0].cpu().numpy()
        
        # Apply SpecAugment
        spec_aug = SpecAugment.spec_augment(
            spec,
            time_mask_param=25,
            freq_mask_param=15,
            num_time_masks=2,
            num_freq_masks=2
        )
        
        augmented[i, 0] = torch.from_numpy(spec_aug)
    
    return augmented.to(spectrograms.device)


def train_epoch(model, dataloader, criterion, optimizer, device, use_mixup=True, use_spec_augment=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    
    for batch in pbar:
        spectrograms = batch['spectrogram'].to(device)
        labels = batch['label'].to(device)
        
        # Apply SpecAugment
        if use_spec_augment and np.random.random() < 0.8:
            spectrograms = apply_spec_augment(spectrograms)
        
        # Apply Mixup
        if use_mixup and np.random.random() < 0.5:
            spectrograms, labels_a, labels_b, lam = mixup_data(spectrograms, labels, alpha=0.2)
            
            # Forward pass
            outputs = model(spectrograms)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            # Standard forward pass
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss/len(pbar):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            spectrograms = batch['spectrogram'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total


def train_advanced_model(
    train_dataset,
    val_dataset,
    num_epochs=100,
    batch_size=32,
    learning_rate=0.001,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_dir='models',
    use_label_smoothing=True,
    use_mixup=True,
    use_spec_augment=True
):
    """
    Train advanced model with all modern techniques.
    """
    print("=" * 70)
    print("🚀 ADVANCED CNN-LSTM TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"\n🎯 Advanced Techniques:")
    print(f"  ✅ Label Smoothing: {use_label_smoothing}")
    print(f"  ✅ Mixup: {use_mixup}")
    print(f"  ✅ SpecAugment: {use_spec_augment}")
    print(f"  ✅ Cosine Annealing LR: True")
    print(f"  ✅ Gradient Clipping: True")
    print(f"  ✅ SE Blocks: True")
    print(f"  ✅ Multi-Head Attention: True")
    print("=" * 70)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Model
    model = AdvancedCNNLSTM(
        num_classes=len(EMOTION_LABELS),
        hidden_size=256,
        num_lstm_layers=2,
        dropout=0.4,
        num_attention_heads=4
    ).to(device)
    
    print(f"\n📊 Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function
    if use_label_smoothing:
        criterion = LabelSmoothingLoss(num_classes=len(EMOTION_LABELS), smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler - Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart every 10 epochs
        T_mult=2,  # Double the period after each restart
        eta_min=1e-6
    )
    
    # Training loop
    best_val_acc = 0
    patience = 20
    patience_counter = 0
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n🔥 Starting training...\n")
    
    for epoch in range(num_epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*70}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            use_mixup=use_mixup,
            use_spec_augment=use_spec_augment
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Print results
        print(f"\n📊 Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            save_path = os.path.join(save_dir, 'best_advanced_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, save_path)
            
            print(f"  🌟 New best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping triggered after {patience} epochs without improvement")
            break
    
    print("\n" + "=" * 70)
    print(f"✅ Training completed!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    print("=" * 70)
    
    return model, best_val_acc


if __name__ == "__main__":
    # Load datasets
    print("Loading datasets...")
    
    train_dataset = EmotionDataset(
        data_dir=os.path.join(DATA_DIR, 'train'),
        max_samples_per_class=None,
        augment=False  # We do augmentation in training loop
    )
    
    val_dataset = EmotionDataset(
        data_dir=os.path.join(DATA_DIR, 'val'),
        max_samples_per_class=None,
        augment=False
    )
    
    # Train
    model, best_acc = train_advanced_model(
        train_dataset,
        val_dataset,
        num_epochs=100,
        batch_size=32,
        learning_rate=0.001
    )
    
    print(f"\n🎉 Training finished with best accuracy: {best_acc:.2f}%")
