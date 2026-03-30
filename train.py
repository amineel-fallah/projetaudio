"""
Training script for Speech Emotion Recognition
"""

import os
import sys
import argparse
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (EPOCHS, BATCH_SIZE, LEARNING_RATE, NUM_CLASSES, 
                    MODEL_DIR, EMOTION_LABELS)
from src.model import get_model
from src.dataset import create_dataloaders


class FocalLossWithSmoothing(nn.Module):
    """
    Focal Loss with Label Smoothing for better emotion detection.
    Focal Loss helps focus on hard-to-classify examples (neutral, happy, sad).
    """
    
    def __init__(self, gamma=2.0, smoothing=0.1, weight=None):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.weight = weight
        
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        
        # Apply label smoothing
        with torch.no_grad():
            smooth_target = torch.zeros_like(pred)
            smooth_target.fill_(self.smoothing / (n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        # Compute log softmax
        log_probs = F.log_softmax(pred, dim=-1)
        probs = torch.exp(log_probs)
        
        # Focal weight: (1 - p)^gamma
        focal_weight = (1 - probs) ** self.gamma
        
        # Compute focal loss with smoothed labels
        loss = -focal_weight * smooth_target * log_probs
        
        # Apply class weights if provided
        if self.weight is not None:
            weight = self.weight.to(pred.device)
            loss = loss * weight.unsqueeze(0)
        
        return loss.sum(dim=-1).mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing to prevent overconfidence."""
    
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_probs = torch.nn.functional.log_softmax(pred, dim=-1)
        
        # Create smoothed labels
        with torch.no_grad():
            smooth_target = torch.zeros_like(log_probs)
            smooth_target.fill_(self.smoothing / (n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        # Apply class weights if provided
        if self.weight is not None:
            smooth_target = smooth_target * self.weight.unsqueeze(0)
        
        loss = (-smooth_target * log_probs).sum(dim=-1)
        return loss.mean()


class Trainer:
    """Training class for emotion recognition models."""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, device: str = None,
                 learning_rate: float = LEARNING_RATE,
                 class_weights: torch.Tensor = None,
                 label_smoothing: float = 0.1,
                 use_focal_loss: bool = True,
                 focal_gamma: float = 2.0):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Learning rate
            class_weights: Optional class weights for imbalanced data
            label_smoothing: Label smoothing factor (0.1 = 10% smoothing)
            use_focal_loss: Use Focal Loss for hard examples (recommended)
            focal_gamma: Focal loss gamma parameter
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss function - Focal Loss with smoothing helps with all emotions
        if use_focal_loss:
            self.criterion = FocalLossWithSmoothing(
                gamma=focal_gamma,
                smoothing=label_smoothing,
                weight=class_weights
            )
            print(f"Using Focal Loss (gamma={focal_gamma}) with label smoothing ({label_smoothing})")
        elif label_smoothing > 0:
            self.criterion = LabelSmoothingCrossEntropy(
                smoothing=label_smoothing,
                weight=class_weights.to(self.device) if class_weights is not None else None
            )
        else:
            if class_weights is not None:
                class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Learning rate scheduler - cosine annealing for better convergence
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'val_f1': [], 'per_class_f1': []
        }
        
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
        f1 = f1_score(all_targets, all_preds, average='macro')
        
        return avg_loss, accuracy, f1
    
    def train(self, epochs: int = EPOCHS, save_path: str = None) -> dict:
        """
        Full training loop.
        
        Args:
            epochs: Number of epochs
            save_path: Path to save best model
            
        Returns:
            Training history
        """
        best_f1 = 0
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 30)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_f1 = self.validate()
            
            # Update scheduler (CosineAnnealingWarmRestarts takes epoch number)
            self.scheduler.step(epoch)
            
            # Log metrics
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            
            # Save best model
            if val_f1 > best_f1 and save_path:
                best_f1 = val_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1': val_f1,
                }, save_path)
                print(f"Saved best model with F1: {val_f1:.4f}")
        
        return self.history
    
    def plot_history(self, save_path: str = None):
        """Plot training history."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_title('Loss')
        axes[0].legend()
        
        # Accuracy
        axes[1].plot(self.history['train_acc'], label='Train')
        axes[1].plot(self.history['val_acc'], label='Val')
        axes[1].set_title('Accuracy')
        axes[1].legend()
        
        # F1 Score
        axes[2].plot(self.history['val_f1'], label='Val F1')
        axes[2].axhline(y=0.8, color='r', linestyle='--', label='Target')
        axes[2].set_title('Macro F1 Score')
        axes[2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()


def evaluate_model(model: nn.Module, test_loader: DataLoader, 
                   device: str = None) -> dict:
    """
    Evaluate model on test set.
    
    Returns:
        Dictionary with metrics and confusion matrix
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Compute metrics
    results = {
        'accuracy': np.mean(np.array(all_preds) == np.array(all_targets)),
        'f1_macro': f1_score(all_targets, all_preds, average='macro'),
        'f1_per_class': f1_score(all_targets, all_preds, average=None),
        'confusion_matrix': confusion_matrix(all_targets, all_preds),
        'classification_report': classification_report(
            all_targets, all_preds, target_names=EMOTION_LABELS
        )
    }
    
    return results


def plot_confusion_matrix(cm: np.ndarray, labels: list, save_path: str = None):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def main(epochs: int = None):
    """Main training function."""
    print("=" * 50)
    print("Speech Emotion Recognition - Training")
    print("=" * 50)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create model directory
    Path(MODEL_DIR).mkdir(exist_ok=True)
    
    # Check if dataset exists
    ravdess_path = Path("data/ravdess")
    if not ravdess_path.exists() or len(list(ravdess_path.glob("Actor_*"))) == 0:
        print("\nError: RAVDESS dataset not found!")
        print("Please run: python download_ravdess.py")
        return
    
    # Create model with improved architecture
    model = get_model("cnn_lstm", num_classes=len(EMOTION_LABELS), 
                      hidden_size=256, dropout=0.4, use_attention=True)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create datasets with augmentation
    print("\nLoading dataset...")
    from src.dataset import RAVDESSDataset

    # Load training data to compute class weights
    train_dataset = RAVDESSDataset(str(ravdess_path), split="train", augment=True)
    print(f"Total train samples (with augmentation): {len(train_dataset)}")

    if len(train_dataset) == 0:
        print("Error: No samples found in dataset!")
        return

    # Compute class weights to handle imbalance
    train_labels = [train_dataset.labels[i] for i in train_dataset.indices]
    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(NUM_CLASSES),
        y=train_labels
    )
    class_weights = torch.FloatTensor(class_weights)
    print(f"Class weights: {class_weights}")

    train_loader, val_loader, _ = create_dataloaders(
        str(ravdess_path),
        batch_size=BATCH_SIZE,
        num_workers=0,
        augment=True  # Enable augmentation
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    if len(val_loader.dataset) == 0:
        print("Error: Validation split is empty. Check dataset structure.")
        return
    
    # Create trainer with class weights and label smoothing
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=LEARNING_RATE,
        class_weights=class_weights,
        label_smoothing=0.1
    )
    
    # Train
    train_epochs = epochs if epochs is not None else EPOCHS
    print(f"\nStarting training for {train_epochs} epochs...")
    print("-" * 50)
    
    save_path = os.path.join(MODEL_DIR, "best_model.pt")
    history = trainer.train(epochs=train_epochs, save_path=save_path)
    
    # Save training history plot
    print("\nTraining complete!")
    print(f"Best model saved to: {save_path}")
    
    # Final metrics
    print("\n" + "=" * 50)
    print("Final Results:")
    print(f"  Best Val F1: {max(history['val_f1']):.4f}")
    print(f"  Best Val Acc: {max(history['val_acc']):.4f}")
    print("=" * 50)
    

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Train SER model")
        parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
        args = parser.parse_args()
        main(epochs=args.epochs)
