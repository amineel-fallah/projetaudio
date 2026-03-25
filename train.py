"""
Training script for Speech Emotion Recognition
"""

import os
import sys
import argparse
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (EPOCHS, BATCH_SIZE, LEARNING_RATE, NUM_CLASSES, 
                    MODEL_DIR, EMOTION_LABELS)
from src.model import get_model
from src.dataset import create_dataloaders


class Trainer:
    """Training class for emotion recognition models."""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, device: str = None,
                 learning_rate: float = LEARNING_RATE,
                 class_weights: torch.Tensor = None):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Learning rate
            class_weights: Optional class weights for imbalanced data
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss function with class weights
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'val_f1': []
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
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
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
    
    # Create model
    model = get_model("cnn_lstm", num_classes=len(EMOTION_LABELS))
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create datasets
    print("\nLoading dataset...")
    from src.dataset import RAVDESSDataset

    full_dataset = RAVDESSDataset(str(ravdess_path), split="train", augment=False)
    print(f"Total train samples: {len(full_dataset)}")

    if len(full_dataset) == 0:
        print("Error: No samples found in dataset!")
        return

    train_loader, val_loader, _ = create_dataloaders(
        str(ravdess_path),
        batch_size=BATCH_SIZE,
        num_workers=0
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    if len(val_loader.dataset) == 0:
        print("Error: Validation split is empty. Check dataset structure.")
        return
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=LEARNING_RATE
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
