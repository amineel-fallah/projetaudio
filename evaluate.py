"""
Evaluation script for Speech Emotion Recognition
Computes detailed metrics and generates visualizations
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score,
    accuracy_score, precision_score, recall_score
)
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import EMOTION_LABELS, MODEL_DIR, RAVDESS_DIR
from src.model import get_model
from src.dataset import RAVDESSDataset
from src.utils import get_device, load_checkpoint
from torch.utils.data import DataLoader


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set.
    
    Returns:
        Dictionary with predictions, labels, and metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    results = {
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
        'f1_per_class': f1_score(all_labels, all_preds, average=None),
        'precision_macro': precision_score(all_labels, all_preds, average='macro'),
        'recall_macro': recall_score(all_labels, all_preds, average='macro'),
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }
    
    return results


def plot_confusion_matrix(cm, labels, save_path=None, normalize=False):
    """Plot confusion matrix."""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix to {save_path}")
    plt.close()


def plot_per_class_metrics(results, labels, save_path=None):
    """Plot per-class F1 scores."""
    f1_scores = results['f1_per_class']
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.RdYlGn(f1_scores)
    bars = plt.bar(labels, f1_scores, color=colors)
    
    # Add value labels
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.axhline(y=0.8, color='red', linestyle='--', label='Target (0.80)')
    plt.axhline(y=results['f1_macro'], color='blue', linestyle='--', 
                label=f"Macro F1 ({results['f1_macro']:.3f})")
    
    plt.xlabel('Emotion')
    plt.ylabel('F1 Score')
    plt.title('Per-Class F1 Scores')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved per-class metrics to {save_path}")
    plt.close()


def print_classification_report(labels, predictions, class_names):
    """Print detailed classification report."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(labels, predictions, target_names=class_names))


def main():
    """Main evaluation function."""
    print("=" * 60)
    print("Speech Emotion Recognition - Evaluation")
    print("=" * 60)
    
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    
    # Find best model
    model_path = os.path.join(MODEL_DIR, "best_model.pt")
    if not os.path.exists(model_path):
        print(f"No trained model found at {model_path}")
        print("Please train the model first using: python train.py")
        return
    
    # Load model
    model = get_model("cnn_lstm", num_classes=len(EMOTION_LABELS))
    load_checkpoint(model, model_path)
    model = model.to(device)
    
    # Create test dataset
    test_dataset = RAVDESSDataset(RAVDESS_DIR, split="test", augment=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Evaluate
    results = evaluate_model(model, test_loader, device)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Accuracy:        {results['accuracy']:.4f}")
    print(f"F1 Macro:        {results['f1_macro']:.4f}")
    print(f"F1 Weighted:     {results['f1_weighted']:.4f}")
    print(f"Precision Macro: {results['precision_macro']:.4f}")
    print(f"Recall Macro:    {results['recall_macro']:.4f}")
    
    # Check target
    if results['f1_macro'] >= 0.80:
        print("\n✓ Target F1 score (>0.80) achieved!")
    else:
        print(f"\n✗ Target F1 score (>0.80) not reached. Current: {results['f1_macro']:.4f}")
    
    # Print per-class scores
    print("\nPer-class F1 scores:")
    for emotion, score in zip(EMOTION_LABELS, results['f1_per_class']):
        status = "✓" if score >= 0.80 else "✗"
        print(f"  {status} {emotion}: {score:.4f}")
    
    # Classification report
    print_classification_report(results['labels'], results['predictions'], EMOTION_LABELS)
    
    # Save visualizations
    output_dir = Path("logs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_confusion_matrix(
        results['confusion_matrix'], 
        EMOTION_LABELS,
        save_path=output_dir / "confusion_matrix.png"
    )
    
    plot_confusion_matrix(
        results['confusion_matrix'], 
        EMOTION_LABELS,
        save_path=output_dir / "confusion_matrix_normalized.png",
        normalize=True
    )
    
    plot_per_class_metrics(
        results, 
        EMOTION_LABELS,
        save_path=output_dir / "per_class_f1.png"
    )
    
    print(f"\nVisualizations saved to {output_dir}/")


if __name__ == "__main__":
    main()
