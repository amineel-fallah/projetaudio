"""
Inference script for Speech Emotion Recognition
Load a trained model and predict emotions from audio files
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import EMOTION_LABELS, MODEL_DIR, SAMPLE_RATE
from src.model import get_model
from src.features import load_audio, extract_mel_spectrogram
from src.utils import get_device, load_checkpoint


def predict_file(model, file_path: str, device: str) -> dict:
    """
    Predict emotion from a single audio file.
    
    Args:
        model: Trained model
        file_path: Path to audio file
        device: Device to use
        
    Returns:
        Dictionary with prediction results
    """
    # Load and process audio
    signal = load_audio(file_path)
    mel_spec = extract_mel_spectrogram(signal)
    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
    
    # Convert to tensor
    input_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)
    input_tensor = input_tensor.to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
    
    # Get results
    pred_idx = np.argmax(probs)
    
    return {
        'file': file_path,
        'predicted_emotion': EMOTION_LABELS[pred_idx],
        'confidence': float(probs[pred_idx]),
        'probabilities': {EMOTION_LABELS[i]: float(probs[i]) for i in range(len(EMOTION_LABELS))}
    }


def predict_directory(model, dir_path: str, device: str) -> list:
    """
    Predict emotions for all audio files in a directory.
    
    Args:
        model: Trained model
        dir_path: Path to directory
        device: Device to use
        
    Returns:
        List of prediction results
    """
    results = []
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    
    dir_path = Path(dir_path)
    for file_path in dir_path.rglob('*'):
        if file_path.suffix.lower() in audio_extensions:
            try:
                result = predict_file(model, str(file_path), device)
                results.append(result)
                print(f"✓ {file_path.name}: {result['predicted_emotion']} ({result['confidence']:.1%})")
            except Exception as e:
                print(f"✗ {file_path.name}: Error - {e}")
    
    return results


def format_results(result: dict) -> str:
    """Format prediction results for display."""
    lines = []
    lines.append(f"\nFile: {result['file']}")
    lines.append(f"Predicted emotion: {result['predicted_emotion']}")
    lines.append(f"Confidence: {result['confidence']:.1%}")
    lines.append("\nAll probabilities:")
    
    sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
    for emotion, prob in sorted_probs:
        bar = '█' * int(prob * 30)
        marker = ' ← ' if emotion == result['predicted_emotion'] else '   '
        lines.append(f"  {emotion:10s} [{bar:30s}] {prob:.1%}{marker}")
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Speech Emotion Recognition Inference')
    parser.add_argument('input', type=str, help='Audio file or directory path')
    parser.add_argument('--model', type=str, default='models/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Speech Emotion Recognition - Inference")
    print("=" * 50)
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    model_path = args.model
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train a model first using: python train.py")
        return
    
    model = get_model("cnn_lstm", num_classes=len(EMOTION_LABELS))
    load_checkpoint(model, model_path)
    model = model.to(device)
    print(f"Loaded model from {model_path}")
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        result = predict_file(model, str(input_path), device)
        print(format_results(result))
        results = [result]
        
    elif input_path.is_dir():
        print(f"\nProcessing directory: {input_path}")
        results = predict_directory(model, str(input_path), device)
        print(f"\nProcessed {len(results)} files")
        
    else:
        print(f"Error: {args.input} not found")
        return
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
