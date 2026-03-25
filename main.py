#!/usr/bin/env python
"""
Main entry point for Speech Emotion Recognition project
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Speech Emotion Recognition using Deep Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py download    # Download RAVDESS dataset
  python main.py train       # Train the model
  python main.py evaluate    # Evaluate on test set
  python main.py demo        # Launch Gradio demo
  python main.py predict audio.wav  # Predict emotion from file
        """
    )
    
    parser.add_argument('action', choices=['download', 'train', 'evaluate', 'demo', 'predict', 'realtime'],
                        help='Action to perform')
    parser.add_argument('input', nargs='?', default=None,
                        help='Input file for prediction')
    parser.add_argument('--model', default='models/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    
    args = parser.parse_args()
    
    if args.action == 'download':
        print("Downloading RAVDESS dataset...")
        import download_ravdess
        download_ravdess.main()
        
    elif args.action == 'train':
        print("Starting training...")
        import train
        train.main(epochs=args.epochs)
        
    elif args.action == 'evaluate':
        print("Evaluating model...")
        import evaluate
        evaluate.main()
        
    elif args.action == 'demo':
        print("Launching Gradio demo...")
        import app
        demo = app.create_demo_interface()
        demo.launch()
        
    elif args.action == 'predict':
        if args.input is None:
            print("Error: Please provide an input file for prediction")
            sys.exit(1)
        print(f"Predicting emotion for: {args.input}")
        import inference
        sys.argv = ['inference.py', args.input, '--model', args.model]
        inference.main()
        
    elif args.action == 'realtime':
        print("Starting real-time recognition...")
        from src import realtime
        realtime.main()


if __name__ == "__main__":
    main()
