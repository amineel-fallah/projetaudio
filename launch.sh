#!/bin/bash

echo "================================"
echo "🎭 EMOTION RECOGNITION - QUICK START"
echo "================================"
echo ""

# Vérifier si le modèle existe
if [ -f "models/best_advanced_model.pth" ]; then
    echo "✅ Model found! Launching interface..."
    python app_exceptional.py
else
    echo "⚠️  No trained model found"
    echo ""
    echo "Choose an option:"
    echo "1) Launch demo with untrained model (random predictions)"
    echo "2) Train model first (recommended, takes 2-3 hours)"
    echo "3) Use basic CNN-LSTM model (37% accuracy)"
    echo ""
    read -p "Your choice (1/2/3): " choice
    
    case $choice in
        1)
            echo "🚀 Launching demo with untrained model..."
            python app_exceptional.py
            ;;
        2)
            echo "🔥 Starting training..."
            python train_advanced.py
            echo ""
            echo "✅ Training complete! Launching interface..."
            python app_exceptional.py
            ;;
        3)
            echo "🚀 Launching with basic CNN-LSTM..."
            python app.py
            ;;
        *)
            echo "❌ Invalid choice"
            exit 1
            ;;
    esac
fi
