# Speech Emotion Recognition using Deep Learning

## 🎯 Project Objectives

Speech Emotion Recognition (SER) system using deep learning to classify emotions from speech audio. This project classifies 6 core emotions:
- 😐 Neutral
- 😊 Happy
- 😢 Sad
- 😠 Angry
- 😨 Fearful
- 😲 Surprised

## 📁 Project Structure

```
projetaudio/
├── config.py              # Configuration parameters
├── requirements.txt       # Python dependencies
├── download_ravdess.py    # Dataset download script
├── train.py              # Model training script
├── app.py                # Gradio web interface
├── src/
│   ├── __init__.py
│   ├── features.py       # Audio feature extraction (MFCC, mel-spectrogram)
│   ├── model.py          # CNN-LSTM neural network architecture
│   └── dataset.py        # PyTorch dataset classes
├── data/
│   └── ravdess/          # RAVDESS dataset (to download)
├── models/               # Saved model checkpoints
└── logs/                 # Training logs
```

## 🛠️ Technologies Used

- **Python Libraries**: Librosa (MFCC/mel-spectrograms), PyTorch, scikit-learn
- **Deep Learning**: CNN-LSTM hybrid architecture
- **Deployment**: Gradio for web interface
- **Tracking**: MLflow for experiment logging

## 📊 Dataset

**Primary**: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- 24 professional actors (12 male, 12 female)
- 7,356 audio files
- 8 emotions at 2 emotional intensities

Download from: https://zenodo.org/record/1188976

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python download_ravdess.py
```

### 3. Train Model

```bash
python train.py
```

### 4. Launch Demo

```bash
python app.py
```

Then open http://localhost:7860 in your browser.

## 🏗️ Model Architecture

### CNN-LSTM Hybrid

1. **Feature Extraction**: 13 MFCCs + deltas, mel-spectrograms via Librosa
2. **CNN Layers**: Extract local spectral patterns from mel-spectrograms
3. **LSTM Layers**: Capture temporal dynamics across time frames
4. **Classification**: Fully connected layers with softmax output

### Training Parameters
- **Batch size**: 64
- **Learning rate**: 1e-4
- **Epochs**: 50
- **Loss**: Cross-entropy with class weights

## 📈 Expected Performance

- **Target**: >0.80 macro F1-score
- **Metrics**: Per-class F1-score, confusion matrix

## 🔧 Data Augmentation

- Adding background noise
- Pitch shifting
- Time stretching

## 📝 Features Extracted

| Feature | Description |
|---------|-------------|
| MFCC | 13 coefficients + delta + delta-delta |
| Mel-spectrogram | 128 mel bands |
| Chromagram | 12 pitch classes |

## 🎤 Real-time Pipeline

1. Audio input via microphone (sounddevice)
2. Feature extraction
3. Model inference
4. Emotion visualization

## 📚 References

- RAVDESS: Livingstone SR, Russo FA (2018)
- Librosa: McFee B. et al.
- PyTorch: Paszke A. et al.

## 👤 Author

Projet Audio - Speech Emotion Recognition

---

*Developed for educational purposes*
