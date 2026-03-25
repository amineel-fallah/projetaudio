#!/usr/bin/env python3
"""
Analyse la précision du modèle sur RAVDESS avec les vraies étiquettes
"""

import os
import sys
import librosa
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.wav2vec2_emotion import get_wav2vec2_classifier

# Mapping RAVDESS: 3ème chiffre dans le nom de fichier
RAVDESS_EMOTIONS = {
    '01': 'neutral',
    '02': 'neutral',  # calm → neutral
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'angry',    # disgust → angry (comme notre mapping)
    '08': 'surprised'
}

def parse_ravdess_filename(filename):
    """Extrait l'émotion du nom de fichier RAVDESS."""
    parts = filename.split('-')
    if len(parts) >= 3:
        emotion_code = parts[2]
        return RAVDESS_EMOTIONS.get(emotion_code, None)
    return None

def test_ravdess_accuracy(model, max_files=50):
    """Test la précision sur RAVDESS."""
    
    # Trouver les fichiers RAVDESS
    audio_files = []
    for root, dirs, files in os.walk('data'):
        for file in files:
            if file.endswith('.wav') and '-' in file:
                filepath = os.path.join(root, file)
                true_emotion = parse_ravdess_filename(file)
                if true_emotion:
                    audio_files.append((filepath, true_emotion, file))
    
    if not audio_files:
        print("❌ Aucun fichier RAVDESS trouvé!")
        return
    
    print(f"✅ {len(audio_files)} fichiers RAVDESS trouvés")
    
    # Échantillonner uniformément par émotion
    by_emotion = defaultdict(list)
    for filepath, emotion, filename in audio_files:
        by_emotion[emotion].append((filepath, emotion, filename))
    
    # Prendre N fichiers par émotion
    per_emotion = max_files // len(by_emotion)
    test_files = []
    for emotion, files in by_emotion.items():
        test_files.extend(files[:per_emotion])
    
    print(f"📊 Test sur {len(test_files)} fichiers balancés\n")
    
    # Tester
    correct = 0
    total = 0
    confusion = defaultdict(lambda: defaultdict(int))
    confidences = []
    
    for filepath, true_emotion, filename in test_files:
        try:
            audio, sr = librosa.load(filepath, sr=16000)
            pred_emotion, probs = model.predict(audio, sr, return_all=True)
            
            total += 1
            confidence = probs[pred_emotion]
            confidences.append(confidence)
            
            if pred_emotion == true_emotion:
                correct += 1
                status = "✅"
            else:
                status = "❌"
            
            confusion[true_emotion][pred_emotion] += 1
            
            if total <= 10:  # Afficher les 10 premiers
                print(f"{status} {filename[:30]:30s} Vrai: {true_emotion:10s} Prédit: {pred_emotion:10s} ({confidence:.0%})")
        
        except Exception as e:
            print(f"⚠️  Erreur: {filename}: {e}")
    
    # Résultats
    accuracy = correct / total * 100 if total > 0 else 0
    avg_confidence = np.mean(confidences) * 100 if confidences else 0
    
    print("\n" + "=" * 70)
    print("📊 RÉSULTATS")
    print("=" * 70)
    print(f"🎯 Précision: {correct}/{total} = {accuracy:.1f}%")
    print(f"📈 Confiance moyenne: {avg_confidence:.1f}%")
    
    # Matrice de confusion
    print("\n📉 Matrice de Confusion (Vrai → Prédit):")
    print("-" * 70)
    
    emotions = sorted(set(list(confusion.keys())))
    
    # Header
    print(f"{'Vrai ↓':12s}", end='')
    for pred in emotions:
        print(f"{pred:10s}", end='')
    print(f" {'Total':>6s} {'Précision':>10s}")
    
    # Lignes
    for true_emo in emotions:
        print(f"{true_emo:12s}", end='')
        row_total = sum(confusion[true_emo].values())
        row_correct = confusion[true_emo][true_emo]
        
        for pred_emo in emotions:
            count = confusion[true_emo][pred_emo]
            if count > 0:
                print(f"{count:10d}", end='')
            else:
                print(f"{'·':>10s}", end='')
        
        precision = row_correct / row_total * 100 if row_total > 0 else 0
        print(f" {row_total:6d} {precision:9.1f}%")
    
    print("\n💡 Analyse:")
    if accuracy < 30:
        print("   ⚠️  TRÈS FAIBLE (<30%) - Modèle inefficace")
    elif accuracy < 50:
        print("   ⚠️  FAIBLE (30-50%) - Besoin d'amélioration majeure")
    elif accuracy < 70:
        print("   ⚙️  MOYEN (50-70%) - Besoin de fine-tuning")
    elif accuracy < 85:
        print("   ✅ BON (70-85%) - Performance acceptable")
    else:
        print("   🌟 EXCELLENT (>85%) - Très bonne performance")
    
    if avg_confidence < 40:
        print("   ⚠️  Confiance trop faible - Modèle peu sûr de ses prédictions")
    elif avg_confidence < 60:
        print("   ⚙️  Confiance moyenne - Modèle hésitant")
    else:
        print("   ✅ Bonne confiance - Modèle sûr de ses prédictions")
    
    print("\n" + "=" * 70)

def main():
    print("=" * 70)
    print("🎭 ANALYSE DE PRÉCISION SUR RAVDESS")
    print("=" * 70)
    print()
    
    model = get_wav2vec2_classifier()
    test_ravdess_accuracy(model, max_files=60)

if __name__ == "__main__":
    main()
