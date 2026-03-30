#!/usr/bin/env python3
"""
Test le modèle Wav2Vec2 avec de vrais fichiers audio
"""

import os
import sys
import librosa
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.wav2vec2_emotion import get_wav2vec2_classifier

def test_audio_file(model, filepath):
    """Test un fichier audio et affiche les résultats."""
    try:
        # Charger l'audio
        audio, sr = librosa.load(filepath, sr=16000)
        
        # Prédire
        emotion, probs = model.predict(audio, sr, return_all=True)
        
        print(f"\n📁 Fichier: {Path(filepath).name}")
        print(f"   🎯 Émotion détectée: {emotion.upper()}")
        print(f"   📊 Distribution:")
        
        for emo, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 40)
            print(f"      {emo:12s} {prob:6.1%} {bar}")
        
        return emotion, probs
        
    except Exception as e:
        print(f"\n❌ Erreur avec {filepath}: {e}")
        return None, None

def main():
    print("=" * 70)
    print("🎭 TEST DU MODÈLE WAV2VEC2 AVEC VRAIS FICHIERS AUDIO")
    print("=" * 70)
    
    # Charger le modèle
    print("\n🚀 Chargement du modèle...")
    model = get_wav2vec2_classifier()
    
    # Chercher des fichiers audio
    audio_dirs = ['test_audios', 'data/RAVDESS', 'data']
    audio_files = []
    
    for directory in audio_dirs:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith(('.wav', '.mp3', '.flac')):
                        audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        print("\n⚠️  Aucun fichier audio trouvé!")
        print("   Créez des audios avec:")
        print("   1. Gradio interface (lancez app.py)")
        print("   2. Téléchargez RAVDESS: python download_ravdess.py")
        print("   3. Téléchargez TESS: bash download_tess.sh")
        return
    
    print(f"\n✅ {len(audio_files)} fichier(s) audio trouvé(s)")
    
    # Limiter à 10 fichiers pour le test
    test_files = audio_files[:10]
    
    # Tester chaque fichier
    results = {}
    for filepath in test_files:
        emotion, probs = test_audio_file(model, filepath)
        if emotion:
            results[filepath] = emotion
    
    # Résumé
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ DES RÉSULTATS")
    print("=" * 70)
    
    if results:
        from collections import Counter
        emotion_counts = Counter(results.values())
        
        print("\n🎯 Distribution des émotions détectées:")
        for emo, count in emotion_counts.most_common():
            pct = count / len(results) * 100
            bar = "█" * int(pct / 2)
            print(f"   {emo:12s} {count:2d} ({pct:5.1f}%) {bar}")
        
        if len(set(results.values())) == 1:
            print("\n⚠️  PROBLÈME: Toutes les détections donnent la même émotion!")
            print("   Le modèle ne distingue pas les émotions correctement.")
        else:
            print("\n✅ Le modèle détecte différentes émotions!")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
