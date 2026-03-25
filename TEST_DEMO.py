#!/usr/bin/env python
"""
Script de test rapide du modèle Wav2Vec2
"""
import numpy as np
from src.wav2vec2_emotion import get_wav2vec2_classifier

print("🧪 Test du modèle Wav2Vec2")
print("=" * 60)

# Charger le modèle
print("\n1️⃣ Chargement du modèle...")
model = get_wav2vec2_classifier()
print("   ✅ Modèle chargé!")

# Test avec audio aléatoire
print("\n2️⃣ Test avec audio aléatoire (3 secondes)...")
dummy_audio = np.random.randn(16000 * 3)
emotion, probs = model.predict(dummy_audio, return_all=True)

print(f"\n🎭 Émotion prédite: {emotion.upper()}")
print("\n📊 Distribution:")
for emo, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
    bar = "█" * int(prob * 40)
    print(f"   {emo:10s} {prob:6.1%} {bar}")

print("\n✅ Test réussi! Le modèle fonctionne correctement.")
print("\n💡 Lancez l'interface: python app.py")
