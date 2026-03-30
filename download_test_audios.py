#!/usr/bin/env python3
"""
Script pour télécharger des audios de test depuis RAVDESS
"""

import os
import sys

print("=" * 70)
print("  🎧 Téléchargement d'Audios de Test")
print("=" * 70)

print("\n📋 Ce script télécharge 6 exemples audio (1 par émotion)")
print("   Source: RAVDESS Dataset (domaine public)")
print("   Taille: ~300 KB total")

# Créer dossier
os.makedirs("test_audios", exist_ok=True)

# Note: Les URLs RAVDESS réelles nécessitent un accès Zenodo
# Voici un exemple de structure

print("\n⚠️  IMPORTANT:")
print("   Les fichiers RAVDESS sont disponibles sur Zenodo:")
print("   https://zenodo.org/record/1188976")
print("\n📥 Pour télécharger:")
print("   1. Visitez le lien ci-dessus")
print("   2. Téléchargez le dataset (ou un sous-ensemble)")
print("   3. Extrayez les fichiers dans test_audios/")
print("\n💡 Alternative rapide:")
print("   Enregistrez-vous directement avec le micro de l'interface!")
print("   python app.py → Cliquer micro → Enregistrer")

print("\n" + "=" * 70)
print("  ℹ️  Utilisez plutôt le micro pour des tests rapides!")
print("=" * 70)

# Créer un fichier exemple avec instructions
with open("test_audios/README.txt", "w") as f:
    f.write("""📁 Dossier Test Audios

Pour tester votre système de reconnaissance d'émotions:

OPTION 1 (RECOMMANDÉ): Enregistrer avec le micro
  1. python app.py
  2. Cliquer sur le bouton micro 🎙️
  3. Enregistrer 3-5 secondes
  4. Observer les résultats

OPTION 2: Télécharger RAVDESS
  1. Visiter: https://zenodo.org/record/1188976
  2. Télécharger le dataset
  3. Placer des fichiers .wav ici
  4. Upload dans l'interface

OPTION 3: Utiliser vos propres fichiers
  1. Placer vos fichiers .wav ou .mp3 ici
  2. Upload dans l'interface

Consultez AUDIO_TEST_SOURCES.md pour plus de détails!
""")

print("\n✅ Dossier test_audios/ créé")
print("📄 Instructions dans test_audios/README.txt")
