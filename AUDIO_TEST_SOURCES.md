# 🎧 Guide des Audios de Test

## 🎯 Sources d'Audio pour Tester Votre Projet

Ce guide vous propose plusieurs façons d'obtenir des audios de test pour évaluer l'efficacité de votre système de reconnaissance d'émotions.

---

## 🎤 Option 1: Enregistrer Vous-Même (RECOMMANDÉ)

### Phrases à Enregistrer

#### 😐 **NEUTRAL** (Neutre)
```
1. "Bonjour, comment allez-vous aujourd'hui ?"
2. "Je vais à la bibliothèque cet après-midi."
3. "Il est treize heures et quinze minutes."
4. "Le temps est nuageux ce matin."
5. "Je dois terminer ce rapport."
```
**Conseil**: Parlez de manière monotone, sans émotion particulière.

#### 😊 **HAPPY** (Heureux)
```
1. "C'est fantastique ! Je suis super content !"
2. "J'ai réussi mon examen ! Quelle joie !"
3. "Nous partons en vacances demain ! Génial !"
4. "J'adore cette musique, elle est magnifique !"
5. "Quel beau cadeau, merci beaucoup !"
```
**Conseil**: Souriez en parlant, voix enjouée et dynamique.

#### 😢 **SAD** (Triste)
```
1. "C'est vraiment dommage... je suis déçu."
2. "Je me sens seul aujourd'hui."
3. "Cette nouvelle m'a vraiment attristé."
4. "Je n'arrive pas à surmonter cette épreuve."
5. "Tout semble si difficile en ce moment."
```
**Conseil**: Voix basse, lente, sans énergie.

#### 😡 **ANGRY** (Colère)
```
1. "C'est inacceptable ! Je ne suis pas d'accord !"
2. "J'en ai assez de cette situation !"
3. "Comment osez-vous me parler ainsi ?!"
4. "Cela me met hors de moi !"
5. "Je refuse catégoriquement cette décision !"
```
**Conseil**: Voix forte, tendue, articulation marquée.

#### 😨 **FEARFUL** (Peur)
```
1. "Oh non... j'ai peur que ça arrive..."
2. "Je ne sais pas si je vais y arriver..."
3. "Cela m'inquiète beaucoup, vraiment beaucoup."
4. "J'ai un mauvais pressentiment à ce sujet."
5. "Je tremble rien que d'y penser."
```
**Conseil**: Voix tremblante, hésitante, anxieuse.

#### 😲 **SURPRISED** (Surprise)
```
1. "Quoi ?! Vraiment ?! C'est incroyable !"
2. "Je n'en crois pas mes oreilles !"
3. "Wow ! Je ne m'attendais pas à ça !"
4. "C'est une surprise totale !"
5. "Ça alors ! Quelle nouvelle !"
```
**Conseil**: Voix haute, exclamative, dynamique.

### Comment Enregistrer

**Méthode 1: Directement dans l'interface**
1. Lancer `python app.py`
2. Cliquer sur le micro 🎙️
3. Enregistrer une phrase (3-5 secondes)
4. Observer le résultat

**Méthode 2: Avec Audacity (gratuit)**
1. Télécharger Audacity: https://www.audacityteam.org/
2. Enregistrer vos phrases
3. Exporter en WAV ou MP3
4. Upload dans l'interface

---

## 📥 Option 2: Télécharger des Datasets Publics

### 1. RAVDESS (Recommandé)
**Dataset utilisé pour l'entraînement !**

- **URL**: https://zenodo.org/record/1188976
- **Contenu**: 7,356 fichiers audio
- **Acteurs**: 24 professionnels (12 H, 12 F)
- **Émotions**: 8 (incluant les 6 de votre modèle)
- **Format**: WAV, 16-bit, 48 kHz
- **Taille**: ~5 GB

**Structure des noms**:
```
03-01-01-01-01-01-01.wav
│  │  │  │  │  │  └─ Acteur (01-24)
│  │  │  │  │  └─ Répétition (01-02)
│  │  │  │  └─ Intensité (01=normal, 02=fort)
│  │  │  └─ Statement ("kids"=01, "dogs"=02)
│  │  └─ Émotion (01=neutral, 03=happy, 04=sad, 05=angry, 06=fearful, 08=surprised)
│  └─ Canal vocal (01=speech, 02=song)
└─ Modalité (03=audio-video)
```

**Télécharger un sous-ensemble**:
```bash
# Installer zenodo_get
pip install zenodo_get

# Télécharger RAVDESS
zenodo_get 1188976
```

### 2. TESS (Toronto Emotional Speech Set)
- **URL**: https://tspace.library.utoronto.ca/handle/1807/24487
- **Contenu**: 2,800 fichiers
- **Actrices**: 2 (jeune et âgée)
- **Émotions**: 7 émotions
- **Format**: WAV
- **Taille**: ~500 MB

### 3. CREMA-D
- **URL**: https://github.com/CheyneyComputerScience/CREMA-D
- **Contenu**: 7,442 clips
- **Acteurs**: 91 (48 H, 43 F)
- **Émotions**: 6 émotions
- **Taille**: ~2 GB

### 4. EmoDB (Allemand, mais utile)
- **URL**: http://emodb.bilderbar.info/
- **Contenu**: 535 fichiers
- **Langue**: Allemand
- **Émotions**: 7 émotions
- **Gratuit**: Oui

---

## 🌐 Option 3: Audios Gratuits en Ligne

### Sites de Sons Gratuits

1. **Freesound.org**
   - URL: https://freesound.org/
   - Rechercher: "emotional speech", "angry voice", "happy speech"
   - Licence: Creative Commons (vérifier)

2. **Pixabay Audio**
   - URL: https://pixabay.com/sound-effects/search/voice/
   - Rechercher: émotions vocales
   - Licence: Gratuit

3. **BBC Sound Effects**
   - URL: https://sound-effects.bbcrewind.co.uk/
   - Rechercher: "human voice"
   - Licence: Usage éducatif

### YouTube to Audio

1. Chercher sur YouTube: "emotional speech examples"
2. Utiliser un convertisseur YouTube → MP3 (online-convert.com)
3. Télécharger et tester
4. **Attention**: Respecter les droits d'auteur !

---

## 🤖 Option 4: Générer des Audios Synthétiques

### Avec Google Text-to-Speech

```python
from gtts import gTTS
import os

emotions = {
    "neutral": "Bonjour comment allez vous",
    "happy": "C'est fantastique je suis ravi !",
    "sad": "Je suis triste aujourd'hui...",
}

for emotion, text in emotions.items():
    tts = gTTS(text=text, lang='fr', slow=False)
    tts.save(f"test_{emotion}.mp3")
    print(f"✅ {emotion}.mp3 créé")
```

**Installation**:
```bash
pip install gtts
```

### Avec pyttsx3 (local)

```python
import pyttsx3

engine = pyttsx3.init()

# Ajuster la voix
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)  # Voix française

# Générer
phrases = {
    "neutral": "Bonjour comment allez vous",
    "happy": "C'est fantastique !",
}

for emotion, text in phrases.items():
    engine.save_to_file(text, f'test_{emotion}.wav')
    engine.runAndWait()
```

**Note**: Les synthèses vocales ne sont PAS idéales pour tester les émotions (manque d'expressivité naturelle).

---

## 📁 Option 5: Fichiers de Test Inclus

### Télécharger depuis le Dataset RAVDESS

**Script de téléchargement rapide**:

```python
import requests
import os

# URLs d'exemples RAVDESS (domaine public)
samples = {
    "neutral": "https://zenodo.org/record/1188976/files/Actor_01/03-01-01-01-01-01-01.wav",
    "happy": "https://zenodo.org/record/1188976/files/Actor_01/03-01-03-01-01-01-01.wav",
    "sad": "https://zenodo.org/record/1188976/files/Actor_01/03-01-04-01-01-01-01.wav",
    "angry": "https://zenodo.org/record/1188976/files/Actor_01/03-01-05-01-01-01-01.wav",
    "fearful": "https://zenodo.org/record/1188976/files/Actor_01/03-01-06-01-01-01-01.wav",
    "surprised": "https://zenodo.org/record/1188976/files/Actor_01/03-01-08-01-01-01-01.wav",
}

os.makedirs("test_audios", exist_ok=True)

for emotion, url in samples.items():
    print(f"📥 Téléchargement {emotion}...")
    r = requests.get(url)
    if r.status_code == 200:
        with open(f"test_audios/{emotion}.wav", "wb") as f:
            f.write(r.content)
        print(f"   ✅ {emotion}.wav sauvegardé")
    else:
        print(f"   ❌ Erreur téléchargement")

print("\n✅ Tous les fichiers téléchargés dans test_audios/")
```

---

## 🎯 Stratégie de Test Recommandée

### Phase 1: Test Rapide (5 min)
1. **Enregistrer vous-même** 2 phrases par émotion (12 total)
2. Tester dans l'interface
3. Noter la précision globale

### Phase 2: Test Approfondi (30 min)
1. Télécharger **10-20 fichiers RAVDESS**
2. Tester chaque fichier
3. Comparer avec labels réels
4. Calculer accuracy

### Phase 3: Test Robustesse (1h)
1. Tester avec **différentes voix** (homme/femme/enfant)
2. Tester avec **différents accents**
3. Tester avec **bruit de fond**
4. Tester avec **volumes variés**

---

## 📊 Évaluer la Précision

### Script d'Évaluation Automatique

```python
import os
from src.wav2vec2_emotion import get_wav2vec2_classifier
from src.features import load_audio

# Charger modèle
model = get_wav2vec2_classifier()

# Définir ground truth (labels réels)
test_files = {
    "test_audios/neutral.wav": "neutral",
    "test_audios/happy.wav": "happy",
    "test_audios/sad.wav": "sad",
    "test_audios/angry.wav": "angry",
    "test_audios/fearful.wav": "fearful",
    "test_audios/surprised.wav": "surprised",
}

correct = 0
total = len(test_files)

for file_path, true_emotion in test_files.items():
    if os.path.exists(file_path):
        audio, sr = load_audio(file_path)
        pred_emotion, probs = model.predict(audio, sr, return_all=True)
        
        is_correct = pred_emotion == true_emotion
        correct += int(is_correct)
        
        symbol = "✅" if is_correct else "❌"
        print(f"{symbol} {file_path}")
        print(f"   Prédit: {pred_emotion} (conf: {probs[pred_emotion]:.1%})")
        print(f"   Réel: {true_emotion}")

accuracy = correct / total * 100
print(f"\n📊 Précision: {accuracy:.1f}% ({correct}/{total})")
```

---

## 💡 Conseils pour de Bons Tests

### ✅ À Faire
- Tester **minimum 10 fichiers par émotion**
- Varier les **voix** (homme/femme)
- Tester dans **conditions réelles** (bruit, qualité micro)
- **Documenter** les erreurs

### ❌ À Éviter
- Tester seulement 1-2 fichiers
- Utiliser uniquement voix synthétique
- Ignorer les erreurs
- Ne pas comparer avec ground truth

---

## 🎬 Exemples de Résultats Attendus

### Bon Résultat
```
🎭 Émotion détectée: HAPPY
📊 Distribution:
   happy      87% ████████████████████████████████████
   surprised  8%  ████
   neutral    3%  █
```

### Résultat Moyen
```
🎭 Émotion détectée: SAD
📊 Distribution:
   sad        62% █████████████████████████
   neutral    28% ████████████
   fearful    10% ████
```

### Résultat Faible (à investiguer)
```
🎭 Émotion détectée: NEUTRAL
📊 Distribution:
   neutral    35% ██████████████
   angry      33% █████████████
   sad        32% █████████████
```

---

## 📚 Ressources Supplémentaires

### Datasets Complets
1. **IEMOCAP**: https://sail.usc.edu/iemocap/ (inscription requise)
2. **MSP-IMPROV**: https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Improv.html
3. **SAVEE**: http://kahlan.eps.surrey.ac.uk/savee/

### Articles de Référence
1. "RAVDESS: The Ryerson Audio-Visual Database of Emotional Speech and Song"
2. "Speech Emotion Recognition: A Comprehensive Survey"

---

## 🆘 Aide

Si vous avez besoin d'aide pour télécharger ou préparer des audios:
- Consultez `README_NEW.md`
- Relancez `python app.py` et testez avec le micro
- Contactez-moi pour assistance

---

**Bon Testing ! 🚀**
