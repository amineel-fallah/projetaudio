# 🗂️ Datasets Alternatifs à RAVDESS

## 🎯 Top 5 Datasets Faciles d'Accès

### 1. 🌟 **TESS** (Toronto Emotional Speech Set)
**⭐ RECOMMANDÉ - Le Plus Facile !**

- **Accès**: ✅ Téléchargement direct gratuit
- **Taille**: 500 MB (bien plus petit que RAVDESS!)
- **Fichiers**: 2,800 audio
- **Émotions**: 7 (angry, disgust, fear, happy, neutral, pleasant surprise, sad)
- **Actrices**: 2 (une jeune, une âgée)
- **Format**: WAV, haute qualité
- **Lien**: https://tspace.library.utoronto.ca/handle/1807/24487

#### Comment Télécharger TESS
```bash
# Option 1: Lien direct
wget https://tspace.library.utoronto.ca/bitstream/1807/24487/1/TESS_Toronto_emotional_speech_set_data.zip

# Option 2: Via navigateur
# Aller sur le lien et cliquer "Download"
```

#### Structure TESS
```
TESS/
├── OAF_angry/     # Old Actress - Angry
├── OAF_disgust/
├── OAF_fear/
├── OAF_happy/
├── OAF_neutral/
├── OAF_ps/        # Pleasant Surprise
├── OAF_sad/
├── YAF_angry/     # Young Actress - Angry
├── YAF_disgust/
└── ...
```

---

### 2. 🎤 **CREMA-D**
**Très Populaire et Varié**

- **Accès**: ✅ Gratuit, GitHub
- **Taille**: 2 GB
- **Fichiers**: 7,442 clips
- **Émotions**: 6 (anger, disgust, fear, happy, neutral, sad)
- **Acteurs**: 91 (48 hommes, 43 femmes, multi-ethnique)
- **Format**: WAV, 16 kHz
- **Lien**: https://github.com/CheyneyComputerScience/CREMA-D

#### Comment Télécharger CREMA-D
```bash
# Clone le repo
git clone https://github.com/CheyneyComputerScience/CREMA-D.git

# Ou télécharger le ZIP
wget https://github.com/CheyneyComputerScience/CREMA-D/archive/refs/heads/master.zip
```

#### Avantages CREMA-D
- ✅ Diversité ethnique
- ✅ Nombreux acteurs
- ✅ Clips courts (2-3 secondes)
- ✅ Annotations détaillées

---

### 3. 📚 **EmoDB** (Berlin Database)
**Dataset Académique Classique**

- **Accès**: ✅ Gratuit
- **Taille**: 150 MB (très petit!)
- **Fichiers**: 535 utterances
- **Émotions**: 7 (anger, boredom, disgust, fear, happiness, sadness, neutral)
- **Acteurs**: 10 (5 hommes, 5 femmes)
- **Langue**: Allemand (mais émotions universelles!)
- **Format**: WAV, 16 kHz
- **Lien**: http://emodb.bilderbar.info/

#### Comment Télécharger EmoDB
```bash
# Téléchargement direct
wget http://emodb.bilderbar.info/download/download.zip

# Extraire
unzip download.zip
```

#### Note EmoDB
- Langue allemande mais **émotions universelles**
- Bon pour tester la robustesse du modèle
- Très utilisé dans la recherche

---

### 4. 🎭 **SAVEE** (Surrey Audio-Visual Expressed Emotion)
**Haute Qualité Audio**

- **Accès**: ✅ Gratuit (inscription simple)
- **Taille**: 400 MB
- **Fichiers**: 480 utterances
- **Émotions**: 7 (anger, disgust, fear, happiness, neutral, sadness, surprise)
- **Acteurs**: 4 hommes britanniques
- **Format**: WAV, haute qualité
- **Lien**: http://kahlan.eps.surrey.ac.uk/savee/

#### Comment Télécharger SAVEE
1. Visiter le site
2. Remplir formulaire simple (nom, email)
3. Recevoir lien de téléchargement
4. Télécharger le ZIP

---

### 5. 🌍 **JL-Corpus**
**Multilingue!**

- **Accès**: ✅ Gratuit, Zenodo
- **Taille**: 100 MB
- **Fichiers**: 2,400 phrases
- **Langues**: 4 (anglais, français, allemand, chinois)
- **Émotions**: 5 (angry, neutral, happy, sad, scared)
- **Acteurs**: 4 par langue
- **Format**: WAV
- **Lien**: https://zenodo.org/record/3561618

#### Comment Télécharger JL-Corpus
```bash
# Via zenodo_get
pip install zenodo_get
zenodo_get 3561618

# Ou téléchargement direct depuis le site
```

---

## 🚀 Script de Téléchargement Automatique

### Pour TESS (Le Plus Simple)

```python
#!/usr/bin/env python3
"""
Télécharger et préparer TESS pour test
"""
import os
import urllib.request
import zipfile

print("📥 Téléchargement de TESS...")

# URL TESS
url = "https://tspace.library.utoronto.ca/bitstream/1807/24487/1/TESS_Toronto_emotional_speech_set_data.zip"
zip_path = "TESS.zip"

# Télécharger
print("⏳ Téléchargement en cours... (500 MB)")
urllib.request.urlretrieve(url, zip_path)
print("✅ Téléchargement terminé!")

# Extraire
print("📦 Extraction...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("test_audios/TESS")
print("✅ Extraction terminée!")

# Nettoyage
os.remove(zip_path)
print("🧹 Nettoyage terminé!")

print("\n✅ TESS prêt dans: test_audios/TESS/")
print("📁 2,800 fichiers audio disponibles!")
```

---

## 📊 Comparaison Rapide

| Dataset | Taille | Fichiers | Acteurs | Accès | Qualité |
|---------|--------|----------|---------|-------|---------|
| **RAVDESS** | 5 GB | 7,356 | 24 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **TESS** ✨ | 500 MB | 2,800 | 2 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **CREMA-D** | 2 GB | 7,442 | 91 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **EmoDB** | 150 MB | 535 | 10 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **SAVEE** | 400 MB | 480 | 4 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **JL-Corpus** | 100 MB | 2,400 | 16 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

**Légende Accès**:
- ⭐⭐⭐⭐⭐ = Téléchargement direct
- ⭐⭐⭐⭐ = Inscription simple
- ⭐⭐⭐ = Procédure requise

---

## 🎯 Notre Recommandation

### Pour Tests Rapides
**→ TESS** (500 MB, facile d'accès, bonne qualité)

### Pour Tests Approfondis
**→ CREMA-D** (plus de diversité, nombreux acteurs)

### Pour Tests Multilingues
**→ JL-Corpus** (4 langues, compact)

---

## 📥 Script Complet de Téléchargement

```bash
#!/bin/bash
# download_alternative_datasets.sh

echo "🎧 Téléchargement Datasets Alternatifs"
echo "======================================"

# Créer dossier
mkdir -p test_audios/datasets

# Option 1: TESS (Recommandé)
read -p "Télécharger TESS? (500MB) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📥 Téléchargement TESS..."
    cd test_audios/datasets
    wget https://tspace.library.utoronto.ca/bitstream/1807/24487/1/TESS_Toronto_emotional_speech_set_data.zip
    unzip TESS_Toronto_emotional_speech_set_data.zip
    rm TESS_Toronto_emotional_speech_set_data.zip
    cd ../..
    echo "✅ TESS prêt!"
fi

# Option 2: CREMA-D
read -p "Télécharger CREMA-D? (2GB) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📥 Téléchargement CREMA-D..."
    cd test_audios/datasets
    git clone https://github.com/CheyneyComputerScience/CREMA-D.git
    cd ../..
    echo "✅ CREMA-D prêt!"
fi

# Option 3: EmoDB
read -p "Télécharger EmoDB? (150MB) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📥 Téléchargement EmoDB..."
    cd test_audios/datasets
    wget http://emodb.bilderbar.info/download/download.zip
    unzip download.zip -d EmoDB
    rm download.zip
    cd ../..
    echo "✅ EmoDB prêt!"
fi

echo ""
echo "✅ Téléchargements terminés!"
echo "📁 Fichiers dans: test_audios/datasets/"
```

---

## 🧪 Tester avec les Nouveaux Datasets

### Script de Test Automatique

```python
#!/usr/bin/env python3
"""
Tester votre modèle sur différents datasets
"""
import os
import glob
from src.wav2vec2_emotion import get_wav2vec2_classifier
from src.features import load_audio

# Charger modèle
print("🚀 Chargement du modèle...")
model = get_wav2vec2_classifier()

# Mapping émotions TESS → Notre modèle
emotion_map = {
    'angry': 'angry',
    'disgust': 'angry',  # On mappe disgust vers angry
    'fear': 'fearful',
    'happy': 'happy',
    'neutral': 'neutral',
    'ps': 'surprised',  # Pleasant surprise
    'sad': 'sad'
}

# Tester TESS
tess_path = "test_audios/TESS/"
if os.path.exists(tess_path):
    print("\n📊 Test sur TESS...")
    
    results = {'correct': 0, 'total': 0}
    
    for emotion_folder in glob.glob(f"{tess_path}*/"):
        emotion_name = os.path.basename(emotion_folder.rstrip('/')).split('_')[-1].lower()
        
        if emotion_name in emotion_map:
            expected = emotion_map[emotion_name]
            
            # Tester 10 fichiers de cette émotion
            files = glob.glob(f"{emotion_folder}*.wav")[:10]
            
            for file_path in files:
                try:
                    audio, sr = load_audio(file_path)
                    predicted, _ = model.predict(audio, sr, return_all=True)
                    
                    if predicted == expected:
                        results['correct'] += 1
                    results['total'] += 1
                    
                except Exception as e:
                    print(f"Erreur: {file_path}")
    
    accuracy = results['correct'] / results['total'] * 100
    print(f"\n✅ Précision TESS: {accuracy:.1f}% ({results['correct']}/{results['total']})")

else:
    print("⚠️  TESS non trouvé. Télécharger d'abord!")
```

---

## 📚 Ressources Supplémentaires

### Autres Datasets (Plus Complexes)

1. **IEMOCAP** (Interactif)
   - Lien: https://sail.usc.edu/iemocap/
   - Nécessite inscription académique
   - 12 heures d'audio
   - Très utilisé en recherche

2. **MSP-IMPROV** (Improvisé)
   - Lien: https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Improv.html
   - Émotions naturelles
   - Conversations improvisées

3. **MELD** (Multimodal)
   - Lien: https://affective-meld.github.io/
   - Audio + Vidéo + Texte
   - Dialogues TV (Friends, etc.)

---

## 💡 Conseils d'Utilisation

### Quel Dataset Choisir?

**Pour débuter**: 
→ **TESS** (petit, facile, qualité)

**Pour comparer**:
→ **CREMA-D** (diversité)

**Pour challenge**:
→ **EmoDB** (autre langue)

**Pour production**:
→ **RAVDESS** + **TESS** + **CREMA-D** (ensemble)

---

## 🎯 Action Immédiate

### Télécharger TESS Maintenant

```bash
# Créer script
cat > download_tess.sh << 'EOF'
#!/bin/bash
mkdir -p test_audios/TESS
cd test_audios/TESS
echo "📥 Téléchargement TESS (500 MB)..."
wget https://tspace.library.utoronto.ca/bitstream/1807/24487/1/TESS_Toronto_emotional_speech_set_data.zip
echo "📦 Extraction..."
unzip TESS_Toronto_emotional_speech_set_data.zip
rm TESS_Toronto_emotional_speech_set_data.zip
echo "✅ TESS prêt! 2,800 fichiers disponibles."
cd ../..
EOF

# Lancer
chmod +x download_tess.sh
./download_tess.sh
```

---

**Avec TESS, vous avez 2,800 audios de qualité professionnelle pour tester ! 🎉**
