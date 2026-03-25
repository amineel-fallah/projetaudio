# 🚀 Quick Start Guide

## Démarrage Ultra-Rapide (2 minutes)

### Étape 1: Installation
```bash
# Cloner le projet (si pas encore fait)
git clone <votre-repo>
cd projetaudio

# Installer Plotly (nouvelle dépendance)
pip install plotly
```

### Étape 2: Lancer l'application
```bash
# Option 1: Script automatique
./run.sh

# Option 2: Direct
python app.py
```

### Étape 3: Ouvrir l'interface
Ouvrez votre navigateur sur: **http://localhost:7860**

## 🎤 Premier Test

1. **Cliquez sur l'onglet "🎤 Analyse"**
2. **Cliquez sur le bouton micro** 🎙️
3. **Parlez avec émotion** pendant 3-5 secondes
4. **Observez les résultats** automatiquement !

## 📊 Comprendre les Résultats

### Carte Résultat Principale
- **Grand Emoji** : Émotion détectée
- **Nom en Couleur** : Label de l'émotion
- **Score de Confiance** : Certitude du modèle (0-100%)
- **Barres** : Distribution de toutes les émotions

### Visualisations
- **Waveform** : Forme d'onde de votre audio
- **Spectrogramme** : Analyse fréquentielle (couleurs = intensité)
- **Radar** : Vue circulaire de toutes les émotions
- **Jauge** : Niveau de confiance visuel

### Onglet Historique
- **Timeline** : Toutes vos prédictions passées
- **Émojis Colorés** : Chaque point = une prédiction
- **Rafraîchir** : Mettre à jour la vue
- **Effacer** : Réinitialiser l'historique

## 💡 Conseils pour Meilleurs Résultats

### ✅ À Faire
- Parler 3-5 secondes
- Environnement calme
- Exprimer clairement l'émotion
- Utiliser un micro correct

### ❌ À Éviter
- Bruit de fond important
- Audio trop court (<2s)
- Voix trop faible
- Micro saturé

## 🎯 Exemples d'Utilisation

### Test des 6 Émotions

1. **Neutre** 😐
   - Phrase: "Bonjour, comment allez-vous aujourd'hui ?"
   - Ton: Calme, monotone

2. **Heureux** 😊
   - Phrase: "C'est fantastique ! Je suis super content !"
   - Ton: Enjoué, souriant

3. **Triste** 😢
   - Phrase: "C'est vraiment dommage... je suis déçu"
   - Ton: Bas, mélancolique

4. **Colère** 😡
   - Phrase: "C'est inacceptable ! Je ne suis pas d'accord !"
   - Ton: Fort, tendu

5. **Peur** 😨
   - Phrase: "Oh non... j'ai peur que ça arrive..."
   - Ton: Tremblant, inquiet

6. **Surprise** 😲
   - Phrase: "Quoi ?! Vraiment ?! C'est incroyable !"
   - Ton: Exclamatif, étonné

## 🐛 Problèmes Courants

### L'interface ne se lance pas
```bash
# Vérifier Python
python --version  # Doit être 3.8+

# Réinstaller dépendances
pip install -r requirements.txt
pip install plotly
```

### Le modèle ne se charge pas
```bash
# Forcer le téléchargement
python src/wav2vec2_emotion.py
```

### Audio non détecté
- Vérifier permissions micro (navigateur)
- Essayer upload fichier à la place
- Formats supportés: .wav, .mp3, .flac

### Prédictions incorrectes
- Audio trop court ? (min 2s)
- Bruit de fond ?
- Émotion clairement exprimée ?

## 📈 Interpréter les Scores

### Confiance > 70%
✅ Excellente prédiction - Très fiable

### Confiance 50-70%
⚠️ Bonne prédiction - Assez fiable

### Confiance < 50%
❌ Prédiction incertaine - Plusieurs émotions possibles

## 🎓 Aller Plus Loin

### Tester avec Fichiers Audio
1. Préparez des fichiers .wav ou .mp3
2. Cliquez "Upload" au lieu de "Micro"
3. Sélectionnez votre fichier
4. Analyse instantanée !

### Explorer l'Historique
- Consultez vos patterns émotionnels
- Comparez différentes prises
- Identifiez les erreurs du modèle

### Lire la Documentation
- README.md : Documentation complète
- src/wav2vec2_emotion.py : Code du modèle
- app.py : Code de l'interface

## 🆘 Support

### Questions ?
1. Lire README.md (documentation complète)
2. Vérifier les issues GitHub
3. Ouvrir une nouvelle issue

### Bugs ?
Ouvrez une issue avec:
- Description du problème
- Étapes pour reproduire
- Message d'erreur (si applicable)
- Configuration système

---

**Prêt à démarrer ?** Lancez `./run.sh` ou `python app.py` ! 🚀
