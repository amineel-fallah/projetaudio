#!/bin/bash
# Script pour télécharger TESS (Toronto Emotional Speech Set)

echo "════════════════════════════════════════════════════════════════"
echo "  📥 Téléchargement Dataset TESS"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Dataset: Toronto Emotional Speech Set"
echo "Taille: ~500 MB"
echo "Fichiers: 2,800 audio clips"
echo "Émotions: 7 (angry, disgust, fear, happy, neutral, surprise, sad)"
echo ""

# Vérifier si déjà téléchargé
if [ -d "test_audios/TESS" ]; then
    echo "⚠️  TESS semble déjà téléchargé dans test_audios/TESS/"
    read -p "Télécharger à nouveau? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Annulé."
        exit 0
    fi
fi

# Créer dossier
mkdir -p test_audios/TESS
cd test_audios/TESS

# URL TESS
URL="https://tspace.library.utoronto.ca/bitstream/1807/24487/1/TESS_Toronto_emotional_speech_set_data.zip"

echo "📥 Téléchargement depuis University of Toronto..."
echo "   (Cela peut prendre quelques minutes)"
echo ""

# Télécharger avec wget ou curl
if command -v wget &> /dev/null; then
    wget -O TESS.zip "$URL"
elif command -v curl &> /dev/null; then
    curl -L -o TESS.zip "$URL"
else
    echo "❌ wget ou curl non trouvé!"
    echo "💡 Installer wget: sudo apt-get install wget"
    exit 1
fi

# Vérifier téléchargement
if [ ! -f "TESS.zip" ]; then
    echo "❌ Erreur de téléchargement!"
    exit 1
fi

echo ""
echo "✅ Téléchargement terminé!"
echo "📦 Extraction des fichiers..."

# Extraire
unzip -q TESS.zip

# Vérifier extraction
if [ $? -eq 0 ]; then
    echo "✅ Extraction réussie!"
    
    # Compter fichiers
    file_count=$(find . -name "*.wav" | wc -l)
    echo "📊 $file_count fichiers WAV trouvés"
    
    # Nettoyer
    rm TESS.zip
    echo "🧹 Fichier ZIP supprimé"
    
    cd ../..
    
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  ✅ TESS Prêt!"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "📁 Emplacement: test_audios/TESS/"
    echo "📊 Fichiers: $file_count audio clips"
    echo ""
    echo "🚀 Tester maintenant:"
    echo "   1. python app.py"
    echo "   2. Upload des fichiers depuis test_audios/TESS/"
    echo ""
else
    echo "❌ Erreur d'extraction!"
    exit 1
fi
