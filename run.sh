#!/bin/bash
# Lanceur simplifié pour Speech Emotion Recognition Studio

echo "════════════════════════════════════════════════════════════════════"
echo "  🎭 Speech Emotion Recognition Studio"
echo "  Powered by Wav2Vec2 Transformer"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python n'est pas installé"
    exit 1
fi

echo "✅ Python trouvé: $(python --version)"

# Check dependencies
echo ""
echo "📦 Vérification des dépendances..."

MISSING=""
for pkg in torch transformers gradio plotly; do
    if ! python -c "import $pkg" 2>/dev/null; then
        MISSING="$MISSING $pkg"
    fi
done

if [ -n "$MISSING" ]; then
    echo "⚠️  Dépendances manquantes:$MISSING"
    echo "💡 Installation automatique..."
    pip install -q $MISSING
    if [ $? -ne 0 ]; then
        echo "❌ Erreur lors de l'installation"
        exit 1
    fi
    echo "✅ Dépendances installées!"
fi

echo "✅ Toutes les dépendances sont présentes"

# Launch app
echo ""
echo "🚀 Lancement de l'application..."
echo "   Interface: http://localhost:7860"
echo "   Appuyez sur Ctrl+C pour arrêter"
echo ""

python app.py
