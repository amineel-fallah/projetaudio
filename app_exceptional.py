"""
INTERFACE WEB EXCEPTIONNELLE
Design moderne avec glassmorphism, animations fluides, et visualisations avancées
"""

import gradio as gr
import numpy as np
import torch
import librosa
import plotly.graph_objects as go
from scipy import signal as scipy_signal
import os
from datetime import datetime

# Import models
from src.model_advanced import AdvancedCNNLSTM
from src.features import extract_mel_spectrogram
from config import *

# Emojis pour chaque émotion
EMOTION_EMOJIS = {
    'neutral': '😐',
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'fearful': '😨',
    'surprised': '😲'
}

# Couleurs pour chaque émotion
EMOTION_COLORS = {
    'neutral': '#95a5a6',
    'happy': '#f39c12',
    'sad': '#3498db',
    'angry': '#e74c3c',
    'fearful': '#9b59b6',
    'surprised': '#1abc9c'
}

class EmotionPredictor:
    """Prédicteur d'émotions avec modèle avancé."""
    
    def __init__(self, model_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Loading advanced model on {self.device}...")
        
        self.model = AdvancedCNNLSTM(num_classes=len(EMOTION_LABELS)).to(self.device)
        
        # Charger les poids si disponibles
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model loaded from {model_path}")
        else:
            print("⚠️  No trained weights found - using random initialization")
            print("   Train the model first with: python train_advanced.py")
        
        self.model.eval()
        self.history = []
    
    def predict(self, audio, sample_rate):
        """Prédire l'émotion."""
        try:
            # Resample si nécessaire
            if sample_rate != SAMPLE_RATE:
                num_samples = int(len(audio) * SAMPLE_RATE / sample_rate)
                audio = scipy_signal.resample(audio, num_samples)
                sample_rate = SAMPLE_RATE
            
            # Extraire mel-spectrogram
            mel_spec = extract_mel_spectrogram(audio, sample_rate)
            
            # Préparer pour le modèle
            mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Prédiction
            with torch.no_grad():
                logits = self.model(mel_tensor)
                probs = torch.softmax(logits, dim=1)[0]
            
            # Convertir en dict
            emotion_probs = {
                emotion: float(probs[i])
                for i, emotion in enumerate(EMOTION_LABELS)
            }
            
            # Émotion dominante
            top_emotion = max(emotion_probs.items(), key=lambda x: x[1])[0]
            
            # Ajouter à l'historique
            self.history.append({
                'emotion': top_emotion,
                'timestamp': datetime.now(),
                'probs': emotion_probs
            })
            
            # Garder les 10 derniers
            if len(self.history) > 10:
                self.history = self.history[-10:]
            
            return top_emotion, emotion_probs, mel_spec
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            import traceback
            traceback.print_exc()
            raise


# Initialiser le prédicteur
PREDICTOR = EmotionPredictor(model_path='models/best_advanced_model.pth')


def create_waveform_plot(audio, sr):
    """Crée un graphique de forme d'onde animé."""
    time = np.linspace(0, len(audio) / sr, len(audio))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time,
        y=audio,
        mode='lines',
        line=dict(color='#3498db', width=1),
        fill='tozeroy',
        fillcolor='rgba(52, 152, 219, 0.3)',
        name='Audio'
    ))
    
    fig.update_layout(
        title="🎵 Forme d'onde",
        xaxis_title="Temps (s)",
        yaxis_title="Amplitude",
        template="plotly_dark",
        height=250,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.2)'
    )
    
    return fig


def create_spectrogram_plot(mel_spec):
    """Crée un spectrogramme coloré."""
    fig = go.Figure(data=go.Heatmap(
        z=mel_spec,
        colorscale='Viridis',
        colorbar=dict(title='dB')
    ))
    
    fig.update_layout(
        title="🎨 Spectrogramme Mel",
        xaxis_title="Frames temporelles",
        yaxis_title="Bandes Mel",
        template="plotly_dark",
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.2)'
    )
    
    return fig


def create_radar_chart(probs):
    """Crée un graphique radar des émotions."""
    emotions = list(probs.keys())
    values = [probs[e] * 100 for e in emotions]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Fermer le polygone
        theta=emotions + [emotions[0]],
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.4)',
        line=dict(color='#3498db', width=2),
        name='Émotions'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix='%'
            ),
            bgcolor='rgba(0,0,0,0.2)'
        ),
        showlegend=False,
        title="📊 Distribution des Émotions",
        template="plotly_dark",
        height=400,
        margin=dict(l=80, r=80, t=80, b=80),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_confidence_gauge(confidence):
    """Crée une jauge de confiance."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        title={'text': "🎯 Confiance"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#3498db"},
            'steps': [
                {'range': [0, 33], 'color': "rgba(231, 76, 60, 0.3)"},
                {'range': [33, 66], 'color': "rgba(241, 196, 15, 0.3)"},
                {'range': [66, 100], 'color': "rgba(46, 204, 113, 0.3)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=50, r=50, t=80, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig


def create_timeline(history):
    """Crée une timeline d'historique."""
    if not history:
        return "Aucun historique"
    
    timeline_html = "<div style='display: flex; gap: 10px; flex-wrap: wrap; padding: 10px;'>"
    
    for item in reversed(history[-10:]):
        emoji = EMOTION_EMOJIS[item['emotion']]
        color = EMOTION_COLORS[item['emotion']]
        conf = item['probs'][item['emotion']] * 100
        
        timeline_html += f"""
        <div style='
            background: linear-gradient(135deg, {color}33 0%, {color}11 100%);
            border: 2px solid {color};
            border-radius: 15px;
            padding: 10px 15px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        '>
            <div style='font-size: 32px;'>{emoji}</div>
            <div style='font-size: 14px; font-weight: bold; color: white;'>{item['emotion']}</div>
            <div style='font-size: 12px; color: #bdc3c7;'>{conf:.0f}%</div>
        </div>
        """
    
    timeline_html += "</div>"
    return timeline_html


def predict_emotion(audio_input):
    """Fonction principale de prédiction."""
    if audio_input is None:
        return (
            "❌ Aucun audio fourni",
            "Veuillez enregistrer ou uploader un fichier audio",
            None, None, None, None, ""
        )
    
    try:
        # Extraire audio et sample rate
        sr, audio = audio_input
        
        # Convertir en float32 et normaliser
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        
        # Stéréo → Mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Prédire
        emotion, probs, mel_spec = PREDICTOR.predict(audio, sr)
        
        # Créer les visualisations
        waveform_fig = create_waveform_plot(audio, sr)
        spec_fig = create_spectrogram_plot(mel_spec)
        radar_fig = create_radar_chart(probs)
        gauge_fig = create_confidence_gauge(probs[emotion])
        timeline_html = create_timeline(PREDICTOR.history)
        
        # Message de résultat
        emoji = EMOTION_EMOJIS[emotion]
        conf = probs[emotion] * 100
        
        result_msg = f"## {emoji} **{emotion.upper()}** ({conf:.1f}%)"
        
        # Détail des probabilités
        details = "### 📊 Probabilités:\n"
        for emo in sorted(probs.keys(), key=lambda x: probs[x], reverse=True):
            bar_length = int(probs[emo] * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            details += f"**{emo:12s}** {bar} {probs[emo]*100:5.1f}%\n"
        
        return (
            result_msg,
            details,
            waveform_fig,
            spec_fig,
            radar_fig,
            gauge_fig,
            timeline_html
        )
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Erreur: {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        return (error_msg, "", None, None, None, None, "")


# CSS personnalisé - Glassmorphism
CUSTOM_CSS = """
#main_container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

.gradio-container {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(10px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
}

#title {
    text-align: center;
    font-size: 3em;
    font-weight: bold;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 20px 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

#subtitle {
    text-align: center;
    font-size: 1.3em;
    color: #bdc3c7;
    margin-bottom: 30px;
}

.glass-box {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

#result_emotion {
    font-size: 2.5em !important;
    text-align: center;
    padding: 30px;
    background: linear-gradient(135deg, rgba(52, 152, 219, 0.2) 0%, rgba(142, 68, 173, 0.2) 100%);
    border-radius: 20px;
    border: 2px solid rgba(52, 152, 219, 0.5);
    box-shadow: 0 8px 25px rgba(52, 152, 219, 0.3);
    animation: glow 2s ease-in-out infinite;
}

@keyframes glow {
    0%, 100% { box-shadow: 0 8px 25px rgba(52, 152, 219, 0.3); }
    50% { box-shadow: 0 8px 35px rgba(52, 152, 219, 0.6); }
}

#timeline_container {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 15px;
    padding: 15px;
    margin-top: 20px;
}
"""

# Créer l'interface
with gr.Blocks(css=CUSTOM_CSS, title="🎭 Emotion Recognition Studio", theme=gr.themes.Soft()) as demo:
    
    # Header
    gr.HTML('<h1 id="title">🎭 Emotion Recognition Studio</h1>')
    gr.HTML('<p id="subtitle">Reconnaissance d\'émotions vocales avec CNN-LSTM avancé</p>')
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input audio
            gr.HTML('<div class="glass-box"><h3>🎤 Enregistrement Audio</h3></div>')
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="numpy",
                label="Enregistrez ou uploadez votre audio",
                waveform_options={"waveform_color": "#3498db"}
            )
            
            predict_btn = gr.Button("🎯 Analyser l'Émotion", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            # Résultats
            gr.HTML('<div class="glass-box"><h3>📊 Résultats</h3></div>')
            result_emotion = gr.Markdown(elem_id="result_emotion")
            result_details = gr.Markdown()
    
    # Visualisations
    gr.HTML('<div class="glass-box"><h3>📈 Visualisations</h3></div>')
    
    with gr.Row():
        waveform_plot = gr.Plot(label="Forme d'onde")
        spec_plot = gr.Plot(label="Spectrogramme")
    
    with gr.Row():
        radar_plot = gr.Plot(label="Distribution des émotions")
        gauge_plot = gr.Plot(label="Confiance")
    
    # Timeline
    gr.HTML('<h3>🕐 Historique des Détections</h3>')
    timeline = gr.HTML(elem_id="timeline_container")
    
    # Footer
    gr.HTML("""
    <div style='text-align: center; margin-top: 40px; padding: 20px; opacity: 0.7;'>
        <p>🚀 Powered by Advanced CNN-LSTM with Attention</p>
        <p>✨ SpecAugment | Mixup | Label Smoothing | Multi-Head Attention</p>
    </div>
    """)
    
    # Event handler
    predict_btn.click(
        fn=predict_emotion,
        inputs=[audio_input],
        outputs=[result_emotion, result_details, waveform_plot, spec_plot, radar_plot, gauge_plot, timeline]
    )

if __name__ == "__main__":
    print("=" * 70)
    print("🎭 EMOTION RECOGNITION STUDIO")
    print("=" * 70)
    print("🚀 Launching interface...")
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
