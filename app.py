"""
Gradio Web Interface for Speech Emotion Recognition
Dark glassmorphism design – real-time emotion prediction
"""

import gradio as gr
import torch
import numpy as np
from scipy import signal as scipy_signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import EMOTION_LABELS, SAMPLE_RATE, MODEL_DIR
from src.model import get_model
from src.features import load_audio, extract_mel_spectrogram

# ──────────────────────────────── CONSTANTS ──────────────────────────────────

EMOTION_META = {
    "neutral":   {"emoji": "😐", "color": "#94a3b8", "hex": "#94a3b8"},
    "happy":     {"emoji": "😊", "color": "#facc15", "hex": "#facc15"},
    "sad":       {"emoji": "😢", "color": "#60a5fa", "hex": "#60a5fa"},
    "angry":     {"emoji": "😡", "color": "#f87171", "hex": "#f87171"},
    "fearful":   {"emoji": "😨", "color": "#c084fc", "hex": "#c084fc"},
    "surprised": {"emoji": "😲", "color": "#34d399", "hex": "#34d399"},
}

TEMPERATURE = 1.4   # soften overconfident logits

DARK_CSS = """
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: #0f1117 !important;
    color: #e2e8f0 !important;
}

/* remove Gradio's default white card */
.gradio-container > .main > .wrap { background: transparent !important; }

footer { display: none !important; }

/* ── hero banner ── */
.ser-hero {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4c1d95 70%, #1e3a5f 100%);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 16px;
    border: 1px solid rgba(139,92,246,.35);
    box-shadow: 0 8px 40px rgba(99,102,241,.25);
    position: relative;
    overflow: hidden;
}
.ser-hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(circle at 80% 30%, rgba(139,92,246,.15) 0%, transparent 60%);
    pointer-events: none;
}
.ser-hero h1 { margin: 0 0 6px; font-size: 28px; font-weight: 700; color: #fff; }
.ser-hero p  { margin: 0; color: rgba(255,255,255,.75); font-size: 15px; }

/* ── badge chips ── */
.chip-row { display: flex; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }
.chip {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;
    background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.2);
    color: #fff;
}

/* ── glass card ── */
.glass-card {
    background: rgba(30,27,75,.55);
    border: 1px solid rgba(139,92,246,.2);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 24px rgba(0,0,0,.35);
}

/* ── emotion result badge ── */
.emotion-result {
    display: flex; align-items: center; gap: 14px;
    background: rgba(99,102,241,.15);
    border: 1px solid rgba(139,92,246,.35);
    border-radius: 14px;
    padding: 16px 20px;
    margin-top: 10px;
}
.emotion-result .emo-emoji { font-size: 42px; line-height: 1; }
.emotion-result .emo-label { font-size: 22px; font-weight: 700; color: #c4b5fd; }
.emotion-result .emo-conf  { font-size: 13px; color: #94a3b8; margin-top: 2px; }

/* ── Gradio overrides ── */
.gr-button.primary, button.primary {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    border: none !important; border-radius: 10px !important;
    color: #fff !important; font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(99,102,241,.4) !important;
    transition: transform .15s, box-shadow .15s !important;
}
.gr-button.primary:hover, button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,.55) !important;
}
label, .label-wrap span { color: #a5b4fc !important; font-weight: 500 !important; }
.gr-form, .gr-box, .gr-panel, .block {
    background: rgba(15,17,23,.6) !important;
    border-color: rgba(139,92,246,.2) !important;
    border-radius: 12px !important;
}
input[type=range]::-webkit-slider-thumb { background: #8b5cf6 !important; }
input[type=range]::-webkit-slider-runnable-track { background: rgba(139,92,246,.3) !important; }
.tab-nav button { color: #94a3b8 !important; }
.tab-nav button.selected { color: #a5b4fc !important; border-bottom-color: #8b5cf6 !important; }
"""

# ───────────────────────────── MODEL LOADING ──────────────────────────────

model = None

def load_model(model_path: str = "models/best_model.pt"):
    """Load trained model – shape-compatible partial load when architecture differs."""
    global model

    model = get_model("cnn_lstm", num_classes=len(EMOTION_LABELS))

    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        saved_state = checkpoint.get("model_state_dict", checkpoint)

        # Get the new model's current state dict
        model_state = model.state_dict()

        # Only load weights that have a matching name AND matching shape
        compatible = {
            k: v for k, v in saved_state.items()
            if k in model_state and model_state[k].shape == v.shape
        }
        skipped = [k for k in saved_state if k not in compatible]

        model_state.update(compatible)
        model.load_state_dict(model_state)

        print(f"✓ Loaded {len(compatible)}/{len(saved_state)} compatible weight tensors from {model_path}")
        if skipped:
            print(f"  ↪ Skipped {len(skipped)} incompatible tensors (new layers will use random init): {skipped[:4]}")
    else:
        print("⚠ No trained model found – using random weights (demo mode)")

    model.eval()
    return model


# ──────────────────────────── AUDIO HELPERS ────────────────────────────────

def _pad_or_truncate(signal: np.ndarray, target_length: int) -> np.ndarray:
    if len(signal) < target_length:
        return np.pad(signal, (0, target_length - len(signal)))
    return signal[:target_length]


def _normalize_audio(signal: np.ndarray) -> np.ndarray:
    signal = signal.astype(np.float32)
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal /= peak
    rms = np.sqrt(np.mean(signal ** 2) + 1e-8)
    signal = signal * (0.1 / rms)
    return np.clip(signal, -1.0, 1.0).astype(np.float32)


def _highpass(signal: np.ndarray, sr: int = SAMPLE_RATE, cutoff: float = 80.0) -> np.ndarray:
    b, a = scipy_signal.butter(4, cutoff / (sr / 2), btype="highpass")
    return scipy_signal.filtfilt(b, a, signal).astype(np.float32)


def _time_shift(signal: np.ndarray, shift: int, target: int) -> np.ndarray:
    if shift > 0:
        return _pad_or_truncate(np.concatenate([np.zeros(shift, dtype=signal.dtype), signal]), target)
    if shift < 0:
        s = abs(shift)
        return _pad_or_truncate(signal[s:] if s < len(signal) else np.zeros(target, dtype=signal.dtype), target)
    return _pad_or_truncate(signal, target)


def _prepare_audio(audio, duration_sec: float = 4.0, apply_denoise: bool = True) -> np.ndarray:
    sr, data = audio
    data = np.asarray(data)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if sr != SAMPLE_RATE:
        data = scipy_signal.resample(data, int(len(data) * SAMPLE_RATE / sr))
    if apply_denoise:
        data = _highpass(data, cutoff=80.0)
    target = int(SAMPLE_RATE * duration_sec)
    data = _pad_or_truncate(data, target)
    return _normalize_audio(data)


# ──────────────────────────── INFERENCE ───────────────────────────────────

def _predict_single(signal: np.ndarray) -> np.ndarray:
    global model
    if model is None:
        load_model()
    mel = extract_mel_spectrogram(signal)
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)
    tensor = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        # Temperature scaling → sharper / softer probs
        probs = torch.softmax(logits / TEMPERATURE, dim=1)
    return probs.squeeze().numpy()


def predict_emotion(audio, duration_sec=4.0, num_shifts=2,
                    shift_step=0.5, apply_denoise=True, enable_smoothing=True):
    global model
    if model is None:
        load_model()

    if audio is None:
        return {e: 0.0 for e in EMOTION_LABELS}, None, _empty_html()

    data = _prepare_audio(audio, duration_sec=duration_sec, apply_denoise=apply_denoise)
    target = int(SAMPLE_RATE * duration_sec)
    segments = [_pad_or_truncate(data, target)]

    if enable_smoothing and num_shifts > 0 and shift_step > 0:
        step = int(SAMPLE_RATE * shift_step)
        for k in range(1, int(num_shifts) + 1):
            segments.append(_time_shift(data,  k * step, target))
            segments.append(_time_shift(data, -k * step, target))

    stack = np.vstack([_predict_single(s) for s in segments])
    probs = np.mean(stack, axis=0)

    prob_dict = {EMOTION_LABELS[i]: float(probs[i]) for i in range(len(EMOTION_LABELS))}

    # Dominant emotion
    top_idx  = int(np.argmax(probs))
    top_name = EMOTION_LABELS[top_idx]
    top_conf = float(probs[top_idx])

    fig  = _create_visualization(extract_mel_spectrogram(segments[0]), prob_dict)
    html = _result_html(top_name, top_conf, prob_dict)

    return prob_dict, fig, html


# ──────────────────────────── VISUALIZATION ───────────────────────────────

def _create_visualization(mel_spec, probabilities):
    """Dark-themed mel-spectrogram + emotion bar chart."""
    mel_norm = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

    plt.rcParams.update({
        "figure.facecolor": "#0f1117",
        "axes.facecolor": "#1a1d2e",
        "text.color": "#e2e8f0",
        "axes.labelcolor": "#94a3b8",
        "xtick.color": "#64748b",
        "ytick.color": "#64748b",
        "axes.edgecolor": "#334155",
        "grid.color": "#1e293b",
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor("#0f1117")

    # ── Mel spectrogram ──
    ax_mel = axes[0]
    im = ax_mel.imshow(mel_norm, aspect="auto", origin="lower",
                       cmap="magma", interpolation="bilinear")
    ax_mel.set_title("Mel-Spectrogram", fontsize=13, fontweight="bold", pad=10, color="#c4b5fd")
    ax_mel.set_xlabel("Time Frames", fontsize=10)
    ax_mel.set_ylabel("Mel Bands", fontsize=10)
    cbar = plt.colorbar(im, ax=ax_mel, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#64748b")

    # ── Emotion bars ──
    ax_bar = axes[1]
    emotions = list(probabilities.keys())
    probs    = list(probabilities.values())
    colors   = [EMOTION_META.get(e, {}).get("color", "#6366f1") for e in emotions]

    # Sort by probability
    order  = np.argsort(probs)
    s_emo  = [emotions[i] for i in order]
    s_prob = [probs[i]    for i in order]
    s_col  = [colors[i]   for i in order]

    bars = ax_bar.barh(
        [f"{EMOTION_META.get(e,{}).get('emoji','🎯')} {e}" for e in s_emo],
        s_prob, color=s_col, alpha=0.85, height=0.65,
        edgecolor="rgba(255,255,255,0)"
    )

    # Highlight top
    top_local = s_emo.index(EMOTION_LABELS[int(np.argmax(probs))])
    bars[top_local].set_linewidth(2.5)
    bars[top_local].set_edgecolor("#fff")
    bars[top_local].set_alpha(1.0)

    # Value labels
    for bar, val in zip(bars, s_prob):
        ax_bar.text(min(val + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                    f"{val:.0%}", va="center", ha="left",
                    fontsize=9, color="#e2e8f0", fontweight="600")

    ax_bar.set_xlim(0, 1.0)
    ax_bar.set_title("Emotion Distribution", fontsize=13, fontweight="bold", pad=10, color="#c4b5fd")
    ax_bar.set_xlabel("Confidence", fontsize=10)
    ax_bar.grid(axis="x", linestyle="--", alpha=0.3)
    ax_bar.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(pad=2.0)
    return fig


# ──────────────────────────── HTML RESULT CARD ─────────────────────────────

def _empty_html():
    return """
    <div style="text-align:center;padding:20px;color:#64748b;font-family:Inter,sans-serif;">
        🎤 Enregistrez ou importez un audio pour commencer l'analyse
    </div>"""


def _result_html(top_name: str, confidence: float, all_probs: dict) -> str:
    meta  = EMOTION_META.get(top_name, {"emoji": "🎯", "color": "#6366f1"})
    emoji = meta["emoji"]
    color = meta["color"]

    level = ("Très élevée" if confidence > 0.7
             else "Élevée" if confidence > 0.5
             else "Modérée" if confidence > 0.35
             else "Faible")

    bars_html = ""
    for name, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        m   = EMOTION_META.get(name, {"emoji": "🎯", "color": "#6366f1"})
        pct = int(prob * 100)
        is_top = "font-weight:700;" if name == top_name else "opacity:.75;"
        bars_html += f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;{is_top}">
            <span style="width:22px;text-align:center">{m['emoji']}</span>
            <span style="width:80px;font-size:12px;color:#cbd5e1">{name}</span>
            <div style="flex:1;background:rgba(255,255,255,.08);border-radius:6px;height:8px;overflow:hidden;">
                <div style="width:{pct}%;height:100%;border-radius:6px;
                            background:{m['color']};transition:width .4s ease;"></div>
            </div>
            <span style="width:38px;font-size:12px;text-align:right;color:#94a3b8">{pct}%</span>
        </div>"""

    return f"""
    <div style="font-family:Inter,sans-serif;padding:4px 0;">
        <div style="display:flex;align-items:center;gap:14px;
                    background:rgba(99,102,241,.12);border:1px solid rgba(139,92,246,.3);
                    border-radius:14px;padding:16px 20px;margin-bottom:14px;">
            <span style="font-size:48px;line-height:1">{emoji}</span>
            <div>
                <div style="font-size:22px;font-weight:700;color:{color};text-transform:capitalize">
                    {top_name}
                </div>
                <div style="font-size:12px;color:#94a3b8;margin-top:3px">
                    Confiance : <b style="color:{color}">{confidence:.0%}</b>
                    &nbsp;·&nbsp; Niveau : <b style="color:#e2e8f0">{level}</b>
                </div>
            </div>
        </div>
        <div style="background:rgba(15,17,23,.5);border:1px solid rgba(139,92,246,.15);
                    border-radius:12px;padding:14px 16px;">
            <div style="font-size:12px;color:#64748b;margin-bottom:10px;font-weight:600;
                        letter-spacing:.5px;text-transform:uppercase">Distribution complète</div>
            {bars_html}
        </div>
    </div>"""


# ──────────────────────────── GRADIO INTERFACE ─────────────────────────────

def create_demo_interface():
    load_model()

    with gr.Blocks(title="🎤 Speech Emotion Recognition", css=DARK_CSS) as demo:

        # ── Hero banner ──
        gr.HTML("""
        <div class="ser-hero">
            <div class="chip-row">
                <span class="chip">🔴 Live</span>
                <span class="chip">🧠 Deep Learning</span>
                <span class="chip">🎵 CNN-LSTM + Attention</span>
                <span class="chip">6 Émotions</span>
            </div>
            <h1>🎤 Speech Emotion Recognition</h1>
            <p>Analysez vos prises de parole et visualisez les émotions détectées en temps réel avec un modèle CNN-LSTM attentionnel.</p>
        </div>
        """)

        with gr.Tabs():
            # ── Tab 1: Analyse ──────────────────────────────────────────────
            with gr.Tab("🔍 Analyse"):
                with gr.Row(equal_height=False):

                    # ── Left column: inputs ──
                    with gr.Column(scale=1, min_width=300):
                        gr.HTML('<div class="glass-card">')

                        audio_input = gr.Audio(
                            sources=["microphone", "upload"],
                            type="numpy",
                            label="🎙 Audio (microphone ou fichier)",
                        )

                        analyze_btn = gr.Button("🔍 Analyser l'émotion", variant="primary", size="lg")

                        gr.HTML('</div>')

                        with gr.Accordion("⚙️ Options avancées", open=False):
                            duration_sl = gr.Slider(2.0, 6.0, value=4.0, step=0.5,
                                                    label="Durée d'analyse (s)")
                            shift_sl    = gr.Slider(0, 4, value=2, step=1,
                                                    label="Fenêtres TTA (0 = désactivé)")
                            step_sl     = gr.Slider(0.25, 1.0, value=0.5, step=0.25,
                                                    label="Pas de décalage TTA (s)")
                            denoise_cb  = gr.Checkbox(value=True,
                                                      label="Filtre coupe-bas 80 Hz")
                            smooth_cb   = gr.Checkbox(value=True,
                                                      label="Lissage multi-fenêtres (TTA)")

                    # ── Right column: outputs ──
                    with gr.Column(scale=2, min_width=400):
                        result_html = gr.HTML(
                            value=_empty_html(),
                            label="Résultat principal"
                        )

                        emotion_label = gr.Label(
                            num_top_classes=6,
                            label="Scores par émotion"
                        )

                        viz_out = gr.Plot(label="Visualisation audio")

            # ── Tab 2: À propos ─────────────────────────────────────────────
            with gr.Tab("ℹ️ À propos"):
                gr.HTML("""
                <div style="font-family:Inter,sans-serif;max-width:680px;line-height:1.7;color:#cbd5e1">
                    <h2 style="color:#a5b4fc">Architecture du modèle</h2>
                    <p>Le pipeline utilise un modèle <b>CNN-LSTM bidirectionnel avec attention temporelle</b> :</p>
                    <ul>
                        <li>4 blocs CNN (32→64→128→256 filtres) extraient les motifs spectraux</li>
                        <li>2 couches LSTM bidirectionnelles capturent la dynamique temporelle</li>
                        <li><b>Attention pooling</b> : pondère les frames les plus discriminants</li>
                        <li>Normalisation par température (T=1.4) pour calibrer la confiance</li>
                        <li>TTA (Test-Time Augmentation) : moyenne sur plusieurs décalages temporels</li>
                    </ul>
                    <h2 style="color:#a5b4fc">Émotions reconnues</h2>
                    <div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:8px">
                        <span style="padding:6px 14px;border-radius:20px;background:rgba(148,163,184,.15);border:1px solid #94a3b8">😐 Neutral</span>
                        <span style="padding:6px 14px;border-radius:20px;background:rgba(250,204,21,.15);border:1px solid #facc15">😊 Happy</span>
                        <span style="padding:6px 14px;border-radius:20px;background:rgba(96,165,250,.15);border:1px solid #60a5fa">😢 Sad</span>
                        <span style="padding:6px 14px;border-radius:20px;background:rgba(248,113,113,.15);border:1px solid #f87171">😡 Angry</span>
                        <span style="padding:6px 14px;border-radius:20px;background:rgba(192,132,252,.15);border:1px solid #c084fc">😨 Fearful</span>
                        <span style="padding:6px 14px;border-radius:20px;background:rgba(52,211,153,.15);border:1px solid #34d399">😲 Surprised</span>
                    </div>
                    <h2 style="color:#a5b4fc;margin-top:20px">Conseils d'utilisation</h2>
                    <ul>
                        <li>Parlez 3–5 secondes à volume constant</li>
                        <li>Évitez les bruits de fond (activez le filtre coupe-bas si nécessaire)</li>
                        <li>Augmentez les fenêtres TTA pour stabiliser la prédiction</li>
                        <li>Essayez différentes émotions pour apprécier la discrimination</li>
                    </ul>
                    <p style="margin-top:16px;color:#64748b;font-size:13px">
                        Modèle entraîné sur le dataset <b>RAVDESS</b> (24 acteurs, 8 émotions → 6 classes condensées).
                    </p>
                </div>
                """)

        # ── Event wiring ──────────────────────────────────────────────────────
        _outputs = [emotion_label, viz_out, result_html]
        _inputs  = [audio_input, duration_sl, shift_sl, step_sl, denoise_cb, smooth_cb]

        analyze_btn.click(fn=predict_emotion, inputs=_inputs, outputs=_outputs)
        audio_input.change(fn=predict_emotion, inputs=_inputs, outputs=_outputs)

    return demo


# ──────────────────────────── ENTRY POINT ─────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Speech Emotion Recognition  –  Demo Interface")
    print("=" * 55)

    port = int(os.getenv("GRADIO_SERVER_PORT", "7861"))

    demo = create_demo_interface()
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=None if port <= 0 else port,
    )
