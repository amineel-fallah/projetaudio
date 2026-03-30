"""
Feature Extraction Module for Speech Emotion Recognition
Extracts MFCCs, mel-spectrograms using scipy (no numba dependency)
Enhanced with prosodic features for better emotion detection
"""

import numpy as np
import soundfile as sf
from scipy import signal as scipy_signal
from scipy.fft import rfft
from scipy.ndimage import uniform_filter1d
import torch
from config import SAMPLE_RATE, DURATION, N_MFCC, N_MELS, HOP_LENGTH, N_FFT


def hz_to_mel(hz):
    """Convert Hz to Mel scale."""
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    """Convert Mel to Hz."""
    return 700 * (10 ** (mel / 2595) - 1)


def get_mel_filterbank(sr, n_fft, n_mels, fmin=0, fmax=None):
    """Create mel filterbank matrix."""
    if fmax is None:
        fmax = sr / 2
    
    # Mel points
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    # FFT bin frequencies
    fft_bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    
    # Create filterbank
    n_freq = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_freq))
    
    for i in range(n_mels):
        left = fft_bins[i]
        center = fft_bins[i + 1]
        right = fft_bins[i + 2]
        
        for j in range(left, center):
            if center != left:
                filterbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right != center:
                filterbank[i, j] = (right - j) / (right - center)
    
    return filterbank


def compute_energy_features(signal: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Compute energy-related features that help distinguish emotions.
    High energy: angry, happy, surprised
    Low energy: sad, neutral
    """
    frame_length = int(0.025 * sr)  # 25ms frames
    hop = int(0.010 * sr)  # 10ms hop
    
    # Frame the signal
    n_frames = 1 + (len(signal) - frame_length) // hop
    frames = np.zeros((n_frames, frame_length))
    for i in range(n_frames):
        start = i * hop
        frames[i] = signal[start:start + frame_length]
    
    # RMS energy per frame
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    
    # Zero crossing rate per frame
    zcr = np.sum(np.abs(np.diff(np.sign(frames), axis=1)), axis=1) / (2 * frame_length)
    
    return rms, zcr


def compute_pitch_features(signal: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """
    Estimate fundamental frequency (F0) using autocorrelation.
    Higher pitch: surprised, happy, fearful
    Lower pitch: sad, angry, neutral
    """
    # Use autocorrelation method
    min_period = int(sr / 500)  # Max F0 = 500 Hz
    max_period = int(sr / 50)   # Min F0 = 50 Hz
    
    correlation = np.correlate(signal, signal, mode='full')
    correlation = correlation[len(correlation)//2:]
    
    # Find peaks in the valid range
    correlation = correlation[min_period:max_period]
    if len(correlation) > 0:
        peak_idx = np.argmax(correlation) + min_period
        f0 = sr / peak_idx if peak_idx > 0 else 0
    else:
        f0 = 0
    
    return f0


def load_audio(file_path: str, sr: int = SAMPLE_RATE, duration: float = DURATION) -> np.ndarray:
    """
    Load and preprocess audio file.
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate (default: 16kHz)
        duration: Max duration in seconds
        
    Returns:
        Audio signal as numpy array
    """
    # Load audio file using soundfile (faster and no numba dependency)
    signal, file_sr = sf.read(file_path)
    
    # Convert to mono if stereo
    if len(signal.shape) > 1:
        signal = signal.mean(axis=1)
    
    # Resample if necessary using scipy (no numba dependency)
    if file_sr != sr:
        num_samples = int(len(signal) * sr / file_sr)
        signal = scipy_signal.resample(signal, num_samples)
    
    # Truncate to duration
    max_samples = int(sr * duration)
    if len(signal) > max_samples:
        signal = signal[:max_samples]
    
    # Pad or truncate to fixed length
    target_length = int(sr * duration)
    if len(signal) < target_length:
        signal = np.pad(signal, (0, target_length - len(signal)), mode='constant')
    else:
        signal = signal[:target_length]
    
    return signal.astype(np.float32)


def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    """Convert power spectrogram to decibel units."""
    S_db = 10.0 * np.log10(np.maximum(amin, S))
    S_db -= 10.0 * np.log10(np.maximum(amin, ref))
    if top_db is not None:
        S_db = np.maximum(S_db, S_db.max() - top_db)
    return S_db


def compute_delta(data, width=9, order=1):
    """Compute delta features (scipy-based, no librosa)."""
    half_width = width // 2
    window = np.arange(-half_width, half_width + 1)
    
    # Pad the data
    padded = np.pad(data, ((0, 0), (half_width, half_width)), mode='edge')
    
    delta = np.zeros_like(data)
    denominator = 2 * sum([i**2 for i in range(1, half_width + 1)])
    
    for t in range(data.shape[1]):
        delta[:, t] = np.sum(window * padded[:, t:t + width], axis=1) / denominator
    
    if order > 1:
        return compute_delta(delta, width, order - 1)
    return delta


def extract_mfcc(signal: np.ndarray, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC) -> np.ndarray:
    """
    Extract MFCC features with delta and delta-delta (scipy-based).
    
    Args:
        signal: Audio signal
        sr: Sample rate
        n_mfcc: Number of MFCCs
        
    Returns:
        MFCC features with deltas (shape: 3*n_mfcc x time_frames)
    """
    # First get mel spectrogram
    mel_spec = extract_mel_spectrogram(signal, sr, N_MELS)
    
    # Convert from dB back to power for DCT
    mel_power = 10 ** (mel_spec / 10)
    
    # Apply DCT to get MFCCs
    from scipy.fftpack import dct
    mfcc = dct(mel_power, type=2, axis=0, norm='ortho')[:n_mfcc]
    
    # Compute delta and delta-delta
    mfcc_delta = compute_delta(mfcc)
    mfcc_delta2 = compute_delta(mfcc, order=2)
    
    # Stack features
    features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
    
    return features


def extract_mel_spectrogram(signal: np.ndarray, sr: int = SAMPLE_RATE, 
                            n_mels: int = N_MELS, 
                            enhanced: bool = True,
                            target_length: int = 128) -> np.ndarray:
    """
    Extract mel-spectrogram features (scipy-based, no librosa/numba).
    Enhanced version captures more emotional nuances.
    
    Args:
        signal: Audio signal
        sr: Sample rate
        n_mels: Number of mel bands
        enhanced: Apply enhancement for emotion detection
        target_length: Fixed time dimension for the output
        
    Returns:
        Log mel-spectrogram (shape: n_mels x target_length)
    """
    # Pre-emphasis filter to boost high frequencies (helps with emotion detection)
    if enhanced:
        signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
    
    # Compute STFT using scipy
    frequencies, times, Zxx = scipy_signal.stft(
        signal, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH, 
        window='hann', padded=True
    )
    
    # Power spectrogram
    power_spec = np.abs(Zxx) ** 2
    
    # Get mel filterbank and apply
    mel_filter = get_mel_filterbank(sr, N_FFT, n_mels, fmin=20, fmax=sr//2 - 500)
    mel_spec = np.dot(mel_filter, power_spec)
    
    # Convert to dB with better dynamic range
    log_mel_spec = power_to_db(mel_spec, ref=np.max(mel_spec), top_db=100)
    
    # Apply per-channel energy normalization (PCEN) approximation for robustness
    if enhanced:
        # Simple PCEN-like normalization
        log_mel_spec = (log_mel_spec - log_mel_spec.mean(axis=1, keepdims=True)) / (
            log_mel_spec.std(axis=1, keepdims=True) + 1e-8
        )
    
    # Ensure fixed time dimension
    current_length = log_mel_spec.shape[1]
    if current_length > target_length:
        # Truncate
        log_mel_spec = log_mel_spec[:, :target_length]
    elif current_length < target_length:
        # Pad with zeros
        pad_width = target_length - current_length
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='constant')
    
    return log_mel_spec


def extract_chromagram(signal: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract chromagram features (scipy-based).
    
    Args:
        signal: Audio signal
        sr: Sample rate
        
    Returns:
        Chromagram (shape: 12 x time_frames)
    """
    # Compute STFT
    frequencies, times, Zxx = scipy_signal.stft(
        signal, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH, 
        window='hann', padded=True
    )
    
    # Power spectrogram
    power_spec = np.abs(Zxx) ** 2
    
    # Map frequencies to chroma (12 pitch classes)
    n_chroma = 12
    n_freq = power_spec.shape[0]
    
    # Create chroma filterbank
    chroma_filter = np.zeros((n_chroma, n_freq))
    freq_bins = np.fft.rfftfreq(N_FFT, 1/sr)
    
    for i, f in enumerate(freq_bins):
        if f > 0:
            # Convert frequency to pitch class (0-11)
            pitch = 12 * np.log2(f / 440) + 69  # MIDI note number
            chroma_bin = int(pitch) % 12
            chroma_filter[chroma_bin, i] += 1
    
    # Normalize filterbank
    chroma_filter = chroma_filter / (chroma_filter.sum(axis=1, keepdims=True) + 1e-8)
    
    # Apply filterbank
    chroma = np.dot(chroma_filter, power_spec)
    
    return chroma


def extract_all_features(file_path: str) -> dict:
    """
    Extract all audio features from a file.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dictionary containing all features
    """
    signal = load_audio(file_path)
    
    features = {
        'mfcc': extract_mfcc(signal),
        'mel_spectrogram': extract_mel_spectrogram(signal),
        'chromagram': extract_chromagram(signal),
        'signal': signal
    }
    
    return features


def features_to_tensor(features: dict, feature_type: str = 'mel_spectrogram') -> torch.Tensor:
    """
    Convert features to PyTorch tensor.
    
    Args:
        features: Dictionary of features
        feature_type: Type of feature to convert
        
    Returns:
        PyTorch tensor (shape: 1 x feature_dim x time_frames)
    """
    feat = features[feature_type]
    tensor = torch.FloatTensor(feat).unsqueeze(0)  # Add channel dimension
    return tensor


# Data augmentation functions
def add_noise(signal: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """Add random noise to audio signal."""
    noise = np.random.randn(len(signal))
    augmented = signal + noise_factor * noise
    return augmented.astype(signal.dtype)


def pitch_shift(signal: np.ndarray, sr: int = SAMPLE_RATE, 
                n_steps: float = 2.0) -> np.ndarray:
    """
    Shift pitch of audio signal using resampling (scipy-based).
    Note: This is a simplified implementation.
    """
    # Calculate the ratio for pitch shift
    shift_ratio = 2 ** (n_steps / 12)
    
    # Resample to shift pitch
    new_length = int(len(signal) / shift_ratio)
    shifted = scipy_signal.resample(signal, new_length)
    
    # Resample back to original length to maintain duration
    result = scipy_signal.resample(shifted, len(signal))
    
    return result.astype(signal.dtype)


def time_stretch(signal: np.ndarray, rate: float = 1.1) -> np.ndarray:
    """
    Time-stretch audio signal using resampling (scipy-based).
    """
    new_length = int(len(signal) / rate)
    stretched = scipy_signal.resample(signal, new_length)
    return stretched.astype(signal.dtype)


if __name__ == "__main__":
    # Test feature extraction
    print("Feature Extraction Module")
    print("=" * 40)
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Duration: {DURATION} seconds")
    print(f"MFCC coefficients: {N_MFCC}")
    print(f"Mel bands: {N_MELS}")
