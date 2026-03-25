"""
Data Augmentation Module for Speech Emotion Recognition
Provides various audio augmentation techniques for robustness
"""

import numpy as np
import librosa
from typing import Tuple, Optional
import random


class AudioAugmenter:
    """
    Audio augmentation class with multiple transformation techniques.
    Helps improve model robustness by creating variations of training data.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def add_gaussian_noise(self, signal: np.ndarray, 
                           noise_factor: float = 0.005) -> np.ndarray:
        """
        Add Gaussian noise to audio signal.
        
        Args:
            signal: Input audio signal
            noise_factor: Noise amplitude (0.001-0.01 recommended)
            
        Returns:
            Noisy signal
        """
        noise = np.random.randn(len(signal))
        augmented = signal + noise_factor * noise
        return augmented.astype(signal.dtype)
    
    def add_background_noise(self, signal: np.ndarray, 
                             noise_signal: np.ndarray,
                             snr_db: float = 10) -> np.ndarray:
        """
        Add background noise with specified SNR.
        
        Args:
            signal: Clean speech signal
            noise_signal: Background noise signal
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            Mixed signal
        """
        # Ensure same length
        if len(noise_signal) < len(signal):
            noise_signal = np.tile(noise_signal, int(np.ceil(len(signal) / len(noise_signal))))
        noise_signal = noise_signal[:len(signal)]
        
        # Calculate scaling factor for desired SNR
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise_signal ** 2)
        
        snr_linear = 10 ** (snr_db / 10)
        scale = np.sqrt(signal_power / (snr_linear * noise_power + 1e-10))
        
        return signal + scale * noise_signal
    
    def pitch_shift(self, signal: np.ndarray, 
                    n_steps: float = 2.0) -> np.ndarray:
        """
        Shift pitch of audio signal.
        
        Args:
            signal: Input signal
            n_steps: Number of semitones to shift (-12 to 12)
            
        Returns:
            Pitch-shifted signal
        """
        return librosa.effects.pitch_shift(
            signal, sr=self.sample_rate, n_steps=n_steps
        )
    
    def time_stretch(self, signal: np.ndarray, 
                     rate: float = 1.0) -> np.ndarray:
        """
        Time-stretch audio signal without changing pitch.
        
        Args:
            signal: Input signal
            rate: Stretch factor (0.8-1.2 recommended)
                  rate > 1: faster (shorter)
                  rate < 1: slower (longer)
            
        Returns:
            Time-stretched signal
        """
        return librosa.effects.time_stretch(signal, rate=rate)
    
    def change_volume(self, signal: np.ndarray, 
                      gain_db: float = 0) -> np.ndarray:
        """
        Change audio volume.
        
        Args:
            signal: Input signal
            gain_db: Volume change in dB (-10 to 10 recommended)
            
        Returns:
            Volume-adjusted signal
        """
        gain_linear = 10 ** (gain_db / 20)
        return signal * gain_linear
    
    def shift_time(self, signal: np.ndarray, 
                   shift_max: float = 0.2) -> np.ndarray:
        """
        Randomly shift audio in time.
        
        Args:
            signal: Input signal
            shift_max: Maximum shift as fraction of signal length
            
        Returns:
            Time-shifted signal
        """
        shift = int(len(signal) * shift_max * (random.random() * 2 - 1))
        if shift > 0:
            return np.pad(signal[shift:], (0, shift), mode='constant')
        else:
            return np.pad(signal[:shift], (-shift, 0), mode='constant')
    
    def add_reverb(self, signal: np.ndarray, 
                   reverb_amount: float = 0.3) -> np.ndarray:
        """
        Add simple reverb effect.
        
        Args:
            signal: Input signal
            reverb_amount: Reverb intensity (0-0.5)
            
        Returns:
            Signal with reverb
        """
        # Simple delay-based reverb
        delay_samples = int(0.03 * self.sample_rate)  # 30ms delay
        reverb = np.zeros_like(signal)
        reverb[delay_samples:] = signal[:-delay_samples] * reverb_amount
        return signal + reverb
    
    def random_augment(self, signal: np.ndarray, 
                       p: float = 0.5) -> np.ndarray:
        """
        Apply random augmentations with probability p.
        
        Args:
            signal: Input signal
            p: Probability of applying each augmentation
            
        Returns:
            Augmented signal
        """
        augmented = signal.copy()
        
        # Noise
        if random.random() < p:
            noise_factor = random.uniform(0.001, 0.01)
            augmented = self.add_gaussian_noise(augmented, noise_factor)
        
        # Pitch shift
        if random.random() < p * 0.5:  # Less frequent
            n_steps = random.uniform(-3, 3)
            augmented = self.pitch_shift(augmented, n_steps)
        
        # Time stretch
        if random.random() < p * 0.3:
            rate = random.uniform(0.9, 1.1)
            augmented = self.time_stretch(augmented, rate)
            # Adjust length back
            target_len = len(signal)
            if len(augmented) > target_len:
                augmented = augmented[:target_len]
            else:
                augmented = np.pad(augmented, (0, target_len - len(augmented)))
        
        # Volume change
        if random.random() < p:
            gain_db = random.uniform(-6, 6)
            augmented = self.change_volume(augmented, gain_db)
        
        # Time shift
        if random.random() < p * 0.3:
            augmented = self.shift_time(augmented, 0.1)
        
        return augmented
    
    def augment_batch(self, signals: np.ndarray, 
                      num_augmentations: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment a batch of signals.
        
        Args:
            signals: Array of shape (n_samples, signal_length)
            num_augmentations: Number of augmented versions per sample
            
        Returns:
            Tuple of (augmented_signals, indices mapping to original)
        """
        augmented = []
        indices = []
        
        for i, signal in enumerate(signals):
            # Original
            augmented.append(signal)
            indices.append(i)
            
            # Augmented versions
            for _ in range(num_augmentations):
                aug_signal = self.random_augment(signal)
                augmented.append(aug_signal)
                indices.append(i)
        
        return np.array(augmented), np.array(indices)


class SpecAugment:
    """
    SpecAugment: Augmentation techniques for spectrograms.
    Based on the paper "SpecAugment: A Simple Data Augmentation Method 
    for Automatic Speech Recognition"
    """
    
    @staticmethod
    def time_mask(spec: np.ndarray, max_mask: int = 20, 
                  num_masks: int = 1) -> np.ndarray:
        """
        Apply time masking to spectrogram.
        
        Args:
            spec: Spectrogram of shape (freq, time)
            max_mask: Maximum mask width
            num_masks: Number of masks to apply
            
        Returns:
            Masked spectrogram
        """
        spec = spec.copy()
        _, time_steps = spec.shape
        
        for _ in range(num_masks):
            t = random.randint(0, min(max_mask, time_steps - 1))
            t0 = random.randint(0, time_steps - t)
            spec[:, t0:t0 + t] = 0
            
        return spec
    
    @staticmethod
    def freq_mask(spec: np.ndarray, max_mask: int = 20,
                  num_masks: int = 1) -> np.ndarray:
        """
        Apply frequency masking to spectrogram.
        
        Args:
            spec: Spectrogram of shape (freq, time)
            max_mask: Maximum mask height
            num_masks: Number of masks to apply
            
        Returns:
            Masked spectrogram
        """
        spec = spec.copy()
        freq_bins, _ = spec.shape
        
        for _ in range(num_masks):
            f = random.randint(0, min(max_mask, freq_bins - 1))
            f0 = random.randint(0, freq_bins - f)
            spec[f0:f0 + f, :] = 0
            
        return spec
    
    @staticmethod
    def spec_augment(spec: np.ndarray, 
                     time_mask_param: int = 20,
                     freq_mask_param: int = 20,
                     num_time_masks: int = 2,
                     num_freq_masks: int = 2) -> np.ndarray:
        """
        Apply full SpecAugment (time + frequency masking).
        
        Args:
            spec: Input spectrogram
            time_mask_param: Max time mask width
            freq_mask_param: Max frequency mask height
            num_time_masks: Number of time masks
            num_freq_masks: Number of frequency masks
            
        Returns:
            Augmented spectrogram
        """
        spec = SpecAugment.time_mask(spec, time_mask_param, num_time_masks)
        spec = SpecAugment.freq_mask(spec, freq_mask_param, num_freq_masks)
        return spec


if __name__ == "__main__":
    print("Audio Augmentation Module")
    print("=" * 40)
    print("Available augmentations:")
    print("- Gaussian noise")
    print("- Background noise mixing")
    print("- Pitch shifting")
    print("- Time stretching")
    print("- Volume change")
    print("- Time shifting")
    print("- Reverb")
    print("- SpecAugment (time/freq masking)")
