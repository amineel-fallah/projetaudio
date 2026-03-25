"""
Download RAVDESS Dataset
Ryerson Audio-Visual Database of Emotional Speech and Song
"""

import os
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

# RAVDESS is available on Zenodo
# Note: Manual download may be required from:
# https://zenodo.org/record/1188976

DATA_DIR = "data"
RAVDESS_DIR = "data/ravdess"


def download_file(url: str, destination: str):
    """Download file with progress bar."""
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=destination, reporthook=t.update_to)


def setup_directories():
    """Create necessary directories."""
    Path(DATA_DIR).mkdir(exist_ok=True)
    Path(RAVDESS_DIR).mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    print("✓ Created project directories")


def download_ravdess_sample():
    """
    Download a sample of RAVDESS dataset.
    
    Note: The full RAVDESS dataset should be downloaded manually from:
    https://zenodo.org/record/1188976
    
    This function creates sample structure for testing.
    """
    print("\n" + "=" * 60)
    print("RAVDESS Dataset Setup")
    print("=" * 60)
    
    print("""
    The RAVDESS dataset must be downloaded manually from Zenodo:
    
    1. Go to: https://zenodo.org/record/1188976
    2. Download "Audio_Speech_Actors_01-24.zip" (215 MB)
    3. Extract to: data/ravdess/
    
    Expected structure:
    data/
    └── ravdess/
        ├── Actor_01/
        │   ├── 03-01-01-01-01-01-01.wav
        │   ├── 03-01-01-01-01-02-01.wav
        │   └── ...
        ├── Actor_02/
        └── ...
    
    File naming convention:
    Modality-Vocal_channel-Emotion-Intensity-Statement-Repetition-Actor.wav
    
    Emotion codes:
    01 = neutral
    02 = calm
    03 = happy
    04 = sad
    05 = angry
    06 = fearful
    07 = disgust
    08 = surprised
    """)


def verify_dataset():
    """Verify RAVDESS dataset is properly installed."""
    ravdess_path = Path(RAVDESS_DIR)
    
    if not ravdess_path.exists():
        print("✗ RAVDESS directory not found")
        return False
    
    actor_dirs = list(ravdess_path.glob("Actor_*"))
    if len(actor_dirs) == 0:
        print("✗ No actor directories found")
        return False
    
    total_files = sum(len(list(d.glob("*.wav"))) for d in actor_dirs)
    
    print(f"\n✓ Found {len(actor_dirs)} actors")
    print(f"✓ Found {total_files} audio files")
    
    if total_files >= 1000:
        print("✓ Dataset appears complete!")
        return True
    else:
        print("⚠ Dataset may be incomplete")
        return False


def create_sample_data():
    """Create dummy sample data for testing the pipeline."""
    import numpy as np
    from scipy.io import wavfile
    
    print("\nCreating sample data for testing...")
    
    sample_dir = Path(RAVDESS_DIR) / "Actor_99"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample audio files for each emotion
    emotions = ["01", "03", "04", "05", "06", "08"]  # neutral, happy, sad, angry, fearful, surprised
    
    for emotion in emotions:
        # Generate random audio (white noise with different frequencies for different emotions)
        sr = 16000
        duration = 3  # seconds
        t = np.linspace(0, duration, sr * duration)
        
        # Add some variation based on emotion
        freq = 200 + int(emotion) * 50
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
        audio += 0.1 * np.random.randn(len(audio))
        audio = (audio * 32767).astype(np.int16)
        
        # Save with RAVDESS naming convention
        filename = f"03-01-{emotion}-01-01-01-99.wav"
        filepath = sample_dir / filename
        wavfile.write(str(filepath), sr, audio)
    
    print(f"✓ Created {len(emotions)} sample audio files in {sample_dir}")
    print("  (These are synthetic samples for testing only)")


def main():
    """Main setup function."""
    print("=" * 60)
    print("Speech Emotion Recognition - Dataset Setup")
    print("=" * 60)
    
    # Setup directories
    setup_directories()
    
    # Check for existing dataset
    if verify_dataset():
        print("\nDataset is ready for training!")
    else:
        download_ravdess_sample()
        
        # Ask user what to do
        print("\nOptions:")
        print("1. Download RAVDESS manually and run this script again")
        print("2. Create sample data for testing (type 'sample')")
        
        try:
            choice = input("\nYour choice (sample/skip): ").strip().lower()
            if choice == "sample":
                create_sample_data()
                print("\n✓ Sample data created. You can now test the pipeline.")
        except:
            print("\nSkipping sample data creation.")
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
