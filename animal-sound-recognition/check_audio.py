import librosa
import numpy as np
import matplotlib.pyplot as plt

def analyze_audio(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None, mono=True)
    
    # Print basic info
    duration = len(y) / sr
    print(f"File: {file_path}")
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Samples: {len(y)}")
    
    # Calculate and print statistics
    print("\nAudio Statistics:")
    print(f"  Min: {np.min(y):.6f}")
    print(f"  Max: {np.max(y):.6f}")
    print(f"  Mean: {np.mean(y):.6f}")
    print(f"  RMS: {np.sqrt(np.mean(np.square(y))):.6f}")
    
    # Plot the waveform
    plt.figure(figsize=(12, 4))
    plt.plot(np.linspace(0, duration, len(y)), y)
    plt.title("Audio Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    
    # Save the plot
    output_file = file_path.replace('.mp3', '_waveform.png')
    plt.savefig(output_file)
    print(f"\nWaveform plot saved to: {output_file}")

if __name__ == "__main__":
    audio_file = "data/BatA.mp3"
    analyze_audio(audio_file)
