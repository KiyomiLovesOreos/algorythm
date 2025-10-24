"""
Test FrequencyScopeVisualizer and SpectrogramVisualizer

Verify that both actually visualize audio properly.
"""

import numpy as np
from algorythm.synth import Synth
from algorythm.visualization import FrequencyScopeVisualizer, SpectrogramVisualizer
from algorythm.export import Exporter


def test_frequency_scope():
    """Test frequency scope visualization."""
    print("\n" + "=" * 60)
    print("Testing FrequencyScopeVisualizer")
    print("=" * 60)
    
    # Create signal with known frequencies
    sample_rate = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Mix of 220Hz, 440Hz, 880Hz
    signal = (
        np.sin(2 * np.pi * 220 * t) +
        np.sin(2 * np.pi * 440 * t) +
        np.sin(2 * np.pi * 880 * t)
    ) / 3.0
    
    # Test visualization
    viz = FrequencyScopeVisualizer(
        sample_rate=sample_rate,
        fft_size=2048,
        freq_range=(50, 2000),
        debug=True
    )
    
    # Generate spectrum
    print("\nGenerating spectrum...")
    spectrum = viz.generate(signal[:4410])  # First 0.1s
    print(f"Spectrum shape: {spectrum.shape}")
    print(f"Spectrum range: {spectrum.min():.2f} to {spectrum.max():.2f} dB")
    
    # Generate image
    print("\nGenerating image...")
    image = viz.to_image_data(signal[:4410], height=480, width=640)
    print(f"Image shape: {image.shape}")
    print(f"Image range: {image.min():.3f} to {image.max():.3f}")
    print(f"Non-zero pixels: {np.count_nonzero(image)}/{image.size}")
    
    if np.count_nonzero(image) > 0:
        print("✓ Frequency scope has visualization data")
    else:
        print("✗ Frequency scope is empty!")
    
    # Export as video
    print("\nExporting video...")
    exporter = Exporter()
    exporter.export(
        signal,
        'test_frequency_scope.mp4',
        sample_rate=sample_rate,
        visualizer=viz,
        video_width=1280,
        video_height=720,
        video_fps=30
    )
    print("✓ Video exported")


def test_spectrogram():
    """Test spectrogram visualization."""
    print("\n" + "=" * 60)
    print("Testing SpectrogramVisualizer")
    print("=" * 60)
    
    # Create signal with frequency sweep
    sample_rate = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Frequency sweep from 200Hz to 2000Hz
    freq_sweep = 200 + (1800 * t / duration)
    signal = np.sin(2 * np.pi * freq_sweep * t)
    
    # Test visualization
    viz = SpectrogramVisualizer(
        sample_rate=sample_rate,
        window_size=2048,
        hop_size=512,
        debug=True
    )
    
    # Generate spectrogram
    print("\nGenerating spectrogram...")
    spectrogram = viz.generate(signal)
    print(f"Spectrogram shape: {spectrogram.shape}")
    print(f"Spectrogram range: {spectrogram.min():.2f} to {spectrogram.max():.2f} dB")
    
    # Generate image
    print("\nGenerating image...")
    image = viz.to_image_data(signal, height=480, width=640)
    print(f"Image shape: {image.shape}")
    print(f"Image range: {image.min():.3f} to {image.max():.3f}")
    print(f"Non-zero pixels: {np.count_nonzero(image)}/{image.size}")
    
    if np.count_nonzero(image) > 0:
        print("✓ Spectrogram has visualization data")
    else:
        print("✗ Spectrogram is empty!")
    
    # Export as video
    print("\nExporting video...")
    exporter = Exporter()
    exporter.export(
        signal,
        'test_spectrogram.mp4',
        sample_rate=sample_rate,
        visualizer=viz,
        video_width=1280,
        video_height=720,
        video_fps=30
    )
    print("✓ Video exported")


def main():
    print("\n" + "=" * 60)
    print("FREQUENCY VISUALIZATION FIX TEST")
    print("=" * 60)
    print("\nTesting if frequency scope and spectrogram actually visualize audio")
    
    test_frequency_scope()
    test_spectrogram()
    
    print("\n" + "=" * 60)
    print("✓ All tests complete!")
    print("=" * 60)
    print("\nCheck ~/Music/ for:")
    print("  - test_frequency_scope.mp4")
    print("  - test_spectrogram.mp4")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
