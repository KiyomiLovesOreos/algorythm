"""
Quick Test: Visualization Improvements

A quick test to verify the visualization improvements work correctly.
"""

import numpy as np
from algorythm.visualization import (
    WaveformVisualizer,
    SpectrogramVisualizer,
    FrequencyScopeVisualizer,
    CircularVisualizer
)


def main():
    print("\n" + "=" * 60)
    print("Quick Visualization Test")
    print("=" * 60)
    
    # Create short test signal
    sample_rate = 44100
    duration = 0.5  # Short duration
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.sin(2 * np.pi * 440 * t) * 0.5
    
    print("\n1. Testing WaveformVisualizer (with debug):")
    viz1 = WaveformVisualizer(sample_rate=sample_rate, debug=True)
    img1 = viz1.to_image_data(signal, height=480, width=1280, line_thickness=3, center_line=True)
    print(f"   ✓ Generated {img1.shape} image with {np.count_nonzero(img1)} non-zero pixels")
    
    print("\n2. Testing SpectrogramVisualizer (with debug):")
    viz2 = SpectrogramVisualizer(sample_rate=sample_rate, debug=True)
    spec = viz2.generate(signal)
    print(f"   ✓ Generated {spec.shape} spectrogram")
    
    print("\n3. Testing FrequencyScopeVisualizer:")
    viz3 = FrequencyScopeVisualizer(sample_rate=sample_rate, debug=False)
    spectrum = viz3.generate(signal)
    print(f"   ✓ Generated spectrum with {len(spectrum)} frequency bins")
    
    print("\n4. Testing CircularVisualizer:")
    viz4 = CircularVisualizer(sample_rate=sample_rate, num_bars=32, debug=False)
    circular = viz4.generate(signal)
    circular_img = viz4.to_image_data(signal, height=512, width=512)
    print(f"   ✓ Generated circular visualization: {circular_img.shape}")
    
    print("\n5. Testing colored spectrogram (without matplotlib):")
    colored = viz2.to_colored_image(spec, colormap='viridis')
    print(f"   ✓ Generated colored spectrogram: {colored.shape}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
