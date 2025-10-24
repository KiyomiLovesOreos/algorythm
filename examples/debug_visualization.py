"""
Example: Debug Visualization Features

This example demonstrates the debug features and improvements
in the visualization module.
"""

import numpy as np
from algorythm.synth import Synth, Filter, ADSR
from algorythm.visualization import (
    WaveformVisualizer,
    SpectrogramVisualizer,
    FrequencyScopeVisualizer,
    CircularVisualizer,
    VideoRenderer
)


def test_visualizer_output():
    """Test visualizers with debug output."""
    print("=" * 60)
    print("Testing Visualizer Debug Output")
    print("=" * 60)
    
    # Create test signal
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a complex signal with multiple frequencies
    signal = (
        np.sin(2 * np.pi * 440 * t) * 0.3 +  # A4
        np.sin(2 * np.pi * 554.37 * t) * 0.2 +  # C#5
        np.sin(2 * np.pi * 659.25 * t) * 0.1    # E5
    )
    
    print("\n1. Testing WaveformVisualizer with debug mode:")
    waveform_viz = WaveformVisualizer(sample_rate=sample_rate, debug=True)
    waveform_data = waveform_viz.generate(signal)
    waveform_img = waveform_viz.to_image_data(signal, height=480, width=1280)
    print(f"   ✓ Waveform data shape: {waveform_data.shape}")
    print(f"   ✓ Waveform image shape: {waveform_img.shape}")
    
    print("\n2. Testing SpectrogramVisualizer with debug mode:")
    spec_viz = SpectrogramVisualizer(sample_rate=sample_rate, debug=True)
    spec_data = spec_viz.generate(signal)
    print(f"   ✓ Spectrogram shape: {spec_data.shape}")
    print(f"   ✓ Frequency range: {spec_viz.get_frequency_axis()[0]:.1f} - {spec_viz.get_frequency_axis()[-1]:.1f} Hz")
    
    print("\n3. Testing FrequencyScopeVisualizer:")
    freq_viz = FrequencyScopeVisualizer(sample_rate=sample_rate, debug=False)
    spectrum = freq_viz.generate(signal)
    freqs, mags = freq_viz.filter_frequency_range(spectrum)
    print(f"   ✓ Spectrum length: {len(spectrum)}")
    print(f"   ✓ Filtered frequency range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz")
    
    print("\n4. Testing CircularVisualizer:")
    circular_viz = CircularVisualizer(sample_rate=sample_rate, num_bars=64, debug=False)
    circular_data = circular_viz.generate(signal)
    circular_img = circular_viz.to_image_data(signal, height=512, width=512)
    print(f"   ✓ Circular data shape: {circular_data.shape}")
    print(f"   ✓ Circular image shape: {circular_img.shape}")
    
    print("\n" + "=" * 60)
    print("All visualizers tested successfully!")
    print("=" * 60)


def test_video_renderer_progress():
    """Test video renderer with progress tracking."""
    print("\n" + "=" * 60)
    print("Testing Video Renderer with Progress Tracking")
    print("=" * 60)
    
    # Create a simple test signal
    sample_rate = 44100
    duration = 2.0  # Short duration for quick test
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.sin(2 * np.pi * 440 * t) * 0.5
    
    # Progress callback
    def progress_callback(current, total):
        percent = 100 * current / total
        bar_length = 40
        filled = int(bar_length * current / total)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f'\r   Progress: [{bar}] {percent:.1f}%', end='', flush=True)
    
    print("\nRendering video with progress tracking...")
    
    # Create renderer with debug mode
    renderer = VideoRenderer(
        width=1280,
        height=720,
        fps=30,
        sample_rate=sample_rate,
        debug=True
    )
    
    # Create visualizer
    visualizer = WaveformVisualizer(sample_rate=sample_rate, debug=True)
    
    # Render frames with progress callback
    frames = renderer.render_frames(
        signal,
        visualizer,
        progress_callback=progress_callback
    )
    
    print(f"\n   ✓ Rendered {len(frames)} frames")
    print("=" * 60)


def test_enhanced_waveform_rendering():
    """Test enhanced waveform rendering with thicker lines and center line."""
    print("\n" + "=" * 60)
    print("Testing Enhanced Waveform Rendering")
    print("=" * 60)
    
    # Create a synth with envelope
    synth = Synth(
        waveform='saw',
        filter=Filter.lowpass(cutoff=2000, resonance=0.6),
        envelope=ADSR(attack=0.1, decay=0.2, sustain=0.7, release=0.5)
    )
    
    # Generate a melody
    notes = [440, 494, 523, 587, 523, 494, 440]
    duration = 0.5
    signal_parts = []
    
    for freq in notes:
        note = synth.generate_note(frequency=freq, duration=duration)
        signal_parts.append(note)
    
    signal = np.concatenate(signal_parts)
    
    print("\n1. Basic waveform (thin line, no center):")
    viz1 = WaveformVisualizer(debug=False)
    img1 = viz1.to_image_data(signal, height=300, width=1200, line_thickness=1, center_line=False)
    print(f"   ✓ Shape: {img1.shape}, Non-zero pixels: {np.count_nonzero(img1)}")
    
    print("\n2. Enhanced waveform (thick line, with center):")
    viz2 = WaveformVisualizer(debug=False)
    img2 = viz2.to_image_data(signal, height=300, width=1200, line_thickness=3, center_line=True)
    print(f"   ✓ Shape: {img2.shape}, Non-zero pixels: {np.count_nonzero(img2)}")
    
    print(f"   ✓ Enhanced version has {np.count_nonzero(img2) / np.count_nonzero(img1):.1f}x more pixels")
    
    print("=" * 60)


def test_colored_spectrogram():
    """Test colored spectrogram generation."""
    print("\n" + "=" * 60)
    print("Testing Colored Spectrogram")
    print("=" * 60)
    
    # Create a frequency sweep
    sample_rate = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Linear frequency sweep from 200 Hz to 2000 Hz
    start_freq = 200
    end_freq = 2000
    freq = np.linspace(start_freq, end_freq, len(t))
    signal = np.sin(2 * np.pi * np.cumsum(freq) / sample_rate) * 0.5
    
    # Create spectrogram
    viz = SpectrogramVisualizer(sample_rate=sample_rate, debug=True)
    spec_data = viz.generate(signal)
    
    print("\nGenerating colored spectrograms with different colormaps:")
    
    colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'hot', 'cool']
    for cmap in colormaps:
        try:
            colored = viz.to_colored_image(spec_data, colormap=cmap)
            print(f"   ✓ {cmap:10s}: {colored.shape} ({colored.dtype})")
        except Exception as e:
            print(f"   ✗ {cmap:10s}: {e}")
    
    print("=" * 60)


def main():
    """Run all debug tests."""
    print("\n" + "=" * 60)
    print("Algorythm Visualization Debug & Testing Suite")
    print("=" * 60)
    print("\nThis suite tests the improved visualization features")
    print("including debug output, progress tracking, and enhanced rendering.")
    print("=" * 60)
    
    try:
        test_visualizer_output()
        test_enhanced_waveform_rendering()
        test_colored_spectrogram()
        test_video_renderer_progress()
        
        print("\n" + "=" * 60)
        print("✓ All tests completed successfully!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
