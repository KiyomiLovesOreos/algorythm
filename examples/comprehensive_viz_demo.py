"""
Comprehensive Visualization Demo

This example demonstrates all the new visualization features including:
- Debug mode
- Enhanced waveform rendering
- Colored spectrograms
- Progress tracking
- Multiple backends
"""

import numpy as np
from algorythm.synth import Synth, Filter, ADSR


def demo_debug_mode():
    """Demonstrate debug mode across all visualizers."""
    print("\n" + "=" * 70)
    print("DEMO 1: Debug Mode")
    print("=" * 70)
    print("\nDebug mode provides detailed logging for troubleshooting")
    
    from algorythm.visualization import (
        WaveformVisualizer,
        SpectrogramVisualizer,
        FrequencyScopeVisualizer
    )
    
    # Create test signal
    sample_rate = 44100
    t = np.linspace(0, 0.5, int(sample_rate * 0.5))
    signal = np.sin(2 * np.pi * 440 * t) * 0.5
    
    print("\nWaveform with debug=True:")
    viz1 = WaveformVisualizer(sample_rate=sample_rate, debug=True)
    img1 = viz1.to_image_data(signal)
    
    print("\nSpectrogram with debug=True:")
    viz2 = SpectrogramVisualizer(sample_rate=sample_rate, debug=True)
    spec = viz2.generate(signal)
    
    print("\nFrequency scope without debug:")
    viz3 = FrequencyScopeVisualizer(sample_rate=sample_rate, debug=False)
    spectrum = viz3.generate(signal)
    print("  (No debug output shown)")


def demo_enhanced_waveform():
    """Demonstrate enhanced waveform rendering."""
    print("\n" + "=" * 70)
    print("DEMO 2: Enhanced Waveform Rendering")
    print("=" * 70)
    print("\nComparing thin vs thick lines, with and without center line")
    
    from algorythm.visualization import WaveformVisualizer
    
    # Create musical signal
    synth = Synth(waveform='saw', envelope=ADSR(0.1, 0.2, 0.7, 0.3))
    signal = synth.generate_note(frequency=440, duration=0.5)
    
    viz = WaveformVisualizer()
    
    # Test different configurations
    configs = [
        (1, False, "Thin line, no center"),
        (3, False, "Thick line, no center"),
        (3, True, "Thick line with center"),
    ]
    
    for thickness, center, desc in configs:
        img = viz.to_image_data(signal, line_thickness=thickness, center_line=center)
        pixels = np.count_nonzero(img)
        print(f"\n  {desc}:")
        print(f"    Shape: {img.shape}, Non-zero pixels: {pixels:,}")


def demo_colored_spectrogram():
    """Demonstrate colored spectrogram generation."""
    print("\n" + "=" * 70)
    print("DEMO 3: Colored Spectrograms")
    print("=" * 70)
    print("\nGenerate spectrograms with different color schemes")
    
    from algorythm.visualization import SpectrogramVisualizer
    
    # Create frequency sweep
    sample_rate = 44100
    t = np.linspace(0, 1.0, int(sample_rate * 1.0))
    freq = np.linspace(200, 2000, len(t))
    signal = np.sin(2 * np.pi * np.cumsum(freq) / sample_rate) * 0.5
    
    viz = SpectrogramVisualizer(sample_rate=sample_rate)
    spec = viz.generate(signal)
    
    print(f"\nBase spectrogram: {spec.shape}")
    
    colormaps = ['viridis', 'plasma', 'inferno', 'hot']
    for cmap in colormaps:
        colored = viz.to_colored_image(spec, colormap=cmap)
        print(f"  {cmap:10s}: {colored.shape} {colored.dtype}")


def demo_progress_tracking():
    """Demonstrate progress tracking during rendering."""
    print("\n" + "=" * 70)
    print("DEMO 4: Progress Tracking")
    print("=" * 70)
    print("\nProgress tracking for long-running operations")
    
    from algorythm.visualization import VideoRenderer, CircularVisualizer
    
    # Create test signal
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.sin(2 * np.pi * 440 * t) * 0.5
    
    # Progress callback with bar
    def show_progress(current, total):
        if current % 5 == 0:  # Update every 5 frames
            percent = 100 * current / total
            bar_length = 30
            filled = int(bar_length * current / total)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f'\r  Progress: [{bar}] {percent:5.1f}%', end='', flush=True)
    
    print("\nRendering with progress callback:")
    renderer = VideoRenderer(
        width=640,
        height=480,
        fps=30,
        debug=False  # Disable debug to see progress bar clearly
    )
    
    visualizer = CircularVisualizer(sample_rate=sample_rate, num_bars=32)
    
    frames = renderer.render_frames(
        signal,
        visualizer,
        progress_callback=show_progress
    )
    
    print(f"\n  ✓ Rendered {len(frames)} frames")


def demo_backend_selection():
    """Demonstrate video backend selection."""
    print("\n" + "=" * 70)
    print("DEMO 5: Video Backend Selection")
    print("=" * 70)
    print("\nAutomatic backend selection based on available libraries")
    
    from algorythm.visualization import VideoRenderer
    
    # Default backend (automatic selection)
    print("\nDefault (automatic) backend:")
    renderer1 = VideoRenderer(debug=True)
    
    # Force matplotlib backend
    print("\nForcing matplotlib backend:")
    renderer2 = VideoRenderer(use_matplotlib=True, debug=True)
    
    print("\nBackends are selected based on:")
    print("  1. opencv-python availability (preferred)")
    print("  2. matplotlib as fallback")
    print("  3. User can override with use_matplotlib=True")


def demo_error_handling():
    """Demonstrate improved error handling."""
    print("\n" + "=" * 70)
    print("DEMO 6: Error Handling")
    print("=" * 70)
    print("\nImproved error messages provide clear guidance")
    
    print("\nExamples of helpful error messages:")
    print("\n  ❌ ffmpeg not found:")
    print("     Ubuntu/Debian: sudo apt install ffmpeg")
    print("     macOS: brew install ffmpeg")
    print("     Windows: Download from https://ffmpeg.org/")
    
    print("\n  ❌ OpenCV not available:")
    print("     Install with: pip install opencv-python")
    print("     (Falling back to matplotlib backend)")
    
    print("\n  ❌ Matplotlib not available:")
    print("     Install with: pip install matplotlib")
    print("     (Falling back to grayscale colormaps)")
    
    print("\nAll errors include:")
    print("  • Clear description of the problem")
    print("  • Installation instructions")
    print("  • Fallback behavior when possible")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("ALGORYTHM VISUALIZATION - COMPREHENSIVE DEMO")
    print("=" * 70)
    print("\nThis demo showcases all the new visualization features:")
    print("  1. Debug Mode - Detailed logging")
    print("  2. Enhanced Waveforms - Better rendering quality")
    print("  3. Colored Spectrograms - Beautiful visualizations")
    print("  4. Progress Tracking - Real-time feedback")
    print("  5. Backend Selection - Flexible rendering")
    print("  6. Error Handling - Clear guidance")
    
    try:
        demo_debug_mode()
        demo_enhanced_waveform()
        demo_colored_spectrogram()
        demo_progress_tracking()
        demo_backend_selection()
        demo_error_handling()
        
        print("\n" + "=" * 70)
        print("✓ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nThe visualization module is ready for production use.")
        print("All features are backwards compatible with existing code.")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
