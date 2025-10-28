"""
Test MP4 Export with Visualization

This tests if MP4 export actually creates a video file with visualization.
"""

import numpy as np
from algorythm.synth import Synth, Filter, ADSR
from algorythm.export import Exporter
from algorythm.visualization import WaveformVisualizer, CircularVisualizer


def test_mp4_export():
    """Test MP4 export functionality."""
    print("\n" + "=" * 70)
    print("Testing MP4 Export with Visualization")
    print("=" * 70)
    
    # Create a simple musical signal
    print("\n1. Creating audio signal...")
    synth = Synth(
        waveform='saw',
        filter=Filter.lowpass(cutoff=2000, resonance=0.6),
        envelope=ADSR(attack=0.1, decay=0.2, sustain=0.7, release=0.5)
    )
    
    # Generate a short melody
    notes = [440, 494, 523, 587]  # A, B, C, D
    duration = 0.5
    signal_parts = []
    
    for freq in notes:
        note = synth.generate_note(frequency=freq, duration=duration)
        signal_parts.append(note)
    
    signal = np.concatenate(signal_parts)
    print(f"   ✓ Generated {len(signal)} samples ({len(signal)/44100:.2f}s)")
    
    # Test 1: Export with WaveformVisualizer
    print("\n2. Exporting MP4 with WaveformVisualizer...")
    exporter = Exporter()
    waveform_viz = WaveformVisualizer(sample_rate=44100, debug=False)
    
    exporter.export(
        signal,
        'test_waveform.mp4',
        sample_rate=44100,
        visualizer=waveform_viz,
        video_width=1280,
        video_height=720,
        video_fps=30
    )
    print("   ✓ Waveform MP4 export complete")
    
    # Test 2: Export with CircularVisualizer
    print("\n3. Exporting MP4 with CircularVisualizer...")
    circular_viz = CircularVisualizer(sample_rate=44100, num_bars=64)
    
    exporter.export(
        signal,
        'test_circular.mp4',
        sample_rate=44100,
        visualizer=circular_viz,
        video_width=1080,
        video_height=1080,
        video_fps=30
    )
    print("   ✓ Circular MP4 export complete")
    
    # Test 3: Export without visualizer (should use default)
    print("\n4. Exporting MP4 with default visualizer...")
    exporter.export(
        signal,
        'test_default.mp4',
        sample_rate=44100,
        video_width=1280,
        video_height=720,
        video_fps=30
    )
    print("   ✓ Default MP4 export complete")
    
    print("\n" + "=" * 70)
    print("✓ All MP4 exports completed!")
    print("=" * 70)
    print("\nCheck ~/Music directory for the following files:")
    print("  - test_waveform.mp4 (with waveform visualization)")
    print("  - test_circular.mp4 (with circular visualization)")
    print("  - test_default.mp4 (with default visualization)")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    try:
        test_mp4_export()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
