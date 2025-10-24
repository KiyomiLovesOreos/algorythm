"""
Complete MP4 Export Demo - All Visualizers

This example demonstrates MP4 video export with all available visualizers.
"""

import numpy as np
from algorythm.synth import Synth, Filter, ADSR
from algorythm.export import Exporter
from algorythm.visualization import (
    WaveformVisualizer,
    SpectrogramVisualizer,
    FrequencyScopeVisualizer,
    CircularVisualizer,
    OscilloscopeVisualizer,
    ParticleVisualizer
)


def create_musical_signal():
    """Create a musical signal for visualization."""
    print("Creating musical signal...")
    
    synth = Synth(
        waveform='saw',
        filter=Filter.lowpass(cutoff=2000, resonance=0.6),
        envelope=ADSR(attack=0.1, decay=0.2, sustain=0.7, release=0.5)
    )
    
    # Generate a melody
    notes = [440, 494, 523, 587, 659, 587, 523, 494]  # A, B, C, D, E, D, C, B
    duration = 0.5
    signal_parts = []
    
    for freq in notes:
        note = synth.generate_note(frequency=freq, duration=duration)
        signal_parts.append(note)
    
    signal = np.concatenate(signal_parts)
    print(f"✓ Created {len(signal)/44100:.2f}s audio signal")
    return signal


def export_with_all_visualizers():
    """Export MP4 files with all visualizers."""
    print("\n" + "=" * 70)
    print("MP4 Export Demo - All Visualizers")
    print("=" * 70)
    
    signal = create_musical_signal()
    exporter = Exporter()
    sample_rate = 44100
    
    visualizers = [
        ("waveform", WaveformVisualizer(sample_rate=sample_rate), 1280, 720),
        ("spectrogram", SpectrogramVisualizer(sample_rate=sample_rate), 1280, 720),
        ("frequency_scope", FrequencyScopeVisualizer(sample_rate=sample_rate), 1280, 720),
        ("circular", CircularVisualizer(sample_rate=sample_rate, num_bars=64), 1080, 1080),
        ("oscilloscope", OscilloscopeVisualizer(sample_rate=sample_rate, mode='waveform'), 1280, 720),
        ("particle", ParticleVisualizer(sample_rate=sample_rate, num_particles=100), 1280, 720),
    ]
    
    print(f"\nExporting {len(visualizers)} different visualizations...")
    print()
    
    for name, viz, width, height in visualizers:
        print(f"Exporting {name}...")
        filename = f'demo_{name}.mp4'
        
        try:
            exporter.export(
                signal,
                filename,
                sample_rate=sample_rate,
                visualizer=viz,
                video_width=width,
                video_height=height,
                video_fps=30
            )
            print(f"  ✓ {filename} complete\n")
        except Exception as e:
            print(f"  ✗ {filename} failed: {e}\n")
    
    print("=" * 70)
    print("✓ All exports complete!")
    print("=" * 70)
    print("\nGenerated files in ~/Music/:")
    print("  • demo_waveform.mp4 - Traditional waveform display")
    print("  • demo_spectrogram.mp4 - Frequency content over time")
    print("  • demo_frequency_scope.mp4 - Real-time frequency bars")
    print("  • demo_circular.mp4 - Circular/radial frequency display")
    print("  • demo_oscilloscope.mp4 - Oscilloscope-style waveform")
    print("  • demo_particle.mp4 - Particle-based reactive visualization")
    print("=" * 70 + "\n")


def main():
    """Run the complete demo."""
    print("\n" + "=" * 70)
    print("ALGORYTHM - Complete MP4 Visualization Export Demo")
    print("=" * 70)
    print("\nThis demo exports audio with 6 different visualization styles.")
    print("Each creates a synchronized MP4 video file.")
    print("=" * 70)
    
    try:
        export_with_all_visualizers()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
