"""
Example: MP4 Export with Visualizations

This example demonstrates how to export audio with synchronized
video visualizations to MP4 format.
"""

import numpy as np
from algorythm.synth import Synth, Filter, ADSR
from algorythm.export import Exporter
from algorythm.visualization import (
    WaveformVisualizer,
    SpectrogramVisualizer,
    FrequencyScopeVisualizer,
    CircularVisualizer
)


def example_basic_mp4():
    """Export a simple synth sound with waveform visualization."""
    print("=" * 60)
    print("Example 1: Basic MP4 Export with Waveform")
    print("=" * 60)
    
    # Create a synth sound
    synth = Synth(
        waveform='saw',
        filter=Filter.lowpass(cutoff=2000, resonance=0.6),
        envelope=ADSR(attack=0.1, decay=0.2, sustain=0.7, release=0.5)
    )
    
    # Generate a simple melody
    notes = [440, 494, 523, 587, 523, 494, 440]  # A, B, C, D, C, B, A
    duration = 0.5
    signal_parts = []
    
    for freq in notes:
        note = synth.generate_note(frequency=freq, duration=duration)
        signal_parts.append(note)
    
    # Combine into one signal
    signal = np.concatenate(signal_parts)
    
    # Create visualizer
    visualizer = WaveformVisualizer(sample_rate=44100, window_size=2048)
    
    # Export as MP4
    exporter = Exporter()
    exporter.export(
        signal,
        'melody_waveform.mp4',
        sample_rate=44100,
        visualizer=visualizer,
        video_width=1280,
        video_height=720,
        video_fps=30
    )
    print("\n✓ Exported to ~/Music/melody_waveform.mp4\n")


def example_spectrogram_mp4():
    """Export with spectrogram visualization."""
    print("=" * 60)
    print("Example 2: MP4 Export with Spectrogram")
    print("=" * 60)
    
    # Create a frequency sweep
    sample_rate = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Linear frequency sweep from 200 Hz to 2000 Hz
    start_freq = 200
    end_freq = 2000
    freq = np.linspace(start_freq, end_freq, len(t))
    signal = np.sin(2 * np.pi * np.cumsum(freq) / sample_rate)
    
    # Apply envelope
    envelope = np.ones_like(signal)
    fade_samples = int(0.1 * sample_rate)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    signal *= envelope * 0.5
    
    # Create spectrogram visualizer
    visualizer = SpectrogramVisualizer(
        sample_rate=sample_rate,
        window_size=2048,
        hop_size=512
    )
    
    # Export as MP4
    exporter = Exporter()
    exporter.export(
        signal,
        'frequency_sweep_spectrogram.mp4',
        sample_rate=sample_rate,
        visualizer=visualizer,
        video_width=1920,
        video_height=1080,
        video_fps=30
    )
    print("\n✓ Exported to ~/Music/frequency_sweep_spectrogram.mp4\n")


def example_frequency_scope_mp4():
    """Export with frequency scope visualization."""
    print("=" * 60)
    print("Example 3: MP4 Export with Frequency Scope")
    print("=" * 60)
    
    # Create a chord progression
    synth = Synth(waveform='sine')
    sample_rate = 44100
    
    # Chord frequencies (C major, F major, G major, C major)
    chords = [
        [261.63, 329.63, 392.00],  # C E G
        [349.23, 440.00, 523.25],  # F A C
        [392.00, 493.88, 587.33],  # G B D
        [261.63, 329.63, 392.00],  # C E G
    ]
    
    signal_parts = []
    for chord in chords:
        # Generate each note in the chord
        chord_signal = np.zeros(int(sample_rate * 1.0))
        for freq in chord:
            note = synth.generate_note(frequency=freq, duration=1.0)
            chord_signal += note[:len(chord_signal)]
        chord_signal /= len(chord)
        signal_parts.append(chord_signal)
    
    signal = np.concatenate(signal_parts)
    
    # Create frequency scope visualizer
    visualizer = FrequencyScopeVisualizer(
        sample_rate=sample_rate,
        fft_size=4096,
        freq_range=(50, 2000)
    )
    
    # Export as MP4
    exporter = Exporter()
    exporter.export(
        signal,
        'chord_progression_spectrum.mp4',
        sample_rate=sample_rate,
        visualizer=visualizer,
        video_width=1280,
        video_height=720,
        video_fps=30
    )
    print("\n✓ Exported to ~/Music/chord_progression_spectrum.mp4\n")


def example_circular_mp4():
    """Export with circular visualization."""
    print("=" * 60)
    print("Example 4: MP4 Export with Circular Visualizer")
    print("=" * 60)
    
    # Create a rhythmic pattern
    synth = Synth(waveform='square', envelope=ADSR(attack=0.01, decay=0.1, sustain=0.3, release=0.2))
    sample_rate = 44100
    
    # Create a simple beat pattern
    beat_freqs = [100, 150, 200, 150] * 4  # Bass-like pattern
    signal_parts = []
    
    for freq in beat_freqs:
        note = synth.generate_note(frequency=freq, duration=0.125)
        signal_parts.append(note)
    
    signal = np.concatenate(signal_parts)
    
    # Create circular visualizer
    visualizer = CircularVisualizer(
        sample_rate=sample_rate,
        num_bars=64,
        inner_radius=0.3,
        bar_width=0.8,
        smoothing=0.7
    )
    
    # Export as MP4
    exporter = Exporter()
    exporter.export(
        signal,
        'beat_pattern_circular.mp4',
        sample_rate=sample_rate,
        visualizer=visualizer,
        video_width=1080,
        video_height=1080,  # Square format for circular viz
        video_fps=30
    )
    print("\n✓ Exported to ~/Music/beat_pattern_circular.mp4\n")


def example_no_visualizer():
    """Export MP4 without specifying a visualizer (uses default)."""
    print("=" * 60)
    print("Example 5: MP4 Export with Default Visualizer")
    print("=" * 60)
    
    # Create a simple tone
    sample_rate = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.sin(2 * np.pi * 440 * t) * 0.5
    
    # Export as MP4 without specifying visualizer (will use WaveformVisualizer)
    exporter = Exporter()
    exporter.export(
        signal,
        'simple_tone_default.mp4',
        sample_rate=sample_rate,
        video_width=1280,
        video_height=720,
        video_fps=30
    )
    print("\n✓ Exported to ~/Music/simple_tone_default.mp4\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("MP4 Export Examples for Algorythm")
    print("=" * 60)
    print("\nThese examples demonstrate exporting audio with")
    print("synchronized video visualizations to MP4 format.")
    print("\nNote: Requires opencv-python and ffmpeg installed:")
    print("  pip install opencv-python")
    print("  (and ffmpeg on your system)")
    print("=" * 60 + "\n")
    
    # Run examples
    try:
        example_basic_mp4()
        example_spectrogram_mp4()
        example_frequency_scope_mp4()
        example_circular_mp4()
        example_no_visualizer()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("Check your ~/Music directory for the generated MP4 files.")
        print("=" * 60 + "\n")
        
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease install required dependencies:")
        print("  pip install opencv-python")
        print("  (and ensure ffmpeg is installed on your system)")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nSome examples may have failed. Check the error messages above.")
