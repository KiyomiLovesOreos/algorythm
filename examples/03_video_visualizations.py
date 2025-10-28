"""
Example: Video Visualizations

Create stunning MP4 videos with different visualization styles.
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


def create_music():
    """Create a musical piece for visualization."""
    print("Creating music...")
    
    synth = Synth(
        waveform='saw',
        filter=Filter.lowpass(cutoff=2000, resonance=0.6),
        envelope=ADSR(attack=0.1, decay=0.2, sustain=0.7, release=0.5)
    )
    
    # Create a chord progression
    chords = [
        [261.63, 329.63, 392.00],  # C major
        [293.66, 369.99, 440.00],  # D minor
        [392.00, 493.88, 587.33],  # G major
        [261.63, 329.63, 392.00],  # C major
    ]
    
    signal_parts = []
    for chord in chords:
        chord_signal = np.zeros(int(44100 * 1.0))
        for freq in chord:
            note = synth.generate_note(frequency=freq, duration=1.0)
            chord_signal += note[:len(chord_signal)]
        chord_signal /= len(chord)
        signal_parts.append(chord_signal)
    
    signal = np.concatenate(signal_parts)
    print(f"✓ Created {len(signal)/44100:.1f}s of music")
    return signal


def example_1_waveform_video():
    """Create a video with waveform visualization."""
    print("\n" + "=" * 60)
    print("Example 1: Waveform Video (HD 720p)")
    print("=" * 60)
    
    signal = create_music()
    
    visualizer = WaveformVisualizer(sample_rate=44100)
    exporter = Exporter()
    
    print("Rendering video...")
    exporter.export(
        signal,
        'video_waveform.mp4',
        sample_rate=44100,
        visualizer=visualizer,
        video_width=1280,
        video_height=720,
        video_fps=30
    )
    print("✓ Created video_waveform.mp4")


def example_2_circular_video():
    """Create a video with circular visualization."""
    print("\n" + "=" * 60)
    print("Example 2: Circular Video (Square)")
    print("=" * 60)
    
    signal = create_music()
    
    visualizer = CircularVisualizer(
        sample_rate=44100,
        num_bars=64,
        inner_radius=0.3
    )
    exporter = Exporter()
    
    print("Rendering video...")
    exporter.export(
        signal,
        'video_circular.mp4',
        sample_rate=44100,
        visualizer=visualizer,
        video_width=1080,
        video_height=1080,
        video_fps=30
    )
    print("✓ Created video_circular.mp4")


def example_3_spectrogram_video():
    """Create a video with spectrogram visualization."""
    print("\n" + "=" * 60)
    print("Example 3: Spectrogram Video")
    print("=" * 60)
    
    signal = create_music()
    
    visualizer = SpectrogramVisualizer(
        sample_rate=44100,
        window_size=2048,
        hop_size=512
    )
    exporter = Exporter()
    
    print("Rendering video...")
    exporter.export(
        signal,
        'video_spectrogram.mp4',
        sample_rate=44100,
        visualizer=visualizer,
        video_width=1280,
        video_height=720,
        video_fps=30
    )
    print("✓ Created video_spectrogram.mp4")


def example_4_frequency_bars():
    """Create a video with frequency bar visualization."""
    print("\n" + "=" * 60)
    print("Example 4: Frequency Bars Video")
    print("=" * 60)
    
    signal = create_music()
    
    visualizer = FrequencyScopeVisualizer(
        sample_rate=44100,
        fft_size=2048,
        freq_range=(50, 2000)
    )
    exporter = Exporter()
    
    print("Rendering video...")
    exporter.export(
        signal,
        'video_frequency_bars.mp4',
        sample_rate=44100,
        visualizer=visualizer,
        video_width=1280,
        video_height=720,
        video_fps=30
    )
    print("✓ Created video_frequency_bars.mp4")


def main():
    print("\n" + "=" * 60)
    print("VIDEO VISUALIZATION EXAMPLES")
    print("=" * 60)
    print("\nCreate beautiful MP4 videos with synchronized visualizations!")
    print("\nNote: This will take a few minutes to render all videos.")
    
    try:
        example_1_waveform_video()
        example_2_circular_video()
        example_3_spectrogram_video()
        example_4_frequency_bars()
        
        print("\n" + "=" * 60)
        print("✓ All videos complete! Check ~/Music/ for the files.")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure ffmpeg is installed and PIL/Pillow is available.")


if __name__ == '__main__':
    main()
