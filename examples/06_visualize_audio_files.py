"""
Example: Visualize Existing Audio Files

Learn how to load MP3/WAV/OGG files and create visualizations from them.
"""

import numpy as np
from algorythm.audio_loader import load_audio, visualize_audio_file, AudioFile
from algorythm.visualization import (
    WaveformVisualizer,
    CircularVisualizer,
    SpectrogramVisualizer,
    FrequencyScopeVisualizer
)
from algorythm.synth import Synth
from algorythm.export import Exporter


def create_sample_audio():
    """Create a sample audio file for demonstration."""
    print("\n" + "=" * 60)
    print("Creating Sample Audio File")
    print("=" * 60)
    
    # Create a simple melody
    synth = Synth(waveform='saw')
    melody = [261.63, 293.66, 329.63, 392.00, 440.00, 392.00, 329.63, 261.63]
    
    signal_parts = []
    for freq in melody:
        note = synth.generate_note(frequency=freq, duration=0.5)
        signal_parts.append(note)
    
    signal = np.concatenate(signal_parts)
    
    # Export as WAV to home Music directory
    from pathlib import Path
    output_path = Path.home() / 'Music' / 'sample_audio.wav'
    
    exporter = Exporter()
    exporter.export(signal, str(output_path), sample_rate=44100)
    
    print(f"✓ Created {output_path}")
    return str(output_path)


def example_1_load_and_visualize():
    """Load audio and create visualization (simple way)."""
    print("\n" + "=" * 60)
    print("Example 1: Quick Visualization")
    print("=" * 60)
    
    # Create sample audio
    audio_file = create_sample_audio()
    
    # Create visualizer
    viz = CircularVisualizer(sample_rate=44100, num_bars=64)
    
    # One-line visualization!
    print("\nCreating visualization video...")
    visualize_audio_file(
        input_file=audio_file,
        output_file='visualized_quick.mp4',
        visualizer=viz,
        video_width=1280,
        video_height=720,
        video_fps=30
    )
    
    print("✓ Created visualized_quick.mp4")


def example_2_audio_file_class():
    """Use AudioFile class for more control."""
    print("\n" + "=" * 60)
    print("Example 2: AudioFile Class")
    print("=" * 60)
    
    # Create sample audio
    audio_file = create_sample_audio()
    
    # Load audio
    print("\nLoading audio file...")
    audio = AudioFile(audio_file)
    
    print(f"✓ Loaded: {audio}")
    print(f"  Duration: {audio.duration:.2f}s")
    print(f"  Sample rate: {audio.sample_rate}Hz")
    print(f"  Samples: {audio.num_samples:,}")
    
    # Create waveform visualization
    print("\nCreating waveform visualization...")
    viz = WaveformVisualizer(sample_rate=audio.sample_rate)
    audio.visualize(
        'visualized_waveform.mp4',
        visualizer=viz,
        video_width=1920,
        video_height=1080,
        video_fps=30
    )
    
    print("✓ Created visualized_waveform.mp4")


def example_3_load_partial():
    """Load only part of an audio file."""
    print("\n" + "=" * 60)
    print("Example 3: Load Partial Audio")
    print("=" * 60)
    
    # Create sample audio
    audio_file = create_sample_audio()
    
    # Load only middle 2 seconds
    print("\nLoading middle 2 seconds...")
    audio = AudioFile(audio_file, offset=1.0, duration=2.0)
    
    print(f"✓ Loaded: {audio}")
    print(f"  Duration: {audio.duration:.2f}s")
    
    # Create spectrogram
    print("\nCreating spectrogram visualization...")
    viz = SpectrogramVisualizer(sample_rate=audio.sample_rate)
    audio.visualize(
        'visualized_partial.mp4',
        visualizer=viz,
        video_width=1280,
        video_height=720
    )
    
    print("✓ Created visualized_partial.mp4")


def example_4_multiple_visualizations():
    """Create multiple visualizations from same audio."""
    print("\n" + "=" * 60)
    print("Example 4: Multiple Visualizations")
    print("=" * 60)
    
    # Create sample audio
    audio_file = create_sample_audio()
    
    # Load once
    print("\nLoading audio file...")
    audio = AudioFile(audio_file)
    
    # Create multiple visualizations
    visualizers = [
        ('waveform', WaveformVisualizer(sample_rate=audio.sample_rate)),
        ('circular', CircularVisualizer(sample_rate=audio.sample_rate, num_bars=64)),
        ('spectrum', FrequencyScopeVisualizer(sample_rate=audio.sample_rate)),
    ]
    
    for name, viz in visualizers:
        print(f"\nCreating {name} visualization...")
        audio.visualize(
            f'multi_{name}.mp4',
            visualizer=viz,
            video_width=1280,
            video_height=720,
            video_fps=30
        )
        print(f"  ✓ Created multi_{name}.mp4")
    
    print("\n✓ All visualizations created")


def example_5_manual_control():
    """Manual control over loading and visualization."""
    print("\n" + "=" * 60)
    print("Example 5: Manual Control")
    print("=" * 60)
    
    # Create sample audio
    audio_file = create_sample_audio()
    
    # Load audio manually
    print("\nLoading audio with manual control...")
    signal, sample_rate = load_audio(
        audio_file,
        target_sample_rate=44100,  # Force specific sample rate
        duration=3.0,              # First 3 seconds
        offset=0.5                 # Skip first 0.5s
    )
    
    print(f"✓ Loaded {len(signal)/sample_rate:.2f}s at {sample_rate}Hz")
    
    # Create custom visualization
    print("\nCreating custom visualization...")
    exporter = Exporter()
    viz = CircularVisualizer(
        sample_rate=sample_rate,
        num_bars=128,              # More bars
        inner_radius=0.2           # Smaller center
    )
    
    exporter.export(
        signal,
        'visualized_custom.mp4',
        sample_rate=sample_rate,
        visualizer=viz,
        video_width=1080,
        video_height=1080,         # Square
        video_fps=60               # High framerate
    )
    
    print("✓ Created visualized_custom.mp4")


def main():
    print("\n" + "=" * 60)
    print("VISUALIZE EXISTING AUDIO FILES")
    print("=" * 60)
    print("\nCreate music videos from MP3, WAV, OGG, or FLAC files!")
    
    print("\n" + "=" * 60)
    print("NOTE: These examples use a generated WAV file.")
    print("To use your own MP3 files:")
    print("  1. Install pydub: pip install pydub")
    print("  2. Replace 'sample_audio.wav' with your MP3 path")
    print("=" * 60)
    
    try:
        example_1_load_and_visualize()
        example_2_audio_file_class()
        example_3_load_partial()
        example_4_multiple_visualizations()
        example_5_manual_control()
        
        print("\n" + "=" * 60)
        print("✓ All examples complete!")
        print("=" * 60)
        print("\nCreated files in ~/Music/:")
        print("  • sample_audio.wav")
        print("  • visualized_quick.mp4")
        print("  • visualized_waveform.mp4")
        print("  • visualized_partial.mp4")
        print("  • multi_waveform.mp4")
        print("  • multi_circular.mp4")
        print("  • multi_spectrum.mp4")
        print("  • visualized_custom.mp4")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nIf you see 'pydub' errors, install it with:")
        print("  pip install pydub")


if __name__ == '__main__':
    main()
