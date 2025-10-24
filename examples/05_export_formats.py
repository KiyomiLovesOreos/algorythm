"""
Example: Export Formats

Learn how to export audio in different formats (WAV, MP3, OGG, FLAC, MP4).
"""

import numpy as np
from algorythm.synth import Synth, Filter, ADSR
from algorythm.export import Exporter
from algorythm.visualization import CircularVisualizer


def create_demo_music():
    """Create a short musical piece for export."""
    synth = Synth(
        waveform='saw',
        filter=Filter.lowpass(cutoff=1500, resonance=0.6),
        envelope=ADSR(attack=0.1, decay=0.2, sustain=0.7, release=0.5)
    )
    
    # Simple melody
    melody = [261.63, 293.66, 329.63, 392.00, 329.63, 293.66, 261.63]
    signal_parts = []
    
    for freq in melody:
        note = synth.generate_note(frequency=freq, duration=0.5)
        signal_parts.append(note)
    
    return np.concatenate(signal_parts)


def example_1_wav_export():
    """Export as WAV (uncompressed)."""
    print("\n" + "=" * 60)
    print("Example 1: WAV Export (Uncompressed)")
    print("=" * 60)
    
    signal = create_demo_music()
    exporter = Exporter()
    
    # 16-bit WAV
    print("  Exporting 16-bit WAV...")
    exporter.export(signal, 'export_16bit.wav', sample_rate=44100, bit_depth=16)
    
    # 24-bit WAV
    print("  Exporting 24-bit WAV...")
    exporter.export(signal, 'export_24bit.wav', sample_rate=44100, bit_depth=24)
    
    print("✓ Created WAV files")


def example_2_mp3_export():
    """Export as MP3 (compressed)."""
    print("\n" + "=" * 60)
    print("Example 2: MP3 Export (Compressed)")
    print("=" * 60)
    
    signal = create_demo_music()
    exporter = Exporter()
    
    # Different quality settings
    qualities = ['low', 'medium', 'high']
    
    for quality in qualities:
        print(f"  Exporting MP3 ({quality} quality)...")
        exporter.export(
            signal,
            f'export_mp3_{quality}.mp3',
            sample_rate=44100,
            quality=quality
        )
    
    print("✓ Created MP3 files (or WAV fallbacks if pydub not available)")


def example_3_flac_export():
    """Export as FLAC (lossless compression)."""
    print("\n" + "=" * 60)
    print("Example 3: FLAC Export (Lossless)")
    print("=" * 60)
    
    signal = create_demo_music()
    exporter = Exporter()
    
    print("  Exporting FLAC...")
    exporter.export(
        signal,
        'export_lossless.flac',
        sample_rate=44100,
        quality='high'
    )
    
    print("✓ Created FLAC file (or WAV fallback)")


def example_4_ogg_export():
    """Export as OGG Vorbis."""
    print("\n" + "=" * 60)
    print("Example 4: OGG Export")
    print("=" * 60)
    
    signal = create_demo_music()
    exporter = Exporter()
    
    print("  Exporting OGG...")
    exporter.export(
        signal,
        'export_ogg.ogg',
        sample_rate=44100,
        quality='high'
    )
    
    print("✓ Created OGG file (or WAV fallback)")


def example_5_mp4_video():
    """Export as MP4 video with visualization."""
    print("\n" + "=" * 60)
    print("Example 5: MP4 Video Export")
    print("=" * 60)
    
    signal = create_demo_music()
    exporter = Exporter()
    
    # Create visualizer
    visualizer = CircularVisualizer(
        sample_rate=44100,
        num_bars=64,
        inner_radius=0.3
    )
    
    print("  Rendering MP4 video with circular visualization...")
    exporter.export(
        signal,
        'export_video.mp4',
        sample_rate=44100,
        visualizer=visualizer,
        video_width=1280,
        video_height=720,
        video_fps=30
    )
    
    print("✓ Created MP4 video")


def example_6_custom_location():
    """Export to custom location."""
    print("\n" + "=" * 60)
    print("Example 6: Custom Export Location")
    print("=" * 60)
    
    signal = create_demo_music()
    
    # Default exporter saves to ~/Music
    exporter_default = Exporter()
    print("  Exporting to default location (~/Music/)...")
    exporter_default.export(signal, 'in_music_folder.wav')
    
    # Custom directory
    exporter_custom = Exporter(default_directory='/tmp')
    print("  Exporting to /tmp/...")
    exporter_custom.export(signal, 'in_tmp_folder.wav')
    
    # Absolute path (ignores default directory)
    print("  Exporting with absolute path...")
    exporter_default.export(signal, '/tmp/absolute_path.wav')
    
    print("✓ Created files in different locations")


def main():
    print("\n" + "=" * 60)
    print("EXPORT FORMATS EXAMPLES")
    print("=" * 60)
    print("\nLearn how to export audio in different formats!")
    print("\nNote: Some formats require additional libraries:")
    print("  - MP3/OGG: pip install pydub")
    print("  - FLAC: pip install pydub")
    print("  - MP4: PIL/Pillow + ffmpeg")
    
    example_1_wav_export()
    example_2_mp3_export()
    example_3_flac_export()
    example_4_ogg_export()
    example_5_mp4_video()
    example_6_custom_location()
    
    print("\n" + "=" * 60)
    print("✓ All examples complete!")
    print("=" * 60)
    print("\nFiles created in:")
    print("  - ~/Music/ (most files)")
    print("  - /tmp/ (custom location examples)")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
