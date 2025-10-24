"""
Command-line interface for algorythm.
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Algorythm - A Python Library for Algorithmic Music',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  algorythm --version                      Show version information
  algorythm --example basic                Run basic example
  algorythm formats                        Show supported formats
  algorythm info song.mp3                  Show audio file info
  algorythm visualize song.mp3             Create visualization video
  algorythm visualize song.mp3 -v circular Create circular visualization
  algorythm visualize song.mp3 --bars 128  Use more bars (128)
  algorythm visualize song.mp3 --color purple --background dark
  algorythm visualize song.mp3 --offset 30 --duration 60

For more information, visit: https://github.com/KiyomiLovesOreos/synthesia
        """
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.4.0'
    )
    
    parser.add_argument(
        '--example',
        choices=['basic', 'composition', 'advanced'],
        help='Run an example script'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Visualize command
    visualize_parser = subparsers.add_parser(
        'visualize',
        help='Create visualization video from audio file'
    )
    visualize_parser.add_argument(
        'input_file',
        help='Input audio file (MP3, WAV, OGG, FLAC)'
    )
    visualize_parser.add_argument(
        '-o', '--output',
        help='Output video file (default: input_video.mp4)'
    )
    visualize_parser.add_argument(
        '-v', '--visualizer',
        choices=['waveform', 'circular', 'spectrum', 'spectrogram', 'oscilloscope'],
        default='circular',
        help='Visualizer type (default: circular)'
    )
    visualize_parser.add_argument(
        '-w', '--width',
        type=int,
        default=1920,
        help='Video width (default: 1920)'
    )
    visualize_parser.add_argument(
        '--height',
        type=int,
        default=1080,
        help='Video height (default: 1080)'
    )
    visualize_parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Video FPS (default: 30)'
    )
    visualize_parser.add_argument(
        '--offset',
        type=float,
        default=0.0,
        help='Start offset in seconds (default: 0)'
    )
    visualize_parser.add_argument(
        '--duration',
        type=float,
        help='Duration to process in seconds (default: entire file)'
    )
    visualize_parser.add_argument(
        '--bars',
        type=int,
        default=64,
        help='Number of bars for circular visualizer (default: 64)'
    )
    visualize_parser.add_argument(
        '--color',
        choices=['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta'],
        default='cyan',
        help='Color scheme (default: cyan)'
    )
    visualize_parser.add_argument(
        '--background',
        choices=['black', 'white', 'dark', 'light'],
        default='black',
        help='Background color (default: black)'
    )
    visualize_parser.add_argument(
        '--debug',
        action='store_true',
        help='Show debug information'
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        'info',
        help='Show audio file information'
    )
    info_parser.add_argument(
        'input_file',
        help='Input audio file'
    )
    
    # List formats command
    formats_parser = subparsers.add_parser(
        'formats',
        help='List supported audio formats'
    )
    
    args = parser.parse_args()
    
    if args.command == 'visualize':
        visualize_audio(args)
    elif args.command == 'info':
        show_audio_info(args)
    elif args.command == 'formats':
        show_supported_formats()
    elif args.example:
        run_example(args.example)
    else:
        parser.print_help()


def run_example(example_name):
    """Run an example script."""
    print(f"Running {example_name} example...")
    
    if example_name == 'basic':
        from algorythm.synth import Synth, Filter, ADSR
        from algorythm.export import Exporter
        
        warm_pad = Synth(
            waveform='saw',
            filter=Filter.lowpass(cutoff=2000, resonance=0.6),
            envelope=ADSR(attack=1.5, decay=0.5, sustain=0.8, release=2.0)
        )
        
        note_signal = warm_pad.generate_note(frequency=440.0, duration=3.0)
        print(f"Generated warm pad: {len(note_signal)} samples")
        
        exporter = Exporter()
        exporter.export(note_signal, 'warm_pad.wav', sample_rate=44100)
        print("Exported to 'warm_pad.wav'")
        
    elif example_name == 'composition':
        from algorythm.synth import Synth, Filter, ADSR
        from algorythm.sequence import Motif, Scale
        from algorythm.structure import Composition, Reverb
        
        warm_pad = Synth(
            waveform='saw',
            filter=Filter.lowpass(cutoff=2000, resonance=0.6),
            envelope=ADSR(attack=1.5, decay=0.5, sustain=0.8, release=2.0)
        )
        
        melody = Motif.from_intervals([0, 2, 4, 7], scale=Scale.major('C'))
        
        final_track = Composition(tempo=120) \
            .add_track('Bassline', warm_pad) \
            .repeat_motif(melody, bars=8) \
            .transpose(semitones=5) \
            .add_fx(Reverb(mix=0.4))
        
        audio = final_track.render(file_path='epic_track.wav', quality='high')
        print(f"Rendered composition: {len(audio)} samples")
        print("Exported to 'epic_track.wav'")
        
    elif example_name == 'advanced':
        from algorythm.synth import SynthPresets
        from algorythm.sequence import Motif, Scale, Arpeggiator
        from algorythm.structure import Composition, Reverb, Delay
        
        composition = Composition(tempo=128)
        
        bass_synth = SynthPresets.bass()
        bass_motif = Motif.from_intervals([0, 0, 0, 0], scale=Scale.major('A', octave=2))
        composition.add_track('Bass', bass_synth).repeat_motif(bass_motif, bars=4)
        
        lead_synth = SynthPresets.pluck()
        lead_motif = Motif.from_intervals([0, 2, 4, 7, 9, 7, 4, 2], scale=Scale.major('A', octave=4))
        arpeggiator = Arpeggiator(pattern='up-down', octaves=2)
        arpeggiated = arpeggiator.arpeggiate(lead_motif)
        
        composition.add_track('Lead', lead_synth) \
            .repeat_motif(arpeggiated, bars=2) \
            .add_fx(Delay(delay_time=0.25, feedback=0.3, mix=0.3)) \
            .add_fx(Reverb(mix=0.2))
        
        pad_synth = SynthPresets.warm_pad()
        pad_motif = Motif.from_intervals([0, 4, 7], scale=Scale.major('A', octave=3))
        composition.add_track('Pad', pad_synth) \
            .repeat_motif(pad_motif, bars=4) \
            .add_fx(Reverb(mix=0.5))
        
        audio = composition.render(file_path='advanced_track.wav', quality='high')
        print(f"Rendered advanced composition: {len(audio)} samples")
        print("Exported to 'advanced_track.wav'")


def visualize_audio(args):
    """Create visualization video from audio file."""
    from algorythm.audio_loader import visualize_audio_file, AudioFile
    from algorythm.visualization import (
        WaveformVisualizer,
        CircularVisualizer,
        FrequencyScopeVisualizer,
        SpectrogramVisualizer,
        OscilloscopeVisualizer
    )
    
    input_path = Path(args.input_file)
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    # Determine output filename
    if args.output:
        output_path = args.output
    else:
        output_path = input_path.stem + '_video.mp4'
    
    print(f"\n{'=' * 60}")
    print(f"Algorythm Audio Visualizer")
    print(f"{'=' * 60}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Visualizer: {args.visualizer}")
    print(f"Resolution: {args.width}x{args.height} @ {args.fps}fps")
    
    if args.offset > 0 or args.duration:
        offset_str = f"{args.offset:.1f}s" if args.offset > 0 else "start"
        duration_str = f"{args.duration:.1f}s" if args.duration else "end"
        print(f"Time range: {offset_str} to {duration_str}")
    
    print(f"{'=' * 60}\n")
    
    # Load audio to get info
    try:
        print("Loading audio file...")
        audio = AudioFile(
            str(input_path),
            offset=args.offset,
            duration=args.duration
        )
        print(f"✓ Loaded {audio.duration:.2f}s at {audio.sample_rate}Hz")
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\nFor MP3/OGG/FLAC support:")
        print("  1. Install pydub: pip install pydub")
        print("  2. Install ffmpeg:")
        print("     Ubuntu/Debian: sudo apt install ffmpeg")
        print("     macOS: brew install ffmpeg")
        print("     Windows: Download from https://ffmpeg.org/")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error loading audio: {e}")
        import traceback
        if '--debug' in sys.argv:
            traceback.print_exc()
        sys.exit(1)
    
    # Create visualizer
    print(f"\nCreating {args.visualizer} visualizer...")
    
    # Color mapping
    color_map = {
        'blue': (100, 150, 255),
        'red': (255, 100, 100),
        'green': (100, 255, 150),
        'purple': (200, 100, 255),
        'orange': (255, 180, 100),
        'cyan': (100, 255, 255),
        'magenta': (255, 100, 255),
    }
    
    bg_color_map = {
        'black': (0, 0, 0),
        'white': (255, 255, 255),
        'dark': (20, 20, 30),
        'light': (240, 240, 245),
    }
    
    fg_color = color_map.get(args.color, (100, 255, 255))
    bg_color = bg_color_map.get(args.background, (0, 0, 0))
    
    if args.visualizer == 'waveform':
        viz = WaveformVisualizer(sample_rate=audio.sample_rate, debug=args.debug)
    elif args.visualizer == 'circular':
        viz = CircularVisualizer(
            sample_rate=audio.sample_rate,
            num_bars=args.bars,
            debug=args.debug
        )
    elif args.visualizer == 'spectrum':
        viz = FrequencyScopeVisualizer(sample_rate=audio.sample_rate, debug=args.debug)
    elif args.visualizer == 'spectrogram':
        viz = SpectrogramVisualizer(sample_rate=audio.sample_rate, debug=args.debug)
    elif args.visualizer == 'oscilloscope':
        viz = OscilloscopeVisualizer(sample_rate=audio.sample_rate, debug=args.debug)
    else:
        viz = CircularVisualizer(
            sample_rate=audio.sample_rate,
            num_bars=args.bars,
            debug=args.debug
        )
    
    # Create visualization
    print(f"Rendering video (this may take a few minutes)...")
    print(f"Color scheme: {args.color} on {args.background}")
    
    try:
        # Use VideoRenderer directly for better control over colors
        from algorythm.visualization import VideoRenderer
        from algorythm.export import Exporter
        
        # Create renderer with custom colors
        renderer = VideoRenderer(
            width=args.width,
            height=args.height,
            fps=args.fps,
            sample_rate=audio.sample_rate,
            background_color=bg_color,
            foreground_color=fg_color,
            debug=args.debug
        )
        
        # Render video with audio
        renderer.render_frames(
            audio.signal,
            viz,
            output_path=output_path
        )
    except Exception as e:
        print(f"\n❌ Error creating video: {e}")
        import traceback
        if args.debug:
            traceback.print_exc()
        print("\nTroubleshooting:")
        print("  - Make sure ffmpeg is installed")
        print("  - Check if opencv-python is installed: pip install opencv-python")
        print("  - Try with --debug flag for more details")
        sys.exit(1)
    
    # Only print success if we got here without exceptions
    print(f"\n{'=' * 60}")
    print(f"✓ Success! Video saved to: {output_path}")
    print(f"{'=' * 60}\n")


def show_audio_info(args):
    """Show information about audio file."""
    from algorythm.audio_loader import AudioFile
    
    input_path = Path(args.input_file)
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    try:
        print(f"\n{'=' * 60}")
        print(f"Audio File Information")
        print(f"{'=' * 60}")
        
        audio = AudioFile(str(input_path))
        
        print(f"File:        {input_path.name}")
        print(f"Path:        {input_path.absolute()}")
        print(f"Format:      {input_path.suffix.upper()[1:]}")
        print(f"Duration:    {audio.duration:.2f} seconds")
        print(f"Sample rate: {audio.sample_rate:,} Hz")
        print(f"Samples:     {audio.num_samples:,}")
        print(f"Size:        {input_path.stat().st_size / 1024:.1f} KB")
        
        # Calculate time representation
        minutes = int(audio.duration // 60)
        seconds = audio.duration % 60
        print(f"Time:        {minutes}:{seconds:05.2f}")
        
        print(f"{'=' * 60}\n")
        
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\nFor MP3/OGG/FLAC support:")
        print("  1. Install pydub: pip install pydub")
        print("  2. Install ffmpeg:")
        print("     Ubuntu/Debian: sudo apt install ffmpeg")
        print("     macOS: brew install ffmpeg")
        print("     Windows: Download from https://ffmpeg.org/")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error reading audio file: {e}")
        import traceback
        if '--debug' in sys.argv:
            traceback.print_exc()
        sys.exit(1)


def show_supported_formats():
    """Show supported audio and video formats with dependency status."""
    print(f"\n{'=' * 60}")
    print("Supported Formats")
    print(f"{'=' * 60}")
    
    # Check dependencies
    from algorythm.audio_loader import HAS_PYDUB
    
    try:
        import cv2
        has_opencv = True
    except ImportError:
        has_opencv = False
    
    try:
        import matplotlib
        has_matplotlib = True
    except ImportError:
        has_matplotlib = False
    
    # Check ffmpeg
    has_ffmpeg = False
    if HAS_PYDUB:
        try:
            from pydub.utils import which
            has_ffmpeg = bool(which("ffmpeg") or which("avconv"))
        except:
            pass
    
    print("\nAudio Input Formats:")
    print("  ✓ WAV  - Uncompressed audio (always supported)")
    
    if HAS_PYDUB and has_ffmpeg:
        print("  ✓ MP3  - MPEG Audio Layer 3")
        print("  ✓ OGG  - Ogg Vorbis")
        print("  ✓ FLAC - Free Lossless Audio Codec")
        print("  ✓ M4A  - MPEG-4 Audio")
        print("  ✓ AAC  - Advanced Audio Coding")
    elif HAS_PYDUB and not has_ffmpeg:
        print("  ❌ MP3  - Requires ffmpeg")
        print("  ❌ OGG  - Requires ffmpeg")
        print("  ❌ FLAC - Requires ffmpeg")
        print("  ❌ M4A  - Requires ffmpeg")
        print("  ❌ AAC  - Requires ffmpeg")
    else:
        print("  ❌ MP3  - Requires pydub and ffmpeg")
        print("  ❌ OGG  - Requires pydub and ffmpeg")
        print("  ❌ FLAC - Requires pydub and ffmpeg")
        print("  ❌ M4A  - Requires pydub and ffmpeg")
        print("  ❌ AAC  - Requires pydub and ffmpeg")
    
    print("\nVideo Output Formats:")
    if has_opencv and has_ffmpeg:
        print("  ✓ MP4  - MPEG-4 Video (OpenCV backend)")
    elif has_matplotlib and has_ffmpeg:
        print("  ✓ MP4  - MPEG-4 Video (Matplotlib backend)")
    elif has_opencv or has_matplotlib:
        print("  ❌ MP4  - Requires ffmpeg")
    else:
        print("  ❌ MP4  - Requires opencv-python/matplotlib and ffmpeg")
    
    print("\nVisualizer Types:")
    print("  • waveform     - Audio waveform over time")
    print("  • circular     - Circular frequency bars (default)")
    print("  • spectrum     - Frequency spectrum scope")
    print("  • spectrogram  - Time-frequency heatmap")
    print("  • oscilloscope - Real-time oscilloscope display")
    
    print("\nDependency Status:")
    print(f"  numpy:      ✓ (required)")
    print(f"  pydub:      {'✓ Installed' if HAS_PYDUB else '❌ Not installed'}")
    print(f"  ffmpeg:     {'✓ Available' if has_ffmpeg else '❌ Not found'}")
    print(f"  opencv:     {'✓ Installed' if has_opencv else '❌ Not installed'}")
    print(f"  matplotlib: {'✓ Installed' if has_matplotlib else '❌ Not installed'}")
    
    if not HAS_PYDUB or not has_ffmpeg:
        print("\n" + "=" * 60)
        print("Installation Instructions:")
        if not HAS_PYDUB:
            print("\n1. Install pydub:")
            print("   pip install pydub")
        if not has_ffmpeg:
            print("\n2. Install ffmpeg:")
            print("   Ubuntu/Debian: sudo apt install ffmpeg")
            print("   macOS: brew install ffmpeg")
            print("   Windows: Download from https://ffmpeg.org/")
        if not has_opencv and not has_matplotlib:
            print("\n3. Install video backend (optional):")
            print("   pip install opencv-python  # Fast")
            print("   # OR")
            print("   pip install matplotlib     # Slower but no C deps")
    
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
