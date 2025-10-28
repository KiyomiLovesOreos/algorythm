"""
Command-line interface for algorythm.
"""

import argparse
import sys
import numpy as np
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
    
    # Generate music command
    generate_parser = subparsers.add_parser(
        'generate',
        help='Generate music from command line'
    )
    generate_parser.add_argument(
        '-i', '--instrument',
        default='pluck',
        help='Instrument preset (default: pluck)'
    )
    generate_parser.add_argument(
        '-s', '--scale',
        default='C:major',
        help='Scale (format: NOTE:TYPE, e.g., C:major, A:minor)'
    )
    generate_parser.add_argument(
        '-n', '--notes',
        default='0,2,4,5,7',
        help='Note intervals (comma-separated, e.g., 0,2,4,5,7)'
    )
    generate_parser.add_argument(
        '-d', '--durations',
        default='0.5',
        help='Note durations (single value or comma-separated)'
    )
    generate_parser.add_argument(
        '-t', '--tempo',
        type=int,
        default=120,
        help='Tempo in BPM (default: 120)'
    )
    generate_parser.add_argument(
        '-r', '--repeat',
        type=int,
        default=1,
        help='Number of times to repeat (default: 1)'
    )
    generate_parser.add_argument(
        '-o', '--output',
        default='output.wav',
        help='Output file (default: output.wav)'
    )
    generate_parser.add_argument(
        '--effects',
        help='Effects chain (comma-separated, e.g., reverb,delay,chorus)'
    )
    
    # List presets command
    presets_parser = subparsers.add_parser(
        'presets',
        help='List available instrument presets'
    )
    
    # List effects command
    effects_parser = subparsers.add_parser(
        'effects',
        help='List available audio effects'
    )
    
    # Apply effects command
    apply_fx_parser = subparsers.add_parser(
        'apply-fx',
        help='Apply effects to audio file'
    )
    apply_fx_parser.add_argument(
        'input_file',
        help='Input audio file'
    )
    apply_fx_parser.add_argument(
        '-e', '--effects',
        required=True,
        help='Effects chain (comma-separated, e.g., reverb,delay,distortion)'
    )
    apply_fx_parser.add_argument(
        '-o', '--output',
        help='Output file (default: input_with_fx.wav)'
    )
    apply_fx_parser.add_argument(
        '--params',
        help='Effect parameters in JSON format'
    )
    
    # Quick generate command
    quick_parser = subparsers.add_parser(
        'quick',
        help='Quick music generation with presets'
    )
    quick_parser.add_argument(
        'style',
        choices=['ambient', 'techno', 'chill', 'upbeat', 'experimental', 'minimal'],
        help='Music style'
    )
    quick_parser.add_argument(
        '-o', '--output',
        default='quick_track.wav',
        help='Output file (default: quick_track.wav)'
    )
    quick_parser.add_argument(
        '-l', '--length',
        type=int,
        default=30,
        help='Track length in seconds (default: 30)'
    )
    
    # Interactive mode
    interactive_parser = subparsers.add_parser(
        'interactive',
        help='Start interactive mode'
    )
    
    args = parser.parse_args()
    
    if args.command == 'visualize':
        visualize_audio(args)
    elif args.command == 'info':
        show_audio_info(args)
    elif args.command == 'formats':
        show_supported_formats()
    elif args.command == 'generate':
        generate_music(args)
    elif args.command == 'presets':
        list_presets()
    elif args.command == 'effects':
        list_effects()
    elif args.command == 'apply-fx':
        apply_effects(args)
    elif args.command == 'quick':
        quick_generate(args)
    elif args.command == 'interactive':
        interactive_mode()
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


def generate_music(args):
    """Generate music from command line arguments."""
    from algorythm import SynthPresets, Motif, Scale, Track, Composition, RenderEngine, Exporter
    from algorythm.effects import (
        Reverb, Delay, Chorus, Flanger, Phaser, 
        Distortion, Overdrive, Fuzz, Compressor, Tremolo, Vibrato,
        Limiter, Gate, RingModulator, BitCrusher, EffectChain
    )
    
    print(f"\n{'=' * 60}")
    print("Algorythm Music Generator")
    print(f"{'=' * 60}\n")
    
    # Parse scale
    try:
        note, scale_type = args.scale.split(':')
        if scale_type == 'major':
            scale = Scale.major(note, 4)
        elif scale_type == 'minor':
            scale = Scale.minor(note, 4)
        elif scale_type == 'pentatonic':
            scale = Scale.pentatonic(note, 4)
        else:
            scale = Scale.major(note, 4)
    except:
        print(f"Invalid scale format. Using C major.")
        scale = Scale.major('C', 4)
    
    # Parse notes
    intervals = [int(x.strip()) for x in args.notes.split(',')]
    
    # Parse durations
    if ',' in args.durations:
        durations = [float(x.strip()) for x in args.durations.split(',')]
    else:
        durations = [float(args.durations)] * len(intervals)
    
    # Get instrument
    instrument = get_preset_by_name(args.instrument)
    
    # Create motif using intervals and scale
    motif = Motif(intervals=intervals, scale=scale, durations=durations)
    
    # Create composition
    comp = Composition(tempo=args.tempo)
    track = Track("Main", instrument)
    
    # Repeat motif
    for _ in range(args.repeat):
        track.add_motif(motif)
    
    # Add effects if specified
    if args.effects:
        effects = args.effects.split(',')
        for effect_name in effects:
            effect = create_effect_by_name(effect_name.strip())
            if effect:
                track.add_effect(effect)
                print(f"  ✓ Added effect: {effect_name}")
    
    comp.add_track(track)
    
    # Render
    print(f"\nGenerating music...")
    print(f"  Instrument: {args.instrument}")
    print(f"  Scale: {args.scale}")
    print(f"  Notes: {intervals}")
    print(f"  Tempo: {args.tempo} BPM")
    print(f"  Repeats: {args.repeat}")
    
    engine = RenderEngine(sample_rate=44100)
    audio = engine.render(comp)
    
    # Export
    exporter = Exporter()
    exporter.export_wav(audio, args.output, sample_rate=44100)
    
    duration = len(audio) / 44100
    print(f"\n✓ Generated: {args.output}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Samples: {len(audio)}")
    print(f"{'=' * 60}\n")


def list_presets():
    """List all available instrument presets."""
    print(f"\n{'=' * 60}")
    print("Available Instrument Presets")
    print(f"{'=' * 60}\n")
    
    presets = {
        'Basic Sounds': ['warm_pad', 'pluck', 'bass', 'lead'],
        'Advanced Sounds': ['organ', 'bell', 'strings', 'guitar', 'drum', 'brass']
    }
    
    for category, preset_list in presets.items():
        print(f"{category}:")
        for preset in preset_list:
            print(f"  • {preset}")
        print()
    
    print("Usage:")
    print("  algorythm generate -i <preset_name> -o output.wav")
    print(f"{'=' * 60}\n")


def list_effects():
    """List all available audio effects."""
    print(f"\n{'=' * 60}")
    print("Available Audio Effects")
    print(f"{'=' * 60}\n")
    
    effects = {
        'Time-Based': [
            ('reverb', 'Adds space and depth'),
            ('delay', 'Echo effect with feedback'),
        ],
        'Modulation': [
            ('chorus', 'Thickening effect'),
            ('flanger', 'Sweeping comb filter'),
            ('phaser', 'Notch filter sweeps'),
            ('tremolo', 'Amplitude modulation'),
            ('vibrato', 'Pitch modulation'),
        ],
        'Distortion': [
            ('distortion', 'Waveshaping distortion'),
            ('overdrive', 'Smooth tube saturation'),
            ('fuzz', 'Extreme clipping'),
        ],
        'Dynamics': [
            ('compressor', 'Dynamic range control'),
            ('limiter', 'Peak limiting'),
            ('gate', 'Noise gate'),
        ],
        'Special': [
            ('ringmod', 'Ring modulator'),
            ('bitcrush', 'Lo-fi digital distortion'),
        ]
    }
    
    for category, effect_list in effects.items():
        print(f"{category}:")
        for name, desc in effect_list:
            print(f"  • {name:15s} - {desc}")
        print()
    
    print("Usage:")
    print("  algorythm generate --effects reverb,delay,chorus")
    print("  algorythm apply-fx input.wav -e distortion,reverb")
    print(f"{'=' * 60}\n")


def apply_effects(args):
    """Apply effects to an existing audio file."""
    from algorythm.audio_loader import load_audio
    from algorythm.effects import EffectChain
    from algorythm.export import Exporter
    import json
    
    print(f"\n{'=' * 60}")
    print("Algorythm Effect Processor")
    print(f"{'=' * 60}\n")
    
    # Load audio
    print(f"Loading: {args.input_file}")
    try:
        audio, sample_rate = load_audio(args.input_file)
        print(f"  ✓ Loaded: {len(audio)} samples at {sample_rate} Hz")
    except Exception as e:
        print(f"  ✗ Error loading file: {e}")
        return
    
    # Parse effect parameters if provided
    params = {}
    if args.params:
        try:
            params = json.loads(args.params)
        except:
            print("Warning: Could not parse parameters JSON")
    
    # Create effect chain
    effects_list = [e.strip() for e in args.effects.split(',')]
    chain = EffectChain()
    
    for effect_name in effects_list:
        effect_params = params.get(effect_name, {})
        effect = create_effect_by_name(effect_name, **effect_params)
        if effect:
            chain.add_effect(effect)
            print(f"  ✓ Added: {effect_name}")
        else:
            print(f"  ✗ Unknown effect: {effect_name}")
    
    # Apply effects
    print(f"\nProcessing...")
    processed = chain.apply(audio, sample_rate)
    
    # Export
    output_file = args.output or args.input_file.replace('.wav', '_with_fx.wav')
    exporter = Exporter()
    exporter.export_wav(processed, output_file, sample_rate)
    
    print(f"\n✓ Saved: {output_file}")
    print(f"{'=' * 60}\n")


def quick_generate(args):
    """Quick music generation with style presets."""
    from algorythm import (
        Composition, Track, Motif, Scale, SynthPresets,
        PhysicalModelSynth, RenderEngine, Exporter
    )
    from algorythm.effects import (
        Reverb, Delay, Chorus, Distortion, Compressor,
        RingModulator, Phaser, Tremolo
    )
    
    print(f"\n{'=' * 60}")
    print(f"Quick Generate: {args.style.upper()}")
    print(f"{'=' * 60}\n")
    
    comp = Composition(tempo=120)
    scale = Scale.major('C', 4)
    
    if args.style == 'ambient':
        # Ambient: slow pads with reverb
        pad = SynthPresets.strings()
        motif = Motif(intervals=[0, 2, 4], 
                     durations=[8.0, 8.0, 8.0])
        track = Track("Pad")
        track.add_motif(motif, instrument=pad)
        track.add_effect(Reverb(room_size=0.9, wet_level=0.6))
        track.add_effect(Chorus(mix=0.4))
        comp.add_track(track)
        comp.tempo = 60
        
    elif args.style == 'techno':
        # Techno: bass + drums
        bass = SynthPresets.bass()
        drum = SynthPresets.drum()
        
        bass_motif = Motif(intervals=[scale.get_frequency(-12)] * 8, durations=[0.5] * 8)
        bass_track = Track("Bass")
        bass_track.add_motif(bass_motif, instrument=bass)
        bass_track.add_effect(Distortion(drive=2.0, mix=0.3))
        
        drum_motif = Motif(intervals=[scale.get_frequency(-12)] * 16, durations=[0.25] * 16)
        drum_track = Track("Drums")
        drum_track.add_motif(drum_motif, instrument=drum)
        drum_track.add_effect(Compressor(threshold=-15.0, ratio=4.0))
        
        comp.add_track(bass_track)
        comp.add_track(drum_track)
        comp.tempo = 130
        
    elif args.style == 'chill':
        # Chill: guitar + soft pads
        guitar = SynthPresets.guitar()
        pad = SynthPresets.warm_pad()
        
        guitar_motif = Motif(intervals=scale.ascending(8), durations=[1.0] * 8)
        guitar_track = Track("Guitar")
        guitar_track.add_motif(guitar_motif, instrument=guitar)
        guitar_track.add_effect(Reverb(room_size=0.6, wet_level=0.3))
        guitar_track.add_effect(Delay(delay_time=0.375, feedback=0.4, wet_level=0.3))
        
        pad_motif = Motif(intervals=[0, 4, 7], 
                         durations=[8.0, 8.0, 8.0])
        pad_track = Track("Pad")
        pad_track.add_motif(pad_motif, instrument=pad)
        pad_track.add_effect(Chorus(mix=0.3))
        
        comp.add_track(guitar_track)
        comp.add_track(pad_track)
        comp.tempo = 90
        
    elif args.style == 'upbeat':
        # Upbeat: lead + bass + pluck
        lead = SynthPresets.lead()
        bass = SynthPresets.bass()
        pluck = SynthPresets.pluck()
        
        lead_motif = Motif(intervals=scale.ascending(8) + scale.descending(8), 
                          durations=[0.25] * 16)
        lead_track = Track("Lead")
        lead_track.add_motif(lead_motif, instrument=lead)
        lead_track.add_effect(Delay(delay_time=0.25, feedback=0.3, wet_level=0.4))
        
        bass_motif = Motif(intervals=[scale.get_frequency(-12)] * 4, durations=[1.0] * 4)
        bass_track = Track("Bass")
        bass_track.add_motif(bass_motif, instrument=bass)
        
        pluck_motif = Motif(intervals=[0, 2, 4, 5],
                           durations=[0.5] * 4)
        pluck_track = Track("Pluck")
        pluck_track.add_motif(pluck_motif, instrument=pluck)
        pluck_track.add_effect(Reverb(room_size=0.5, wet_level=0.2))
        
        comp.add_track(lead_track)
        comp.add_track(bass_track)
        comp.add_track(pluck_track)
        comp.tempo = 140
        
    elif args.style == 'experimental':
        # Experimental: physical models + ring mod
        string = PhysicalModelSynth(model_type='string')
        wind = PhysicalModelSynth(model_type='wind')
        
        string_motif = Motif(intervals=[scale.get_frequency(i) for i in [0, 3, 7, 10, 7, 3]],
                            durations=[1.5] * 6)
        string_track = Track("String")
        string_track.add_motif(string_motif, instrument=string)
        string_track.add_effect(RingModulator(carrier_freq=200.0, mix=0.3))
        string_track.add_effect(Reverb(room_size=0.8, wet_level=0.5))
        
        wind_motif = Motif(intervals=[scale.get_frequency(i) for i in [5, 7, 9, 12]],
                          durations=[2.0] * 4)
        wind_track = Track("Wind")
        wind_track.add_motif(wind_motif, instrument=wind)
        wind_track.add_effect(Phaser(rate=0.3, mix=0.5))
        
        comp.add_track(string_track)
        comp.add_track(wind_track)
        comp.tempo = 80
        
    elif args.style == 'minimal':
        # Minimal: single instrument with effects
        organ = SynthPresets.organ()
        motif = Motif(intervals=[0, 4], durations=[4.0, 4.0])
        track = Track("Organ")
        track.add_motif(motif, instrument=organ)
        track.add_effect(Tremolo(rate=6.0, depth=0.3))
        track.add_effect(Reverb(room_size=0.7, wet_level=0.4))
        comp.add_track(track)
        comp.tempo = 100
    
    # Render
    print(f"Generating {args.style} track...")
    print(f"  Tempo: {comp.tempo} BPM")
    print(f"  Target length: {args.length} seconds")
    
    engine = RenderEngine(sample_rate=44100)
    audio = engine.render(comp)
    
    # Trim or loop to desired length
    target_samples = args.length * 44100
    if len(audio) < target_samples:
        # Loop to fill
        repeats = (target_samples // len(audio)) + 1
        audio = np.tile(audio, repeats)[:target_samples]
    else:
        audio = audio[:target_samples]
    
    # Export
    exporter = Exporter()
    exporter.export_wav(audio, args.output, sample_rate=44100)
    
    print(f"\n✓ Generated: {args.output}")
    print(f"  Duration: {len(audio) / 44100:.2f} seconds")
    print(f"{'=' * 60}\n")


def interactive_mode():
    """Start interactive mode for music creation."""
    print(f"\n{'=' * 70}")
    print("Algorythm Interactive Mode")
    print(f"{'=' * 70}\n")
    print("Commands:")
    print("  play <preset> <note>     - Play a note")
    print("  generate <style>         - Generate quick track")
    print("  list presets            - List instruments")
    print("  list effects            - List effects")
    print("  help                    - Show this help")
    print("  quit                    - Exit")
    print(f"{'=' * 70}\n")
    
    while True:
        try:
            cmd = input("algorythm> ").strip()
            
            if not cmd:
                continue
            elif cmd == 'quit' or cmd == 'exit':
                print("Goodbye!")
                break
            elif cmd == 'help':
                print("\nCommands:")
                print("  play <preset> <note>     - Play a note")
                print("  generate <style>         - Generate quick track")
                print("  list presets            - List instruments")
                print("  list effects            - List effects")
                print("  quit                    - Exit\n")
            elif cmd == 'list presets':
                list_presets()
            elif cmd == 'list effects':
                list_effects()
            elif cmd.startswith('play '):
                parts = cmd.split()
                if len(parts) >= 3:
                    preset_name = parts[1]
                    note = parts[2]
                    print(f"Playing {preset_name} at {note}...")
                    # TODO: Implement playback
                else:
                    print("Usage: play <preset> <note>")
            elif cmd.startswith('generate '):
                style = cmd.split()[1] if len(cmd.split()) > 1 else 'chill'
                print(f"Generating {style} track...")
                # TODO: Call quick_generate
            else:
                print(f"Unknown command: {cmd}. Type 'help' for commands.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break


def get_preset_by_name(name):
    """Get instrument preset by name."""
    from algorythm import SynthPresets
    
    presets = {
        'warm_pad': SynthPresets.warm_pad,
        'pluck': SynthPresets.pluck,
        'bass': SynthPresets.bass,
        'lead': SynthPresets.lead,
        'organ': SynthPresets.organ,
        'bell': SynthPresets.bell,
        'strings': SynthPresets.strings,
        'guitar': SynthPresets.guitar,
        'drum': SynthPresets.drum,
        'brass': SynthPresets.brass,
    }
    
    preset_func = presets.get(name.lower(), presets['pluck'])
    return preset_func()


def create_effect_by_name(name, **kwargs):
    """Create effect instance by name."""
    from algorythm.effects import (
        Reverb, Delay, Chorus, Flanger, Phaser,
        Distortion, Overdrive, Fuzz, Compressor, Limiter, Gate,
        Tremolo, Vibrato, RingModulator, BitCrusher
    )
    
    effects = {
        'reverb': lambda: Reverb(room_size=kwargs.get('room_size', 0.7), 
                                    wet_level=kwargs.get('wet_level', 0.3)),
        'delay': lambda: Delay(delay_time=kwargs.get('delay_time', 0.5),
                                  feedback=kwargs.get('feedback', 0.5),
                                  wet_level=kwargs.get('wet_level', 0.5)),
        'chorus': lambda: Chorus(mix=kwargs.get('mix', 0.5)),
        'flanger': lambda: Flanger(mix=kwargs.get('mix', 0.5)),
        'phaser': lambda: Phaser(mix=kwargs.get('mix', 0.5)),
        'tremolo': lambda: Tremolo(rate=kwargs.get('rate', 5.0), 
                                      depth=kwargs.get('depth', 0.5)),
        'vibrato': lambda: Vibrato(rate=kwargs.get('rate', 5.0)),
        'distortion': lambda: Distortion(drive=kwargs.get('drive', 5.0)),
        'overdrive': lambda: Overdrive(drive=kwargs.get('drive', 2.0)),
        'fuzz': lambda: Fuzz(gain=kwargs.get('gain', 10.0)),
        'compressor': lambda: Compressor(threshold=kwargs.get('threshold', -20.0),
                                          ratio=kwargs.get('ratio', 4.0)),
        'limiter': lambda: Limiter(threshold=kwargs.get('threshold', -1.0)),
        'gate': lambda: Gate(threshold=kwargs.get('threshold', -40.0)),
        'ringmod': lambda: RingModulator(carrier_freq=kwargs.get('carrier_freq', 440.0)),
        'bitcrush': lambda: BitCrusher(bit_depth=kwargs.get('bit_depth', 8)),
    }
    
    effect_func = effects.get(name.lower())
    return effect_func() if effect_func else None


if __name__ == '__main__':
    main()
