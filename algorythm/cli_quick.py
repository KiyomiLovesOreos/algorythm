#!/usr/bin/env python3
"""
Simple CLI for Algorythm - Quick music generation commands.
Usage: algorythm-quick [command] [options]
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog='algorythm-quick',
        description='Quick music generation with Algorythm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  algorythm-quick melody                           # Generate melody with defaults
  algorythm-quick melody -i bell -t 140            # Bell at 140 BPM
  algorythm-quick beat -t 128 -b 8                 # 8-bar beat at 128 BPM
  algorythm-quick style ambient -l 60              # 60s ambient track
  algorythm-quick fx reverb input.wav              # Add reverb
  algorythm-quick list                             # List presets and effects
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Melody command
    melody_parser = subparsers.add_parser('melody', help='Generate a melody')
    melody_parser.add_argument('-i', '--instrument', default='pluck',
                              help='Instrument (pluck, bass, lead, bell, etc.)')
    melody_parser.add_argument('-s', '--scale', default='C:major',
                              help='Scale (e.g., C:major, A:minor)')
    melody_parser.add_argument('-t', '--tempo', type=int, default=120,
                              help='Tempo in BPM')
    melody_parser.add_argument('-b', '--bars', type=int, default=4,
                              help='Number of bars')
    melody_parser.add_argument('-o', '--output', default='melody.wav',
                              help='Output file')
    
    # Beat command
    beat_parser = subparsers.add_parser('beat', help='Generate a drum beat')
    beat_parser.add_argument('-t', '--tempo', type=int, default=120,
                            help='Tempo in BPM')
    beat_parser.add_argument('-b', '--bars', type=int, default=4,
                            help='Number of bars')
    beat_parser.add_argument('-o', '--output', default='beat.wav',
                            help='Output file')
    
    # Style command
    style_parser = subparsers.add_parser('style', help='Generate by style')
    style_parser.add_argument('style', choices=['ambient', 'chill', 'upbeat', 'minimal'],
                             help='Music style')
    style_parser.add_argument('-l', '--length', type=int, default=30,
                             help='Length in seconds')
    style_parser.add_argument('-o', '--output', default='track.wav',
                             help='Output file')
    
    # FX command
    fx_parser = subparsers.add_parser('fx', help='Apply effect to file')
    fx_parser.add_argument('effect', 
                          choices=['reverb', 'delay', 'chorus', 'distortion', 'compressor'],
                          help='Effect to apply')
    fx_parser.add_argument('input', help='Input audio file')
    fx_parser.add_argument('-o', '--output', help='Output file (auto if not specified)')
    fx_parser.add_argument('--param', action='append', 
                          help='Effect parameter (key=value)')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List presets and effects')
    list_parser.add_argument('what', nargs='?', choices=['presets', 'effects', 'all'],
                            default='all', help='What to list')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    from algorythm.cli_tools import (
        quick_melody, quick_beat, apply_effect_to_file,
        list_all_presets, list_all_effects, generate_style
    )
    
    try:
        if args.command == 'melody':
            print(f"🎵 Generating melody...")
            duration = quick_melody(
                instrument=args.instrument,
                scale=args.scale,
                tempo=args.tempo,
                bars=args.bars,
                output=args.output
            )
            print(f"✓ Created: {args.output} ({duration:.1f}s)")
            
        elif args.command == 'beat':
            print(f"🥁 Generating beat...")
            duration = quick_beat(
                tempo=args.tempo,
                bars=args.bars,
                output=args.output
            )
            print(f"✓ Created: {args.output} ({duration:.1f}s)")
            
        elif args.command == 'style':
            print(f"🎨 Generating {args.style} track...")
            duration = generate_style(
                style=args.style,
                length=args.length,
                output=args.output
            )
            print(f"✓ Created: {args.output} ({duration:.1f}s)")
            
        elif args.command == 'fx':
            print(f"🎚️  Applying {args.effect}...")
            params = {}
            if args.param:
                for p in args.param:
                    k, v = p.split('=')
                    params[k] = float(v)
            
            output = apply_effect_to_file(
                args.input,
                args.effect,
                args.output,
                **params
            )
            print(f"✓ Created: {output}")
            
        elif args.command == 'list':
            if args.what in ['presets', 'all']:
                print("\n📦 Available Presets:")
                presets = list_all_presets()
                for category, items in presets.items():
                    print(f"\n  {category.upper()}:")
                    for item in items:
                        print(f"    • {item}")
            
            if args.what in ['effects', 'all']:
                print("\n🎚️  Available Effects:")
                effects = list_all_effects()
                for category, items in effects.items():
                    print(f"\n  {category.upper()}:")
                    for item in items:
                        print(f"    • {item}")
                print()
                
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
