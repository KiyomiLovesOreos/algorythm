"""
Command-line interface for algorythm.
"""

import argparse
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Algorythm - A Python Library for Algorithmic Music',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  algorythm --version      Show version information
  algorythm --help         Show this help message

For more information, visit: https://github.com/KiyomiLovesOreos/synthesia
        """
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.1.0'
    )
    
    parser.add_argument(
        '--example',
        choices=['basic', 'composition', 'advanced'],
        help='Run an example script'
    )
    
    args = parser.parse_args()
    
    if args.example:
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


if __name__ == '__main__':
    main()
