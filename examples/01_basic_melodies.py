"""
Example: Basic Melody Creation

Learn how to create simple melodies using the synthesizer.
Perfect for beginners!
"""

import numpy as np
from algorythm.synth import Synth
from algorythm.export import Exporter


def example_1_simple_tone():
    """Play a simple 440Hz tone (A4)."""
    print("\n" + "=" * 60)
    print("Example 1: Simple Tone")
    print("=" * 60)
    
    synth = Synth(waveform='sine')
    signal = synth.generate_note(frequency=440, duration=1.0)
    
    exporter = Exporter()
    exporter.export(signal, 'simple_tone.wav')
    print("✓ Created simple_tone.wav")


def example_2_melody():
    """Create a simple melody."""
    print("\n" + "=" * 60)
    print("Example 2: Simple Melody")
    print("=" * 60)
    
    synth = Synth(waveform='sine')
    
    # C major scale
    notes = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
    note_names = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C']
    
    signal_parts = []
    for freq, name in zip(notes, note_names):
        print(f"  Playing {name}...")
        note = synth.generate_note(frequency=freq, duration=0.5)
        signal_parts.append(note)
    
    signal = np.concatenate(signal_parts)
    
    exporter = Exporter()
    exporter.export(signal, 'melody_scale.wav')
    print("✓ Created melody_scale.wav")


def example_3_different_waveforms():
    """Compare different waveform types."""
    print("\n" + "=" * 60)
    print("Example 3: Different Waveforms")
    print("=" * 60)
    
    waveforms = ['sine', 'square', 'saw', 'triangle']
    exporter = Exporter()
    
    for waveform in waveforms:
        print(f"  Creating {waveform} waveform...")
        synth = Synth(waveform=waveform)
        signal = synth.generate_note(frequency=440, duration=1.0)
        exporter.export(signal, f'waveform_{waveform}.wav')
    
    print("✓ Created 4 waveform examples")


def example_4_twinkle_twinkle():
    """Create 'Twinkle Twinkle Little Star'."""
    print("\n" + "=" * 60)
    print("Example 4: Twinkle Twinkle Little Star")
    print("=" * 60)
    
    synth = Synth(waveform='sine')
    
    # Note frequencies (C major scale)
    C, D, E, F, G, A = 261.63, 293.66, 329.63, 349.23, 392.00, 440.00
    
    # Twinkle Twinkle melody
    melody = [
        (C, 0.5), (C, 0.5), (G, 0.5), (G, 0.5),
        (A, 0.5), (A, 0.5), (G, 1.0),
        (F, 0.5), (F, 0.5), (E, 0.5), (E, 0.5),
        (D, 0.5), (D, 0.5), (C, 1.0)
    ]
    
    signal_parts = []
    for freq, duration in melody:
        note = synth.generate_note(frequency=freq, duration=duration)
        signal_parts.append(note)
    
    signal = np.concatenate(signal_parts)
    
    exporter = Exporter()
    exporter.export(signal, 'twinkle_twinkle.wav')
    print("✓ Created twinkle_twinkle.wav")


def main():
    print("\n" + "=" * 60)
    print("BASIC MELODY EXAMPLES")
    print("=" * 60)
    print("\nLearn the basics of melody creation with Algorythm!")
    
    example_1_simple_tone()
    example_2_melody()
    example_3_different_waveforms()
    example_4_twinkle_twinkle()
    
    print("\n" + "=" * 60)
    print("✓ All examples complete! Check ~/Music/ for the files.")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
