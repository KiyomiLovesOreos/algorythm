"""
Example: Generative Music

Create music using algorithmic composition techniques.
"""

import numpy as np
from algorythm.synth import Synth, Filter, ADSR
from algorythm.export import Exporter


def example_1_random_walk():
    """Create a melody using random walk algorithm."""
    print("\n" + "=" * 60)
    print("Example 1: Random Walk Melody")
    print("=" * 60)
    
    synth = Synth(
        waveform='sine',
        envelope=ADSR(attack=0.05, decay=0.2, sustain=0.5, release=0.3)
    )
    
    # Start at middle C
    current_freq = 261.63
    melody = [current_freq]
    
    # Random walk for 16 notes
    np.random.seed(42)  # For reproducibility
    for _ in range(15):
        # Move up or down by a small amount
        change = np.random.choice([-2, -1, 0, 1, 2])
        # Convert semitones to frequency multiplier
        current_freq *= 2 ** (change / 12)
        # Keep in reasonable range
        current_freq = np.clip(current_freq, 130.81, 523.25)
        melody.append(current_freq)
    
    signal_parts = []
    for freq in melody:
        note = synth.generate_note(frequency=freq, duration=0.3)
        signal_parts.append(note)
    
    signal = np.concatenate(signal_parts)
    
    exporter = Exporter()
    exporter.export(signal, 'generative_random_walk.wav')
    print("✓ Created generative_random_walk.wav")


def example_2_pentatonic_pattern():
    """Create music using a pentatonic scale pattern."""
    print("\n" + "=" * 60)
    print("Example 2: Pentatonic Pattern")
    print("=" * 60)
    
    synth = Synth(
        waveform='square',
        filter=Filter.lowpass(cutoff=2000, resonance=0.5),
        envelope=ADSR(attack=0.01, decay=0.15, sustain=0.4, release=0.2)
    )
    
    # C pentatonic scale
    pentatonic = [261.63, 293.66, 329.63, 392.00, 440.00]
    
    # Generate a pattern
    np.random.seed(123)
    pattern_length = 32
    pattern = np.random.choice(len(pentatonic), pattern_length)
    
    signal_parts = []
    for idx in pattern:
        freq = pentatonic[idx]
        note = synth.generate_note(frequency=freq, duration=0.25)
        signal_parts.append(note)
    
    signal = np.concatenate(signal_parts)
    
    exporter = Exporter()
    exporter.export(signal, 'generative_pentatonic.wav')
    print("✓ Created generative_pentatonic.wav")


def example_3_arpeggiator():
    """Create arpeggios from chord progressions."""
    print("\n" + "=" * 60)
    print("Example 3: Arpeggiator")
    print("=" * 60)
    
    synth = Synth(
        waveform='sine',
        envelope=ADSR(attack=0.01, decay=0.1, sustain=0.3, release=0.2)
    )
    
    # Chord progression (C, Am, F, G)
    chords = [
        [261.63, 329.63, 392.00],    # C major
        [220.00, 261.63, 329.63],    # A minor
        [174.61, 220.00, 261.63],    # F major
        [196.00, 246.94, 293.66],    # G major
    ]
    
    signal_parts = []
    for chord in chords:
        # Arpeggiate up
        for freq in chord:
            note = synth.generate_note(frequency=freq, duration=0.2)
            signal_parts.append(note)
        # Arpeggiate down
        for freq in reversed(chord):
            note = synth.generate_note(frequency=freq, duration=0.2)
            signal_parts.append(note)
    
    signal = np.concatenate(signal_parts)
    
    exporter = Exporter()
    exporter.export(signal, 'generative_arpeggio.wav')
    print("✓ Created generative_arpeggio.wav")


def example_4_euclidean_rhythm():
    """Create rhythms using Euclidean algorithm."""
    print("\n" + "=" * 60)
    print("Example 4: Euclidean Rhythm")
    print("=" * 60)
    
    synth = Synth(
        waveform='square',
        envelope=ADSR(attack=0.001, decay=0.05, sustain=0.1, release=0.05)
    )
    
    def euclidean_rhythm(steps, pulses):
        """Generate Euclidean rhythm pattern."""
        pattern = [0] * steps
        bucket = 0
        for i in range(steps):
            bucket += pulses
            if bucket >= steps:
                bucket -= steps
                pattern[i] = 1
        return pattern
    
    # Create a 16-step pattern with 5 pulses
    pattern = euclidean_rhythm(16, 5)
    
    signal_parts = []
    for hit in pattern:
        if hit:
            note = synth.generate_note(frequency=220, duration=0.15)
        else:
            note = np.zeros(int(44100 * 0.15))
        signal_parts.append(note)
    
    # Repeat pattern 4 times
    signal = np.concatenate(signal_parts * 4)
    
    exporter = Exporter()
    exporter.export(signal, 'generative_euclidean.wav')
    print("✓ Created generative_euclidean.wav")


def example_5_markov_chain():
    """Create a melody using Markov chain."""
    print("\n" + "=" * 60)
    print("Example 5: Markov Chain Melody")
    print("=" * 60)
    
    synth = Synth(
        waveform='triangle',
        envelope=ADSR(attack=0.05, decay=0.2, sustain=0.6, release=0.3)
    )
    
    # Define note transitions (simplified Markov chain)
    notes = [261.63, 293.66, 329.63, 349.23, 392.00]  # C, D, E, F, G
    
    # Transition probabilities (each row sums to 1)
    transitions = np.array([
        [0.2, 0.3, 0.3, 0.1, 0.1],  # From C
        [0.3, 0.2, 0.3, 0.1, 0.1],  # From D
        [0.2, 0.2, 0.2, 0.2, 0.2],  # From E
        [0.1, 0.2, 0.3, 0.2, 0.2],  # From F
        [0.2, 0.2, 0.2, 0.2, 0.2],  # From G
    ])
    
    # Generate melody
    np.random.seed(456)
    current_note = 0  # Start with C
    melody_indices = [current_note]
    
    for _ in range(31):
        # Choose next note based on transition probabilities
        current_note = np.random.choice(len(notes), p=transitions[current_note])
        melody_indices.append(current_note)
    
    signal_parts = []
    for idx in melody_indices:
        freq = notes[idx]
        note = synth.generate_note(frequency=freq, duration=0.35)
        signal_parts.append(note)
    
    signal = np.concatenate(signal_parts)
    
    exporter = Exporter()
    exporter.export(signal, 'generative_markov.wav')
    print("✓ Created generative_markov.wav")


def main():
    print("\n" + "=" * 60)
    print("GENERATIVE MUSIC EXAMPLES")
    print("=" * 60)
    print("\nCreate music using algorithmic composition techniques!")
    
    example_1_random_walk()
    example_2_pentatonic_pattern()
    example_3_arpeggiator()
    example_4_euclidean_rhythm()
    example_5_markov_chain()
    
    print("\n" + "=" * 60)
    print("✓ All examples complete! Check ~/Music/ for the files.")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
