"""
Instrument Showcase Example

Demonstrates the wide variety of available instrument presets in Algorythm.
Creates a short composition using many different instrument types.
"""

from algorythm.synth import SynthPresets, Oscillator
import numpy as np

# Sample rate
SAMPLE_RATE = 44100

def showcase_keyboards():
    """Showcase keyboard instruments."""
    print("\n=== Keyboard Instruments ===")
    
    instruments = {
        'Piano': SynthPresets.piano(),
        'Electric Piano': SynthPresets.electric_piano(),
        'Organ': SynthPresets.organ(),
        'Harpsichord': SynthPresets.harpsichord(),
        'Clavinet': SynthPresets.clavinet(),
    }
    
    # Play a C major chord
    frequencies = [261.63, 329.63, 392.00]  # C, E, G
    
    for name, instrument in instruments.items():
        print(f"  • {name}")
        audio = np.zeros(int(1.5 * SAMPLE_RATE))
        for freq in frequencies:
            note = instrument.generate_note(freq, 1.5, SAMPLE_RATE)
            audio[:len(note)] += note / len(frequencies)
    
    return audio


def showcase_strings():
    """Showcase string instruments."""
    print("\n=== String Instruments ===")
    
    instruments = {
        'Violin': SynthPresets.violin(),
        'Cello': SynthPresets.cello(),
        'Guitar': SynthPresets.guitar(),
        'Strings Ensemble': SynthPresets.strings(),
        'Harp': SynthPresets.harp(),
        'Banjo': SynthPresets.banjo(),
        'Sitar': SynthPresets.sitar(),
    }
    
    melody = [440.0, 493.88, 523.25, 587.33]  # A, B, C, D
    
    for name, instrument in instruments.items():
        print(f"  • {name}")
        audio = np.array([])
        for freq in melody:
            note = instrument.generate_note(freq, 0.5, SAMPLE_RATE)
            audio = np.concatenate([audio, note])
    
    return audio


def showcase_winds():
    """Showcase wind instruments."""
    print("\n=== Wind Instruments ===")
    
    instruments = {
        'Flute': SynthPresets.flute(),
        'Clarinet': SynthPresets.clarinet(),
        'Trumpet': SynthPresets.trumpet(),
        'Saxophone': SynthPresets.saxophone(),
    }
    
    melody = [523.25, 587.33, 659.25, 698.46, 783.99]  # C, D, E, F, G
    
    for name, instrument in instruments.items():
        print(f"  • {name}")
        audio = np.array([])
        for freq in melody:
            note = instrument.generate_note(freq, 0.4, SAMPLE_RATE)
            audio = np.concatenate([audio, note])
    
    return audio


def showcase_brass():
    """Showcase brass instruments."""
    print("\n=== Brass Instruments ===")
    
    instruments = {
        'Brass Section': SynthPresets.brass(),
        'Synth Brass': SynthPresets.synth_brass(),
    }
    
    chord = [261.63, 329.63, 392.00, 493.88]  # C, E, G, B
    
    for name, instrument in instruments.items():
        print(f"  • {name}")
        audio = np.zeros(int(2.0 * SAMPLE_RATE))
        for freq in chord:
            note = instrument.generate_note(freq, 2.0, SAMPLE_RATE)
            audio[:len(note)] += note / len(chord)
    
    return audio


def showcase_mallets():
    """Showcase mallet percussion instruments."""
    print("\n=== Mallet Percussion ===")
    
    instruments = {
        'Bell': SynthPresets.bell(),
        'Marimba': SynthPresets.marimba(),
        'Xylophone': SynthPresets.xylophone(),
        'Vibraphone': SynthPresets.vibraphone(),
        'Glockenspiel': SynthPresets.glockenspiel(),
        'Kalimba': SynthPresets.kalimba(),
        'Music Box': SynthPresets.music_box(),
        'Steel Drum': SynthPresets.steel_drum(),
    }
    
    arpeggio = [523.25, 659.25, 783.99, 1046.50]  # C, E, G, C
    
    for name, instrument in instruments.items():
        print(f"  • {name}")
        audio = np.array([])
        for freq in arpeggio:
            note = instrument.generate_note(freq, 0.6, SAMPLE_RATE)
            audio = np.concatenate([audio, note])
    
    return audio


def showcase_drums():
    """Showcase drum sounds."""
    print("\n=== Drum Sounds ===")
    
    instruments = {
        'Kick Drum': SynthPresets.kick_drum(),
        'Snare Drum': SynthPresets.snare_drum(),
        'Tom Drum': SynthPresets.tom_drum(),
        'Hi-Hat': SynthPresets.hi_hat(),
        'Cymbal': SynthPresets.cymbal(),
    }
    
    frequencies = {
        'Kick Drum': 60.0,
        'Snare Drum': 180.0,
        'Tom Drum': 120.0,
        'Hi-Hat': 200.0,
        'Cymbal': 300.0,
    }
    
    for name, instrument in instruments.items():
        print(f"  • {name}")
        freq = frequencies[name]
        audio = instrument.generate_note(freq, 0.3, SAMPLE_RATE)
    
    return audio


def showcase_synths():
    """Showcase synthesizer sounds."""
    print("\n=== Synthesizer Sounds ===")
    
    instruments = {
        'Synth Lead': SynthPresets.synth_lead(),
        'Synth Pluck': SynthPresets.synth_pluck(),
        'Arp Synth': SynthPresets.arp_synth(),
        'Acid Bass': SynthPresets.acid_bass(),
        'Fat Bass': SynthPresets.fat_bass(),
        'Sub Bass': SynthPresets.sub_bass(),
        'Warm Pad': SynthPresets.warm_pad(),
        'Ambient Pad': SynthPresets.ambient_pad(),
        'Choir': SynthPresets.choir(),
    }
    
    # Different patterns for different types
    lead_melody = [523.25, 587.33, 659.25, 783.99, 659.25, 587.33]
    bass_note = 130.81  # C2
    pad_chord = [261.63, 329.63, 392.00]  # C, E, G
    
    for name, instrument in instruments.items():
        print(f"  • {name}")
        
        if 'Bass' in name:
            audio = instrument.generate_note(bass_note, 1.0, SAMPLE_RATE)
        elif 'Pad' in name or 'Choir' in name:
            audio = np.zeros(int(2.0 * SAMPLE_RATE))
            for freq in pad_chord:
                note = instrument.generate_note(freq, 2.0, SAMPLE_RATE)
                audio[:len(note)] += note / len(pad_chord)
        else:
            audio = np.array([])
            for freq in lead_melody[:4]:
                note = instrument.generate_note(freq, 0.3, SAMPLE_RATE)
                audio = np.concatenate([audio, note])
    
    return audio


def main():
    """Run the instrument showcase."""
    print("=" * 60)
    print("ALGORYTHM INSTRUMENT SHOWCASE")
    print("=" * 60)
    print("\nDemonstrating 50+ instrument presets available in Algorythm")
    
    # Showcase each category
    showcase_keyboards()
    showcase_strings()
    showcase_winds()
    showcase_brass()
    showcase_mallets()
    showcase_drums()
    showcase_synths()
    
    print("\n" + "=" * 60)
    print("All instruments showcased successfully!")
    print("\nYou can use any of these presets in your compositions:")
    print("  from algorythm.synth import SynthPresets")
    print("  instrument = SynthPresets.piano()")
    print("  audio = instrument.generate_note(440.0, 1.0)")
    print("=" * 60)


if __name__ == '__main__':
    main()
