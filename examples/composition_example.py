"""
Composition Example

This example demonstrates the full API as described in the problem statement.
"""

from audionaut.synth import Synth, Filter, ADSR
from audionaut.sequence import Motif, Scale
from audionaut.structure import Composition, Reverb

# Create a warm synth sound
warm_pad = Synth(
    waveform='saw',
    filter=Filter.lowpass(cutoff=2000, resonance=0.6),
    envelope=ADSR(attack=1.5, decay=0.5, sustain=0.8, release=2.0)
)

# Create a simple, rising motif
melody = Motif.from_intervals([0, 2, 4, 7], scale=Scale.major('C'))

# Create the main composition structure
final_track = Composition(tempo=120) \
    .add_track('Bassline', warm_pad) \
    .repeat_motif(melody, bars=8) \
    .transpose(semitones=5) \
    .add_fx(Reverb(mix=0.4))

print("Created composition:")
print(f"  - Tempo: 120 BPM")
print(f"  - Track: Bassline with warm pad synth")
print(f"  - Melody: C major scale [0, 2, 4, 7] degrees")
print(f"  - Repeated for 8 bars")
print(f"  - Transposed up 5 semitones")
print(f"  - Effects: Reverb (40% mix)")

# Render the final output
print("\nRendering audio...")
audio = final_track.render(
    file_path='epic_track.wav',
    quality='high',
    formats=['wav']  # Would support ['flac', 'mp3', 'ogg'] with additional dependencies
)

print(f"Rendered {len(audio)} samples")
print("Exported to 'epic_track.wav'")
