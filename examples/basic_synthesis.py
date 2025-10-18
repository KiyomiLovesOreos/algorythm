"""
Basic Synthesis Example

This example demonstrates the core API for creating a simple synth sound.
"""

from algorythm.synth import Synth, Filter, ADSR

# Create a warm synth sound (Example from the problem statement)
warm_pad = Synth(
    waveform='saw',
    filter=Filter.lowpass(cutoff=2000, resonance=0.6),
    envelope=ADSR(attack=1.5, decay=0.5, sustain=0.8, release=2.0)
)

# Generate a single note
note_signal = warm_pad.generate_note(frequency=440.0, duration=3.0)

print(f"Generated a warm pad sound:")
print(f"  - Waveform: saw")
print(f"  - Filter: lowpass at 2000 Hz")
print(f"  - Envelope: A=1.5s, D=0.5s, S=0.8, R=2.0s")
print(f"  - Signal length: {len(note_signal)} samples")

# Export to WAV file
from algorythm.export import Exporter

exporter = Exporter()
exporter.export(note_signal, 'warm_pad.wav', sample_rate=44100)
print("\nExported to 'warm_pad.wav'")
