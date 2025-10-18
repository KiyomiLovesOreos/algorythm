"""
Advanced Example

This example demonstrates more advanced features including multiple tracks,
different synth presets, and various musical transformations.
"""

from audionaut.synth import SynthPresets
from audionaut.sequence import Motif, Scale, Arpeggiator
from audionaut.structure import Composition, Reverb, Delay

# Create a composition
composition = Composition(tempo=128)

# Add a bassline track
bass_synth = SynthPresets.bass()
bass_motif = Motif.from_intervals([0, 0, 0, 0], scale=Scale.major('A', octave=2))
composition.add_track('Bass', bass_synth).repeat_motif(bass_motif, bars=4)

# Add a lead melody track
lead_synth = SynthPresets.pluck()
lead_motif = Motif.from_intervals([0, 2, 4, 7, 9, 7, 4, 2], scale=Scale.major('A', octave=4))

# Apply arpeggiator
arpeggiator = Arpeggiator(pattern='up-down', octaves=2)
arpeggiated = arpeggiator.arpeggiate(lead_motif)

composition.add_track('Lead', lead_synth) \
    .repeat_motif(arpeggiated, bars=2) \
    .add_fx(Delay(delay_time=0.25, feedback=0.3, mix=0.3)) \
    .add_fx(Reverb(mix=0.2))

# Add a pad track
pad_synth = SynthPresets.warm_pad()
pad_motif = Motif.from_intervals([0, 4, 7], scale=Scale.major('A', octave=3))
composition.add_track('Pad', pad_synth) \
    .repeat_motif(pad_motif, bars=4) \
    .add_fx(Reverb(mix=0.5))

print("Created advanced composition:")
print("  Tracks:")
print("    - Bass: Using bass preset with A major scale")
print("    - Lead: Using pluck preset with arpeggiated melody")
print("    - Pad: Using warm pad preset with chord progression")
print("\n  Effects:")
print("    - Lead: Delay (0.25s) + Reverb (20%)")
print("    - Pad: Reverb (50%)")

# Render
print("\nRendering...")
audio = composition.render(file_path='advanced_track.wav', quality='high')
print(f"Exported to 'advanced_track.wav' ({len(audio)} samples)")
