"""
Volume Control Demo

This example demonstrates the volume control features in algorythm,
including track volume, master volume, volume utilities, and fades.
"""

from algorythm.synth import Synth, Filter, ADSR, SynthPresets
from algorythm.sequence import Motif, Scale
from algorythm.structure import Composition, Reverb, VolumeControl
from algorythm.export import Exporter
import numpy as np

print("=== Algorythm Volume Control Demo ===\n")

# Create different synths for multiple tracks
print("1. Creating synths...")
bass_synth = SynthPresets.bass()
lead_synth = SynthPresets.pluck()
pad_synth = SynthPresets.warm_pad()

# Create motifs
print("2. Creating motifs...")
bass_line = Motif.from_intervals([0, 0, 5, 5], scale=Scale.minor('C', octave=2))
bass_line.durations = [1.0, 1.0, 1.0, 1.0]

melody = Motif.from_intervals([0, 2, 4, 5, 7, 5, 4, 2], scale=Scale.minor('C', octave=4))
melody.durations = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

pad_chords = Motif.from_intervals([0, 3, 7], scale=Scale.minor('C', octave=3))
pad_chords.durations = [4.0, 4.0, 4.0]

# Create composition with volume control
print("3. Creating composition with volume control...")
comp = Composition(tempo=120)

# Add tracks with different volumes
comp.add_track('Bass', bass_synth).repeat_motif(bass_line, bars=4)
comp.set_track_volume('Bass', 0.8)  # Slightly quieter bass

comp.add_track('Lead', lead_synth).repeat_motif(melody, bars=4)
comp.set_track_volume('Lead', 1.0)  # Full volume lead

comp.add_track('Pad', pad_synth).repeat_motif(pad_chords, bars=4)
comp.set_track_volume('Pad', 0.4)  # Quiet background pad

# Set master volume
comp.set_master_volume(0.9)

# Add fade in and fade out
comp.fade_in(1.0).fade_out(2.0)

print("4. Rendering with volume control...")
audio = comp.render()

# Export
print("5. Exporting to WAV...")
exporter = Exporter()
exporter.export(audio, 'volume_demo.wav', sample_rate=44100)
print("   Exported: volume_demo.wav")

# Demonstrate VolumeControl utilities
print("\n6. VolumeControl Utility Demonstrations:")

# dB to linear conversion
print("\n   a) Decibel conversions:")
for db in [-20, -10, -6, -3, 0, 3, 6]:
    linear = VolumeControl.db_to_linear(db)
    print(f"      {db:>3} dB = {linear:.4f} linear")

# Linear to dB conversion
print("\n   b) Linear to dB conversions:")
for linear in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
    db = VolumeControl.linear_to_db(linear)
    print(f"      {linear:.2f} linear = {db:>6.2f} dB")

# Apply volume with dB
print("\n   c) Creating test signal and applying volume...")
test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))

# Apply -6 dB attenuation
attenuated = VolumeControl.apply_db_volume(test_signal, -6.0)
print(f"      Original peak: {np.max(np.abs(test_signal)):.4f}")
print(f"      After -6 dB:   {np.max(np.abs(attenuated)):.4f}")

# Normalize to -3 dB
normalized = VolumeControl.normalize(test_signal, target_db=-3.0)
print(f"      Normalized to -3 dB: {np.max(np.abs(normalized)):.4f}")

# Apply fades with different curves
print("\n   d) Testing fade curves...")
for curve_type in ['linear', 'exponential', 'logarithmic']:
    faded = VolumeControl.fade(
        test_signal,
        fade_in=0.1,
        fade_out=0.1,
        sample_rate=44100,
        curve=curve_type
    )
    exporter.export(faded, f'fade_{curve_type}_demo.wav', sample_rate=44100)
    print(f"      Exported: fade_{curve_type}_demo.wav")

print("\n=== Demo Complete! ===")
print("\nFiles created:")
print("  - volume_demo.wav (full composition with volume control)")
print("  - fade_linear_demo.wav (linear fade curve)")
print("  - fade_exponential_demo.wav (exponential fade curve)")
print("  - fade_logarithmic_demo.wav (logarithmic fade curve)")
