"""
Simple example showing the new instruments and effects.
This creates a short musical piece using the new features.
"""

from algorythm import *
from algorythm.effects import *


def main():
    print("Creating a short composition with new instruments and effects...")
    
    # Create scale
    scale = Scale.major('C', 4)
    
    # Create simple melodies
    guitar_notes = [scale.note(0), scale.note(2), scale.note(4), scale.note(5)]
    guitar_melody = Motif(notes=guitar_notes, durations=[1.0, 1.0, 1.0, 1.0])
    
    strings_chord = [scale.note(0), scale.note(2), scale.note(4)]
    strings_motif = Motif(notes=strings_chord, durations=[4.0, 4.0, 4.0])
    
    drum_note = [scale.note(-12)]  # Low C
    drum_rhythm = Motif(notes=drum_note * 8, durations=[0.5] * 8)
    
    # Create instruments using new presets
    guitar = SynthPresets.guitar()
    strings = SynthPresets.strings()
    drum = SynthPresets.drum()
    
    # Create composition
    comp = Composition(tempo=100)
    
    # Add guitar track with reverb
    guitar_track = Track("Guitar")
    guitar_track.add_motif(guitar_melody, instrument=guitar, start_time=0.0)
    guitar_track.add_effect(ReverbFX(room_size=0.6, wet_level=0.3))
    comp.add_track(guitar_track)
    
    # Add strings track with chorus
    strings_track = Track("Strings")
    strings_track.add_motif(strings_motif, instrument=strings, start_time=0.0)
    strings_track.add_effect(ChorusFX(depth=0.003, rate=1.5, mix=0.4))
    comp.add_track(strings_track)
    
    # Add drum track with compression
    drum_track = Track("Drums")
    drum_track.add_motif(drum_rhythm, instrument=drum, start_time=0.0)
    drum_track.add_effect(Compressor(threshold=-15.0, ratio=4.0))
    comp.add_track(drum_track)
    
    # Render
    print("Rendering...")
    engine = RenderEngine(sample_rate=44100)
    audio = engine.render(comp)
    
    # Export
    exporter = Exporter()
    output_file = "simple_example_new_features.wav"
    exporter.export_wav(audio, output_file)
    print(f"✓ Saved to {output_file}")
    print(f"  Duration: {len(audio) / 44100:.2f} seconds")


if __name__ == "__main__":
    main()
