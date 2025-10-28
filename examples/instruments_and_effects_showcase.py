"""
Showcase of all instruments and effects in Algorythm.

This example demonstrates:
- Physical modeling synth (strings, drums, wind)
- Additive synthesis
- Pad synth with detuned voices
- All effects (reverb, delay, chorus, flanger, phaser, distortion, etc.)
- Effect chains
"""

from algorythm import *


def main():
    # Create composition
    comp = Composition(tempo=120, time_signature=(4, 4))
    
    # ========== INSTRUMENTS SHOWCASE ==========
    
    # 1. Physical Model Synths
    print("Creating physical model instruments...")
    
    # String instrument (guitar-like)
    string_synth = PhysicalModelSynth(
        model_type='string',
        damping=0.996,
        brightness=0.7
    )
    
    # Drum instrument
    drum_synth = PhysicalModelSynth(
        model_type='drum',
        damping=0.98,
        brightness=0.6
    )
    
    # Wind instrument
    wind_synth = PhysicalModelSynth(
        model_type='wind',
        damping=0.995,
        brightness=0.5
    )
    
    # 2. Additive Synth (organ-like)
    print("Creating additive synth...")
    organ = AdditiveeSynth(
        num_harmonics=8,
        harmonic_amplitudes=[1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05],
        envelope=ADSR(attack=0.01, decay=0.1, sustain=1.0, release=0.1)
    )
    
    # 3. Pad Synth (lush strings)
    print("Creating pad synth...")
    pad = PadSynth(
        num_voices=7,
        detune_amount=0.1,
        waveform='saw',
        envelope=ADSR(attack=2.0, decay=1.0, sustain=0.8, release=3.0)
    )
    
    # 4. FM Synth presets
    bell = SynthPresets.bell()
    brass = SynthPresets.brass()
    
    # 5. Standard synth presets
    bass = SynthPresets.bass()
    lead = SynthPresets.lead()
    pluck = SynthPresets.pluck()
    
    # ========== SCALES AND MOTIFS ==========
    
    scale = Scale.major('C', 4)
    
    # Create motifs for different instruments
    string_motif = Motif(notes=scale.ascending(8), durations=[0.5] * 8)
    drum_motif = Motif(notes=[scale.tonic()] * 4, durations=[1.0] * 4)
    organ_motif = Motif(notes=[scale.note(0), scale.note(2), scale.note(4), scale.note(5)],
                        durations=[1.0, 1.0, 1.0, 1.0])
    pad_chord = Motif(notes=[scale.note(0), scale.note(2), scale.note(4)], durations=[8.0] * 3)
    bell_arp = Arpeggiator(chord=Chord.major(scale.tonic()), pattern='up', duration=4.0)
    
    # ========== TRACKS WITH INSTRUMENTS ==========
    
    print("Creating tracks...")
    
    # String track
    string_track = Track("Strings")
    string_track.add_motif(string_motif, instrument=string_synth, start_time=0.0)
    
    # Drum track
    drum_track = Track("Drums")
    drum_track.add_motif(drum_motif, instrument=drum_synth, start_time=0.0)
    
    # Organ track
    organ_track = Track("Organ")
    organ_track.add_motif(organ_motif, instrument=organ, start_time=2.0)
    
    # Pad track
    pad_track = Track("Pad")
    pad_track.add_motif(pad_chord, instrument=pad, start_time=0.0)
    
    # Bell arpeggio track
    bell_track = Track("Bell")
    bell_track.add_motif(bell_arp.generate(), instrument=bell, start_time=4.0)
    
    # Bass track
    bass_motif = Motif(notes=[scale.note(-12)] * 4, durations=[1.0] * 4)
    bass_track = Track("Bass")
    bass_track.add_motif(bass_motif, instrument=bass, start_time=0.0)
    
    # ========== EFFECTS SHOWCASE ==========
    
    print("Adding effects...")
    
    # Create effect chains for different tracks
    
    # Reverb effect
    reverb = ReverbFX(room_size=0.7, damping=0.5, wet_level=0.3)
    string_track.add_effect(reverb)
    
    # Delay effect
    delay = DelayFX(delay_time=0.375, feedback=0.5, wet_level=0.4)
    bell_track.add_effect(delay)
    
    # Chorus effect
    chorus = ChorusFX(depth=0.003, rate=1.5, mix=0.5)
    organ_track.add_effect(chorus)
    
    # Distortion + Reverb chain
    distortion = DistortionFX(drive=3.0, tone=0.6, mix=0.7)
    reverb2 = ReverbFX(room_size=0.5, damping=0.6, wet_level=0.2)
    bass_track.add_effect(distortion)
    bass_track.add_effect(reverb2)
    
    # Flanger effect
    flanger = FlangerFX(depth=0.002, rate=0.3, feedback=0.6, mix=0.5)
    pad_track.add_effect(flanger)
    
    # Phaser effect
    phaser = PhaserFX(rate=0.5, depth=1.0, feedback=0.5, mix=0.4)
    # You can add this to any track: string_track.add_effect(phaser)
    
    # Compressor (dynamics control)
    compressor = Compressor(threshold=-15.0, ratio=4.0, attack=0.005, release=0.1)
    drum_track.add_effect(compressor)
    
    # Tremolo (amplitude modulation)
    tremolo = TremoloFX(rate=5.0, depth=0.5, waveform='sine')
    # Example: organ_track.add_effect(tremolo)
    
    # Ring Modulator (metallic effect)
    ring_mod = RingModulator(carrier_freq=440.0, mix=0.3)
    # Example for experimental sounds: bell_track.add_effect(ring_mod)
    
    # Bit Crusher (lo-fi effect)
    bitcrusher = BitCrusherFX(bit_depth=8, sample_rate_reduction=0.5, mix=0.6)
    # Example for retro sounds: drum_track.add_effect(bitcrusher)
    
    # ========== ADD TRACKS TO COMPOSITION ==========
    
    comp.add_track(string_track)
    comp.add_track(drum_track)
    comp.add_track(organ_track)
    comp.add_track(pad_track)
    comp.add_track(bell_track)
    comp.add_track(bass_track)
    
    # ========== RENDER ==========
    
    print("Rendering composition...")
    engine = RenderEngine(sample_rate=44100)
    audio = engine.render(comp)
    
    # Export
    exporter = Exporter()
    output_file = "instruments_and_effects_showcase.wav"
    exporter.export_wav(audio, output_file)
    print(f"Saved to {output_file}")
    
    # ========== ADDITIONAL EFFECTS EXAMPLES ==========
    
    print("\nAvailable effects not used in this example:")
    print("- Overdrive: Smooth tube-like distortion")
    print("- Fuzz: Extreme clipping distortion")
    print("- Limiter: Peak limiting for mastering")
    print("- Gate: Noise gate for removing low-level signals")
    print("- Vibrato: Pitch modulation effect")
    print("- AutoPan: Stereo auto-panning")
    
    print("\nEffect chain example:")
    print("fx_chain = FXChain()")
    print("fx_chain.add_effect(DistortionFX(drive=5.0))")
    print("fx_chain.add_effect(ChorusFX(mix=0.3))")
    print("fx_chain.add_effect(ReverbFX(room_size=0.8))")
    print("processed = fx_chain.apply(audio)")
    
    print("\nAll instrument presets:")
    print("- SynthPresets.warm_pad()")
    print("- SynthPresets.pluck()")
    print("- SynthPresets.bass()")
    print("- SynthPresets.lead()")
    print("- SynthPresets.organ()")
    print("- SynthPresets.bell()")
    print("- SynthPresets.strings()")
    print("- SynthPresets.guitar()")
    print("- SynthPresets.drum()")
    print("- SynthPresets.brass()")


if __name__ == "__main__":
    main()
