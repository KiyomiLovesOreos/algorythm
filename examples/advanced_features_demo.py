"""
Advanced Example: Demonstrating All User-Facing Capabilities

This example showcases all the new features added to the Algorythm library:
- Noise waveforms
- Extended effects (Flanger, Distortion, Compression)
- Chord objects
- L-System generation
- Cellular Automata
- Data sonification
- Parameter automation
- Sample playback
- Visualization (waveform, spectrogram, frequency scope)
"""

from algorythm.synth import Synth, Oscillator, Filter, ADSR, SynthPresets
from algorythm.sequence import Motif, Scale, Chord, Arpeggiator
from algorythm.structure import Composition, Reverb, Delay, Flanger, Distortion, Compression
from algorythm.generative import LSystem, CellularAutomata
from algorythm.automation import Automation, DataSonification
from algorythm.visualization import WaveformVisualizer, SpectrogramVisualizer
from algorythm.sampler import Sample
import numpy as np


def create_generative_composition():
    """Create a composition using generative techniques."""
    
    # 1. L-System Generated Melody
    print("Generating L-System melody...")
    lsys = LSystem(
        axiom='A',
        rules={'A': 'AB', 'B': 'AC', 'C': 'A'},
        iterations=3
    )
    lsys.generate()
    
    # Map symbols to scale degrees
    symbol_map = {'A': 0, 'B': 2, 'C': 4}
    lsys_motif = lsys.to_motif(symbol_map, scale=Scale.minor('A', 4))
    
    # 2. Cellular Automata Rhythm
    print("Generating Cellular Automata rhythm...")
    ca = CellularAutomata(width=16, height=8)
    ca.evolve()
    ca_motif = ca.to_motif(row=-1, scale=Scale.pentatonic_minor('E', 3))
    
    # 3. Chord Progression
    print("Creating chord progression...")
    chords = [
        Chord.minor('A', 3),
        Chord.major('C', 3),
        Chord.major('G', 3),
        Chord.minor('E', 3),
    ]
    
    # Create composition
    comp = Composition(tempo=120, sample_rate=44100)
    
    # Track 1: L-System Melody with noise-based synth
    print("Adding L-System melody track with noise synthesis...")
    noise_synth = Synth(
        waveform='noise',
        filter=Filter.lowpass(cutoff=800, resonance=0.3),
        envelope=ADSR(attack=0.01, decay=0.2, sustain=0.1, release=0.3)
    )
    comp.add_track('Noise Percussion', noise_synth)
    comp.repeat_motif(lsys_motif, bars=2)
    
    # Add distortion and compression
    comp.add_fx(Distortion(drive=0.5, tone=0.6, mix=0.7))
    comp.add_fx(Compression(threshold=-15.0, ratio=3.0, makeup_gain=1.2))
    
    # Track 2: CA-based bass line with flanger
    print("Adding Cellular Automata bass track with flanger...")
    bass_synth = SynthPresets.bass()
    comp.add_track('CA Bass', bass_synth)
    comp.repeat_motif(ca_motif, bars=2)
    comp.transpose(semitones=-12)  # Drop an octave
    comp.add_fx(Flanger(rate=0.3, depth=0.5, feedback=0.2, mix=0.4))
    
    # Track 3: Chord progression with arpeggiation
    print("Adding arpeggiated chord progression...")
    pad_synth = SynthPresets.warm_pad()
    comp.add_track('Chord Pad', pad_synth)
    
    # Arpeggiate each chord
    arp = Arpeggiator(pattern='up-down', octaves=2)
    for i, chord in enumerate(chords):
        chord_motif = chord.to_motif(duration=0.25)
        arpeggiated = arp.arpeggiate(chord_motif)
        # Add to track starting at bar i
        for note_idx, (interval, duration) in enumerate(zip(arpeggiated.intervals, arpeggiated.durations)):
            freq = arpeggiated.scale.get_frequency(interval)
            start_time = (i * 4 + note_idx * 0.25) * (60.0 / comp.tempo)
            comp.current_track.add_note(freq, start_time, duration * (60.0 / comp.tempo))
    
    comp.add_fx(Reverb(mix=0.5, room_size=0.7))
    
    return comp


def demonstrate_data_sonification():
    """Demonstrate data sonification with automation."""
    
    print("\n=== Data Sonification Demo ===")
    
    # Example: Sonify a simple dataset (e.g., temperature readings)
    data = [20, 22, 25, 23, 28, 30, 27, 24, 21, 19]
    
    print(f"Original data: {data}")
    
    # Create data sonification
    ds = DataSonification(data, param_range=(0.0, 1.0), scaling='linear')
    
    # Map to different musical parameters
    pitches = ds.to_pitch_sequence(scale=Scale.major('C', 4))
    rhythm = ds.to_rhythm_pattern(min_duration=0.25, max_duration=1.0)
    volumes = ds.to_volume_envelope(min_volume=0.3, max_volume=1.0)
    
    print(f"Mapped to {len(pitches)} pitches")
    print(f"Rhythm pattern: {len(rhythm)} durations")
    print(f"Volume envelope: {len(volumes)} levels")
    
    # Create composition with sonified data
    comp = Composition(tempo=120)
    pluck_synth = SynthPresets.pluck()
    comp.add_track('Data Sonification', pluck_synth)
    
    # Add notes with data-driven parameters
    current_time = 0.0
    for freq, duration, volume in zip(pitches, rhythm, volumes):
        # Scale volume
        comp.current_track.add_note(freq, current_time, duration)
        current_time += duration
    
    return comp


def demonstrate_automation():
    """Demonstrate parameter automation."""
    
    print("\n=== Parameter Automation Demo ===")
    
    # Create automation curves
    fade_in = Automation.fade_in(duration=2.0, target_value=1.0)
    fade_out = Automation.fade_out(duration=2.0, start_value=1.0)
    
    # Generate automation curves
    fade_in_curve = fade_in.generate_curve(num_points=50)
    fade_out_curve = fade_out.generate_curve(num_points=50)
    
    print(f"Fade-in curve: {len(fade_in_curve)} points")
    print(f"Fade-out curve: {len(fade_out_curve)} points")
    
    # Create exponential filter cutoff automation
    filter_auto = Automation(
        start_value=200.0,
        end_value=5000.0,
        duration=4.0,
        curve_type='exponential'
    )
    
    print(f"Filter automation at t=0: {filter_auto.get_value(0.0):.1f} Hz")
    print(f"Filter automation at t=2: {filter_auto.get_value(2.0):.1f} Hz")
    print(f"Filter automation at t=4: {filter_auto.get_value(4.0):.1f} Hz")


def demonstrate_visualization(audio_signal, sample_rate=44100):
    """Demonstrate audio visualization."""
    
    print("\n=== Audio Visualization Demo ===")
    
    # Waveform visualization
    waveform_viz = WaveformVisualizer(sample_rate=sample_rate, window_size=1024)
    waveform_data = waveform_viz.generate(audio_signal)
    print(f"Waveform visualization: {waveform_data.shape}")
    
    # Spectrogram visualization
    spec_viz = SpectrogramVisualizer(
        sample_rate=sample_rate,
        window_size=2048,
        hop_size=512
    )
    spectrogram = spec_viz.generate(audio_signal)
    print(f"Spectrogram shape: {spectrogram.shape}")
    print(f"Frequency range: 0 - {sample_rate/2} Hz")
    print(f"Time frames: {spectrogram.shape[1]}")


def main():
    """Main function to run all demonstrations."""
    
    print("=" * 60)
    print("Algorythm Advanced Features Demonstration")
    print("=" * 60)
    
    # 1. Create generative composition
    print("\n1. Creating generative composition...")
    comp = create_generative_composition()
    
    # 2. Demonstrate data sonification
    data_comp = demonstrate_data_sonification()
    
    # 3. Demonstrate automation
    demonstrate_automation()
    
    # 4. Render and visualize
    print("\n4. Rendering composition...")
    audio_signal = comp.render()
    
    if len(audio_signal) > 0:
        print(f"Rendered audio: {len(audio_signal)} samples ({len(audio_signal)/44100:.2f} seconds)")
        
        # Demonstrate visualization
        demonstrate_visualization(audio_signal)
        
        # Export audio
        print("\n5. Exporting audio...")
        comp.render(file_path='generative_demo.wav', quality='high', formats=['wav'])
        print("Exported to: generative_demo.wav")
    else:
        print("Warning: No audio generated")
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
