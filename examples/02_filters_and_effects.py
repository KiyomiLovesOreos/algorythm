"""
Example: Filters and Effects

Learn how to use filters and envelopes to shape your sound.
"""

import numpy as np
from algorythm.synth import Synth, Filter, ADSR
from algorythm.export import Exporter


def example_1_lowpass_filter():
    """Demonstrate lowpass filter effect."""
    print("\n" + "=" * 60)
    print("Example 1: Lowpass Filter")
    print("=" * 60)
    
    synth = Synth(waveform='saw')
    exporter = Exporter()
    
    # Without filter
    print("  Creating unfiltered saw wave...")
    signal1 = synth.generate_note(frequency=220, duration=2.0)
    exporter.export(signal1, 'filter_none.wav')
    
    # With lowpass filter
    print("  Creating lowpass filtered saw wave...")
    synth_filtered = Synth(
        waveform='saw',
        filter=Filter.lowpass(cutoff=500, resonance=0.7)
    )
    signal2 = synth_filtered.generate_note(frequency=220, duration=2.0)
    exporter.export(signal2, 'filter_lowpass.wav')
    
    print("✓ Created filter examples")


def example_2_highpass_filter():
    """Demonstrate highpass filter effect."""
    print("\n" + "=" * 60)
    print("Example 2: Highpass Filter")
    print("=" * 60)
    
    synth_highpass = Synth(
        waveform='saw',
        filter=Filter.highpass(cutoff=1000, resonance=0.5)
    )
    
    signal = synth_highpass.generate_note(frequency=220, duration=2.0)
    
    exporter = Exporter()
    exporter.export(signal, 'filter_highpass.wav')
    print("✓ Created highpass filter example")


def example_3_bandpass_filter():
    """Demonstrate bandpass filter effect."""
    print("\n" + "=" * 60)
    print("Example 3: Bandpass Filter")
    print("=" * 60)
    
    synth_bandpass = Synth(
        waveform='saw',
        filter=Filter.bandpass(center=800, bandwidth=400)
    )
    
    signal = synth_bandpass.generate_note(frequency=220, duration=2.0)
    
    exporter = Exporter()
    exporter.export(signal, 'filter_bandpass.wav')
    print("✓ Created bandpass filter example")


def example_4_adsr_envelope():
    """Demonstrate ADSR envelope shaping."""
    print("\n" + "=" * 60)
    print("Example 4: ADSR Envelope")
    print("=" * 60)
    
    exporter = Exporter()
    
    # Short plucky sound
    print("  Creating plucky sound (fast decay)...")
    synth1 = Synth(
        waveform='square',
        envelope=ADSR(attack=0.01, decay=0.2, sustain=0.1, release=0.1)
    )
    signal1 = synth1.generate_note(frequency=440, duration=1.0)
    exporter.export(signal1, 'envelope_pluck.wav')
    
    # Pad sound
    print("  Creating pad sound (slow attack)...")
    synth2 = Synth(
        waveform='sine',
        envelope=ADSR(attack=0.5, decay=0.3, sustain=0.7, release=1.0)
    )
    signal2 = synth2.generate_note(frequency=440, duration=2.0)
    exporter.export(signal2, 'envelope_pad.wav')
    
    # Organ sound
    print("  Creating organ sound (no decay)...")
    synth3 = Synth(
        waveform='sine',
        envelope=ADSR(attack=0.01, decay=0.0, sustain=1.0, release=0.1)
    )
    signal3 = synth3.generate_note(frequency=440, duration=1.0)
    exporter.export(signal3, 'envelope_organ.wav')
    
    print("✓ Created 3 envelope examples")


def example_5_filter_sweep():
    """Create a filter sweep effect."""
    print("\n" + "=" * 60)
    print("Example 5: Filter Sweep")
    print("=" * 60)
    
    synth = Synth(waveform='saw')
    base_signal = synth.generate_note(frequency=110, duration=4.0)
    
    # Manual filter sweep (simplified)
    sample_rate = 44100
    chunk_size = len(base_signal) // 10
    filtered_parts = []
    
    for i in range(10):
        cutoff = 200 + (i * 200)  # Sweep from 200Hz to 2000Hz
        print(f"  Filtering at {cutoff}Hz...")
        synth_filtered = Synth(
            waveform='saw',
            filter=Filter.lowpass(cutoff=cutoff, resonance=0.7)
        )
        start = i * chunk_size
        end = start + chunk_size
        chunk = base_signal[start:end]
        
        # Re-synthesize just this chunk with the filter
        freq = 110
        t = np.arange(len(chunk)) / sample_rate
        chunk_synth = np.sin(2 * np.pi * freq * t)
        
        # Apply filter (simplified - just use the synth's filter)
        filtered_chunk = synth_filtered.generate_note(frequency=110, duration=len(chunk)/sample_rate)
        filtered_parts.append(filtered_chunk[:len(chunk)])
    
    signal = np.concatenate(filtered_parts)
    
    exporter = Exporter()
    exporter.export(signal, 'filter_sweep.wav')
    print("✓ Created filter sweep example")


def main():
    print("\n" + "=" * 60)
    print("FILTERS AND EFFECTS EXAMPLES")
    print("=" * 60)
    print("\nLearn how to shape your sound with filters and envelopes!")
    
    example_1_lowpass_filter()
    example_2_highpass_filter()
    example_3_bandpass_filter()
    example_4_adsr_envelope()
    example_5_filter_sweep()
    
    print("\n" + "=" * 60)
    print("✓ All examples complete! Check ~/Music/ for the files.")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
