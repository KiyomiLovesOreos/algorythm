from algorythm.synth import Synth, Oscillator, ADSR, Filter
from algorythm.sequence import Rhythm, Motif, Scale
from algorythm.structure import Composition, Track, VolumeControl
import numpy as np

# 1. Initialize Composition
comp = Composition(tempo=120) # Standard tempo

# 2. Define drum sounds using basic synths
# Kick Drum: Low frequency sine wave with a fast decay
kick_synth = Synth(
    waveform='sine',
    envelope=ADSR(attack=0.001, decay=0.2, sustain=0.0, release=0.1),
    filter=Filter.lowpass(cutoff=100, resonance=0.8)
)

# Snare Drum: Noise with a short decay and a mid-range filter
snare_synth = Synth(
    waveform='noise',
    envelope=ADSR(attack=0.001, decay=0.2, sustain=0.0, release=0.1),
    filter=Filter.bandpass(cutoff=2000, resonance=0.5)
)

# Hi-Hat: High frequency noise with a very short decay
hihat_synth = Synth(
    waveform='noise',
    envelope=ADSR(attack=0.001, decay=0.05, sustain=0.0, release=0.05),
    filter=Filter.highpass(cutoff=7000, resonance=0.1)
)

# 3. Define rhythmic patterns
# Kick pattern: every beat
kick_pattern = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0] # 4/4 rhythm, 16th notes
kick_motif = Motif.from_intervals([0 for _ in kick_pattern], scale=Scale.chromatic('C', 4), durations=[0.25 if x == 1 else 0.0 for x in kick_pattern])

# Snare pattern: on beats 2 and 4
snare_pattern = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0] # 4/4 rhythm, 16th notes
snare_motif = Motif.from_intervals([0 for _ in snare_pattern], scale=Scale.chromatic('C', 4), durations=[0.25 if x == 1 else 0.0 for x in snare_pattern])

# Hi-Hat pattern: eighth notes
hihat_pattern = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] # 4/4 rhythm, 16th notes
hihat_motif = Motif.from_intervals([0 for _ in hihat_pattern], scale=Scale.chromatic('C', 4), durations=[0.125 if x == 1 else 0.0 for x in hihat_pattern])

# 4. Add drum tracks to the composition
comp.add_track('Kick', kick_synth) \
    .repeat_motif(kick_motif, bars=8) \
    .add_fx(VolumeControl(volume=1.2)) # Boost kick volume

comp.add_track('Snare', snare_synth) \
    .repeat_motif(snare_motif, bars=8) \
    .add_fx(VolumeControl(volume=1.0))

comp.add_track('Hi-Hat', hihat_synth) \
    .repeat_motif(hihat_motif, bars=8) \
    .add_fx(VolumeControl(volume=0.8))

# 5. Render the composition
print("Rendering drum beat...")
audio_output = comp.render(file_path='test_example_2_drums.wav', quality='high')
print(f"Rendered drum beat to test_example_2_drums.wav with {len(audio_output)} samples.")
print(f"Duration: {len(audio_output) / comp.sample_rate:.2f} seconds.")
