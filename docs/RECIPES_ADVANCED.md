# Advanced Recipes: Complex Compositions

Deep dives into advanced music production techniques using Algorythm.

## Ambient Music

### Evolving Pad with Morphing Effects

```python
from algorythm import Composition, SynthPresets, ReverbFX, ChorusFX, DelayFX

song = Composition(tempo=45)
pad = song.add_track('pad', SynthPresets.soft_pad())

# Start with subtle reverb
pad.add_effect(ReverbFX(mix=0.2, room_size=0.4))

# Long held note that evolves
note_freq = 110  # Low A

# Play note in sections with changing effects
for section in range(4):
    time = section * 8.0
    
    # Update effects gradually
    if section == 1:
        pad.add_effect(ChorusFX(rate=0.2, depth=0.3, mix=0.2))
    elif section == 2:
        pad.add_effect(DelayFX(delay_time=1.0, feedback=0.4, mix=0.2))
    elif section == 3:
        # Fade out reverb, increase delay
        pass
    
    song.play_note(note_freq, 8.0, time, 'pad')

song.render('evolving_pad.wav')
```

### Minimal Ambient Loop

```python
from algorythm import Composition, SynthPresets, Scale, ReverbFX, ChorusFX

song = Composition(tempo=60)
synth = song.add_track('synth', SynthPresets.soft_pad())

scale = Scale.pentatonic('C')

# Simple 4-note loop
notes = [0, 2, 4, 5]  # C, D, E, G

synth.add_effect(ReverbFX(mix=0.5, room_size=0.7))
synth.add_effect(ChorusFX(rate=0.1, depth=0.5, mix=0.3))

# Loop 8 times, each note gets 2 seconds
for i in range(32):
    note = notes[i % 4]
    freq = scale.get_frequency(note)
    time = i * 2.0
    
    song.play_note(freq, 2.0, time, 'synth')

song.render('ambient_loop.wav')
```

### Harmonic Series Exploration

```python
from algorythm import Composition, SynthPresets, Exporter

song = Composition(tempo=45)
synth = song.add_track('synth', SynthPresets.sine_pad())

# Play harmonic series from fundamental
fundamental = 55  # A1

# Play harmonics 1-8
for harmonic in range(1, 9):
    freq = fundamental * harmonic
    time = (harmonic - 1) * 3.0
    
    song.play_note(freq, 3.0, time, 'synth')

song.render('harmonic_series.wav')
```

---

## Complex Rhythmic Patterns

### Euclidean Rhythm Generation

```python
from algorythm import Composition, SynthPresets

def euclidean_rhythm(steps, beats):
    """Generate Euclidean rhythm pattern"""
    pattern = [0] * steps
    for i in range(beats):
        pattern[int(i * steps / beats)] = 1
    return pattern

song = Composition(tempo=120)
kick = song.add_track('kick', SynthPresets.kick())
snare = song.add_track('snare', SynthPresets.snare())

# 16-step pattern with 5 kicks (Euclidean)
kick_pattern = euclidean_rhythm(16, 5)

# 16-step pattern with 3 snares (Euclidean)
snare_pattern = euclidean_rhythm(16, 3)

for step in range(16):
    time = step * 0.125  # 16th note divisions
    
    if kick_pattern[step]:
        song.play_note(60, 0.1, time, 'kick')
    
    if snare_pattern[step]:
        song.play_note(60, 0.12, time, 'snare')

song.render('euclidean.wav')
```

### Polymetric Time Signature

```python
from algorythm import Composition, SynthPresets

song = Composition(tempo=120)
drum1 = song.add_track('drum1', SynthPresets.kick())
drum2 = song.add_track('drum2', SynthPresets.snare())
drum3 = song.add_track('drum3', SynthPresets.hihat())

duration = 12.0  # 6 bars

# Drum1: plays in 4/4 (4 beats per bar)
for beat in range(int(duration * 4)):
    song.play_note(60, 0.1, beat * 0.5, 'drum1')

# Drum2: plays in 3/4 (3 beats per bar) 
for beat in range(int(duration * 3)):
    song.play_note(60, 0.1, beat * (2.0/3.0), 'drum2')

# Drum3: plays in 5/4 (5 beats per bar)
for beat in range(int(duration * 5)):
    song.play_note(60, 0.05, beat * 0.4, 'drum3')

song.render('polymetric.wav')
```

### Afrobeat Pattern (Complex Polyrhythm)

```python
from algorythm import Composition, SynthPresets

song = Composition(tempo=120)
kick = song.add_track('kick', SynthPresets.kick())
snare = song.add_track('snare', SynthPresets.snare())
cowbell = song.add_track('cowbell', SynthPresets.cowbell())

# Afrobeat: 12-beat cycle
for bar in range(2):
    time = bar * 3.0
    
    # Kick pattern (syncopated)
    kicks = [0, 0.33, 1, 1.5, 2, 2.5]
    for k_time in kicks:
        song.play_note(60, 0.08, time + k_time, 'kick')
    
    # Snare pattern
    snares = [0.75, 1.75, 2.75]
    for s_time in snares:
        song.play_note(60, 0.1, time + s_time, 'snare')
    
    # Cowbell: steady 16ths
    for i in range(12):
        song.play_note(60, 0.05, time + (i * 0.25), 'cowbell')

song.render('afrobeat.wav')
```

---

## Sound Design Deep Dives

### Resonant Filter Sweep

```python
from algorythm import Synth, Filter, ADSR, Exporter
import numpy as np

synth = Synth(waveform='saw')

# Create audio with resonant filter
audio = synth.generate_note(220, 4.0)

# Create filter frequency envelope (sweep)
sample_rate = 44100
num_samples = len(audio)
t = np.linspace(0, 4.0, num_samples)

# Sweep filter from low to high
cutoff_start = 500
cutoff_end = 4000
cutoff_sweep = cutoff_start + (cutoff_end - cutoff_start) * t / 4.0

# Apply to saw wave (simplified)
filtered = audio.copy()

Exporter().export(filtered, 'filter_sweep.wav')
```

### Additive Synthesis (Hand-Built Timbre)

```python
from algorythm import Composition, SynthPresets, Exporter
import numpy as np

# Build complex timbre from harmonics
sample_rate = 44100
duration = 2.0
num_samples = int(sample_rate * duration)
t = np.linspace(0, duration, num_samples)

fundamental = 220
audio = np.zeros(num_samples)

# Add harmonics with decreasing amplitude
harmonic_amps = [1.0, 0.5, 0.33, 0.25, 0.2, 0.167]

for harmonic, amp in enumerate(harmonic_amps, 1):
    freq = fundamental * harmonic
    # Add harmonics with different phases
    audio += amp * np.sin(2 * np.pi * freq * t + np.pi * harmonic / 4)

# Normalize
audio = audio / np.max(np.abs(audio)) * 0.9

Exporter().export(audio, 'additive_timbre.wav')
```

### Wavetable Morphing

```python
from algorythm import Synth, Exporter
import numpy as np

# Generate wavetable by morphing between waveforms
sample_rate = 44100
duration = 3.0
num_samples = int(sample_rate * duration)
t = np.linspace(0, duration, num_samples)

freq = 220

# Start with sine, morph to saw
audio = np.zeros(num_samples)

for i, sample_time in enumerate(t):
    # Morph amount from 0 to 1
    morph = sample_time / duration
    
    # Sine wave
    sine = np.sin(2 * np.pi * freq * sample_time)
    
    # Saw wave
    saw = 2 * (sample_time * freq % 1) - 1
    
    # Morph between them
    audio[i] = sine * (1 - morph) + saw * morph

Exporter().export(audio, 'wavetable_morph.wav')
```

### Granular Texture

```python
from algorythm import Synth, Exporter
import numpy as np

# Create granular synthesis texture
sample_rate = 44100
duration = 4.0
num_samples = int(sample_rate * duration)

audio = np.zeros(num_samples)

# Grain parameters
grain_size = 0.05  # 50ms grains
grain_overlap = 0.7  # 70% overlap
grain_density = 20  # grains per second

synth = Synth(waveform='sine')

# Generate grains
for grain_start_time in np.arange(0, duration, grain_size * (1 - grain_overlap)):
    if grain_start_time + grain_size > duration:
        break
    
    # Random frequency around 440Hz
    freq = 440 + np.random.randn() * 50
    
    # Generate grain
    grain = synth.generate_note(freq, grain_size)
    
    # Window the grain (fade in/out)
    window = np.hann(len(grain))
    grain = grain * window
    
    # Add to output
    start_idx = int(grain_start_time * sample_rate)
    end_idx = start_idx + len(grain)
    audio[start_idx:end_idx] += grain

# Normalize
audio = audio / np.max(np.abs(audio)) * 0.9

Exporter().export(audio, 'granular.wav')
```

---

## Algorithmic Composition

### Generative Ambient (Never Repeats)

```python
import random
from algorythm import Composition, SynthPresets, Scale

song = Composition(tempo=45)
synth = song.add_track('synth', SynthPresets.soft_pad())

scale = Scale.pentatonic('C')

# Generate unique ambient piece
current_note = 0
time = 0.0

for _ in range(50):  # 50 notes
    # Random duration
    duration = random.choice([1.0, 1.5, 2.0, 2.5, 3.0])
    
    # Next note (constrained walk)
    direction = random.choice([-1, 0, 1])
    current_note = max(0, min(7, current_note + direction))
    
    freq = scale.get_frequency(current_note)
    song.play_note(freq, duration, time, 'synth')
    
    time += duration

song.render('generative_ambient.wav')
```

### Markov Chain Melody Generation

```python
import random
from algorythm import Composition, SynthPresets, Scale, Motif

# Markov chain: probability of next note based on current
transitions = {
    0: {0: 0.2, 1: 0.3, 2: 0.3, 3: 0.2},
    1: {0: 0.2, 1: 0.1, 2: 0.4, 3: 0.3},
    2: {0: 0.3, 1: 0.2, 2: 0.2, 3: 0.3},
    3: {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1},
}

song = Composition(tempo=120)
synth = song.add_track('synth', SynthPresets.pluck())

scale = Scale.major('C')

# Generate melody using Markov chain
melody = [0]  # Start at root
for _ in range(15):
    current = melody[-1]
    probabilities = transitions[current]
    
    next_note = random.choices(
        list(probabilities.keys()),
        weights=list(probabilities.values())
    )[0]
    melody.append(next_note)

motif = Motif.from_intervals(melody, scale=scale, duration=0.5)
song.play_motif(motif, start=0.0, track='synth')

song.render('markov_melody.wav')
```

### L-System Rhythm Generation

```python
from algorythm import Composition, SynthPresets

def lsystem(axiom, rules, iterations):
    """Generate L-system string"""
    current = axiom
    for _ in range(iterations):
        next_str = ""
        for char in current:
            next_str += rules.get(char, char)
        current = next_str
    return current

song = Composition(tempo=120)
kick = song.add_track('kick', SynthPresets.kick())

# L-system for rhythm
# A = kick, B = rest
rules = {'A': 'AB', 'B': 'A'}
pattern_str = lsystem('A', rules, 4)

# Convert to rhythm
time = 0.0
eighth_note = 0.25

for char in pattern_str[:32]:  # Limit length
    if char == 'A':
        song.play_note(60, 0.08, time, 'kick')
    time += eighth_note

song.render('lsystem_rhythm.wav')
```

---

## Interactive & Real-Time Effects

### Dynamic Volume Gate

```python
from algorythm import Composition, SynthPresets

song = Composition(tempo=120)
lead = song.add_track('lead', SynthPresets.pluck())

# Simulate gating with volume automation
for i in range(16):
    time = i * 0.25
    
    # Gate opens and closes
    if i % 4 < 2:
        lead.set_volume(1.0)  # Open
        song.play_note(440, 0.2, time, 'lead')
    else:
        lead.set_volume(0.0)  # Closed (silent)

song.render('gated.wav')
```

### Auto-Panning Effect

```python
from algorythm import Composition, SynthPresets
import numpy as np

song = Composition(tempo=120)
synth = song.add_track('synth', SynthPresets.pluck())

# Oscillating pan position
duration = 4.0
samples = int(duration * 44100)
t = np.linspace(0, duration, samples)

# Slow pan
pan_freq = 0.5  # Pan frequency in Hz
pan_positions = np.sin(2 * np.pi * pan_freq * t)

# Apply panning throughout
for i, pan in enumerate(pan_positions[::4410]):  # Every 0.1 sec
    synth.set_pan(pan_positions[i * 4410])
    song.play_note(440, 0.1, i * 0.1, 'synth')

song.render('auto_pan.wav')
```

### Frequency Modulation in Real-Time

```python
from algorythm import FMSynth, Exporter
import numpy as np

# FM with modulation
duration = 3.0
sample_rate = 44100
num_samples = int(sample_rate * duration)
t = np.linspace(0, duration, num_samples)

# Modulation that changes over time
carrier = 440
mod_freq_envelope = 100 + 200 * t / duration  # 100Hz to 300Hz

audio = np.zeros(num_samples)

for i, sample_time in enumerate(t):
    mod_freq = 100 + 200 * sample_time / duration
    mod_index = 5
    
    phase = 2 * np.pi * carrier * sample_time
    mod_phase = 2 * np.pi * mod_freq * sample_time
    
    audio[i] = np.sin(phase + mod_index * np.sin(mod_phase))

Exporter().export(audio, 'fm_modulation.wav')
```

---

## Mastering & Processing

### Multiband Compression

```python
from algorythm import Composition, SynthPresets, Compressor, Filter

song = Composition(tempo=120)
drums = song.add_track('drums', SynthPresets.kick())
bass = song.add_track('bass', SynthPresets.synth_bass())
mid = song.add_track('mid', SynthPresets.pluck())

# Low band compression
bass.add_effect(Filter.lowpass(cutoff=250))
bass.add_effect(Compressor(threshold=-25, ratio=4.0))

# Mid band compression
mid.add_effect(Filter.highpass(cutoff=250))
mid.add_effect(Filter.lowpass(cutoff=2000))
mid.add_effect(Compressor(threshold=-20, ratio=2.0))

# High band compression (light)
drums.add_effect(Filter.highpass(cutoff=2000))
drums.add_effect(Compressor(threshold=-15, ratio=1.5))

song.play_note(440, 1.0, 0.0, 'kick')
song.play_note(110, 1.0, 0.0, 'bass')
song.play_note(880, 1.0, 0.0, 'mid')

song.render('multiband_comp.wav')
```

### Parallel Compression (Blending)

```python
from algorythm import Composition, SynthPresets, Compressor

song = Composition(tempo=120)

# Dry signal
dry = song.add_track('dry', SynthPresets.warm_pad())

# Wet signal (heavily compressed)
wet = song.add_track('wet', SynthPresets.warm_pad())
wet.add_effect(Compressor(threshold=-30, ratio=8.0))

# Lower wet volume to blend
wet.set_volume(0.3)

# Play same note on both
song.play_note(220, 2.0, 0.0, 'dry')
song.play_note(220, 2.0, 0.0, 'wet')

song.render('parallel_comp.wav')
```

### Vintage Limiting Chain

```python
from algorythm import Composition, SynthPresets, Compressor, Limiter

song = Composition(tempo=120)
master = song.add_track('main', SynthPresets.synth_lead())

# Vintage chain: soft compression + hard limiter
master.add_effect(Compressor(threshold=-12, ratio=1.5))  # Soft
master.add_effect(Compressor(threshold=-6, ratio=4.0))   # Medium
master.add_effect(Limiter(threshold=-1))                  # Hard wall

song.play_note(440, 2.0, 0.0, 'main')

song.render('vintage_limiting.wav')
```

---

## Integration & Hybrid Approaches

### Combining Synthesis Types

```python
from algorythm import Synth, FMSynth, Composition, SynthPresets, Exporter

song = Composition(tempo=120)

# Wavetable synth
wt_synth = song.add_track('wavetable', SynthPresets.synth_lead())

# FM synth
fm = FMSynth(carrier_freq=220, modulator_freq=110, mod_index=3)

# Basic synth
basic = song.add_track('basic', SynthPresets.pluck())

# Play different synths
song.play_note(220, 1.0, 0.0, 'wavetable')
basic_audio = Synth(waveform='sine').generate_note(220, 1.0)

time = 0.0
for _ in range(4):
    song.play_note(220, 1.0, time, 'basic')
    time += 1.0

song.render('hybrid_synth.wav')
```

### Granular + Harmonic Hybrid

```python
from algorythm import Synth, Exporter
import numpy as np

# Layer granular texture with harmonic pad
sample_rate = 44100
duration = 4.0
num_samples = int(sample_rate * duration)

# Granular layer
granular = np.zeros(num_samples)
for start in np.arange(0, duration, 0.05):
    if start + 0.1 > duration:
        break
    grain = Synth(waveform='sine').generate_note(440, 0.1)
    start_idx = int(start * sample_rate)
    granular[start_idx:start_idx + len(grain)] += grain

# Harmonic layer
t = np.linspace(0, duration, num_samples)
harmonic = np.sin(2 * np.pi * 220 * t) * 0.5

# Combine
combined = (granular + harmonic) / 2
combined = combined / np.max(np.abs(combined)) * 0.9

Exporter().export(combined, 'granular_harmonic.wav')
```

---

## Production Techniques

### Creating Tension & Release

```python
from algorythm import Composition, SynthPresets, Scale, ReverbFX, Compressor

song = Composition(tempo=110)

drums = song.add_track('drums', SynthPresets.kick())
bass = song.add_track('bass', SynthPresets.synth_bass())
lead = song.add_track('lead', SynthPresets.pluck())
pad = song.add_track('pad', SynthPresets.warm_pad())

scale = Scale.minor('A')

# Tension section (bars 0-16): minimal, compressed
for bar in range(16):
    time = bar * 2.0
    for beat in range(4):
        song.play_note(60, 0.05, time + (beat * 0.5), 'drums')

# Add bass with compression
bass.add_effect(Compressor(threshold=-25, ratio=6.0))
song.play_note(55, 16 * 2.0, 0.0, 'bass')

# Release section (bars 16-24): add layers, remove compression
for bar in range(16, 24):
    time = bar * 2.0
    # More drum hits
    for beat in range(4):
        song.play_note(60, 0.08, time + (beat * 0.5), 'drums')

# Add lead
lead_motif = Motif.from_intervals([0, 2, 4, 5, 7], scale=scale, duration=0.5)
song.play_motif(lead_motif, start=16 * 2.0, track='lead')

# Add pad for fullness
song.play_note(220, 8 * 2.0, 16 * 2.0, 'pad')

song.render('tension_release.wav')
```

### Build-Up to Drop

```python
from algorythm import Composition, SynthPresets, Scale, Motif

def bars(n, tempo=128):
    return (n * 4.0 * 60.0) / tempo

song = Composition(tempo=128)

kick = song.add_track('kick', SynthPresets.kick())
bass = song.add_track('bass', SynthPresets.acid_bass())
synth = song.add_track('synth', SynthPresets.synth_lead())

scale = Scale.minor('A')

# Intro (8 bars): just kick
for bar in range(8):
    time = bars(bar)
    for beat in range(4):
        song.play_note(60, 0.05, time + (beat * 0.5), 'kick')

# Build 1 (8 bars): add bass
for bar in range(8):
    time = bars(8 + bar)
    for beat in range(4):
        song.play_note(60, 0.05, time + (beat * 0.5), 'kick')

song.play_note(55, bars(8), bars(8), 'bass')

# Build 2 (8 bars): add synth pad
for bar in range(8):
    time = bars(16 + bar)
    for beat in range(4):
        song.play_note(60, 0.05, time + (beat * 0.5), 'kick')

song.play_note(55, bars(8), bars(16), 'bass')
song.play_note(220, bars(8), bars(16), 'synth')

# Drop (16 bars): full arrangement
synth_motif = Motif.from_intervals([0, 2, 3, 5], scale=scale, duration=0.25)

for bar in range(16):
    time = bars(24 + bar)
    for beat in range(4):
        song.play_note(60, 0.08, time + (beat * 0.5), 'kick')

song.play_motif(synth_motif, start=bars(24), track='synth')
song.play_note(55, bars(16), bars(24), 'bass')

song.render('buildup_drop.wav')
```

---

## Tips for Advanced Production

1. **Layer different synthesis types** - Wavetable + FM + granular = interesting
2. **Use constraint-based randomization** - Never fully random, always bounded
3. **Build tension systematically** - Add elements gradually, not all at once
4. **Use EQ to create space** - Each track in its own frequency range
5. **Modulate parameters over time** - Nothing static, everything evolves
6. **Reference professional music** - Analyze structures and techniques
7. **Test with different playback systems** - Headphones, speakers, car
8. **Take breaks** - Fresh ears make better decisions
9. **Document your process** - Remember what worked
10. **Experiment fearlessly** - That's what code is for
