# Cookbook: Copy & Paste Recipes

Ready-to-use code snippets for common tasks. Copy, paste, adapt.

## Basic Sound Generation

### Simple Beep

```python
from algorythm import Synth, Exporter

synth = Synth(waveform='sine')
audio = synth.generate_note(440, 0.5)
Exporter().export(audio, 'beep.wav')
```

### Tone Sweep (Ascending)

```python
import numpy as np
from algorythm import Exporter

sample_rate = 44100
duration = 2.0
num_samples = int(sample_rate * duration)
t = np.linspace(0, duration, num_samples)

# Sweep from 200 Hz to 800 Hz
freq_start = 200
freq_end = 800
frequencies = np.linspace(freq_start, freq_end, num_samples)

# Generate phase and sine wave
phase = 2 * np.pi * np.cumsum(frequencies) / sample_rate
audio = np.sin(phase[:num_samples])

Exporter().export(audio, 'sweep.wav')
```

### White Noise

```python
import numpy as np
from algorythm import Exporter

sample_rate = 44100
duration = 2.0
samples = int(sample_rate * duration)
audio = np.random.uniform(-0.5, 0.5, samples)

Exporter().export(audio, 'noise.wav')
```

## Melodies

### Simple Scale Exercise

```python
from algorythm import Synth, Scale, Motif, Exporter
import numpy as np

scale = Scale.major('C')
melody = Motif.from_intervals([0, 1, 2, 3, 4, 5, 6, 7], scale=scale, duration=0.5)

synth = Synth(waveform='sine')
notes = []
for note in melody.notes:
    audio = synth.generate_note(note['frequency'], note['duration'])
    notes.append(audio)

full_audio = np.concatenate(notes)
Exporter().export(full_audio, 'scale.wav')
```

### Happy Melody

```python
from algorythm import Composition, SynthPresets, Scale, Motif

song = Composition(tempo=120)
song.add_track('melody', SynthPresets.pluck())

scale = Scale.major('C')
melody = Motif.from_intervals(
    [0, 2, 4, 5, 7, 5, 4, 2, 0],
    scale=scale,
    duration=0.5
)

song.play_motif(melody, start=0.0, track='melody')
song.render('happy.wav')
```

### Sad Melody

```python
from algorythm import Composition, SynthPresets, Scale, Motif

song = Composition(tempo=90)  # Slower tempo
song.add_track('melody', SynthPresets.soft_pad())

scale = Scale.minor('A')
melody = Motif.from_intervals(
    [0, -2, -4, -5, -7, -5, -4, -2, 0],
    scale=scale,
    duration=0.75  # Longer notes
)

song.play_motif(melody, start=0.0, track='melody')
song.render('sad.wav')
```

## Drums

### Simple 4/4 Beat

```python
from algorythm import Composition, SynthPresets

song = Composition(tempo=120)
song.add_track('kick', SynthPresets.kick())
song.add_track('snare', SynthPresets.snare())

# One bar at a time
for bar in range(4):
    time = bar * 2.0  # 2 seconds per bar at 120 BPM
    
    # Kick on 1 and 3
    song.play_note(60, 0.1, time + 0.0, 'kick')
    song.play_note(60, 0.1, time + 1.0, 'kick')
    
    # Snare on 2 and 4
    song.play_note(60, 0.1, time + 0.5, 'snare')
    song.play_note(60, 0.1, time + 1.5, 'snare')

song.render('beat.wav')
```

### House Beat (4 to the Floor)

```python
from algorythm import Composition, SynthPresets

song = Composition(tempo=128)
song.add_track('kick', SynthPresets.kick())
song.add_track('clap', SynthPresets.clap())
song.add_track('hihat', SynthPresets.hihat())

# 4 bars
for bar in range(4):
    time = bar * 2.0
    
    # Kick on every beat (4/4)
    for beat in range(4):
        song.play_note(60, 0.08, time + (beat * 0.5), 'kick')
    
    # Clap on 2 and 4
    song.play_note(60, 0.12, time + 0.5, 'clap')
    song.play_note(60, 0.12, time + 1.5, 'clap')
    
    # Hi-hats on offbeats
    for i in [1, 3, 5, 7]:
        song.play_note(60, 0.05, time + (i * 0.25), 'hihat')

song.render('house.wav')
```

## Chords and Harmony

### Simple Chord Pad

```python
from algorythm import Composition, Chord, SynthPresets

song = Composition(tempo=100)
song.add_track('pad', SynthPresets.warm_pad())

chords = [
    Chord.major('C'),
    Chord.major('F'),
    Chord.major('G'),
    Chord.major('C')
]

for i, chord in enumerate(chords):
    time = i * 2.0
    freqs = chord.get_frequencies()
    for freq in freqs:
        song.play_note(freq, 2.0, time, 'pad')

song.render('chords.wav')
```

### Arpeggiated Chord

```python
from algorythm import Composition, Chord, SynthPresets

song = Composition(tempo=120)
song.add_track('arp', SynthPresets.pluck())

chord = Chord.major('C')
freqs = chord.get_frequencies()

time = 0.0
note_duration = 0.25

for freq in freqs:
    song.play_note(freq, note_duration, time, 'arp')
    time += note_duration

song.render('arpeggio.wav')
```

## Full Compositions

### 8-Bar Loop

```python
from algorythm import Composition, SynthPresets, Scale, Motif

def bars(n, tempo=120):
    return (n * 4.0 * 60.0) / tempo

song = Composition(tempo=120)

# Tracks
kick = song.add_track('kick', SynthPresets.kick())
bass = song.add_track('bass', SynthPresets.synth_bass())
melody = song.add_track('melody', SynthPresets.pluck())

scale = Scale.minor('A')
bass_pattern = Motif.from_intervals([0, 0, 4, 4], scale=scale, octave=2, duration=0.5)
melody_pattern = Motif.from_intervals([0, 2, 3, 5, 7], scale=scale, duration=0.5)

# Drums
for bar in range(8):
    time = bar * 2.0
    song.play_note(60, 0.1, time + 0.0, 'kick')
    song.play_note(60, 0.1, time + 1.0, 'kick')

# Bass and Melody (loop twice)
for loop in range(2):
    start = bars(loop * 4)
    song.play_motif(bass_pattern, start=start, track='bass')
    song.play_motif(melody_pattern, start=start, track='melody')

song.render('loop.wav')
```

### Minute-Long Track

```python
from algorythm import Composition, SynthPresets, Scale, Motif, ReverbFX, Compressor

def bars(n, tempo=120):
    return (n * 4.0 * 60.0) / tempo

song = Composition(tempo=120)

# Add tracks
drums = song.add_track('drums', SynthPresets.kick())
bass = song.add_track('bass', SynthPresets.synth_bass())
melody = song.add_track('melody', SynthPresets.pluck())
pad = song.add_track('pad', SynthPresets.warm_pad())

# Effects
melody.add_effect(ReverbFX(mix=0.2))
pad.add_effect(ReverbFX(mix=0.4))

# Scale
scale = Scale.minor('A')

# Patterns
bass_pattern = Motif.from_intervals([0, 0, 4, 4], scale=scale, octave=2, duration=0.5)
melody_pattern = Motif.from_intervals([0, 2, 3, 5, 7], scale=scale, duration=0.5)

# Intro (0-8 bars) - just drums
for bar in range(8):
    time = bar * 2.0
    song.play_note(60, 0.1, time + 0.0, 'drums')
    song.play_note(60, 0.1, time + 1.0, 'drums')

# Verse (8-24 bars) - add bass
for bar in range(16):
    time = bars(8) + (bar * 2.0)
    song.play_note(60, 0.1, time + 0.0, 'drums')
    song.play_note(60, 0.1, time + 1.0, 'drums')

song.play_motif(bass_pattern, start=bars(8), track='bass')

# Chorus (24-32 bars) - add melody and pad
for bar in range(8):
    time = bars(24) + (bar * 2.0)
    song.play_note(60, 0.1, time + 0.0, 'drums')
    song.play_note(60, 0.1, time + 1.0, 'drums')

song.play_motif(melody_pattern, start=bars(24), track='melody')
song.play_motif(bass_pattern, start=bars(24), track='bass')

pad_notes = Motif.from_intervals([0, 2, 4], scale=scale, duration=8.0)
song.play_motif(pad_notes, start=bars(24), track='pad')

# Master
song.add_master_effect(Compressor(threshold=-10, ratio=2.0))

song.render('track.wav')
```

## Effects Recipes

### Vocal-Like Effect

```python
from algorythm import Composition, SynthPresets, Scale, Motif
from algorythm import ReverbFX, DelayFX, Compressor

song = Composition(tempo=120)
track = song.add_track('voice', SynthPresets.pluck())

# Vocal-like effects
track.add_effect(Compressor(threshold=-15, ratio=3.0))
track.add_effect(DelayFX(delay_time=0.375, feedback=0.2, mix=0.1))
track.add_effect(ReverbFX(mix=0.3, room_size=0.4))

scale = Scale.major('C')
melody = Motif.from_intervals([0, 2, 4, 5, 7, 5, 4, 2], scale=scale, duration=0.5)

song.play_motif(melody, start=0.0, track='voice')
song.render('vocal.wav')
```

### Lo-Fi Hip-Hop Effect

```python
from algorythm import Composition, SynthPresets, Scale, Motif
from algorythm import BitCrusherFX, DistortionFX, ReverbFX

song = Composition(tempo=85)
track = song.add_track('lofi', SynthPresets.warm_pad())

# Lo-fi effects
track.add_effect(BitCrusherFX(bit_depth=8, sample_rate=11025))
track.add_effect(DistortionFX(drive=2.0, tone=0.4, mix=0.2))
track.add_effect(ReverbFX(mix=0.4, room_size=0.3, damping=0.8))

scale = Scale.pentatonic('C')
melody = Motif.from_intervals([0, 2, 4, 5], scale=scale, duration=1.0)

song.play_motif(melody, start=0.0, track='lofi')
song.render('lofi.wav')
```

### Ambient Pad

```python
from algorythm import Composition, SynthPresets
from algorythm import ChorusFX, DelayFX, ReverbFX

song = Composition(tempo=60)
track = song.add_track('ambient', SynthPresets.soft_pad())

# Ambient effects
track.add_effect(ChorusFX(rate=0.2, depth=0.5, mix=0.5))
track.add_effect(DelayFX(delay_time=1.0, feedback=0.5, mix=0.3))
track.add_effect(ReverbFX(mix=0.6, room_size=0.9, damping=0.3))

# Long pad note
song.play_note(220, 16.0, start=0.0, track='ambient')

song.render('ambient.wav')
```

## Visualization

### Basic Frequency Scope

```python
from algorythm import visualize_audio_file, FrequencyScopeVisualizer

# Assumes you have 'audio.wav' from previous recipes

viz = FrequencyScopeVisualizer(
    sample_rate=44100,
    num_bars=64,
    color=(0, 255, 100)
)

visualize_audio_file(
    'audio.wav',
    'audio.mp4',
    viz,
    video_width=1920,
    video_height=1080,
    video_fps=30
)
```

### Waveform Visualization

```python
from algorythm import visualize_audio_file, WaveformVisualizer

viz = WaveformVisualizer(
    sample_rate=44100,
    color=(100, 200, 255),
    line_width=2
)

visualize_audio_file(
    'audio.wav',
    'waveform.mp4',
    viz,
    video_width=1920,
    video_height=1080,
    video_fps=30
)
```

### Spectrogram

```python
from algorythm import visualize_audio_file, SpectrogramVisualizer

viz = SpectrogramVisualizer(
    sample_rate=44100,
    fft_size=2048,
    colormap='viridis'
)

visualize_audio_file(
    'audio.wav',
    'spectrogram.mp4',
    viz
)
```

## Data and Randomization

### Random Melody

```python
import random
from algorythm import Composition, SynthPresets, Scale, Motif

song = Composition(tempo=120)
song.add_track('random', SynthPresets.pluck())

scale = Scale.pentatonic('C')

# Generate random melody
random_degrees = [random.randint(0, 7) for _ in range(16)]

melody = Motif.from_intervals(random_degrees, scale=scale, duration=0.25)
song.play_motif(melody, start=0.0, track='random')
song.render('random.wav')
```

### Constrained Random Walk

```python
import random
from algorythm import Composition, SynthPresets, Scale, Motif

song = Composition(tempo=120)
song.add_track('walk', SynthPresets.pluck())

scale = Scale.major('C')

# Random walk (each note constrains next)
melody = [0]  # Start at root
for _ in range(15):
    # Next note is within 2 steps
    next_note = melody[-1] + random.randint(-2, 2)
    next_note = max(0, min(7, next_note))  # Keep in range
    melody.append(next_note)

motif = Motif.from_intervals(melody, scale=scale, duration=0.25)
song.play_motif(motif, start=0.0, track='walk')
song.render('walk.wav')
```

## Export Options

### WAV (Lossless)

```python
song.render('output.wav')  # Best quality, largest file
```

### MP3 (Compressed)

```python
song.render('output.mp3')  # Medium quality, smaller file
```

### FLAC (Lossless Compressed)

```python
song.render('output.flac')  # Lossless, smaller than WAV
```

## Quick Experiments

### Microtonal Scale

```python
from algorythm import Composition, SynthPresets, Tuning, Scale

song = Composition(tempo=120)
song.add_track('microtonal', SynthPresets.pluck())

# 24-tone equal temperament (quarter tones)
tuning = Tuning('24-TET')
scale = Scale.major('C', tuning=tuning)

# Play scale degree every 1 instead of 2 (half steps)
from algorythm import Motif
melody = Motif.from_intervals(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    scale=scale,
    duration=0.25
)

song.play_motif(melody, start=0.0, track='microtonal')
song.render('microtonal.wav')
```

### FM Synthesis Exploration

```python
from algorythm import FMSynth, Exporter

# Metallic sound
fm = FMSynth(
    carrier_freq=200,
    modulator_freq=600,
    mod_index=10
)
audio = fm.generate(duration=2.0)
Exporter().export(audio, 'metallic.wav')

# Bell-like sound
fm = FMSynth(
    carrier_freq=100,
    modulator_freq=150,
    mod_index=5
)
audio = fm.generate(duration=3.0)
Exporter().export(audio, 'bell.wav')
```

## Time Savers

### Quick Tempo Calculator

```python
def get_note_duration(bpm, note_type):
    """Get duration in seconds for note type at given BPM"""
    beat_duration = 60.0 / bpm
    
    note_types = {
        'whole': 4.0,
        'half': 2.0,
        'quarter': 1.0,
        'eighth': 0.5,
        'sixteenth': 0.25,
        'd_quarter': 1.5,
        'd_eighth': 0.75,
        'd_sixteenth': 0.375,
    }
    
    return beat_duration * note_types[note_type]

# Usage:
duration = get_note_duration(120, 'quarter')  # 0.5 seconds
```

#### Bars to Seconds Helper

```python
def bars_to_time(num_bars, bpm=120):
    """Convert bar count to seconds"""
    return (num_bars * 4.0 * 60.0) / bpm

# Usage:
time = bars_to_time(8, 120)  # 16.0 seconds
```

---

## Advanced Sound Design

### Pad with Velocity Control

```python
from algorythm import Composition, SynthPresets, Scale

song = Composition(tempo=60)
pad = song.add_track('pad', SynthPresets.warm_pad())

scale = Scale.major('C')

# Play same note with different volumes
volumes = [0.3, 0.5, 0.7, 0.9, 0.7, 0.5, 0.3]
for i, vol in enumerate(volumes):
    time = i * 2.0
    song.play_note(scale.get_frequency(0), 2.0, time, 'pad')
    pad.set_volume(vol)

song.render('pad_swell.wav')
```

### String Ensemble Sound

```python
from algorythm import Composition, SynthPresets, Scale, Chord

song = Composition(tempo=80)

# Three string sections for lush sound
violin = song.add_track('violin', SynthPresets.strings())
viola = song.add_track('viola', SynthPresets.cello())
cello = song.add_track('cello', SynthPresets.upright_bass())

scale = Scale.major('D')
chord = Chord.major('D')

freqs = chord.get_frequencies()

# High strings
song.play_note(freqs[-1], 4.0, 0.0, 'violin')

# Mid strings
song.play_note(freqs[len(freqs)//2], 4.0, 0.0, 'viola')

# Low strings
song.play_note(freqs[0], 4.0, 0.0, 'cello')

song.render('strings.wav')
```

### FM Bass with Modulation

```python
from algorythm import FMSynth, Exporter

# Deep bass with FM
fm = FMSynth(
    carrier_freq=55,      # Sub bass frequency
    modulator_freq=30,
    mod_index=8
)
audio = fm.generate(duration=4.0)
Exporter().export(audio, 'fm_bass.wav')
```

### Plucked String Simulation

```python
from algorythm import Synth, ADSR, Filter, Exporter

synth = Synth(
    waveform='triangle',
    envelope=ADSR(attack=0.001, decay=0.8, sustain=0.0, release=0.2),
    filter=Filter.lowpass(cutoff=3000, resonance=0.3)
)

# Simulate plucked note
audio = synth.generate_note(220, 1.0)
Exporter().export(audio, 'pluck.wav')
```

### Bell Tone (Long Decay)

```python
from algorythm import Synth, ADSR, Filter, Exporter

synth = Synth(
    waveform='sine',
    envelope=ADSR(attack=0.05, decay=3.0, sustain=0.1, release=1.0),
    filter=Filter.lowpass(cutoff=4000, resonance=0.5)
)

audio = synth.generate_note(440, 4.0)
Exporter().export(audio, 'bell.wav')
```

---

## Advanced Melodies

### Melodic Sequence with Variations

```python
from algorythm import Composition, SynthPresets, Scale, Motif

song = Composition(tempo=120)
lead = song.add_track('lead', SynthPresets.synth_lead())

scale = Scale.major('C')

# Base pattern
base = [0, 2, 4, 5, 7]

# Variation 1: transpose up
var1 = [x + 2 for x in base]

# Variation 2: reverse
var2 = list(reversed(base))

# Variation 3: rhythmic change
motif1 = Motif.from_intervals(base, scale=scale, duration=0.5)
motif2 = Motif.from_intervals(var1, scale=scale, duration=0.25)
motif3 = Motif.from_intervals(var2, scale=scale, duration=0.75)

song.play_motif(motif1, start=0.0, track='lead')
song.play_motif(motif2, start=4.0, track='lead')
song.play_motif(motif3, start=8.0, track='lead')

song.render('melody_sequence.wav')
```

### Call and Response Pattern

```python
from algorythm import Composition, SynthPresets, Scale, Motif

song = Composition(tempo=120)
synth = song.add_track('synth', SynthPresets.pluck())

scale = Scale.pentatonic('C')

# Call (question)
call = Motif.from_intervals([0, 2, 4, 5], scale=scale, duration=0.5)

# Response (answer) - inverted
response = Motif.from_intervals([5, 4, 2, 0], scale=scale, duration=0.5)

# Repeat pattern 4 times
for i in range(4):
    time = i * 4.0
    song.play_motif(call, start=time, track='synth')
    song.play_motif(response, start=time + 2.0, track='synth')

song.render('call_response.wav')
```

### Fugue-like Canon (Simple)

```python
from algorythm import Composition, SynthPresets, Scale, Motif

song = Composition(tempo=100)
voice1 = song.add_track('voice1', SynthPresets.sine_pad())
voice2 = song.add_track('voice2', SynthPresets.sine_pad())
voice3 = song.add_track('voice3', SynthPresets.sine_pad())

scale = Scale.minor('A')

# Single melody phrase
phrase = Motif.from_intervals([0, 2, 3, 5, 7, 5, 3, 2], scale=scale, duration=0.5)

# Stagger voices (canon effect)
song.play_motif(phrase, start=0.0, track='voice1')
song.play_motif(phrase, start=2.0, track='voice2')
song.play_motif(phrase, start=4.0, track='voice3')

song.render('canon.wav')
```

---

## Advanced Drum Programming

### Breakbeat Variation

```python
from algorythm import Composition, SynthPresets

song = Composition(tempo=110)
kick = song.add_track('kick', SynthPresets.kick())
snare = song.add_track('snare', SynthPresets.snare())
hihat = song.add_track('hihat', SynthPresets.hihat())

# Breakbeat pattern (syncopated)
for bar in range(4):
    time = bar * 2.0
    
    # Kick: syncopated
    song.play_note(60, 0.08, time + 0.0, 'kick')
    song.play_note(60, 0.08, time + 0.5, 'kick')
    song.play_note(60, 0.08, time + 1.0, 'kick')
    song.play_note(60, 0.08, time + 1.6, 'kick')
    
    # Snare: off-beat
    song.play_note(60, 0.1, time + 0.75, 'snare')
    song.play_note(60, 0.1, time + 1.75, 'snare')
    
    # Hi-hat: tight, fast
    for i in range(16):
        song.play_note(60, 0.03, time + (i * 0.125), 'hihat')

song.render('breakbeat.wav')
```

### Swing Shuffle Feel

```python
from algorythm import Composition, SynthPresets

song = Composition(tempo=100)
kick = song.add_track('kick', SynthPresets.kick())
snare = song.add_track('snare', SynthPresets.snare())

# Shuffle: triplet feel
triplet_unit = 1.0 / 3.0  # Triplet divisions

for bar in range(4):
    time = bar * 2.0
    
    # Swing the hi-hats on triplets
    for triplet in range(6):
        hat_time = time + (triplet * triplet_unit)
        if triplet % 3 != 1:  # Skip middle triplet
            song.play_note(60, 0.05, hat_time, 'snare')
    
    # Keep kick on beats
    song.play_note(60, 0.1, time + 0.0, 'kick')
    song.play_note(60, 0.1, time + 1.0, 'kick')

song.render('swing.wav')
```

### Polyrhythmic Drums

```python
from algorythm import Composition, SynthPresets

song = Composition(tempo=120)
kick = song.add_track('kick', SynthPresets.kick())
tom = song.add_track('tom', SynthPresets.tom())
cymbal = song.add_track('cymbal', SynthPresets.cymbal())

duration = 4.0  # 4 bars

# Kick: 4/4 pattern
for i in range(8):
    song.play_note(60, 0.1, (i * 0.5), 'kick')

# Tom: 3/4 pattern (triplet feel)
for i in range(6):
    song.play_note(60, 0.08, (i * (2.0/3.0)), 'tom')

# Cymbal: 5/4 pattern
for i in range(5):
    song.play_note(60, 0.1, (i * 0.8), 'cymbal')

song.render('polyrhythm.wav')
```

---

## Advanced Harmony & Chords

### Suspended Chords (Sus2, Sus4)

```python
from algorythm import Composition, Chord, SynthPresets

song = Composition(tempo=100)
pad = song.add_track('pad', SynthPresets.warm_pad())

# Sus2: Root, 2nd, 5th
sus2 = Chord.sus2('C')

# Sus4: Root, 4th, 5th
sus4 = Chord.sus4('C')

# Play progression
time = 0.0
for chord in [sus2, sus4, sus2, sus4]:
    freqs = chord.get_frequencies()
    for freq in freqs:
        song.play_note(freq, 2.0, time, 'pad')
    time += 2.0

song.render('suspended.wav')
```

### Chord Inversions

```python
from algorythm import Composition, Chord, SynthPresets

song = Composition(tempo=100)
piano = song.add_track('piano', SynthPresets.piano())

# C major chord in different inversions
root = Chord.major('C')
first_inv = Chord.major_first_inversion('C')
second_inv = Chord.major_second_inversion('C')

inversions = [root, first_inv, second_inv]

for i, chord in enumerate(inversions):
    time = i * 2.0
    freqs = chord.get_frequencies()
    for freq in freqs:
        song.play_note(freq, 2.0, time, 'piano')

song.render('inversions.wav')
```

### Extended Chords (9th, 11th, 13th)

```python
from algorythm import Composition, Chord, SynthPresets

song = Composition(tempo=100)
pad = song.add_track('pad', SynthPresets.warm_pad())

# Extended chords
maj9 = Chord.major9('C')
maj11 = Chord.major11('C')
maj13 = Chord.major13('C')

chords = [maj9, maj11, maj13]

for i, chord in enumerate(chords):
    time = i * 2.5
    freqs = chord.get_frequencies()
    for freq in freqs:
        song.play_note(freq, 2.5, time, 'pad')

song.render('extended.wav')
```

### Smooth Voice Leading

```python
from algorythm import Composition, Chord, SynthPresets

song = Composition(tempo=90)
strings = song.add_track('strings', SynthPresets.strings())

# Progression with smooth voice leading
chords = [
    Chord.major('C'),
    Chord.major('F'),
    Chord.major('G'),
    Chord.major('C')
]

for i, chord in enumerate(chords):
    time = i * 3.0
    
    # Get frequencies
    freqs = chord.get_frequencies()
    
    # Play each note
    for j, freq in enumerate(freqs):
        song.play_note(freq, 3.0, time, 'strings')

song.render('voice_leading.wav')
```

---

## Effects Chains

### Slapback Echo (Rockabilly)

```python
from algorythm import Composition, SynthPresets, DelayFX, ReverbFX

song = Composition(tempo=120)
lead = song.add_track('lead', SynthPresets.pluck())

# Slapback: short delay, no feedback
lead.add_effect(DelayFX(delay_time=0.375, feedback=0.0, mix=0.4))
lead.add_effect(ReverbFX(mix=0.2, room_size=0.3))

# Play something
song.play_note(440, 0.5, 0.0, 'lead')
song.play_note(440, 0.5, 0.5, 'lead')

song.render('slapback.wav')
```

### Dub Reggae (Heavy Reverb + Delay)

```python
from algorythm import Composition, SynthPresets, DelayFX, ReverbFX

song = Composition(tempo=90)
bass = song.add_track('bass', SynthPresets.synth_bass())

# Dub effects: lots of reverb and delay
bass.add_effect(DelayFX(delay_time=0.5, feedback=0.6, mix=0.5))
bass.add_effect(ReverbFX(mix=0.7, room_size=0.8, damping=0.2))

# Play bass line
song.play_note(55, 1.0, 0.0, 'bass')
song.play_note(55, 1.0, 1.0, 'bass')

song.render('dub.wav')
```

### Granular Pad (Stuttering Effect)

```python
from algorythm import Composition, SynthPresets, PhaserFX, ChorusFX

song = Composition(tempo=60)
pad = song.add_track('pad', SynthPresets.soft_pad())

# Granular effect through phaser + chorus
pad.add_effect(PhaserFX(rate=0.5, depth=0.6, mix=0.6))
pad.add_effect(ChorusFX(rate=0.3, depth=0.5, mix=0.4))

song.play_note(220, 8.0, 0.0, 'pad')

song.render('granular_pad.wav')
```

### Radio Filter Effect

```python
from algorythm import Composition, SynthPresets, Filter, Exporter

song = Composition(tempo=120)
voice = song.add_track('voice', SynthPresets.pluck())

# Simulate AM radio: heavy filter
voice.add_effect(Filter.lowpass(cutoff=2000, resonance=0.8))
voice.add_effect(Filter.highpass(cutoff=500, resonance=0.5))

song.play_note(440, 2.0, 0.0, 'voice')

song.render('radio.wav')
```

---

## Advanced Composition

### 12-Bar Blues Structure

```python
from algorythm import Composition, SynthPresets, Scale, Motif, Chord
from algorythm import ReverbFX, Compressor

def bars(n, tempo=120):
    return (n * 4.0 * 60.0) / tempo

song = Composition(tempo=120)

kick = song.add_track('kick', SynthPresets.kick())
bass = song.add_track('bass', SynthPresets.synth_bass())
lead = song.add_track('lead', SynthPresets.pluck())

scale = Scale.blues('E')
lead.add_effect(ReverbFX(mix=0.2))

# 12-bar blues: I-I-I-I-IV-IV-I-I-V-IV-I-V

# Chords
chords = [
    Chord.dominant7('E'),    # I
    Chord.dominant7('E'),    # I
    Chord.dominant7('E'),    # I
    Chord.dominant7('E'),    # I
    Chord.dominant7('A'),    # IV
    Chord.dominant7('A'),    # IV
    Chord.dominant7('E'),    # I
    Chord.dominant7('E'),    # I
    Chord.dominant7('B'),    # V
    Chord.dominant7('A'),    # IV
    Chord.dominant7('E'),    # I
    Chord.dominant7('B'),    # V
]

# Drums all the way
for bar in range(12):
    time = bars(bar)
    for beat in range(4):
        song.play_note(60, 0.08, time + (beat * 0.5), 'kick')

# Bass follows chord progression
for i, chord in enumerate(chords):
    time = bars(i)
    root = chord.get_frequencies()[0]
    song.play_note(root, 2.0, time, 'bass')

# Lead melody
lead_motif = Motif.from_intervals([0, 2, 3, 5, 7], scale=scale, duration=0.5)
song.play_motif(lead_motif, start=bars(0), track='lead')

song.render('12bar_blues.wav')
```

### Song with Dynamics (Verse→Chorus)

```python
from algorythm import Composition, SynthPresets, Scale, Motif

def bars(n, tempo=120):
    return (n * 4.0 * 60.0) / tempo

song = Composition(tempo=120)

drums = song.add_track('drums', SynthPresets.kick())
bass = song.add_track('bass', SynthPresets.synth_bass())
melody = song.add_track('melody', SynthPresets.pluck())
pad = song.add_track('pad', SynthPresets.warm_pad())

scale = Scale.major('C')

# Intro (4 bars, minimal)
for bar in range(4):
    time = bars(bar)
    song.play_note(60, 0.1, time + 0.0, 'drums')
    song.play_note(60, 0.1, time + 1.0, 'drums')

# Verse 1 (8 bars, add bass)
for bar in range(8):
    time = bars(4 + bar)
    song.play_note(60, 0.1, time + 0.0, 'drums')
    song.play_note(60, 0.1, time + 1.0, 'drums')

bass_motif = Motif.from_intervals([0, 0, 4, 4], scale=scale, octave=2, duration=0.5)
song.play_motif(bass_motif, start=bars(4), track='bass')

# Chorus (8 bars, add melody + pad)
for bar in range(8):
    time = bars(12 + bar)
    song.play_note(60, 0.1, time + 0.0, 'drums')
    song.play_note(60, 0.1, time + 1.0, 'drums')

melody_motif = Motif.from_intervals([0, 2, 4, 5, 7], scale=scale, duration=0.5)
song.play_motif(bass_motif, start=bars(12), track='bass')
song.play_motif(melody_motif, start=bars(12), track='melody')

pad_note = Motif.from_intervals([0, 2, 4], scale=scale, duration=8.0)
song.play_motif(pad_note, start=bars(12), track='pad')

# Outro (4 bars, drop melody)
for bar in range(4):
    time = bars(20 + bar)
    song.play_note(60, 0.1, time + 0.0, 'drums')
    song.play_note(60, 0.1, time + 1.0, 'drums')

song.render('dynamic_song.wav')
```

### Layered Strings Arrangement

```python
from algorythm import Composition, SynthPresets, Chord, Scale

song = Composition(tempo=70)

violins = song.add_track('violins', SynthPresets.strings())
violas = song.add_track('violas', SynthPresets.cello())
cellos = song.add_track('cellos', SynthPresets.upright_bass())

scale = Scale.major('D')

# Chord progression
chords = [
    Chord.major('D'),
    Chord.major('A'),
    Chord.major('G'),
    Chord.major('D')
]

for i, chord in enumerate(chords):
    time = i * 4.0
    freqs = chord.get_frequencies()
    
    # High strings (violins)
    song.play_note(freqs[-1] * 2, 4.0, time, 'violins')
    
    # Mid strings (violas)
    song.play_note(freqs[len(freqs)//2], 4.0, time, 'violas')
    
    # Low strings (cellos)
    song.play_note(freqs[0], 4.0, time, 'cellos')

song.render('string_arrangement.wav')
```

---

## Generative & Algorithmic

### Random Melody within Constraints

```python
import random
from algorythm import Composition, SynthPresets, Scale, Motif

song = Composition(tempo=120)
synth = song.add_track('random', SynthPresets.pluck())

scale = Scale.pentatonic('C')

# Generate random constrained melody
notes = [0]  # Start at root
for _ in range(15):
    # Next note within 2 semitones of current
    options = [notes[-1] - 2, notes[-1] - 1, notes[-1], 
               notes[-1] + 1, notes[-1] + 2]
    next_note = random.choice(options)
    next_note = max(0, min(7, next_note))  # Keep in range
    notes.append(next_note)

melody = Motif.from_intervals(notes, scale=scale, duration=0.25)
song.play_motif(melody, start=0.0, track='random')

song.render('constrained_random.wav')
```

### Repeat with Variation

```python
from algorythm import Composition, SynthPresets, Scale, Motif

song = Composition(tempo=120)
lead = song.add_track('lead', SynthPresets.pluck())

scale = Scale.major('C')

# Base pattern
base = [0, 2, 4, 5, 7, 5, 4, 2]

# Variation: add notes
variation1 = [0, 1, 2, 3, 4, 5, 6, 7]

# Variation: change rhythm
motif1 = Motif.from_intervals(base, scale=scale, duration=0.5)
motif2 = Motif.from_intervals(variation1, scale=scale, duration=0.25)

# Play pattern 4 times with variations
song.play_motif(motif1, start=0.0, track='lead')
song.play_motif(motif2, start=4.0, track='lead')
song.play_motif(motif1, start=8.0, track='lead')
song.play_motif(motif2, start=12.0, track='lead')

song.render('repeat_variation.wav')
```

### Pattern Accumulation

```python
from algorythm import Composition, SynthPresets, Scale, Motif

song = Composition(tempo=140)
synth1 = song.add_track('synth1', SynthPresets.pluck())
synth2 = song.add_track('synth2', SynthPresets.pluck())
synth3 = song.add_track('synth3', SynthPresets.pluck())

scale = Scale.minor('A')

# Pattern 1
p1 = Motif.from_intervals([0, 0, 0, 0], scale=scale, duration=0.25)

# Pattern 2
p2 = Motif.from_intervals([0, 2, 3, 5], scale=scale, duration=0.25)

# Pattern 3
p3 = Motif.from_intervals([5, 3, 2, 0], scale=scale, duration=0.25)

# Bar 1: Just pattern 1
song.play_motif(p1, start=0.0, track='synth1')

# Bar 2: Pattern 1 + 2
song.play_motif(p1, start=2.0, track='synth1')
song.play_motif(p2, start=2.0, track='synth2')

# Bar 3: Pattern 1 + 2 + 3
song.play_motif(p1, start=4.0, track='synth1')
song.play_motif(p2, start=4.0, track='synth2')
song.play_motif(p3, start=4.0, track='synth3')

song.render('accumulation.wav')
```

---

## Experimental & Creative

### Shepard Tone (Infinite Ascending)

```python
import numpy as np
from algorythm import Exporter

# Shepard tone: notes spiral upward but never get higher
sample_rate = 44100
duration = 4.0
num_samples = int(sample_rate * duration)
t = np.linspace(0, duration, num_samples)

# Multiple sine waves at octave intervals
audio = np.zeros(num_samples)
base_freq = 440

for octave in range(4):
    freq = base_freq * (2 ** octave)
    
    # Frequency sweep within octave
    freq_sweep = freq + (50 * t)
    
    # Amplitude envelope (fade in/out)
    envelope = np.sin(np.pi * t / duration)
    
    # Add to audio
    audio += np.sin(2 * np.pi * freq_sweep * t) * envelope

audio = audio / np.max(np.abs(audio))
Exporter().export(audio, 'shepard.wav')
```

### Glitch Effect (Bit Crushing + Stuttering)

```python
import numpy as np
from algorythm import Synth, Exporter

synth = Synth(waveform='saw')
audio = synth.generate_note(220, 2.0)

# Bit crush effect
bit_depth = 6
audio_crushed = np.round(audio * (2 ** bit_depth)) / (2 ** bit_depth)

# Stutter effect
stutter_size = 441  # 10ms at 44100Hz
for i in range(0, len(audio_crushed) - stutter_size, stutter_size * 2):
    # Repeat every other stutter window
    audio_crushed[i:i + stutter_size] = audio_crushed[i:i + stutter_size].copy()

Exporter().export(audio_crushed, 'glitch.wav')
```

### Binaural Beats (Brain Entrainment)

```python
import numpy as np
from algorythm import Exporter

# Binaural beat: Left ear 440Hz, Right ear 450Hz = 10Hz difference
sample_rate = 44100
duration = 10.0
num_samples = int(sample_rate * duration)
t = np.linspace(0, duration, num_samples)

# Left channel: 440 Hz
left = np.sin(2 * np.pi * 440 * t)

# Right channel: 450 Hz (10 Hz difference)
right = np.sin(2 * np.pi * 450 * t)

# Stereo audio
stereo = np.column_stack([left, right])

Exporter().export(stereo, 'binaural.wav')
```

### Karplus-Strong String Synthesis

```python
import numpy as np
from algorythm import Exporter

def karplus_strong(freq, duration=1.0, sample_rate=44100):
    # Buffer size based on frequency
    buffer_size = int(sample_rate / freq)
    
    # Initialize with noise
    buffer = np.random.uniform(-0.5, 0.5, buffer_size)
    
    # Generate audio
    num_samples = int(sample_rate * duration)
    audio = np.zeros(num_samples)
    
    for i in range(num_samples):
        # Output sample
        output = buffer[i % buffer_size]
        audio[i] = output
        
        # Update buffer (simple averaging)
        new_sample = (buffer[i % buffer_size] + 
                      buffer[(i + 1) % buffer_size]) / 2
        buffer[i % buffer_size] = new_sample * 0.99  # Decay
    
    return audio

# Generate plucked string
audio = karplus_strong(330, 2.0)
audio = audio / np.max(np.abs(audio))

Exporter().export(audio, 'karplus_strong.wav')
```

---

## Mixing & Mastering Recipes

### Proper Gain Staging

```python
from algorythm import Composition, SynthPresets, Compressor, Limiter

song = Composition(tempo=120)

kick = song.add_track('kick', SynthPresets.kick())
bass = song.add_track('bass', SynthPresets.synth_bass())
melody = song.add_track('melody', SynthPresets.pluck())

# Set proper levels (leave headroom)
kick.set_volume(0.8)    # Loudest element
bass.set_volume(0.7)
melody.set_volume(0.6)

# Add per-track compression for control
kick.add_effect(Compressor(threshold=-15, ratio=3.0))
bass.add_effect(Compressor(threshold=-20, ratio=2.0))

# Master compression and limiter
song.add_master_effect(Compressor(threshold=-10, ratio=2.0))
song.add_master_effect(Limiter(threshold=-2))

# Test
song.play_note(440, 1.0, 0.0, 'kick')
song.play_note(55, 1.0, 0.0, 'bass')
song.play_note(440, 1.0, 0.0, 'melody')

song.render('proper_gain.wav')
```

### EQ Separation (Bass, Mids, Treble)

```python
from algorythm import Composition, SynthPresets, Filter

song = Composition(tempo=120)

bass_track = song.add_track('bass', SynthPresets.synth_bass())
mid_track = song.add_track('mids', SynthPresets.warm_pad())
treble_track = song.add_track('treble', SynthPresets.pluck())

# EQ each track to its frequency range
# Bass: keep only <250Hz
bass_track.add_effect(Filter.lowpass(cutoff=250))

# Mids: 250-2000Hz
mid_track.add_effect(Filter.highpass(cutoff=250))
mid_track.add_effect(Filter.lowpass(cutoff=2000))

# Treble: keep >2000Hz
treble_track.add_effect(Filter.highpass(cutoff=2000))

# Play test tones
song.play_note(110, 1.0, 0.0, 'bass')
song.play_note(440, 1.0, 0.0, 'mids')
song.play_note(2000, 1.0, 0.0, 'treble')

song.render('eq_separation.wav')
```

### Wide Stereo Arrangement

```python
from algorythm import Composition, SynthPresets, ChorusFX

song = Composition(tempo=120)

left = song.add_track('left', SynthPresets.pluck())
center = song.add_track('center', SynthPresets.synth_lead())
right = song.add_track('right', SynthPresets.pluck())

# Pan to create width
left.set_pan(-0.8)
center.set_pan(0.0)
right.set_pan(0.8)

# Add chorus to widen further
left.add_effect(ChorusFX(rate=0.3, depth=0.3, mix=0.4))
right.add_effect(ChorusFX(rate=0.4, depth=0.3, mix=0.4))

# Play
song.play_note(440, 1.0, 0.0, 'left')
song.play_note(880, 1.0, 0.0, 'center')
song.play_note(440, 1.0, 0.0, 'right')

song.render('wide_stereo.wav')
```
