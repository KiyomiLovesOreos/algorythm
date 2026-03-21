# Intermediate Recipes: Building Blocks to Full Tracks

Step up from basic recipes to more sophisticated techniques.

## Layering Basics

### Simple Two-Track Composition

```python
from algorythm import Composition, SynthPresets, Scale, Motif

song = Composition(tempo=120)
melody = song.add_track('melody', SynthPresets.pluck())
bass = song.add_track('bass', SynthPresets.synth_bass())

scale = Scale.major('C')

# Melody pattern
melody_notes = [0, 2, 4, 5, 7, 5, 4, 2]
melody_motif = Motif.from_intervals(melody_notes, scale=scale, duration=0.5)

# Bass pattern (follows root)
bass_notes = [0, 0, 0, 0]
bass_motif = Motif.from_intervals(bass_notes, scale=scale, octave=2, duration=0.5)

# Play both
song.play_motif(melody_motif, start=0.0, track='melody')
song.play_motif(bass_motif, start=0.0, track='bass')

song.render('two_track.wav')
```

### Drums + Melodic Element

```python
from algorythm import Composition, SynthPresets, Scale, Motif

song = Composition(tempo=120)
drums = song.add_track('drums', SynthPresets.kick())
synth = song.add_track('synth', SynthPresets.pluck())

scale = Scale.pentatonic('C')

# Drum pattern (4 bars)
for bar in range(4):
    time = bar * 2.0
    song.play_note(60, 0.1, time + 0.0, 'drums')
    song.play_note(60, 0.1, time + 1.0, 'drums')

# Melodic line over drums
melody = Motif.from_intervals([0, 2, 4, 5, 7], scale=scale, duration=0.5)
song.play_motif(melody, start=0.0, track='synth')

song.render('drums_melody.wav')
```

### Three-Part Harmony

```python
from algorythm import Composition, SynthPresets, Chord

song = Composition(tempo=100)
high = song.add_track('high', SynthPresets.strings())
mid = song.add_track('mid', SynthPresets.warm_pad())
low = song.add_track('low', SynthPresets.upright_bass())

# Chord progression
chords = [
    Chord.major('C'),
    Chord.major('F'),
    Chord.major('G'),
    Chord.major('C')
]

for i, chord in enumerate(chords):
    time = i * 3.0
    freqs = chord.get_frequencies()
    
    # High notes (violins)
    song.play_note(freqs[-1] * 2, 3.0, time, 'high')
    
    # Mid notes (pad)
    song.play_note(freqs[1], 3.0, time, 'mid')
    
    # Low notes (bass)
    song.play_note(freqs[0], 3.0, time, 'low')

song.render('three_harmony.wav')
```

---

## Effects Integration

### Adding Reverb to Create Space

```python
from algorythm import Composition, SynthPresets, Scale, Motif, ReverbFX

song = Composition(tempo=120)

# Dry track (no effects)
dry = song.add_track('dry', SynthPresets.pluck())

# Wet track (with reverb)
wet = song.add_track('wet', SynthPresets.pluck())
wet.add_effect(ReverbFX(mix=0.3, room_size=0.5))

scale = Scale.major('C')
melody = Motif.from_intervals([0, 2, 4, 5, 7], scale=scale, duration=0.5)

# Play same melody on both
song.play_motif(melody, start=0.0, track='dry')
song.play_motif(melody, start=0.0, track='wet')

song.render('reverb_comparison.wav')
```

### Delay as a Creative Tool

```python
from algorythm import Composition, SynthPresets, Scale, Motif, DelayFX

song = Composition(tempo=120)

# Track with delay that creates a "slap"
lead = song.add_track('lead', SynthPresets.pluck())
lead.add_effect(DelayFX(delay_time=0.375, feedback=0.2, mix=0.3))

scale = Scale.major('C')
melody = Motif.from_intervals([0, 2, 4, 5, 7], scale=scale, duration=0.5)

song.play_motif(melody, start=0.0, track='lead')

song.render('delay_slap.wav')
```

### Layered Effects

```python
from algorythm import Composition, SynthPresets, ReverbFX, DelayFX, ChorusFX

song = Composition(tempo=120)
pad = song.add_track('pad', SynthPresets.warm_pad())

# Stack multiple effects
pad.add_effect(ChorusFX(rate=0.3, depth=0.3, mix=0.3))
pad.add_effect(DelayFX(delay_time=0.5, feedback=0.2, mix=0.2))
pad.add_effect(ReverbFX(mix=0.2, room_size=0.4))

song.play_note(220, 4.0, 0.0, 'pad')

song.render('layered_effects.wav')
```

---

## Rhythm Building

### Simple Drum Programming

```python
from algorythm import Composition, SynthPresets

song = Composition(tempo=120)
kick = song.add_track('kick', SynthPresets.kick())
snare = song.add_track('snare', SynthPresets.snare())
hihat = song.add_track('hihat', SynthPresets.hihat())

# 1 bar pattern, repeat 4 times
for bar in range(4):
    time = bar * 2.0
    
    # Kick: on beats 1 and 3
    song.play_note(60, 0.1, time + 0.0, 'kick')
    song.play_note(60, 0.1, time + 1.0, 'kick')
    
    # Snare: on beats 2 and 4
    song.play_note(60, 0.1, time + 0.5, 'snare')
    song.play_note(60, 0.1, time + 1.5, 'snare')
    
    # Hi-hat: every 8th note
    for i in range(8):
        song.play_note(60, 0.05, time + (i * 0.25), 'hihat')

song.render('drum_pattern.wav')
```

### Kick & Bass Lock

```python
from algorythm import Composition, SynthPresets, Scale

song = Composition(tempo=120)
kick = song.add_track('kick', SynthPresets.kick())
bass = song.add_track('bass', SynthPresets.synth_bass())

scale = Scale.minor('A')

# Lock kick and bass on same timing
for beat in range(8):
    time = beat * 0.5
    
    # Kick hits
    song.play_note(60, 0.08, time, 'kick')
    
    # Bass hits same time for tightness
    song.play_note(55, 0.5, time, 'bass')

song.render('kick_bass_lock.wav')
```

### Finger Snapping (Synthetic)

```python
from algorythm import Composition, Synth, Exporter
import numpy as np

song = Composition(tempo=120)

# Create snapping sound
synth = Synth(waveform='square')

# Snap has fast attack and quick decay
for beat in range(8):
    time = beat * 0.5
    
    # Generate snap sound
    snap = synth.generate_note(8000, 0.05)  # High frequency
    
    # This is simplified - in real usage you'd add the snap at the time
    song.play_note(8000, 0.05, time, 'snare')

song.render('finger_snap.wav')
```

---

## Harmonic Concepts

### Using Inversions for Movement

```python
from algorythm import Composition, Chord, SynthPresets

song = Composition(tempo=100)
pad = song.add_track('pad', SynthPresets.warm_pad())

# Same chord, different inversions = different feel
chords = [
    ('C', 'root'),      # C E G
    ('C', 'first'),     # E G C
    ('C', 'second'),    # G C E
]

for i, (note, inversion) in enumerate(chords):
    time = i * 2.0
    
    if inversion == 'root':
        chord = Chord.major(note)
    elif inversion == 'first':
        chord = Chord.major_first_inversion(note)
    else:
        chord = Chord.major_second_inversion(note)
    
    freqs = chord.get_frequencies()
    for freq in freqs:
        song.play_note(freq, 2.0, time, 'pad')

song.render('inversions.wav')
```

### Parallel Chord Motion

```python
from algorythm import Composition, Chord, SynthPresets

song = Composition(tempo=100)
strings = song.add_track('strings', SynthPresets.strings())

# Move all voices together (parallel motion)
start_notes = Chord.major('C').get_frequencies()
end_notes = Chord.major('G').get_frequencies()

for i, (start_freq, end_freq) in enumerate(zip(start_notes, end_notes)):
    song.play_note(start_freq, 2.0, 0.0, 'strings')
    song.play_note(end_freq, 2.0, 2.0, 'strings')

song.render('parallel_motion.wav')
```

### Creating Tension with Chord Extensions

```python
from algorythm import Composition, Chord, SynthPresets

song = Composition(tempo=100)
synth = song.add_track('synth', SynthPresets.warm_pad())

# Build tension: simple → extended
chords = [
    Chord.major('C'),      # Stable
    Chord.major7('C'),     # Slightly tense
    Chord.major9('C'),     # More tense
    Chord.major11('C'),    # Very tense
]

for i, chord in enumerate(chords):
    time = i * 2.0
    freqs = chord.get_frequencies()
    
    for freq in freqs:
        song.play_note(freq, 2.0, time, 'synth')

song.render('tension_build.wav')
```

---

## Melodic Techniques

### Call and Answer (Simple)

```python
from algorythm import Composition, SynthPresets, Scale, Motif

song = Composition(tempo=120)
synth = song.add_track('synth', SynthPresets.pluck())

scale = Scale.major('C')

# Call phrase
call = Motif.from_intervals([0, 2, 4, 5], scale=scale, duration=0.5)

# Answer (slightly different)
answer = Motif.from_intervals([5, 4, 2, 0], scale=scale, duration=0.5)

# Dialogue
song.play_motif(call, start=0.0, track='synth')
song.play_motif(answer, start=2.0, track='synth')
song.play_motif(call, start=4.0, track='synth')
song.play_motif(answer, start=6.0, track='synth')

song.render('call_answer.wav')
```

### Rhythmic Variation

```python
from algorythm import Composition, SynthPresets, Scale, Motif

song = Composition(tempo=120)
lead = song.add_track('lead', SynthPresets.pluck())

scale = Scale.major('C')
notes = [0, 2, 4, 5, 7]

# Same notes, different rhythms
rhythm1 = Motif.from_intervals(notes, scale=scale, duration=0.5)
rhythm2 = Motif.from_intervals(notes, scale=scale, duration=0.25)
rhythm3 = Motif.from_intervals(notes, scale=scale, duration=0.75)

song.play_motif(rhythm1, start=0.0, track='lead')
song.play_motif(rhythm2, start=4.0, track='lead')
song.play_motif(rhythm3, start=8.0, track='lead')

song.render('rhythm_variation.wav')
```

### Interval Patterns

```python
from algorythm import Composition, SynthPresets, Scale, Motif

song = Composition(tempo=120)
synth = song.add_track('synth', SynthPresets.pluck())

scale = Scale.major('C')

# Step pattern (by 1)
step = Motif.from_intervals([0, 1, 2, 3, 4, 5, 6, 7], scale=scale, duration=0.25)

# Skip pattern (by 2)
skip = Motif.from_intervals([0, 2, 4, 6], scale=scale, duration=0.5)

# Jump pattern (by 3)
jump = Motif.from_intervals([0, 3, 6], scale=scale, duration=0.75)

song.play_motif(step, start=0.0, track='synth')
song.play_motif(skip, start=4.0, track='synth')
song.play_motif(jump, start=8.0, track='synth')

song.render('interval_patterns.wav')
```

---

## Simple Arrangements

### Verse-Chorus Structure

```python
from algorythm import Composition, SynthPresets, Scale, Motif

song = Composition(tempo=120)
drums = song.add_track('drums', SynthPresets.kick())
bass = song.add_track('bass', SynthPresets.synth_bass())
lead = song.add_track('lead', SynthPresets.pluck())

scale = Scale.major('C')

# Verse (8 bars): drums + bass only
for bar in range(8):
    time = bar * 2.0
    song.play_note(60, 0.1, time + 0.0, 'drums')
    song.play_note(60, 0.1, time + 1.0, 'drums')

bass_motif = Motif.from_intervals([0, 0, 4, 4], scale=scale, octave=2, duration=0.5)
song.play_motif(bass_motif, start=0.0, track='bass')

# Chorus (8 bars): add lead
for bar in range(8, 16):
    time = bar * 2.0
    song.play_note(60, 0.1, time + 0.0, 'drums')
    song.play_note(60, 0.1, time + 1.0, 'drums')

song.play_motif(bass_motif, start=16.0, track='bass')

lead_motif = Motif.from_intervals([0, 2, 4, 5, 7], scale=scale, duration=0.5)
song.play_motif(lead_motif, start=16.0, track='lead')

song.render('verse_chorus.wav')
```

### Building Intensity

```python
from algorythm import Composition, SynthPresets, Scale, Motif, ReverbFX

song = Composition(tempo=120)
drums = song.add_track('drums', SynthPresets.kick())
bass = song.add_track('bass', SynthPresets.synth_bass())
pad = song.add_track('pad', SynthPresets.warm_pad())

scale = Scale.minor('A')

# Section 1: Just drums (4 bars)
for bar in range(4):
    time = bar * 2.0
    for beat in range(4):
        song.play_note(60, 0.08, time + (beat * 0.5), 'drums')

# Section 2: Add bass (4 bars)
for bar in range(4, 8):
    time = bar * 2.0
    for beat in range(4):
        song.play_note(60, 0.08, time + (beat * 0.5), 'drums')

song.play_note(55, 8.0, 8.0, 'bass')

# Section 3: Add pad (4 bars)
for bar in range(8, 12):
    time = bar * 2.0
    for beat in range(4):
        song.play_note(60, 0.08, time + (beat * 0.5), 'drums')

song.play_note(55, 8.0, 16.0, 'bass')
song.play_note(220, 8.0, 16.0, 'pad')

song.render('building_intensity.wav')
```

---

## Production Practices

### Test Mixing Levels

```python
from algorythm import Composition, SynthPresets

song = Composition(tempo=120)
drums = song.add_track('drums', SynthPresets.kick())
bass = song.add_track('bass', SynthPresets.synth_bass())
lead = song.add_track('lead', SynthPresets.pluck())
pad = song.add_track('pad', SynthPresets.warm_pad())

# Set levels
drums.set_volume(1.0)   # Reference level
bass.set_volume(0.8)
lead.set_volume(0.7)
pad.set_volume(0.5)

# Play test tones
song.play_note(60, 1.0, 0.0, 'drums')
song.play_note(55, 1.0, 0.0, 'bass')
song.play_note(440, 1.0, 0.0, 'lead')
song.play_note(220, 1.0, 0.0, 'pad')

song.render('level_test.wav')
```

### A/B Testing Effects

```python
from algorythm import Composition, SynthPresets, ReverbFX

song = Composition(tempo=120)

# Without reverb
dry = song.add_track('dry', SynthPresets.pluck())

# With reverb
wet = song.add_track('wet', SynthPresets.pluck())
wet.add_effect(ReverbFX(mix=0.3))

# Play same note on both (A/B comparison)
song.play_note(440, 1.0, 0.0, 'dry')
song.play_note(440, 1.0, 1.0, 'wet')

song.render('ab_test.wav')
```

### Frequency Separation

```python
from algorythm import Composition, SynthPresets

song = Composition(tempo=120)

# Low frequencies
bass = song.add_track('bass', SynthPresets.synth_bass())
bass.set_volume(0.9)

# Mid frequencies
mid = song.add_track('mid', SynthPresets.pluck())
mid.set_volume(0.7)

# High frequencies
high = song.add_track('high', SynthPresets.synth_lead())
high.set_volume(0.6)

song.play_note(55, 1.0, 0.0, 'bass')
song.play_note(440, 1.0, 0.0, 'mid')
song.play_note(2000, 1.0, 0.0, 'high')

song.render('frequency_separation.wav')
```

---

## Practical Compositions

### Simple Song (30 seconds)

```python
from algorythm import Composition, SynthPresets, Scale, Motif

def bars(n):
    return (n * 4.0 * 60.0) / 120

song = Composition(tempo=120)

kick = song.add_track('kick', SynthPresets.kick())
bass = song.add_track('bass', SynthPresets.synth_bass())
lead = song.add_track('lead', SynthPresets.pluck())

scale = Scale.major('C')

# Intro (4 bars)
for bar in range(4):
    time = bars(bar)
    song.play_note(60, 0.1, time + 0.0, 'kick')
    song.play_note(60, 0.1, time + 1.0, 'kick')

# Main (12 bars): add bass and melody
for bar in range(12):
    time = bars(4 + bar)
    song.play_note(60, 0.1, time + 0.0, 'kick')
    song.play_note(60, 0.1, time + 1.0, 'kick')

bass_motif = Motif.from_intervals([0, 0, 4, 4], scale=scale, octave=2, duration=0.5)
song.play_motif(bass_motif, start=bars(4), track='bass')

lead_motif = Motif.from_intervals([0, 2, 4, 5, 7], scale=scale, duration=0.5)
song.play_motif(lead_motif, start=bars(4), track='lead')

# Outro (4 bars): drop lead
for bar in range(4):
    time = bars(16 + bar)
    song.play_note(60, 0.1, time + 0.0, 'kick')
    song.play_note(60, 0.1, time + 1.0, 'kick')

song.play_motif(bass_motif, start=bars(16), track='bass')

song.render('simple_song.wav')
```

### Looped Section (Extended)

```python
from algorythm import Composition, SynthPresets, Scale, Motif

song = Composition(tempo=120)
synth1 = song.add_track('synth1', SynthPresets.pluck())
synth2 = song.add_track('synth2', SynthPresets.pluck())

scale = Scale.pentatonic('C')

# Pattern 1
p1 = Motif.from_intervals([0, 0, 0, 0], scale=scale, duration=0.25)

# Pattern 2
p2 = Motif.from_intervals([0, 2, 4, 5], scale=scale, duration=0.25)

# Loop combination 8 times (32 seconds at 120 BPM)
for loop in range(8):
    start = loop * 4.0
    song.play_motif(p1, start=start, track='synth1')
    song.play_motif(p2, start=start, track='synth2')

song.render('extended_loop.wav')
```

---

## Tips for Intermediate Work

1. **Mix as you go** - Set volumes and panning early
2. **Use effects strategically** - Not on everything
3. **Build arrangement gradually** - Add tracks one at a time
4. **Reference your levels** - Test frequently
5. **Use headroom** - Leave room for mastering
6. **Keep it simple** - Too many layers gets muddy
7. **Test on different systems** - Ears get fatigued
8. **Document your settings** - Remember what worked
9. **Use structure** - Verse, chorus, bridge
10. **Listen critically** - Take breaks, come back fresh
