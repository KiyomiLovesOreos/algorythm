# Composition Guide

Build complete multi-track arrangements with Algorythm.

## Composition Basics

The `Composition` class is your workspace for arranging tracks.

### Creating a Composition

```python
from algorythm import Composition

song = Composition(
    tempo=120,           # BPM
    time_signature=(4, 4),  # Time signature
    sample_rate=44100    # Audio sample rate
)
```

## Tracks

Tracks hold instruments and their audio.

### Adding Tracks

```python
from algorythm import Composition, SynthPresets

song = Composition(tempo=120)

# Add a track with an instrument
melody_track = song.add_track('melody', SynthPresets.pluck())
bass_track = song.add_track('bass', SynthPresets.synth_bass())
drums_track = song.add_track('drums', SynthPresets.kick())
```

Each track has:
- A name (for reference)
- An instrument (synthesizer)
- An effect chain (optional)
- Volume control

### Track Properties

```python
# Set track volume
melody_track.set_volume(0.8)  # 0.0 to 1.0

# Mute/unmute
melody_track.mute()
melody_track.unmute()

# Solo (mute all others)
melody_track.solo()
```

## Adding Notes and Patterns

### Playing Single Notes

```python
# Play a note at a specific time
song.play_note(
    frequency=440,      # Hz
    duration=1.0,       # Seconds
    start=0.0,          # Start time
    track='melody'      # Track name
)
```

### Playing Motifs

```python
from algorythm import Scale, Motif

scale = Scale.major('C')
melody = Motif.from_intervals([0, 2, 4, 5, 7], scale=scale, duration=0.5)

# Play motif at specified time
song.play_motif(
    motif=melody,
    start=0.0,          # Start time in seconds
    track='melody'
)
```

### Repeating Patterns

```python
# Repeat a motif multiple times
melody = Motif.from_intervals([0, 2, 4, 2], scale=Scale.major('C'), duration=0.5)

for i in range(4):
    start_time = i * 2.0  # Each repetition starts 2 seconds apart
    song.play_motif(melody, start=start_time, track='melody')
```

### Playing Chords

```python
from algorythm import Chord

chord = Chord.major('C')
frequencies = chord.get_frequencies()

# Play all notes of chord at same time
for freq in frequencies:
    song.play_note(freq, duration=2.0, start=0.0, track='melody')
```

## Arrangement Structure

Build songs section by section.

### Sections

```python
from algorythm import Composition, SynthPresets, Scale, Motif

song = Composition(tempo=120)
song.add_track('melody', SynthPresets.pluck())
song.add_track('bass', SynthPresets.synth_bass())

scale = Scale.minor('A')

# Intro (0-8 seconds)
intro_melody = Motif.from_intervals([0, 2, 3], scale=scale, duration=1.0)
song.play_motif(intro_melody, start=0.0, track='melody')

# Verse (8-24 seconds)
verse_melody = Motif.from_intervals([0, 2, 3, 5, 7], scale=scale, duration=0.5)
for i in range(4):
    song.play_motif(verse_melody, start=8.0 + i*4.0, track='melody')

# Chorus (24-40 seconds)
chorus_melody = Motif.from_intervals([7, 5, 3, 0], scale=scale, duration=0.5)
for i in range(4):
    song.play_motif(chorus_melody, start=24.0 + i*4.0, track='melody')

song.render('full_song.wav')
```

## Working with Time

### Tempo and Timing

```python
# Convert bars to seconds
bars_to_seconds = (bars * 4) / (tempo / 60)

# At 120 BPM:
# 1 bar = 2 seconds
# 1 beat = 0.5 seconds

# Helper function
def bars_to_time(bars, tempo=120):
    return (bars * 4) / (tempo / 60)

# Use it
start_time = bars_to_time(4)  # Start at bar 4
song.play_motif(melody, start=start_time, track='melody')
```

### Bars and Beats

```python
# Work in bars
TEMPO = 120
BAR_LENGTH = (4 * 60) / TEMPO  # 2.0 seconds at 120 BPM

# Arrange by bars
intro_start = 0 * BAR_LENGTH
verse_start = 4 * BAR_LENGTH
chorus_start = 12 * BAR_LENGTH

song.play_motif(intro_pattern, start=intro_start, track='melody')
song.play_motif(verse_pattern, start=verse_start, track='melody')
song.play_motif(chorus_pattern, start=chorus_start, track='melody')
```

## Automation

Automate parameters over time.

### Parameter Automation

```python
from algorythm import Automation

# Create automation curve
volume_automation = Automation(
    start_value=0.0,
    end_value=1.0,
    duration=4.0,
    curve='linear'  # 'linear', 'exponential', 'logarithmic'
)

# Apply to track
track.automate_parameter('volume', volume_automation, start_time=0.0)
```

### Automation Curves

```python
# Linear fade in
fade_in = Automation(start_value=0.0, end_value=1.0, duration=2.0, curve='linear')

# Exponential swell
swell = Automation(start_value=0.0, end_value=1.0, duration=3.0, curve='exponential')

# Logarithmic fade out
fade_out = Automation(start_value=1.0, end_value=0.0, duration=2.0, curve='logarithmic')
```

### Multiple Automation Tracks

```python
from algorythm import AutomationTrack

# Create automation track
auto_track = AutomationTrack()

# Add multiple automations
auto_track.add(Automation(0.0, 1.0, 2.0), start=0.0)
auto_track.add(Automation(1.0, 0.5, 2.0), start=4.0)
auto_track.add(Automation(0.5, 1.0, 2.0), start=8.0)

# Apply to parameter
track.apply_automation_track('volume', auto_track)
```

## Effects on Tracks

### Adding Effects

```python
from algorythm import ReverbFX, DelayFX, Compressor

track = song.add_track('melody', SynthPresets.pluck())

# Add individual effects
track.add_effect(Compressor(threshold=-15, ratio=3.0))
track.add_effect(DelayFX(delay_time=0.375, feedback=0.3))
track.add_effect(ReverbFX(mix=0.3))
```

### Effect Chains

```python
from algorythm import FXChain, EQ, Compressor, ReverbFX

# Build chain
chain = FXChain()
chain.add(EQ(low_gain=-3, mid_gain=2, high_gain=1))
chain.add(Compressor(threshold=-12, ratio=4.0))
chain.add(ReverbFX(mix=0.25))

# Add to track
track.add_effect_chain(chain)
```

## Mixing

Balance levels and frequencies.

### Volume Mixing

```python
# Set relative levels
song.get_track('melody').set_volume(0.8)
song.get_track('bass').set_volume(1.0)
song.get_track('drums').set_volume(0.7)
song.get_track('pad').set_volume(0.5)
```

General mixing guidelines:
- Bass: loudest (0.9-1.0)
- Drums: loud (0.7-0.9)
- Lead melody: medium-loud (0.6-0.8)
- Pads/backgrounds: quiet (0.3-0.5)

### Panning

```python
# Pan tracks in stereo field
track.set_pan(-1.0)  # Full left
track.set_pan(0.0)   # Center
track.set_pan(1.0)   # Full right

# Common panning
song.get_track('bass').set_pan(0.0)      # Center
song.get_track('kick').set_pan(0.0)      # Center
song.get_track('snare').set_pan(0.0)     # Center
song.get_track('melody').set_pan(-0.3)   # Left-ish
song.get_track('harmony').set_pan(0.3)   # Right-ish
song.get_track('pad').set_pan(0.0)       # Center (wide)
```

### Master Effects

Apply effects to the entire mix:

```python
from algorythm import Compressor, Limiter, EQ

# Master compression
song.add_master_effect(Compressor(
    threshold=-10,
    ratio=2.0,
    attack=0.01,
    release=0.1
))

# Master EQ
song.add_master_effect(EQ(
    low_gain=1,
    mid_gain=0,
    high_gain=2
))

# Master limiter (prevents clipping)
song.add_master_effect(Limiter(threshold=-1))
```

## Rendering

Export your composition to audio.

### Basic Rendering

```python
# Render to WAV
song.render('my_song.wav')

# Render to MP3
song.render('my_song.mp3')

# Render to FLAC
song.render('my_song.flac')
```

### Render Options

```python
from algorythm import Exporter

exporter = Exporter()

# Custom quality
exporter.export(
    audio_data,
    'output.mp3',
    bitrate='320k'  # MP3 bitrate
)

# Normalize audio
exporter.export(
    audio_data,
    'output.wav',
    normalize=True  # Maximize volume without clipping
)
```

## Complete Example

Full song with multiple sections:

```python
from algorythm import (
    Composition, SynthPresets, Scale, Motif, Chord,
    ReverbFX, DelayFX, Compressor
)

# Setup
TEMPO = 120
BAR = (4 * 60) / TEMPO  # Bar length in seconds

song = Composition(tempo=TEMPO)

# Add tracks
melody = song.add_track('melody', SynthPresets.pluck())
bass = song.add_track('bass', SynthPresets.synth_bass())
pad = song.add_track('pad', SynthPresets.warm_pad())
drums = song.add_track('kick', SynthPresets.kick())

# Add effects
melody.add_effect(DelayFX(delay_time=0.375, feedback=0.3, mix=0.2))
melody.add_effect(ReverbFX(mix=0.2))
pad.add_effect(ReverbFX(mix=0.5, room_size=0.8))

# Set volumes
melody.set_volume(0.7)
bass.set_volume(1.0)
pad.set_volume(0.4)
drums.set_volume(0.8)

# Create patterns
scale = Scale.minor('A')
intro_melody = Motif.from_intervals([0, 2, 3], scale=scale, duration=1.0)
verse_melody = Motif.from_intervals([0, 2, 3, 5, 7, 5, 3, 2], scale=scale, duration=0.5)
bass_line = Motif.from_intervals([0, 0, 0, 0], scale=scale, octave=2, duration=1.0)

# Intro (0-8 bars)
for i in range(2):
    song.play_motif(intro_melody, start=i*4*BAR, track='melody')

# Verse (8-24 bars)
for i in range(4):
    song.play_motif(verse_melody, start=(8+i*4)*BAR, track='melody')
    song.play_motif(bass_line, start=(8+i*4)*BAR, track='bass')

# Add pad in background (starts at bar 16)
pad_chord = Motif.from_intervals([0, 2, 4], scale=scale, duration=8.0)
song.play_motif(pad_chord, start=16*BAR, track='pad')

# Master compression
song.add_master_effect(Compressor(threshold=-10, ratio=2.0))

# Render
song.render('complete_song.wav')
print("Song complete!")
```

## Tips

1. Plan your arrangement before coding
2. Work in bars/beats for easier timing
3. Start with basic structure, add details later
4. Use automation for dynamics and interest
5. Leave headroom - don't max out all levels
6. Add master limiting to prevent clipping
7. Reference professional tracks for mixing balance
8. Export to WAV first, then convert to MP3 if needed
