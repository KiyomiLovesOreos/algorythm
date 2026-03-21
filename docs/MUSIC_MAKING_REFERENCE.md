# Music Making Reference

Quick reference for actually composing music with Algorythm. Keep this open while you work.

## Basic Song Structure

```python
from algorythm import Composition, SynthPresets, Scale, Motif

# Create composition
song = Composition(tempo=120)

# Add tracks
song.add_track('drums', SynthPresets.kick())
song.add_track('bass', SynthPresets.synth_bass())
song.add_track('melody', SynthPresets.pluck())
song.add_track('pad', SynthPresets.warm_pad())

# Create your patterns here...

# Render
song.render('output.wav')
```

## Time and Tempo

### Convert Bars to Seconds

```python
# At 120 BPM, 4/4 time:
# 1 bar = 2.0 seconds
# 1 beat = 0.5 seconds

def bars(n, tempo=120):
    return (n * 4.0 * 60.0) / tempo

# Use it:
intro = bars(0)      # 0.0
verse = bars(4)      # 8.0
chorus = bars(12)    # 24.0
```

### Tempo Reference

| BPM | Bar (4/4) | Beat | Eighth | Sixteenth |
|-----|-----------|------|--------|-----------|
| 60  | 4.0s      | 1.0s | 0.5s   | 0.25s     |
| 90  | 2.67s     | 0.67s| 0.33s  | 0.17s     |
| 120 | 2.0s      | 0.5s | 0.25s  | 0.125s    |
| 140 | 1.71s     | 0.43s| 0.21s  | 0.11s     |
| 160 | 1.5s      | 0.375s| 0.19s | 0.09s     |

## Note Duration Quick Reference

At 120 BPM:
```python
whole_note = 2.0
half_note = 1.0
quarter_note = 0.5
eighth_note = 0.25
sixteenth_note = 0.125

dotted_half = 1.5
dotted_quarter = 0.75
dotted_eighth = 0.375
```

## Scales and Keys

### Common Scales

```python
from algorythm import Scale

# Major keys (bright, happy)
Scale.major('C')  # No sharps/flats
Scale.major('G')  # 1 sharp
Scale.major('D')  # 2 sharps
Scale.major('F')  # 1 flat

# Minor keys (dark, emotional)
Scale.minor('A')  # Relative to C major
Scale.minor('E')  # Relative to G major
Scale.minor('D')  # Relative to F major

# Pentatonic (safe, melodic)
Scale.pentatonic('C')
Scale.pentatonic('A')

# Blues (bluesy, groovy)
Scale.blues('C')
Scale.blues('A')
```

### Key Signature Reference

**Major Keys:**
- C: No accidentals
- G: F#
- D: F#, C#
- A: F#, C#, G#
- F: Bb
- Bb: Bb, Eb
- Eb: Bb, Eb, Ab

**Minor Keys:**
- A: No accidentals
- E: F#
- B: F#, C#
- D: Bb
- G: Bb, Eb

## Creating Melodies

### By Scale Degrees

```python
scale = Scale.major('C')

# Scale degrees (0 = root, 1 = second, etc.)
melody = Motif.from_intervals(
    [0, 2, 4, 5, 7, 5, 4, 2],  # C D E F G F E D
    scale=scale,
    duration=0.5
)
```

### Common Melodic Patterns

```python
# Rising scale
[0, 1, 2, 3, 4, 5, 6, 7]

# Falling scale
[7, 6, 5, 4, 3, 2, 1, 0]

# Arpeggio (triad)
[0, 2, 4, 7, 4, 2, 0]

# Arpeggio (7th chord)
[0, 2, 4, 6, 4, 2, 0]

# Pentatonic riff
[0, 2, 3, 5, 7]  # in pentatonic scale

# Blues lick
[0, 3, 4, 5, 4, 3, 0]  # in blues scale
```

### Melody Tips

- Keep most intervals between 1-3 scale degrees (stepwise motion)
- Large jumps (5+) create drama but use sparingly
- Return to root note (0) for resolution
- Repeat patterns for catchiness
- Vary rhythm more than pitch for interest

## Creating Bass Lines

### Basic Patterns

```python
scale = Scale.minor('A')

# Root note pattern
bass = Motif.from_intervals(
    [0, 0, 0, 0],
    scale=scale,
    octave=2,  # Two octaves down
    duration=1.0
)

# Root-fifth pattern
bass = Motif.from_intervals(
    [0, 0, 4, 4],
    scale=scale,
    octave=2,
    duration=0.5
)

# Walking bass
bass = Motif.from_intervals(
    [0, 1, 2, 3, 4, 3, 2, 1],
    scale=scale,
    octave=2,
    duration=0.5
)

# Octave jump
bass = Motif.from_intervals(
    [0, 7, 0, 7],
    scale=scale,
    octave=2,
    duration=0.5
)
```

### Bass Tips

- Keep bass in octave 2 or 3
- Use root notes of chords
- Simple is better - let it groove
- Quarter or eighth notes typically
- Lock with drums for tight rhythm

## Drum Patterns

### Basic Patterns

```python
# 4/4 Rock beat
def rock_beat(song, start, length):
    for bar in range(length):
        time = start + (bar * 2.0)  # 2 seconds per bar at 120 BPM
        
        # Kick on 1 and 3
        song.play_note(60, 0.1, time + 0.0, 'kick')
        song.play_note(60, 0.1, time + 1.0, 'kick')
        
        # Snare on 2 and 4
        song.play_note(60, 0.1, time + 0.5, 'snare')
        song.play_note(60, 0.1, time + 1.5, 'snare')
        
        # Hi-hats on eighths
        for i in range(8):
            song.play_note(60, 0.05, time + (i * 0.25), 'hihat')

# 4/4 House beat
def house_beat(song, start, length):
    for bar in range(length):
        time = start + (bar * 2.0)
        
        # Kick on every quarter note
        for i in range(4):
            song.play_note(60, 0.1, time + (i * 0.5), 'kick')
        
        # Clap on 2 and 4
        song.play_note(60, 0.1, time + 0.5, 'clap')
        song.play_note(60, 0.1, time + 1.5, 'clap')
        
        # Hi-hats on offbeats
        for i in [1, 3, 5, 7]:
            song.play_note(60, 0.05, time + (i * 0.25), 'hihat')

# Hip-hop beat
def hiphop_beat(song, start, length):
    for bar in range(length):
        time = start + (bar * 2.0)
        
        # Kick pattern
        song.play_note(60, 0.1, time + 0.0, 'kick')
        song.play_note(60, 0.1, time + 0.75, 'kick')
        song.play_note(60, 0.1, time + 1.5, 'kick')
        
        # Snare on 2 and 4
        song.play_note(60, 0.15, time + 0.5, 'snare')
        song.play_note(60, 0.15, time + 1.5, 'snare')
        
        # Hi-hats on sixteenths
        for i in range(16):
            vel = 0.08 if i % 2 == 0 else 0.04
            song.play_note(60, vel, time + (i * 0.125), 'hihat')
```

### Drum Pattern Grid (120 BPM)

```
Beat:  1   +   2   +   3   +   4   +
Time:  0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75

Rock:
Kick:  X           X
Snare:         X           X
HiHat: X   X   X   X   X   X   X   X

House:
Kick:  X       X       X       X
Clap:          X               X
HiHat:     X       X       X       X

Hip-Hop:
Kick:  X           X   X
Snare:         X               X
HiHat: X X X X X X X X X X X X X X X X
```

## Chord Progressions

### Common Progressions (by scale degree)

```python
from algorythm import Chord, Scale

scale = Scale.major('C')

# I-IV-V-I (most common)
chords = [
    Chord.from_scale_degrees([0, 2, 4], scale),  # I
    Chord.from_scale_degrees([3, 5, 7], scale),  # IV
    Chord.from_scale_degrees([4, 6, 8], scale),  # V
    Chord.from_scale_degrees([0, 2, 4], scale),  # I
]

# I-V-vi-IV (pop)
chords = [
    Chord.major('C'),   # I
    Chord.major('G'),   # V
    Chord.minor('A'),   # vi
    Chord.major('F'),   # IV
]

# ii-V-I (jazz)
chords = [
    Chord.minor7('D'),    # ii7
    Chord.dominant7('G'), # V7
    Chord.major7('C'),    # Imaj7
]

# i-VII-VI-VII (minor)
scale_minor = Scale.minor('A')
chords = [
    Chord.minor('A'),   # i
    Chord.major('G'),   # VII
    Chord.major('F'),   # VI
    Chord.major('G'),   # VII
]
```

### Playing Chords

```python
# As pad (all notes together, long duration)
def play_chord_pad(song, chord, start, duration, track):
    freqs = chord.get_frequencies()
    for freq in freqs:
        song.play_note(freq, duration, start, track)

# As arpeggio (notes one by one)
def play_chord_arp(song, chord, start, note_duration, track):
    freqs = chord.get_frequencies()
    for i, freq in enumerate(freqs):
        song.play_note(freq, note_duration, start + (i * note_duration), track)

# Example usage:
for i, chord in enumerate(chords):
    time = i * 2.0  # 2 seconds per chord
    play_chord_pad(song, chord, time, 2.0, 'pad')
```

## Song Structure Templates

### Pop Song Structure (3-4 minutes)

```python
TEMPO = 120
song = Composition(tempo=TEMPO)

# Setup tracks...

# Structure
intro =     bars(0)   # 0-8 seconds (4 bars)
verse1 =    bars(4)   # 8-24 seconds (8 bars)
chorus1 =   bars(12)  # 24-32 seconds (4 bars)
verse2 =    bars(16)  # 32-48 seconds (8 bars)
chorus2 =   bars(24)  # 48-56 seconds (4 bars)
bridge =    bars(28)  # 56-64 seconds (4 bars)
chorus3 =   bars(32)  # 64-72 seconds (4 bars)
outro =     bars(36)  # 72-80 seconds (4 bars)

# Add patterns at each section...
```

### Electronic Track Structure

```python
# Build-up structure
intro =      bars(0)   # Minimal
buildup1 =   bars(8)   # Add bass
buildup2 =   bars(16)  # Add melody
drop =       bars(32)  # Full energy
breakdown =  bars(48)  # Strip down
buildup3 =   bars(56)  # Build again
drop2 =      bars(64)  # Full again
outro =      bars(80)  # Wind down
```

## Arrangement Tips

### Layer by Frequency

```
High:     Melody, hi-hats, cymbals
Mid-high: Chords, snare
Mid:      Rhythm guitar, vocals
Mid-low:  Bass melody, toms
Low:      Bass root, kick
```

### Build Energy

Start minimal, add layers:
1. Drums only
2. + Bass
3. + Chords/pad
4. + Melody
5. Full arrangement

### Standard Track Count

- Minimal: 3-5 tracks (drums, bass, melody)
- Normal: 6-10 tracks
- Full: 12-16 tracks
- More than 20 gets cluttered

### Typical Mix Levels

```python
# Volume (0.0 - 1.0)
kick.set_volume(1.0)       # Loudest
bass.set_volume(0.9)       # Almost as loud
snare.set_volume(0.8)      # Loud
melody.set_volume(0.7)     # Medium-loud
chords.set_volume(0.5)     # Medium
pad.set_volume(0.4)        # Quiet
hihat.set_volume(0.6)      # Medium
```

### Typical Panning

```python
# Pan (-1.0 to 1.0, -1=left, 0=center, 1=right)
kick.set_pan(0.0)          # Center
bass.set_pan(0.0)          # Center
snare.set_pan(0.0)         # Center
melody.set_pan(-0.2)       # Slight left
harmony.set_pan(0.2)       # Slight right
hihat.set_pan(0.3)         # Right
pad.set_pan(0.0)           # Center (stereo width from reverb)
```

## Effects Quick Settings

### For Melody

```python
melody.add_effect(ReverbFX(mix=0.2, room_size=0.5))
melody.add_effect(DelayFX(delay_time=0.375, feedback=0.3, mix=0.2))
```

### For Bass

```python
bass.add_effect(Compressor(threshold=-15, ratio=4.0))
bass.add_effect(DistortionFX(drive=2.0, tone=0.4, mix=0.3))
```

### For Pads

```python
pad.add_effect(ChorusFX(rate=0.3, depth=0.4, mix=0.5))
pad.add_effect(ReverbFX(mix=0.6, room_size=0.8, damping=0.4))
```

### For Drums

```python
# Kick - compression only
kick.add_effect(Compressor(threshold=-10, ratio=3.0))

# Snare - reverb for room
snare.add_effect(ReverbFX(mix=0.2, room_size=0.3))

# Hi-hat - subtle effects
hihat.add_effect(PhaserFX(rate=0.2, depth=0.3, mix=0.3))
```

### Master Chain

```python
# Always last
song.add_master_effect(Compressor(threshold=-10, ratio=2.0))
song.add_master_effect(Limiter(threshold=-1))
```

## Common Mistakes to Avoid

1. **Too many tracks** - Keep it simple, 8-12 is plenty
2. **Everything at full volume** - Leave headroom, not everything should be 1.0
3. **No low end** - Don't forget bass
4. **No high end** - Add brightness with hi-hats or high melody
5. **Clashing frequencies** - Use EQ to separate bass and kick
6. **Too much reverb** - Keep mix under 0.4 for most things
7. **No dynamics** - Vary volume and intensity throughout song
8. **Ignoring song structure** - Songs need verses, choruses, breaks
9. **All notes on grid** - Add slight timing variations for human feel
10. **No silence** - Rests and breaks are important

## Workflow Checklist

1. **Set tempo** - Pick BPM and stick to it
2. **Choose key** - Pick a scale and use it
3. **Drums first** - Get the rhythm foundation
4. **Add bass** - Lock with drums
5. **Add chords** - Harmonic foundation
6. **Add melody** - The hook
7. **Arrange** - Structure with intro/verse/chorus
8. **Mix** - Set volumes and panning
9. **Effects** - Add reverb, delay, compression
10. **Master** - Final compression and limiting
11. **Export** - Render to file

## Quick Example: Full Song

```python
from algorythm import Composition, SynthPresets, Scale, Motif

# Setup
TEMPO = 120
song = Composition(tempo=TEMPO)

# Tracks
kick = song.add_track('kick', SynthPresets.kick())
snare = song.add_track('snare', SynthPresets.snare())
bass = song.add_track('bass', SynthPresets.synth_bass())
melody = song.add_track('melody', SynthPresets.pluck())

# Scale
scale = Scale.minor('A')

# Patterns
bass_line = Motif.from_intervals([0, 0, 4, 4], scale=scale, octave=2, duration=0.5)
melody_line = Motif.from_intervals([0, 2, 3, 5, 7], scale=scale, duration=0.5)

# Helper
def bars(n):
    return (n * 4.0 * 60.0) / TEMPO

# Drums
def add_drums(start, bars_count):
    for bar in range(bars_count):
        time = start + (bar * 2.0)
        song.play_note(60, 0.1, time + 0.0, 'kick')
        song.play_note(60, 0.1, time + 1.0, 'kick')
        song.play_note(60, 0.1, time + 0.5, 'snare')
        song.play_note(60, 0.1, time + 1.5, 'snare')

# Arrange
add_drums(bars(0), 16)  # Drums throughout

song.play_motif(bass_line, start=bars(4), track='bass')  # Bass at bar 4
song.play_motif(melody_line, start=bars(8), track='melody')  # Melody at bar 8

# Effects
melody.add_effect(ReverbFX(mix=0.2))
bass.add_effect(Compressor(threshold=-15, ratio=3.0))

# Master
song.add_master_effect(Limiter(threshold=-1))

# Render
song.render('my_song.wav')
```

## Keep This In Mind

- **Tempo** dictates everything - calculate bar length first
- **Scale** keeps you in key - stick to one for simplicity
- **Less is more** - Simple patterns repeated are better than complex chaos
- **Test often** - Render and listen frequently
- **Copy what works** - Reference songs you like and recreate their structure
