# Sequencing Guide

Learn how to create melodies, rhythms, and musical patterns.

## Scales

Scales define which notes are available in a key.

### Creating Scales

```python
from algorythm import Scale

# Major scales
c_major = Scale.major('C')
d_major = Scale.major('D')

# Minor scales
a_minor = Scale.minor('A')
e_minor = Scale.minor('E')

# Other modes
d_dorian = Scale.dorian('D')
e_phrygian = Scale.phrygian('E')
f_lydian = Scale.lydian('F')
g_mixolydian = Scale.mixolydian('G')
b_locrian = Scale.locrian('B')

# Exotic scales
harmonic_minor = Scale.harmonic_minor('A')
melodic_minor = Scale.melodic_minor('A')
pentatonic = Scale.pentatonic('C')
blues = Scale.blues('C')
```

### Using Scales

Get frequencies for scale degrees:

```python
scale = Scale.major('C')

# Get the root note (C)
root = scale.get_frequency(0)

# Get the third (E)
third = scale.get_frequency(2)

# Get octaves higher
high_c = scale.get_frequency(7)  # One octave up
very_high_c = scale.get_frequency(14)  # Two octaves up

# Negative numbers go down
low_c = scale.get_frequency(-7)  # One octave down
```

## Motifs

Motifs are melodic patterns.

### Creating Motifs

```python
from algorythm import Motif, Scale

scale = Scale.major('C')

# From scale intervals (most common)
melody = Motif.from_intervals(
    intervals=[0, 2, 4, 5, 7, 5, 4, 2],  # Scale degrees
    scale=scale,
    duration=0.5,  # Each note length
    octave=4       # Starting octave
)

# From absolute frequencies
melody = Motif.from_frequencies(
    frequencies=[261.63, 293.66, 329.63, 349.23],  # Hz
    durations=[0.5, 0.5, 0.5, 1.0]  # Individual durations
)

# From note names
melody = Motif.from_notes(
    notes=['C4', 'D4', 'E4', 'F4', 'G4'],
    duration=0.5
)
```

### Motif Transformations

Transform motifs musically:

```python
melody = Motif.from_intervals([0, 2, 4, 5], scale=Scale.major('C'))

# Transpose up or down
higher = melody.transpose(2)  # 2 scale degrees up
lower = melody.transpose(-3)  # 3 scale degrees down

# Reverse the melody
backwards = melody.retrograde()

# Invert intervals
inverted = melody.invert()

# Change speed
faster = melody.augment(0.5)  # Half duration (2x speed)
slower = melody.augment(2.0)  # Double duration (half speed)

# Repeat
repeated = melody.repeat(4)  # Play 4 times
```

### Accessing Notes

```python
melody = Motif.from_intervals([0, 2, 4], scale=Scale.major('C'))

# Access individual notes
for note in melody.notes:
    print(f"Frequency: {note['frequency']} Hz")
    print(f"Duration: {note['duration']} seconds")
```

## Rhythms

Rhythms define timing patterns without pitch.

### Creating Rhythms

```python
from algorythm import Rhythm

# From durations (in beats)
rhythm = Rhythm.from_durations([1, 1, 0.5, 0.5, 2])

# Common patterns
quarter_notes = Rhythm.from_durations([1, 1, 1, 1])
eighth_notes = Rhythm.from_durations([0.5] * 8)
syncopated = Rhythm.from_durations([0.75, 0.25, 1, 0.5, 0.5])

# From string notation (x = hit, . = rest)
rhythm = Rhythm.from_pattern('x..x.x..')

# Euclidean rhythms (evenly distributed hits)
rhythm = Rhythm.euclidean(hits=5, length=8)  # 5 hits in 8 steps
```

### Rhythm Operations

```python
rhythm = Rhythm.from_durations([1, 0.5, 0.5])

# Repeat
repeated = rhythm.repeat(4)

# Reverse
backwards = rhythm.reverse()

# Combine rhythms
combined = rhythm1 + rhythm2
```

## Chords

Chords are multiple notes played together.

### Creating Chords

```python
from algorythm import Chord, Scale

scale = Scale.major('C')

# Common chords
c_major = Chord.major('C')
a_minor = Chord.minor('A')
g7 = Chord.dominant7('G')
dm7 = Chord.minor7('D')

# From scale degrees
chord = Chord.from_scale_degrees([0, 2, 4], scale)  # I chord (C-E-G)

# From intervals
chord = Chord.from_intervals([0, 4, 7])  # Major triad in semitones
```

### Chord Types

Available chord types:
- `major(root)` - Major triad
- `minor(root)` - Minor triad
- `diminished(root)` - Diminished triad
- `augmented(root)` - Augmented triad
- `major7(root)` - Major 7th
- `minor7(root)` - Minor 7th
- `dominant7(root)` - Dominant 7th
- `diminished7(root)` - Diminished 7th

### Using Chords

```python
chord = Chord.major('C')

# Get all frequencies
freqs = chord.get_frequencies()

# Play as arpeggio (notes one after another)
arpeggio = chord.arpeggiate(duration=0.25)

# Play as block chord (all together)
# Generate each note and mix them
```

## Arpeggios

Arpeggios play chord notes in sequence.

### Creating Arpeggios

```python
from algorythm import Arpeggiator, Chord

chord = Chord.major('C')

# Basic arpeggio
arp = Arpeggiator(chord=chord)
pattern = arp.generate(
    pattern='up',           # Direction
    duration=0.25,          # Note length
    num_repetitions=2       # Times through chord
)

# Pattern types
patterns = [
    'up',           # Bottom to top
    'down',         # Top to bottom
    'up_down',      # Up then down
    'down_up',      # Down then up
    'random'        # Random order
]
```

### Advanced Arpeggios

```python
# Custom pattern (indices into chord)
arp.generate(
    pattern=[0, 2, 1, 2],   # Custom sequence
    duration=0.25
)

# With octave range
arp.generate(
    pattern='up',
    octaves=2,              # Span 2 octaves
    duration=0.25
)
```

## Microtonal Tunings

Go beyond 12-tone equal temperament.

### Using Different Tunings

```python
from algorythm import Tuning, Scale

# 19-tone equal temperament
tuning_19 = Tuning('19-TET')
scale = Scale.major('C', tuning=tuning_19)

# 24-tone (quarter tones)
tuning_24 = Tuning('24-TET')

# Just intonation
just = Tuning.just_intonation()

# Pythagorean tuning
pyth = Tuning.pythagorean()

# Custom tuning (specify cents for each degree)
custom = Tuning(tuning_system=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100])
```

### Equal Temperament Systems

```python
# Create any equal division of the octave
tuning_31 = Tuning.equal_temperament(31)  # 31-TET
tuning_53 = Tuning.equal_temperament(53)  # 53-TET
```

## Practical Examples

### Simple Melody

```python
from algorythm import Scale, Motif, Synth, Exporter
import numpy as np

scale = Scale.major('C')
melody = Motif.from_intervals([0, 2, 4, 5, 7, 5, 4, 2], scale=scale, duration=0.5)

synth = Synth(waveform='sine')
notes = []
for note in melody.notes:
    audio = synth.generate_note(note['frequency'], note['duration'])
    notes.append(audio)

full_audio = np.concatenate(notes)
Exporter().export(full_audio, 'melody.wav')
```

### Arpeggio with Bass

```python
from algorythm import Composition, SynthPresets, Chord, Arpeggiator, Motif, Scale

song = Composition(tempo=120)
song.add_track('arp', SynthPresets.pluck())
song.add_track('bass', SynthPresets.synth_bass())

# Arpeggio
chord = Chord.major('C')
arp = Arpeggiator(chord=chord)
arp_pattern = arp.generate('up_down', duration=0.25, num_repetitions=4)

# Bass line
scale = Scale.major('C')
bass = Motif.from_intervals([0, 0, 0, 0], scale=scale, octave=2, duration=1.0)

song.play_motif(arp_pattern, start=0.0, track='arp')
song.play_motif(bass, start=0.0, track='bass')

song.render('arp_with_bass.wav')
```

### Rhythmic Pattern

```python
from algorythm import Rhythm, Composition, SynthPresets

# Create a rhythm
rhythm = Rhythm.euclidean(hits=5, length=8)

song = Composition(tempo=120)
song.add_track('drums', SynthPresets.kick())

# Play the rhythm
for i, hit in enumerate(rhythm.pattern):
    if hit:  # If there's a hit at this position
        time = i * 0.5  # Each step is 0.5 seconds
        # Play a note at this time
        # (Implementation depends on your composition structure)

song.render('rhythm.wav')
```

### Chord Progression

```python
from algorythm import Chord, Scale, Motif, Composition, SynthPresets

scale = Scale.major('C')
song = Composition(tempo=90)
song.add_track('chords', SynthPresets.warm_pad())

# I - IV - V - I progression in C major
chords = [
    Chord.major('C'),
    Chord.major('F'),
    Chord.major('G'),
    Chord.major('C')
]

# Play each chord as a motif
time = 0.0
for chord in chords:
    # Convert chord to motif
    freqs = chord.get_frequencies()
    motif = Motif.from_frequencies(freqs, durations=[2.0] * len(freqs))
    song.play_motif(motif, start=time, track='chords')
    time += 2.0

song.render('chord_progression.wav')
```

## Tips

1. Start with simple intervals before complex melodies
2. Use scales to keep melodies in key
3. Euclidean rhythms create interesting patterns
4. Combine motif transformations for variation
5. Arpeggios add movement to chords
6. Experiment with microtonal tunings for unique sounds
