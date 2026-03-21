# Music Theory for Algorythm

Essential music theory concepts for composing with Algorythm. You don't need to know all this, but it helps.

## Basic Concepts

### Pitch and Frequency

- **Pitch**: How high or low a note sounds
- **Frequency**: Measured in Hz (cycles per second)
- **A4** (concert pitch): 440 Hz (standard reference)
- **Octave**: Doubling or halving frequency (e.g., A3 = 220 Hz, A5 = 880 Hz)

### Notes

12 notes in Western music (semitones apart):
```
C  C# D  D# E  F  F# G  G# A  A# B
0  1  2  3  4  5  6  7  8  9  10 11
```

Notes repeat in octaves. Each note can have # (sharp, up 1 semitone) or b (flat, down 1 semitone).

### Intervals

Distance between two notes (in semitones):
```
Semitones | Name           | Quality
0         | Unison         | Perfect
1         | Minor second   | Minor
2         | Major second   | Major
3         | Minor third    | Minor
4         | Major third    | Major
5         | Perfect fourth | Perfect
6         | Tritone        | Diminished
7         | Perfect fifth  | Perfect
8         | Minor sixth    | Minor
9         | Major sixth    | Major
10        | Minor seventh  | Minor
11        | Major seventh  | Major
12        | Octave         | Perfect
```

In Algorythm terms (scale degrees in major scale):
```
Degree | Interval Name
0      | Root/Unison
1      | Major second (whole step)
2      | Major third
3      | Perfect fourth
4      | Perfect fifth
5      | Major sixth
6      | Major seventh
7      | Octave
```

## Scales

### Major Scale Pattern

```
C  D  E  F  G  A  B  C
0  2  4  5  7  9  11 12

Pattern: 2-2-1-2-2-2-1 semitones apart
```

All major scales follow this pattern. C major has no sharps or flats.

### Minor Scale (Natural Minor)

```
A  B  C  D  E  F  G  A
0  2  3  5  7  8  10 12

Pattern: 2-1-2-2-1-2-2 semitones apart
```

Relative to major: A minor = C major (same notes, different root).

### Harmonic Minor

```
A  B  C  D  E  F  G# A

Pattern: 2-1-2-2-1-3-1 semitones apart
```

Raised 7th degree. Used for dramatic minor sound.

### Melodic Minor

```
A  B  C  D  E  F# G# A (ascending)
A  G  F  E  D  C  B  A (descending)

Pattern: 2-1-2-2-2-2-1 (up), 2-1-2-2-1-2-2 (down)
```

Raised 6th and 7th going up, natural minor going down. Used in classical music.

### Pentatonic Scale

```
C  D  E  G  A  (C)

Pattern: 2-2-3-2-3 semitones apart
```

5 notes per octave. Sounds "Asian" or "folk-like". Hard to make sound bad.

### Blues Scale

```
C  Eb F  Gb G  Bb (C)

Pattern: 3-2-1-1-3-2 semitones apart
```

Major pentatonic + flat 5th. Essential for blues, jazz, rock.

## Chord Theory

### Chord Construction

Built by stacking thirds (every other note in a scale).

```
Triad (3-note chord):
Root + Major third (4 semitones) + Perfect fifth (7 semitones from root)

7th Chord (4-note chord):
Root + Major third + Perfect fifth + Major seventh (11 semitones from root)
```

### Major Chord

```
C major:    C  E  G
Intervals:  0  4  7 (semitones from root)
In scale:   0  2  4 (scale degrees in C major)
```

Sound: Bright, happy, resolved.

### Minor Chord

```
A minor:    A  C  E
Intervals:  0  3  7 (semitones from root)
In scale:   0  2  4 (scale degrees in A minor)
```

Sound: Dark, sad, resolved.

### Dominant 7th

```
G7:         G  B  D  F
Intervals:  0  4  7  10 (semitones from root)
```

Sound: Unresolved tension, pulls to next chord. Creates movement.

### Minor 7th

```
A minor 7:  A  C  E  G
Intervals:  0  3  7  10
```

Sound: Laid-back, jazzy, dark but not tense.

### Major 7th

```
C major 7:  C  E  G  B
Intervals:  0  4  7  11
```

Sound: Sophisticated, jazzy, open.

## Chord Progressions

### Why Progressions Matter

Chords create harmonic movement. The ear expects certain progressions. Breaking expectations creates interest.

### Roman Numeral Notation

In any key, chords are numbered by position:
```
In C major:
I    = C major    (1-3-5)
ii   = D minor    (2-4-6)
iii  = E minor    (3-5-7)
IV   = F major    (4-6-1)
V    = G major    (5-7-2)
vi   = A minor    (6-1-3)
vii° = B diminished (7-2-4)
```

Uppercase = major, lowercase = minor, ° = diminished.

### Classic Progressions

**I-IV-V-I (and variations)**
```
C - F - G - C (or G-C-F-C, C-G-F-C, etc)
Most common. Used in thousands of songs.
```

**I-V-vi-IV**
```
C - G - Am - F
Pop music standard. Emotional but catchy.
```

**vi-IV-I-V**
```
Am - F - C - G
Moody pop. Often used as loop.
```

**ii-V-I**
```
Dm - G - C
Jazz standard. Creates smooth movement.
```

**I-vi-IV-V**
```
C - Am - F - G
Doo-wop progression. Uplifting.
```

**I-IV-vi-V**
```
C - F - Am - G
Adds tension. Creates drama.
```

### Cadences (Chord Ending Phrases)

**Authentic Cadence (V-I)**
```
G - C
Conclusive. Ends sections and songs.
```

**Plagal Cadence (IV-I)**
```
F - C
"Amen cadence." Gentle ending.
```

**Deceptive Cadence (V-vi)**
```
G - Am
Unexpected. Creates surprise, continues tension.
```

**Half Cadence (I-V)**
```
C - G
Unresolved. Asks a question.
```

## Harmony and Melody Relationship

### Consonance and Dissonance

**Consonant Intervals** (stable, resolved):
- Unison (0 semitones)
- Perfect fourth (5 semitones)
- Perfect fifth (7 semitones)
- Major/minor thirds (3-4 semitones)
- Major/minor sixths (8-9 semitones)
- Octave (12 semitones)

**Dissonant Intervals** (tense, unresolved):
- Minor second (1 semitone) - Very tense
- Major second (2 semitones) - Tense
- Tritone (6 semitones) - Very tense ("devil's interval")
- Minor seventh (10 semitones) - Tense

### Voice Leading

**How to connect chords smoothly:**

1. **Stepwise motion**: Move notes by smallest amount (1-2 semitones)
2. **Common tones**: Keep notes that are in both chords
3. **Avoid parallel fifths**: Don't have two voices both move up/down by 5 semitones
4. **Balance**: Keep voices within reasonable range

Example:
```
C major (C E G) → F major (F A C)
Bad:  C→F, E→A, G→C (big jumps)
Good: C→C, E→F, G→A (smooth, smallest movement)
```

## Melody Composition

### Melodic Contour

Shape of melody creates emotional content:

- **Ascending**: Builds energy, hope
- **Descending**: Releases tension, sadness
- **Arch**: Up then down (natural, satisfying)
- **Undulating**: Up and down repeatedly (interesting, complex)

### Melodic Strategies

**Repetition**: Repeat phrases for catchiness
```
[0 2 4 5 7] [0 2 4 5 7] - Too repetitive
[0 2 4 5 7] [0 2 4 5 9] - Variation in last note
```

**Sequence**: Repeat pattern at different pitch
```
Starting: 0 2 4 5 → (repeat at +2) → 2 4 6 7
```

**Inversion**: Turn melody upside down
```
Original:  0 2 4 5 7
Inverted:  0 -2 -4 -5 -7 (flip intervals)
```

**Augmentation/Diminution**: Change note lengths
```
Original:  0.5 0.5 0.5 1.0 (eighth eighth eighth quarter)
Augmented: 1.0 1.0 1.0 2.0 (double durations)
```

**Call and Response**: Short melody + answer melody
```
Melody 1: [0 2 4 5] (question)
Melody 2: [5 4 2 0] (answer)
```

## Rhythm Concepts

### Meters and Time Signatures

```
4/4 (common time): 4 beats per measure, quarter note = 1 beat
     Used in: pop, rock, blues, hip-hop (most music)

3/4 (waltz time): 3 beats per measure
     Used in: waltzes, country, some pop

2/4 (cut time): 2 beats per measure (fast)
     Used in: marches, polkas

6/8 (compound): 6 eighth notes per measure (feels like 2 groups of 3)
     Used in: folk, some pop
```

### Rhythmic Values

```
At 120 BPM (quarter note = 0.5 seconds):

Whole note:      4.0 seconds  (4 beats)
Half note:       2.0 seconds  (2 beats)
Quarter note:    0.5 seconds  (1 beat)
Eighth note:     0.25 seconds (1/2 beat)
Sixteenth note:  0.125 seconds (1/4 beat)

Dotted notes (1.5x duration):
Dotted half:     3.0 seconds
Dotted quarter:  0.75 seconds
Dotted eighth:   0.375 seconds
```

### Syncopation

Emphasis on off-beats creates rhythm interest:
```
Regular:    X . X . X . X .  (beat 1, 3)
Syncopated: . X . X . X . X  (beat 2, 4)
Off-beat:   . . X . . . X .  (beat 2.5, 4.5)
```

## Tension and Resolution

### Building Tension

1. **Dissonant harmony**: Use V, vii°, or unstable chords
2. **Rhythmic tension**: Syncopation, faster notes
3. **Ascending motion**: Pitch rises
4. **Dynamic increase**: Get louder
5. **Harmonic rhythm**: Change chords faster

### Resolution

1. **Consonant harmony**: Return to I, iv, vi
2. **Descending motion**: Pitch falls
3. **Slower rhythm**: Longer note values
4. **Dynamic decrease**: Get quieter
5. **Harmonic stillness**: Hold on chord longer

## Modulation (Key Changes)

### Types of Modulation

**Pivot Chord Modulation**: Use common chord between keys
```
C major (I) - F major (IV in C) [pivot] - F major (I)
Smooth transition using shared chord
```

**Relative Key Modulation**: Switch to relative minor/major
```
C major → A minor (same notes, different root)
```

**Up a semitone**: Instant key jump
```
C major → C# major (common in pop for energy boost)
```

**V-I Modulation**: Use V of new key to V to new I
```
C major → [G major chord] → D major
```

## Just Intonation vs Equal Temperament

### 12-Tone Equal Temperament (Standard)

All semitones are mathematically equal (ratio 2^(1/12)).

Pros:
- Can play in any key
- Simple, standard
- Transposing is easy

Cons:
- No interval is acoustically pure (except octave)
- Sounds slightly "out of tune" to ears trained in just intonation

### Just Intonation

Intervals based on simple ratios (3:2 for perfect fifth, 5:4 for major third).

Pros:
- Acoustically pure
- Sounds natural, "in tune"
- Chords resonate

Cons:
- Key dependent
- Hard to transpose
- Microtonal tuning needed

## Useful Ratios (Just Intonation)

```
Interval          | Ratio  | Cents
Minor second      | 16:15  | 112
Major second      | 9:8    | 204
Minor third       | 6:5    | 316
Major third       | 5:4    | 386
Perfect fourth    | 4:3    | 498
Perfect fifth     | 3:2    | 702
Minor sixth       | 8:5    | 814
Major sixth       | 5:3    | 884
Minor seventh     | 9:5    | 1018
Major seventh     | 15:8   | 1088
```

Cents: 100 cents = 1 semitone. Useful for tuning systems.

## Tips for Composition

1. **Know what you're breaking**: Learn rules before breaking them
2. **Listen to reference**: Analyze songs you like
3. **Less is more**: Simple harmony is often better
4. **Context matters**: Theory is guidelines, not law
5. **Trust your ears**: If it sounds good, it is good
6. **Experiment**: Try unexpected progressions
7. **Resolve tension**: Don't leave listeners hanging (usually)
8. **Create contrast**: Vary between consonance and dissonance
9. **Use inversions**: First and second inversions add interest
10. **Understand function**: Know why each chord is there
