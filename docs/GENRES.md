# Genre-Specific Guides

Step-by-step recipes for different music styles.

## Electronic Dance Music (EDM)

### Key Characteristics
- Fast BPM (120-140)
- Repetitive, hypnotic patterns
- Heavy bass and kick
- Build-ups and drops
- Synth leads and pads

### Structure
```
Intro:     8 bars - minimal elements
Build 1:   16 bars - add elements
Build 2:   16 bars - more elements
Drop:      16 bars - full energy
Breakdown: 8 bars - strip down
Build 3:   16 bars - rebuild
Drop 2:    16 bars - full again
Outro:     8 bars - wind down
```

### Recipe

```python
from algorythm import Composition, SynthPresets, Scale, Motif
from algorythm import ReverbFX, DelayFX, Compressor, Limiter

def bars(n, tempo=128):
    return (n * 4.0 * 60.0) / tempo

song = Composition(tempo=128)

# Add tracks
kick = song.add_track('kick', SynthPresets.kick())
bass = song.add_track('bass', SynthPresets.acid_bass())
synth = song.add_track('synth', SynthPresets.synth_lead())
pad = song.add_track('pad', SynthPresets.warm_pad())

# Add effects
synth.add_effect(ReverbFX(mix=0.2))
pad.add_effect(ReverbFX(mix=0.5))

# Scale - often minor or pentatonic
scale = Scale.minor('A')

# Patterns
bass_line = Motif.from_intervals([0, 0, 0, 0], scale=scale, octave=2, duration=0.25)
synth_line = Motif.from_intervals([0, 2, 3, 5, 7], scale=scale, duration=0.125)

# Intro - kick only (8 bars)
for bar in range(8):
    time = bar * 2.0
    for beat in range(4):
        song.play_note(60, 0.05, time + (beat * 0.5), 'kick')

# Build 1 - add bass (16 bars)
for bar in range(16):
    time = bars(8) + (bar * 2.0)
    for beat in range(4):
        song.play_note(60, 0.05, time + (beat * 0.5), 'kick')

song.play_motif(bass_line, start=bars(8), track='bass')

# Build 2 - add synth (16 bars)
for bar in range(16):
    time = bars(24) + (bar * 2.0)
    for beat in range(4):
        song.play_note(60, 0.05, time + (beat * 0.5), 'kick')

song.play_motif(bass_line, start=bars(24), track='bass')
song.play_motif(synth_line, start=bars(24), track='synth')

# Drop - add pad (16 bars) - FULL ENERGY
for bar in range(16):
    time = bars(40) + (bar * 2.0)
    for beat in range(4):
        song.play_note(60, 0.05, time + (beat * 0.5), 'kick')

song.play_motif(bass_line, start=bars(40), track='bass')
song.play_motif(synth_line, start=bars(40), track='synth')

pad_notes = Motif.from_intervals([0, 2, 4], scale=scale, duration=4.0)
song.play_motif(pad_notes, start=bars(40), track='pad')

# Master
song.add_master_effect(Compressor(threshold=-10, ratio=2.0))
song.add_master_effect(Limiter(threshold=-1))

song.render('edm_track.wav')
```

## Hip-Hop / Rap

### Key Characteristics
- Slower BPM (80-100)
- Heavy, punchy drums
- Groovy bass with syncopation
- Sparse, off-beat hi-hats
- Loop-based arrangement

### Drum Pattern

```python
# Hi-hop beat signature
for bar in range(bars_count):
    time = bar * 2.0
    
    # Kick pattern (syncopated)
    song.play_note(60, 0.1, time + 0.0, 'kick')
    song.play_note(60, 0.1, time + 0.75, 'kick')
    song.play_note(60, 0.1, time + 1.5, 'kick')
    
    # Snare on 2 and 4
    song.play_note(60, 0.15, time + 0.5, 'snare')
    song.play_note(60, 0.15, time + 1.5, 'snare')
    
    # Hi-hats on offbeats (tight, quiet)
    for i in [1, 3, 5, 7]:
        song.play_note(60, 0.04, time + (i * 0.25), 'hihat')
```

### Key Elements
- Kick and snare create pocket
- Bass sits behind kick
- Chops and stabs on off-beats
- Sample loops are common

## Rock/Alternative

### Key Characteristics
- BPM varies (90-120)
- Live-sounding drums
- Prominent bass and guitars
- Power chords
- Dynamic arrangement with quiet and loud sections

### Drum Pattern

```python
# Rock beat
for bar in range(bars_count):
    time = bar * 2.0
    
    # Kick on 1 and 3
    song.play_note(60, 0.1, time + 0.0, 'kick')
    song.play_note(60, 0.1, time + 1.0, 'kick')
    
    # Snare on 2 and 4
    song.play_note(60, 0.12, time + 0.5, 'snare')
    song.play_note(60, 0.12, time + 1.5, 'snare')
    
    # Hi-hats on eighth notes
    for i in range(8):
        song.play_note(60, 0.05, time + (i * 0.25), 'hihat')
```

### Structure
- Verse: Quiet, sparse
- Pre-chorus: Building tension
- Chorus: Full, loud, all elements
- Bridge: Different feel, buildup
- Final chorus: Biggest energy

## Ambient / Chill

### Key Characteristics
- Slow BPM (60-80)
- Long, sustained notes
- Lots of reverb and space
- Sparse arrangements
- Emphasis on atmosphere over groove

### Recipe

```python
from algorythm import Composition, SynthPresets, Scale
from algorythm import ChorusFX, DelayFX, ReverbFX

song = Composition(tempo=60)  # Slow
pad = song.add_track('pad', SynthPresets.soft_pad())
bass = song.add_track('bass', SynthPresets.bass())

# Very reverby, spacious effects
pad.add_effect(ChorusFX(rate=0.1, depth=0.5, mix=0.5))
pad.add_effect(DelayFX(delay_time=1.0, feedback=0.6, mix=0.3))
pad.add_effect(ReverbFX(mix=0.7, room_size=0.9, damping=0.2))

scale = Scale.major('C')

# Long notes
pad_note = scale.get_frequency(0)
song.play_note(pad_note, 16.0, 0.0, 'pad')  # 16 seconds!

# Bass underneath
song.play_note(60, 16.0, 0.0, 'bass')

song.render('ambient.wav')
```

## Jazz

### Key Characteristics
- Complex chords (7ths, extensions)
- Swing feel (triplet subdivisions)
- Improvisation-based
- Walking bass
- Tight rhythm section

### Chord Progression

```python
from algorythm import Composition, SynthPresets, Chord

song = Composition(tempo=120)
piano = song.add_track('piano', SynthPresets.piano())
bass = song.add_track('bass', SynthPresets.bass())

# Classic ii-V-I progression
chords = [
    Chord.minor7('D'),    # ii7
    Chord.dominant7('G'), # V7
    Chord.major7('C'),    # Imaj7
    Chord.major7('C'),    # Imaj7 (hold)
]

for i, chord in enumerate(chords):
    time = i * 2.0
    freqs = chord.get_frequencies()
    
    # Play chord as block
    for freq in freqs:
        song.play_note(freq, 2.0, time, 'piano')

song.render('jazz.wav')
```

## Lofi Hip-Hop

### Key Characteristics
- Relaxing, lo-fi sound
- Bit-crushed audio
- Vinyl crackle feel
- Pentatonic melodies
- Chill vibes (60-90 BPM)

### Recipe

```python
from algorythm import Composition, SynthPresets, Scale, Motif
from algorythm import BitCrusherFX, DistortionFX, ReverbFX

song = Composition(tempo=85)
sample = song.add_track('sample', SynthPresets.soft_pad())

# Lo-fi effects
sample.add_effect(BitCrusherFX(bit_depth=8, sample_rate=11025))
sample.add_effect(DistortionFX(drive=2.0, tone=0.4, mix=0.2))
sample.add_effect(ReverbFX(mix=0.4, room_size=0.5, damping=0.7))

scale = Scale.pentatonic('C')

# Simple, loopy melody
melody = Motif.from_intervals([0, 2, 4, 5], scale=scale, duration=1.0)

# Loop it 4 times
for loop in range(4):
    song.play_motif(melody, start=loop * 8.0, track='sample')

song.render('lofi.wav')
```

## House / Techno

### Key Characteristics
- Four-on-the-floor kick (4/4)
- 120-130 BPM
- Hypnotic, repetitive patterns
- Emphasis on beat and groove
- Minimal melody

### Kick Pattern

```python
# Four-on-the-floor (kick on every beat)
for bar in range(bars_count):
    time = bar * 2.0
    
    # Kick on every quarter note (4 per bar)
    for beat in range(4):
        song.play_note(60, 0.08, time + (beat * 0.5), 'kick')

# Everything locks to this grid
```

## Classical / Orchestral (Simulated)

### Key Characteristics
- Complex harmonies
- Polyphonic (many voices)
- Dynamics vary greatly
- Longer note values
- Often in major or minor keys

### Recipe

```python
from algorythm import Composition, SynthPresets, Scale, Chord

song = Composition(tempo=90)

# Separate instrumental tracks
violins = song.add_track('violins', SynthPresets.strings())
cello = song.add_track('cello', SynthPresets.cello())
bass = song.add_track('bass', SynthPresets.upright_bass())

scale = Scale.major('C')

# Chord progression
chords = [
    Chord.major('C'),
    Chord.major('G'),
    Chord.major('F'),
    Chord.major('C'),
]

# Play each chord with multiple instruments
for i, chord in enumerate(chords):
    time = i * 3.0
    freqs = chord.get_frequencies()
    
    # Violins: high notes
    high_freq = freqs[-1]
    song.play_note(high_freq, 3.0, time, 'violins')
    
    # Cello: middle notes
    mid_freq = freqs[len(freqs)//2]
    song.play_note(mid_freq, 3.0, time, 'cello')
    
    # Bass: lowest notes
    low_freq = freqs[0]
    song.play_note(low_freq, 3.0, time, 'bass')

song.render('classical.wav')
```

## Synthwave

### Key Characteristics
- Retro 80s vibe
- Synthesizers and digital sounds
- Dark, moody atmosphere
- Steady, driving beat
- Often minor keys, chromatic movements

### Key Elements
- Heavy reverb on vocals/synths
- Driving synth bass
- Pad layers
- Arpeggiated leads
- 80s drum sounds

### Typical Arrangement
- Kick and bass establish vibe
- Pad creates atmosphere
- Synth lead over top
- Heavy reverb on all elements

## Ambient/Generative

### Key Characteristics
- No strict tempo
- Evolving patterns
- Lots of space and silence
- Randomization creates uniqueness
- Meditative quality

### Recipe

```python
import random
from algorythm import Composition, SynthPresets, Scale, Motif

song = Composition(tempo=60)
synth = song.add_track('synth', SynthPresets.soft_pad())

scale = Scale.pentatonic('C')

# Generate random notes
for _ in range(20):
    degree = random.choice([0, 2, 4, 5, 7])
    duration = random.choice([2.0, 3.0, 4.0, 5.0])
    time = sum([2.0, 3.0, 4.0, 5.0][:-1])  # Accumulate times
    
    song.play_note(scale.get_frequency(degree), duration, time, 'synth')

song.render('ambient.wav')
```

## Tips for All Genres

1. **Reference Track**: Find a song you love in the style, analyze its structure
2. **Tempo**: Get the BPM right - makes HUGE difference
3. **Scale**: Pick one and stick to it
4. **Simple Foundation**: Drums + Bass first, then build
5. **Arrangement**: Take at least 50% of song for intro
6. **Mixdown**: Leave headroom, don't max everything at 1.0
7. **Effects**: Use them to create atmosphere, not hide problems
8. **Dynamics**: Vary energy, volume, and intensity throughout
9. **Test**: Listen frequently, take breaks, come back fresh
10. **Reference**: Compare to professional productions in genre
