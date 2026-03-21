# Getting Started with Algorythm

This guide will walk you through the basics of making music with Algorythm.

## Installation

First, make sure you have Python 3.7 or newer. Then install Algorythm:

```bash
cd algorythm
pip install -e .
```

This installs the core library with NumPy, PyDub, and PyYAML.

### Optional Dependencies

For video visualization:
```bash
pip install moviepy pillow
```

For audio playback:
```bash
pip install sounddevice
```

## Your First Sound

Let's generate a simple tone:

```python
from algorythm import Synth, Exporter

# Create a synthesizer
synth = Synth(waveform='sine')

# Generate a 440Hz tone (A4) for 1 second
audio = synth.generate_note(frequency=440, duration=1.0)

# Export to WAV file
exporter = Exporter()
exporter.export(audio, 'my_first_sound.wav')
```

Run this script and you'll get a pure sine wave tone.

## Different Waveforms

Try different waveform types to hear how they sound:

```python
from algorythm import Synth, Exporter
import numpy as np

exporter = Exporter()

# Try each waveform
for waveform in ['sine', 'square', 'saw', 'triangle']:
    synth = Synth(waveform=waveform)
    audio = synth.generate_note(440, 1.0)
    exporter.export(audio, f'tone_{waveform}.wav')
```

- `sine` - smooth, pure tone
- `square` - harsh, hollow sound
- `saw` - bright, buzzy sound
- `triangle` - softer than square, still has character

## Making a Melody

Instead of manually specifying frequencies, use scales and motifs:

```python
from algorythm import Synth, Scale, Motif, Exporter
import numpy as np

# Create instrument
synth = Synth(waveform='sine')

# Define a scale
scale = Scale.major('C')  # C major scale

# Create a motif (melody pattern)
melody = Motif.from_intervals(
    intervals=[0, 2, 4, 5, 7, 5, 4, 2],  # Scale degrees
    scale=scale,
    duration=0.5  # Each note is 0.5 seconds
)

# Generate the audio
notes = []
for note in melody.notes:
    audio = synth.generate_note(note['frequency'], note['duration'])
    notes.append(audio)

# Concatenate all notes
full_melody = np.concatenate(notes)

# Export
exporter = Exporter()
exporter.export(full_melody, 'my_melody.wav')
```

## Using Presets

Instead of building synths from scratch, use presets:

```python
from algorythm import SynthPresets, Composition, Scale, Motif

# Use a preset instrument
instrument = SynthPresets.pluck()

# Create a melody
melody = Motif.from_intervals([0, 2, 4, 5, 7], scale=Scale.major('C'))

# Make a composition
song = Composition(tempo=120)
song.add_track('melody', instrument)
song.play_motif(melody, start=0.0)
song.render('preset_song.wav')
```

Available preset categories:
- Synth: `synth_lead()`, `synth_pad()`, `synth_bass()`
- Plucked: `pluck()`, `harp()`, `guitar()`
- Keys: `piano()`, `electric_piano()`, `bell()`
- Brass: `brass()`, `trumpet()`
- Strings: `strings()`, `violin()`
- Drums: `kick()`, `snare()`, `hihat()`, `clap()`

See `SYNTHESIS.md` for the complete list.

## Adding Effects

Make your sound more interesting with effects:

```python
from algorythm import Composition, SynthPresets, Scale, Motif, ReverbFX, DelayFX

instrument = SynthPresets.pluck()
melody = Motif.from_intervals([0, 2, 4, 5], scale=Scale.major('C'))

song = Composition(tempo=120)
track = song.add_track('melody', instrument)

# Add effects to the track
track.add_effect(ReverbFX(mix=0.3, room_size=0.7))
track.add_effect(DelayFX(delay_time=0.5, feedback=0.3, mix=0.2))

song.play_motif(melody, start=0.0)
song.render('with_effects.wav')
```

## Multi-Track Composition

Build a complete song with multiple instruments:

```python
from algorythm import Composition, SynthPresets, Scale, Motif

song = Composition(tempo=120)

# Add multiple tracks
melody_track = song.add_track('melody', SynthPresets.pluck())
bass_track = song.add_track('bass', SynthPresets.synth_bass())
drums_track = song.add_track('drums', SynthPresets.kick())

# Create patterns
scale = Scale.major('C')
melody = Motif.from_intervals([0, 2, 4, 5, 7], scale=scale)
bass = Motif.from_intervals([0, 0, 0, 0], scale=scale, octave=2)

# Arrange the song
song.play_motif(melody, start=0.0, track='melody')
song.play_motif(bass, start=0.0, track='bass')

# Render
song.render('full_song.wav')
```

## What's Next?

Now that you know the basics, explore:

- `SYNTHESIS.md` - Learn about synthesis engines and presets
- `SEQUENCING.md` - Advanced melody and rhythm techniques
- `EFFECTS.md` - All available effects and how to use them
- `COMPOSITION.md` - Building complex arrangements
- `GENERATIVE.md` - Algorithmic composition

Or check out the example scripts in the `examples/` folder.
