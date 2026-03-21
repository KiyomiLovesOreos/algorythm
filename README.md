# Algorythm

A Python library for making music with code.

## What is it?

Algorythm lets you create music by writing Python. You can synthesize sounds, build melodies, add effects, and export your tracks to audio files. It's designed to be simple enough for beginners but flexible enough for complex compositions.

## Install

Clone this repo and install:

```bash
pip install -e .
```

## Quick Start

Here's how to make your first sound:

```python
from algorythm import Synth, Exporter

# Make a synth and play a note
synth = Synth(waveform='sine')
audio = synth.generate_note(frequency=440, duration=1.0)

# Save it
exporter = Exporter()
exporter.export(audio, 'my_sound.wav')
```

Want to make something more musical? Use the composition tools:

```python
from algorythm import Composition, SynthPresets, Scale, Motif

# Pick an instrument preset
instrument = SynthPresets.pluck()

# Create a melody in C major
melody = Motif.from_intervals([0, 2, 4, 5, 7], scale=Scale.major('C'))

# Build a composition
song = Composition(tempo=120)
song.add_track('melody', instrument)
song.play_motif(melody, start=0.0)
song.render('my_song.wav')
```

## What can you do with it?

### Synthesis
- Multiple synthesis engines (basic oscillators, FM, wavetable, physical modeling, additive)
- 50+ instrument presets ready to use
- Build custom instruments from scratch

### Sequencing
- Create melodies with motifs and scales
- Build rhythms and drum patterns
- Arpeggiate chords automatically
- Support for microtonal tunings

### Effects
- Time-based: reverb, delay, chorus, flanger, phaser
- Dynamics: compression, limiting, gating
- Distortion: overdrive, fuzz, distortion
- Modulation: tremolo, vibrato, auto-pan, ring modulation
- Creative: stutter, freeze, beat repeat, bit crushing

### Composition
- Multi-track arrangements
- Tempo and time signature control
- Parameter automation over time
- Effect chains per track

### Generative Music
- L-Systems for melodic generation
- Cellular automata for rhythm patterns
- Constraint-based composition
- Genetic algorithm improvisation

### Audio Processing
- Load and process existing audio files
- Sample-based synthesis
- Granular synthesis
- Data sonification (turn data into sound)

### Visualization
- Render audio to video with visualizations
- Waveform, spectrogram, frequency scope
- Oscilloscope and piano roll views
- Multiple visualizer styles

### Export
- WAV, MP3, FLAC formats
- Video export with audio visualization
- Optimized streaming renderer for video

## Documentation

Check the docs folder for detailed guides:

- `GETTING_STARTED.md` - Your first steps with Algorythm
- `SYNTHESIS.md` - How synthesis works and available instruments
- `SEQUENCING.md` - Creating melodies, rhythms, and patterns
- `EFFECTS.md` - Complete effects reference
- `COMPOSITION.md` - Building complete tracks
- `GENERATIVE.md` - Algorithmic composition techniques
- `VISUALIZATION.md` - Creating audio visualizations
- `API_REFERENCE.md` - Complete API documentation

## Examples

The `examples/` folder has working examples:

- `01_basic_melodies.py` - Simple note and melody creation
- `02_filters_and_effects.py` - Using filters and effects
- `03_video_visualizations.py` - Creating video from audio
- `04_generative_music.py` - Algorithmic composition
- `05_export_formats.py` - Exporting to different formats

Run any example:

```bash
python examples/01_basic_melodies.py
```

## Requirements

- Python 3.7+
- NumPy
- PyDub (for MP3/format conversion)
- PyYAML

Optional for video:
- moviepy
- Pillow

Optional for playback:
- sounddevice

## License

See LICENSE file.

## Contributing

This is a personal project but if you want to contribute, feel free to open an issue or PR.
