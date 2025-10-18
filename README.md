# Algorythm

**A Python Library for Algorithmic Music**

Designed to take code as an input and give audio as an output, like the main animation library for Python by 3Blue1Brown (Manim).

## Core Philosophy (Manim-Inspired)

The library is **declarative** and uses **Object-Oriented Composition**. Users describe what the music should be, not how to generate every sample.

- **Composition by Chaining**: Define elements (Synths, Motifs, Structures) and chain them together to form the final piece
- **Time as a Variable**: Time, duration, and tempo are handled by the library engine, abstracting away complex sample rate math

## Installation

```bash
pip install -e .
```

For additional export format support (FLAC, MP3, OGG):
```bash
pip install -e ".[export]"
```

## Library Structure

| Module | Purpose | Key Classes/Concepts |
|--------|---------|---------------------|
| `algorythm.synth` | Defines sound sources and timbres | `Oscillator`, `Filter`, `ADSR`, `SynthPresets`, `Synth` |
| `algorythm.sequence` | Handles rhythmic and melodic patterns | `Motif`, `Rhythm`, `Arpeggiator`, `Scale` |
| `algorythm.structure` | Arranges and composes the final track | `Track`, `Composition`, `EffectChain`, `Reverb` |
| `algorythm.export` | Renders and saves the final audio | `RenderEngine`, `Exporter` |

## Quick Start

### Basic Synthesis

Define a custom instrument (a Synth) once, then reference it repeatedly:

```python
from algorythm.synth import Synth, Filter, ADSR

# Create a warm synth sound
warm_pad = Synth(
    waveform='saw',
    filter=Filter.lowpass(cutoff=2000, resonance=0.6),
    envelope=ADSR(attack=1.5, decay=0.5, sustain=0.8, release=2.0)
)
```

### Sequencing and Composition

Define a musical idea (Motif) and apply it to a structured timeline (Track):

```python
from algorythm.sequence import Motif, Scale
from algorythm.structure import Composition, Reverb

# Create a simple, rising motif
melody = Motif.from_intervals([0, 2, 4, 7], scale=Scale.major('C'))

# Create the main composition structure
final_track = Composition(tempo=120) \
    .add_track('Bassline', warm_pad) \
    .repeat_motif(melody, bars=8) \
    .transpose(semitones=5) \
    .add_fx(Reverb(mix=0.4))
```

### Rendering and Export

The render function handles the heavy lifting, generating audio samples:

```python
# Render the final output
final_track.render(
    file_path='epic_track.wav',
    quality='high',
    formats=['wav']  # Supports ['flac', 'mp3', 'ogg'] with additional dependencies
)
```

## Examples

The `examples/` directory contains several demonstration scripts:

- `basic_synthesis.py` - Simple synth sound generation
- `composition_example.py` - Full composition workflow (as shown in the problem statement)
- `advanced_example.py` - Multiple tracks, effects, and transformations

Run an example:
```bash
cd examples
python composition_example.py
```

## API Reference

### Synth Module

**Oscillator**: Generates basic waveforms (sine, square, saw, triangle)
```python
from algorythm.synth import Oscillator
osc = Oscillator(waveform='sine', frequency=440.0, amplitude=1.0)
```

**Filter**: Applies frequency filtering (lowpass, highpass, bandpass, notch)
```python
from algorythm.synth import Filter
lpf = Filter.lowpass(cutoff=1000, resonance=0.5)
```

**ADSR**: Attack, Decay, Sustain, Release envelope
```python
from algorythm.synth import ADSR
envelope = ADSR(attack=0.1, decay=0.2, sustain=0.7, release=0.3)
```

**Synth**: Main synthesizer combining oscillator, filter, and envelope
```python
from algorythm.synth import Synth, Filter, ADSR
synth = Synth(waveform='saw', filter=Filter.lowpass(2000), envelope=ADSR())
```

**SynthPresets**: Pre-configured synth sounds
```python
from algorythm.synth import SynthPresets
warm_pad = SynthPresets.warm_pad()
pluck = SynthPresets.pluck()
bass = SynthPresets.bass()
```

### Sequence Module

**Scale**: Musical scale definitions
```python
from algorythm.sequence import Scale
c_major = Scale.major('C', octave=4)
a_minor = Scale.minor('A', octave=3)
```

**Motif**: Musical motif with intervals and durations
```python
from algorythm.sequence import Motif, Scale
melody = Motif.from_intervals([0, 2, 4, 7], scale=Scale.major('C'))
transposed = melody.transpose(semitones=5)
reversed_melody = melody.reverse()
inverted = melody.invert()
```

**Arpeggiator**: Generates arpeggios from motifs
```python
from algorythm.sequence import Arpeggiator
arp = Arpeggiator(pattern='up-down', octaves=2)
arpeggiated = arp.arpeggiate(melody)
```

### Structure Module

**Composition**: Main composition container
```python
from algorythm.structure import Composition
comp = Composition(tempo=120)
comp.add_track('Lead', synth).repeat_motif(melody, bars=4)
```

**Effects**: Audio effects (Reverb, Delay, Chorus)
```python
from algorythm.structure import Reverb, Delay
reverb = Reverb(mix=0.3, room_size=0.5)
delay = Delay(delay_time=0.5, feedback=0.3, mix=0.3)
```

### Export Module

**Exporter**: Exports audio to various formats
```python
from algorythm.export import Exporter
exporter = Exporter()
exporter.export(audio_signal, 'output.wav', sample_rate=44100, quality='high')
```

## Technical Considerations

- **Audio Backend**: Pure Python/NumPy implementation for simplicity and portability
  - For production use, consider binding to native C++ audio libraries (JUCE, RTCMix) via pyo3 or ctypes
- **Export Formats**: 
  - WAV export is built-in
  - FLAC, MP3, and OGG require additional dependencies (install with `pip install -e ".[export]"`)
  - Uses Python wrappers for ffmpeg or soundfile for reliable multi-format encoding

## Development

Install development dependencies:
```bash
pip install -e ".[dev]"
```

Run tests:
```bash
pytest
```

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
