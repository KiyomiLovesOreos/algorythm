# Live Coding Guide

## Overview

Algorythm provides two interfaces for live coding and interactive music creation:

1. **Terminal DAW (Recommended)** - Terminal-based interface using Textual
2. **Live Coding GUI** - Tkinter-based graphical interface (requires system tk libraries)

## Terminal DAW - `algorythm studio`

### Quick Start

```bash
algorythm studio
```

The Terminal DAW starts in **Live Coding view by default**, providing a full-featured terminal interface for music creation.

### Features

- **Live Coding View** (Default, press `5`) - Python REPL for algorithmic composition
- **Tracker View** (press `1`) - Classic vertical composition
- **Piano Roll View** (press `2`) - Grid-based note editing
- **Arranger View** (press `3`) - Pattern arrangement
- **Mixer View** (press `4`) - Track levels and effects
- **Instrument/FX Editor** (press `6`) - Parameter editing

### Controls

| Key | Action |
|-----|--------|
| `1-6` | Switch between views |
| `Tab` | Next view |
| `Shift+Tab` | Previous view |
| `Ctrl+R` | Run code (in Live Coding view) |
| `Ctrl+P` | Play audio (in Live Coding view) |
| `Ctrl+S` | Save project/audio |
| `Ctrl+O` | Open project |
| `Space` | Play/Stop playback |
| `Q` | Quit |

### Live Coding View Usage

The Live Coding view provides a Python REPL with full access to the Algorythm library:

```python
# Example code in Live Coding view
from algorythm.synth import Synth, ADSR
from algorythm.sequence import Scale, Motif
from algorythm.structure import Composition, Reverb
import numpy as np

# Create composition
comp = Composition(tempo=120)

# Define synth
synth = Synth(
    waveform='saw',
    envelope=ADSR(attack=0.05, decay=0.2, sustain=0.6, release=0.4)
)

# Create melody
scale = Scale.minor('C', octave=4)
motif = Motif.from_intervals([0, 2, 3, 5, 7], scale=scale)

# Add to composition
comp.add_track('melody', synth) \
    .repeat_motif(motif, bars=2) \
    .add_fx(Reverb(mix=0.3))

# Render audio
audio = comp.render()
print(f"Generated {len(audio)} samples")
print(f"Duration: {len(audio) / 44100:.2f}s")

# Store result for playback/export
result = audio
```

**Important**: Set `result = audio` at the end to make audio available for playback/export.

### Project Files

Save and load projects:

```bash
# Save current project (Ctrl+S in app)
# Opens as project.agp by default

# Load existing project
algorythm studio myproject.agp
```

## Live Coding GUI - `algorythm-live`

### Requirements

The GUI requires tkinter system libraries:

```bash
# Ubuntu/Debian
sudo apt install python3-tk

# Fedora
sudo dnf install python3-tkinter

# Arch Linux
sudo pacman -S tk

# macOS
# Usually included with Python
```

### Launch

```bash
algorythm-live
```

If tkinter is not available, the command will display installation instructions and suggest using `algorythm studio` instead.

### Features

- Visual code editor with syntax highlighting
- Real-time output console
- Built-in audio playback (requires pyaudio)
- Example code templates
- Save audio to WAV files

### Controls

- `Ctrl+Enter` or `Ctrl+R` - Run code
- Click "Play" - Play generated audio
- Click "Stop" - Stop playback
- Click "Save Audio" - Export to WAV
- Dropdown menu - Load example code

## Examples

### Simple Synthesis

```python
from algorythm.synth import Synth
s = Synth(waveform='sine')
audio = s.generate_note(440, 1.0)
result = audio
```

### Euclidean Rhythm

```python
from algorythm.generative import euclid
from algorythm.synth import Synth
import numpy as np

pattern = euclid(hits=5, steps=16)
synth = Synth(waveform='square')

audio_parts = []
for step in pattern:
    if step:
        note = synth.generate_note(440, 0.1)
    else:
        note = np.zeros(int(0.1 * 44100))
    audio_parts.append(note)

result = np.concatenate(audio_parts)
```

### Full Composition

```python
from algorythm.synth import SynthPresets
from algorythm.sequence import Scale, Motif
from algorythm.structure import Composition, Reverb, Delay

comp = Composition(tempo=130)

# Bass line
bass = SynthPresets.bass()
bass_motif = Motif.from_intervals([0, 0, 0, 0], 
                                   scale=Scale.minor('A', octave=2))
comp.add_track('bass', bass).repeat_motif(bass_motif, bars=4)

# Lead melody
lead = SynthPresets.pluck()
lead_motif = Motif.from_intervals([0, 2, 4, 7, 9, 7, 4, 2],
                                   scale=Scale.minor('A', octave=4))
comp.add_track('lead', lead) \
    .repeat_motif(lead_motif, bars=2) \
    .add_fx(Delay(delay_time=0.25, feedback=0.3)) \
    .add_fx(Reverb(mix=0.2))

result = comp.render()
```

## Troubleshooting

### "Tkinter GUI Not Available"

Use the Terminal DAW instead:
```bash
algorythm studio
```

This provides the same Live Coding functionality without requiring system GUI libraries.

### "No audio to play"

Make sure your code sets `result = audio` at the end. This makes the audio available for playback.

### Import Errors

Make sure algorythm is properly installed:
```bash
pip install -e .
# or
pip install algorythm
```

### Audio Playback Not Working

Install pyaudio for real-time playback:
```bash
pip install pyaudio
```

## Tips

1. **Start Simple** - Begin with basic synthesis and gradually add complexity
2. **Use Print Statements** - Debug your code with print() to see what's happening
3. **Experiment** - Try different synth parameters, scales, and effects
4. **Save Often** - Use Ctrl+S to save your work in the Terminal DAW
5. **Check Output** - The output console shows execution results and errors

## Additional Resources

- Main documentation: `README.md`
- Example scripts: `examples/` directory
- API documentation: `docs/` directory
- Command reference: `algorythm --help`
- View formats: `algorythm formats`
- Available effects: `algorythm effects`
- Instrument presets: `algorythm presets`

## Quick Reference

### Common Imports

```python
from algorythm.synth import Synth, ADSR, Filter, SynthPresets
from algorythm.sequence import Scale, Motif, Chord
from algorythm.structure import Composition, Reverb, Delay, Chorus
from algorythm.generative import LSystem, euclid, markov_chain
import numpy as np
```

### Common Scales

```python
Scale.major('C', octave=4)
Scale.minor('A', octave=3)
Scale.pentatonic('E', octave=4)
Scale.chromatic('C', octave=4)
```

### Common Effects

```python
Reverb(mix=0.3)
Delay(delay_time=0.25, feedback=0.4)
Chorus(mix=0.5)
Distortion(drive=2.0)
Compressor(threshold=-20.0, ratio=4.0)
```

## Getting Help

- Press `Ctrl+P` in Terminal DAW for command palette
- Check examples dropdown in Live Coding GUI
- Run `algorythm --help` for command reference
- Visit the GitHub repository for more examples and documentation
