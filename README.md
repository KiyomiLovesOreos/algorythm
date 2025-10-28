# Algorythm

A terminal-based digital audio workstation (DAW) for algorithmic music composition in Python.

## Overview

Algorythm is a powerful Python library for creating music with code. It provides a flexible and expressive set of tools for synthesis, sequencing, and arrangement. Whether you are a developer, a musician, or an artist, Algorythm allows you to explore the intersection of code and music.

The project also includes a vision for a full-featured, keyboard-driven, terminal-based DAW experience with multiple composition workflows, including a piano roll, a tracker, and live coding.

## Features

*   **Powerful Synthesis Engines**: Basic, FM, Wavetable, Physical Modeling, Additive, and Granular synthesis.
*   **Extensive Audio Effects**: A wide range of effects including reverb, delay, chorus, distortion, compression, and more.
*   **50+ Instrument Presets**: A large collection of ready-to-use instrument presets.
*   **Advanced Sequencing**: Create complex musical patterns with motifs, rhythms, arpeggiators, and scales.
*   **Generative Music Tools**: Explore algorithmic composition with L-Systems, Cellular Automata, and more.
*   **Automation**: Automate any parameter with curves.
*   **Video Visualization**: Render beautiful visualizations of your music to MP4 video, with optimized performance.
*   **Flexible Export**: Export your creations to WAV, MP3, and FLAC formats.

## Getting Started

Install Algorythm using pip:
```bash
pip install -e .
```

## Example Usage

Create your first song with just a few lines of Python:
```python
from algorythm.synth import SynthPresets
from algorythm.sequence import Motif, Scale
from algorythm.structure import Composition

# 1. Pick an instrument
my_instrument = SynthPresets.pluck()

# 2. Create a melody
my_melody = Motif.from_intervals([0, 2, 4, 5], scale=Scale.major('C'))

# 3. Put it all together
song = Composition(tempo=120)
song.add_track('MyTrack', my_instrument)
song.repeat_motif(my_melody, bars=2)
song.render('my_first_song.wav')

print("Your song is ready! Check 'my_first_song.wav'")
```

## Instruments & Effects

Algorythm comes with over 50 instrument presets and a wide variety of audio effects. You can find a complete list in the [Instruments and Effects Guide](docs/INSTRUMENTS_AND_EFFECTS.md).

## Video Visualization

Create stunning video visualizations of your audio files with optimized performance. The new streaming renderer is 4-5x faster and uses 80-90% less memory.

```python
from algorythm.audio_loader import visualize_audio_file
from algorythm.visualization import CircularVisualizer

viz = CircularVisualizer(sample_rate=44100, num_bars=64)
visualize_audio_file('song.mp3', 'output.mp4', viz,
                     video_width=1280, video_height=720, video_fps=24)
```

## Terminal DAW

Algorythm is evolving into a terminal-based DAW. The planned features include:
*   **Multiple Composition Views**: Piano Roll, Tracker, and Arranger.
*   **Live Coding**: A Python REPL to manipulate your composition in real-time.
*   **Keyboard-First Workflow**: A fast and efficient, keyboard-driven interface.

For more details, see the [Terminal DAW Design Doc](docs/TERMINAL_DAW_DESIGN_DOC.md).

## Documentation

For more detailed information, please refer to the documentation in the `docs` directory.
- [Beginner's Guide](docs/BEGINNER_GUIDE.md)
- [Complete Feature List](docs/COMPLETE_FEATURE_LIST.md)
- [Instruments and Effects Guide](docs/INSTRUMENTS_AND_EFFECTS.md)
- [Video Optimization Guide](docs/VIDEO_OPTIMIZATION_GUIDE.md)

## Installation

```bash
pip install algorythm
```
Optional dependencies for playback and GUI can be installed with:
```bash
pip install algorythm[playback]
pip install algorythm[gui]
pip install algorythm[all]
```
