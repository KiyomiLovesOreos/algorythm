# Algorythm v0.2.0 Quick Start Guide

## Installation

```bash
cd ~/Projects/algorythm

# Install with all features
pip install --user --break-system-packages -e ".[all]"

# Or basic installation
pip install --user --break-system-packages -e .
```

## New in v0.2.0

### 1. FM Synthesis

```python
from algorythm.synth import FMSynth, ADSR

fm = FMSynth(
    modulation_index=3.0,
    mod_freq_ratio=2.0,
    envelope=ADSR(attack=0.01, decay=0.3, sustain=0.2, release=0.5)
)

audio = fm.generate_note(440.0, 1.0)
```

### 2. Wavetable Synthesis

```python
from algorythm.synth import WavetableSynth
import numpy as np

wt = WavetableSynth.from_waveforms(['sine', 'saw', 'square'])

# With morphing
morph = np.linspace(0, 1, 44100)  # Sweep through wavetable
audio = wt.generate_note(440.0, 1.0, morph_automation=morph)
```

### 3. New Effects

```python
from algorythm.structure import EQ, Phaser, Tremolo, Bitcrusher

# EQ - Frequency shaping
eq = EQ(low_gain=1.5, mid_gain=1.0, high_gain=0.7)
audio = eq.apply(audio)

# Phaser - Sweeping notches
phaser = Phaser(rate=0.5, depth=0.7)
audio = phaser.apply(audio)

# Tremolo - Amplitude modulation
tremolo = Tremolo(rate=5.0, depth=0.6)
audio = tremolo.apply(audio)

# Bitcrusher - Lo-fi distortion
bitcrusher = Bitcrusher(bit_depth=4, sample_rate_reduction=4.0)
audio = bitcrusher.apply(audio)
```

### 4. Interactive Playback

```python
from algorythm.playback import AudioPlayer

player = AudioPlayer()
player.play(audio, blocking=True)
player.close()
```

### 5. Live Coding GUI

```bash
# Launch the GUI
algorythm-live

# Or from Python
python -m algorythm.live_gui
```

## Complete Example

```python
from algorythm.synth import FMSynth, ADSR
from algorythm.sequence import Scale, Motif
from algorythm.structure import Composition, Phaser, EQ, Reverb
from algorythm.playback import AudioPlayer

# Create FM synth
fm = FMSynth(
    modulation_index=2.5,
    mod_freq_ratio=2.0,
    envelope=ADSR(attack=0.01, decay=0.2, sustain=0.3, release=0.5)
)

# Create composition
comp = Composition(tempo=128)

# Add melody with effects
scale = Scale.minor('A', octave=5)
motif = Motif.from_intervals([0, 2, 3, 5, 7], scale=scale)

comp.add_track('melody', fm) \
    .repeat_motif(motif, bars=4) \
    .add_fx(Phaser(rate=0.4, depth=0.6)) \
    .add_fx(EQ(low_gain=1.2, high_gain=0.8)) \
    .add_fx(Reverb(mix=0.3))

# Render
audio = comp.render()

# Play
player = AudioPlayer()
player.play(audio, blocking=True)
player.close()

# Save
from algorythm.export import Exporter
exporter = Exporter()
exporter.export(audio, 'output.wav')
```

## Run Demo

```bash
cd ~/Projects/algorythm/examples
python3 new_features_v2_demo.py
```

## Features Summary

**Synthesis:**
- ✅ Basic Synthesis (sine, saw, square, triangle, noise)
- ✅ FM Synthesis (NEW)
- ✅ Wavetable Synthesis (NEW)

**Effects:**
- ✅ Reverb, Delay, Chorus, Flanger, Distortion, Compression
- ✅ EQ (NEW)
- ✅ Phaser (NEW)
- ✅ Tremolo (NEW)
- ✅ Bitcrusher (NEW)

**Composition:**
- ✅ Multi-track composition
- ✅ Scales, Chords, Motifs
- ✅ L-System, Cellular Automata
- ✅ Data Sonification
- ✅ Parameter Automation

**Playback & Interaction:**
- ✅ Real-time playback (NEW)
- ✅ Streaming audio (NEW)
- ✅ Live composition updates (NEW)
- ✅ Live coding GUI (NEW)

**Export:**
- ✅ WAV, FLAC, MP3, OGG
- ✅ Multiple bit depths

**Visualization:**
- ✅ Waveform, Spectrogram, Frequency Scope
- ✅ Oscilloscope, Piano Roll
- ✅ Video rendering

## Documentation

- Full API: See `README.md`
- New Features: See `NEW_FEATURES_V0.2.md`
- Implementation: See `IMPLEMENTATION_V0.2.md`

## Next Steps

1. Try the demo: `python3 examples/new_features_v2_demo.py`
2. Launch the GUI: `algorythm-live`
3. Read the docs: `NEW_FEATURES_V0.2.md`
4. Explore examples: `examples/`

Enjoy composing with Algorythm v0.2.0! 🎵
