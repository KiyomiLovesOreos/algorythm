# Algorythm v0.2.0 - New Features

## What's New

This release adds significant new capabilities for advanced synthesis, effects processing, and interactive composition.

### 🎹 Advanced Synthesis

#### FM Synthesis
Frequency Modulation synthesis for complex, metallic, and bell-like timbres.

```python
from algorythm.synth import FMSynth, ADSR

fm_synth = FMSynth(
    carrier_waveform='sine',
    modulator_waveform='sine',
    modulation_index=3.0,      # Amount of modulation
    mod_freq_ratio=2.0,         # Modulator frequency ratio
    envelope=ADSR(attack=0.01, decay=0.3, sustain=0.2, release=0.5)
)

# Generate a note
audio = fm_synth.generate_note(frequency=440.0, duration=1.0)
```

**Features:**
- Carrier and modulator waveform selection (sine, square, saw, triangle)
- Adjustable modulation index for controlling harmonic complexity
- Frequency ratio control for different harmonic relationships
- Optional filter and envelope

#### Wavetable Synthesis
Morphing between multiple waveforms for evolving, dynamic timbres.

```python
from algorythm.synth import WavetableSynth, ADSR
import numpy as np

# Create wavetable from named waveforms
wt_synth = WavetableSynth.from_waveforms(
    waveforms=['sine', 'triangle', 'saw', 'square'],
    envelope=ADSR(attack=0.05, decay=0.2, sustain=0.6, release=0.3)
)

# Generate with automatic morphing over time
duration = 1.0
morph = np.linspace(0, 1, int(duration * 44100))  # Sweep through wavetable

audio = wt_synth.generate_note(
    frequency=440.0,
    duration=duration,
    morph_automation=morph
)
```

**Features:**
- Load custom wavetables or use built-in waveforms
- Real-time morphing between waveforms
- Position automation for evolving sounds
- Smooth interpolation between wavetable positions

### 🎛️ New Audio Effects

#### EQ (Equalizer)
3-band equalizer for frequency shaping.

```python
from algorythm.structure import EQ

eq = EQ(
    low_gain=1.5,     # Boost bass
    mid_gain=1.0,     # Keep mids neutral
    high_gain=0.6,    # Cut highs
    low_freq=200,     # Low/mid crossover
    high_freq=2000    # Mid/high crossover
)

processed = eq.apply(audio)
```

#### Phaser
Sweeping notch filter effect for classic phaser sounds.

```python
from algorythm.structure import Phaser

phaser = Phaser(
    rate=0.5,          # LFO rate in Hz
    depth=0.7,         # Modulation depth
    stages=4,          # Number of all-pass stages (2-12)
    feedback=0.5,      # Feedback amount
    mix=0.5            # Wet/dry mix
)

processed = phaser.apply(audio)
```

#### Tremolo
Amplitude modulation for classic tremolo effects.

```python
from algorythm.structure import Tremolo

tremolo = Tremolo(
    rate=5.0,          # Modulation rate in Hz
    depth=0.6,         # Modulation depth
    waveform='sine'    # LFO waveform: sine, square, triangle
)

processed = tremolo.apply(audio)
```

#### Bitcrusher
Lo-fi digital distortion and sample rate reduction.

```python
from algorythm.structure import Bitcrusher

bitcrusher = Bitcrusher(
    bit_depth=4,                    # Reduce to 4-bit (1-16)
    sample_rate_reduction=4.0,      # Reduce sample rate by 4x
    mix=1.0                         # Wet/dry mix
)

processed = bitcrusher.apply(audio)
```

### 🔊 Interactive Playback

Real-time audio playback for immediate feedback during composition.

```python
from algorythm.playback import AudioPlayer

# Create player
player = AudioPlayer(sample_rate=44100)

# Play audio (non-blocking)
player.play(audio, blocking=False)

# Or play and wait for completion
player.play(audio, blocking=True)

# Stop playback
player.stop()

# Clean up
player.close()
```

**Requires:** `pip install pyaudio`

#### Streaming Player
Generate and play audio on-the-fly.

```python
from algorythm.playback import StreamingPlayer

def generate_audio(num_samples):
    """Generate audio chunk in real-time"""
    # Your generation code here
    return audio_chunk

player = StreamingPlayer(
    generator_callback=generate_audio,
    sample_rate=44100
)

player.start()  # Start streaming
# ... streaming continues ...
player.stop()   # Stop streaming
player.close()
```

#### Live Composition Player
Update compositions while they're playing.

```python
from algorythm.playback import LiveCompositionPlayer

player = LiveCompositionPlayer()

# Play with looping
player.play(audio, loop=True)

# Update the audio while playing
new_audio = generate_new_audio()
player.update_audio(new_audio)

player.stop()
```

### 🎨 Live Coding GUI

Interactive GUI for composing music with real-time feedback.

**Launch the GUI:**
```bash
# Command line
algorythm-live

# Or in Python
python -m algorythm.live_gui
```

**Features:**
- Code editor with syntax highlighting
- Real-time audio playback
- Built-in examples
- Output console for debugging
- Save generated audio
- Keyboard shortcut: `Ctrl+Enter` to run code

**GUI Code Example:**
```python
from algorythm.synth import FMSynth, ADSR
from algorythm.sequence import Scale

# Code in the GUI editor
fm = FMSynth(modulation_index=3.0)
scale = Scale.major('C', octave=5)

audio_parts = []
for i in range(8):
    note = fm.generate_note(scale.get_frequency(i), 0.3)
    audio_parts.append(note)

result = np.concatenate(audio_parts)
# GUI will automatically play 'result'
```

**Requires:** Built-in tkinter (included with Python) + pyaudio for playback

### 📦 Installation

#### Basic Installation
```bash
pip install -e .
```

#### With Playback Support
```bash
pip install -e ".[playback]"
```

#### With GUI Support
```bash
pip install -e ".[gui]"
```

#### All Features
```bash
pip install -e ".[all]"
```

### 🎵 Complete Example

Combine all new features:

```python
from algorythm.synth import FMSynth, WavetableSynth, ADSR, Filter
from algorythm.sequence import Scale, Motif
from algorythm.structure import (
    Composition, EQ, Phaser, Tremolo, 
    Delay, Reverb
)
from algorythm.playback import AudioPlayer

# Create FM synth for melody
fm_synth = FMSynth(
    modulation_index=2.5,
    mod_freq_ratio=3.0,
    envelope=ADSR(attack=0.01, decay=0.2, sustain=0.4, release=0.6)
)

# Create wavetable synth for bass
wt_synth = WavetableSynth.from_waveforms(
    waveforms=['saw', 'square'],
    envelope=ADSR(attack=0.05, decay=0.3, sustain=0.7, release=0.2),
    filter=Filter.lowpass(cutoff=800, resonance=0.6)
)

# Create composition
comp = Composition(tempo=128)

# Add FM melody
scale = Scale.minor('A', octave=5)
melody = Motif.from_intervals([0, 2, 3, 5, 7, 5, 3, 2], scale=scale)

comp.add_track('melody', fm_synth) \
    .repeat_motif(melody, bars=4) \
    .add_fx(Phaser(rate=0.4, depth=0.6)) \
    .add_fx(Delay(delay_time=0.375, feedback=0.3, mix=0.25)) \
    .add_fx(Reverb(mix=0.3))

# Add wavetable bass
bass_scale = Scale.minor('A', octave=2)
bass_motif = Motif.from_intervals([0, 0, 3, 3], scale=bass_scale)

comp.add_track('bass', wt_synth) \
    .repeat_motif(bass_motif, bars=4) \
    .add_fx(Tremolo(rate=4.0, depth=0.3)) \
    .add_fx(EQ(low_gain=1.5, mid_gain=0.8, high_gain=0.5))

# Render
audio = comp.render()

# Play
player = AudioPlayer()
player.play(audio, blocking=True)
player.close()

# Save
from algorythm.export import Exporter
exporter = Exporter()
exporter.export(audio, 'my_track.wav')
```

## Migration Guide

### From v0.1.0 to v0.2.0

All existing code remains compatible. New features are additive:

```python
# Old code still works
from algorythm.synth import Synth
synth = Synth(waveform='saw')

# New synthesis methods available
from algorythm.synth import FMSynth, WavetableSynth
fm = FMSynth(modulation_index=3.0)
wt = WavetableSynth.from_waveforms(['sine', 'saw'])

# Old effects still work
from algorythm.structure import Reverb, Delay
comp.add_fx(Reverb(mix=0.3))

# New effects available
from algorythm.structure import EQ, Phaser, Tremolo, Bitcrusher
comp.add_fx(Phaser(rate=0.5))
```

## Performance Notes

- **FM Synthesis:** Fast, CPU-efficient
- **Wavetable Synthesis:** Slightly more CPU intensive due to interpolation
- **Effects:** EQ uses FFT (more expensive), others are time-domain (fast)
- **Playback:** Uses separate thread, minimal latency
- **GUI:** Runs code in background thread, keeps UI responsive

## Known Limitations

1. **Playback** requires `pyaudio` which can be tricky to install on some systems
2. **GUI** requires tkinter (usually included with Python)
3. **Wavetable morphing** automation must match audio length
4. **Effects** are simplified implementations suitable for algorithmic music

## Future Plans

- MIDI file import/export
- More synthesis methods (granular, additive)
- Advanced filter designs
- Multi-core rendering
- VST plugin support

## Changelog

### v0.2.0 (2025-10-21)

**Added:**
- FM Synthesis (`FMSynth`)
- Wavetable Synthesis (`WavetableSynth`)
- EQ effect (3-band equalizer)
- Phaser effect
- Tremolo effect
- Bitcrusher effect
- Real-time audio playback module (`algorythm.playback`)
- Live coding GUI (`algorythm.live_gui`)
- Streaming audio player
- Live composition player
- `algorythm-live` command for launching GUI

**Changed:**
- Updated version to 0.2.0
- Enhanced README with new examples
- Added installation options for playback and GUI

**Dependencies:**
- Optional: `pyaudio>=0.2.11` for playback
- tkinter (built-in) for GUI
