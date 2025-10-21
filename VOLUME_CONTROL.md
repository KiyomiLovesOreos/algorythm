# Volume Control in Algorythm

Algorythm now includes comprehensive volume control features at multiple levels: track volume, master volume, real-time playback volume, and utility functions for volume conversions and manipulation.

## Features Overview

### 1. Track-Level Volume Control

Control the volume of individual tracks in a composition:

```python
from algorythm import Composition, SynthPresets
from algorythm.sequence import Motif, Scale

comp = Composition(tempo=120)

# Add tracks
comp.add_track('Bass', SynthPresets.bass())
comp.add_track('Lead', SynthPresets.pluck())
comp.add_track('Pad', SynthPresets.warm_pad())

# Set different volumes for each track
comp.set_track_volume('Bass', 0.8)   # 80% volume
comp.set_track_volume('Lead', 1.0)   # 100% volume (full)
comp.set_track_volume('Pad', 0.4)    # 40% volume (background)
```

### 2. Master Volume Control

Control the overall volume of the entire composition:

```python
# Set master volume for the entire composition
comp.set_master_volume(0.9)  # 90% master volume
```

The master volume is applied after all tracks are mixed together, allowing you to control the final output level without adjusting individual tracks.

### 3. Fade In/Out

Add smooth fades at the beginning and end of your composition:

```python
# Add 1 second fade in and 2 second fade out
comp.fade_in(1.0).fade_out(2.0)
```

Fades are applied during rendering and work with the chaining API style.

### 4. Real-Time Playback Volume

Control volume during real-time audio playback:

```python
from algorythm.playback import AudioPlayer, StreamingPlayer

# AudioPlayer volume control
player = AudioPlayer()
player.set_volume(0.7)  # 70% volume
player.play(audio)

# StreamingPlayer volume control
def generate_audio(frames):
    return np.sin(2 * np.pi * 440 * np.arange(frames) / 44100)

stream_player = StreamingPlayer(generate_audio)
stream_player.set_volume(0.5)  # 50% volume
stream_player.start()
```

## VolumeControl Utility Class

The `VolumeControl` class provides static methods for volume manipulation and conversion:

### Decibel Conversions

Convert between decibels (dB) and linear amplitude:

```python
from algorythm import VolumeControl

# Convert dB to linear amplitude
linear = VolumeControl.db_to_linear(-6.0)  # Returns ~0.5
linear = VolumeControl.db_to_linear(-3.0)  # Returns ~0.707
linear = VolumeControl.db_to_linear(0.0)   # Returns 1.0
linear = VolumeControl.db_to_linear(6.0)   # Returns ~2.0

# Convert linear amplitude to dB
db = VolumeControl.linear_to_db(0.5)  # Returns ~-6.02 dB
db = VolumeControl.linear_to_db(1.0)  # Returns 0.0 dB
db = VolumeControl.linear_to_db(2.0)  # Returns ~6.02 dB
```

### Apply Volume

Apply volume changes to audio signals:

```python
import numpy as np

# Create a test signal
signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))

# Apply linear volume (multiply by 0.5)
quieter = VolumeControl.apply_volume(signal, 0.5)

# Apply volume in decibels (-6 dB attenuation)
quieter_db = VolumeControl.apply_db_volume(signal, -6.0)
```

### Normalize Audio

Normalize audio signals to a target dB level:

```python
# Normalize to -3 dB peak level (common for music mastering)
normalized = VolumeControl.normalize(signal, target_db=-3.0)

# Normalize to 0 dB (maximum level without clipping)
normalized_max = VolumeControl.normalize(signal, target_db=0.0)
```

### Advanced Fades

Apply fades with different curve types:

```python
# Linear fade (constant rate)
faded_linear = VolumeControl.fade(
    signal,
    fade_in=1.0,
    fade_out=2.0,
    sample_rate=44100,
    curve='linear'
)

# Exponential fade (smooth, natural sounding)
faded_exp = VolumeControl.fade(
    signal,
    fade_in=1.0,
    fade_out=2.0,
    sample_rate=44100,
    curve='exponential'
)

# Logarithmic fade (perception-based)
faded_log = VolumeControl.fade(
    signal,
    fade_in=1.0,
    fade_out=2.0,
    sample_rate=44100,
    curve='logarithmic'
)
```

## Complete Example

Here's a complete example combining all volume control features:

```python
from algorythm import (
    Composition, VolumeControl,
    SynthPresets, Motif, Scale, Reverb
)

# Create composition
comp = Composition(tempo=120)

# Create motifs
bass_line = Motif.from_intervals([0, 0, 5, 5], scale=Scale.minor('C', octave=2))
melody = Motif.from_intervals([0, 2, 4, 5, 7], scale=Scale.minor('C', octave=4))

# Add tracks with individual volumes
comp.add_track('Bass', SynthPresets.bass())
comp.repeat_motif(bass_line, bars=4)
comp.set_track_volume('Bass', 0.8)  # Slightly quieter bass

comp.add_track('Lead', SynthPresets.pluck())
comp.repeat_motif(melody, bars=4)
comp.add_fx(Reverb(mix=0.3))
comp.set_track_volume('Lead', 1.0)  # Full volume lead

# Set master volume and fades
comp.set_master_volume(0.9)
comp.fade_in(1.5).fade_out(2.0)

# Render
audio = comp.render('my_track.wav')

# Post-process with VolumeControl utilities if needed
audio_normalized = VolumeControl.normalize(audio, target_db=-3.0)
audio_faded = VolumeControl.fade(
    audio_normalized,
    fade_in=0.5,
    fade_out=1.0,
    curve='exponential'
)
```

## Volume Guidelines

### Recommended Levels

- **Track Volume**: Use values between 0.0 and 1.0 for most cases
  - Background elements: 0.3 - 0.5
  - Supporting elements: 0.6 - 0.8
  - Lead elements: 0.8 - 1.0
  
- **Master Volume**: Keep below 1.0 to leave headroom
  - For mixing: 0.7 - 0.9
  - For final output: Use normalization instead
  
- **Normalization**: Common target levels
  - `-3 dB`: Standard for music (recommended)
  - `-6 dB`: Conservative, more headroom
  - `-1 dB`: Near maximum, for loud productions
  - `0 dB`: Maximum (use with caution, can cause clipping)

### Preventing Clipping

The composition renderer automatically normalizes output to prevent clipping, but you can also:

1. Set master volume below 1.0
2. Adjust individual track volumes
3. Use `VolumeControl.normalize()` with appropriate target dB

### Fade Duration Guidelines

- **Short fades** (0.1 - 0.5s): For transitions between sections
- **Medium fades** (0.5 - 2.0s): For song intros/outros
- **Long fades** (2.0 - 5.0s): For atmospheric or ambient music

## API Reference

### Composition Methods

- `set_track_volume(track_name: str, volume: float) -> Composition`
- `set_master_volume(volume: float) -> Composition`
- `fade_in(duration: float) -> Composition`
- `fade_out(duration: float) -> Composition`

### VolumeControl Static Methods

- `db_to_linear(db: float) -> float`
- `linear_to_db(linear: float) -> float`
- `apply_volume(signal: np.ndarray, volume: float) -> np.ndarray`
- `apply_db_volume(signal: np.ndarray, db: float) -> np.ndarray`
- `normalize(signal: np.ndarray, target_db: float = -3.0) -> np.ndarray`
- `fade(signal: np.ndarray, fade_in: float = 0.0, fade_out: float = 0.0, sample_rate: int = 44100, curve: str = 'linear') -> np.ndarray`

### AudioPlayer/StreamingPlayer Methods

- `set_volume(volume: float)`

## Demo Script

Run the included demo to see all volume control features in action:

```bash
cd examples
python volume_control_demo.py
```

This will create several audio files demonstrating:
- Multi-track composition with individual volumes
- Master volume control
- Fade in/out effects
- Different fade curve types (linear, exponential, logarithmic)
- Volume conversion utilities
