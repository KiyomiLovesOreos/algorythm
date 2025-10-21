# Algorythm v0.3.0 Release Notes

**Release Date**: October 21, 2025

## 🎉 What's New

Version 0.3.0 introduces two major features that significantly enhance the music production workflow:

1. **Comprehensive Volume Control System**
2. **Automatic Export to ~/Music Directory**

---

## 🎚️ Volume Control System

Professional-grade volume control at every level of your composition.

### Track-Level Volume

Control individual track volumes for perfect mixing:

```python
comp = Composition(tempo=120)
comp.add_track('Bass', bass_synth)
comp.set_track_volume('Bass', 0.7)  # 70% volume

comp.add_track('Lead', lead_synth)
comp.set_track_volume('Lead', 1.0)  # 100% volume

comp.add_track('Pad', pad_synth)
comp.set_track_volume('Pad', 0.4)   # 40% volume (background)
```

### Master Volume

Control the overall composition volume:

```python
comp.set_master_volume(0.85)  # 85% master volume
```

### Smooth Fades

Add professional fade-in and fade-out effects:

```python
comp.fade_in(1.0).fade_out(2.0)  # 1s fade in, 2s fade out
```

### VolumeControl Utility Class

Powerful tools for audio manipulation:

```python
from algorythm import VolumeControl

# Convert between dB and linear
linear = VolumeControl.db_to_linear(-6.0)  # Returns ~0.5
db = VolumeControl.linear_to_db(0.5)       # Returns ~-6.02 dB

# Apply volume to signals
quieter = VolumeControl.apply_volume(signal, 0.5)
quieter_db = VolumeControl.apply_db_volume(signal, -6.0)

# Normalize to target level
normalized = VolumeControl.normalize(signal, target_db=-3.0)

# Advanced fades with curve types
faded = VolumeControl.fade(
    signal,
    fade_in=1.0,
    fade_out=2.0,
    curve='exponential'  # 'linear', 'exponential', 'logarithmic'
)
```

### Playback Volume Control

Control volume during real-time playback:

```python
from algorythm.playback import AudioPlayer

player = AudioPlayer()
player.set_volume(0.7)  # 70% playback volume
player.play(audio)
```

---

## 💾 Export to ~/Music Directory

Automatic organization of your generated music.

### Default Behavior

Files now automatically save to your Music directory:

```python
from algorythm.export import Exporter

exporter = Exporter()
exporter.export(audio, 'my_track.wav')  # Saves to ~/Music/my_track.wav
```

### Subdirectory Organization

Keep projects organized with subdirectories:

```python
# Saves to ~/Music/project1/song.wav
exporter.export(audio, 'project1/song.wav')

# Saves to ~/Music/album/tracks/track01.wav
exporter.export(audio, 'album/tracks/track01.wav')
```

Subdirectories are created automatically!

### Absolute Paths

Still need to save somewhere specific? Use absolute paths:

```python
exporter.export(audio, '/tmp/test.wav')  # Exact location
```

### Custom Default Directory

Set your own default directory:

```python
exporter = Exporter(default_directory='/home/user/myproject')
exporter.export(audio, 'track.wav')  # Saves to /home/user/myproject/track.wav
```

### Integration with Composition

Works seamlessly with the render method:

```python
comp = Composition(tempo=120)
# ... build composition ...

# Saves to ~/Music/my_song.wav
comp.render('my_song.wav')

# Saves to ~/Music/album/track01.wav
comp.render('album/track01.wav')
```

---

## 🎵 Complete Example

Here's both features working together:

```python
from algorythm import (
    Composition, VolumeControl,
    SynthPresets, Motif, Scale, Reverb
)

# Create composition
comp = Composition(tempo=130)

# Add tracks with volume control
bass = Motif.from_intervals([0, 0, 5, 5], scale=Scale.minor('A', octave=2))
lead = Motif.from_intervals([0, 2, 4, 7, 9], scale=Scale.minor('A', octave=4))

comp.add_track('Bass', SynthPresets.bass())
comp.repeat_motif(bass, bars=4)
comp.set_track_volume('Bass', 0.7)

comp.add_track('Lead', SynthPresets.pluck())
comp.repeat_motif(lead, bars=4)
comp.add_fx(Reverb(mix=0.3))
comp.set_track_volume('Lead', 1.0)

# Apply master volume and fades
comp.set_master_volume(0.85)
comp.fade_in(1.0).fade_out(2.0)

# Render and export to ~/Music/myproject/song.wav
audio = comp.render('myproject/song.wav')

# Your track is now in ~/Music/myproject/song.wav
```

---

## 📚 Documentation

### New Documentation Files

- **VOLUME_CONTROL.md** - Comprehensive volume control guide
- **VOLUME_CONTROL_QUICKREF.md** - Quick reference card
- **EXPORT_MUSIC_FOLDER.md** - Export behavior guide
- **CHANGELOG.md** - Version history
- **RECENT_UPDATES.md** - Feature overview

### Demo Scripts

- **examples/volume_control_demo.py** - Complete demonstration

Run it with:
```bash
cd examples
python volume_control_demo.py
```

---

## 🔄 Migration Guide

### From v0.2.x

#### Export Behavior Change

**Old behavior**: Relative paths saved to current working directory  
**New behavior**: Relative paths save to `~/Music`

#### Migration Options

1. **Accept new behavior** (recommended)
   - Your music files will be automatically organized in `~/Music`

2. **Use absolute paths**
   ```python
   import os
   exporter.export(audio, os.path.abspath('track.wav'))
   ```

3. **Set current directory as default**
   ```python
   exporter = Exporter(default_directory='.')
   ```

### No Breaking Changes

- All existing code using absolute paths works unchanged
- Track.volume attribute was already present, now accessible via API
- All methods support method chaining as before

---

## ✨ Key Features Summary

### Volume Control
- ✅ Track-level volume control
- ✅ Master volume control
- ✅ Smooth fade in/out effects
- ✅ dB ↔ linear conversions
- ✅ Audio normalization
- ✅ Advanced fade curves (3 types)
- ✅ Real-time playback volume

### Export System
- ✅ Automatic ~/Music directory export
- ✅ Subdirectory creation
- ✅ Absolute path support
- ✅ Custom default directory
- ✅ Path feedback
- ✅ Seamless integration

---

## 🧪 Testing

All features thoroughly tested:
- ✅ Volume control (track, master, fades)
- ✅ VolumeControl utilities
- ✅ Export to ~/Music
- ✅ Subdirectory creation
- ✅ Absolute path support
- ✅ Custom default directory
- ✅ Full integration tests
- ✅ Backwards compatibility

---

## 💻 Installation

### Upgrade from pip
```bash
pip install --upgrade algorythm
```

### Install from source
```bash
git clone https://github.com/KiyomiLovesOreos/algorythm
cd algorythm
git checkout v0.3.0
pip install -e .
```

---

## 🙏 Feedback

We'd love to hear your feedback! Please report issues or suggestions on our GitHub repository.

---

## 📖 Learn More

- **README.md** - Main documentation
- **VOLUME_CONTROL.md** - Volume control deep dive
- **EXPORT_MUSIC_FOLDER.md** - Export system details
- **examples/** - Sample scripts

---

**Enjoy creating music with Algorythm v0.3.0!** 🎵
