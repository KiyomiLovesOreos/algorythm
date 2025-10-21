# Recent Updates to Algorythm

## Summary

Two major feature additions have been made to algorythm:

1. **Comprehensive Volume Control** - Fine-grained control over audio levels
2. **~/Music Directory Export** - Automatic organization of generated music

## 1. Volume Control Features

### Track-Level Volume
Control individual track volumes in your composition:

```python
comp = Composition(tempo=120)
comp.add_track('Bass', bass_synth)
comp.set_track_volume('Bass', 0.7)  # 70% volume
```

### Master Volume
Control the overall composition volume:

```python
comp.set_master_volume(0.85)  # 85% master volume
```

### Fade Effects
Add smooth fade-in and fade-out:

```python
comp.fade_in(1.0).fade_out(2.0)  # 1s fade in, 2s fade out
```

### VolumeControl Utility Class
Powerful static methods for volume manipulation:

```python
from algorythm import VolumeControl

# Convert between dB and linear
linear = VolumeControl.db_to_linear(-6.0)  # 0.5012
db = VolumeControl.linear_to_db(0.5)       # -6.02 dB

# Apply volume to signals
quieter = VolumeControl.apply_volume(signal, 0.5)
quieter = VolumeControl.apply_db_volume(signal, -6.0)

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
Real-time volume control during playback:

```python
from algorythm.playback import AudioPlayer

player = AudioPlayer()
player.set_volume(0.7)  # 70% playback volume
player.play(audio)
```

## 2. ~/Music Directory Export

### Default Behavior
Files are now automatically saved to `~/Music`:

```python
from algorythm.export import Exporter

exporter = Exporter()
exporter.export(audio, 'my_track.wav')  # Saves to ~/Music/my_track.wav
```

### Subdirectory Organization
Organize your tracks in subdirectories:

```python
# Saves to ~/Music/project1/song.wav
exporter.export(audio, 'project1/song.wav')

# Saves to ~/Music/beats/2024/track01.wav
exporter.export(audio, 'beats/2024/track01.wav')
```

### Absolute Paths Still Work
Use absolute paths when you need specific locations:

```python
exporter.export(audio, '/tmp/test.wav')  # Exact path
```

### Custom Default Directory
Specify a different default directory:

```python
exporter = Exporter(default_directory='/home/user/myproject')
exporter.export(audio, 'track.wav')  # Saves to /home/user/myproject/track.wav
```

### With Composition
The `render()` method also uses `~/Music`:

```python
comp = Composition(tempo=120)
# ... build composition ...

# Saves to ~/Music/my_song.wav
comp.render('my_song.wav')

# Saves to ~/Music/album/track01.wav
comp.render('album/track01.wav')
```

## Complete Example

Here's an example using both features together:

```python
from algorythm import (
    Composition, VolumeControl,
    SynthPresets, Motif, Scale, Reverb
)

# Create composition
comp = Composition(tempo=130)

# Add tracks
bass = Motif.from_intervals([0, 0, 5, 5], scale=Scale.minor('A', octave=2))
lead = Motif.from_intervals([0, 2, 4, 7, 9], scale=Scale.minor('A', octave=4))

comp.add_track('Bass', SynthPresets.bass())
comp.repeat_motif(bass, bars=4)
comp.set_track_volume('Bass', 0.7)  # Quiet bass

comp.add_track('Lead', SynthPresets.pluck())
comp.repeat_motif(lead, bars=4)
comp.add_fx(Reverb(mix=0.3))
comp.set_track_volume('Lead', 1.0)  # Full volume lead

# Apply master volume and fades
comp.set_master_volume(0.85)
comp.fade_in(1.0).fade_out(2.0)

# Render and export to ~/Music/myproject/song.wav
audio = comp.render('myproject/song.wav')

# The file is automatically saved to ~/Music
# Full path: /home/user/Music/myproject/song.wav
```

## Documentation Files

### Volume Control
- `VOLUME_CONTROL.md` - Comprehensive guide
- `VOLUME_CONTROL_QUICKREF.md` - Quick reference
- `VOLUME_CONTROL_SUMMARY.md` - Implementation details
- `examples/volume_control_demo.py` - Demo script

### Export
- `EXPORT_MUSIC_FOLDER.md` - Export guide
- `EXPORT_UPDATE_SUMMARY.md` - Implementation details

### General
- `README.md` - Updated with new features
- `RECENT_UPDATES.md` - This file

## Demo Scripts

### Volume Control Demo
```bash
cd examples
python volume_control_demo.py
```

Creates demonstration files showing:
- Multi-track composition with volume control
- Different fade curve types
- VolumeControl utility functions

All files are saved to `~/Music/`

## Key Benefits

### Volume Control
- Professional mixing capabilities
- Easy balance between tracks
- Smooth fades for polished output
- Industry-standard dB conversions
- Real-time playback control

### ~/Music Export
- Better organization
- Standard location following OS conventions
- Easy discovery by music players
- Automatic directory creation
- Flexible with absolute path support

## Migration Notes

### Exporter Behavior Change
- **Old**: Relative paths saved to current working directory
- **New**: Relative paths save to `~/Music`

### Migration Options
1. **Accept new behavior** (recommended) - Files go to `~/Music`
2. **Use absolute paths** - `exporter.export(audio, os.path.abspath('track.wav'))`
3. **Set current directory** - `Exporter(default_directory='.')`

## Testing

All features have been thoroughly tested:
- ✓ Volume control (track, master, fades)
- ✓ VolumeControl utilities (conversions, normalization, fades)
- ✓ Export to ~/Music (default behavior)
- ✓ Subdirectory creation
- ✓ Absolute path support
- ✓ Custom default directory
- ✓ Full integration (composition + volume + export)
- ✓ Backwards compatibility

## Finding Your Files

After exporting, find your files at:

```bash
# List all your tracks
ls ~/Music/*.wav

# List project subdirectory
ls ~/Music/myproject/

# Play a file (Linux)
aplay ~/Music/my_track.wav

# Play a file (macOS)
afplay ~/Music/my_track.wav
```

## Next Steps

Try the new features:

1. Run the volume control demo:
   ```bash
   cd examples
   python volume_control_demo.py
   ```

2. Check your `~/Music` directory for the generated files

3. Create your own compositions using volume control and enjoy automatic organization!

---

For detailed documentation, see:
- `VOLUME_CONTROL.md` for volume control
- `EXPORT_MUSIC_FOLDER.md` for export behavior
- `README.md` for full API reference
