# Export to ~/Music Directory

## Overview

The `Exporter` class now automatically saves audio files to your `~/Music` directory by default, making it easier to organize and find your generated tracks.

## Default Behavior

When you use a relative file path, the file is saved to `~/Music`:

```python
from algorythm.export import Exporter

exporter = Exporter()

# This saves to ~/Music/my_track.wav
exporter.export(audio_signal, 'my_track.wav', sample_rate=44100)
```

## Using Subdirectories

You can organize your tracks into subdirectories within `~/Music`:

```python
# Saves to ~/Music/project1/song.wav
exporter.export(audio_signal, 'project1/song.wav')

# Saves to ~/Music/beats/2024/track01.wav
exporter.export(audio_signal, 'beats/2024/track01.wav')
```

Subdirectories are automatically created if they don't exist.

## Using Absolute Paths

If you need to save to a specific location outside of `~/Music`, use an absolute path:

```python
# Saves to /tmp/test.wav (exact path)
exporter.export(audio_signal, '/tmp/test.wav')

# Saves to /home/user/projects/track.wav (exact path)
exporter.export(audio_signal, '/home/user/projects/track.wav')
```

## Custom Default Directory

You can specify a different default directory when creating the exporter:

```python
# Use /home/user/myproject as the default directory
exporter = Exporter(default_directory='/home/user/myproject')

# This saves to /home/user/myproject/track.wav
exporter.export(audio_signal, 'track.wav')

# This saves to /home/user/myproject/songs/track01.wav
exporter.export(audio_signal, 'songs/track01.wav')
```

## With Composition.render()

The `Composition.render()` method also uses the `~/Music` directory:

```python
from algorythm import Composition, SynthPresets, Motif, Scale

comp = Composition(tempo=120)
comp.add_track('Lead', SynthPresets.pluck())
# ... add notes/motifs ...

# Saves to ~/Music/my_composition.wav
comp.render('my_composition.wav')

# Saves to ~/Music/project/song.wav
comp.render('project/song.wav')

# Saves to /tmp/test.wav (absolute path)
comp.render('/tmp/test.wav')
```

## Complete Example

```python
from algorythm import Composition, SynthPresets, Motif, Scale, Reverb
from algorythm.export import Exporter

# Create composition
comp = Composition(tempo=120)

# Add tracks
melody = Motif.from_intervals([0, 2, 4, 5, 7], scale=Scale.major('C', octave=4))
comp.add_track('Lead', SynthPresets.pluck())
comp.repeat_motif(melody, bars=4)
comp.add_fx(Reverb(mix=0.3))

# Render and export to ~/Music/myproject/final_mix.wav
audio = comp.render('myproject/final_mix.wav')

# Or use exporter directly with custom directory
custom_exporter = Exporter(default_directory='/home/user/albumproject')
custom_exporter.export(audio, 'track_01.wav')  # Saves to /home/user/albumproject/track_01.wav
```

## Finding Your Files

After exporting, your files are in:

```bash
# List all WAV files in Music directory
ls ~/Music/*.wav

# List all files in a specific subdirectory
ls ~/Music/myproject/

# Play a file (Linux with aplay)
aplay ~/Music/my_track.wav

# Play a file (macOS)
afplay ~/Music/my_track.wav
```

## Benefits

- **Organization**: All your generated music in one place
- **Standard Location**: Follows OS conventions for music files
- **Easy Access**: Music players can find your files automatically
- **Flexible**: Still supports absolute paths when needed
- **Automatic Setup**: `~/Music` directory is created if it doesn't exist

## Migration from Previous Version

If you had scripts that used relative paths in the current working directory, you have two options:

### Option 1: Use Absolute Paths
```python
import os
# Save to current directory
exporter.export(audio, os.path.abspath('my_track.wav'))
```

### Option 2: Use Current Directory as Default
```python
# Set current directory as default
exporter = Exporter(default_directory='.')
exporter.export(audio, 'my_track.wav')  # Saves in current directory
```

### Option 3: Update Paths
```python
# Old: saved to current directory
exporter.export(audio, 'track.wav')

# New: explicitly use ~/Music (same as default)
exporter.export(audio, 'track.wav')  # Now in ~/Music

# Or use absolute path for old behavior
exporter.export(audio, './track.wav')  # Stays in current directory with ./
```

## API Reference

### Exporter Constructor

```python
Exporter(default_directory: Optional[str] = None)
```

- `default_directory`: Default directory for exports. If `None`, uses `~/Music`

### Path Resolution Rules

1. **Absolute path** (starts with `/`): Used as-is
2. **Relative path**: Prepended with default directory
3. **Subdirectories**: Automatically created if they don't exist

### Methods

All export methods (`export()`, `export_stereo()`) now:
- Accept relative or absolute paths
- Print the full path where the file was saved
- Create necessary parent directories automatically
