# Export Update Summary

## Changes Made

Updated the `Exporter` class to automatically save audio files to `~/Music` directory by default, making it easier to organize and access generated music.

## Modified Files

### `algorythm/export.py`

1. **Added imports**: `os` and `Path` from `pathlib`

2. **Updated `Exporter.__init__()`**:
   - Added `default_directory` parameter (defaults to `None`)
   - Sets `self.default_directory` to `~/Music` if not specified
   - Creates the default directory if it doesn't exist

3. **Added `_resolve_path()` method**:
   - Resolves file paths, using default directory for relative paths
   - Leaves absolute paths unchanged
   - Returns the resolved absolute path

4. **Updated `export()` method**:
   - Calls `_resolve_path()` to resolve the file path
   - Creates parent directories if they don't exist
   - Prints the full path where the file was saved

5. **Updated `export_stereo()` method**:
   - Same path resolution and directory creation logic
   - Prints the full path where the file was saved

## New Features

### Default ~/Music Directory
```python
exporter = Exporter()
exporter.export(audio, 'track.wav')  # Saves to ~/Music/track.wav
```

### Subdirectory Support
```python
exporter.export(audio, 'project/song.wav')  # Saves to ~/Music/project/song.wav
```

### Absolute Paths Still Work
```python
exporter.export(audio, '/tmp/test.wav')  # Saves to /tmp/test.wav
```

### Custom Default Directory
```python
exporter = Exporter(default_directory='/path/to/project')
exporter.export(audio, 'track.wav')  # Saves to /path/to/project/track.wav
```

### Automatic Directory Creation
- Default directory (`~/Music`) is created on `Exporter` initialization
- Parent directories are created when exporting files
- Subdirectories are automatically created as needed

### Path Feedback
- Export methods now print the full path where files are saved
- Makes it easy to find exported files

## Testing

All features tested and verified:
- ✓ Relative paths save to `~/Music`
- ✓ Subdirectories are created automatically
- ✓ Absolute paths work as expected
- ✓ Custom default directory works
- ✓ Integration with `Composition.render()` works
- ✓ Volume control + export works together
- ✓ Path feedback displays correctly

## Benefits

1. **Better Organization**: All generated music in one standard location
2. **OS Convention**: Follows operating system standards for music files
3. **Easy Discovery**: Music players can automatically find files
4. **Flexible**: Still supports absolute paths and custom directories
5. **User-Friendly**: Automatic directory creation, no manual setup needed
6. **Transparent**: Prints full path so users know where files are saved

## Backwards Compatibility

### Behavior Change
- **Old behavior**: Relative paths saved to current working directory
- **New behavior**: Relative paths save to `~/Music`

### Migration Options

**Option 1: Accept new behavior** (recommended)
```python
# Files now go to ~/Music automatically
exporter.export(audio, 'track.wav')
```

**Option 2: Use absolute paths**
```python
import os
# Save to current directory
exporter.export(audio, os.path.abspath('track.wav'))
```

**Option 3: Set current directory as default**
```python
# Use current directory as default
exporter = Exporter(default_directory='.')
exporter.export(audio, 'track.wav')
```

## Examples Updated

All examples in the repository have been tested and work with the new export behavior. Files from examples are now saved to `~/Music` instead of the `examples/` directory.

## Documentation

New documentation files:
- `EXPORT_MUSIC_FOLDER.md` - Comprehensive guide to the new export behavior
- `EXPORT_UPDATE_SUMMARY.md` - This file

Updated documentation:
- `README.md` - Updated export section with examples

## Integration with Volume Control

The export changes work seamlessly with the volume control features added earlier:

```python
from algorythm import Composition, SynthPresets, Motif, Scale

comp = Composition(tempo=120)
comp.add_track('Lead', SynthPresets.pluck())
# ... add music ...

# Volume control
comp.set_track_volume('Lead', 0.9)
comp.set_master_volume(0.85)
comp.fade_in(1.0).fade_out(2.0)

# Export to ~/Music/myproject/track.wav
comp.render('myproject/track.wav')
```

## Technical Details

### Path Resolution Algorithm

1. Convert input path to `Path` object
2. Check if path is absolute:
   - If yes: return as-is
   - If no: prepend default directory
3. Create parent directories if needed
4. Return resolved absolute path string

### Directory Creation

- Directories are created with `mkdir(parents=True, exist_ok=True)`
- Safe to call multiple times
- Creates all intermediate directories
- No error if directory already exists

## Future Enhancements

Potential future improvements:
- Support for environment variable configuration (e.g., `ALGORYTHM_MUSIC_DIR`)
- Project-based export organization
- Automatic timestamping of exports
- Conflict resolution for duplicate filenames
