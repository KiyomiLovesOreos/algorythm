# Bug Fix: MP3 Recognition and CLI Enhancement

## Summary

Fixed the issue where algorythm wasn't recognizing pydub for MP3 visualization and enhanced the CLI with better error handling and more features.

## Problems Fixed

### 1. **Dependency Mismatch**
- **Problem:** `requirements.txt` listed `pydub>=0.25.1`, but `setup.py` only had it in `extras_require["export"]`
- **Impact:** Users installing with `pip install algorythm` didn't get pydub automatically
- **Fix:** Added pydub to main `install_requires` in `setup.py`

### 2. **Poor Error Messages**
- **Problem:** Generic import errors that didn't explain what was missing
- **Impact:** Users confused about why MP3 files wouldn't load
- **Fix:** Added comprehensive error messages with platform-specific installation instructions

### 3. **No ffmpeg Detection**
- **Problem:** pydub requires ffmpeg for MP3 support, but there was no check
- **Impact:** Cryptic errors when ffmpeg was missing
- **Fix:** Added early ffmpeg detection with helpful error messages

## Changes Made

### 1. `setup.py`
```python
install_requires=[
    "numpy>=1.19.0",
    "pydub>=0.25.1",   # NOW INCLUDED BY DEFAULT
    "Pillow>=8.0.0",
]
```

### 2. `algorythm/audio_loader.py`
- Added module-level `HAS_PYDUB` flag for early detection
- Enhanced `load_audio_pydub()` with better error handling
- Added ffmpeg availability check
- Improved error messages with OS-specific installation commands

### 3. `algorythm/cli.py`

#### New Features:
- **`formats` command** - Shows supported formats and dependency status
- **Color options** - `--color` flag (blue, red, green, purple, orange, cyan, magenta)
- **Background options** - `--background` flag (black, white, dark, light)
- **Bar count** - `--bars` flag for circular visualizer customization
- **Debug mode** - `--debug` flag for troubleshooting

#### Enhanced Error Handling:
- Better ImportError messages with installation instructions
- Separate RuntimeError handling for ffmpeg issues
- FileNotFoundError handling
- Debug traceback option

#### New Command Examples:
```bash
# Check what's supported
algorythm formats

# Get file info
algorythm info song.mp3

# Create visualization with options
algorythm visualize song.mp3 --color purple --background dark --bars 128

# Process section of file
algorythm visualize song.mp3 --offset 30 --duration 60
```

## Testing

All changes tested and verified:

✅ MP3 loading works correctly
✅ pydub detection works
✅ ffmpeg detection works  
✅ Error messages are clear and helpful
✅ CLI commands work as expected
✅ `formats` command shows accurate dependency status
✅ `info` command displays MP3 file information
✅ `visualize` command creates videos from MP3 files
✅ Color and customization options work

## User Impact

### Before:
```bash
$ pip install algorythm
$ algorythm visualize song.mp3
Error: No module named 'pydub'  # Unclear what to do
```

### After:
```bash
$ pip install algorythm
$ algorythm formats
# Shows exactly what's installed and what's needed

$ algorythm visualize song.mp3
# Works automatically if pydub was installed
# OR shows clear instructions if pydub/ffmpeg missing:
# ❌ Error: pydub is required to load MP3/OGG/FLAC files.
# Install it with: pip install pydub
# Or reinstall algorythm with: pip install --upgrade algorythm
```

## Installation Instructions for Users

### Standard Installation (includes MP3 support):
```bash
pip install --upgrade algorythm
```

### Install ffmpeg:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/
```

### Verify Installation:
```bash
algorythm formats
```

## Files Modified

1. `setup.py` - Added pydub to install_requires
2. `algorythm/audio_loader.py` - Enhanced error handling and ffmpeg detection
3. `algorythm/cli.py` - Added features and better error messages

## Files Created

1. `CLI_MP3_GUIDE.md` - Comprehensive CLI usage guide
2. `BUGFIX_MP3_CLI_SUMMARY.md` - This file

## Backward Compatibility

✅ All existing functionality preserved
✅ No breaking changes to API
✅ Old code continues to work
✅ New features are optional enhancements

## Next Steps

Users should:
1. Update to the latest version: `pip install --upgrade algorythm`
2. Run `algorythm formats` to check dependencies
3. Install any missing dependencies (pydub, ffmpeg)
4. Try the new CLI features!

## Example Workflow

```bash
# Check dependencies
$ algorythm formats

# Get info about a file
$ algorythm info my_song.mp3

# Create visualization
$ algorythm visualize my_song.mp3 \
    -v circular \
    --bars 128 \
    --color purple \
    --background dark \
    -o my_video.mp4
```

Perfect! 🎵
