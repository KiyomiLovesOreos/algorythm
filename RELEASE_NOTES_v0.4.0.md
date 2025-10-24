# Algorythm v0.4.0 Release Notes

## 🎉 Major Release - Audio File Visualization & Enhanced CLI

Released: October 2024

## What's New

### 🎵 Audio File Visualization

Load existing audio files and create visualizations!

- **Load MP3, WAV, OGG, FLAC files**
- **Apply any visualizer to your music**
- **Three usage methods**: one-line function, class-based, or manual
- **Partial loading**: Load specific sections (offset/duration)
- **Automatic resampling**: Handle different sample rates

```python
from algorythm.audio_loader import visualize_audio_file
from algorythm.visualization import CircularVisualizer

viz = CircularVisualizer(sample_rate=44100, num_bars=64)
visualize_audio_file('my_song.mp3', 'my_video.mp4', viz)
```

### 🖥️ Enhanced CLI

New command-line interface for quick audio visualization!

```bash
# Create visualization
algorythm visualize song.mp3 -v circular -w 1920 --height 1080

# Show audio info
algorythm info song.mp3

# Run examples
algorythm --example basic
```

**Features:**
- One-command visualization creation
- Five visualizer types to choose from
- Custom resolution, FPS, and duration
- Audio file information display
- Built-in example scripts

### 🐛 Bug Fixes

**FrequencyScopeVisualizer & SpectrogramVisualizer**
- Fixed: Now actually visualize audio data!
- Added missing `to_image_data()` methods
- Both now render properly in videos

### 🎨 Previous Features (v0.4.0 Development)

- Debug mode on all visualizers
- Enhanced waveform rendering
- Colored spectrograms
- Progress tracking
- Multiple video backends (OpenCV/matplotlib/PIL)
- Fully working MP4 export with synchronized audio

## New Modules

- **algorythm.audio_loader** - Audio file loading and visualization
  - `load_audio()` - Load audio from files
  - `visualize_audio_file()` - One-line visualization
  - `AudioFile` - Object-oriented interface

## New Examples

- **06_visualize_audio_files.py** - Complete guide to loading and visualizing audio files

## Updated

- **algorythm/cli.py** - Enhanced command-line interface
- **algorythm/__init__.py** - Export audio loader functions
- **examples/README.md** - Added new example

## Documentation

New documentation files:
- **CLI_GUIDE.md** - Complete CLI documentation
- **AUDIO_FILE_VISUALIZATION.md** - Audio loading API reference
- **VISUALIZATION_FIX.md** - Bug fix details
- **VISUALIZATION_FIX_SUMMARY.txt** - Quick fix summary

## Installation

### New Installation

```bash
pipx install algorythm
```

### Upgrade from v0.3.0

```bash
pipx upgrade algorythm
```

Or if installed with pip:

```bash
pip install --upgrade algorythm
```

## Quick Start

### CLI Usage

```bash
# Show version
algorythm --version

# Create visualization
algorythm visualize my_song.mp3

# Show audio info
algorythm info my_song.mp3

# Custom settings
algorythm visualize song.mp3 -v waveform -w 1280 --height 720 --duration 30
```

### Python Usage

```python
# Load and visualize audio file
from algorythm.audio_loader import AudioFile
from algorythm.visualization import CircularVisualizer

audio = AudioFile('song.mp3')
viz = CircularVisualizer(sample_rate=audio.sample_rate)
audio.visualize('output.mp4', visualizer=viz)
```

## Dependencies

### Required
- numpy >= 1.19.0
- Pillow >= 8.0.0

### Optional
- **pydub** - For MP3/OGG/FLAC loading
- **opencv-python** - For faster video rendering
- **matplotlib** - For colored spectrograms

### System
- **ffmpeg** - For video export (required)

## Supported Formats

### Input Audio
- WAV (always supported)
- MP3, OGG, FLAC (with pydub)
- M4A, WMA (with pydub + ffmpeg)

### Output Video
- MP4 with H.264 video and AAC audio

## All Visualizers

All 6 visualizers now working:

1. ✅ **WaveformVisualizer** - Traditional waveform
2. ✅ **SpectrogramVisualizer** - Frequency over time (FIXED)
3. ✅ **FrequencyScopeVisualizer** - Real-time bars (FIXED)
4. ✅ **CircularVisualizer** - Circular/radial display
5. ✅ **OscilloscopeVisualizer** - Oscilloscope style
6. ✅ **ParticleVisualizer** - Particle physics

## Backwards Compatibility

✅ **100% Compatible**

All existing code from v0.3.0 works without changes. This release only adds new features.

## Breaking Changes

None! This is a pure feature addition release.

## Performance

- WAV loading: Very fast (built-in)
- MP3 loading: Fast (with pydub)
- Video rendering: 
  - PIL: ~2 fps (fallback)
  - OpenCV: ~30-60 fps (recommended)

## Known Issues

None reported.

## Use Cases

### Music Videos
```bash
algorythm visualize my_track.mp3 -v circular
```

### Podcast Clips
```bash
algorythm visualize podcast.mp3 --duration 30 -v waveform
```

### Social Media
```bash
algorythm visualize song.mp3 -w 1080 --height 1080 --offset 60 --duration 15
```

### Audio Analysis
```bash
algorythm visualize audio.mp3 -v spectrogram --offset 90 --duration 30
```

## Testing

All features tested and verified:
- ✅ Audio file loading (WAV, MP3, OGG, FLAC)
- ✅ All 6 visualizers working
- ✅ CLI commands functional
- ✅ Video export working
- ✅ Partial loading working
- ✅ Multiple visualizations from one file

## Upgrading

After upgrading, test the new features:

```bash
# Verify version
algorythm --version

# Test visualization
algorythm visualize ~/Music/test.wav -v circular --duration 5

# Test info command
algorythm info ~/Music/test.wav
```

## Examples Gallery

Videos created during testing:
- test_frequency_scope.mp4 - Frequency visualization
- test_spectrogram.mp4 - Spectrogram display
- visualized_quick.mp4 - Quick circular visualization
- visualized_waveform.mp4 - HD waveform
- multi_circular.mp4 - Circular bars
- cli_final_test.mp4 - CLI-created video

## Contributors

- Fixed FrequencyScopeVisualizer and SpectrogramVisualizer
- Implemented audio file loading
- Enhanced CLI with visualization commands
- Created comprehensive documentation

## Future Plans

Potential features for v0.5.0:
- Real-time visualization
- Custom color schemes via CLI
- Batch processing command
- Progress bars for long renders
- Audio format conversion
- Video filters and effects

## Changelog Summary

```
v0.4.0 (October 2024)
  Added:
    + Audio file loading (MP3, WAV, OGG, FLAC)
    + visualize_audio_file() one-line function
    + AudioFile class for object-oriented usage
    + Enhanced CLI with visualize and info commands
    + CLI support for 5 visualizer types
    + Custom resolution, FPS, duration in CLI
    + Example 06: Visualize audio files
    + CLI_GUIDE.md documentation
    + AUDIO_FILE_VISUALIZATION.md reference
  
  Fixed:
    * FrequencyScopeVisualizer now renders properly
    * SpectrogramVisualizer now renders properly
    * Both have working to_image_data() methods
  
  Updated:
    * Version 0.3.0 → 0.4.0
    * CLI with subcommands and better help
    * Package metadata and dependencies
    * Documentation and examples
```

## Thank You

Thank you for using Algorythm! Create amazing music visualizations! 🎵🎬✨

## Links

- GitHub: https://github.com/KiyomiLovesOreos/algorythm
- Documentation: See README.md and docs in repository
- Examples: See examples/ directory

## Support

For issues or questions:
1. Check the documentation (CLI_GUIDE.md, AUDIO_FILE_VISUALIZATION.md)
2. Run example scripts (algorythm --example basic)
3. Open an issue on GitHub

---

**Version**: 0.4.0  
**Release Date**: October 2024  
**Status**: Production Ready ✅
