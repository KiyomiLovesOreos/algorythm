# Changelog

All notable changes to Algorythm will be documented in this file.

## [Unreleased]

### Fixed - MP3 Recognition and Enhanced CLI (2025-10-24)

#### CLI Color Parameter Bug Fix
- **Fixed TypeError**: Fixed `Exporter.export() got an unexpected keyword argument 'background_color'` error
- **Correct Parameter Flow**: Color parameters now correctly passed to `VideoRenderer` instead of `Exporter`
- **Working Color Options**: `--color` and `--background` flags now function properly
- **Fixed ffmpeg Subprocess**: Fixed `capture_output` conflict in subprocess call (line 901)

#### Dependency Management
- **Fixed pydub Installation**: Added `pydub>=0.25.1` to `install_requires` in setup.py (previously only in extras_require)
- **Automatic MP3 Support**: MP3/OGG/FLAC support now installs automatically with algorythm
- **Better Error Messages**: Enhanced error handling with platform-specific installation instructions

#### Audio Loading Improvements
- **Added HAS_PYDUB Flag**: Module-level flag for early pydub detection
- **ffmpeg Detection**: Automatic detection of ffmpeg/avconv availability
- **Enhanced Error Handling**: Clear, actionable error messages when dependencies are missing
- **Multiple Error Types**: Separate handling for ImportError, RuntimeError, and FileNotFoundError

#### CLI Enhancements
- **New Command: `formats`**: Check supported formats and dependency status
  - Shows audio input formats (WAV, MP3, OGG, FLAC, M4A, AAC)
  - Shows video output formats
  - Displays dependency status (pydub, ffmpeg, opencv, matplotlib)
  - Provides installation instructions for missing dependencies

- **Enhanced `visualize` Command**:
  - `--color` option: Choose foreground color (blue, red, green, purple, orange, cyan, magenta)
  - `--background` option: Choose background color (black, white, dark, light)
  - `--bars` option: Customize bar count for circular visualizer
  - `--debug` flag: Enable debug mode with detailed error messages
  - Better progress indicators and status messages

- **Improved `info` Command**:
  - Enhanced error handling
  - Better formatting
  - Clearer dependency messages

#### Documentation
- **CLI_MP3_GUIDE.md**: Complete guide for MP3 visualization with CLI
- **CLI_QUICK_REF.md**: Quick reference card for all CLI commands
- **BUGFIX_MP3_CLI_SUMMARY.md**: Detailed summary of changes

#### Example Usage
```bash
# Check what's supported
algorythm formats

# Create colorful visualization
algorythm visualize song.mp3 --color purple --background dark --bars 128

# Process section of file
algorythm visualize song.mp3 --offset 30 --duration 60

# High-quality 4K video
algorythm visualize song.mp3 -w 3840 --height 2160 --fps 60
```

### Added - MP4 Export for Visualizers

#### Direct MP4 Export
- **MP4 Format Support**: Export audio with synchronized visualizations to MP4 video files
- **Integrated with Exporter**: Use `Exporter.export()` with `.mp4` file extension
- **Multiple Visualizers**: Support for all 6 built-in visualizers
- **Automatic Fallback**: Uses default WaveformVisualizer if none specified

#### Video Configuration
- **Customizable Dimensions**: Set video width and height (HD, Full HD, 4K, custom)
- **Frame Rate Control**: Adjustable FPS (24, 30, 60, etc.)
- **Progress Indicators**: Real-time progress feedback during rendering
- **Quality Settings**: Uses ffmpeg with optimized encoding settings

#### Supported Visualizers
- WaveformVisualizer - Shows amplitude over time
- SpectrogramVisualizer - Displays frequency content as heatmap
- FrequencyScopeVisualizer - Real-time frequency spectrum bars
- CircularVisualizer - Radial frequency display
- OscilloscopeVisualizer - Classic oscilloscope views
- ParticleVisualizer - Animated particles reacting to audio

#### Documentation
- **MP4_EXPORT_GUIDE.md**: Complete guide with examples and best practices
- **examples/mp4_export_example.py**: 5 detailed examples showcasing different visualizers
- **Test Coverage**: Added tests for MP4 export functionality

#### Dependencies
- Requires `opencv-python` and `ffmpeg` for MP4 export
- Graceful fallback to WAV export if dependencies missing
- Optional install via `pip install -e ".[video]"`

## [0.3.0] - 2025-10-21

### Added - Volume Control System

#### Track and Master Volume
- **Track Volume Control**: Set individual track volumes via `Composition.set_track_volume(name, volume)`
- **Master Volume Control**: Control overall composition volume via `Composition.set_master_volume(volume)`
- **Method Chaining**: All volume methods support fluent API chaining

#### Fade Effects
- **Fade In/Out**: Add smooth fades via `Composition.fade_in(duration)` and `fade_out(duration)`
- Applied automatically during rendering
- Integrated with composition workflow

#### VolumeControl Utility Class
- **dB Conversions**: `db_to_linear()` and `linear_to_db()` static methods
- **Volume Application**: `apply_volume()` and `apply_db_volume()` for signal processing
- **Normalization**: `normalize(signal, target_db)` for level control
- **Advanced Fades**: `fade()` with three curve types:
  - `'linear'` - Constant rate fade
  - `'exponential'` - Smooth, natural fade
  - `'logarithmic'` - Perception-based fade

#### Playback Volume Control
- **AudioPlayer**: Added `set_volume()` method for real-time volume control
- **StreamingPlayer**: Added `set_volume()` method for streaming audio
- Volume applied before output to prevent clipping

### Added - Export to ~/Music Directory

#### Default Directory
- **Automatic Organization**: Files now save to `~/Music` by default
- **Subdirectory Support**: Relative paths with subdirectories automatically created
- **Path Resolution**: Smart resolution of relative vs absolute paths

#### Exporter Enhancements
- **Custom Default Directory**: New `default_directory` parameter in `Exporter()`
- **Automatic Directory Creation**: Creates `~/Music` and subdirectories as needed
- **Path Feedback**: Prints full path where files are saved
- **Absolute Path Support**: Absolute paths still work for specific locations

#### Integration
- Works seamlessly with `Composition.render()`
- All export methods updated (`export()`, `export_stereo()`)
- Parent directories created automatically

### Documentation

#### New Files
- `VOLUME_CONTROL.md` - Comprehensive volume control guide
- `VOLUME_CONTROL_QUICKREF.md` - Quick reference card
- `VOLUME_CONTROL_SUMMARY.md` - Implementation details
- `EXPORT_MUSIC_FOLDER.md` - Export behavior guide
- `EXPORT_UPDATE_SUMMARY.md` - Export implementation details
- `RECENT_UPDATES.md` - Combined feature overview
- `CHANGELOG.md` - This file

#### Updated Files
- `README.md` - Added volume control and export documentation
- `examples/README.md` - Added volume control demo description
- `algorythm/__init__.py` - Export VolumeControl class

#### Demo Scripts
- `examples/volume_control_demo.py` - Comprehensive demonstration of all volume features

### Changed
- **Exporter Behavior**: Relative paths now save to `~/Music` instead of current directory
- **Export Feedback**: All export methods now print the full output path
- **Composition Rendering**: Now applies master volume and fades during render

### Technical Details
- Volume application order: Track effects → Track volume → Mix → Master volume → Fades → Normalization
- Path resolution: Absolute paths used as-is, relative paths prepended with default directory
- All changes are backwards compatible with absolute path usage

### Migration Notes
For users upgrading from 0.2.x:
- Relative export paths now save to `~/Music` by default
- To save in current directory: use absolute paths or `Exporter(default_directory='.')`
- Track volume attribute was already present, now accessible via API
- No breaking changes to existing code using absolute paths

---

## [0.2.0] - Previous Release

### Added
- FM Synthesis (`FMSynth`)
- Wavetable Synthesis (`WavetableSynth`)
- Live GUI for real-time composition
- Enhanced effects (Bitcrusher, Tremolo, Phaser)
- Real-time playback support
- Visualization enhancements

---

## [0.1.0] - Initial Release

### Added
- Core synthesis engine (Oscillator, Filter, ADSR, Synth)
- Sequence module (Motif, Scale, Chord, Arpeggiator)
- Structure module (Track, Composition, Effects)
- Export module (WAV, FLAC, MP3, OGG support)
- Generative composition tools (L-System, Cellular Automata)
- Automation and data sonification
- Audio visualization
- Sample playback and manipulation
- Basic CLI
