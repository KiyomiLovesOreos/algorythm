# Changelog

All notable changes to Algorythm will be documented in this file.

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
