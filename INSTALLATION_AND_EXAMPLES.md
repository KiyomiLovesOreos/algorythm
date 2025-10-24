# Algorythm v0.4.0 - Installation & Examples Guide

## ✅ Installation Complete

The package has been successfully upgraded to **v0.4.0** and installed!

## What's New in v0.4.0

### 🎨 Enhanced Visualization
- Debug mode on all visualizers
- Enhanced waveform rendering (thick lines, center reference)
- Colored spectrograms with matplotlib colormaps
- Progress tracking for long operations
- Multiple backend support (OpenCV/matplotlib/PIL)

### 🎬 Working MP4 Export
- **Actually creates real video files!**
- Synchronized audio and video
- All 6 visualizers supported
- H.264/AAC encoding
- Progress feedback

### 📚 New Examples (5 numbered tutorials)
- 01_basic_melodies.py - Start here!
- 02_filters_and_effects.py
- 03_video_visualizations.py
- 04_generative_music.py
- 05_export_formats.py

## Quick Start

### Run Examples

```bash
cd /home/yurei/Projects/algorythm/examples

# Beginner - Start here!
python 01_basic_melodies.py

# Create a video
python 03_video_visualizations.py

# Test everything works
python quick_viz_test.py
```

### Use in Your Code

```python
from algorythm.synth import Synth, Filter, ADSR
from algorythm.export import Exporter
from algorythm.visualization import CircularVisualizer

# Create music
synth = Synth(
    waveform='saw',
    filter=Filter.lowpass(cutoff=2000, resonance=0.6),
    envelope=ADSR(attack=0.1, decay=0.2, sustain=0.7, release=0.5)
)
signal = synth.generate_note(frequency=440, duration=2.0)

# Export as MP4 video with visualization
exporter = Exporter()
visualizer = CircularVisualizer(sample_rate=44100, num_bars=64)

exporter.export(
    signal,
    'my_video.mp4',
    sample_rate=44100,
    visualizer=visualizer,
    video_width=1920,
    video_height=1080,
    video_fps=30
)
```

## Example Scripts Overview

### Beginner Level

**01_basic_melodies.py** ⭐ Start here!
- Simple 440Hz tone
- C major scale
- Different waveforms (sine, square, saw, triangle)
- Twinkle Twinkle Little Star

**02_filters_and_effects.py**
- Lowpass/highpass/bandpass filters
- ADSR envelopes (pluck, pad, organ)
- Filter sweeps

**05_export_formats.py**
- WAV, MP3, OGG, FLAC exports
- Quality settings
- Custom locations

### Intermediate Level

**03_video_visualizations.py**
- Create MP4 videos
- 4 visualization styles
- HD quality output
- Chord progressions

**04_generative_music.py**
- Random walk melodies
- Pentatonic patterns
- Euclidean rhythms
- Markov chains
- Arpeggiators

## Output Location

All files saved to `~/Music/` by default:
- Audio: WAV, MP3, OGG, FLAC
- Video: MP4

## Dependencies

### Already Installed
✓ numpy
✓ Pillow (PIL)

### Optional (for better performance)
```bash
# Faster video rendering
pip install --user opencv-python

# Alternative video backend + colored spectrograms
pip install --user matplotlib

# Compressed audio formats
pip install --user pydub
```

### System Requirements
```bash
# Required for video export
sudo apt install ffmpeg  # Ubuntu/Debian
brew install ffmpeg      # macOS
```

## Documentation

- **README_COMPLETE.md** - Quick start guide
- **MP4_EXPORT_COMPLETE.md** - Video export details
- **VISUALIZATION_IMPROVEMENTS.md** - All visualization features
- **VISUALIZATION_QUICK_REF.md** - Quick reference
- **examples/README.md** - Example index
- **UPGRADE_COMPLETE_SUMMARY.txt** - Full summary

## Testing

All features tested and verified:
```bash
# Quick validation (< 1 second)
python examples/quick_viz_test.py

# Comprehensive demo
python examples/comprehensive_viz_demo.py

# All visualizers
python examples/complete_mp4_demo.py
```

## Troubleshooting

### Import errors
```python
# Test imports
python -c "from algorythm.synth import Synth; print('OK')"
```

### Video export not working
- Check ffmpeg is installed: `ffmpeg -version`
- Check PIL available: `python -c "from PIL import Image; print('OK')"`
- Use debug mode: `WaveformVisualizer(debug=True)`

### Examples not found
```bash
# Make sure you're in the right directory
cd /home/yurei/Projects/algorythm/examples
ls *.py
```

## What's Next?

1. Run the beginner examples
2. Experiment with parameters
3. Create your own melodies
4. Try video export
5. Explore generative algorithms
6. Build something awesome!

## Support

- Check the documentation in the project directory
- Run examples to see features in action
- Use debug mode for troubleshooting
- All error messages include helpful instructions

## Version Info

- **Version**: 0.4.0
- **Python**: 3.7+
- **Installation**: pipx (user-level)
- **Status**: Production Ready ✅

Enjoy creating with Algorythm! 🎵🎬
