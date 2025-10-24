# 🎵 Algorythm v0.4.0 - START HERE

## Welcome!

You've successfully upgraded Algorythm to v0.4.0 with enhanced visualizations and working MP4 video export!

## Quick Test (30 seconds)

```bash
cd /home/yurei/Projects/algorythm/examples
python 01_basic_melodies.py
```

This will create several audio files in `~/Music/` to verify everything works.

## What's New?

1. **Debug Mode** - See what's happening under the hood
2. **Enhanced Visualizations** - Better looking waveforms
3. **MP4 Video Export** - Actually creates real videos!
4. **5 New Examples** - Learn by doing
5. **Complete Documentation** - Everything you need to know

## Next Steps

### 1. Run the Examples (Recommended Order)

```bash
cd /home/yurei/Projects/algorythm/examples

# Start here - basics
python 01_basic_melodies.py

# Learn sound design
python 02_filters_and_effects.py

# Create videos (takes a few minutes)
python 03_video_visualizations.py

# Try algorithmic composition
python 04_generative_music.py

# Explore export options
python 05_export_formats.py
```

### 2. Read the Documentation

- **INSTALLATION_AND_EXAMPLES.md** - Complete guide
- **examples/README.md** - Example overview
- **VISUALIZATION_QUICK_REF.md** - Feature reference

### 3. Try It in Your Code

```python
from algorythm.synth import Synth
from algorythm.export import Exporter
from algorythm.visualization import CircularVisualizer

# Create music
synth = Synth(waveform='saw')
signal = synth.generate_note(frequency=440, duration=2.0)

# Export as video!
exporter = Exporter()
viz = CircularVisualizer(sample_rate=44100)
exporter.export(signal, 'my_first_video.mp4', visualizer=viz)
```

## Output Location

All files are saved to `~/Music/` by default:
- Audio files: WAV, MP3, OGG, FLAC
- Video files: MP4

## Need Help?

- Check the examples - they show how to use every feature
- Use debug mode: `WaveformVisualizer(debug=True)`
- Read error messages - they include installation instructions
- Check the documentation in the project directory

## Features at a Glance

### Audio Synthesis
- 4 waveforms: sine, square, saw, triangle
- Filters: lowpass, highpass, bandpass
- ADSR envelopes
- Multiple oscillators

### Visualization
- 6 visualizer types
- Debug mode
- Enhanced rendering
- Progress tracking

### Export
- WAV, MP3, OGG, FLAC
- MP4 video with visualization
- Custom quality settings
- Multiple backends

### Examples
- Basic melodies
- Filter effects
- Video creation
- Generative music
- Export options

## What to Explore

1. **Beginners**: Start with 01_basic_melodies.py
2. **Sound Designers**: Check out 02_filters_and_effects.py
3. **Video Creators**: Try 03_video_visualizations.py
4. **Algorithmic Music**: Explore 04_generative_music.py
5. **Developers**: Read the source code and documentation

## Version Info

- **Version**: 0.4.0
- **Status**: Production Ready ✅
- **Python**: 3.7+
- **Installation**: pipx (user-level)

## Have Fun!

Create amazing music and visualizations with Algorythm! 🎵🎬✨

---

**Quick Links:**
- Examples: `/home/yurei/Projects/algorythm/examples/`
- Documentation: See *.md files in project root
- Output: `~/Music/`
