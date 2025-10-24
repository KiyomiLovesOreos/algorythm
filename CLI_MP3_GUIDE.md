# Algorythm CLI - MP3 Visualization Guide

## Overview

The Algorythm CLI now has full support for MP3 (and other audio formats) with enhanced visualization options.

## Installation

```bash
# Install algorythm with MP3 support
pip install algorythm

# Make sure ffmpeg is installed
# Ubuntu/Debian:
sudo apt install ffmpeg

# macOS:
brew install ffmpeg

# Windows:
# Download from https://ffmpeg.org/
```

## Quick Start

### Check Supported Formats

```bash
algorythm formats
```

This shows:
- Which audio formats are supported (WAV, MP3, OGG, FLAC, etc.)
- Which dependencies are installed (pydub, ffmpeg, opencv, matplotlib)
- Installation instructions if anything is missing

### Get Audio File Info

```bash
algorythm info song.mp3
```

Shows:
- Duration, sample rate, file size
- Format details
- Sample count

### Create Visualization

Basic usage:
```bash
algorythm visualize song.mp3
```

With options:
```bash
# Choose visualizer type
algorythm visualize song.mp3 -v circular
algorythm visualize song.mp3 -v waveform
algorythm visualize song.mp3 -v spectrum
algorythm visualize song.mp3 -v spectrogram
algorythm visualize song.mp3 -v oscilloscope

# Set output file
algorythm visualize song.mp3 -o my_video.mp4

# Customize appearance
algorythm visualize song.mp3 --color purple --background dark
algorythm visualize song.mp3 --bars 128  # More frequency bars

# Set video parameters
algorythm visualize song.mp3 -w 1920 --height 1080 --fps 60

# Process part of the file
algorythm visualize song.mp3 --offset 30 --duration 60  # 60s starting at 30s
```

## Visualizer Types

| Type | Description | Best For |
|------|-------------|----------|
| `circular` | Circular frequency bars (default) | Music with strong beat |
| `waveform` | Audio waveform over time | Detailed amplitude view |
| `spectrum` | Frequency spectrum scope | Frequency content analysis |
| `spectrogram` | Time-frequency heatmap | Classical, ambient |
| `oscilloscope` | Real-time oscilloscope | Technical analysis |

## Color Options

**Foreground colors:**
- `blue`, `red`, `green`, `purple`, `orange`, `cyan` (default), `magenta`

**Background colors:**
- `black` (default), `white`, `dark`, `light`

## Advanced Examples

### High-quality music video
```bash
algorythm visualize song.mp3 \
  -v circular \
  --bars 128 \
  --color purple \
  --background dark \
  -w 3840 --height 2160 \
  --fps 60 \
  -o song_4k.mp4
```

### Preview a section
```bash
algorythm visualize long_song.mp3 \
  --offset 60 \
  --duration 30 \
  -o preview.mp4
```

### Multiple visualizations
```bash
# Create different versions
algorythm visualize song.mp3 -v circular -o song_circular.mp4
algorythm visualize song.mp3 -v waveform -o song_waveform.mp4
algorythm visualize song.mp3 -v spectrogram -o song_spectrogram.mp4
```

## Troubleshooting

### "pydub not found" error
```bash
pip install pydub
```

### "ffmpeg not found" error
- **Ubuntu/Debian:** `sudo apt install ffmpeg`
- **macOS:** `brew install ffmpeg`
- **Windows:** Download from https://ffmpeg.org/

### Video export fails
```bash
# Install video backend
pip install opencv-python

# Or use matplotlib (slower but no C dependencies)
pip install matplotlib
```

### Debug mode
```bash
algorythm visualize song.mp3 --debug
```

Shows detailed error messages and processing information.

## Python API

You can also use the same functionality in Python:

```python
from algorythm.audio_loader import AudioFile
from algorythm.visualization import CircularVisualizer

# Load MP3
audio = AudioFile('song.mp3')

# Create visualizer
viz = CircularVisualizer(sample_rate=audio.sample_rate, num_bars=128)

# Export video
audio.visualize('output.mp4', visualizer=viz, video_width=1920, video_height=1080)
```

## Features Added

✅ Full MP3/OGG/FLAC support via pydub
✅ Better error messages with installation instructions
✅ `formats` command to check dependencies
✅ Customizable colors and backgrounds
✅ Debug mode for troubleshooting
✅ More visualizer options
✅ Configurable bar count for circular visualizer
✅ Offset and duration support
✅ Automatic ffmpeg detection

## Notes

- MP3 support requires both **pydub** and **ffmpeg**
- Video export requires **ffmpeg** and either **opencv-python** or **matplotlib**
- WAV files work without any extra dependencies
- Processing time depends on video length and resolution
- Higher FPS and resolution = longer processing time
