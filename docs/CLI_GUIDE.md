# Algorythm CLI Guide

## Overview

Algorythm now includes a powerful command-line interface (CLI) for quick audio visualization and information without writing code!

## Installation

The CLI is automatically installed when you install Algorythm:

```bash
pipx install algorythm
```

## Available Commands

### Show Version

```bash
algorythm --version
```

Shows the installed version of Algorythm.

### Show Help

```bash
algorythm --help
```

Shows all available commands and options.

### Visualize Audio Files

Create visualization videos from audio files:

```bash
algorythm visualize <input_file> [options]
```

**Basic Usage:**

```bash
# Create visualization with default settings
algorythm visualize song.mp3

# Specify output file
algorythm visualize song.mp3 -o my_video.mp4

# Choose visualizer type
algorythm visualize song.mp3 -v waveform
algorythm visualize song.mp3 -v circular
algorythm visualize song.mp3 -v spectrum
algorythm visualize song.mp3 -v spectrogram
algorythm visualize song.mp3 -v oscilloscope
```

**Advanced Options:**

```bash
# Custom resolution and FPS
algorythm visualize song.mp3 -w 3840 --height 2160 --fps 60

# Process only part of the file
algorythm visualize song.mp3 --offset 60 --duration 30

# 720p waveform video
algorythm visualize song.mp3 -v waveform -w 1280 --height 720

# Square format for Instagram
algorythm visualize song.mp3 -w 1080 --height 1080
```

**All Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--output` | `-o` | Output video file | `input_video.mp4` |
| `--visualizer` | `-v` | Visualizer type | `circular` |
| `--width` | `-w` | Video width | `1920` |
| `--height` | | Video height | `1080` |
| `--fps` | | Frames per second | `30` |
| `--offset` | | Start offset in seconds | `0.0` |
| `--duration` | | Duration to process | entire file |

**Visualizer Types:**

- `waveform` - Traditional waveform display
- `circular` - Circular/radial bars (default)
- `spectrum` - Real-time frequency bars
- `spectrogram` - Frequency over time heatmap
- `oscilloscope` - Oscilloscope-style display

### Show Audio Information

Display information about an audio file:

```bash
algorythm info <audio_file>
```

**Example:**

```bash
algorythm info song.mp3
```

**Output:**

```
============================================================
Audio File Information
============================================================
File:        song.mp3
Path:        /home/user/Music/song.mp3
Format:      MP3
Duration:    182.45 seconds
Sample rate: 44,100 Hz
Samples:     8,044,050
Size:        4,321.2 KB
Time:        3:02.45
============================================================
```

### Run Examples

Run built-in example scripts:

```bash
# Basic synthesis example
algorythm --example basic

# Composition example
algorythm --example composition

# Advanced example with multiple tracks
algorythm --example advanced
```

## Complete Examples

### Example 1: Quick Visualization

Create a video with default settings (circular visualizer, 1920x1080):

```bash
algorythm visualize my_song.mp3
```

Output: `my_song_video.mp4` in the current directory.

### Example 2: High-Quality 4K Video

Create a 4K video at 60fps:

```bash
algorythm visualize my_song.mp3 \
  -o my_song_4k.mp4 \
  -v waveform \
  -w 3840 \
  --height 2160 \
  --fps 60
```

### Example 3: Instagram-Ready Clip

Create a square video of a 15-second clip:

```bash
algorythm visualize my_song.mp3 \
  -o instagram_clip.mp4 \
  -v circular \
  -w 1080 \
  --height 1080 \
  --offset 45 \
  --duration 15
```

### Example 4: Podcast Intro

Create visualization for the first 30 seconds:

```bash
algorythm visualize podcast_episode.mp3 \
  -o podcast_intro.mp4 \
  -v waveform \
  -w 1280 \
  --height 720 \
  --duration 30
```

### Example 5: Multiple Visualizers

Create different versions with different visualizers:

```bash
# Waveform version
algorythm visualize song.mp3 -o song_waveform.mp4 -v waveform

# Circular version
algorythm visualize song.mp3 -o song_circular.mp4 -v circular

# Spectrum version
algorythm visualize song.mp3 -o song_spectrum.mp4 -v spectrum
```

### Example 6: Music Video Section

Visualize the chorus (1:30 to 2:00):

```bash
algorythm visualize song.mp3 \
  -o chorus.mp4 \
  -v spectrogram \
  --offset 90 \
  --duration 30
```

## Supported Formats

### Input Audio Formats

- **Always supported:** WAV
- **With pydub installed:** MP3, OGG, FLAC, M4A, WMA

To enable MP3/OGG/FLAC support:

```bash
pip install pydub
```

### Output Video Format

- MP4 with H.264 video and AAC audio

## Resolution Presets

Common video resolutions:

| Name | Width | Height | Command |
|------|-------|--------|---------|
| HD 720p | 1280 | 720 | `-w 1280 --height 720` |
| Full HD | 1920 | 1080 | `-w 1920 --height 1080` (default) |
| 2K | 2560 | 1440 | `-w 2560 --height 1440` |
| 4K | 3840 | 2160 | `-w 3840 --height 2160` |
| Instagram Square | 1080 | 1080 | `-w 1080 --height 1080` |
| YouTube Short | 1080 | 1920 | `-w 1080 --height 1920` |

## Performance Tips

1. **Use duration/offset** for large files to reduce processing time
2. **Lower resolution** for faster processing during testing
3. **Lower FPS** (e.g., `--fps 24`) for faster rendering
4. **Install opencv-python** for faster video rendering:
   ```bash
   pip install opencv-python
   ```

## Troubleshooting

### Command Not Found

If `algorythm` command is not found, ensure pipx bin directory is in PATH:

```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH="$HOME/.local/bin:$PATH"
```

Or reinstall:

```bash
pipx reinstall algorythm
```

### MP3 Files Not Working

Install pydub:

```bash
pip install pydub
```

### Video Not Creating

Install ffmpeg:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Fedora
sudo dnf install ffmpeg
```

### Slow Rendering

Try these options:

1. Use shorter duration: `--duration 10`
2. Lower resolution: `-w 1280 --height 720`
3. Lower FPS: `--fps 24`
4. Install opencv-python: `pip install opencv-python`

## Workflow Examples

### Music Producer Workflow

```bash
# 1. Check audio file info
algorythm info track.wav

# 2. Create preview (first 10 seconds)
algorythm visualize track.wav -o preview.mp4 --duration 10 -v circular

# 3. Create full HD version
algorythm visualize track.wav -o track_hd.mp4 -v circular

# 4. Create Instagram square version
algorythm visualize track.wav -o track_ig.mp4 -w 1080 --height 1080
```

### Podcast Workflow

```bash
# 1. Check episode length
algorythm info episode.mp3

# 2. Create intro clip (first minute)
algorythm visualize episode.mp3 \
  -o episode_intro.mp4 \
  -v waveform \
  -w 1280 --height 720 \
  --duration 60

# 3. Create highlight clip (best part)
algorythm visualize episode.mp3 \
  -o episode_highlight.mp4 \
  -v waveform \
  --offset 1200 \
  --duration 30
```

### DJ Set Workflow

```bash
# Visualize different sections
algorythm visualize dj_set.mp3 -o set_intro.mp4 --duration 30 -v spectrum
algorythm visualize dj_set.mp3 -o set_buildup.mp4 --offset 600 --duration 30 -v circular
algorythm visualize dj_set.mp3 -o set_drop.mp4 --offset 900 --duration 30 -v oscilloscope
```

## Integration with Scripts

Use the CLI in shell scripts:

```bash
#!/bin/bash
# Batch process all MP3 files in a directory

for file in *.mp3; do
  echo "Processing $file..."
  algorythm visualize "$file" \
    -o "${file%.mp3}_video.mp4" \
    -v circular \
    -w 1920 --height 1080
done
```

## Comparison with Python API

| Feature | CLI | Python API |
|---------|-----|------------|
| Quick visualization | ✅ Very easy | Requires code |
| File info | ✅ One command | Requires code |
| Custom processing | ❌ Limited | ✅ Full control |
| Batch processing | ✅ Shell scripts | ✅ Python loops |
| Integration | ✅ Any workflow | ✅ Python apps |
| Parameters | ✅ Preset options | ✅ Full customization |

Use the CLI for quick tasks, use the Python API for custom needs.

## See Also

- `AUDIO_FILE_VISUALIZATION.md` - Python API documentation
- `examples/06_visualize_audio_files.py` - Python examples
- `VISUALIZATION_QUICK_REF.md` - Visualizer reference

## Version

This guide is for Algorythm v0.4.0+

## Feedback

The CLI is designed to be simple and powerful. Enjoy creating music visualizations from the command line! 🎵🎬✨
