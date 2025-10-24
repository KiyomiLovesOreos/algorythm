# Algorythm CLI Quick Reference

## Basic Commands

```bash
# Show help
algorythm --help
algorythm visualize --help

# Check version
algorythm --version

# Check supported formats and dependencies
algorythm formats

# Show audio file info
algorythm info song.mp3
```

## Visualize Audio

### Simple
```bash
algorythm visualize song.mp3
```

### Choose Visualizer
```bash
algorythm visualize song.mp3 -v waveform
algorythm visualize song.mp3 -v circular
algorythm visualize song.mp3 -v spectrum
algorythm visualize song.mp3 -v spectrogram
algorythm visualize song.mp3 -v oscilloscope
```

### Customize Appearance
```bash
# Colors: blue, red, green, purple, orange, cyan, magenta
algorythm visualize song.mp3 --color purple

# Backgrounds: black, white, dark, light
algorythm visualize song.mp3 --background dark

# Number of bars (circular visualizer)
algorythm visualize song.mp3 --bars 128

# Combine options
algorythm visualize song.mp3 --color purple --background dark --bars 128
```

### Video Settings
```bash
# Set resolution
algorythm visualize song.mp3 -w 3840 --height 2160  # 4K

# Set frame rate
algorythm visualize song.mp3 --fps 60

# Set output file
algorythm visualize song.mp3 -o my_video.mp4
```

### Time Range
```bash
# Skip first 30 seconds
algorythm visualize song.mp3 --offset 30

# Process 60 seconds
algorythm visualize song.mp3 --duration 60

# Process 60s starting at 30s
algorythm visualize song.mp3 --offset 30 --duration 60
```

### Debug
```bash
algorythm visualize song.mp3 --debug
```

## Complete Examples

### HD YouTube Video
```bash
algorythm visualize song.mp3 \
  -v circular \
  --bars 128 \
  --color cyan \
  --background black \
  -w 1920 --height 1080 \
  --fps 30 \
  -o youtube.mp4
```

### 4K Music Video
```bash
algorythm visualize song.mp3 \
  -v circular \
  --bars 256 \
  --color purple \
  --background dark \
  -w 3840 --height 2160 \
  --fps 60 \
  -o 4k_video.mp4
```

### Quick Preview
```bash
algorythm visualize long_song.mp3 \
  --offset 60 \
  --duration 30 \
  -w 1280 --height 720 \
  -o preview.mp4
```

### Waveform Style
```bash
algorythm visualize song.mp3 \
  -v waveform \
  --color green \
  --background black \
  -o waveform.mp4
```

### Spectrogram Analysis
```bash
algorythm visualize song.mp3 \
  -v spectrogram \
  --background black \
  -o spectrogram.mp4
```

## Supported Formats

**Input:** MP3, WAV, OGG, FLAC, M4A, AAC
**Output:** MP4

## Requirements

- **Python:** 3.7+
- **Required:** numpy, pydub, Pillow
- **For MP3:** ffmpeg
- **For video:** opencv-python OR matplotlib

## Installation

```bash
# Install algorythm
pip install algorythm

# Install ffmpeg
# Ubuntu/Debian:
sudo apt install ffmpeg

# macOS:
brew install ffmpeg

# Optional: video backend
pip install opencv-python
```

## Troubleshooting

```bash
# Check what's installed
algorythm formats

# Debug mode
algorythm visualize song.mp3 --debug

# Test with info first
algorythm info song.mp3
```

## Color Combinations

**Vibrant:**
- `--color magenta --background black`
- `--color orange --background dark`

**Professional:**
- `--color cyan --background black`
- `--color blue --background dark`

**Retro:**
- `--color green --background black`
- `--color purple --background dark`

**Light Mode:**
- `--color blue --background light`
- `--color purple --background white`
