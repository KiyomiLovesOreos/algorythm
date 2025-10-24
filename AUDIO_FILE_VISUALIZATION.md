# Audio File Visualization Feature

## Overview

You can now load existing audio files (MP3, WAV, OGG, FLAC) and create visualization videos from them!

## Features

✅ Load audio from common formats (MP3, WAV, OGG, FLAC)
✅ Apply any visualizer to your audio
✅ Load partial audio (offset and duration)
✅ Resample to different sample rates
✅ Create multiple visualizations from same audio
✅ Simple one-line API for quick results
✅ Advanced API for full control

## Quick Start

### Method 1: One-Line Visualization

```python
from algorythm.audio_loader import visualize_audio_file
from algorythm.visualization import CircularVisualizer

viz = CircularVisualizer(sample_rate=44100, num_bars=64)

visualize_audio_file(
    input_file='my_song.mp3',
    output_file='my_song_video.mp4',
    visualizer=viz
)
```

### Method 2: AudioFile Class

```python
from algorythm.audio_loader import AudioFile
from algorythm.visualization import WaveformVisualizer

# Load audio
audio = AudioFile('my_song.mp3')
print(f"Duration: {audio.duration:.2f}s")

# Create visualization
viz = WaveformVisualizer(sample_rate=audio.sample_rate)
audio.visualize('output.mp4', visualizer=viz)
```

### Method 3: Manual Control

```python
from algorythm.audio_loader import load_audio
from algorythm.export import Exporter
from algorythm.visualization import SpectrogramVisualizer

# Load audio
signal, sample_rate = load_audio('my_song.mp3')

# Create visualization
viz = SpectrogramVisualizer(sample_rate=sample_rate)
exporter = Exporter()
exporter.export(
    signal,
    'output.mp4',
    sample_rate=sample_rate,
    visualizer=viz
)
```

## API Reference

### visualize_audio_file()

One-line function to create visualizations.

```python
visualize_audio_file(
    input_file: str,        # Path to audio file
    output_file: str,       # Path to output MP4
    visualizer,             # Visualizer instance
    video_width: int = 1920,
    video_height: int = 1080,
    video_fps: int = 30,
    offset: float = 0.0,    # Start offset in seconds
    duration: float = None  # Duration to process (None = all)
)
```

### AudioFile Class

Object-oriented interface for audio files.

```python
# Constructor
audio = AudioFile(
    filepath: str,
    target_sample_rate: int = None,  # Resample if needed
    duration: float = None,          # Load only N seconds
    offset: float = 0.0              # Skip first N seconds
)

# Properties
audio.duration      # Duration in seconds
audio.sample_rate   # Sample rate in Hz
audio.num_samples   # Number of samples
audio.signal        # Raw audio data (numpy array)

# Methods
audio.visualize(
    output_file: str,
    visualizer,
    video_width: int = 1920,
    video_height: int = 1080,
    video_fps: int = 30
)
```

### load_audio()

Low-level function to load audio data.

```python
signal, sample_rate = load_audio(
    filepath: str,
    target_sample_rate: int = None,  # Resample to this rate
    duration: float = None,          # Load only N seconds
    offset: float = 0.0              # Start from N seconds
)
```

## Examples

### Example 1: Quick Visualization

```python
from algorythm.audio_loader import visualize_audio_file
from algorythm.visualization import CircularVisualizer

viz = CircularVisualizer(sample_rate=44100)
visualize_audio_file('song.mp3', 'song_video.mp4', viz)
```

### Example 2: Load Partial Audio

```python
from algorythm.audio_loader import AudioFile

# Load only 30 seconds starting at 1 minute
audio = AudioFile('song.mp3', offset=60.0, duration=30.0)
print(f"Loaded {audio.duration}s of audio")
```

### Example 3: Multiple Visualizations

```python
from algorythm.audio_loader import AudioFile
from algorythm.visualization import *

# Load once
audio = AudioFile('song.mp3')

# Create multiple videos
visualizers = {
    'waveform': WaveformVisualizer(sample_rate=audio.sample_rate),
    'circular': CircularVisualizer(sample_rate=audio.sample_rate),
    'spectrum': FrequencyScopeVisualizer(sample_rate=audio.sample_rate),
}

for name, viz in visualizers.items():
    audio.visualize(f'song_{name}.mp4', visualizer=viz)
```

### Example 4: Resample Audio

```python
from algorythm.audio_loader import load_audio

# Force specific sample rate
signal, sample_rate = load_audio(
    'song.mp3',
    target_sample_rate=48000  # Resample to 48kHz
)
```

### Example 5: High Quality Settings

```python
from algorythm.audio_loader import visualize_audio_file
from algorythm.visualization import WaveformVisualizer

viz = WaveformVisualizer(sample_rate=44100)
visualize_audio_file(
    'song.mp3',
    'song_4k.mp4',
    visualizer=viz,
    video_width=3840,   # 4K
    video_height=2160,
    video_fps=60        # High framerate
)
```

## Supported Formats

### Always Supported
- **WAV** - Uncompressed audio (built-in support)

### With pydub Installed
- **MP3** - MPEG Audio Layer 3
- **OGG** - Ogg Vorbis
- **FLAC** - Free Lossless Audio Codec
- **M4A** - MPEG-4 Audio
- **WMA** - Windows Media Audio (with ffmpeg)
- And more...

Install pydub:
```bash
pip install pydub
```

## Supported Visualizers

All visualizers work with audio files:

1. **WaveformVisualizer** - Traditional waveform
2. **SpectrogramVisualizer** - Frequency over time
3. **FrequencyScopeVisualizer** - Real-time bars
4. **CircularVisualizer** - Circular/radial display
5. **OscilloscopeVisualizer** - Oscilloscope style
6. **ParticleVisualizer** - Particle physics

## Performance

- **WAV Loading**: Very fast (built-in)
- **MP3 Loading**: Fast (requires pydub)
- **Video Rendering**: Same as synthesized audio
  - PIL backend: ~2 frames/second
  - OpenCV: ~30-60 frames/second

## Use Cases

### Music Videos
Create visualizations for your music tracks:
```python
visualize_audio_file('my_track.mp3', 'my_video.mp4', viz)
```

### Podcast Visualizations
Add visuals to podcast episodes:
```python
# First 5 minutes with waveform
audio = AudioFile('podcast.mp3', duration=300)
viz = WaveformVisualizer(sample_rate=audio.sample_rate)
audio.visualize('podcast_intro.mp4', visualizer=viz)
```

### Audio Analysis
Visualize specific sections of audio:
```python
# Analyze chorus (1:30 to 2:00)
audio = AudioFile('song.mp3', offset=90, duration=30)
viz = SpectrogramVisualizer(sample_rate=audio.sample_rate)
audio.visualize('chorus_analysis.mp4', visualizer=viz)
```

### Social Media Content
Create short clips with visualizations:
```python
# 15-second Instagram clip
audio = AudioFile('song.mp3', offset=45, duration=15)
viz = CircularVisualizer(sample_rate=audio.sample_rate)
audio.visualize(
    'instagram_clip.mp4',
    visualizer=viz,
    video_width=1080,
    video_height=1080  # Square for Instagram
)
```

## Error Handling

### File Not Found
```python
try:
    audio = AudioFile('song.mp3')
except FileNotFoundError:
    print("Audio file not found!")
```

### Unsupported Format
```python
try:
    audio = AudioFile('song.mp3')
except ImportError as e:
    print("Install pydub for MP3 support:")
    print("  pip install pydub")
```

## Complete Example

```python
from algorythm.audio_loader import AudioFile
from algorythm.visualization import CircularVisualizer

# Load audio
print("Loading audio...")
audio = AudioFile('my_song.mp3')

# Show info
print(f"Title: my_song.mp3")
print(f"Duration: {audio.duration:.2f}s")
print(f"Sample rate: {audio.sample_rate}Hz")
print(f"Samples: {audio.num_samples:,}")

# Create visualization
print("\nCreating video...")
viz = CircularVisualizer(
    sample_rate=audio.sample_rate,
    num_bars=64,
    inner_radius=0.3
)

audio.visualize(
    'my_song_video.mp4',
    visualizer=viz,
    video_width=1920,
    video_height=1080,
    video_fps=30
)

print("✓ Done! Video saved to ~/Music/my_song_video.mp4")
```

## Tips

1. **Use offset/duration** for large files to reduce processing time
2. **Resample to 44.1kHz** for consistency across different sources
3. **Try different visualizers** to find what works best for your audio
4. **Use square dimensions** (1080x1080) for social media
5. **Higher FPS** (60) looks smoother but takes longer to render

## Troubleshooting

### MP3 files not loading
Install pydub: `pip install pydub`

### Video not creating
Install ffmpeg: `sudo apt install ffmpeg` (Linux) or `brew install ffmpeg` (macOS)

### Slow rendering
- Use offset/duration to process only what you need
- Consider installing opencv-python for faster rendering
- Lower the FPS or resolution for drafts

## See Also

- `examples/06_visualize_audio_files.py` - Complete examples
- `VISUALIZATION_FIX.md` - Visualization improvements
- `MP4_EXPORT_COMPLETE.md` - Video export details

## Status

🎉 **PRODUCTION READY**

Load audio files and create visualizations easily! Perfect for:
- Music videos
- Podcast visuals
- Audio analysis
- Social media content
- Educational materials
