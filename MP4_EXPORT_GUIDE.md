# MP4 Export Quick Reference

## Overview
The MP4 export feature allows you to create video files with synchronized audio visualizations. This is perfect for sharing your algorithmic compositions on platforms like YouTube, social media, or for creating visual presentations of your music.

## Requirements
```bash
pip install opencv-python
```

You also need `ffmpeg` installed on your system:
- **Linux**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from https://ffmpeg.org/

## Basic Usage

### Simple Export with Default Visualizer
```python
from algorythm.export import Exporter
import numpy as np

# Create your audio signal
signal = np.sin(2 * np.pi * 440 * np.linspace(0, 3, 132300))

# Export as MP4 (uses default waveform visualizer)
exporter = Exporter()
exporter.export(signal, 'my_sound.mp4', sample_rate=44100)
```

### Export with Custom Visualizer
```python
from algorythm.export import Exporter
from algorythm.visualization import WaveformVisualizer

# Create a visualizer
visualizer = WaveformVisualizer(sample_rate=44100, window_size=2048)

# Export with custom settings
exporter = Exporter()
exporter.export(
    signal,
    'my_sound.mp4',
    sample_rate=44100,
    visualizer=visualizer,
    video_width=1920,   # Full HD width
    video_height=1080,  # Full HD height
    video_fps=30        # 30 frames per second
)
```

## Available Visualizers

### 1. WaveformVisualizer
Displays the audio waveform over time.
```python
from algorythm.visualization import WaveformVisualizer

visualizer = WaveformVisualizer(
    sample_rate=44100,
    window_size=2048,
    downsample_factor=1
)
```

### 2. SpectrogramVisualizer
Shows frequency content over time (spectrogram).
```python
from algorythm.visualization import SpectrogramVisualizer

visualizer = SpectrogramVisualizer(
    sample_rate=44100,
    window_size=2048,
    hop_size=512,
    window_type='hann'  # Options: 'hann', 'hamming', 'blackman'
)
```

### 3. FrequencyScopeVisualizer
Displays real-time frequency spectrum (like an equalizer).
```python
from algorythm.visualization import FrequencyScopeVisualizer

visualizer = FrequencyScopeVisualizer(
    sample_rate=44100,
    fft_size=2048,
    freq_range=(20.0, 20000.0)  # Display frequency range
)
```

### 4. CircularVisualizer
Creates a circular/radial frequency display.
```python
from algorythm.visualization import CircularVisualizer

visualizer = CircularVisualizer(
    sample_rate=44100,
    num_bars=64,
    inner_radius=0.2,
    bar_width=0.8,
    smoothing=0.5
)
```

### 5. OscilloscopeVisualizer
Shows oscilloscope-style display.
```python
from algorythm.visualization import OscilloscopeVisualizer

visualizer = OscilloscopeVisualizer(
    sample_rate=44100,
    window_size=1024,
    mode='waveform'  # Options: 'waveform', 'lissajous', 'phase'
)
```

### 6. ParticleVisualizer
Animated particles that react to audio.
```python
from algorythm.visualization import ParticleVisualizer

visualizer = ParticleVisualizer(
    sample_rate=44100,
    num_particles=100,
    decay=0.95,
    sensitivity=1.0
)
```

## Video Settings

### Resolution Options
```python
# 4K Ultra HD
video_width=3840, video_height=2160

# Full HD (1080p)
video_width=1920, video_height=1080

# HD (720p)
video_width=1280, video_height=720

# SD
video_width=640, video_height=480

# Square (for Instagram/social)
video_width=1080, video_height=1080
```

### Frame Rate Options
```python
video_fps=24  # Cinema standard
video_fps=30  # Standard video
video_fps=60  # Smooth/high frame rate
```

## Complete Example

```python
from algorythm.synth import Synth, Filter, ADSR
from algorythm.export import Exporter
from algorythm.visualization import SpectrogramVisualizer
import numpy as np

# Create a synth sound
synth = Synth(
    waveform='saw',
    filter=Filter.lowpass(cutoff=2000, resonance=0.6),
    envelope=ADSR(attack=0.1, decay=0.2, sustain=0.7, release=0.5)
)

# Generate notes
notes = [440, 494, 523, 587]  # A, B, C, D
signal_parts = [synth.generate_note(frequency=f, duration=0.5) for f in notes]
signal = np.concatenate(signal_parts)

# Create spectrogram visualizer
visualizer = SpectrogramVisualizer(
    sample_rate=44100,
    window_size=2048,
    hop_size=512
)

# Export as MP4
exporter = Exporter()
exporter.export(
    signal,
    'melody_with_spectrogram.mp4',
    sample_rate=44100,
    visualizer=visualizer,
    video_width=1920,
    video_height=1080,
    video_fps=30
)
```

## Tips and Best Practices

1. **Start Small**: Test with shorter clips and lower resolutions first
2. **Choose Appropriate Visualizers**: 
   - Waveform: Good for general use, shows amplitude
   - Spectrogram: Best for showing frequency changes over time
   - Frequency Scope: Good for bass-heavy or rhythmic content
   - Circular: Visually striking for electronic music
3. **Resolution vs. Render Time**: Higher resolutions take longer to render
4. **Frame Rate**: 30 FPS is a good balance between smoothness and file size
5. **File Location**: By default, files are saved to `~/Music` directory
6. **Progress Monitoring**: The export process shows progress indicators

## Troubleshooting

### "opencv-python not installed"
```bash
pip install opencv-python
```

### "ffmpeg not found"
Install ffmpeg on your system:
- Linux: `sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: Download from https://ffmpeg.org/

### Video exports as WAV instead
This happens when opencv-python or ffmpeg is not properly installed. Check the error messages in the console.

### Slow rendering
- Reduce video resolution
- Use lower FPS (24 or 30 instead of 60)
- Shorten the audio clip for testing

## File Formats Comparison

| Format | Use Case | Notes |
|--------|----------|-------|
| WAV | Pure audio, best quality | Large file size |
| MP3 | Compressed audio | Smaller, widely supported |
| FLAC | Lossless compression | Good balance |
| OGG | Open source compression | Good for web |
| **MP4** | **Audio with visualization** | **Video format, shareable** |

## See Also
- `examples/mp4_export_example.py` - Detailed examples
- `algorythm.visualization` - All visualizer documentation
- `algorythm.export` - Export module documentation
