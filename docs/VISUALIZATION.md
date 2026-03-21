# Visualization Guide

Create video visualizations of your audio.

## Basic Visualization

Render audio to video with visual effects.

### Quick Start

```python
from algorythm import visualize_audio_file, WaveformVisualizer

# Create visualizer
viz = WaveformVisualizer(
    sample_rate=44100,
    color=(0, 255, 100)  # RGB color
)

# Visualize audio file
visualize_audio_file(
    audio_path='my_song.wav',
    output_path='my_song.mp4',
    visualizer=viz,
    video_width=1920,
    video_height=1080,
    video_fps=30
)
```

## Visualizer Types

Different visualizers show audio in different ways.

### Waveform Visualizer

Shows the audio waveform over time:

```python
from algorythm import WaveformVisualizer

viz = WaveformVisualizer(
    sample_rate=44100,
    color=(0, 255, 100),      # Line color (R, G, B)
    line_width=2,             # Thickness
    background=(0, 0, 0)      # Background color
)
```

### Frequency Scope

Shows real-time frequency content:

```python
from algorythm import FrequencyScopeVisualizer

viz = FrequencyScopeVisualizer(
    sample_rate=44100,
    num_bars=64,              # Number of frequency bars
    color=(100, 200, 255),
    background=(0, 0, 0)
)
```

More bars = more detail but slower rendering.

### Spectrogram

Shows frequency content over time:

```python
from algorythm import SpectrogramVisualizer

viz = SpectrogramVisualizer(
    sample_rate=44100,
    fft_size=2048,            # FFT window size
    hop_length=512,           # Hop between windows
    colormap='viridis'        # Color scheme
)
```

Colormaps: `'viridis'`, `'plasma'`, `'inferno'`, `'magma'`, `'coolwarm'`

### Oscilloscope

Classic oscilloscope display:

```python
from algorythm import OscilloscopeVisualizer

viz = OscilloscopeVisualizer(
    sample_rate=44100,
    color=(0, 255, 0),
    line_width=3,
    background=(0, 0, 0)
)
```

### Piano Roll

Shows notes as a piano roll:

```python
from algorythm import PianoRollVisualizer

viz = PianoRollVisualizer(
    sample_rate=44100,
    note_color=(100, 150, 255),
    background=(20, 20, 20)
)
```

Note: Requires pitch detection from audio.

### Circular Visualizer

Circular frequency display:

```python
from algorythm import CircularVisualizer

viz = CircularVisualizer(
    sample_rate=44100,
    num_bars=64,
    radius=200,               # Circle radius
    color=(255, 100, 150),
    background=(0, 0, 0)
)
```

## Video Rendering

### Basic Rendering

```python
from algorythm import visualize_audio_file, FrequencyScopeVisualizer

viz = FrequencyScopeVisualizer(sample_rate=44100, num_bars=64)

visualize_audio_file(
    'input.wav',
    'output.mp4',
    viz
)
```

### Custom Video Settings

```python
visualize_audio_file(
    audio_path='input.wav',
    output_path='output.mp4',
    visualizer=viz,
    video_width=1920,         # HD resolution
    video_height=1080,
    video_fps=30,             # Frame rate
    video_bitrate='8000k'     # Video quality
)
```

Common resolutions:
- 1920x1080 (1080p HD)
- 1280x720 (720p)
- 3840x2160 (4K)

### Performance Options

```python
# Faster rendering (lower quality)
visualize_audio_file(
    'input.wav',
    'output.mp4',
    viz,
    video_width=1280,
    video_height=720,
    video_fps=24,             # Lower FPS
    video_bitrate='4000k'     # Lower bitrate
)

# Best quality (slower)
visualize_audio_file(
    'input.wav',
    'output.mp4',
    viz,
    video_width=3840,
    video_height=2160,
    video_fps=60,
    video_bitrate='20000k'
)
```

## Advanced Usage

### Custom Visualizer

Create your own visualizer:

```python
from algorythm import VideoRenderer
import numpy as np
from PIL import Image, ImageDraw

class CustomVisualizer:
    def __init__(self, sample_rate, color=(255, 255, 255)):
        self.sample_rate = sample_rate
        self.color = color
    
    def render_frame(self, audio_chunk, width, height):
        """
        Render a single frame.
        
        Args:
            audio_chunk: Audio samples for this frame
            width: Frame width
            height: Frame height
        
        Returns:
            PIL Image
        """
        # Create blank frame
        img = Image.new('RGB', (width, height), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw something based on audio
        amplitude = np.abs(audio_chunk).mean()
        radius = int(amplitude * height / 2)
        
        center_x = width // 2
        center_y = height // 2
        
        draw.ellipse(
            [center_x - radius, center_y - radius,
             center_x + radius, center_y + radius],
            fill=self.color
        )
        
        return img

# Use it
viz = CustomVisualizer(sample_rate=44100)
visualize_audio_file('input.wav', 'output.mp4', viz)
```

### Multiple Visualizers

Combine different visualizers:

```python
from algorythm import VideoRenderer, WaveformVisualizer, FrequencyScopeVisualizer
from PIL import Image

class ComboVisualizer:
    def __init__(self, sample_rate):
        self.waveform = WaveformVisualizer(sample_rate)
        self.frequency = FrequencyScopeVisualizer(sample_rate)
    
    def render_frame(self, audio_chunk, width, height):
        # Render both visualizers
        half_height = height // 2
        
        wave_frame = self.waveform.render_frame(audio_chunk, width, half_height)
        freq_frame = self.frequency.render_frame(audio_chunk, width, half_height)
        
        # Combine vertically
        combined = Image.new('RGB', (width, height))
        combined.paste(wave_frame, (0, 0))
        combined.paste(freq_frame, (0, half_height))
        
        return combined

viz = ComboVisualizer(sample_rate=44100)
visualize_audio_file('input.wav', 'output.mp4', viz)
```

## Loading and Processing Audio

### Load Audio Files

```python
from algorythm import load_audio, AudioFile

# Load audio
audio_file = load_audio('my_song.wav')

# Access properties
print(f"Duration: {audio_file.duration} seconds")
print(f"Sample rate: {audio_file.sample_rate} Hz")
print(f"Channels: {audio_file.channels}")

# Get audio data
audio_data = audio_file.get_audio_data()
```

### Process Audio Before Visualization

```python
from algorythm import load_audio, AudioFile, visualize_audio_file
from algorythm import FrequencyScopeVisualizer

# Load audio
audio = load_audio('input.wav')

# Process audio (e.g., normalize)
audio_data = audio.get_audio_data()
normalized = audio_data / np.max(np.abs(audio_data))

# Save processed audio
processed_audio = AudioFile(
    audio_data=normalized,
    sample_rate=audio.sample_rate
)
processed_audio.save('processed.wav')

# Visualize processed audio
viz = FrequencyScopeVisualizer(sample_rate=audio.sample_rate)
visualize_audio_file('processed.wav', 'output.mp4', viz)
```

## Direct Rendering from Composition

```python
from algorythm import Composition, SynthPresets, Scale, Motif
from algorythm import VideoRenderer, WaveformVisualizer

# Create composition
song = Composition(tempo=120)
song.add_track('melody', SynthPresets.pluck())

scale = Scale.major('C')
melody = Motif.from_intervals([0, 2, 4, 5, 7], scale=scale)
song.play_motif(melody, start=0.0, track='melody')

# Render to audio first
song.render('temp.wav')

# Then visualize
viz = WaveformVisualizer(sample_rate=44100)
visualize_audio_file('temp.wav', 'output.mp4', viz)
```

## Color Schemes

### Pre-defined Colors

```python
# Neon green
color = (0, 255, 100)

# Electric blue
color = (0, 150, 255)

# Hot pink
color = (255, 20, 147)

# Orange
color = (255, 165, 0)

# Purple
color = (128, 0, 255)
```

### Dynamic Colors

Change color based on audio:

```python
class DynamicColorVisualizer:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.base_viz = FrequencyScopeVisualizer(sample_rate)
    
    def render_frame(self, audio_chunk, width, height):
        # Change color based on amplitude
        amplitude = np.abs(audio_chunk).mean()
        
        # Interpolate color
        r = int(255 * amplitude)
        g = int(255 * (1 - amplitude))
        b = 128
        
        self.base_viz.color = (r, g, b)
        return self.base_viz.render_frame(audio_chunk, width, height)
```

## Optimization Tips

1. **Lower resolution for testing**: Use 720p while developing, 1080p for final
2. **Reduce FPS**: 24fps is fine for most visualizations
3. **Fewer bars**: 32-64 bars is usually enough
4. **Use streaming renderer**: The built-in renderer is optimized
5. **Close other apps**: Video rendering is CPU/memory intensive

### Fast Preview

```python
# Quick preview settings
visualize_audio_file(
    'input.wav',
    'preview.mp4',
    viz,
    video_width=854,      # 480p
    video_height=480,
    video_fps=15,         # Low FPS
    video_bitrate='2000k'
)
```

### Production Quality

```python
# High quality final render
visualize_audio_file(
    'input.wav',
    'final.mp4',
    viz,
    video_width=1920,     # 1080p
    video_height=1080,
    video_fps=30,
    video_bitrate='8000k'
)
```

## Common Issues

### Out of Memory

Reduce video dimensions or use streaming:
```python
# Use smaller dimensions
video_width=1280
video_height=720
```

### Slow Rendering

- Lower FPS (24 instead of 60)
- Reduce resolution
- Use fewer frequency bars
- Close other applications

### Audio/Video Sync Issues

Ensure consistent sample rate:
```python
viz = WaveformVisualizer(sample_rate=44100)
# Make sure audio file is also 44100 Hz
```

## Complete Example

```python
from algorythm import (
    Composition, SynthPresets, Scale, Motif,
    visualize_audio_file, FrequencyScopeVisualizer
)

# Create music
song = Composition(tempo=120)
melody = song.add_track('melody', SynthPresets.pluck())
bass = song.add_track('bass', SynthPresets.synth_bass())

scale = Scale.minor('A')
melody_pattern = Motif.from_intervals([0, 2, 3, 5, 7], scale=scale, duration=0.5)
bass_pattern = Motif.from_intervals([0, 0, 0, 0], scale=scale, octave=2, duration=1.0)

song.play_motif(melody_pattern, start=0.0, track='melody')
song.play_motif(bass_pattern, start=0.0, track='bass')

# Render audio
song.render('music.wav')

# Create visualization
viz = FrequencyScopeVisualizer(
    sample_rate=44100,
    num_bars=64,
    color=(0, 255, 150)
)

# Render video
visualize_audio_file(
    'music.wav',
    'music_video.mp4',
    viz,
    video_width=1920,
    video_height=1080,
    video_fps=30
)

print("Video created: music_video.mp4")
```

## Requirements

For video visualization, install:
```bash
pip install moviepy pillow
```
