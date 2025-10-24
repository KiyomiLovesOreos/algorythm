# 🎬 Video Export Guide

## Overview

Algorythm now supports exporting your music as **MP4 videos with synchronized visualizations**! Simply add `video=True` to your render call and choose from multiple customizable visualizers.

---

## 🚀 Quick Start

### Installation Requirements

```bash
# Install algorythm with video support
pip install opencv-python

# Ensure ffmpeg is installed (system-wide)
# Ubuntu/Debian: sudo apt-get install ffmpeg
# macOS: brew install ffmpeg
# Windows: Download from ffmpeg.org
```

### Simplest Example

```python
from algorythm.synth import SynthPresets
from algorythm.sequence import Motif, Scale
from algorythm.structure import Composition

# Create your song
song = Composition(tempo=120)
song.add_track('Lead', SynthPresets.pluck())
song.repeat_motif(Motif.from_intervals([0, 2, 4, 7], scale=Scale.major('C')), bars=4)

# Export as video - that's it!
song.render('my_song.mp4', video=True)
```

---

## 🎨 Available Visualizers

### 1. **Spectrum Visualizer** (Default)
Displays frequency content as vertical bars.

```python
song.render('output.mp4', video=True, video_config={
    'visualizer': 'spectrum',
    'fft_size': 2048,                    # FFT size (higher = more detail)
    'freq_range': (20.0, 20000.0)        # Frequency range to display
})
```

**Best for:** Electronic music, bass-heavy tracks, general purpose

---

### 2. **Waveform Visualizer**
Shows the audio waveform over time.

```python
song.render('output.mp4', video=True, video_config={
    'visualizer': 'waveform',
    'window_size': 1024                  # Number of samples to display
})
```

**Best for:** Acoustic music, vocals, showing dynamics

---

### 3. **Circular Visualizer** ⭕
Radial frequency display in a circle pattern.

```python
song.render('output.mp4', video=True, video_config={
    'visualizer': 'circular',
    'num_bars': 64,                      # Number of bars around the circle
    'inner_radius': 0.2,                 # Inner circle size (0.0 to 1.0)
    'bar_width': 0.8,                    # Bar width/length (0.0 to 1.0)
    'smoothing': 0.5                     # Animation smoothing (0.0 to 1.0)
})
```

**Best for:** Dance music, EDM, visually striking content

---

### 4. **Particle Visualizer** ✨
Animated particles that react to the music.

```python
song.render('output.mp4', video=True, video_config={
    'visualizer': 'particle',
    'num_particles': 100,                # Number of particles
    'decay': 0.95,                       # Particle velocity decay (0.0 to 1.0)
    'sensitivity': 1.0                   # Sensitivity to audio changes
})
```

**Best for:** Ambient music, experimental, artistic content

---

### 5. **Spectrogram Visualizer** 📊
Shows frequency content over time (like a heatmap).

```python
song.render('output.mp4', video=True, video_config={
    'visualizer': 'spectrogram',
    'window_size': 2048,                 # FFT window size
    'hop_size': 512                      # Hop size between windows
})
```

**Best for:** Analysis, educational content, complex music

---

## 🎨 Color Customization

All visualizers support custom colors:

```python
video_config = {
    'visualizer': 'spectrum',
    'background_color': (0, 0, 0),        # RGB (0-255) - Black
    'foreground_color': (255, 100, 255)   # RGB (0-255) - Pink
}
```

### Color Presets

```python
# Neon Cyan on Black
'background_color': (0, 0, 0), 'foreground_color': (0, 255, 255)

# Purple on Dark Blue
'background_color': (10, 10, 50), 'foreground_color': (200, 100, 255)

# Gold on Black
'background_color': (0, 0, 0), 'foreground_color': (255, 215, 0)

# Green Matrix Style
'background_color': (0, 0, 0), 'foreground_color': (0, 255, 100)

# Red on Dark
'background_color': (20, 0, 0), 'foreground_color': (255, 50, 50)

# Retro Orange
'background_color': (5, 5, 15), 'foreground_color': (255, 150, 0)
```

---

## 📐 Resolution & Quality Settings

### Common Resolutions

```python
# Full HD (1080p) - Recommended
video_config = {'width': 1920, 'height': 1080, 'fps': 30}

# HD (720p) - Smaller file size
video_config = {'width': 1280, 'height': 720, 'fps': 30}

# 4K (Ultra HD) - High quality, large file
video_config = {'width': 3840, 'height': 2160, 'fps': 30}

# Square (Instagram/Social)
video_config = {'width': 1080, 'height': 1080, 'fps': 30}

# Vertical (TikTok/Stories)
video_config = {'width': 1080, 'height': 1920, 'fps': 30}

# Cinematic
video_config = {'width': 2560, 'height': 1080, 'fps': 60}
```

### Frame Rate Options

- **24 fps** - Cinematic, film-like
- **30 fps** - Standard, recommended
- **60 fps** - Smooth, high motion (larger file)

---

## 🎯 Complete Examples

### Example 1: YouTube Music Video

```python
from algorythm.synth import SynthPresets
from algorythm.sequence import Motif, Scale
from algorythm.structure import Composition, Reverb

# Create composition
song = Composition(tempo=128)
song.add_track('Lead', SynthPresets.lead())
song.repeat_motif(Motif.from_intervals([0, 2, 4, 7, 9, 7, 4, 2], scale=Scale.major('D')), bars=8)
song.add_fx(Reverb(mix=0.4))

# Export as 1080p video with circular visualizer
song.render('youtube_video.mp4', video=True, video_config={
    'visualizer': 'circular',
    'width': 1920,
    'height': 1080,
    'fps': 30,
    'background_color': (0, 0, 0),
    'foreground_color': (0, 200, 255),
    'num_bars': 80,
    'smoothing': 0.7
})
```

---

### Example 2: Instagram Post

```python
# Square format for Instagram
song.render('instagram_post.mp4', video=True, video_config={
    'visualizer': 'spectrum',
    'width': 1080,
    'height': 1080,
    'fps': 30,
    'background_color': (30, 0, 50),
    'foreground_color': (255, 100, 200)
})
```

---

### Example 3: Artistic Particle Effect

```python
# Slow, atmospheric with particles
ambient = Composition(tempo=80)
ambient.add_track('Pad', SynthPresets.warm_pad())
ambient.repeat_motif(Motif.from_intervals([0, 3, 5, 7], scale=Scale.minor('A')), bars=16)

ambient.render('ambient_art.mp4', video=True, video_config={
    'visualizer': 'particle',
    'width': 1920,
    'height': 1080,
    'fps': 30,
    'background_color': (5, 5, 15),
    'foreground_color': (150, 100, 255),
    'num_particles': 200,
    'decay': 0.98,
    'sensitivity': 1.5
})
```

---

### Example 4: Multi-Format Export

```python
# Export both audio and video
song.render('track', formats=['wav', 'mp3'])           # Audio files
song.render('track_video.mp4', video=True)             # Video file
```

---

## 🔧 Advanced Configuration

### All Available Options

```python
complete_config = {
    # Basic Settings
    'visualizer': 'spectrum',           # Type of visualization
    'width': 1920,                      # Video width in pixels
    'height': 1080,                     # Video height in pixels
    'fps': 30,                          # Frames per second
    
    # Colors
    'background_color': (0, 0, 0),      # RGB background
    'foreground_color': (255, 255, 255), # RGB foreground
    
    # Spectrum Visualizer Options
    'fft_size': 2048,                   # FFT size
    'freq_range': (20.0, 20000.0),      # Frequency range
    
    # Waveform Visualizer Options
    'window_size': 1024,                # Waveform window size
    
    # Circular Visualizer Options
    'num_bars': 64,                     # Number of bars
    'inner_radius': 0.2,                # Inner radius (0.0-1.0)
    'bar_width': 0.8,                   # Bar width (0.0-1.0)
    'smoothing': 0.5,                   # Smoothing (0.0-1.0)
    
    # Particle Visualizer Options
    'num_particles': 100,               # Particle count
    'decay': 0.95,                      # Velocity decay (0.0-1.0)
    'sensitivity': 1.0,                 # Audio sensitivity
    
    # Spectrogram Visualizer Options
    'hop_size': 512,                    # Hop size for STFT
}
```

---

## 💡 Tips & Best Practices

### 1. **Choose the Right Visualizer**
- **Spectrum/Circular**: Great for music with strong bass or rhythm
- **Waveform**: Best for acoustic or vocal-focused tracks
- **Particle**: Perfect for ambient or experimental music
- **Spectrogram**: Ideal for complex arrangements or educational content

### 2. **Performance Considerations**
- Higher resolutions (4K) take longer to render
- More particles = slower rendering
- 60 fps takes ~2x longer than 30 fps
- Start with lower settings, then increase for final export

### 3. **Color Selection**
- High contrast makes visuals pop (e.g., bright on dark)
- Match colors to your music genre or mood
- Use complementary colors for visual interest

### 4. **Frame Rate**
- 30 fps is perfect for most use cases
- Use 60 fps only for fast, energetic music
- 24 fps for a cinematic feel

### 5. **File Size**
- Longer videos = larger files
- Higher resolutions = larger files
- Use 720p for web preview, 1080p for final export

---

## 🐛 Troubleshooting

### "opencv-python not found"
```bash
pip install opencv-python
```

### "ffmpeg not found"
Install ffmpeg on your system:
- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from ffmpeg.org

### Video renders but no audio
Make sure ffmpeg is properly installed and accessible from command line:
```bash
ffmpeg -version
```

### Rendering is slow
- Reduce resolution (e.g., 720p instead of 1080p)
- Reduce fps (e.g., 24 instead of 30)
- Reduce particle count or visual complexity
- Use spectrum visualizer (fastest) instead of particles

### Colors look wrong
- Make sure RGB values are 0-255, not 0.0-1.0
- Check that background and foreground colors have good contrast

---

## 📚 Quick Reference

### Minimal Video Export
```python
song.render('output.mp4', video=True)
```

### Custom Visualizer
```python
song.render('output.mp4', video=True, video_config={
    'visualizer': 'circular',
    'background_color': (0, 0, 0),
    'foreground_color': (255, 100, 200)
})
```

### High Quality Export
```python
song.render('output.mp4', video=True, video_config={
    'width': 3840,
    'height': 2160,
    'fps': 60,
    'visualizer': 'spectrum'
})
```

---

## 🎉 You're Ready!

Start creating beautiful music videos with your algorythm compositions! Experiment with different visualizers and colors to find your unique style.

**Happy creating!** 🎵✨🎬
