# 🎬 Video Export Quick Reference

## One-Line Video Export
```python
song.render('output.mp4', video=True)
```

## Visualizer Types
```python
'spectrum'      # Frequency bars
'waveform'      # Audio waveform
'circular'      # Radial display
'particle'      # Animated particles
'spectrogram'   # Frequency heatmap
```

## Quick Templates

### YouTube (1080p)
```python
song.render('youtube.mp4', video=True, video_config={
    'visualizer': 'circular',
    'width': 1920, 'height': 1080, 'fps': 30
})
```

### Instagram Square
```python
song.render('instagram.mp4', video=True, video_config={
    'width': 1080, 'height': 1080
})
```

### TikTok Vertical
```python
song.render('tiktok.mp4', video=True, video_config={
    'width': 1080, 'height': 1920
})
```

## Color Presets
```python
# Cyan
'background_color': (0, 0, 0), 'foreground_color': (0, 255, 255)

# Pink
'background_color': (20, 0, 20), 'foreground_color': (255, 0, 200)

# Gold
'background_color': (0, 0, 0), 'foreground_color': (255, 215, 0)
```

## Common Options
```python
video_config = {
    'visualizer': 'spectrum',           # Required
    'width': 1920,                      # Default: 1920
    'height': 1080,                     # Default: 1080
    'fps': 30,                          # Default: 30
    'background_color': (0, 0, 0),      # RGB 0-255
    'foreground_color': (255, 255, 255) # RGB 0-255
}
```

## Visualizer-Specific Options

### Spectrum
```python
'fft_size': 2048              # Detail level
'freq_range': (20, 20000)     # Frequency range
```

### Circular
```python
'num_bars': 64                # Number of bars
'inner_radius': 0.2           # Inner size (0-1)
'smoothing': 0.5              # Smoothing (0-1)
```

### Particle
```python
'num_particles': 100          # Particle count
'decay': 0.95                 # Decay rate (0-1)
'sensitivity': 1.0            # Responsiveness
```

## Installation
```bash
pip install opencv-python
# Also need ffmpeg installed
```

## See Full Docs
- VIDEO_BEGINNER_GUIDE.md
- VIDEO_EXPORT_GUIDE.md
