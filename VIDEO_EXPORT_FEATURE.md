# 🎬 Video Export Feature - Summary

## What's New?

Algorythm now has **full video export capabilities**! Create stunning music videos with synchronized visualizations in just one line of code.

---

## Quick Start

```python
# Just add video=True to your render call!
song.render('output.mp4', video=True)
```

---

## Installation

```bash
# Install video support
pip install opencv-python

# Install ffmpeg (system-wide)
# Ubuntu/Debian: sudo apt-get install ffmpeg
# macOS: brew install ffmpeg
# Windows: Download from ffmpeg.org
```

Or install with setup.py:
```bash
pip install -e ".[video]"     # Just video support
pip install -e ".[all]"        # Everything including video
```

---

## 5 Visualizer Types

1. **Spectrum** - Frequency bars (like an equalizer)
2. **Waveform** - Audio waveform display
3. **Circular** - Radial frequency display
4. **Particle** - Animated particle system
5. **Spectrogram** - Frequency heatmap over time

---

## Key Features

✅ **Easy to Use** - Just add `video=True`  
✅ **Highly Customizable** - Colors, resolutions, visualizer options  
✅ **Multiple Resolutions** - 720p, 1080p, 4K, custom sizes  
✅ **Social Media Ready** - Square, vertical, widescreen formats  
✅ **MP4 Output** - Universal format with audio embedded  
✅ **5 Unique Visualizers** - Each with extensive options  
✅ **Color Customization** - RGB background and foreground colors  
✅ **Performance Options** - Adjustable FPS and quality settings  

---

## Example Usage

### Basic
```python
song.render('video.mp4', video=True)
```

### With Customization
```python
song.render('video.mp4', video=True, video_config={
    'visualizer': 'circular',
    'width': 1920,
    'height': 1080,
    'fps': 30,
    'background_color': (0, 0, 0),
    'foreground_color': (255, 100, 200),
    'num_bars': 64,
    'smoothing': 0.7
})
```

---

## Documentation Files

- **VIDEO_BEGINNER_GUIDE.md** - Start here if you're new!
- **VIDEO_EXPORT_GUIDE.md** - Complete reference with all options
- **examples/simple_video_export.py** - Simple working example
- **examples/video_export_example.py** - Advanced examples

---

## Visualizer Options Summary

### Spectrum Visualizer
- `fft_size` - FFT size (default: 2048)
- `freq_range` - Frequency range tuple (default: (20, 20000))

### Waveform Visualizer
- `window_size` - Display window size (default: 1024)

### Circular Visualizer
- `num_bars` - Number of bars (default: 64)
- `inner_radius` - Inner circle size (default: 0.2)
- `bar_width` - Bar width (default: 0.8)
- `smoothing` - Animation smoothing (default: 0.5)

### Particle Visualizer
- `num_particles` - Particle count (default: 100)
- `decay` - Velocity decay (default: 0.95)
- `sensitivity` - Audio sensitivity (default: 1.0)

### Spectrogram Visualizer
- `window_size` - FFT window (default: 2048)
- `hop_size` - Hop size (default: 512)

---

## Common Use Cases

### YouTube Music Video
```python
song.render('youtube.mp4', video=True, video_config={
    'visualizer': 'circular',
    'width': 1920,
    'height': 1080,
    'fps': 30
})
```

### Instagram Square Post
```python
song.render('instagram.mp4', video=True, video_config={
    'width': 1080,
    'height': 1080
})
```

### TikTok Vertical Video
```python
song.render('tiktok.mp4', video=True, video_config={
    'width': 1080,
    'height': 1920
})
```

---

## New Classes Added

### `algorythm.visualization`
- `CircularVisualizer` - Radial frequency display
- `ParticleVisualizer` - Particle-based visualization
- Enhanced `VideoRenderer` with full MP4 export

### `algorythm.structure`
- `Composition.render()` - Enhanced with `video` and `video_config` parameters

---

## Performance Notes

- Video rendering is CPU-intensive
- Higher resolutions take longer
- Particle visualizer is the most demanding
- Spectrum visualizer is the fastest
- Typical render times:
  - 720p @ 30fps: ~1-2x real-time
  - 1080p @ 30fps: ~2-4x real-time
  - 4K @ 60fps: ~8-15x real-time

---

## Requirements

**Python Packages:**
- `opencv-python>=4.5.0` (for video encoding)
- `numpy>=1.19.0` (already required)

**System Requirements:**
- `ffmpeg` (for audio muxing)

---

## Backwards Compatibility

✅ **100% backwards compatible**  
- All existing code works without changes
- Video export is opt-in with `video=True`
- No impact on audio-only exports

---

## Next Steps

1. **Read** `VIDEO_BEGINNER_GUIDE.md` for a gentle introduction
2. **Try** `examples/simple_video_export.py` for your first video
3. **Explore** `examples/video_export_example.py` for advanced usage
4. **Reference** `VIDEO_EXPORT_GUIDE.md` for complete documentation

---

## Future Enhancements

Potential future additions:
- Real-time preview window
- Custom shader effects
- Multi-layer compositing
- Text/title overlays
- Transition effects between sections
- Beat-synchronized animations
- Custom color gradients
- 3D visualizations

---

## Contributing

Have ideas for new visualizers or effects? Contributions welcome!

---

**Enjoy creating beautiful music videos with algorythm!** 🎵✨🎬
