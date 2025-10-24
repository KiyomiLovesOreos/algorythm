# Visualization Quick Reference

Quick reference for the improved visualization features.

## Debug Mode

Enable detailed logging on any visualizer:

```python
viz = WaveformVisualizer(debug=True)
# Output: [WaveformVisualizer] Generated waveform image: 1280x480, signal length: 1280
```

## Enhanced Waveform

```python
viz = WaveformVisualizer()
image = viz.to_image_data(
    signal,
    height=720,
    width=1280,
    line_thickness=3,    # 1-5, default: 2
    center_line=True     # default: True
)
```

## Colored Spectrogram

```python
viz = SpectrogramVisualizer()
spec = viz.generate(signal)
colored = viz.to_colored_image(
    spec,
    colormap='viridis',  # viridis, plasma, inferno, magma, hot, cool
    vmin=None,           # auto-detect
    vmax=None            # auto-detect
)
```

## Progress Tracking

```python
def progress(current, total):
    print(f'\rProgress: {100*current/total:.1f}%', end='')

renderer = VideoRenderer()
frames = renderer.render_frames(
    signal,
    visualizer,
    progress_callback=progress
)
```

## Backend Selection

```python
# Automatic (prefers OpenCV)
renderer = VideoRenderer()

# Force matplotlib
renderer = VideoRenderer(use_matplotlib=True)

# Check which backend is used
renderer = VideoRenderer(debug=True)
# Output: [VideoRenderer] Backend: opencv (or matplotlib)
```

## All Visualizer Classes Support Debug

```python
WaveformVisualizer(debug=True)
SpectrogramVisualizer(debug=True)
FrequencyScopeVisualizer(debug=True)
CircularVisualizer(debug=True)
OscilloscopeVisualizer(debug=True)
ParticleVisualizer(debug=True)
PianoRollVisualizer(debug=True)
```

## Run Tests

```bash
# Quick test (< 1 second)
python examples/quick_viz_test.py

# Comprehensive demo
python examples/comprehensive_viz_demo.py

# Debug test suite
python examples/debug_visualization.py
```

## Common Patterns

### High-Quality Waveform Export
```python
viz = WaveformVisualizer()
image = viz.to_image_data(
    signal,
    height=1080,
    width=1920,
    line_thickness=3,
    center_line=True
)
```

### Video with Progress Bar
```python
from algorythm.visualization import VideoRenderer, CircularVisualizer

def progress(curr, total):
    bar = '█' * int(40 * curr / total) + '░' * (40 - int(40 * curr / total))
    print(f'\r[{bar}] {100*curr/total:.0f}%', end='')

renderer = VideoRenderer(debug=False)
viz = CircularVisualizer(num_bars=64)
renderer.render_frames(signal, viz, 'output.mp4', progress)
```

### Debug Mode for Troubleshooting
```python
# Enable debug on everything
viz = SpectrogramVisualizer(debug=True)
renderer = VideoRenderer(debug=True)

# Detailed output helps identify issues
frames = renderer.render_frames(signal, viz)
```

## Tips

- Use `debug=True` when developing or troubleshooting
- Use `line_thickness=3` for better visibility in waveforms
- Progress callbacks are called every 10 frames
- Colored spectrograms work without matplotlib (grayscale fallback)
- Video rendering falls back to matplotlib if opencv unavailable
- All new features are backwards compatible
