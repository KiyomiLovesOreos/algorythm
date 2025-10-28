# Performance Optimization Guide for MP3 Visualizer

## Quick Wins for Faster Rendering

### 1. **Use Lower Resolution** (2-3x speedup)
```python
# Instead of Full HD (slow)
visualize_audio_file('song.mp3', 'output.mp4', 
                     visualizer=viz,
                     video_width=1920,
                     video_height=1080)

# Use 720p (MUCH faster!)
visualize_audio_file('song.mp3', 'output.mp4',
                     visualizer=viz, 
                     video_width=1280,
                     video_height=720)
```

### 2. **Reduce Frame Rate** (20% speedup)
```python
# 30 fps is smooth but slower
video_fps=30

# 24 fps (cinematic) is faster
video_fps=24
```

### 3. **Choose Faster Visualizers**

**Fastest** (recommended for long songs):
- `CircularVisualizer` - Very fast, looks great
- `WaveformVisualizer` - Fastest overall
- `FrequencyScopeVisualizer` - Fast

**Slower** (use for short clips):
- `SpectrogramVisualizer` - Now optimized but still slower
- `ParticleVisualizer` - Physics calculations slow

### 4. **Test with Short Clips First**
```python
# Render just first 10 seconds to test settings
visualize_audio_file('song.mp3', 'test.mp4',
                     visualizer=viz,
                     duration=10.0)  # Only 10 seconds!
```

## Recommended Settings by Song Length

### Short Songs (< 1 minute)
```python
viz = CircularVisualizer(sample_rate=44100, num_bars=128)
visualize_audio_file('short.mp3', 'output.mp4',
                     visualizer=viz,
                     video_width=1920,  # Full HD is fine
                     video_height=1080,
                     video_fps=30)
```

### Medium Songs (1-3 minutes)
```python
viz = CircularVisualizer(sample_rate=44100, num_bars=64)
visualize_audio_file('medium.mp3', 'output.mp4',
                     visualizer=viz,
                     video_width=1280,  # 720p recommended
                     video_height=720,
                     video_fps=24)      # 24fps recommended
```

### Long Songs (> 3 minutes)
```python
viz = WaveformVisualizer(sample_rate=44100)
visualize_audio_file('long.mp3', 'output.mp4',
                     visualizer=viz,
                     video_width=1280,  # 720p
                     video_height=720,
                     video_fps=24)      # 24fps
```

## System Optimizations

### If Your Computer Freezes

1. **Close other applications** - Free up RAM
2. **Use lower settings**:
   ```python
   video_width=854   # 480p - very fast!
   video_height=480
   video_fps=24
   ```
3. **Process in chunks**:
   ```python
   # First 30 seconds
   visualize_audio_file('song.mp3', 'part1.mp4', duration=30)
   
   # Next 30 seconds  
   visualize_audio_file('song.mp3', 'part2.mp4', offset=30, duration=30)
   ```

### Expected Rendering Times

On a typical laptop with the **optimized** code:

| Song Length | Settings | Approximate Time |
|-------------|----------|------------------|
| 1 minute    | 1280x720, 24fps, Circular | 30-60 seconds |
| 2 minutes   | 1280x720, 24fps, Circular | 1-2 minutes |
| 3 minutes   | 1280x720, 24fps, Circular | 1.5-3 minutes |
| 5 minutes   | 1280x720, 24fps, Circular | 2.5-5 minutes |

*Times with 1920x1080 will be 2-3x longer*

## What Was Optimized

The recent optimizations include:

1. **Vectorized FFT computation** - SpectrogramVisualizer now computes all FFTs at once using numpy stride tricks (10-20x faster)
2. **Batch spectrogram rendering** - Process entire audio file once instead of per-frame
3. **Optimized video encoding** - Better memory management during frame writing
4. **Progress feedback** - See rendering progress so you know it's working

## Example: Full Optimization

```python
from algorythm.audio_loader import visualize_audio_file
from algorythm.visualization import CircularVisualizer

# Create fast visualizer
viz = CircularVisualizer(
    sample_rate=44100,
    num_bars=64,        # Fewer bars = faster
    smoothing=0.7       # Smooth animation
)

# Fast rendering settings
visualize_audio_file(
    input_file='my_song.mp3',
    output_file='my_video.mp4',
    visualizer=viz,
    video_width=1280,   # 720p (fast)
    video_height=720,
    video_fps=24,       # 24fps (fast)
    duration=None       # Full song
)
```

## Troubleshooting

### "My computer still freezes"
- Your RAM might be insufficient. Try 480p resolution (854x480)
- Process in 30-second chunks and combine later
- Use WaveformVisualizer (simplest, fastest)

### "It's taking forever"
- Check that you're using 1280x720, not 1920x1080
- Verify fps is 24, not 30 or 60
- Make sure you're using CircularVisualizer or WaveformVisualizer

### "The video looks choppy"
- 24fps is cinematic standard and should look smooth
- If it's truly choppy, increase to 30fps (will take longer)

## Future Optimizations Possible

If you need even more speed:
- GPU acceleration (requires PyTorch/CUDA)
- Multi-threading (complex for video encoding)
- Further downsampling of audio before visualization
- Hardware video encoding (system-dependent)
