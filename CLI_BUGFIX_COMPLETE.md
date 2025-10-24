# CLI Bug Fix: background_color Parameter Error

## Issue

When running `algorythm visualize` with the `--color` or `--background` flags, the following error occurred:

```
❌ Error creating video: Exporter.export() got an unexpected keyword argument 'background_color'
TypeError: Exporter.export() got an unexpected keyword argument 'background_color'
```

## Root Cause

The CLI was trying to pass `background_color` and `foreground_color` parameters to `Exporter.export()`, but that method doesn't accept those parameters. The color parameters should be passed to `VideoRenderer` instead.

## Fix Applied

Modified `algorythm/cli.py` in the `visualize_audio()` function:

### Before (broken):
```python
exporter = Exporter()
exporter.export(
    audio.signal,
    output_path,
    sample_rate=audio.sample_rate,
    visualizer=viz,
    video_width=args.width,
    video_height=args.height,
    video_fps=args.fps,
    background_color=bg_color,  # ❌ Exporter doesn't accept this
    foreground_color=fg_color    # ❌ Exporter doesn't accept this
)
```

### After (fixed):
```python
# Create VideoRenderer directly with custom colors
renderer = VideoRenderer(
    width=args.width,
    height=args.height,
    fps=args.fps,
    sample_rate=audio.sample_rate,
    background_color=bg_color,   # ✓ VideoRenderer accepts this
    foreground_color=fg_color,   # ✓ VideoRenderer accepts this
    debug=args.debug
)

# Render video with audio
renderer.render_frames(
    audio.signal,
    viz,
    output_path=output_path
)
```

## Status

✅ **FIXED** - The CLI now correctly passes color parameters to VideoRenderer

## Usage

The color and background options now work correctly:

```bash
# Basic usage
algorythm visualize song.mp3

# With custom colors
algorythm visualize song.mp3 --color purple --background dark

# With all options
algorythm visualize song.mp3 \
  --color purple \
  --background dark \
  --bars 128 \
  -w 1920 --height 1080 \
  --fps 30
```

## Available Colors

**Foreground (--color):**
- `cyan` (default)
- `purple`
- `blue`
- `red`
- `green`
- `orange`
- `magenta`

**Background (--background):**
- `black` (default)
- `dark`
- `white`
- `light`

## Important Note

To actually create video files, you need to install a video backend:

```bash
# Option 1: OpenCV (faster, recommended)
pip install opencv-python

# Option 2: Matplotlib (slower, but no C dependencies)
pip install matplotlib
```

Both require ffmpeg to be installed on your system:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/
```

## Verify Installation

Check what's installed:

```bash
algorythm formats
```

This will show:
- Which dependencies are installed
- What formats are supported
- What's missing (if anything)

## Testing

The fix has been tested and verified:

```bash
# Test with MP3 file
algorythm visualize song.mp3 --color purple --background dark --debug

# Test with WAV file  
algorythm visualize song.wav --color cyan --background black
```

## Files Modified

- `algorythm/cli.py` - Fixed color parameter passing in `visualize_audio()` function

## Related Documentation

- `CLI_MP3_GUIDE.md` - Complete guide for MP3 visualization
- `CLI_QUICK_REF.md` - Quick reference for all CLI commands
- `BUGFIX_MP3_CLI_SUMMARY.md` - Summary of all MP3-related fixes
