# Troubleshooting

Common problems and solutions.

## No Sound or Silent Output

### Problem: Audio file is empty or silent

**Causes:**
- Amplitude too low (signals near 0)
- No tracks added
- Notes not playing
- Synth not generating audio

**Solutions:**

1. Check amplitude values
```python
# Make sure amplitude is reasonable
synth = Synth(waveform='sine', amplitude=1.0)  # Should be 1.0, not 0.0

# Check note generation
audio = synth.generate_note(440, 1.0)
print(f"Audio min: {audio.min()}, max: {audio.max()}")  # Should not be all zeros
```

2. Verify tracks have audio
```python
song = Composition(tempo=120)
track = song.add_track('test', SynthPresets.pluck())

# Actually add notes
song.play_note(440, 1.0, 0.0, 'test')  # Required!

song.render('output.wav')
```

3. Check duration isn't too short
```python
# Problem: Duration is 0.001 seconds (1 ms)
audio = synth.generate_note(440, 0.001)

# Solution: Make it longer
audio = synth.generate_note(440, 1.0)  # 1 second
```

4. Verify file size
```bash
# Check if file was actually created and has data
ls -lh output.wav

# File should be several MB (not KB)
```

## Audio Is Distorted or Crackling

### Problem: Sound is clipped, harsh, or has noise artifacts

**Causes:**
- Volume levels too high (clipping)
- Multiple tracks overwhelming
- No limiter on master
- Audio values exceed -1.0 to 1.0 range

**Solutions:**

1. Add master limiter
```python
from algorythm import Limiter

song.add_master_effect(Limiter(threshold=-3))
```

2. Lower individual track volumes
```python
# Instead of:
track.set_volume(1.0)

# Use:
track.set_volume(0.8)  # Leave headroom
```

3. Reduce effect mix amounts
```python
# Problem: Effects too wet
reverb = ReverbFX(mix=1.0)  # 100% reverb

# Solution: Reduce mix
reverb = ReverbFX(mix=0.3)  # 30% reverb, 70% dry
```

4. Normalize properly
```python
import numpy as np

# Prevent clipping
max_val = np.abs(audio).max()
if max_val > 1.0:
    audio = audio / (max_val * 1.1)  # Slight headroom
```

## Timing Issues

### Problem: Notes play at wrong times or are out of sync

**Causes:**
- Tempo not matching beat calculations
- Bar-to-second conversion incorrect
- Time calculations off

**Solutions:**

1. Always use helper function
```python
TEMPO = 120

def bars(n):
    return (n * 4.0 * 60.0) / TEMPO

# Then use:
start_time = bars(4)  # 4 bars = 8.0 seconds at 120 BPM
```

2. Verify tempo math
```python
# 120 BPM = 2 beats per second
# 1 quarter note = 0.5 seconds
# 1 bar (4 beats) = 2.0 seconds

bar_duration = (4.0 * 60.0) / 120  # Should be 2.0
print(f"Bar duration: {bar_duration} seconds")
```

3. Use consistent sample rate
```python
# Make sure everything uses same sample rate
song = Composition(tempo=120, sample_rate=44100)

# Export at same rate
Exporter(sample_rate=44100).export(audio, 'output.wav')
```

### Problem: Drums and bass don't sync

**Cause:** Not locking timing together

**Solution:** Use exact same time values
```python
for bar in range(4):
    time = bar * 2.0  # Calculate once
    
    # Use same time for both
    song.play_note(60, 0.1, time + 0.0, 'kick')
    song.play_note(60, 0.1, time + 0.0, 'bass')  # Same start time
```

## Track and Arrangement Issues

### Problem: Can't hear certain tracks

**Causes:**
- Track volume is 0
- Track is muted
- Track name mismatch
- No notes on track

**Solutions:**

1. Check track exists
```python
# Make sure track was added
track = song.add_track('melody', SynthPresets.pluck())

# Use exact same name
song.play_note(440, 1.0, 0.0, 'melody')  # Matches track name
```

2. Check volume
```python
track = song.get_track('melody')
print(f"Volume: {track.volume}")  # Should be > 0
track.set_volume(0.8)
```

3. Check for mute
```python
track.unmute()  # In case it's muted
```

4. Verify notes exist
```python
# Make sure you're adding notes
song.play_note(440, 1.0, 0.0, 'melody')  # Do this!

# Not just:
song.add_track('melody', SynthPresets.pluck())  # Just setup
```

### Problem: Too many overlapping sounds

**Solution:** Simplify or use panning
```python
# Reduce number of simultaneous tracks
# Or pan them in stereo
track1.set_pan(-0.5)  # Left
track2.set_pan(0.5)   # Right

# Or lower volumes
track1.set_volume(0.6)
track2.set_volume(0.6)
track3.set_volume(0.4)
```

## Effects Issues

### Problem: Effects not being applied

**Causes:**
- Effect not added correctly
- Effect parameters wrong
- Effect chain in wrong order

**Solutions:**

1. Verify effect was added
```python
track.add_effect(ReverbFX(mix=0.3))

# You should hear reverb on this track
song.play_note(440, 2.0, 0.0, 'track')
```

2. Check effect parameters
```python
# Problem: mix=0 means no effect
reverb = ReverbFX(mix=0)  # No reverb heard

# Solution:
reverb = ReverbFX(mix=0.3)  # Now you'll hear it
```

3. Reorder effect chain
```python
# Effects apply in order added
# Put compression before distortion for controlled distortion
track.add_effect(Compressor(threshold=-15, ratio=3.0))
track.add_effect(DistortionFX(drive=5.0, mix=1.0))

# Better than:
# track.add_effect(DistortionFX(...))  # First
# track.add_effect(Compressor(...))    # Then - too late!
```

### Problem: Reverb/Delay is too loud or quiet

**Solution:** Adjust mix parameter
```python
# Too quiet (barely noticeable)
ReverbFX(mix=0.1)

# Good balance
ReverbFX(mix=0.3)

# Too much (overpowering)
ReverbFX(mix=0.8)
```

## Export Issues

### Problem: File won't export or export fails

**Causes:**
- Invalid filename
- Missing dependencies
- Disk space
- Permission issues

**Solutions:**

1. Check dependencies
```bash
pip install pydub
pip install moviepy  # For MP4
```

2. Use valid filename
```python
# Good:
song.render('output.wav')
song.render('/path/to/output.mp3')

# Bad:
song.render('output.wav|invalid.wav')  # Invalid char
song.render('')  # Empty name
```

3. Check disk space
```bash
# See available space
df -h
```

4. Use full path
```python
import os

output_dir = os.path.expanduser('~/Music')
output_file = os.path.join(output_dir, 'song.wav')
song.render(output_file)
```

## Video Visualization Issues

### Problem: Video creation slow or fails

**Causes:**
- Too high resolution
- Too many frequency bars
- Low disk space
- Missing dependencies

**Solutions:**

1. Lower resolution for testing
```python
# Instead of:
visualize_audio_file('audio.wav', 'out.mp4', viz,
    video_width=3840, video_height=2160, video_fps=60)

# Use:
visualize_audio_file('audio.wav', 'out.mp4', viz,
    video_width=1280, video_height=720, video_fps=24)

# Then upgrade to 1080p for final
```

2. Reduce FPS
```python
video_fps=24  # Instead of 60
# Still looks smooth, renders faster
```

3. Reduce bars (for frequency scope)
```python
viz = FrequencyScopeVisualizer(sample_rate=44100, num_bars=32)
# Instead of 128 bars
```

4. Check dependencies
```bash
pip install moviepy pillow
```

### Problem: Video is black or has no visualization

**Causes:**
- Wrong visualizer
- Visualizer not rendering
- Color issue

**Solutions:**

1. Use basic visualizer first
```python
from algorythm import WaveformVisualizer

viz = WaveformVisualizer(sample_rate=44100, color=(0, 255, 0))
# Simple waveform usually works
```

2. Verify audio file works
```bash
# Test audio first
ffplay audio.wav

# If audio doesn't work, that's the problem
```

3. Check colors
```python
# RGB color: (R, G, B) from 0-255
color=(255, 0, 0)    # Red
color=(0, 255, 0)    # Green
color=(0, 0, 255)    # Blue
color=(255, 255, 0)  # Yellow

# Not:
color='red'  # Wrong format
```

## Import Issues

### Problem: "ImportError: No module named 'algorythm'"

**Solution:** Install package
```bash
cd /path/to/algorythm
pip install -e .
```

### Problem: "ImportError: cannot import name 'X'"

**Causes:**
- Typo in import name
- Importing from wrong module
- Old version installed

**Solutions:**

1. Check spelling
```python
# Right:
from algorythm import Synth

# Wrong:
from algorythm import Synth2  # Doesn't exist
from algorythm import synth   # Wrong case
```

2. Check module location
```python
# Right:
from algorythm import ReverbFX  # In effects module

# Check __init__.py to see what's exported
```

3. Reinstall
```bash
pip uninstall algorythm
pip install -e .
```

## Performance Issues

### Problem: Generation is very slow

**Causes:**
- Too long duration
- Too many effects
- High sample rate
- CPU throttling

**Solutions:**

1. Test with shorter duration
```python
# Instead of 5 minutes:
song.render('test.wav')  # Just render what you need

# Shorter test:
audio = synth.generate_note(440, 1.0)  # 1 second
```

2. Reduce effects
```python
# Temporary: Remove effects to test
# track.add_effect(ReverbFX(mix=0.3))
# track.add_effect(DelayFX(...))

# Render without them first
```

3. Close other applications
```bash
# Free up RAM and CPU
# Close browsers, IDEs, etc.
```

## Memory Issues

### Problem: "MemoryError" or "out of memory"

**Causes:**
- Composition too long
- Too many tracks
- High sample rate with long duration

**Solutions:**

1. Break into sections
```python
# Instead of one 10-minute file:
for section in ['intro', 'verse', 'chorus']:
    song = Composition(tempo=120)
    # ... add 2 minutes of content
    song.render(f'{section}.wav')

# Then concatenate:
# ffmpeg -i "concat:intro.wav|verse.wav|chorus.wav" full.wav
```

2. Lower sample rate (if acceptable)
```python
# Instead of:
Composition(sample_rate=48000)

# Use:
Composition(sample_rate=44100)  # Standard, uses less memory
```

3. Reduce tracks
```python
# Too many simultaneous tracks uses memory
# Render instruments separately and mix later
```

## Getting Help

1. **Check error message** - Read it carefully
2. **Check documentation** - See GETTING_STARTED.md
3. **Try minimal example** - Simplify to isolate problem
4. **Check tempo calculations** - Most timing issues here
5. **Test with shorter duration** - Narrow down problem scope
6. **Verify file permissions** - Can you write to directory?
7. **Check dependencies** - All packages installed?

## Test Checklist

Before assuming there's a bug:

- [ ] Can you hear anything? (volume > 0?)
- [ ] Did you add tracks?
- [ ] Did you add notes to tracks?
- [ ] Is duration reasonable? (not 0.001 seconds)
- [ ] Is amplitude reasonable? (not all zeros)
- [ ] Did you render/export the file?
- [ ] Do you have a limiter on master?
- [ ] Are track names spelled correctly?
- [ ] Is tempo/BPM reasonable?
- [ ] Did you actually play the audio file?
