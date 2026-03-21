# Quick Reference

Fast lookup for common Algorythm tasks.

## Installation

```bash
pip install -e .
```

## Basic Imports

```python
from algorythm import (
    # Synthesis
    Synth, SynthPresets, Oscillator, Filter, ADSR,
    # Sequencing
    Scale, Motif, Rhythm, Chord, Arpeggiator,
    # Composition
    Composition, Track,
    # Effects
    ReverbFX, DelayFX, ChorusFX, Compressor,
    # Export
    Exporter,
    # Visualization
    visualize_audio_file, FrequencyScopeVisualizer
)
```

## Quick Tasks

### Play a Note

```python
synth = Synth(waveform='sine')
audio = synth.generate_note(440, 1.0)
Exporter().export(audio, 'note.wav')
```

### Use a Preset

```python
instrument = SynthPresets.pluck()
audio = instrument.generate_note(440, 1.0)
```

### Create a Melody

```python
scale = Scale.major('C')
melody = Motif.from_intervals([0, 2, 4, 5, 7], scale=scale)
```

### Simple Song

```python
song = Composition(tempo=120)
song.add_track('melody', SynthPresets.pluck())
melody = Motif.from_intervals([0, 2, 4, 5], scale=Scale.major('C'))
song.play_motif(melody, start=0.0, track='melody')
song.render('song.wav')
```

### Add Effects

```python
track.add_effect(ReverbFX(mix=0.3))
track.add_effect(DelayFX(delay_time=0.5, feedback=0.3))
```

### Create Video

```python
viz = FrequencyScopeVisualizer(sample_rate=44100)
visualize_audio_file('audio.wav', 'video.mp4', viz)
```

## Common Patterns

### Full Song Structure

```python
song = Composition(tempo=120)

# Add tracks
melody = song.add_track('melody', SynthPresets.pluck())
bass = song.add_track('bass', SynthPresets.synth_bass())

# Add effects
melody.add_effect(ReverbFX(mix=0.2))

# Create patterns
scale = Scale.minor('A')
melody_pattern = Motif.from_intervals([0, 2, 3, 5], scale=scale)
bass_pattern = Motif.from_intervals([0, 0, 0, 0], scale=scale, octave=2)

# Arrange
song.play_motif(melody_pattern, start=0.0, track='melody')
song.play_motif(bass_pattern, start=0.0, track='bass')

# Render
song.render('output.wav')
```

### Custom Instrument

```python
instrument = Synth(
    waveform='saw',
    filter=Filter.lowpass(cutoff=2000, resonance=0.5),
    envelope=ADSR(attack=0.01, decay=0.1, sustain=0.7, release=0.3)
)
```

### Chord Progression

```python
chords = [
    Chord.major('C'),
    Chord.major('F'),
    Chord.major('G'),
    Chord.major('C')
]

for i, chord in enumerate(chords):
    # Play chord at time i*2.0
    pass
```

## Waveforms

- `sine` - Pure tone
- `square` - Hollow sound
- `saw` - Bright buzz
- `triangle` - Soft
- `noise` - White noise

## Scales

```python
Scale.major('C')
Scale.minor('A')
Scale.harmonic_minor('E')
Scale.pentatonic('G')
Scale.blues('C')
```

## Preset Categories

```python
# Synth
SynthPresets.synth_lead()
SynthPresets.synth_bass()
SynthPresets.warm_pad()

# Instruments
SynthPresets.pluck()
SynthPresets.piano()
SynthPresets.strings()
SynthPresets.brass()

# Drums
SynthPresets.kick()
SynthPresets.snare()
SynthPresets.hihat()
```

## Common Effects

```python
# Reverb
ReverbFX(mix=0.3, room_size=0.5, damping=0.5)

# Delay
DelayFX(delay_time=0.5, feedback=0.3, mix=0.3)

# Chorus
ChorusFX(rate=0.5, depth=0.3, mix=0.4)

# Distortion
DistortionFX(drive=5.0, tone=0.5, mix=1.0)

# Compression
Compressor(threshold=-20, ratio=4.0, attack=0.005, release=0.1)
```

## Timing Helpers

```python
# At 120 BPM:
# 1 bar = 2.0 seconds
# 1 beat = 0.5 seconds
# 1 eighth = 0.25 seconds

def bars_to_time(bars, tempo=120):
    return (bars * 4) / (tempo / 60)

# Use:
start = bars_to_time(4)  # Start at bar 4
```

## File Formats

```python
# WAV (lossless)
song.render('output.wav')

# MP3 (compressed)
song.render('output.mp3')

# FLAC (lossless compressed)
song.render('output.flac')
```

## Video Settings

```python
# Preview (fast)
visualize_audio_file('in.wav', 'out.mp4', viz,
    video_width=1280, video_height=720, video_fps=24)

# Production (quality)
visualize_audio_file('in.wav', 'out.mp4', viz,
    video_width=1920, video_height=1080, video_fps=30)
```

## Troubleshooting

### No sound
- Check amplitude/volume levels
- Ensure sample rate matches (44100)
- Check for clipping (values > 1.0)

### Audio distorted
- Lower amplitude/volume
- Add Limiter effect
- Reduce effect mix amounts

### Video too slow
- Lower resolution
- Reduce FPS
- Use fewer frequency bars

### Import errors
```bash
# Core
pip install numpy pydub pyyaml

# Video
pip install moviepy pillow

# Playback
pip install sounddevice
```

## Resources

- `docs/GETTING_STARTED.md` - Beginner tutorial
- `docs/SYNTHESIS.md` - All about sound synthesis
- `docs/SEQUENCING.md` - Melodies and patterns
- `docs/EFFECTS.md` - Complete effects guide
- `docs/COMPOSITION.md` - Multi-track arrangements
- `docs/GENERATIVE.md` - Algorithmic music
- `docs/VISUALIZATION.md` - Audio to video
- `docs/API_REFERENCE.md` - Complete API docs
- `examples/` - Working code examples
