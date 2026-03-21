# Effects Guide

Complete reference for all audio effects in Algorythm.

## Effect Basics

Effects modify audio signals. Apply them to individual tracks or entire compositions.

### Adding Effects to Tracks

```python
from algorythm import Composition, SynthPresets, ReverbFX, DelayFX

song = Composition(tempo=120)
track = song.add_track('melody', SynthPresets.pluck())

# Add effects
track.add_effect(ReverbFX(mix=0.3))
track.add_effect(DelayFX(delay_time=0.5, feedback=0.3))
```

### Effect Chains

Chain multiple effects in order:

```python
from algorythm import FXChain, ReverbFX, DelayFX, ChorusFX

chain = FXChain()
chain.add(ChorusFX(rate=0.5, depth=0.3))
chain.add(DelayFX(delay_time=0.375, feedback=0.4))
chain.add(ReverbFX(mix=0.2, room_size=0.6))

track.add_effect_chain(chain)
```

Effects are applied in the order you add them.

## Time-Based Effects

Effects that use delays and repeats.

### Reverb

Simulates acoustic spaces:

```python
from algorythm import ReverbFX

reverb = ReverbFX(
    mix=0.3,         # Wet/dry balance (0-1)
    room_size=0.5,   # Size of space (0-1)
    damping=0.5      # High frequency absorption (0-1)
)
```

Use cases:
- Small room: `room_size=0.3, damping=0.7`
- Hall: `room_size=0.8, damping=0.3`
- Plate: `room_size=0.5, damping=0.9`

### Delay

Echo effect with feedback:

```python
from algorythm import DelayFX

delay = DelayFX(
    delay_time=0.5,   # Delay time in seconds
    feedback=0.3,     # How much signal feeds back (0-1)
    mix=0.3           # Wet/dry balance (0-1)
)
```

Musical delay times:
- Quarter note at 120 BPM: 0.5 seconds
- Eighth note at 120 BPM: 0.25 seconds
- Dotted eighth at 120 BPM: 0.375 seconds

### Chorus

Thickens sound by detuning copies:

```python
from algorythm import ChorusFX

chorus = ChorusFX(
    rate=0.5,      # LFO rate in Hz
    depth=0.3,     # Modulation depth (0-1)
    mix=0.4        # Wet/dry balance (0-1)
)
```

Subtle for warmth, heavy for shimmer.

### Flanger

Swooshing jet-plane effect:

```python
from algorythm import FlangerFX

flanger = FlangerFX(
    rate=0.5,       # LFO rate in Hz
    depth=0.5,      # Modulation depth (0-1)
    feedback=0.3,   # Feedback amount (0-1)
    mix=0.5         # Wet/dry balance (0-1)
)
```

Higher feedback = more metallic sound.

### Phaser

Sweeping filter effect:

```python
from algorythm import PhaserFX

phaser = PhaserFX(
    rate=0.5,       # LFO rate in Hz
    depth=0.5,      # Modulation depth (0-1)
    feedback=0.3,   # Feedback amount (0-1)
    mix=0.5         # Wet/dry balance (0-1)
)
```

Similar to flanger but smoother.

## Dynamics Effects

Control volume and dynamics.

### Compressor

Reduces dynamic range:

```python
from algorythm import Compressor

compressor = Compressor(
    threshold=-20,    # Level where compression starts (dB)
    ratio=4.0,        # Compression ratio (1:1 to 20:1)
    attack=0.005,     # Attack time in seconds
    release=0.1,      # Release time in seconds
    makeup_gain=0     # Output gain in dB
)
```

Use cases:
- Gentle: `ratio=2.0, threshold=-15`
- Heavy: `ratio=8.0, threshold=-25`
- Limiting: `ratio=20.0, threshold=-10`

### Limiter

Prevents signal from exceeding a threshold:

```python
from algorythm import Limiter

limiter = Limiter(
    threshold=-3,     # Maximum level in dB
    release=0.05      # Release time in seconds
)
```

Use on master track to prevent clipping.

### Gate

Silences signal below threshold:

```python
from algorythm import Gate

gate = Gate(
    threshold=-40,    # Level below which gate closes (dB)
    attack=0.001,     # Attack time in seconds
    release=0.1       # Release time in seconds
)
```

Good for removing noise between notes.

## Distortion Effects

Add harmonics and grit.

### Distortion

Classic distortion:

```python
from algorythm import DistortionFX

distortion = DistortionFX(
    drive=5.0,      # Amount of distortion (1-20)
    tone=0.5,       # Tone control (0-1)
    mix=1.0         # Wet/dry balance (0-1)
)
```

### Overdrive

Warm tube-like distortion:

```python
from algorythm import Overdrive

overdrive = Overdrive(
    drive=3.0,      # Amount of overdrive (1-10)
    tone=0.6,       # Tone control (0-1)
    mix=1.0         # Wet/dry balance (0-1)
)
```

Softer than distortion.

### Fuzz

Aggressive fuzzy distortion:

```python
from algorythm import Fuzz

fuzz = Fuzz(
    drive=8.0,      # Amount of fuzz (1-20)
    tone=0.5,       # Tone control (0-1)
    mix=1.0         # Wet/dry balance (0-1)
)
```

Very aggressive, lots of harmonics.

### Bitcrusher

Digital lo-fi distortion:

```python
from algorythm import BitCrusherFX

bitcrusher = BitCrusherFX(
    bit_depth=8,        # Bits (1-16, lower = more distortion)
    sample_rate=8000,   # Sample rate reduction in Hz
    mix=1.0             # Wet/dry balance (0-1)
)
```

Lower values = more lo-fi.

## Modulation Effects

Vary parameters over time.

### Tremolo

Volume modulation:

```python
from algorythm import TremoloFX

tremolo = TremoloFX(
    rate=5.0,       # Modulation rate in Hz
    depth=0.5,      # Modulation depth (0-1)
    mix=1.0         # Wet/dry balance (0-1)
)
```

### Vibrato

Pitch modulation:

```python
from algorythm import Vibrato

vibrato = Vibrato(
    rate=5.0,       # Modulation rate in Hz
    depth=0.02,     # Modulation depth (0-1)
    mix=1.0         # Wet/dry balance (0-1)
)
```

Keep depth low for natural vibrato.

### Auto-Pan

Automatic stereo panning:

```python
from algorythm import AutoPan

autopan = AutoPan(
    rate=0.5,       # Panning rate in Hz
    depth=0.8,      # Panning depth (0-1)
    waveform='sine' # LFO waveform ('sine', 'triangle', 'square')
)
```

### Ring Modulator

Metallic, bell-like modulation:

```python
from algorythm import RingModulator

ringmod = RingModulator(
    frequency=100,   # Modulation frequency in Hz
    mix=0.5          # Wet/dry balance (0-1)
)
```

Inharmonic and experimental.

## Creative Effects

Unusual effects for sound design.

### Stutter

Repeats small slices:

```python
from algorythm import Stutter

stutter = Stutter(
    slice_length=0.1,   # Slice size in seconds
    repetitions=4,      # Times to repeat each slice
    mix=1.0             # Wet/dry balance (0-1)
)
```

### Beat Repeat

Rhythmic looping effect:

```python
from algorythm import BeatRepeat

beatrepeat = BeatRepeat(
    loop_length=0.5,    # Loop size in seconds
    probability=0.3,    # Chance to trigger (0-1)
    mix=0.5             # Wet/dry balance (0-1)
)
```

Randomly captures and repeats beats.

### Freeze

Freezes audio in time:

```python
from algorythm import Freeze

freeze = Freeze(
    freeze_length=2.0,  # Length to freeze in seconds
    mix=0.5             # Wet/dry balance (0-1)
)
```

### Reverse

Plays audio backwards:

```python
from algorythm import Reverse

reverse = Reverse(
    mix=1.0    # Wet/dry balance (0-1)
)
```

### Filter Sweep

Animated filter sweep:

```python
from algorythm import FilterSweep

sweep = FilterSweep(
    filter_type='lowpass',  # Filter type
    start_freq=200,         # Starting frequency in Hz
    end_freq=5000,          # Ending frequency in Hz
    duration=2.0,           # Sweep duration in seconds
    resonance=0.7           # Filter resonance (0-1)
)
```

## EQ and Filtering

### EQ

Multi-band equalizer:

```python
from algorythm import EQ

eq = EQ(
    low_gain=0,      # Low frequency gain in dB
    mid_gain=3,      # Mid frequency gain in dB
    high_gain=-2,    # High frequency gain in dB
    low_freq=200,    # Low band center frequency
    mid_freq=1000,   # Mid band center frequency
    high_freq=5000   # High band center frequency
)
```

## Practical Examples

### Vocal Chain

```python
from algorythm import FXChain, Compressor, EQ, ReverbFX

vocal_chain = FXChain()
vocal_chain.add(Compressor(threshold=-15, ratio=3.0))
vocal_chain.add(EQ(low_gain=-3, mid_gain=2, high_gain=3))
vocal_chain.add(ReverbFX(mix=0.2, room_size=0.4))

track.add_effect_chain(vocal_chain)
```

### Guitar Effect

```python
from algorythm import FXChain, Overdrive, ChorusFX, DelayFX, ReverbFX

guitar_chain = FXChain()
guitar_chain.add(Overdrive(drive=4.0))
guitar_chain.add(ChorusFX(rate=0.5, depth=0.3))
guitar_chain.add(DelayFX(delay_time=0.375, feedback=0.4))
guitar_chain.add(ReverbFX(mix=0.25))

track.add_effect_chain(guitar_chain)
```

### Lo-Fi Effect

```python
from algorythm import FXChain, BitCrusherFX, DistortionFX, ReverbFX

lofi_chain = FXChain()
lofi_chain.add(BitCrusherFX(bit_depth=8, sample_rate=11025))
lofi_chain.add(DistortionFX(drive=2.0))
lofi_chain.add(ReverbFX(mix=0.4, room_size=0.3, damping=0.8))

track.add_effect_chain(lofi_chain)
```

### Ambient Pad

```python
from algorythm import FXChain, ReverbFX, DelayFX, ChorusFX

ambient_chain = FXChain()
ambient_chain.add(ChorusFX(rate=0.3, depth=0.5))
ambient_chain.add(DelayFX(delay_time=1.0, feedback=0.6, mix=0.3))
ambient_chain.add(ReverbFX(mix=0.6, room_size=0.9, damping=0.3))

track.add_effect_chain(ambient_chain)
```

## Tips

1. Less is more - start with one or two effects
2. Order matters - generally: dynamics → EQ → distortion → modulation → time-based
3. Use mix parameter to blend effects subtly
4. Reverb and delay at end of chain for natural sound
5. Compression before distortion for consistent drive
6. Use limiter on master to prevent clipping
7. Automate effect parameters for movement
8. Test effects on different types of sounds
