# Synthesis Guide

This guide covers all the synthesis engines and instrument presets in Algorythm.

## Basic Synthesis

The `Synth` class is the foundation. It generates audio using oscillators, filters, and envelopes.

### Creating a Synth

```python
from algorythm import Synth, Oscillator, Filter, ADSR

# Simple synth with just a waveform
synth = Synth(waveform='sine')

# Synth with filter
synth = Synth(
    waveform='saw',
    filter=Filter.lowpass(cutoff=2000, resonance=0.5)
)

# Synth with envelope
synth = Synth(
    waveform='square',
    envelope=ADSR(attack=0.1, decay=0.2, sustain=0.7, release=0.3)
)

# Full custom synth
synth = Synth(
    waveform='saw',
    filter=Filter.lowpass(cutoff=1500),
    envelope=ADSR(attack=0.01, decay=0.1, sustain=0.8, release=0.5),
    amplitude=0.8
)
```

### Waveforms

Available waveform types:
- `sine` - Pure tone, no harmonics
- `square` - Hollow, odd harmonics
- `saw` - Bright, all harmonics
- `triangle` - Soft, fewer harmonics than square
- `noise` - White noise (random)

### Filters

Filters shape the frequency content:

```python
from algorythm import Filter

# Lowpass - removes high frequencies
Filter.lowpass(cutoff=1000, resonance=0.5)

# Highpass - removes low frequencies
Filter.highpass(cutoff=500, resonance=0.3)

# Bandpass - keeps a range of frequencies
Filter.bandpass(center=1000, resonance=0.7)

# Notch - removes a range of frequencies
Filter.notch(center=2000, resonance=0.5)
```

Parameters:
- `cutoff`/`center` - Frequency in Hz
- `resonance` - Emphasis at cutoff (0.0-1.0)

### Envelopes

ADSR envelopes control how sound evolves over time:

```python
from algorythm import ADSR

envelope = ADSR(
    attack=0.1,   # Time to reach peak (seconds)
    decay=0.2,    # Time to fall to sustain level
    sustain=0.7,  # Level to hold (0.0-1.0)
    release=0.3   # Time to fade out after note ends
)
```

Common envelope shapes:
- Pluck: fast attack, no sustain, medium release
- Pad: slow attack, high sustain, slow release
- Percussive: very fast attack, no sustain, short release

## Advanced Synthesis Engines

### FM Synthesis

Frequency Modulation creates complex timbres by modulating one oscillator with another:

```python
from algorythm import FMSynth

# Simple FM
fm = FMSynth(
    carrier_freq=440,      # Main frequency
    modulator_freq=880,    # Modulation frequency (2x carrier = metallic)
    mod_index=5            # Modulation amount
)

audio = fm.generate(duration=1.0)
```

Higher `mod_index` creates more harmonics. Try different frequency ratios for different timbres.

### Wavetable Synthesis

Uses custom waveforms stored in a table:

```python
from algorythm import WavetableSynth
import numpy as np

# Create a custom wavetable (single cycle)
wavetable = np.sin(np.linspace(0, 2*np.pi, 2048))

synth = WavetableSynth(wavetable=wavetable)
audio = synth.generate_note(frequency=440, duration=1.0)
```

### Physical Modeling

Simulates physical instruments using Karplus-Strong algorithm:

```python
from algorythm import PhysicalModelSynth

synth = PhysicalModelSynth(
    brightness=0.5,  # Higher = brighter sound
    damping=0.995    # Higher = longer sustain
)

audio = synth.generate_note(frequency=440, duration=2.0)
```

Good for plucked strings, guitar-like sounds.

### Additive Synthesis

Builds sounds from multiple sine waves:

```python
from algorythm import AdditiveeSynth

synth = AdditiveeSynth(
    num_harmonics=8,        # Number of overtones
    harmonic_decay=0.5      # How quickly harmonics fade
)

audio = synth.generate_note(frequency=440, duration=1.0)
```

### Granular Synthesis

Creates textures from small audio grains:

```python
from algorythm import GranularSynth, Sample

# Load a sample
sample = Sample('path/to/audio.wav')

synth = GranularSynth(
    sample=sample,
    grain_size=0.1,      # Grain length in seconds
    density=20,          # Grains per second
    pitch=1.0,           # Pitch shift (1.0 = original)
    spray=0.1            # Random position variation
)

audio = synth.generate(duration=5.0)
```

## Instrument Presets

Instead of building synths manually, use the 50+ presets:

```python
from algorythm import SynthPresets

# Get a preset
instrument = SynthPresets.pluck()
```

### Synth Presets

Modern synthesizer sounds:

- `synth_lead()` - Bright lead synth
- `synth_pad()` - Soft pad for background
- `synth_bass()` - Deep bass synth
- `warm_pad()` - Warm, lush pad
- `bright_lead()` - Cutting lead sound
- `soft_pad()` - Gentle pad texture

### Plucked Instruments

String instruments:

- `pluck()` - Generic plucked string
- `guitar()` - Acoustic guitar tone
- `harp()` - Harp-like sound
- `banjo()` - Bright, twangy

### Keys

Keyboard instruments:

- `piano()` - Acoustic piano
- `electric_piano()` - Electric piano (Rhodes-style)
- `bell()` - Bell-like tone
- `glockenspiel()` - Metallic bell

### Brass

Brass section:

- `brass()` - Generic brass
- `trumpet()` - Trumpet sound
- `trombone()` - Trombone
- `french_horn()` - French horn

### Strings

Orchestral strings:

- `strings()` - String section
- `violin()` - Solo violin
- `cello()` - Cello
- `pizzicato()` - Plucked strings

### Bass

Bass instruments:

- `bass()` - Acoustic bass
- `electric_bass()` - Electric bass
- `sub_bass()` - Sub bass for electronic music
- `upright_bass()` - Upright jazz bass

### Drums

Drum sounds:

- `kick()` - Kick drum
- `snare()` - Snare drum
- `hihat()` - Hi-hat
- `clap()` - Hand clap
- `tom()` - Tom drum
- `cymbal()` - Crash cymbal

### Sound Effects

Special sounds:

- `laser()` - Laser zap
- `explosion()` - Explosion
- `woosh()` - Swoosh effect

### Complete List

Run this to see all presets:

```python
from algorythm import SynthPresets

# Get all preset names
presets = [name for name in dir(SynthPresets) if not name.startswith('_')]
print(f"Available presets: {len(presets)}")
for preset in presets:
    print(f"  - {preset}")
```

## Using Presets in Compositions

```python
from algorythm import Composition, SynthPresets, Scale, Motif

song = Composition(tempo=120)

# Add tracks with different instruments
song.add_track('lead', SynthPresets.synth_lead())
song.add_track('bass', SynthPresets.synth_bass())
song.add_track('pad', SynthPresets.warm_pad())
song.add_track('drums', SynthPresets.kick())

# Create melodies for each
scale = Scale.minor('A')
lead_melody = Motif.from_intervals([0, 3, 5, 7], scale=scale)
bass_line = Motif.from_intervals([0, 0, 0, 0], scale=scale, octave=2)

# Arrange
song.play_motif(lead_melody, start=0.0, track='lead')
song.play_motif(bass_line, start=0.0, track='bass')

song.render('multi_track.wav')
```

## Custom Instruments

Build your own instruments by combining components:

```python
from algorythm import Synth, Filter, ADSR, Oscillator

def my_custom_instrument():
    return Synth(
        waveform='saw',
        filter=Filter.lowpass(cutoff=2000, resonance=0.7),
        envelope=ADSR(attack=0.01, decay=0.1, sustain=0.6, release=0.5),
        amplitude=0.7
    )

# Use it
instrument = my_custom_instrument()
audio = instrument.generate_note(440, 1.0)
```

## Tips

1. Start with presets and modify them if needed
2. Lower resonance values sound more natural
3. Fast attack + short release = percussive
4. Slow attack + long release = pad/ambient
5. Use filters to tame harsh waveforms
6. Layer multiple instruments for richer sounds
