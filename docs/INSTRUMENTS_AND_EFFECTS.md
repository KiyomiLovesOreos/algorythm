# Instruments and Effects Guide

This guide covers all the instruments and effects available in Algorythm.

## Table of Contents
- [Instruments](#instruments)
- [Effects](#effects)
- [Presets](#presets)
- [Usage Examples](#usage-examples)

## Instruments

### Basic Synth
Standard subtractive synthesizer with oscillator, filter, and envelope.

```python
from algorythm import Synth, Filter, ADSR

synth = Synth(
    waveform='saw',  # 'sine', 'square', 'saw', 'triangle', 'noise'
    filter=Filter.lowpass(cutoff=2000, resonance=0.7),
    envelope=ADSR(attack=0.1, decay=0.2, sustain=0.7, release=0.3)
)

audio = synth.generate_note(frequency=440.0, duration=2.0)
```

### FM Synth
Frequency modulation synthesizer for complex harmonic sounds.

```python
from algorythm import FMSynth

fm = FMSynth(
    carrier_waveform='sine',
    modulator_waveform='sine',
    modulation_index=2.0,  # Amount of modulation
    mod_freq_ratio=2.0,    # Modulator frequency ratio
    envelope=ADSR(attack=0.01, decay=0.3, sustain=0.5, release=0.5)
)

audio = fm.generate_note(frequency=440.0, duration=1.0)
```

### Wavetable Synth
Morphs between different waveforms over time.

```python
from algorythm import WavetableSynth

wavetable = WavetableSynth.from_waveforms(
    waveforms=['sine', 'triangle', 'saw', 'square'],
    envelope=ADSR(attack=0.5, decay=0.3, sustain=0.8, release=1.0)
)

# Morph through wavetable (0.0 to 1.0)
audio = wavetable.generate_note(frequency=440.0, duration=2.0, position=0.5)
```

### Physical Model Synth
Simulates real instruments using physical modeling.

```python
from algorythm import PhysicalModelSynth

# String instrument (guitar, harp, etc.)
string = PhysicalModelSynth(
    model_type='string',
    damping=0.996,      # Higher = longer sustain
    brightness=0.7      # Tone brightness
)

# Drum/percussion
drum = PhysicalModelSynth(
    model_type='drum',
    damping=0.98,
    brightness=0.6
)

# Wind instrument (flute, clarinet, etc.)
wind = PhysicalModelSynth(
    model_type='wind',
    damping=0.995,
    brightness=0.5
)

audio = string.generate_note(frequency=440.0, duration=1.0)
```

### Additive Synth
Combines multiple sine wave harmonics.

```python
from algorythm import AdditiveeSynth

organ = AdditiveeSynth(
    num_harmonics=8,
    harmonic_amplitudes=[1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05],
    envelope=ADSR(attack=0.01, decay=0.1, sustain=1.0, release=0.1)
)

audio = organ.generate_note(frequency=440.0, duration=2.0)
```

### Pad Synth
Creates lush pad sounds with multiple detuned voices.

```python
from algorythm import PadSynth

pad = PadSynth(
    num_voices=7,           # More voices = thicker sound
    detune_amount=0.1,      # Amount of detuning (cents)
    waveform='saw',         # Base waveform
    envelope=ADSR(attack=2.0, decay=1.0, sustain=0.8, release=3.0),
    filter=Filter.lowpass(cutoff=2000, resonance=0.5)
)

audio = pad.generate_note(frequency=440.0, duration=4.0)
```

### Granular Synth
Creates textures by manipulating tiny audio grains.

```python
from algorythm import GranularSynth, Sample

sample = Sample(file_path='audio.wav')
granular = GranularSynth(
    sample=sample,
    grain_size=0.05,        # Size of each grain in seconds
    grain_density=20.0,     # Grains per second
    grain_envelope='hann'   # 'rectangular', 'triangular', 'gaussian', 'hann'
)

audio = granular.synthesize(
    duration=5.0,
    position_range=(0.0, 1.0),    # Where to read from sample
    pitch_range=(-12, 12),         # Pitch variation in semitones
    spatial_spread=0.5,            # Stereo spread
    density_variation=0.2          # Randomize density
)
```

## Effects

### Time-Based Effects

#### Reverb
Adds space and depth to sounds.

```python
from algorythm.effects import Reverb

reverb = Reverb(
    room_size=0.7,      # 0.0-1.0 (small room to large hall)
    damping=0.5,        # High frequency damping
    wet_level=0.3       # Mix amount
)

processed = reverb.apply(audio, sample_rate=44100)
```

#### Delay
Creates echo effects.

```python
from algorythm.effects import Delay

delay = Delay(
    delay_time=0.5,     # Delay in seconds
    feedback=0.5,       # Amount of repetition
    wet_level=0.5       # Mix amount
)

processed = delay.apply(audio)
```

### Modulation Effects

#### Chorus
Thickens sounds with modulated delays.

```python
from algorythm.effects import Chorus

chorus = Chorus(
    depth=0.003,        # Modulation depth
    rate=1.5,           # LFO rate in Hz
    mix=0.5             # Mix amount
)

processed = chorus.apply(audio)
```

#### Flanger
Creates sweeping comb filter effects.

```python
from algorythm.effects import Flanger

flanger = Flanger(
    depth=0.002,
    rate=0.25,
    feedback=0.7,
    mix=0.5
)

processed = flanger.apply(audio)
```

#### Phaser
Creates notch filter sweeps.

```python
from algorythm.effects import Phaser

phaser = Phaser(
    rate=0.5,
    depth=1.0,
    feedback=0.5,
    mix=0.5
)

processed = phaser.apply(audio)
```

#### Tremolo
Amplitude modulation effect.

```python
from algorythm.effects import Tremolo

tremolo = Tremolo(
    rate=5.0,           # Modulation rate in Hz
    depth=0.5,          # Modulation depth
    waveform='sine'     # 'sine', 'square', 'triangle'
)

processed = tremolo.apply(audio)
```

#### Vibrato
Pitch modulation effect.

```python
from algorythm.effects import Vibrato

vibrato = Vibrato(
    rate=5.0,
    depth=0.002
)

processed = vibrato.apply(audio)
```

### Distortion Effects

#### Distortion
Waveshaping distortion with tone control.

```python
from algorythm.effects import Distortion

distortion = Distortion(
    drive=5.0,          # Amount of distortion
    tone=0.5,           # Tone control (0.0=dark, 1.0=bright)
    mix=1.0             # Mix amount
)

processed = distortion.apply(audio)
```

#### Overdrive
Smooth tube-like distortion.

```python
from algorythm.effects import Overdrive

overdrive = Overdrive(
    drive=2.0,
    tone=0.7,
    mix=1.0
)

processed = overdrive.apply(audio)
```

#### Fuzz
Extreme clipping distortion.

```python
from algorythm.effects import Fuzz

fuzz = Fuzz(
    gain=10.0,
    mix=1.0
)

processed = fuzz.apply(audio)
```

### Dynamics Effects

#### Compressor
Controls dynamic range.

```python
from algorythm.effects import Compressor

compressor = Compressor(
    threshold=-20.0,    # Threshold in dB
    ratio=4.0,          # Compression ratio
    attack=0.005,       # Attack time in seconds
    release=0.1,        # Release time in seconds
    makeup_gain=1.0     # Output gain
)

processed = compressor.apply(audio)
```

#### Limiter
Prevents signal from exceeding threshold.

```python
from algorythm.effects import Limiter

limiter = Limiter(
    threshold=-1.0,     # Ceiling in dB
    release=0.05
)

processed = limiter.apply(audio)
```

#### Gate
Removes low-level signals.

```python
from algorythm.effects import Gate

gate = Gate(
    threshold=-40.0,    # Gate threshold in dB
    attack=0.001,
    release=0.1
)

processed = gate.apply(audio)
```

### Special Effects

#### Ring Modulator
Creates metallic, bell-like sounds.

```python
from algorythm.effects import RingModulator

ring_mod = RingModulator(
    carrier_freq=440.0,
    mix=0.5
)

processed = ring_mod.apply(audio)
```

#### Bit Crusher
Lo-fi digital distortion.

```python
from algorythm.effects import BitCrusher

bitcrusher = BitCrusher(
    bit_depth=8,                # Target bit depth
    sample_rate_reduction=0.5,  # Sample rate reduction factor
    mix=1.0
)

processed = bitcrusher.apply(audio)
```

#### Auto Pan
Stereo panning modulation.

```python
from algorythm.effects import AutoPan

autopan = AutoPan(
    rate=1.0,
    depth=1.0
)

processed = autopan.apply(audio)
```

### Effect Chain
Combine multiple effects.

```python
from algorythm.effects import EffectChain, Distortion, Chorus, Reverb

chain = EffectChain()
chain.add_effect(Distortion(drive=3.0))
chain.add_effect(Chorus(mix=0.3))
chain.add_effect(Reverb(room_size=0.8, wet_level=0.4))

processed = chain.apply(audio)
```

## Presets

Algorythm includes high-quality presets for a wide variety of instruments:

```python
from algorythm import SynthPresets

# Basic Synth Sounds
warm_pad = SynthPresets.warm_pad()
pluck = SynthPresets.pluck()
bass = SynthPresets.bass()
lead = SynthPresets.lead()

# Keyboard Instruments
piano = SynthPresets.piano()
electric_piano = SynthPresets.electric_piano()
organ = SynthPresets.organ()
harpsichord = SynthPresets.harpsichord()
clavinet = SynthPresets.clavinet()

# String Instruments
guitar = SynthPresets.guitar()
violin = SynthPresets.violin()
cello = SynthPresets.cello()
strings = SynthPresets.strings()
harp = SynthPresets.harp()
banjo = SynthPresets.banjo()
sitar = SynthPresets.sitar()

# Wind Instruments
flute = SynthPresets.flute()
clarinet = SynthPresets.clarinet()
trumpet = SynthPresets.trumpet()
saxophone = SynthPresets.saxophone()

# Brass Instruments
brass = SynthPresets.brass()
synth_brass = SynthPresets.synth_brass()

# Vocal/Choir
choir = SynthPresets.choir()

# Mallet/Percussion Instruments
bell = SynthPresets.bell()
marimba = SynthPresets.marimba()
xylophone = SynthPresets.xylophone()
vibraphone = SynthPresets.vibraphone()
glockenspiel = SynthPresets.glockenspiel()
kalimba = SynthPresets.kalimba()
music_box = SynthPresets.music_box()
steel_drum = SynthPresets.steel_drum()

# Drum Sounds
kick_drum = SynthPresets.kick_drum()
snare_drum = SynthPresets.snare_drum()
hi_hat = SynthPresets.hi_hat()
tom_drum = SynthPresets.tom_drum()
cymbal = SynthPresets.cymbal()
drum = SynthPresets.drum()

# Synth Sounds
synth_lead = SynthPresets.synth_lead()
synth_pluck = SynthPresets.synth_pluck()
arp_synth = SynthPresets.arp_synth()
acid_bass = SynthPresets.acid_bass()
fat_bass = SynthPresets.fat_bass()
sub_bass = SynthPresets.sub_bass()
ambient_pad = SynthPresets.ambient_pad()
noise_sweep = SynthPresets.noise_sweep()

# Use in composition
audio = bell.generate_note(frequency=440.0, duration=2.0)
```

### Preset Categories

**Keyboard Instruments**: Piano, electric piano, organ, harpsichord, clavinet

**String Instruments**: Guitar, violin, cello, strings ensemble, harp, banjo, sitar

**Wind Instruments**: Flute, clarinet, trumpet, saxophone

**Brass**: Brass section, synth brass

**Vocal**: Choir pad

**Mallet Percussion**: Bell, marimba, xylophone, vibraphone, glockenspiel, kalimba, music box, steel drum

**Drums**: Kick, snare, hi-hat, tom, cymbal

**Synth Leads/Basses**: Synth lead, pluck, arp synth, acid bass, fat bass, sub bass

**Pads**: Warm pad, ambient pad, noise sweep

## Usage Examples

### Complete Track with Effects

```python
from algorythm import *

# Create composition
comp = Composition(tempo=120)

# Create instruments
bass = SynthPresets.bass()
lead = SynthPresets.lead()
pad = SynthPresets.strings()

# Create scale and motifs
scale = Scale.major('C', 4)
bass_motif = Motif(notes=[scale.tonic()] * 4, durations=[1.0] * 4)
lead_motif = Motif(notes=scale.ascending(8), durations=[0.5] * 8)
pad_chord = Motif(notes=[scale.note(0), scale.note(2), scale.note(4)], durations=[8.0] * 3)

# Create tracks
bass_track = Track("Bass")
bass_track.add_motif(bass_motif, instrument=bass)

lead_track = Track("Lead")
lead_track.add_motif(lead_motif, instrument=lead)

pad_track = Track("Pad")
pad_track.add_motif(pad_chord, instrument=pad)

# Add effects
from algorythm.effects import Distortion, Reverb, Delay

bass_track.add_effect(Distortion(drive=3.0, mix=0.7))
bass_track.add_effect(Reverb(room_size=0.4, wet_level=0.2))

lead_track.add_effect(Delay(delay_time=0.375, feedback=0.4, wet_level=0.5))
lead_track.add_effect(Reverb(room_size=0.6, wet_level=0.3))

pad_track.add_effect(Chorus(mix=0.4))
pad_track.add_effect(Reverb(room_size=0.8, wet_level=0.4))

# Add to composition
comp.add_track(bass_track)
comp.add_track(lead_track)
comp.add_track(pad_track)

# Render
engine = RenderEngine()
audio = engine.render(comp)

# Export
exporter = Exporter()
exporter.export_wav(audio, "my_track.wav")
```

### Experimental Sound Design

```python
from algorythm import *
from algorythm.effects import *

# Create experimental instrument
granular = GranularSynth.from_file('sample.wav', grain_size=0.03, grain_density=30)

# Generate texture
texture = granular.synthesize(
    duration=10.0,
    position_range=(0.2, 0.8),
    pitch_range=(-7, 7),
    spatial_spread=0.8,
    density_variation=0.3
)

# Apply complex effect chain
chain = EffectChain()
chain.add_effect(RingModulator(carrier_freq=200.0, mix=0.3))
chain.add_effect(Phaser(rate=0.3, mix=0.5))
chain.add_effect(BitCrusher(bit_depth=10, sample_rate_reduction=0.7, mix=0.4))
chain.add_effect(Reverb(room_size=0.9, wet_level=0.5))

processed = chain.apply(texture)
```

## Tips and Best Practices

1. **Layering**: Combine multiple instruments for richer sounds
2. **Effect Order**: Generally: Distortion → Modulation → Time-based (Delay/Reverb)
3. **Subtlety**: Start with low mix values and increase gradually
4. **CPU Usage**: Physical modeling and granular synthesis are CPU-intensive
5. **Experimentation**: Try unconventional effect chains for unique sounds

## See Also

- [BEGINNER_GUIDE.md](BEGINNER_GUIDE.md) - Getting started
- [CHEAT_SHEET.md](CHEAT_SHEET.md) - Quick reference
- [examples/](examples/) - More examples
