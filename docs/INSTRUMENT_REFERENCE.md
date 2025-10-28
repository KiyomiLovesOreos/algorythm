# Algorythm Instrument Quick Reference

## Usage
```python
from algorythm.synth import SynthPresets

# Create an instrument
instrument = SynthPresets.piano()

# Generate a note
audio = instrument.generate_note(frequency=440.0, duration=2.0)
```

## All Available Instruments (45 Presets)

### 🎹 Keyboard (5)
```python
SynthPresets.piano()           # Acoustic piano
SynthPresets.electric_piano()  # Electric piano
SynthPresets.organ()           # Hammond-style organ
SynthPresets.harpsichord()     # Baroque harpsichord
SynthPresets.clavinet()        # Funky clavinet
```

### 🎻 Strings (7)
```python
SynthPresets.guitar()          # Acoustic guitar
SynthPresets.violin()          # Violin
SynthPresets.cello()           # Cello
SynthPresets.strings()         # String ensemble
SynthPresets.harp()            # Concert harp
SynthPresets.banjo()           # 5-string banjo
SynthPresets.sitar()           # Indian sitar
```

### 🎺 Winds (4)
```python
SynthPresets.flute()           # Flute
SynthPresets.clarinet()        # Clarinet
SynthPresets.trumpet()         # Trumpet
SynthPresets.saxophone()       # Saxophone
```

### 🎷 Brass (2)
```python
SynthPresets.brass()           # Brass section
SynthPresets.synth_brass()     # Synth brass
```

### 👥 Vocal (1)
```python
SynthPresets.choir()           # Choir pad
```

### 🔔 Mallet Percussion (8)
```python
SynthPresets.bell()            # Bell
SynthPresets.marimba()         # Marimba
SynthPresets.xylophone()       # Xylophone
SynthPresets.vibraphone()      # Vibraphone
SynthPresets.glockenspiel()    # Glockenspiel
SynthPresets.kalimba()         # Thumb piano
SynthPresets.music_box()       # Music box
SynthPresets.steel_drum()      # Steel pan
```

### 🥁 Drums (6)
```python
SynthPresets.kick_drum()       # Bass drum/kick
SynthPresets.snare_drum()      # Snare
SynthPresets.hi_hat()          # Hi-hat cymbal
SynthPresets.tom_drum()        # Tom drum
SynthPresets.cymbal()          # Crash cymbal
SynthPresets.drum()            # Generic drum
```

### 🎛️ Synthesizers (12)
```python
# Leads
SynthPresets.lead()            # Classic lead
SynthPresets.synth_lead()      # Bright lead

# Plucks
SynthPresets.pluck()           # Plucked synth
SynthPresets.synth_pluck()     # Synth pluck
SynthPresets.arp_synth()       # Arpeggiator synth

# Bass
SynthPresets.bass()            # Standard bass
SynthPresets.acid_bass()       # TB-303 style acid
SynthPresets.fat_bass()        # Thick detuned bass
SynthPresets.sub_bass()        # Sub bass

# Pads
SynthPresets.warm_pad()        # Warm pad
SynthPresets.ambient_pad()     # Ambient pad
SynthPresets.noise_sweep()     # Noise sweep FX
```

## Complete Example

```python
from algorythm.synth import SynthPresets
import numpy as np

# Create a multi-instrument composition
instruments = {
    'melody': SynthPresets.piano(),
    'bass': SynthPresets.sub_bass(),
    'pad': SynthPresets.strings(),
    'lead': SynthPresets.trumpet(),
    'perc': SynthPresets.marimba(),
    'drums': SynthPresets.kick_drum()
}

# Generate notes for each instrument
melody_notes = [261.63, 293.66, 329.63, 349.23]  # C, D, E, F
bass_note = 130.81  # C2

# Create audio for each part
melody_audio = np.concatenate([
    instruments['melody'].generate_note(freq, 0.5)
    for freq in melody_notes
])

bass_audio = instruments['bass'].generate_note(bass_note, 2.0)

# Mix together
# ... (add your mixing logic here)
```

## Synthesis Types Used

- **Synth** - Basic oscillator synthesis
- **FMSynth** - Frequency modulation (bells, brass, electric piano)
- **AdditiveeSynth** - Harmonic additive (organ, clarinet, piano)
- **PhysicalModelSynth** - Physical modeling (strings, drums, winds)
- **PadSynth** - Multi-voice detuned (strings, pads, choir)
- **WavetableSynth** - Wavetable morphing (ambient pad)

## Tips

1. **Layering**: Combine multiple instruments for richer sounds
   ```python
   piano = SynthPresets.piano()
   strings = SynthPresets.strings()
   # Play the same note with both
   ```

2. **Frequency Ranges**:
   - Sub bass: 20-60 Hz
   - Bass: 60-250 Hz
   - Mids: 250-2000 Hz
   - Highs: 2000-20000 Hz

3. **Note Durations**:
   - Drums/percussion: 0.05-0.5s
   - Plucks: 0.3-1.0s
   - Sustained: 1.0-4.0s
   - Pads: 2.0-8.0s

4. **Add Effects**: Chain effects for professional sound
   ```python
   from algorythm.effects import Reverb, Delay
   
   piano = SynthPresets.piano()
   audio = piano.generate_note(440.0, 2.0)
   
   reverb = Reverb(room_size=0.7, wet_level=0.3)
   audio = reverb.apply(audio)
   ```

## See Also
- **INSTRUMENTS_AND_EFFECTS.md** - Detailed guide
- **BEGINNER_GUIDE.md** - Getting started
- **examples/instrument_showcase.py** - Hear all instruments
