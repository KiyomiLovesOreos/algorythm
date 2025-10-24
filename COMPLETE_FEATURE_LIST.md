# Algorythm - Complete Feature List

## 🎹 Synthesis Engines

### Basic Synth
- **Oscillator**: sine, square, saw, triangle, noise waveforms
- **Filter**: lowpass, highpass, bandpass, notch
- **ADSR Envelope**: attack, decay, sustain, release

### FM Synth (Frequency Modulation)
- Complex harmonic timbres
- Carrier and modulator oscillators
- Configurable modulation index and ratio
- Perfect for bells, brass, metallic sounds

### Wavetable Synth
- Morphing between multiple waveforms
- Real-time wavetable position control
- Automation support
- Smooth interpolation between tables

### Physical Model Synth ⭐ NEW
- **String Model**: Karplus-Strong algorithm for plucked strings
- **Drum Model**: Physical percussion simulation
- **Wind Model**: Flute and wind instrument simulation
- Configurable damping and brightness

### Additive Synth ⭐ NEW
- Multiple sine wave harmonics
- Configurable harmonic amplitudes
- Perfect for organs and rich timbres

### Pad Synth ⭐ NEW
- Multiple detuned oscillator voices
- Creates lush, thick pad sounds
- Configurable voice count (3-16 voices)
- Detune control for thickness

### Granular Synth
- Grain-based synthesis
- Configurable grain size and density
- Position, pitch, and spatial control
- Envelope shapes: rectangular, triangular, gaussian, hann

## 🎚️ Audio Effects

### Time-Based Effects
- **Reverb** ⭐: Room simulation with size and damping control
- **Delay** ⭐: Echo effect with feedback and timing control

### Modulation Effects
- **Chorus** ⭐: Thickening with modulated delays
- **Flanger** ⭐: Sweeping comb filter with feedback
- **Phaser** ⭐: Notch filter sweeps
- **Tremolo** ⭐: Amplitude modulation (sine, square, triangle)
- **Vibrato** ⭐: Pitch modulation
- **AutoPan** ⭐: Stereo panning modulation

### Distortion Effects
- **Distortion** ⭐: Waveshaping with tone control
- **Overdrive** ⭐: Smooth tube-like saturation
- **Fuzz** ⭐: Extreme clipping distortion

### Dynamics Effects
- **Compressor** ⭐: Dynamic range control with threshold, ratio, attack, release
- **Limiter** ⭐: Peak limiting for mastering
- **Gate** ⭐: Noise gate for removing low-level signals

### Special Effects
- **Ring Modulator** ⭐: Metallic, bell-like effects
- **Bit Crusher** ⭐: Lo-fi digital distortion with bit depth and sample rate reduction
- **Effect Chain** ⭐: Combine multiple effects in series

⭐ = New in this update

## 🎼 Presets

### Basic Presets
- `SynthPresets.warm_pad()` - Warm pad sound
- `SynthPresets.pluck()` - Plucked string
- `SynthPresets.bass()` - Deep bass
- `SynthPresets.lead()` ⭐ - Bright lead synth

### Advanced Presets
- `SynthPresets.organ()` ⭐ - Additive organ
- `SynthPresets.bell()` - FM bell
- `SynthPresets.strings()` ⭐ - Lush string section
- `SynthPresets.guitar()` ⭐ - Physical model guitar
- `SynthPresets.drum()` ⭐ - Physical model drum
- `SynthPresets.brass()` ⭐ - FM brass sound

## 🎵 Musical Structure

### Sequences
- **Motif**: Musical phrases with notes and durations
- **Rhythm**: Rhythmic patterns
- **Arpeggiator**: Generate arpeggios from chords
- **Scale**: Major, minor, pentatonic, blues, and more
- **Chord**: Major, minor, diminished, augmented, 7th chords
- **Tuning**: Standard and alternative tuning systems

### Composition
- **Track**: Individual instrument tracks
- **Composition**: Multi-track compositions
- Tempo and time signature control
- Volume and panning per track

## 🤖 Generative Tools

- **L-System**: Lindenmayer systems for fractal melodies
- **Cellular Automata**: Conway's Game of Life for music
- **Constraint-Based Composer**: Rule-based composition
- **Genetic Algorithm Improviser**: Evolutionary music generation

## 📊 Automation & Data

- **Automation**: Parameter automation with curves (linear, exponential, bezier, ease in/out)
- **AutomationTrack**: Multi-segment automation
- **Data Sonification**: Convert data to music (CSV, arrays)

## 🎨 Visualization

- **Waveform Visualizer**: Audio waveform plots
- **Spectrogram Visualizer**: Frequency over time
- **Frequency Scope**: Real-time frequency analysis
- **Oscilloscope Visualizer**: XY scope visualization
- **Piano Roll Visualizer**: MIDI-style piano roll
- **Video Renderer**: Export visualizations to video

## 🔊 Audio I/O

### Loading
- **Sample**: Load and manipulate audio files (WAV)
- **Sampler**: Trigger samples with pitch shifting
- **AudioFile**: Load audio with metadata

### Export
- **RenderEngine**: Render compositions to audio
- **Exporter**: Export to WAV, MP3, FLAC formats

### Playback (optional)
- **AudioPlayer**: Real-time playback
- **StreamingPlayer**: Streaming audio player
- **LiveCompositionPlayer**: Real-time composition preview

### Live Coding (optional)
- **LiveCodingGUI**: Interactive live coding interface
- Real-time code editing and playback

## 📝 Example Usage

```python
from algorythm import *
from algorythm.effects import *

# Create instruments
guitar = SynthPresets.guitar()
strings = SynthPresets.strings()
pad = PadSynth(num_voices=7)

# Create scale and motifs
scale = Scale.major('C', 4)
melody = Motif(notes=scale.ascending(8), durations=[0.5] * 8)

# Create composition
comp = Composition(tempo=120)

# Add tracks with effects
track1 = Track("Guitar")
track1.add_motif(melody, instrument=guitar)
track1.add_effect(ReverbFX(room_size=0.6, wet_level=0.3))
track1.add_effect(DelayFX(delay_time=0.375, feedback=0.4))

track2 = Track("Strings")
track2.add_motif(melody, instrument=strings)
track2.add_effect(ChorusFX(mix=0.4))

# Create effect chain
chain = FXChain()
chain.add_effect(DistortionFX(drive=3.0))
chain.add_effect(ReverbFX(room_size=0.8))

comp.add_track(track1)
comp.add_track(track2)

# Render and export
engine = RenderEngine()
audio = engine.render(comp)

exporter = Exporter()
exporter.export_wav(audio, "output.wav")
```

## 📚 Documentation

- **BEGINNER_GUIDE.md** - Getting started guide
- **CHEAT_SHEET.md** - Quick reference
- **INSTRUMENTS_AND_EFFECTS.md** - Complete guide to instruments and effects
- **CLI_GUIDE.md** - Command-line interface guide
- **PERFORMANCE_TIPS.md** - Optimization tips

## 🧪 Testing

- Full test suite with pytest
- 162+ total tests
- All new features tested (22 additional tests)
- Continuous integration ready

## 📦 Installation

```bash
pip install algorythm
```

Optional dependencies:
```bash
pip install algorythm[playback]  # For audio playback
pip install algorythm[gui]       # For live coding GUI
pip install algorythm[all]       # Install everything
```

---

Total Features: 50+ instruments, effects, and tools for algorithmic music composition!
