# API Reference

Complete API documentation for Algorythm.

## Core Modules

### algorythm.synth

Sound synthesis components.

#### Synth

Main synthesizer class.

```python
Synth(
    waveform='sine',        # Waveform type
    filter=None,            # Optional Filter object
    envelope=None,          # Optional ADSR envelope
    amplitude=1.0,          # Output amplitude (0-1)
    sample_rate=44100       # Sample rate in Hz
)
```

**Methods:**
- `generate_note(frequency, duration)` - Generate a single note
- `generate(duration)` - Generate audio without specific pitch

#### Oscillator

Waveform generator.

```python
Oscillator(
    waveform='sine',        # 'sine', 'square', 'saw', 'triangle', 'noise'
    frequency=440.0,        # Frequency in Hz
    amplitude=1.0,          # Amplitude (0-1)
    phase=0.0               # Initial phase in radians
)
```

**Methods:**
- `generate(duration, sample_rate=44100)` - Generate waveform

#### Filter

Frequency filter.

```python
Filter(
    filter_type,            # 'lowpass', 'highpass', 'bandpass', 'notch'
    cutoff,                 # Cutoff frequency in Hz
    resonance=0.5           # Resonance (0-1)
)
```

**Class methods:**
- `Filter.lowpass(cutoff, resonance=0.5)`
- `Filter.highpass(cutoff, resonance=0.5)`
- `Filter.bandpass(center, resonance=0.5)`
- `Filter.notch(center, resonance=0.5)`

**Methods:**
- `apply(signal, sample_rate=44100)` - Apply filter to signal

#### ADSR

Envelope generator.

```python
ADSR(
    attack=0.1,             # Attack time in seconds
    decay=0.1,              # Decay time in seconds
    sustain=0.7,            # Sustain level (0-1)
    release=0.3             # Release time in seconds
)
```

**Methods:**
- `apply(signal, sample_rate=44100)` - Apply envelope to signal
- `generate(duration, sample_rate=44100)` - Generate envelope curve

#### FMSynth

FM synthesis engine.

```python
FMSynth(
    carrier_freq=440,       # Carrier frequency in Hz
    modulator_freq=880,     # Modulator frequency in Hz
    mod_index=5,            # Modulation index
    sample_rate=44100
)
```

**Methods:**
- `generate(duration)` - Generate FM audio
- `generate_note(frequency, duration)` - Generate FM note

#### WavetableSynth

Wavetable synthesis.

```python
WavetableSynth(
    wavetable,              # NumPy array of wavetable
    sample_rate=44100
)
```

**Methods:**
- `generate_note(frequency, duration)` - Generate note from wavetable

#### PhysicalModelSynth

Physical modeling synthesis.

```python
PhysicalModelSynth(
    brightness=0.5,         # Brightness (0-1)
    damping=0.995,          # Damping factor (0-1)
    sample_rate=44100
)
```

**Methods:**
- `generate_note(frequency, duration)` - Generate plucked string sound

#### AdditiveeSynth

Additive synthesis.

```python
AdditiveeSynth(
    num_harmonics=8,        # Number of harmonics
    harmonic_decay=0.5,     # Harmonic amplitude decay
    sample_rate=44100
)
```

**Methods:**
- `generate_note(frequency, duration)` - Generate additive synthesis note

#### SynthPresets

Pre-built instrument presets. All methods are class methods returning Synth objects.

**Synth presets:**
- `synth_lead()`, `synth_bass()`, `synth_pad()`, `warm_pad()`, `soft_pad()`, `bright_lead()`
- `acid_bass()`, `arp_synth()`, `ambient_pad()`

**Plucked:**
- `pluck()`, `guitar()`, `harp()`, `banjo()`

**Keys:**
- `piano()`, `electric_piano()`, `bell()`, `glockenspiel()`

**Brass:**
- `brass()`, `trumpet()`, `trombone()`, `french_horn()`

**Strings:**
- `strings()`, `violin()`, `cello()`, `pizzicato()`, `choir()`

**Woodwinds:**
- `flute()`, `clarinet()`, `oboe()`, `bassoon()`

**Bass:**
- `bass()`, `electric_bass()`, `sub_bass()`, `upright_bass()`

**Drums:**
- `kick()`, `snare()`, `hihat()`, `clap()`, `tom()`, `cymbal()`

**Effects:**
- `laser()`, `explosion()`, `woosh()`

---

### algorythm.sequence

Musical pattern creation.

#### Scale

Musical scales.

```python
Scale(
    root='C',               # Root note
    intervals=[...],        # Scale intervals in semitones
    tuning=None             # Optional Tuning object
)
```

**Class methods:**
- `Scale.major(root)` - Major scale
- `Scale.minor(root)` - Natural minor
- `Scale.harmonic_minor(root)`
- `Scale.melodic_minor(root)`
- `Scale.dorian(root)`, `Scale.phrygian(root)`, `Scale.lydian(root)`, etc.
- `Scale.pentatonic(root)` - Pentatonic scale
- `Scale.blues(root)` - Blues scale
- `Scale.chromatic(root)` - Chromatic scale

**Methods:**
- `get_frequency(degree)` - Get frequency for scale degree
- `get_note_name(degree)` - Get note name for degree

#### Motif

Melodic pattern.

```python
Motif(
    notes=[...],            # List of note dicts
    scale=None              # Optional scale
)
```

**Class methods:**
- `Motif.from_intervals(intervals, scale, duration=0.5, octave=4)`
- `Motif.from_frequencies(frequencies, durations)`
- `Motif.from_notes(notes, duration=0.5)` - Notes like 'C4', 'D#5'

**Methods:**
- `transpose(degrees)` - Transpose by scale degrees
- `retrograde()` - Reverse the motif
- `invert()` - Invert intervals
- `augment(factor)` - Change duration
- `repeat(times)` - Repeat motif

**Properties:**
- `notes` - List of note dictionaries with 'frequency' and 'duration'

#### Rhythm

Rhythmic pattern.

```python
Rhythm(
    pattern=[...]           # List of durations or hits
)
```

**Class methods:**
- `Rhythm.from_durations(durations)` - From duration list
- `Rhythm.from_pattern(pattern)` - From 'x..x.x..' string
- `Rhythm.euclidean(hits, length)` - Euclidean rhythm

**Methods:**
- `repeat(times)` - Repeat rhythm
- `reverse()` - Reverse rhythm

**Properties:**
- `pattern` - Rhythm pattern
- `duration` - Total duration

#### Chord

Musical chords.

```python
Chord(
    root='C',               # Root note
    intervals=[...]         # Chord intervals
)
```

**Class methods:**
- `Chord.major(root)`, `Chord.minor(root)`, `Chord.diminished(root)`, `Chord.augmented(root)`
- `Chord.major7(root)`, `Chord.minor7(root)`, `Chord.dominant7(root)`, `Chord.diminished7(root)`
- `Chord.from_scale_degrees(degrees, scale)`
- `Chord.from_intervals(intervals)`

**Methods:**
- `get_frequencies()` - Get all chord note frequencies
- `arpeggiate(duration=0.25)` - Convert to arpeggio motif

#### Arpeggiator

Arpeggio generator.

```python
Arpeggiator(
    chord                   # Chord object
)
```

**Methods:**
- `generate(pattern='up', duration=0.25, num_repetitions=1, octaves=1)`
  - Patterns: 'up', 'down', 'up_down', 'down_up', 'random', or list of indices

#### Tuning

Alternative tuning systems.

```python
Tuning(
    tuning_system='12-TET',     # Tuning name or cents list
    reference_frequency=440.0,   # Reference freq
    reference_note=69            # Reference MIDI note
)
```

**Class methods:**
- `Tuning.equal_temperament(divisions)` - N-tone equal temperament
- `Tuning.just_intonation()` - Just intonation
- `Tuning.pythagorean()` - Pythagorean tuning

**Pre-defined tunings:**
- '12-TET', '19-TET', '24-TET', 'just_intonation', 'pythagorean'

**Methods:**
- `get_frequency(degree)` - Get frequency for scale degree

---

### algorythm.structure

Composition and arrangement.

#### Composition

Multi-track composition.

```python
Composition(
    tempo=120,                  # BPM
    time_signature=(4, 4),      # Time signature
    sample_rate=44100
)
```

**Methods:**
- `add_track(name, instrument)` - Add a track
- `get_track(name)` - Get track by name
- `play_note(frequency, duration, start, track)` - Play note on track
- `play_motif(motif, start, track)` - Play motif on track
- `add_master_effect(effect)` - Add effect to master
- `render(filename)` - Render to audio file

#### Track

Individual track in composition.

```python
Track(
    name,                   # Track name
    instrument,             # Synth object
    sample_rate=44100
)
```

**Methods:**
- `set_volume(level)` - Set volume (0-1)
- `set_pan(position)` - Set pan (-1 to 1)
- `mute()`, `unmute()` - Mute/unmute track
- `solo()` - Solo this track
- `add_effect(effect)` - Add effect
- `add_effect_chain(chain)` - Add effect chain
- `automate_parameter(param, automation, start_time)` - Automate parameter

#### Effect Classes (in structure module)

Wrapper classes for track effects:
- `Reverb(mix, room_size, damping)`
- `Delay(delay_time, feedback, mix)`
- `Chorus(rate, depth, mix)`
- `Flanger(rate, depth, feedback, mix)`
- `Phaser(rate, depth, feedback, mix)`
- `Distortion(drive, tone, mix)`
- `Compression(threshold, ratio, attack, release)`
- `EQ(low_gain, mid_gain, high_gain, low_freq, mid_freq, high_freq)`
- `Tremolo(rate, depth, mix)`
- `Bitcrusher(bit_depth, sample_rate, mix)`

All have `apply(signal, sample_rate)` method.

---

### algorythm.effects

Standalone effect processors.

All effects have similar constructor pattern and `apply(signal, sample_rate)` method.

**Time-based:**
- `ReverbFX(mix, room_size, damping)`
- `DelayFX(delay_time, feedback, mix)`
- `ChorusFX(rate, depth, mix)`
- `FlangerFX(rate, depth, feedback, mix)`
- `PhaserFX(rate, depth, feedback, mix)`

**Dynamics:**
- `Compressor(threshold, ratio, attack, release, makeup_gain)`
- `Limiter(threshold, release)`
- `Gate(threshold, attack, release)`

**Distortion:**
- `DistortionFX(drive, tone, mix)`
- `Overdrive(drive, tone, mix)`
- `Fuzz(drive, tone, mix)`
- `BitCrusherFX(bit_depth, sample_rate, mix)`

**Modulation:**
- `TremoloFX(rate, depth, mix)`
- `Vibrato(rate, depth, mix)`
- `AutoPan(rate, depth, waveform)`
- `RingModulator(frequency, mix)`

**Creative:**
- `Stutter(slice_length, repetitions, mix)`
- `BeatRepeat(loop_length, probability, mix)`
- `Freeze(freeze_length, mix)`
- `Reverse(mix)`
- `FilterSweep(filter_type, start_freq, end_freq, duration, resonance)`

**Chain:**
- `FXChain()` - Chain multiple effects
  - `add(effect)` - Add effect to chain
  - `apply(signal, sample_rate)` - Process signal

---

### algorythm.generative

Algorithmic composition tools.

#### LSystem

L-System pattern generator.

```python
LSystem(
    axiom,                  # Starting string
    rules={},               # Rewriting rules dict
    iterations=3            # Number of iterations
)
```

**Class methods:**
- `LSystem.fractal_melody()`
- `LSystem.growing_pattern()`
- `LSystem.branching()`

**Methods:**
- `generate()` - Generate L-system string

#### CellularAutomata

Cellular automaton pattern generator.

```python
CellularAutomata(
    rule=30,                # Rule number (0-255)
    width=16,               # Pattern width
    generations=8           # Number of generations
)
```

**Methods:**
- `generate()` - Generate 2D pattern array

#### ConstraintBasedComposer

Constrained melody generator.

```python
ConstraintBasedComposer(
    scale,                  # Scale object
    constraints={}          # Constraint dict
)
```

**Constraints:**
- `min_interval`, `max_interval` - Interval limits
- `prefer_steps` - Prefer stepwise motion
- `avoid_leaps` - Avoid large jumps
- `contour` - 'arch', 'ascending', 'descending'
- `chord_tones` - Preferred scale degrees
- `chord_weight` - Probability of chord tones
- `forbidden_intervals` - Intervals to avoid
- `preferred_intervals` - Preferred intervals

**Methods:**
- `compose(length, start_degree=0)` - Generate melody

#### GeneticAlgorithmImproviser

Evolutionary melody generator.

```python
GeneticAlgorithmImproviser(
    scale,                      # Scale object
    population_size=20,         # Population size
    mutation_rate=0.1,          # Mutation probability
    generations=10              # Number of generations
)
```

**Methods:**
- `evolve(length, fitness_function)` - Evolve melody
  - `fitness_function(melody)` should return numeric score

---

### algorythm.automation

Parameter automation.

#### Automation

Single automation curve.

```python
Automation(
    start_value,            # Starting value
    end_value,              # Ending value
    duration,               # Duration in seconds
    curve='linear'          # 'linear', 'exponential', 'logarithmic'
)
```

**Methods:**
- `get_value(time)` - Get value at time

#### AutomationTrack

Multiple automation curves.

```python
AutomationTrack()
```

**Methods:**
- `add(automation, start=0.0)` - Add automation curve
- `get_value(time)` - Get value at time

#### DataSonification

Data to sound mapping.

```python
DataSonification(
    scale,                  # Scale object
    min_value,              # Min data value
    max_value,              # Max data value
    min_degree=-7,          # Min scale degree
    max_degree=7            # Max scale degree
)
```

**Methods:**
- `map_to_scale(data)` - Map data to scale degrees

---

### algorythm.export

Audio export utilities.

#### Exporter

Export audio to files.

```python
Exporter(
    sample_rate=44100
)
```

**Methods:**
- `export(audio_data, filename, format=None, bitrate='192k', normalize=False)`
  - Supports WAV, MP3, FLAC
  - Auto-detects format from extension

#### RenderEngine

Advanced rendering engine.

```python
RenderEngine(
    sample_rate=44100
)
```

**Methods:**
- `render(composition)` - Render composition to audio array

---

### algorythm.visualization

Audio visualization.

#### visualize_audio_file()

Main visualization function.

```python
visualize_audio_file(
    audio_path,             # Input audio file
    output_path,            # Output video file
    visualizer,             # Visualizer object
    video_width=1920,       # Video width
    video_height=1080,      # Video height
    video_fps=30,           # Frame rate
    video_bitrate='5000k'   # Video quality
)
```

#### Visualizers

All visualizers have `render_frame(audio_chunk, width, height)` method.

- `WaveformVisualizer(sample_rate, color, line_width, background)`
- `FrequencyScopeVisualizer(sample_rate, num_bars, color, background)`
- `SpectrogramVisualizer(sample_rate, fft_size, hop_length, colormap)`
- `OscilloscopeVisualizer(sample_rate, color, line_width, background)`
- `PianoRollVisualizer(sample_rate, note_color, background)`
- `CircularVisualizer(sample_rate, num_bars, radius, color, background)`

#### VideoRenderer

Custom video rendering.

```python
VideoRenderer(
    audio_path,
    visualizer,
    output_path,
    video_width=1920,
    video_height=1080,
    video_fps=30
)
```

**Methods:**
- `render()` - Render video

---

### algorythm.sampler

Sample-based synthesis.

#### Sample

Audio sample.

```python
Sample(
    filepath                # Path to audio file
)
```

**Methods:**
- `get_audio_data()` - Get sample audio
- `get_duration()` - Get duration

#### Sampler

Sample playback.

```python
Sampler(
    sample,                 # Sample object
    pitch=1.0,              # Pitch multiplier
    loop=False              # Enable looping
)
```

**Methods:**
- `generate(duration)` - Generate audio

#### GranularSynth

Granular synthesis.

```python
GranularSynth(
    sample,                 # Sample object
    grain_size=0.1,         # Grain duration
    density=20,             # Grains per second
    pitch=1.0,              # Pitch shift
    spray=0.1               # Position randomness
)
```

**Methods:**
- `generate(duration)` - Generate granular audio

---

### algorythm.audio_loader

Audio file utilities.

#### load_audio()

Load audio file.

```python
load_audio(filepath)        # Returns AudioFile object
```

#### AudioFile

Audio file wrapper.

```python
AudioFile(
    audio_data,             # NumPy array
    sample_rate
)
```

**Properties:**
- `duration` - Duration in seconds
- `sample_rate` - Sample rate
- `channels` - Number of channels

**Methods:**
- `get_audio_data()` - Get audio array
- `save(filepath)` - Save to file

---

## Common Patterns

### Create and render a song

```python
from algorythm import Composition, SynthPresets, Scale, Motif

song = Composition(tempo=120)
song.add_track('melody', SynthPresets.pluck())
melody = Motif.from_intervals([0, 2, 4, 5], scale=Scale.major('C'))
song.play_motif(melody, start=0.0, track='melody')
song.render('output.wav')
```

### Apply effects

```python
from algorythm import ReverbFX, DelayFX

track.add_effect(ReverbFX(mix=0.3))
track.add_effect(DelayFX(delay_time=0.5, feedback=0.3))
```

### Create visualization

```python
from algorythm import visualize_audio_file, FrequencyScopeVisualizer

viz = FrequencyScopeVisualizer(sample_rate=44100)
visualize_audio_file('input.wav', 'output.mp4', viz)
```
