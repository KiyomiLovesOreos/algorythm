# Algorythm Examples

This directory contains example scripts demonstrating the Algorythm library's features.

## Running Examples

You can run examples in two ways:

### 1. Direct Python Execution

```bash
python basic_synthesis.py
python composition_example.py
python advanced_example.py
```

### 2. Via CLI

```bash
# From the repository root
python -m algorythm.cli --example basic
python -m algorythm.cli --example composition
python -m algorythm.cli --example advanced
```

Or if installed:
```bash
algorythm --example basic
algorythm --example composition
algorythm --example advanced
```

## Example Descriptions

### basic_synthesis.py

Demonstrates the core synthesis API:
- Creating a custom synth with waveform, filter, and envelope
- Generating a single note
- Exporting to WAV file

Key concepts:
- `Synth` class
- `Filter.lowpass()` for filtering
- `ADSR` envelope
- `Exporter` for audio output

### composition_example.py

Shows the complete composition workflow as described in the problem statement:
- Creating custom instruments
- Defining melodic motifs with scales
- Building compositions with method chaining
- Transposing music
- Applying effects (reverb)
- Multi-format export

Key concepts:
- `Motif.from_intervals()`
- `Scale.major()` for musical scales
- `Composition` class with fluent API
- Method chaining for composition building
- `Reverb` effect

### advanced_example.py

Demonstrates advanced features:
- Multiple tracks with different instruments
- Synth presets (bass, pluck, warm pad)
- Arpeggiators
- Multiple effects per track
- Complex arrangements

Key concepts:
- `SynthPresets` for quick instrument creation
- `Arpeggiator` for generating arpeggios
- Multiple effects: `Delay` and `Reverb`
- Multi-track composition

### volume_control_demo.py

Demonstrates comprehensive volume control features:
- Track-level volume control
- Master volume for entire composition
- Fade in/out effects
- VolumeControl utility functions
- dB to linear conversions
- Volume normalization
- Different fade curve types (linear, exponential, logarithmic)

Key concepts:
- `set_track_volume()` for individual track control
- `set_master_volume()` for composition-wide control
- `VolumeControl` utility class
- `fade_in()` and `fade_out()` methods
- Volume conversion between dB and linear

## Generated Files

Running these examples will create WAV files:
- `warm_pad.wav` - Single warm pad note
- `epic_track.wav` - Full composition with melody
- `advanced_track.wav` - Multi-track composition
- `volume_demo.wav` - Multi-track composition with volume control
- `fade_linear_demo.wav` - Linear fade curve demonstration
- `fade_exponential_demo.wav` - Exponential fade curve demonstration
- `fade_logarithmic_demo.wav` - Logarithmic fade curve demonstration

**Note:** Generated audio files are excluded from git (see `.gitignore`).

## Next Steps

After running these examples:

1. **Modify Parameters**: Try changing synth parameters, scales, or tempos
2. **Create Your Own**: Use these as templates for your own compositions
3. **Explore the API**: Check the main README for full API documentation
4. **Add Effects**: Experiment with different effect combinations

## Tips

- Start with `basic_synthesis.py` to understand the fundamentals
- Use `composition_example.py` as a template for your projects
- Reference `advanced_example.py` for complex multi-track work
- Adjust tempo, scales, and intervals to create different moods
- Layer multiple tracks for richer compositions
