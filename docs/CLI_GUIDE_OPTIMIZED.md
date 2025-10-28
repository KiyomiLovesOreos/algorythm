# Algorythm CLI Guide

## Quick Start

Algorythm provides two CLI interfaces:

1. **algorythm-quick** - Simple, optimized commands for fast music generation
2. **algorythm** - Full-featured CLI with visualization and advanced options

## Quick CLI (Recommended for Fast Workflows)

### Installation

```bash
pip install algorythm
```

### Generate a Melody

```bash
# Basic melody
python3 -m algorythm.cli_quick melody

# Custom melody
python3 -m algorythm.cli_quick melody -i bell -t 140 -b 4 -o my_melody.wav
```

**Options:**
- `-i, --instrument` - Instrument preset (pluck, bass, lead, bell, pad, organ, strings, guitar, brass)
- `-s, --scale` - Musical scale (e.g., C:major, A:minor, D:pentatonic)
- `-t, --tempo` - Tempo in BPM (default: 120)
- `-b, --bars` - Number of bars (default: 4)
- `-o, --output` - Output filename (default: melody.wav)

### Generate a Beat

```bash
# Basic beat
python3 -m algorythm.cli_quick beat

# Custom beat
python3 -m algorythm.cli_quick beat -t 128 -b 8 -o my_beat.wav
```

**Options:**
- `-t, --tempo` - Tempo in BPM (default: 120)
- `-b, --bars` - Number of bars (default: 4)
- `-o, --output` - Output filename (default: beat.wav)

### Generate by Style

```bash
# Generate ambient track
python3 -m algorythm.cli_quick style ambient -l 60 -o ambient.wav

# Generate chill track
python3 -m algorythm.cli_quick style chill -l 30

# Generate upbeat track
python3 -m algorythm.cli_quick style upbeat -l 45

# Generate minimal track
python3 -m algorythm.cli_quick style minimal -l 20
```

**Styles:**
- `ambient` - Slow pads with heavy reverb (60 BPM)
- `chill` - Relaxed guitar melodies (90 BPM)
- `upbeat` - Fast, energetic leads (140 BPM)
- `minimal` - Simple, sparse composition (100 BPM)

**Options:**
- `-l, --length` - Track length in seconds (default: 30)
- `-o, --output` - Output filename (default: track.wav)

### Apply Effects

```bash
# Add reverb
python3 -m algorythm.cli_quick fx reverb input.wav

# Add delay with custom output
python3 -m algorythm.cli_quick fx delay input.wav -o output_delayed.wav

# Add distortion with parameters
python3 -m algorythm.cli_quick fx distortion input.wav --param drive=8.0 --param tone=0.7
```

**Available Effects:**
- `reverb` - Adds space and depth
- `delay` - Echo effect
- `chorus` - Thickening effect
- `distortion` - Overdrive/distortion
- `compressor` - Dynamic range control

**Options:**
- `-o, --output` - Output filename (auto-generated if not specified)
- `--param key=value` - Set effect parameter (can be used multiple times)

### List Presets and Effects

```bash
# List all
python3 -m algorythm.cli_quick list

# List only presets
python3 -m algorythm.cli_quick list presets

# List only effects
python3 -m algorythm.cli_quick list effects
```

## Full CLI

The full CLI provides additional features for visualization and advanced control.

### Show Version

```bash
python3 -m algorythm.cli --version
```

### List Supported Formats

```bash
python3 -m algorythm.cli formats
```

### Show Audio File Info

```bash
python3 -m algorythm.cli info song.mp3
```

### Create Visualization Video

```bash
# Basic visualization
python3 -m algorythm.cli visualize song.mp3

# Custom visualization
python3 -m algorythm.cli visualize song.mp3 \
  -v circular \
  --bars 128 \
  --color purple \
  --background dark \
  -o output_video.mp4
```

**Visualizer Types:**
- `waveform` - Audio waveform over time
- `circular` - Circular frequency bars (default)
- `spectrum` - Frequency spectrum scope
- `spectrogram` - Time-frequency heatmap
- `oscilloscope` - Real-time oscilloscope display

**Options:**
- `-v, --visualizer` - Visualization type
- `-w, --width` - Video width (default: 1920)
- `--height` - Video height (default: 1080)
- `--fps` - Video FPS (default: 30)
- `--bars` - Number of frequency bars (default: 64)
- `--color` - Color scheme (blue, red, green, purple, orange, cyan, magenta)
- `--background` - Background color (black, white, dark, light)
- `--offset` - Start offset in seconds
- `--duration` - Duration to process in seconds

### Run Examples

```bash
# Basic example
python3 -m algorythm.cli --example basic

# Composition example
python3 -m algorythm.cli --example composition

# Advanced example
python3 -m algorythm.cli --example advanced
```

## Python API for Scripts

You can also use the CLI tools directly in Python:

```python
from algorythm.cli_tools import (
    quick_melody,
    quick_beat,
    apply_effect_to_file,
    generate_style,
    list_all_presets,
    list_all_effects
)

# Generate melody
duration = quick_melody(
    instrument='bell',
    scale='C:major',
    tempo=140,
    bars=4,
    output='my_song.wav'
)

# Generate beat
duration = quick_beat(tempo=128, bars=8, output='beat.wav')

# Generate by style
duration = generate_style(style='ambient', length=60, output='ambient.wav')

# Apply effect
output = apply_effect_to_file(
    'input.wav',
    'reverb',
    output_file='output.wav',
    room_size=0.8,
    wet_level=0.5
)

# List presets
presets = list_all_presets()
print(presets)

# List effects
effects = list_all_effects()
print(effects)
```

## Tips for CLI Usage

### 1. Batch Processing

Generate multiple tracks:

```bash
for tempo in 120 130 140; do
    python3 -m algorythm.cli_quick melody -i lead -t $tempo -o "melody_${tempo}bpm.wav"
done
```

### 2. Style Mixing

Generate different styles and mix them:

```bash
python3 -m algorythm.cli_quick style ambient -l 30 -o ambient.wav
python3 -m algorythm.cli_quick style upbeat -l 30 -o upbeat.wav
# Mix with audio editor
```

### 3. Effect Chains

Apply multiple effects sequentially:

```bash
python3 -m algorythm.cli_quick fx reverb input.wav -o temp.wav
python3 -m algorythm.cli_quick fx delay temp.wav -o output.wav
rm temp.wav
```

### 4. Quick Sketches

Generate ideas quickly:

```bash
# Generate 5 random melodies
for i in {1..5}; do
    python3 -m algorythm.cli_quick melody -i bell -b 2 -o "idea_${i}.wav"
done
```

### 5. Performance Mode

For long tracks, use style generation:

```bash
# Generate 5-minute ambient track
python3 -m algorythm.cli_quick style ambient -l 300 -o long_ambient.wav
```

## Available Instruments

### Basic Presets
- `pluck` - Plucked string sound
- `bass` - Deep bass synth
- `lead` - Bright lead synth
- `pad` - Warm pad sound

### Advanced Presets
- `bell` - FM bell sound
- `organ` - Additive organ
- `strings` - Lush string section
- `guitar` - Physical model guitar
- `brass` - FM brass sound
- `drum` - Physical model drum

## Available Scales

- `C:major`, `G:major`, `D:major`, etc.
- `A:minor`, `E:minor`, `D:minor`, etc.
- `C:pentatonic`, `G:pentatonic`, etc.

Note: Replace C with any note (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)

## Common Workflows

### 1. Create a Song Sketch

```bash
# Generate bassline
python3 -m algorythm.cli_quick melody -i bass -s C:minor -t 110 -b 8 -o bass.wav

# Generate melody
python3 -m algorythm.cli_quick melody -i lead -s C:minor -t 110 -b 8 -o lead.wav

# Generate beat
python3 -m algorythm.cli_quick beat -t 110 -b 8 -o drums.wav

# Add effects
python3 -m algorythm.cli_quick fx reverb bass.wav -o bass_fx.wav
python3 -m algorythm.cli_quick fx delay lead.wav -o lead_fx.wav
```

### 2. Generate Background Music

```bash
# Ambient background
python3 -m algorythm.cli_quick style ambient -l 180 -o background.wav

# Add subtle effects
python3 -m algorythm.cli_quick fx reverb background.wav
```

### 3. Sound Design

```bash
# Generate base sound
python3 -m algorythm.cli_quick melody -i strings -b 1 -o base.wav

# Apply creative effects
python3 -m algorythm.cli_quick fx distortion base.wav --param drive=10.0
```

## Troubleshooting

### Command Not Found

If `algorythm-quick` is not found:
```bash
# Use Python module syntax instead
python3 -m algorythm.cli_quick [command]
```

### Output File Already Exists

The CLI will overwrite existing files. Back up important files first.

### Audio Not Playing

The CLI generates WAV files. Ensure you have a media player installed:
- Linux: `aplay output.wav` or `vlc output.wav`
- macOS: `afplay output.wav` or `open output.wav`
- Windows: `start output.wav`

### Memory Issues with Long Tracks

For tracks longer than 5 minutes, generation may be slow. Consider:
- Generating shorter loops and extending in an audio editor
- Using lower sample rates (not yet supported in CLI)
- Reducing the number of effects

## See Also

- [INSTRUMENTS_AND_EFFECTS.md](INSTRUMENTS_AND_EFFECTS.md) - Complete guide to instruments and effects
- [BEGINNER_GUIDE.md](BEGINNER_GUIDE.md) - Getting started with Python API
- [CHEAT_SHEET.md](CHEAT_SHEET.md) - Quick reference for Python API
- [examples/](examples/) - Python code examples

## Contributing

Found a bug or have a feature request? Open an issue on GitHub!
