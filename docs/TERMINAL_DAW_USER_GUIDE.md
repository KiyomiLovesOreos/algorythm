# Terminal DAW User Guide

## Overview

The Algorythm Terminal DAW is a full-featured Digital Audio Workstation that runs entirely in your terminal. It provides multiple composition workflows including tracker-style editing, piano roll, arrangement, mixing, and live coding.

## Installation

The Terminal DAW requires the `textual` library:

```bash
pip install textual
```

## Launching

Start the Terminal DAW with:

```bash
# From command line
algorythm studio

# Or with a project file
algorythm studio myproject.agp

# Direct launch
python3 -m algorythm.terminal_daw
```

## Main Views

The Terminal DAW has 5 main views that you can switch between:

### 1. Tracker View (Press `1`)

Classic vertical composition view inspired by Renoise and ProTracker.

```
-- Track 01: Lead Synth | Pattern 01 | BPM: 120 --
┌─────┬──────┬─────┬─────┬────────┬────────┐
│ Row │ Note │ Ins │ Vol │ FX Cmd │ FX Val │
├─────┼──────┼─────┼─────┼────────┼────────┤
│>00  │ C-4  │ 01  │ 64  │        │        │
│ 01  │ ---  │ --  │ --  │        │        │
│ 02  │ E-4  │ 01  │ 64  │   F0   │   48   │
└─────┴──────┴─────┴─────┴────────┴────────┘
```

**Controls:**
- `↑/↓` - Move cursor up/down
- `Space` - Add note at cursor
- `Delete` - Remove note
- `A-G` - Enter note letters (planned)
- `0-9` - Edit values (planned)

**Features:**
- Per-step note entry
- Instrument selection per note
- Velocity control
- Effect commands (F0 = Filter, D0 = Delay, etc.)
- Effect parameter values

### 2. Piano Roll View (Press `2`)

Grid-based note editor similar to modern DAWs.

```
-- Pattern 01: Bassline | Track 01: Bass | BPM: 120 --
┌──────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ Step │ 1   │ 2   │ 3   │ 4   │ 5   │ 6   │ 7   │ 8   │
├──────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ G-4  │     │     │     │     │     │     │     │     │
│ E-4  │ [X] │ --- │ --- │ --- │ [X] │ --- │ --- │ --- │
│ D-4  │     │     │ [X] │ --- │     │     │ [X] │ --- │
│ C-4  │ [X] │ --- │ --- │ --- │ --- │ --- │ --- │ --- │
└──────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
```

**Controls:**
- `Arrow Keys` - Move cursor
- `Space` - Toggle note on/off
- `+/-` - Adjust velocity (planned)

**Features:**
- Visual grid representation
- Easy note placement
- Clear pattern visualization

### 3. Arranger View (Press `3`)

Arrange patterns into a full song structure.

```
-- Arranger | Song Length: 6 bars | BPM: 120 --
┌──────────────┬───────┬───────┬───────┬───────┬───────┬───────┐
│ Track        │ Bar 1 │ Bar 2 │ Bar 3 │ Bar 4 │ Bar 5 │ Bar 6 │
├──────────────┼───────┼───────┼───────┼───────┼───────┼───────┤
│ 01: Drums    │ P01   │ P01   │ P02   │ P02   │ P01   │ P01   │
│>02: Bass     │ ---   │ ---   │ P03   │ P03   │ P04   │ P04   │
│ 03: Lead     │ ---   │ ---   │ ---   │ ---   │ P05   │ P05   │
└──────────────┴───────┴───────┴───────┴───────┴───────┴───────┘
```

**Controls:**
- `Arrow Keys` - Navigate grid
- `Enter` - Select/edit pattern
- `S` - Solo track
- `M` - Mute track (planned)

**Features:**
- Pattern placement per track
- Visual song structure
- Track management

### 4. Mixer View (Press `4`)

Control track levels, panning, effects, and see real-time meters.

```
-- Mixer | CPU: 15% --
┌───┬────────────┬──────┬───────────────┬─────┬─────┬────────────────────┐
│ # │ Track Name │ Vol  │ Meter         │ Pan │ M/S │ FX                 │
├───┼────────────┼──────┼───────────────┼─────┼─────┼────────────────────┤
│ 1 │ Drums      │ -6.0 │ [|||||    ]   │ C   │ --- │ Compressor         │
│>2 │ Bass       │ -3.5 │ [|||||||  ]   │ C   │ S   │ EQ, Saturation     │
│ 3 │ Lead       │ -8.0 │ [|||      ]   │ L15 │ --- │ Delay, Reverb      │
└───┴────────────┴──────┴───────────────┴─────┴─────┴────────────────────┘
```

**Controls:**
- `↑/↓` - Select track
- `S` - Toggle solo
- `M` - Toggle mute
- `E` - Edit effects (planned)
- `+/-` - Adjust volume (planned)

**Features:**
- Volume control per track
- Pan control (L/C/R)
- Real-time level meters
- Mute/Solo functionality
- Effects chain display
- CPU usage monitoring

### 5. Live Coding View (Press `5`)

Python REPL with direct access to the project for algorithmic composition.

```
-- Live Coding --
Python REPL with access to 'project' object
────────────────────────────────────────────────────────────
>> from algorythm.generative import euclid
>> seq = euclid(hits=5, steps=16)
>> print(f'Generated {len(seq)} steps')
Generated 16 steps
>> _
────────────────────────────────────────────────────────────
```

**Available Objects:**
- `project` - Current project object
- All Algorythm modules (import as needed)

**Features:**
- Full Python REPL
- Access to generative algorithms
- Real-time pattern manipulation
- Algorithmic composition

## Global Keyboard Shortcuts

- **`1-5`** - Switch to specific view
- **`Tab`** - Next view
- **`Shift+Tab`** - Previous view
- **`Space`** - Toggle playback (Play/Stop)
- **`Ctrl+S`** - Save project
- **`Ctrl+P`** - Command palette (planned)
- **`Q`** - Quit application

## Project Files

Projects are saved in YAML format with `.agp` extension (Algorythm Project).

### File Structure

```yaml
name: My Project
bpm: 120
tracks:
  - name: Drums
    id: 1
    volume: -6.0
    pan: 0
    instrument: Drums
    effects:
      - Compressor
    patterns:
      P01:
        length: 16
        steps:
          - pitch: 60
            instrument: 1
            velocity: 80
            fx_cmd: null
            fx_val: null
          # ... more steps
```

### Saving and Loading

```bash
# Save from within app
Ctrl+S

# Load on startup
algorythm studio myproject.agp
```

## Advanced Features

### Effect Commands (Tracker View)

Effect commands are entered in the FX Cmd and FX Val columns:

- **F0** - Filter Cutoff (00-FF)
- **D0** - Delay Send (00-FF)
- **R0** - Reverb Send (00-FF)
- **70** - Probability (00-64 = 0-100%)
- **C0** - Note Cut (00-FF = timing)
- **P0** - Panning (00=Left, 40=Center, 7F=Right)

Example:
```
│ 04  │ G-4  │ 01  │ 60  │   F0   │   48   │  <- Filter cutoff to 48
│ 08  │ C-4  │ 01  │ 64  │   70   │   32   │  <- 50% probability
```

### Parameter Locks (P-Locks)

Set per-step parameter values that override the track's default settings. Great for dynamic filter sweeps, volume automation, etc.

### Conditional Triggers

Control when notes play based on loop count, probability, etc.

## Tips and Tricks

### Composition Workflow

1. **Start in Tracker** - Enter your basic pattern
2. **Move to Piano Roll** - Refine note placement
3. **Arrange** - Build song structure
4. **Mix** - Balance levels and add effects
5. **Live Code** - Add algorithmic variations

### Keyboard Efficiency

- Learn the number keys (1-5) for instant view switching
- Use Space for quick play/stop during composition
- Ctrl+S regularly to save your work

### Pattern-Based Workflow

1. Create short, reusable patterns
2. Arrange them in different combinations
3. Use variation patterns (P01, P01a, P01b)
4. Keep your patterns organized

### Live Coding Examples

```python
# Generate Euclidean rhythm
from algorythm.generative import euclid
kick_pattern = euclid(hits=4, steps=16)

# Create random melody
import random
from algorythm.sequence import Scale
scale = Scale.major('C', 4)
melody = [random.choice(scale.notes) for _ in range(16)]

# Apply automation
for i in range(16):
    project.tracks[0].patterns['P01'].steps[i].velocity = 64 + i * 2
```

## Troubleshooting

### Application won't start

```bash
# Install textual
pip install textual

# Verify installation
python3 -c "import textual; print(textual.__version__)"
```

### Display issues

- Ensure terminal is at least 80x24 characters
- Use a terminal with Unicode support
- Try different terminal emulators (iTerm2, Alacritty, etc.)

### Performance issues

- Reduce number of tracks
- Simplify effect chains
- Lower pattern resolution
- Check CPU usage in mixer view

## Future Features (Roadmap)

- [ ] Real-time audio playback
- [ ] MIDI input/output
- [ ] More effect commands
- [ ] Automation lanes
- [ ] Sample browser
- [ ] Plugin support
- [ ] Collaborative editing
- [ ] Cloud project sync

## See Also

- [TERMINAL_DAW_DESIGN_DOC.md](../docs/TERMINAL_DAW_DESIGN_DOC.md) - Technical design document
- [INSTRUMENTS_AND_EFFECTS.md](../INSTRUMENTS_AND_EFFECTS.md) - Available sounds and effects
- [BEGINNER_GUIDE.md](../BEGINNER_GUIDE.md) - Getting started with Algorythm

## Support

For issues, questions, or contributions:
- GitHub: https://github.com/yourusername/algorythm
- Documentation: See docs/ folder

---

**Happy composing! 🎵**
