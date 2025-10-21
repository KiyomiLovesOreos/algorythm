# Volume Control Quick Reference

## Import

```python
from algorythm import Composition, VolumeControl
from algorythm.playback import AudioPlayer
```

## Composition Volume Control

```python
comp = Composition(tempo=120)

# Track volume (0.0 to 1.0+)
comp.set_track_volume('TrackName', 0.8)

# Master volume (0.0 to 1.0+)
comp.set_master_volume(0.9)

# Fades (in seconds)
comp.fade_in(1.0).fade_out(2.0)
```

## VolumeControl Utilities

```python
# Convert dB to linear
linear = VolumeControl.db_to_linear(-6.0)  # 0.5012

# Convert linear to dB
db = VolumeControl.linear_to_db(0.5)  # -6.02

# Apply volume to signal
quiet = VolumeControl.apply_volume(signal, 0.5)
quiet = VolumeControl.apply_db_volume(signal, -6.0)

# Normalize to target dB
normalized = VolumeControl.normalize(signal, target_db=-3.0)

# Apply fades with curves
faded = VolumeControl.fade(
    signal, 
    fade_in=1.0, 
    fade_out=2.0,
    curve='exponential'  # 'linear', 'exponential', 'logarithmic'
)
```

## Playback Volume

```python
player = AudioPlayer()
player.set_volume(0.7)  # 70% volume
player.play(audio)
```

## Common Values

| Volume | Linear | dB   | Description |
|--------|--------|------|-------------|
| Silent | 0.0    | -∞   | No sound |
| Quiet  | 0.25   | -12  | Background |
| Half   | 0.5    | -6   | Moderate |
| Normal | 0.7    | -3   | Typical |
| Full   | 1.0    | 0    | Maximum |
| Boost  | 2.0    | +6   | Amplified |

## Typical Mixing Levels

```python
# Balanced mix example
comp.set_track_volume('Bass', 0.8)      # Supporting
comp.set_track_volume('Lead', 1.0)      # Prominent
comp.set_track_volume('Pad', 0.4)       # Background
comp.set_master_volume(0.85)            # Leave headroom
```

## See Also

- Full guide: `VOLUME_CONTROL.md`
- Demo script: `examples/volume_control_demo.py`
- Main README: `README.md`
