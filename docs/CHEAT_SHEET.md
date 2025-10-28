# 🎵 Algorythm Cheat Sheet for Kids

## The 3 Magic Steps 🪄

```python
1. Pick an instrument  →  my_instrument = SynthPresets.pluck()
2. Pick notes         →  my_melody = Motif.from_intervals([0, 2, 4])
3. Make the song      →  song = Composition(tempo=120)
                         song.add_track('Track1', my_instrument)
                         song.repeat_motif(my_melody, bars=4)
                         song.render('output.wav')
```

---

## 🎹 All the Instruments You Can Use

```python
# Basic Sounds
SynthPresets.pluck()      # 🎸 Like a guitar
SynthPresets.bell()       # 🔔 Like a bell
SynthPresets.warm_pad()   # 🌊 Smooth and warm
SynthPresets.bass()       # 🔊 Deep and low
SynthPresets.lead()       # 🎺 Bright and loud

# Advanced Sounds (NEW!)
SynthPresets.organ()      # 🎹 Church organ
SynthPresets.strings()    # 🎻 String section
SynthPresets.guitar()     # 🎸 Acoustic guitar
SynthPresets.drum()       # 🥁 Drum sound
SynthPresets.brass()      # 🎺 Trumpet/trombone
```

---

## 🎼 Making Melodies (The Numbers Game!)

The numbers tell which notes to play:

```python
[0]              # One note (boring!)
[0, 2, 4]        # Three notes going up
[0, 2, 4, 2, 0]  # Up and back down
[0, 4, 7]        # A chord sound
[0, 0, 0, 0]     # Same note (good for bass!)
```

**What the numbers mean:**
- 0 = Starting note
- 1 = One step up
- 2 = Two steps up
- -1 = One step down
- 7 = Seven steps up (high!)

---

## 🎚️ Scales (Happy vs Sad Sounds)

```python
Scale.major('C')    # 😊 Happy and bright
Scale.minor('A')    # 😢 Sad and dark
```

**Popular starting notes:**
- 'C', 'D', 'E', 'F', 'G', 'A'

---

## ⏱️ Speed (Tempo)

```python
tempo=60     # 🐢 Slow and chill
tempo=120    # 🚶 Normal walking speed
tempo=180    # 🏃 Fast and exciting!
```

---

## ✨ Cool Effects

```python
from algorythm.structure import Reverb, Delay

# Add echo/space (like singing in a bathroom!)
song.add_fx(Reverb(mix=0.3))

# Add repeating echo
song.add_fx(Delay(delay_time=0.5, mix=0.3))
```

---

## 📏 How Long to Play (Bars)

```python
repeat_motif(my_melody, bars=2)   # Short
repeat_motif(my_melody, bars=4)   # Medium
repeat_motif(my_melody, bars=8)   # Long
```

---

## 🎨 Cool Effects to Add!

Make your music sound even cooler by adding effects:

```python
from algorythm.effects import *

# Space Effects (makes it sound big!)
Reverb(room_size=0.7, wet_level=0.3)    # 🏰 Like in a big hall
Delay(delay_time=0.5, feedback=0.5)      # 📢 Echo effect

# Wobbly Effects (makes it swirl!)
Chorus(mix=0.5)                          # 🌀 Makes it thick and swirly
Flanger(mix=0.5)                         # 🎭 Jet plane sound
Phaser(mix=0.5)                          # ✨ Spacey swoosh
Tremolo(rate=5.0, depth=0.5)            # 📻 Volume wobble

# Crunchy Effects (makes it rough!)
Distortion(drive=5.0)                    # 🎸 Rock guitar sound
Overdrive(drive=2.0)                     # 🔥 Warm crunch
Fuzz(gain=10.0)                          # 💥 Super fuzzy
BitCrusher(bit_depth=8)                  # 🎮 Video game sound

# How to use effects:
track = Track("My Track")
track.add_effect(Reverb(room_size=0.7))
track.add_effect(Delay(delay_time=0.5))
```

---

## 🎨 Copy-Paste Templates

### Template 1: One Instrument
```python
from algorythm.synth import SynthPresets
from algorythm.sequence import Motif, Scale
from algorythm.structure import Composition

instrument = SynthPresets.pluck()
melody = Motif.from_intervals([0, 2, 4, 5], scale=Scale.major('C'))

song = Composition(tempo=120)
song.add_track('Track1', instrument)
song.repeat_motif(melody, bars=4)
song.render('my_song.wav')
```

### Template 2: Two Instruments (Lead + Bass)
```python
from algorythm.synth import SynthPresets
from algorythm.sequence import Motif, Scale
from algorythm.structure import Composition

lead = SynthPresets.bell()
bass = SynthPresets.bass()

lead_melody = Motif.from_intervals([0, 2, 4, 7], scale=Scale.major('C'))
bass_melody = Motif.from_intervals([0, 0, 0, 0], scale=Scale.major('C'))

song = Composition(tempo=120)
song.add_track('Lead', lead).repeat_motif(lead_melody, bars=4)
song.add_track('Bass', bass).repeat_motif(bass_melody, bars=4)
song.render('two_tracks.wav')
```

### Template 3: With Effects
```python
from algorythm.synth import SynthPresets
from algorythm.sequence import Motif, Scale
from algorythm.structure import Composition, Reverb

instrument = SynthPresets.warm_pad()
melody = Motif.from_intervals([0, 2, 4, 2, 0], scale=Scale.major('C'))

song = Composition(tempo=100)
song.add_track('Pad', instrument)
song.repeat_motif(melody, bars=8)
song.add_fx(Reverb(mix=0.5))  # Lots of echo for dreamy sound!
song.render('dreamy_song.wav')
```

---

## 🎯 Quick Experiments to Try

1. **Make a happy song**: Use `Scale.major('C')` and tempo=140
2. **Make a sad song**: Use `Scale.minor('A')` and tempo=80
3. **Make a fast exciting song**: Use tempo=180 and `SynthPresets.lead()`
4. **Make a calm song**: Use `SynthPresets.warm_pad()`, tempo=60, and Reverb
5. **Make a bass drop**: Use `SynthPresets.bass()` with `[0, 0, 0, 0]` notes

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| Can't find the file | Check `~/Music/` folder! |
| "Module not found" error | Run `pip install -e .` in algorythm folder |
| No sound | Did you add a track and melody? |
| Song too short | Increase the `bars=` number! |
| Song too quiet | Your computer volume might be low! |

---

## 🎓 Level Up!

Once you're comfortable, check out:
- `README.md` - Full documentation
- `examples/` folder - More complex examples
- Try making a full song with intro, verse, chorus!

Remember: **Experiment and have fun!** 🎉
