"""
AGAIN - an algorythm composition
~1:41, key of A minor, 85 BPM

Structure:
  Intro     (bars  0- 3) : pad only, atmospheric
  Buildup   (bars  4- 7) : bass enters bar 4, counter enters bar 5
  Verse 1   (bars  8-15) : full band, lead A/B melody
  Bridge    (bars 16-19) : kick/hat drop, lower lead, descending counter
  Build 2   (bars 20-23) : kick returns bar 21, walking bass, tension
  Chorus    (bars 24-31) : peak — new lead D/E, busy hats, walking bass
  Outro     (bars 32-35) : strip back to pad + counter, long fade
"""

from algorythm.sequence import Motif, Scale
from algorythm.structure import Chorus, Composition, Delay, Reverb
from algorythm.synth import ADSR, Filter, Synth, SynthPresets

# ─── Setup ────────────────────────────────────────────────────────────────────
TEMPO = 85
comp = Composition(tempo=TEMPO)

# ─── Scales ───────────────────────────────────────────────────────────────────
scale_4 = Scale.minor("A", octave=4)  # main melody & pad
scale_3 = Scale.minor("A", octave=3)  # counter-melody & bridge lead
scale_2 = Scale.minor("A", octave=2)  # bass
scale_1 = Scale.minor("A", octave=1)  # kick drum pitch (~55 Hz)

# ─── Motifs ───────────────────────────────────────────────────────────────────

# Lead — verse (octave 4, descending/ascending pair)
motif_mel_a = Motif.from_intervals(
    [4, 3, 2, 0],
    scale=scale_4,
    durations=[0.75, 0.75, 0.75, 1.75],
)
motif_mel_b = Motif.from_intervals(
    [0, 2, 4, 4],
    scale=scale_4,
    durations=[0.5, 0.5, 1.5, 1.5],
)

# Lead — bridge (octave 3, stepwise, introspective)
motif_mel_c = Motif.from_intervals(
    [0, 1, 2, 4],
    scale=scale_3,
    durations=[1.0, 0.75, 0.75, 1.5],
)

# Lead — chorus (syncopated, energetic)
motif_mel_d = Motif.from_intervals(
    [2, 4, 3, 0],
    scale=scale_4,
    durations=[0.5, 1.0, 1.0, 1.5],
)
motif_mel_e = Motif.from_intervals(
    [0, 2, 4, 6],
    scale=scale_4,
    durations=[0.75, 0.75, 0.75, 1.75],
)

# Counter-melody — Am and C arpeggios
motif_counter_am = Motif.from_intervals(
    [0, 2, 4, 2],
    scale=scale_3,
    durations=[1.0, 1.0, 1.0, 1.0],
)
motif_counter_c = Motif.from_intervals(
    [2, 4, 6, 4],
    scale=scale_3,
    durations=[1.0, 1.0, 1.0, 1.0],
)

# Counter-melody — bridge (descending, different feel)
motif_counter_bridge = Motif.from_intervals(
    [4, 2, 0, 2],
    scale=scale_3,
    durations=[1.0, 1.0, 1.0, 1.0],
)

# Pad chords
motif_pad_am = Motif.from_intervals(
    [0, 2, 4, 2],
    scale=scale_4,
    durations=[1.5, 0.5, 1.5, 0.5],
)
motif_pad_c = Motif.from_intervals(
    [2, 4, 6, 4],
    scale=scale_4,
    durations=[1.5, 0.5, 1.5, 0.5],
)

# Bass — root pulse
motif_bass_a = Motif.from_intervals(
    [0, 0, 4, 4],
    scale=scale_2,
    durations=[1.0, 1.0, 1.0, 1.0],
)
# Bass — sustained root
motif_bass_b = Motif.from_intervals(
    [0, 0, 0, 0],
    scale=scale_2,
    durations=[1.0, 1.0, 1.0, 1.0],
)
# Bass — walking (chorus)
motif_bass_c = Motif.from_intervals(
    [0, 2, 4, 6],
    scale=scale_2,
    durations=[0.75, 0.75, 0.75, 1.75],
)

# Drums — kick on beats 1 & 3
motif_kick = Motif.from_intervals(
    [0, 0],
    scale=scale_1,
    durations=[2.0, 2.0],
)
# Hi-hat — quarter notes (full)
motif_hat = Motif.from_intervals(
    [0, 0, 0, 0],
    scale=scale_4,
    durations=[1.0, 1.0, 1.0, 1.0],
)
# Hi-hat — half notes (sparse, for buildup / build 2)
motif_hat_sparse = Motif.from_intervals(
    [0, 0],
    scale=scale_4,
    durations=[2.0, 2.0],
)
# Hi-hat — 8th notes (busy, for chorus off-bars)
motif_hat_busy = Motif.from_intervals(
    [0, 0, 0, 0, 0, 0, 0, 0],
    scale=scale_4,
    durations=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
)

# ─── Synths ───────────────────────────────────────────────────────────────────
lead_synth = SynthPresets.pluck()  # triangle, lowpass — warm
counter_synth = SynthPresets.bell()  # FM, rings out
pad_synth = SynthPresets.warm_pad()
bass_synth = SynthPresets.bass()
kick_synth = SynthPresets.kick_drum()
hat_synth = SynthPresets.hi_hat()

# ─── Arrangement ──────────────────────────────────────────────────────────────
#
# Bars:    0  1  2  3 | 4  5  6  7 | 8  9 10 11 12 13 14 15 | 16 17 18 19 | 20 21 22 23 | 24 25 26 27 28 29 30 31 | 32 33 34 35
# PAD:     Am Am Am C | Am Am C Am | Am C Am Am  C Am Am  C  | Am Am  C Am | Am Am  C Am | Am  C Am Am  C Am Am  C  | Am Am Am Am
# BASS:     -  -  -  -|  a  a  b  a|  a  b  a  a  b  a  a  b|  b  b  b  b |  a  a  c  c |  c  b  c  b  c  b  c  b  |  b  b  b  b
# CNTRL:    -  -  -  -|  -  am c  am| am  c  am am  c  am am  c| br br br br | am  c  am  c| am  c  am  c  am  c  am  c| am  -  -  -
# LEAD:     -  -  -  -|  -  -  -  -|  a  b  a  b  a  b  a  b |  c  c  c  c |  c  a  c  a |  d  e  d  e  d  e  d  e   |  -  -  -  -
# KICK:     -  -  -  -|  -  -  -  -|  x  x  x  x  x  x  x  x|  -  -  -  -  | -  x  x  x |  x  x  x  x  x  x  x  x  |  x  -  -  -
# HAT:      -  -  -  -|  -  -  -  -|  -  s  s  x  x  x  x  x|  -  -  -  -  | -  s  x  x |  x  b  x  b  x  b  x  b   |  -  -  -  -

# --- Pad (all 36 bars) ---
comp.add_track("pad", pad_synth)
# Intro (0-3): Am Am Am C
for bar in [0, 1, 2]:
    comp.repeat_motif(motif_pad_am, bars=1, start_bar=bar)
comp.repeat_motif(motif_pad_c, bars=1, start_bar=3)
# Buildup (4-7): Am Am C Am
for bar in [4, 5, 7]:
    comp.repeat_motif(motif_pad_am, bars=1, start_bar=bar)
comp.repeat_motif(motif_pad_c, bars=1, start_bar=6)
# Verse 1 (8-15): Am C Am Am C Am Am C
for bar in [8, 10, 11, 13, 14]:
    comp.repeat_motif(motif_pad_am, bars=1, start_bar=bar)
for bar in [9, 12, 15]:
    comp.repeat_motif(motif_pad_c, bars=1, start_bar=bar)
# Bridge (16-19): Am Am C Am
for bar in [16, 17, 19]:
    comp.repeat_motif(motif_pad_am, bars=1, start_bar=bar)
comp.repeat_motif(motif_pad_c, bars=1, start_bar=18)
# Build 2 (20-23): Am Am C Am
for bar in [20, 21, 23]:
    comp.repeat_motif(motif_pad_am, bars=1, start_bar=bar)
comp.repeat_motif(motif_pad_c, bars=1, start_bar=22)
# Chorus (24-31): Am C Am Am C Am Am C
for bar in [24, 26, 27, 29, 30]:
    comp.repeat_motif(motif_pad_am, bars=1, start_bar=bar)
for bar in [25, 28, 31]:
    comp.repeat_motif(motif_pad_c, bars=1, start_bar=bar)
# Outro (32-35): Am Am Am Am
for bar in [32, 33, 34, 35]:
    comp.repeat_motif(motif_pad_am, bars=1, start_bar=bar)
comp.add_fx(Reverb(mix=0.5, room_size=0.7, damping=0.4))
comp.add_fx(Chorus(rate=0.6, depth=0.4, mix=0.25))
comp.set_track_volume("pad", 0.45)

# --- Bass (enters bar 4) ---
comp.add_track("bass", bass_synth)
# Buildup (4-7)
for bar in [4, 5, 7]:
    comp.repeat_motif(motif_bass_a, bars=1, start_bar=bar)
comp.repeat_motif(motif_bass_b, bars=1, start_bar=6)
# Verse 1 (8-15)
for bar in [8, 10, 11, 13, 14]:
    comp.repeat_motif(motif_bass_a, bars=1, start_bar=bar)
for bar in [9, 12, 15]:
    comp.repeat_motif(motif_bass_b, bars=1, start_bar=bar)
# Bridge (16-19): sustained root only
for bar in [16, 17, 18, 19]:
    comp.repeat_motif(motif_bass_b, bars=1, start_bar=bar)
# Build 2 (20-23): pulse then walking
for bar in [20, 21]:
    comp.repeat_motif(motif_bass_a, bars=1, start_bar=bar)
for bar in [22, 23]:
    comp.repeat_motif(motif_bass_c, bars=1, start_bar=bar)
# Chorus (24-31): walking alternating with sustained
for bar in [24, 26, 28, 30]:
    comp.repeat_motif(motif_bass_c, bars=1, start_bar=bar)
for bar in [25, 27, 29, 31]:
    comp.repeat_motif(motif_bass_b, bars=1, start_bar=bar)
# Outro (32-35): sustained
for bar in [32, 33, 34, 35]:
    comp.repeat_motif(motif_bass_b, bars=1, start_bar=bar)
comp.add_fx(Reverb(mix=0.15, room_size=0.3, damping=0.7))
comp.set_track_volume("bass", 0.7)

# --- Counter-melody (enters bar 5) ---
comp.add_track("counter", counter_synth)
# Buildup (5-7)
comp.repeat_motif(motif_counter_am, bars=1, start_bar=5)
comp.repeat_motif(motif_counter_c, bars=1, start_bar=6)
comp.repeat_motif(motif_counter_am, bars=1, start_bar=7)
# Verse 1 (8-15)
for bar in [8, 10, 11, 13, 14]:
    comp.repeat_motif(motif_counter_am, bars=1, start_bar=bar)
for bar in [9, 12, 15]:
    comp.repeat_motif(motif_counter_c, bars=1, start_bar=bar)
# Bridge (16-19): descending counter
for bar in [16, 17, 18, 19]:
    comp.repeat_motif(motif_counter_bridge, bars=1, start_bar=bar)
# Build 2 (20-23)
for bar in [20, 22]:
    comp.repeat_motif(motif_counter_am, bars=1, start_bar=bar)
for bar in [21, 23]:
    comp.repeat_motif(motif_counter_c, bars=1, start_bar=bar)
# Chorus (24-31)
for bar in [24, 26, 28, 30]:
    comp.repeat_motif(motif_counter_am, bars=1, start_bar=bar)
for bar in [25, 27, 29, 31]:
    comp.repeat_motif(motif_counter_c, bars=1, start_bar=bar)
# Outro: one last phrase
comp.repeat_motif(motif_counter_am, bars=1, start_bar=32)
comp.add_fx(Reverb(mix=0.55, room_size=0.8, damping=0.3))
comp.set_track_volume("counter", 0.35)

# --- Lead melody (enters bar 8) ---
comp.add_track("lead", lead_synth)
# Verse 1 (8-15): alternating A/B
for bar in [8, 10, 12, 14]:
    comp.repeat_motif(motif_mel_a, bars=1, start_bar=bar)
for bar in [9, 11, 13, 15]:
    comp.repeat_motif(motif_mel_b, bars=1, start_bar=bar)
# Bridge (16-19): lower stepwise motif
for bar in [16, 17, 18, 19]:
    comp.repeat_motif(motif_mel_c, bars=1, start_bar=bar)
# Build 2 (20-23): bridge motif mixed with verse A (tension)
for bar in [20, 22]:
    comp.repeat_motif(motif_mel_c, bars=1, start_bar=bar)
for bar in [21, 23]:
    comp.repeat_motif(motif_mel_a, bars=1, start_bar=bar)
# Chorus (24-31): new syncopated D/E pair
for bar in [24, 26, 28, 30]:
    comp.repeat_motif(motif_mel_d, bars=1, start_bar=bar)
for bar in [25, 27, 29, 31]:
    comp.repeat_motif(motif_mel_e, bars=1, start_bar=bar)
comp.add_fx(Reverb(mix=0.3, room_size=0.5, damping=0.5))
comp.add_fx(Delay(delay_time=0.35, feedback=0.2, mix=0.15))
comp.set_track_volume("lead", 0.75)

# --- Kick drum (enters bar 8, drops for bridge, returns bar 21) ---
comp.add_track("kick", kick_synth)
comp.repeat_motif(motif_kick, bars=8, start_bar=8)  # Verse 1
# Bridge (16-19): no kick
comp.repeat_motif(motif_kick, bars=3, start_bar=21)  # Build 2 tail
comp.repeat_motif(motif_kick, bars=8, start_bar=24)  # Chorus
comp.repeat_motif(motif_kick, bars=1, start_bar=32)  # Outro beat 1
comp.set_track_volume("kick", 0.8)

# --- Hi-hat (enters bar 9, drops for bridge, returns bar 21) ---
comp.add_track("hihat", hat_synth)
# Verse 1: sparse at first (9-11), full from bar 12
for bar in [9, 10, 11]:
    comp.repeat_motif(motif_hat_sparse, bars=1, start_bar=bar)
comp.repeat_motif(motif_hat, bars=4, start_bar=12)
# Bridge (16-19): silence
# Build 2 (21-23): sparse re-enter, then full
for bar in [21, 22]:
    comp.repeat_motif(motif_hat_sparse, bars=1, start_bar=bar)
comp.repeat_motif(motif_hat, bars=1, start_bar=23)
# Chorus (24-31): full on even bars, busy 8th-notes on odd bars
for bar in [24, 26, 28, 30]:
    comp.repeat_motif(motif_hat, bars=1, start_bar=bar)
for bar in [25, 27, 29, 31]:
    comp.repeat_motif(motif_hat_busy, bars=1, start_bar=bar)
comp.set_track_volume("hihat", 0.3)

# ─── Master ───────────────────────────────────────────────────────────────────
comp.set_master_volume(0.82)
comp.fade_in(0.25)
comp.fade_out(0.5)

# ─── Render ───────────────────────────────────────────────────────────────────
print("Rendering AGAIN...")
comp.render(file_path="AGAIN.wav", quality="high", formats=["wav"])
print("Done! -> AGAIN.wav")
