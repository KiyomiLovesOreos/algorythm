"""
R3V3RB - an algorythm composition
~1:36, key of D minor, 190 BPM
Fast breakcore — each section has a distinct layer combination.

Structure + texture fingerprint:
  Intro      (bars  0- 3) : pad only
  Drop 1     (bars  4-19) : kick+snare+hat+bass+lead — raw, no arp, no tom
  Heavy Var  (bars 20-27) : everything — tom+arp enter, snare every beat, 32nd hats, acid bass
  Drop 2     (bars 28-43) : tom DROPS OUT — focused, arp texture, bass_main returns for contrast
  Peak       (bars 44-51) : arp DROPS OUT — tom returns hard, acid bass, 32nd every bar
  Stutter    (bars 52-55) : stripped to kick+snare+hat+bass only — pure rhythm, no lead/tom/arp
  Final Drop (bars 56-71) : everything back — tom+arp+lead D, maximum layers
  Outro      (bars 72-75) : fade
"""

from algorythm.sequence import Motif, Scale
from algorythm.structure import Composition, Delay, Reverb
from algorythm.synth import SynthPresets

# ─── Setup ────────────────────────────────────────────────────────────────────
TEMPO = 190
comp = Composition(tempo=TEMPO)

# ─── Scales ───────────────────────────────────────────────────────────────────
# D minor: D(0) E(1) F(2) G(3) A(4) Bb(5) C(6) D+oct(7)
scale_3 = Scale.minor("D", octave=3)   # lead + arp
scale_2 = Scale.minor("D", octave=2)   # bass
scale_1 = Scale.minor("D", octave=1)   # drums

# ─── Lead motifs ──────────────────────────────────────────────────────────────
motif_lead_a = Motif.from_intervals(        # D→Bb→F→D stab (Drop 1 even bars)
    [0, 5, 2, 0],
    scale=scale_3,
    durations=[0.5, 0.25, 0.25, 3.0],
)
motif_lead_b = Motif.from_intervals(        # Bb→F→D→F grind (Drop 1 odd bars)
    [5, 2, 0, 2],
    scale=scale_3,
    durations=[0.5, 0.5, 0.5, 2.5],
)
motif_lead_c = Motif.from_intervals(        # Bb,F,D,G,Bb glitch (Heavy Var + Drop 2 peak)
    [5, 2, 0, 3, 5],
    scale=scale_3,
    durations=[0.25, 0.25, 0.25, 0.25, 3.0],
)
motif_lead_b2 = Motif.from_intervals(       # G→D→Bb→G — darker variant of B (Drop 2)
    [3, 0, 5, 3],
    scale=scale_3,
    durations=[0.5, 0.5, 0.5, 2.5],
)
motif_lead_peak = Motif.from_intervals(     # F,Bb,D,Bb,F — dense chromatic (Peak, no arp)
    [2, 5, 0, 5, 2],
    scale=scale_3,
    durations=[0.25, 0.25, 0.25, 0.25, 3.0],
)
motif_lead_d = Motif.from_intervals(        # D,Bb,F,G,D — final drop peak
    [0, 5, 2, 3, 0],
    scale=scale_3,
    durations=[0.25, 0.25, 0.25, 0.25, 3.0],
)

# ─── Arp motifs ───────────────────────────────────────────────────────────────
motif_arp_a = Motif.from_intervals(
    [0, 5, 3, 0],
    scale=scale_3,
    durations=[0.5, 0.5, 0.5, 2.5],
)
motif_arp_b = Motif.from_intervals(
    [5, 2, 0, 5],
    scale=scale_3,
    durations=[0.5, 0.5, 0.5, 2.5],
)
motif_arp_c = Motif.from_intervals(        # final drop variant — F,D,Bb,G different shape
    [2, 0, 5, 3],
    scale=scale_3,
    durations=[0.5, 0.5, 0.5, 2.5],
)

# ─── Pad ──────────────────────────────────────────────────────────────────────
motif_pad = Motif.from_intervals(
    [0, 2, 4, 2],
    scale=scale_3,
    durations=[1.0, 1.0, 1.0, 1.0],
)

# ─── Bass motifs ──────────────────────────────────────────────────────────────
motif_bass_main = Motif.from_intervals(     # root pulse (Drop 1, Drop 2 contrast)
    [0, 0, 5, 4],
    scale=scale_2,
    durations=[0.5, 0.5, 1.0, 2.0],
)
motif_bass_acid = Motif.from_intervals(     # rapid acid (Heavy Var, Peak, Final Drop)
    [0, 0, 0, 5, 0, 4],
    scale=scale_2,
    durations=[0.25, 0.25, 0.5, 0.5, 0.5, 2.0],
)
motif_bass_drive = Motif.from_intervals(    # driving staccato (Final Drop variation)
    [0, 0, 5, 5, 0, 4],
    scale=scale_2,
    durations=[0.25, 0.25, 0.25, 0.25, 1.0, 2.0],
)
motif_bass_simple = Motif.from_intervals(   # root only (Stutter, intro, outro)
    [0, 0],
    scale=scale_2,
    durations=[2.0, 2.0],
)

# ─── Drum motifs ──────────────────────────────────────────────────────────────
motif_kick_main = Motif.from_intervals(
    [0, 0, 0, 0, 0],
    scale=scale_1,
    durations=[0.75, 0.5, 0.75, 1.0, 1.0],
)
motif_kick_simple = Motif.from_intervals(
    [0, 0],
    scale=scale_1,
    durations=[2.0, 2.0],
)
motif_kick_break = Motif.from_intervals(
    [0, 0, 0, 0, 0, 0],
    scale=scale_1,
    durations=[0.5, 0.5, 0.5, 0.5, 1.0, 1.0],
)
motif_kick_chaos = Motif.from_intervals(
    [0, 0, 0, 0, 0, 0, 0],
    scale=scale_1,
    durations=[0.5, 0.25, 0.25, 0.5, 0.25, 0.25, 2.0],
)

motif_snare_main = Motif.from_intervals(    # beats 2 & 4
    [0, 0],
    scale=scale_1,
    durations=[2.0, 2.0],
)
motif_snare_every = Motif.from_intervals(   # every beat — industrial
    [0, 0, 0, 0],
    scale=scale_1,
    durations=[1.0, 1.0, 1.0, 1.0],
)
motif_snare_roll = Motif.from_intervals(
    [0, 0, 0, 0],
    scale=scale_1,
    durations=[0.5, 0.5, 1.0, 2.0],
)
motif_snare_chaos = Motif.from_intervals(
    [0, 0, 0],
    scale=scale_1,
    durations=[1.5, 0.5, 2.0],
)

motif_hat_8th = Motif.from_intervals(
    [0] * 8,
    scale=scale_1,
    durations=[0.5] * 8,
)
motif_hat_16th = Motif.from_intervals(
    [0] * 16,
    scale=scale_1,
    durations=[0.25] * 16,
)
motif_hat_32nd = Motif.from_intervals(
    [0] * 32,
    scale=scale_1,
    durations=[0.125] * 32,
)

motif_tom_8th = Motif.from_intervals(
    [0] * 8,
    scale=scale_1,
    durations=[0.5] * 8,
)

motif_cymbal_crash = Motif.from_intervals(
    [0],
    scale=scale_1,
    durations=[4.0],
)

# ─── Synths ───────────────────────────────────────────────────────────────────
lead_synth   = SynthPresets.synth_brass()
arp_synth    = SynthPresets.synth_pluck()
pad_synth    = SynthPresets.ambient_pad()
bass_synth   = SynthPresets.acid_bass()
kick_synth   = SynthPresets.kick_drum()
snare_synth  = SynthPresets.snare_drum()
tom_synth    = SynthPresets.tom_drum()
hat_synth    = SynthPresets.hi_hat()
cymbal_synth = SynthPresets.cymbal()

# ─── Arrangement ──────────────────────────────────────────────────────────────
#
# Each section has a distinct layer signature — silence is as important as presence:
#
# Section    | Bars  | Kick     | Snare      | Hat          | Tom | Arp | Bass  | Lead
# -----------|-------|----------|------------|--------------|-----|-----|-------|------
# Intro      | 0- 3  | —        | —          | —            | —   | —   |simple | —
# Drop 1     | 4-19  | main     | 2&4        | 8th→16th     | —   | —   | main  | A/B
# Heavy Var  | 20-27 | simple   | EVERY BEAT | 32nd         | YES | YES | acid  | C
# Drop 2     | 28-43 | break    | 2&4        | 16th+32nd/4  | —   | YES | MAIN  | B2/C
# Peak       | 44-51 | chaos    | roll       | 32nd every   | YES | —   | acid  | peak
# Stutter    | 52-55 | chaos    | roll       | 32nd         | —   | —   | SIMPLE| —
# Final Drop | 56-71 | chaos    | roll/chaos | 16th+32nd/2  | YES | YES | drive | D
# Outro      | 72-75 | simple   | —          | —            | —   | —   |simple | —

# --- Pad (intro only) ---
comp.add_track("pad", pad_synth)
for bar in range(4):
    comp.repeat_motif(motif_pad, bars=1, start_bar=bar)
comp.add_fx(Reverb(mix=0.8, room_size=0.95, damping=0.15))
comp.set_track_volume("pad", 0.3)

# --- Bass ---
# Deliberately varies between sections for contrast
comp.add_track("bass", bass_synth)
for bar in range(4):
    comp.repeat_motif(motif_bass_simple, bars=1, start_bar=bar)
for bar in range(4, 20):                    # Drop 1: pulse
    comp.repeat_motif(motif_bass_main, bars=1, start_bar=bar)
for bar in range(20, 28):                   # Heavy Var: acid
    comp.repeat_motif(motif_bass_acid, bars=1, start_bar=bar)
for bar in range(28, 44):                   # Drop 2: main returns (contrast!)
    comp.repeat_motif(motif_bass_main, bars=1, start_bar=bar)
for bar in range(44, 52):                   # Peak: acid back
    comp.repeat_motif(motif_bass_acid, bars=1, start_bar=bar)
for bar in range(52, 56):                   # Stutter: just root
    comp.repeat_motif(motif_bass_simple, bars=1, start_bar=bar)
for bar in range(56, 72):                   # Final Drop: driving staccato
    comp.repeat_motif(motif_bass_drive, bars=1, start_bar=bar)
for bar in range(72, 76):
    comp.repeat_motif(motif_bass_simple, bars=1, start_bar=bar)
comp.add_fx(Reverb(mix=0.1, room_size=0.2, damping=0.85))
comp.set_track_volume("bass", 0.85)

# --- Arp (absent in Drop 1, Peak, and Stutter — presence/absence creates shape) ---
comp.add_track("arp", arp_synth)
# Heavy Var (20-27)
for bar in range(20, 28, 2):
    comp.repeat_motif(motif_arp_a, bars=1, start_bar=bar)
for bar in range(21, 28, 2):
    comp.repeat_motif(motif_arp_b, bars=1, start_bar=bar)
# Drop 2 (28-43): arp A/B texture
for bar in range(28, 44, 2):
    comp.repeat_motif(motif_arp_a, bars=1, start_bar=bar)
for bar in range(29, 44, 2):
    comp.repeat_motif(motif_arp_b, bars=1, start_bar=bar)
# Peak (44-51): NO ARP — absence creates impact with tom
# Stutter (52-55): NO ARP
# Final Drop (56-71): arp_c variant (different shape)
for bar in range(56, 72, 2):
    comp.repeat_motif(motif_arp_c, bars=1, start_bar=bar)
for bar in range(57, 72, 2):
    comp.repeat_motif(motif_arp_b, bars=1, start_bar=bar)
comp.add_fx(Reverb(mix=0.3, room_size=0.5, damping=0.5))
comp.add_fx(Delay(delay_time=0.09, feedback=0.2, mix=0.12))
comp.set_track_volume("arp", 0.38)

# --- Lead (absent in Stutter — pure rhythm section) ---
comp.add_track("lead", lead_synth)
# Drop 1 (4-19): A and B alternating
for bar in range(4, 20, 2):
    comp.repeat_motif(motif_lead_a, bars=1, start_bar=bar)
for bar in range(5, 20, 2):
    comp.repeat_motif(motif_lead_b, bars=1, start_bar=bar)
# Heavy Var (20-27): C throughout
for bar in range(20, 28):
    comp.repeat_motif(motif_lead_c, bars=1, start_bar=bar)
# Drop 2 (28-43): B2 and C alternating (B2 is darker variant)
for bar in range(28, 44, 2):
    comp.repeat_motif(motif_lead_b2, bars=1, start_bar=bar)
for bar in range(29, 44, 2):
    comp.repeat_motif(motif_lead_c, bars=1, start_bar=bar)
# Peak (44-51): peak motif — denser, no arp beneath so it fills space
for bar in range(44, 52):
    comp.repeat_motif(motif_lead_peak, bars=1, start_bar=bar)
# Stutter (52-55): NO LEAD — drums + bass only
# Final Drop (56-71): D throughout
for bar in range(56, 72):
    comp.repeat_motif(motif_lead_d, bars=1, start_bar=bar)
comp.add_fx(Reverb(mix=0.12, room_size=0.25, damping=0.75))
comp.add_fx(Delay(delay_time=0.06, feedback=0.08, mix=0.07))
comp.set_track_volume("lead", 0.9)

# --- Kick ---
comp.add_track("kick", kick_synth)
comp.repeat_motif(motif_kick_main, bars=15, start_bar=4)    # Drop 1
comp.repeat_motif(motif_kick_break, bars=1, start_bar=19)   # fill → Heavy Var
comp.repeat_motif(motif_kick_simple, bars=7, start_bar=20)  # Heavy Var (simple contrast)
comp.repeat_motif(motif_kick_break, bars=1, start_bar=27)   # fill → Drop 2
comp.repeat_motif(motif_kick_break, bars=15, start_bar=28)  # Drop 2
comp.repeat_motif(motif_kick_chaos, bars=1, start_bar=43)   # fill → Peak
comp.repeat_motif(motif_kick_chaos, bars=12, start_bar=44)  # Peak + Stutter
comp.repeat_motif(motif_kick_chaos, bars=15, start_bar=56)  # Final Drop
comp.repeat_motif(motif_kick_break, bars=1, start_bar=71)   # fill → Outro
comp.repeat_motif(motif_kick_simple, bars=4, start_bar=72)
comp.set_track_volume("kick", 0.95)

# --- Snare ---
comp.add_track("snare", snare_synth)
comp.repeat_motif(motif_snare_main, bars=16, start_bar=4)   # Drop 1: 2&4
comp.repeat_motif(motif_snare_every, bars=8, start_bar=20)  # Heavy Var: every beat
comp.repeat_motif(motif_snare_main, bars=15, start_bar=28)  # Drop 2: 2&4 again
comp.repeat_motif(motif_snare_roll, bars=1, start_bar=43)   # fill
comp.repeat_motif(motif_snare_roll, bars=8, start_bar=44)   # Peak: roll
comp.repeat_motif(motif_snare_roll, bars=4, start_bar=52)   # Stutter: roll
for bar in range(56, 72, 2):                                # Final Drop: roll/chaos alt
    comp.repeat_motif(motif_snare_roll, bars=1, start_bar=bar)
for bar in range(57, 72, 2):
    comp.repeat_motif(motif_snare_chaos, bars=1, start_bar=bar)
comp.set_track_volume("snare", 0.88)

# --- Hi-hat ---
comp.add_track("hihat", hat_synth)
comp.repeat_motif(motif_hat_8th, bars=4, start_bar=4)       # Drop 1 start
comp.repeat_motif(motif_hat_16th, bars=11, start_bar=8)     # Drop 1 upgrade
comp.repeat_motif(motif_hat_8th, bars=1, start_bar=19)      # taper fill
comp.repeat_motif(motif_hat_32nd, bars=8, start_bar=20)     # Heavy Var: 32nd max
for bar in range(28, 44):                                   # Drop 2: 16th + 32nd/4th
    if bar % 4 == 3:
        comp.repeat_motif(motif_hat_32nd, bars=1, start_bar=bar)
    else:
        comp.repeat_motif(motif_hat_16th, bars=1, start_bar=bar)
comp.repeat_motif(motif_hat_32nd, bars=12, start_bar=44)    # Peak + Stutter: 32nd solid
for bar in range(56, 71):                                   # Final Drop: alt 32nd/16th
    if bar % 2 == 1:
        comp.repeat_motif(motif_hat_32nd, bars=1, start_bar=bar)
    else:
        comp.repeat_motif(motif_hat_16th, bars=1, start_bar=bar)
comp.repeat_motif(motif_hat_8th, bars=1, start_bar=71)      # taper
comp.set_track_volume("hihat", 0.33)

# --- Tom (enters Heavy Var, absent in Drop 2 and Stutter — used for contrast) ---
comp.add_track("tom", tom_synth)
comp.repeat_motif(motif_tom_8th, bars=8, start_bar=20)      # Heavy Var
# Drop 2 (28-43): NO TOM — makes Drop 2 feel focused vs chaotic Heavy Var
comp.repeat_motif(motif_tom_8th, bars=8, start_bar=44)      # Peak: returns hard
# Stutter (52-55): NO TOM — pure rhythm strip
comp.repeat_motif(motif_tom_8th, bars=16, start_bar=56)     # Final Drop
comp.add_fx(Reverb(mix=0.15, room_size=0.3, damping=0.7))
comp.set_track_volume("tom", 0.55)

# --- Cymbal crashes ---
comp.add_track("cymbal", cymbal_synth)
for bar in [4, 20, 28, 44, 56]:
    comp.repeat_motif(motif_cymbal_crash, bars=1, start_bar=bar)
comp.add_fx(Reverb(mix=0.45, room_size=0.65, damping=0.4))
comp.set_track_volume("cymbal", 0.62)

# ─── Master ───────────────────────────────────────────────────────────────────
comp.set_master_volume(0.88)
comp.fade_in(1.5)
comp.fade_out(5.0)

# ─── Render ───────────────────────────────────────────────────────────────────
print("Rendering R3V3RB...")
comp.render(file_path="R3V3RB.wav", quality="high", formats=["wav"])
print("Done! -> R3V3RB.wav")
