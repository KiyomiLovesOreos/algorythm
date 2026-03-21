# Glossary

Music and audio terms used in Algorythm documentation and code.

## Musical Terms

**Amplitude**: How loud a sound is. In digital audio, represented as values from -1.0 to 1.0.

**Arpeggio**: Playing chord notes one at a time instead of all together. Creates flowing, harp-like sound.

**ADSR**: Envelope with four stages:
- Attack: time to reach peak
- Decay: time to fall to sustain level
- Sustain: level to hold
- Release: time to fade out after note ends

**Chord**: Three or more notes played together. Basis of harmony.

**Chromatic**: All 12 semitones. A chromatic scale includes all notes without any gaps.

**Dissonance**: Unstable, tense interval. Creates tension, demands resolution. Examples: tritone, minor 2nd.

**Enharmonic**: Different names for same note. C# and Db are enharmonic.

**Frequency**: How many times per second a sound wave vibrates. Measured in Hertz (Hz). Higher frequency = higher pitch.

**Harmonic**: Overtone above the fundamental frequency. Determines timbre/color of sound.

**Harmony**: How chords sound together. Study of vertical (simultaneous) notes.

**Interval**: Distance between two pitches. Measured in semitones.

**Key**: A scale plus harmonic context. "In the key of C major" means using C major scale.

**Legato**: Smooth, connected notes without gaps.

**Major Scale**: Bright, happy sounding scale. Pattern: 2-2-1-2-2-2-1 semitones.

**Melody**: Sequence of notes forming memorable tune. Horizontal (sequential) aspect of music.

**Minor Scale**: Dark, emotional sounding scale. Pattern: 2-1-2-2-1-2-2 semitones (natural minor).

**Mode**: Type of scale. Major, minor, dorian, phrygian, lydian, mixolydian, locrian.

**Motif**: Short melodic idea or pattern. Building block of composition.

**Octave**: Doubling or halving of frequency. Notes 12 semitones apart.

**Pentatonic**: Scale with 5 notes per octave. Common in many musical styles.

**Pitch**: Perceived highness or lowness of sound. Related to frequency.

**Rhythm**: Pattern of durations. When notes start and stop.

**Root**: Lowest note of a chord. Gives chord its name. C-E-G has root C.

**Semitone**: Smallest interval in Western music. 1/12th of an octave. Also called "half step."

**Syncopation**: Emphasis on off-beats or unexpected rhythmic placements.

**Tempo**: Speed of music. Measured in BPM (beats per minute).

**Timbre**: Color or quality of sound. What makes trumpet sound different from piano.

**Tone**: Single pitch. Also refers to timbre/color.

**Transposition**: Moving music to a different key while keeping same intervals.

## Audio/Technical Terms

**Bit Depth**: Resolution of digital audio. 16-bit (CD quality), 24-bit (high quality), 8-bit (lo-fi).

**Bit Rate**: How much data per second. Higher = better quality but larger file. Measured in kbps (kilobits per second).

**Clipping**: Distortion caused by audio signal exceeding -1.0 to 1.0 range. Sounds harsh and broken.

**Compression**: Reducing dynamic range. Loud parts get quieter, soft parts stay relatively loud. Makes mix "punchier."

**Convolution**: Mathematical operation used for realistic reverb simulation using impulse responses.

**Damping**: Reducing high frequencies over time. Makes sound more muffled or natural.

**dB (Decibel)**: Logarithmic unit measuring sound pressure level. 20 dB = much louder.

**Delay**: Echo effect. Repeats sound after a time interval.

**Distortion**: Adding harmonics by overdriving signal. Creates aggressive, edgy sound.

**DSP (Digital Signal Processing)**: Mathematical manipulation of audio signals.

**Equalization (EQ)**: Boosting or cutting frequencies. Shapes tone of sound.

**Feedback**: Signal feeding back into itself, typically through delay or filter. Can create runaway effect.

**Filter**: Removes or emphasizes frequencies. Types: lowpass, highpass, bandpass, notch.

**Flanger**: Modulated delay effect. Creates "jet plane" swooshing sound.

**Gain**: Volume level of signal. Can be boost (above 0 dB) or cut (below 0 dB).

**Harmonic Distortion**: Adding harmonics through nonlinear processing. More musical than noise.

**Hz (Hertz)**: Unit of frequency. One cycle per second.

**Inharmonic**: Frequencies not related by whole-number ratios. Typical of percussion instruments.

**Latency**: Delay between input and output. Important for real-time audio.

**Limiter**: Special compressor preventing signal from exceeding threshold. Protects against clipping.

**Mix**: Blending multiple audio tracks at appropriate volumes. Creating balance.

**Modulation**: Varying parameter (like pitch or volume) over time using LFO or envelope.

**Noise Floor**: Minimum audible level. Below this, sound is lost to noise.

**Normalization**: Scaling audio so peak level is at 0 dB without clipping.

**Nyquist Frequency**: Highest frequency that can be accurately represented at given sample rate. Half the sample rate.

**Pan**: Position in stereo field. Left (-1) to right (1).

**Phaser**: Modulated all-pass filter creating subtle "whooshing" effect.

**Pitch Shift**: Changing pitch without changing duration. Stretches or compresses frequencies.

**Quantization Noise**: Noise from rounding audio to discrete digital values. More noticeable with lower bit depths.

**Resonance**: Emphasis at cutoff frequency in filter. Higher resonance = more peaked response.

**Reverb**: Spatial effect simulating reflections in acoustic space. Makes sound appear larger/further away.

**Sample**: Single audio value. Audio file is sequence of samples.

**Sample Rate**: How many samples per second. CD = 44.1 kHz, Professional = 48 kHz, High-quality = 96 kHz+.

**Saturation**: Subtle distortion that adds warmth without obvious harshness.

**Synth (Synthesizer)**: Instrument creating sound electronically rather than recording.

**Threshold**: Level at which effect triggers. Common in compressors and gates.

**Tremolo**: Volume modulation. Amplitude variation over time.

**Vibrato**: Pitch modulation. Frequency variation over time.

**Wet/Dry**: Wet = affected by effect, Dry = original unaffected signal. Mix controls ratio.

## Algorythm-Specific Terms

**Composition**: Main class for arranging multiple tracks into a song.

**Exporter**: Class for saving audio to files (WAV, MP3, FLAC, etc).

**FXChain**: Container for multiple effects applied in sequence.

**Motif**: Algorythm's class for melodic patterns. Created from intervals, frequencies, or note names.

**Oscillator**: Generates basic waveforms (sine, square, saw, triangle, noise).

**Preset**: Pre-configured instrument ready to use. SynthPresets has 45+ options.

**Sample**: Audio file loaded for playback or granular synthesis.

**Scale**: Algorythm's class for pitch collections. Major, minor, pentatonic, etc.

**Sequencer**: Tools for creating patterns and sequences (Motif, Rhythm, etc).

**Synth**: Main synthesis class. Combines oscillator, filter, and envelope.

**Track**: Individual instrument in a Composition. Has volume, panning, and effects.

**Tuning**: Alternative pitch systems beyond 12-tone equal temperament.

**Visualizer**: Class for creating video from audio. Various types: waveform, frequency scope, spectrogram, etc.

## Synthesis Terms

**Additive Synthesis**: Building complex sounds from many sine waves (harmonics).

**Carrier**: Main oscillator in FM synthesis that gets modulated.

**FM Synthesis (Frequency Modulation)**: Modulating one oscillator's frequency with another to create complex timbres.

**Fundamental**: Lowest frequency component. Determines perceived pitch.

**Grain**: Small slice of audio. Granular synthesis manipulates many grains.

**Granular Synthesis**: Synthesis using small audio grains triggered and manipulated.

**Modulator**: Oscillator in FM synthesis that modulates the carrier's frequency.

**Oscillator**: Basic sound source generating waveforms.

**Partials**: All frequency components of a sound (fundamental + harmonics).

**Physical Modeling**: Synthesis simulating physical vibrating objects like strings.

**Wavetable**: Pre-computed waveform stored as lookup table. Efficient and flexible.

**Waveform**: Shape of audio signal. Determines timbre.

## File Format Terms

**FLAC**: Free Lossless Audio Codec. Lossless compression without quality loss.

**MP3**: Lossy compression. Smaller files, slight quality loss. Most compatible.

**Quantization**: Process of converting continuous values to discrete digital values.

**WAV**: Waveform Audio File Format. Standard, uncompressed, lossless.

**Lossy**: Compression that removes data. MP3 is lossy.

**Lossless**: Compression that preserves all data. FLAC and WAV are lossless.

## Time and Tempo Terms

**Bar**: Measure of time containing one beat structure. Usually 4 beats in 4/4 time.

**Beat**: Basic unit of musical time.

**BPM**: Beats per minute. Tempo measurement.

**Eighth Note**: Note lasting half a beat (in 4/4 time).

**Measure**: Same as bar.

**Quarter Note**: Note lasting one beat (in 4/4 time).

**Time Signature**: Notation showing beats per measure and which note gets the beat. 4/4, 3/4, 6/8, etc.

**Whole Note**: Note lasting 4 beats (in 4/4 time).

## Common Abbreviations

**ADSR**: Attack, Decay, Sustain, Release

**BPM**: Beats Per Minute

**DAW**: Digital Audio Workstation (like Ableton, Logic, Reaper)

**dB**: Decibel

**DSP**: Digital Signal Processing

**EQ**: Equalization

**FX**: Effects

**Hz**: Hertz (frequency unit)

**kHz**: Kilohertz (1000 Hz)

**LFO**: Low Frequency Oscillator (modulation source)

**MIDI**: Musical Instrument Digital Interface

**MP3**: MPEG-1 Audio Layer III

**ms**: Milliseconds

**Q**: Quality factor (in filters, related to resonance)

**RMS**: Root Mean Square (average loudness)

**WAV**: Waveform Audio Format

## Scales Quick Reference

**Modes (in order)**:
1. Ionian = Major
2. Dorian = Minor with raised 6th
3. Phrygian = Minor with lowered 2nd
4. Lydian = Major with raised 4th
5. Mixolydian = Major with lowered 7th
6. Aeolian = Natural Minor
7. Locrian = Diminished

**Common Scales**:
- Major: Bright, happy
- Minor: Dark, melancholic
- Pentatonic: Safe, melodic (5 notes)
- Blues: Bluesy feel, includes flat 5th
- Chromatic: All 12 semitones
- Whole Tone: Mysterious, unsettling

## Common Chord Types

**Triads** (3-note):
- Major: Root + Major 3rd + Perfect 5th
- Minor: Root + Minor 3rd + Perfect 5th
- Diminished: Root + Minor 3rd + Diminished 5th
- Augmented: Root + Major 3rd + Augmented 5th

**7th Chords** (4-note):
- Major 7th: Root + M3 + P5 + M7
- Minor 7th: Root + m3 + P5 + m7
- Dominant 7th: Root + M3 + P5 + m7
- Half-Diminished 7th: Root + m3 + d5 + m7

## Frequency Reference

**Concert Pitches**:
- A0: 27.5 Hz
- A4 (concert pitch): 440 Hz
- A8: 7040 Hz

**Frequency Ranges**:
- Sub-bass: 20-60 Hz
- Bass: 60-250 Hz
- Low-mids: 250-500 Hz
- Mids: 500-2 kHz
- High-mids: 2-4 kHz
- Treble: 4-20 kHz
- Human hearing: 20-20,000 Hz

## Need More Help?

- See **THEORY.md** for deeper music theory
- See **API_REFERENCE.md** for technical details
- See **QUICK_REFERENCE.md** for quick lookups
