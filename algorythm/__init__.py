"""
Algorythm: A Python Library for Algorithmic Music

Manim-inspired library for generating algorithmic music.
"""

__version__ = "0.4.0"

from algorythm.audio_loader import AudioFile, load_audio, visualize_audio_file
from algorythm.automation import Automation, AutomationTrack, DataSonification
from algorythm.effects import (
    AutoPan,
    BeatRepeat,
    Compressor,
    Effect,
    FilterSweep,
    Freeze,
    Fuzz,
    Gate,
    Limiter,
    Overdrive,
    Reverse,
    RingModulator,
    Stutter,
    Vibrato,
)
from algorythm.effects import BitCrusher as BitCrusherFX
from algorythm.effects import Chorus as ChorusFX
from algorythm.effects import Delay as DelayFX
from algorythm.effects import Distortion as DistortionFX
from algorythm.effects import EffectChain as FXChain
from algorythm.effects import Flanger as FlangerFX
from algorythm.effects import Phaser as PhaserFX
from algorythm.effects import Reverb as ReverbFX
from algorythm.effects import Tremolo as TremoloFX
from algorythm.export import Exporter, RenderEngine
from algorythm.generative import (
    CellularAutomata,
    ConstraintBasedComposer,
    GeneticAlgorithmImproviser,
    LSystem,
)
from algorythm.sampler import GranularSynth, Sample, Sampler
from algorythm.sequence import Arpeggiator, Chord, Motif, Rhythm, Scale, Tuning
from algorythm.structure import (
    EQ,
    Bitcrusher,
    Chorus,
    Composition,
    CompositionSet,
    Compression,
    Delay,
    Distortion,
    EffectChain,
    Flanger,
    Phaser,
    Reverb,
    SpatialAudio,
    Track,
    Tremolo,
    VolumeControl,
)
from algorythm.synth import (
    ADSR,
    AdditiveeSynth,
    Filter,
    FMSynth,
    Oscillator,
    PadSynth,
    PhysicalModelSynth,
    Synth,
    SynthPresets,
    WavetableSynth,
)
from algorythm.visualization import (
    FrequencyScopeVisualizer,
    OscilloscopeVisualizer,
    PianoRollVisualizer,
    SpectrogramVisualizer,
    VideoRenderer,
    WaveformVisualizer,
)

# Optional imports (require additional dependencies)
try:
    from algorythm.playback import AudioPlayer, LiveCompositionPlayer, StreamingPlayer

    _PLAYBACK_AVAILABLE = True
except ImportError:
    _PLAYBACK_AVAILABLE = False


__all__ = [
    # Synthesis
    "Synth",
    "Oscillator",
    "Filter",
    "ADSR",
    "SynthPresets",
    "FMSynth",
    "WavetableSynth",
    "PhysicalModelSynth",
    "AdditiveeSynth",
    "PadSynth",
    # Sequence
    "Motif",
    "Rhythm",
    "Arpeggiator",
    "Scale",
    "Chord",
    "Tuning",
    # Structure
    "Track",
    "Composition",
    "CompositionSet",
    "EffectChain",
    "Reverb",
    "Delay",
    "Chorus",
    "Flanger",
    "Distortion",
    "Compression",
    "EQ",
    "Phaser",
    "Tremolo",
    "Bitcrusher",
    "SpatialAudio",
    "VolumeControl",
    # Effects (new module)
    "Effect",
    "ReverbFX",
    "DelayFX",
    "ChorusFX",
    "FlangerFX",
    "PhaserFX",
    "DistortionFX",
    "Overdrive",
    "Fuzz",
    "Compressor",
    "Limiter",
    "Gate",
    "TremoloFX",
    "Vibrato",
    "BitCrusherFX",
    "AutoPan",
    "RingModulator",
    "FXChain",
    "Stutter",
    "FilterSweep",
    "Freeze",
    "Reverse",
    "BeatRepeat",
    # Export
    "RenderEngine",
    "Exporter",
    # Audio Loading
    "load_audio",
    "visualize_audio_file",
    "AudioFile",
    # Generative
    "LSystem",
    "CellularAutomata",
    "ConstraintBasedComposer",
    "GeneticAlgorithmImproviser",
    # Automation
    "Automation",
    "AutomationTrack",
    "DataSonification",
    # Visualization
    "WaveformVisualizer",
    "SpectrogramVisualizer",
    "FrequencyScopeVisualizer",
    "VideoRenderer",
    "OscilloscopeVisualizer",
    "PianoRollVisualizer",
    # Sampler
    "Sample",
    "Sampler",
    "GranularSynth",
]

# Add playback to __all__ if available
if _PLAYBACK_AVAILABLE:
    __all__.extend(["AudioPlayer", "StreamingPlayer", "LiveCompositionPlayer"])
