"""
Algorythm: A Python Library for Algorithmic Music

A declarative, Manim-inspired library for generating algorithmic music.
"""

__version__ = '0.4.0'

from algorythm.synth import (
    Synth, Oscillator, Filter, ADSR, SynthPresets, 
    FMSynth, WavetableSynth, PhysicalModelSynth, 
    AdditiveeSynth, PadSynth
)
from algorythm.sequence import Motif, Rhythm, Arpeggiator, Scale, Chord, Tuning
from algorythm.structure import (
    Track, Composition, EffectChain, 
    Reverb, Delay, Chorus, Flanger, Distortion, Compression, 
    EQ, Phaser, Tremolo, Bitcrusher, SpatialAudio, VolumeControl
)
from algorythm.effects import (
    Effect, Reverb as ReverbFX, Delay as DelayFX, 
    Chorus as ChorusFX, Flanger as FlangerFX, Phaser as PhaserFX,
    Distortion as DistortionFX, Overdrive, Fuzz,
    Compressor, Limiter, Gate,
    Tremolo as TremoloFX, Vibrato, BitCrusher as BitCrusherFX,
    AutoPan, RingModulator, EffectChain as FXChain
)
from algorythm.export import RenderEngine, Exporter
from algorythm.audio_loader import load_audio, visualize_audio_file, AudioFile
from algorythm.generative import LSystem, CellularAutomata, ConstraintBasedComposer, GeneticAlgorithmImproviser
from algorythm.automation import Automation, AutomationTrack, DataSonification
from algorythm.visualization import WaveformVisualizer, SpectrogramVisualizer, FrequencyScopeVisualizer, VideoRenderer, OscilloscopeVisualizer, PianoRollVisualizer
from algorythm.sampler import Sample, Sampler, GranularSynth

# Optional imports (require additional dependencies)
try:
    from algorythm.playback import AudioPlayer, StreamingPlayer, LiveCompositionPlayer
    _PLAYBACK_AVAILABLE = True
except ImportError:
    _PLAYBACK_AVAILABLE = False

try:
    from algorythm.live_gui import LiveCodingGUI, launch
    _GUI_AVAILABLE = True
except ImportError:
    _GUI_AVAILABLE = False

__all__ = [
    # Synthesis
    'Synth',
    'Oscillator',
    'Filter',
    'ADSR',
    'SynthPresets',
    'FMSynth',
    'WavetableSynth',
    'PhysicalModelSynth',
    'AdditiveeSynth',
    'PadSynth',
    # Sequence
    'Motif',
    'Rhythm',
    'Arpeggiator',
    'Scale',
    'Chord',
    'Tuning',
    # Structure
    'Track',
    'Composition',
    'EffectChain',
    'Reverb',
    'Delay',
    'Chorus',
    'Flanger',
    'Distortion',
    'Compression',
    'EQ',
    'Phaser',
    'Tremolo',
    'Bitcrusher',
    'SpatialAudio',
    'VolumeControl',
    # Effects (new module)
    'Effect',
    'ReverbFX',
    'DelayFX',
    'ChorusFX',
    'FlangerFX',
    'PhaserFX',
    'DistortionFX',
    'Overdrive',
    'Fuzz',
    'Compressor',
    'Limiter',
    'Gate',
    'TremoloFX',
    'Vibrato',
    'BitCrusherFX',
    'AutoPan',
    'RingModulator',
    'FXChain',
    # Export
    'RenderEngine',
    'Exporter',
    # Audio Loading
    'load_audio',
    'visualize_audio_file',
    'AudioFile',
    # Generative
    'LSystem',
    'CellularAutomata',
    'ConstraintBasedComposer',
    'GeneticAlgorithmImproviser',
    # Automation
    'Automation',
    'AutomationTrack',
    'DataSonification',
    # Visualization
    'WaveformVisualizer',
    'SpectrogramVisualizer',
    'FrequencyScopeVisualizer',
    'VideoRenderer',
    'OscilloscopeVisualizer',
    'PianoRollVisualizer',
    # Sampler
    'Sample',
    'Sampler',
    'GranularSynth',
]

# Add playback to __all__ if available
if _PLAYBACK_AVAILABLE:
    __all__.extend(['AudioPlayer', 'StreamingPlayer', 'LiveCompositionPlayer'])

# Add GUI to __all__ if available
if _GUI_AVAILABLE:
    __all__.extend(['LiveCodingGUI', 'launch'])
