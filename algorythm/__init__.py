"""
Algorythm: A Python Library for Algorithmic Music

A declarative, Manim-inspired library for generating algorithmic music.
"""

__version__ = '0.1.0'

from algorythm.synth import Synth, Oscillator, Filter, ADSR, SynthPresets
from algorythm.sequence import Motif, Rhythm, Arpeggiator, Scale, Chord, Tuning
from algorythm.structure import Track, Composition, EffectChain, Reverb, Delay, Chorus, Flanger, Distortion, Compression, SpatialAudio
from algorythm.export import RenderEngine, Exporter
from algorythm.generative import LSystem, CellularAutomata, ConstraintBasedComposer, GeneticAlgorithmImproviser
from algorythm.automation import Automation, AutomationTrack, DataSonification
from algorythm.visualization import WaveformVisualizer, SpectrogramVisualizer, FrequencyScopeVisualizer, VideoRenderer, OscilloscopeVisualizer, PianoRollVisualizer
from algorythm.sampler import Sample, Sampler, GranularSynth

__all__ = [
    # Synthesis
    'Synth',
    'Oscillator',
    'Filter',
    'ADSR',
    'SynthPresets',
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
    'SpatialAudio',
    # Export
    'RenderEngine',
    'Exporter',
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
