"""
Algorythm: A Python Library for Algorithmic Music

A declarative, Manim-inspired library for generating algorithmic music.
"""

__version__ = '0.1.0'

from algorythm.synth import Synth, Oscillator, Filter, ADSR, SynthPresets
from algorythm.sequence import Motif, Rhythm, Arpeggiator, Scale, Chord
from algorythm.structure import Track, Composition, EffectChain, Reverb, Delay, Chorus, Flanger, Distortion, Compression
from algorythm.export import RenderEngine, Exporter
from algorythm.generative import LSystem, CellularAutomata
from algorythm.automation import Automation, AutomationTrack, DataSonification
from algorythm.visualization import WaveformVisualizer, SpectrogramVisualizer, FrequencyScopeVisualizer, VideoRenderer
from algorythm.sampler import Sample, Sampler

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
    # Export
    'RenderEngine',
    'Exporter',
    # Generative
    'LSystem',
    'CellularAutomata',
    # Automation
    'Automation',
    'AutomationTrack',
    'DataSonification',
    # Visualization
    'WaveformVisualizer',
    'SpectrogramVisualizer',
    'FrequencyScopeVisualizer',
    'VideoRenderer',
    # Sampler
    'Sample',
    'Sampler',
]
