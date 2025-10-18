"""
Algorythm: A Python Library for Algorithmic Music

A declarative, Manim-inspired library for generating algorithmic music.
"""

__version__ = '0.1.0'

from algorythm.synth import Synth, Oscillator, Filter, ADSR, SynthPresets
from algorythm.sequence import Motif, Rhythm, Arpeggiator, Scale
from algorythm.structure import Track, Composition, EffectChain, Reverb
from algorythm.export import RenderEngine, Exporter

__all__ = [
    'Synth',
    'Oscillator',
    'Filter',
    'ADSR',
    'SynthPresets',
    'Motif',
    'Rhythm',
    'Arpeggiator',
    'Scale',
    'Track',
    'Composition',
    'EffectChain',
    'Reverb',
    'RenderEngine',
    'Exporter',
]
