"""
Synthesia: A Python Library for Algorithmic Music

A declarative, Manim-inspired library for generating algorithmic music.
"""

__version__ = '0.1.0'

from audionaut.synth import Synth, Oscillator, Filter, ADSR, SynthPresets
from audionaut.sequence import Motif, Rhythm, Arpeggiator, Scale
from audionaut.structure import Track, Composition, EffectChain, Reverb
from audionaut.export import RenderEngine, Exporter

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
