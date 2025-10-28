"""
Optimized CLI tools for Algorythm - simplified and fast music generation.
"""

import numpy as np
from pathlib import Path


def quick_melody(instrument='pluck', scale='C:major', tempo=120, bars=4, output='output.wav'):
    """Generate a quick melody with one command."""
    from algorythm import SynthPresets, Motif, Scale, Composition, Exporter
    
    # Parse scale
    note, scale_type = scale.split(':') if ':' in scale else ('C', 'major')
    if scale_type == 'major':
        s = Scale.major(note, 4)
    elif scale_type == 'minor':
        s = Scale.minor(note, 4)
    else:
        s = Scale.major(note, 4)
    
    # Get instrument
    preset_map = {
        'pluck': SynthPresets.pluck,
        'bass': SynthPresets.bass,
        'lead': SynthPresets.lead,
        'bell': SynthPresets.bell,
        'pad': SynthPresets.warm_pad,
        'organ': SynthPresets.organ,
        'strings': SynthPresets.strings,
        'guitar': SynthPresets.guitar,
        'brass': SynthPresets.brass,
    }
    inst = preset_map.get(instrument, SynthPresets.pluck)()
    
    # Create ascending melody
    motif = Motif(intervals=[0, 2, 4, 5, 7, 5, 4, 2], scale=s, durations=[0.5] * 8)
    
    # Create composition using method chaining
    comp = Composition(tempo=tempo)
    comp.add_track("Main", inst).repeat_motif(motif, bars=bars)
    
    # Render
    # RenderEngine not needed
    audio = comp.render()
    
    # Export
    exporter = Exporter()
    exporter.export(audio, output, sample_rate=44100)
    
    return len(audio) / 44100  # duration


def quick_beat(tempo=120, bars=4, output='beat.wav'):
    """Generate a quick drum beat."""
    from algorythm import SynthPresets, Motif, Scale, Composition, Exporter
    
    scale = Scale.major('C', 2)
    drum = SynthPresets.drum()
    
    # Create 4/4 beat pattern
    motif = Motif(intervals=[0, 0, 0, 0], scale=scale, durations=[0.5, 0.5, 0.5, 0.5])
    
    comp = Composition(tempo=tempo)
    comp.add_track("Drums", drum).repeat_motif(motif, bars=bars * 4)
    
    # RenderEngine not needed
    audio = comp.render()
    
    exporter = Exporter()
    exporter.export(audio, output, sample_rate=44100)
    
    return len(audio) / 44100


def apply_effect_to_file(input_file, effect_name, output_file=None, **params):
    """Apply a single effect to an audio file."""
    from algorythm.audio_loader import load_audio
    from algorythm.export import Exporter
    from algorythm.effects import (
        Reverb, Delay, Chorus, Distortion, Compressor, Phaser, Tremolo
    )
    
    # Load audio
    audio, sr = load_audio(input_file)
    
    # Create effect
    effects_map = {
        'reverb': lambda: Reverb(
            room_size=params.get('room_size', 0.7),
            wet_level=params.get('wet_level', 0.3)
        ),
        'delay': lambda: Delay(
            delay_time=params.get('delay_time', 0.5),
            feedback=params.get('feedback', 0.5),
            wet_level=params.get('wet_level', 0.5)
        ),
        'chorus': lambda: Chorus(mix=params.get('mix', 0.5)),
        'distortion': lambda: Distortion(
            drive=params.get('drive', 5.0),
            tone=params.get('tone', 0.5)
        ),
        'compressor': lambda: Compressor(
            threshold=params.get('threshold', -20.0),
            ratio=params.get('ratio', 4.0)
        ),
        'phaser': lambda: Phaser(mix=params.get('mix', 0.5)),
        'tremolo': lambda: Tremolo(
            rate=params.get('rate', 5.0),
            depth=params.get('depth', 0.5)
        ),
    }
    
    effect = effects_map.get(effect_name.lower())
    if not effect:
        raise ValueError(f"Unknown effect: {effect_name}")
    
    # Apply
    processed = effect().apply(audio, sr)
    
    # Export
    if not output_file:
        p = Path(input_file)
        output_file = str(p.parent / f"{p.stem}_{effect_name}{p.suffix}")
    
    exporter = Exporter()
    exporter.export(processed, output_file, sr)
    
    return output_file


def list_all_presets():
    """Return list of all available presets."""
    return {
        'basic': ['pluck', 'bass', 'lead', 'pad'],
        'advanced': ['bell', 'organ', 'strings', 'guitar', 'brass', 'drum']
    }


def list_all_effects():
    """Return list of all available effects."""
    return {
        'time': ['reverb', 'delay'],
        'modulation': ['chorus', 'phaser', 'tremolo'],
        'dynamics': ['compressor', 'limiter', 'gate'],
        'distortion': ['distortion', 'overdrive', 'fuzz']
    }


def generate_style(style='ambient', length=30, output='track.wav'):
    """Generate music in a specific style."""
    from algorythm import (
        Composition, Motif, Scale, SynthPresets,
        PhysicalModelSynth, Exporter
    )
    from algorythm.effects import Reverb, Delay, Chorus, Compressor
    
    comp = Composition(tempo=120)
    scale = Scale.major('C', 4)
    
    if style == 'ambient':
        pad = SynthPresets.strings()
        motif = Motif(intervals=[0, 2, 4], scale=scale, durations=[8.0, 8.0, 8.0])
        comp.add_track("Pad", pad).repeat_motif(motif, bars=4).add_fx(Reverb(room_size=0.9, wet_level=0.6))
        comp.tempo = 60
        
    elif style == 'chill':
        guitar = SynthPresets.guitar()
        motif = Motif(intervals=[0, 2, 4, 5, 7, 5, 4, 2], scale=scale, durations=[1.0] * 8)
        comp.add_track("Guitar", guitar).repeat_motif(motif, bars=4).add_fx(Reverb(room_size=0.6, wet_level=0.3))
        comp.tempo = 90
        
    elif style == 'upbeat':
        lead = SynthPresets.lead()
        motif = Motif(intervals=[0, 2, 4, 5, 7, 9, 7, 5, 4, 2, 0], scale=scale, durations=[0.25] * 11)
        comp.add_track("Lead", lead).repeat_motif(motif, bars=4).add_fx(Delay(delay_time=0.25, feedback=0.3, wet_level=0.4))
        comp.tempo = 140
        
    elif style == 'minimal':
        organ = SynthPresets.organ()
        motif = Motif(intervals=[0, 4], scale=scale, durations=[4.0, 4.0])
        comp.add_track("Organ", organ).repeat_motif(motif, bars=4).add_fx(Reverb(room_size=0.7, wet_level=0.4))
        comp.tempo = 100
    
    # Render
    # RenderEngine not needed
    audio = comp.render()
    
    # Loop to fill length
    target_samples = int(length * 44100)
    if len(audio) < target_samples:
        repeats = (target_samples // len(audio)) + 1
        audio = np.tile(audio, repeats)[:target_samples]
    else:
        audio = audio[:target_samples]
    
    # Export
    exporter = Exporter()
    exporter.export(audio, output, sample_rate=44100)
    
    return len(audio) / 44100
