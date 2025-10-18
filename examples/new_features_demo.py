"""
New Features Demo for Algorythm Library

Demonstrates the newly implemented advanced capabilities:
- Microtonal tuning systems (19-TET, just intonation, etc.)
- Granular synthesis for texture generation
- 3D spatial audio positioning
- Constraint-based composition
- Genetic algorithm improviser
- Advanced visualizations (oscilloscope, piano roll)
"""

import numpy as np
from algorythm import (
    # Core
    Synth, Scale, Motif, Composition,
    # New: Microtonal
    Tuning,
    # New: Granular synthesis
    Sample, GranularSynth,
    # New: Spatial audio
    SpatialAudio,
    # New: Algorithmic composition
    ConstraintBasedComposer, GeneticAlgorithmImproviser,
    # New: Advanced visualization
    OscilloscopeVisualizer, PianoRollVisualizer
)


def demo_microtonal_support():
    """Demonstrate microtonal tuning systems."""
    print("\n=== Microtonal Support Demo ===")
    
    # Create a 19-tone equal temperament tuning
    tuning_19tet = Tuning.equal_temperament(19)
    print(f"Created {tuning_19tet.tones_per_octave}-tone equal temperament")
    
    # Create a scale using the microtonal tuning
    scale = Scale.major('C', octave=4, tuning=tuning_19tet)
    
    # Generate frequencies - these will be microtonal!
    freqs = [scale.get_frequency(i) for i in range(7)]
    print(f"First 7 frequencies in 19-TET C major: {[f'{f:.2f}' for f in freqs]}")
    
    # Compare with standard 12-TET
    scale_12tet = Scale.major('C', octave=4)
    freqs_12tet = [scale_12tet.get_frequency(i) for i in range(7)]
    print(f"First 7 frequencies in 12-TET C major: {[f'{f:.2f}' for f in freqs_12tet]}")
    
    # Try just intonation
    tuning_ji = Tuning.just_intonation()
    scale_ji = Scale.major('C', octave=4, tuning=tuning_ji)
    freqs_ji = [scale_ji.get_frequency(i) for i in range(7)]
    print(f"First 7 frequencies in just intonation C major: {[f'{f:.2f}' for f in freqs_ji]}")
    
    print("✓ Microtonal tuning systems working!")


def demo_granular_synthesis():
    """Demonstrate granular synthesis."""
    print("\n=== Granular Synthesis Demo ===")
    
    # Create a simple audio sample (sine wave)
    duration = 1.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * 440 * t)
    
    sample = Sample(data=audio_data, sample_rate=sample_rate)
    print(f"Created sample: {len(sample.data)} samples at {sample_rate} Hz")
    
    # Create granular synth
    granular = GranularSynth(
        sample,
        grain_size=0.05,  # 50ms grains
        grain_density=20.0,  # 20 grains per second
        grain_envelope='hann'
    )
    print(f"Initialized granular synth with {granular.grain_size}s grains, density {granular.grain_density}/s")
    
    # Synthesize granular texture
    output = granular.synthesize(
        duration=2.0,
        position_range=(0.2, 0.8),  # Use middle 60% of sample
        pitch_range=(-12.0, 12.0),  # Vary pitch by ±1 octave
        spatial_spread=0.5,  # Some stereo spread
        density_variation=0.3  # 30% density variation
    )
    
    print(f"Generated {len(output)} samples of granular audio")
    print(f"Peak amplitude: {np.max(np.abs(output)):.3f}")
    print("✓ Granular synthesis working!")


def demo_spatial_audio():
    """Demonstrate 3D spatial audio positioning."""
    print("\n=== Spatial Audio Demo ===")
    
    # Create a mono audio signal
    duration = 0.5
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.sin(2 * np.pi * 440 * t)
    
    # Create spatial audio processor
    spatial = SpatialAudio(
        position=(2.0, 1.0, 0.0),
        listener_position=(0.0, 0.0, 0.0),
        distance_model='inverse'
    )
    
    # Calculate spatial properties
    distance = spatial.calculate_distance()
    pan = spatial.calculate_pan()
    attenuation = spatial.calculate_attenuation()
    
    print(f"Sound source at: {spatial.position}")
    print(f"Listener at: {spatial.listener_position}")
    print(f"Distance: {distance:.2f}")
    print(f"Pan: {pan:.2f} (-1=left, 0=center, 1=right)")
    print(f"Attenuation: {attenuation:.2f}")
    
    # Apply spatial processing
    stereo = spatial.apply(signal, sample_rate)
    print(f"Generated stereo output: {stereo.shape}")
    print(f"Left channel RMS: {np.sqrt(np.mean(stereo[0]**2)):.3f}")
    print(f"Right channel RMS: {np.sqrt(np.mean(stereo[1]**2)):.3f}")
    
    # Move sound source
    spatial.set_position(x=-3.0, y=0.0, z=0.0)
    print(f"\nMoved sound to: {spatial.position}")
    print(f"New pan: {spatial.calculate_pan():.2f}")
    
    print("✓ Spatial audio positioning working!")


def demo_constraint_based_composition():
    """Demonstrate constraint-based composition."""
    print("\n=== Constraint-Based Composition Demo ===")
    
    # Create composer with constraints
    composer = ConstraintBasedComposer(scale=Scale.major('C', 4))
    
    # Add musical constraints
    composer.no_large_leaps(max_interval=4)
    composer.prefer_stepwise_motion()
    composer.ending_on_tonic()
    
    print("Constraints added:")
    print("- No melodic leaps larger than 4 semitones")
    print("- At least 70% stepwise motion")
    print("- Must end on tonic (scale degree 0)")
    
    # Generate melody
    motif = composer.generate(length=8, max_attempts=1000)
    
    if motif:
        print(f"\nGenerated melody: {motif.intervals}")
        print(f"Frequencies: {[f'{f:.1f}' for f in motif.get_frequencies()]}")
        
        # Verify constraints
        is_valid = composer.check_constraints(motif.intervals)
        print(f"Satisfies all constraints: {is_valid}")
        print("✓ Constraint-based composition working!")
    else:
        print("Could not find solution within max_attempts")


def demo_genetic_algorithm():
    """Demonstrate genetic algorithm improviser."""
    print("\n=== Genetic Algorithm Improviser Demo ===")
    
    # Define fitness function (prefer ascending melodies)
    print("Fitness function: prefer ascending melodies")
    fitness = GeneticAlgorithmImproviser.fitness_ascending()
    
    # Create GA improviser
    ga = GeneticAlgorithmImproviser(
        fitness,
        scale=Scale.minor('A', 4),
        population_size=30,
        mutation_rate=0.15,
        crossover_rate=0.7
    )
    
    print(f"Population size: {ga.population_size}")
    print(f"Mutation rate: {ga.mutation_rate}")
    print(f"Crossover rate: {ga.crossover_rate}")
    
    # Initialize population
    ga.initialize_population(length=8, range_min=-7, range_max=8)
    print(f"Initialized population with {len(ga.population)} individuals")
    
    # Evolve for multiple generations
    initial_fitness = ga.evaluate_fitness(ga.population[0])
    print(f"Initial best fitness: {initial_fitness:.3f}")
    
    motif = ga.evolve(generations=50)
    
    final_fitness = ga.evaluate_fitness(motif.intervals)
    print(f"Final best fitness: {final_fitness:.3f}")
    print(f"Improvement: {final_fitness - initial_fitness:.3f}")
    print(f"\nEvolved melody: {motif.intervals}")
    print(f"Frequencies: {[f'{f:.1f}' for f in motif.get_frequencies()]}")
    print("✓ Genetic algorithm improviser working!")


def demo_oscilloscope_visualizer():
    """Demonstrate oscilloscope visualization."""
    print("\n=== Oscilloscope Visualizer Demo ===")
    
    # Create test signals
    sample_rate = 44100
    duration = 0.1
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Mono signal
    mono = np.sin(2 * np.pi * 440 * t)
    
    # Stereo signal with phase difference
    stereo = np.vstack([
        np.sin(2 * np.pi * 440 * t),
        np.sin(2 * np.pi * 440 * t + np.pi / 4)  # 45° phase shift
    ])
    
    # Waveform mode
    viz_wave = OscilloscopeVisualizer(mode='waveform', window_size=1024)
    waveform = viz_wave.generate(mono)
    print(f"Waveform mode: generated {len(waveform)} samples")
    
    # Lissajous mode (X-Y plot)
    viz_liss = OscilloscopeVisualizer(mode='lissajous', window_size=512)
    lissajous = viz_liss.generate(stereo)
    print(f"Lissajous mode: generated {lissajous.shape} data")
    
    # Phase correlation mode
    viz_phase = OscilloscopeVisualizer(mode='phase', window_size=1024)
    phase = viz_phase.generate(stereo)
    print(f"Phase mode: generated {len(phase)} correlation values")
    
    # Convert to image
    image = viz_wave.to_image_data(mono, height=256, width=512)
    print(f"Generated image: {image.shape}")
    print("✓ Oscilloscope visualizer working!")


def demo_piano_roll_visualizer():
    """Demonstrate piano roll visualization."""
    print("\n=== Piano Roll Visualizer Demo ===")
    
    # Create piano roll
    viz = PianoRollVisualizer(
        time_resolution=0.1,  # 100ms per column
        pitch_range=(48, 72)  # C3 to C5
    )
    
    # Add some notes (C major scale)
    notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C4 to C5
    for i, midi_note in enumerate(notes):
        viz.add_note(
            midi_note=midi_note,
            start_time=i * 0.25,
            duration=0.25,
            velocity=0.8
        )
    
    print(f"Added {len(viz.notes)} notes to piano roll")
    
    # Generate grid
    duration = 2.0
    grid = viz.to_grid(duration=duration)
    print(f"Generated grid: {grid.shape} (height x width)")
    print(f"Active cells: {np.sum(grid > 0)}")
    
    # Convert to image
    image = viz.to_image_data(duration=duration, height=240, width=640)
    print(f"Generated image: {image.shape}")
    print("✓ Piano roll visualizer working!")


def main():
    """Run all advanced feature demonstrations."""
    print("=" * 60)
    print("Algorythm New Features Demonstration")
    print("=" * 60)
    
    # Run all demos
    demo_microtonal_support()
    demo_granular_synthesis()
    demo_spatial_audio()
    demo_constraint_based_composition()
    demo_genetic_algorithm()
    demo_oscilloscope_visualizer()
    demo_piano_roll_visualizer()
    
    print("\n" + "=" * 60)
    print("All new features demonstrated successfully! ✓")
    print("=" * 60)


if __name__ == '__main__':
    main()
