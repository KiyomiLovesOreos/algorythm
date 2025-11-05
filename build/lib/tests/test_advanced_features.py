"""
Tests for advanced features added to Algorythm library.

Tests for:
- Microtonal support (Tuning class)
- Granular synthesis
- Spatial audio
- Constraint-based composition
- Genetic algorithm improviser
- Advanced visualizers (Oscilloscope, Piano Roll)
"""

import unittest
import numpy as np
from algorythm.sequence import Tuning, Scale, Motif
from algorythm.sampler import Sample, GranularSynth
from algorythm.structure import SpatialAudio
from algorythm.generative import ConstraintBasedComposer, GeneticAlgorithmImproviser
from algorythm.visualization import OscilloscopeVisualizer, PianoRollVisualizer


class TestTuning(unittest.TestCase):
    """Test microtonal tuning system."""
    
    def test_tuning_creation(self):
        """Test creation of tuning systems."""
        tuning = Tuning('12-TET')
        self.assertEqual(tuning.tones_per_octave, 12)
        self.assertEqual(tuning.name, '12-TET')
    
    def test_equal_temperament(self):
        """Test equal temperament creation."""
        tuning = Tuning.equal_temperament(19)
        self.assertEqual(tuning.tones_per_octave, 19)
        self.assertAlmostEqual(tuning.cents[1], 1200 / 19, places=2)
    
    def test_just_intonation(self):
        """Test just intonation tuning."""
        tuning = Tuning.just_intonation()
        self.assertEqual(tuning.name, 'just_intonation')
        self.assertEqual(tuning.tones_per_octave, 12)
    
    def test_pythagorean(self):
        """Test Pythagorean tuning."""
        tuning = Tuning.pythagorean()
        self.assertEqual(tuning.name, 'pythagorean')
    
    def test_get_frequency(self):
        """Test frequency calculation from scale degrees."""
        tuning = Tuning('12-TET')
        # A4 = 440 Hz at degree 0 (with default reference)
        freq = tuning.get_frequency(0)
        self.assertAlmostEqual(freq, 440.0, places=1)
    
    def test_midi_to_frequency(self):
        """Test MIDI note to frequency conversion."""
        tuning = Tuning('12-TET')
        # A4 = MIDI 69 = 440 Hz
        freq = tuning.midi_to_frequency(69)
        self.assertAlmostEqual(freq, 440.0, places=1)
    
    def test_custom_tuning(self):
        """Test custom tuning system."""
        custom_cents = [0, 100, 200, 300, 400, 500, 600]
        tuning = Tuning(custom_cents)
        self.assertEqual(tuning.name, 'custom')
        self.assertEqual(tuning.tones_per_octave, 7)


class TestScaleWithTuning(unittest.TestCase):
    """Test Scale with microtonal tuning support."""
    
    def test_scale_with_custom_tuning(self):
        """Test scale creation with custom tuning."""
        tuning = Tuning.equal_temperament(19)
        scale = Scale('C', 'major', 4, tuning=tuning)
        self.assertIsNotNone(scale.tuning)
        self.assertEqual(scale.tuning.tones_per_octave, 19)
    
    def test_scale_frequency_with_tuning(self):
        """Test frequency calculation with custom tuning."""
        tuning = Tuning('12-TET')
        scale = Scale('A', 'major', 4, tuning=tuning)
        freq = scale.get_frequency(0)
        # Should use tuning system for conversion
        self.assertGreater(freq, 0)


class TestGranularSynth(unittest.TestCase):
    """Test granular synthesis."""
    
    def test_granular_synth_creation(self):
        """Test granular synth initialization."""
        sample = Sample(data=np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)))
        granular = GranularSynth(sample, grain_size=0.05, grain_density=20.0)
        self.assertEqual(granular.grain_size, 0.05)
        self.assertEqual(granular.grain_density, 20.0)
    
    def test_generate_grain(self):
        """Test single grain generation."""
        sample = Sample(data=np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)))
        granular = GranularSynth(sample)
        grain = granular.generate_grain(position=0.5, size=0.05)
        self.assertGreater(len(grain), 0)
    
    def test_synthesize(self):
        """Test granular synthesis over duration."""
        sample = Sample(data=np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)))
        granular = GranularSynth(sample, grain_size=0.02, grain_density=50.0)
        output = granular.synthesize(duration=0.5)
        self.assertGreater(len(output), 0)
        self.assertLessEqual(np.max(np.abs(output)), 1.0)
    
    def test_synthesize_with_pitch_range(self):
        """Test granular synthesis with pitch shifting."""
        sample = Sample(data=np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)))
        granular = GranularSynth(sample)
        output = granular.synthesize(
            duration=0.3,
            pitch_range=(-12.0, 12.0)
        )
        self.assertGreater(len(output), 0)
    
    def test_envelope_types(self):
        """Test different grain envelope types."""
        sample = Sample(data=np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)))
        
        for envelope_type in ['rectangular', 'triangular', 'gaussian', 'hann']:
            granular = GranularSynth(sample, grain_envelope=envelope_type)
            output = granular.synthesize(duration=0.2)
            self.assertGreater(len(output), 0)


class TestSpatialAudio(unittest.TestCase):
    """Test spatial audio positioning."""
    
    def test_spatial_audio_creation(self):
        """Test spatial audio initialization."""
        spatial = SpatialAudio(position=(1.0, 0.0, 0.0))
        self.assertTrue(np.allclose(spatial.position, [1.0, 0.0, 0.0]))
    
    def test_set_position(self):
        """Test setting 3D position."""
        spatial = SpatialAudio()
        spatial.set_position(2.0, 1.0, -1.0)
        self.assertTrue(np.allclose(spatial.position, [2.0, 1.0, -1.0]))
    
    def test_set_listener_position(self):
        """Test setting listener position."""
        spatial = SpatialAudio()
        spatial.set_listener_position(1.0, 0.0, 0.0)
        self.assertTrue(np.allclose(spatial.listener_position, [1.0, 0.0, 0.0]))
    
    def test_calculate_distance(self):
        """Test distance calculation."""
        spatial = SpatialAudio(position=(3.0, 4.0, 0.0))
        distance = spatial.calculate_distance()
        self.assertAlmostEqual(distance, 5.0, places=5)
    
    def test_calculate_pan(self):
        """Test stereo pan calculation."""
        spatial = SpatialAudio(position=(1.0, 0.0, 0.0))
        pan = spatial.calculate_pan()
        self.assertGreater(pan, 0)  # Right of center
        
        spatial.set_position(-1.0, 0.0, 0.0)
        pan = spatial.calculate_pan()
        self.assertLess(pan, 0)  # Left of center
    
    def test_calculate_attenuation(self):
        """Test distance attenuation."""
        spatial = SpatialAudio(position=(2.0, 0.0, 0.0))
        attenuation = spatial.calculate_attenuation()
        self.assertGreater(attenuation, 0)
        self.assertLessEqual(attenuation, 1.0)
    
    def test_apply_spatial_audio(self):
        """Test spatial audio processing."""
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4410))
        spatial = SpatialAudio(position=(1.0, 0.0, 0.0))
        stereo = spatial.apply(signal)
        
        # Should produce stereo output
        self.assertEqual(stereo.shape[0], 2)
        self.assertEqual(stereo.shape[1], len(signal))
    
    def test_distance_models(self):
        """Test different distance attenuation models."""
        for model in ['linear', 'inverse', 'exponential']:
            spatial = SpatialAudio(
                position=(5.0, 0.0, 0.0),
                distance_model=model
            )
            attenuation = spatial.calculate_attenuation()
            self.assertGreater(attenuation, 0)


class TestConstraintBasedComposer(unittest.TestCase):
    """Test constraint-based composition."""
    
    def test_composer_creation(self):
        """Test CBC initialization."""
        composer = ConstraintBasedComposer()
        self.assertIsNotNone(composer.scale)
        self.assertEqual(len(composer.constraints), 0)
    
    def test_add_constraint(self):
        """Test adding custom constraint."""
        composer = ConstraintBasedComposer()
        
        def test_constraint(melody):
            return len(melody) > 0
        
        composer.add_constraint(test_constraint)
        self.assertEqual(len(composer.constraints), 1)
    
    def test_no_large_leaps(self):
        """Test no large leaps constraint."""
        composer = ConstraintBasedComposer()
        composer.no_large_leaps(max_interval=3)
        
        # Valid melody
        self.assertTrue(composer.check_constraints([0, 1, 2, 3, 4]))
        # Invalid melody (large leap)
        self.assertFalse(composer.check_constraints([0, 7, 1]))
    
    def test_prefer_stepwise_motion(self):
        """Test stepwise motion constraint."""
        composer = ConstraintBasedComposer()
        composer.prefer_stepwise_motion()
        
        # Mostly stepwise
        self.assertTrue(composer.check_constraints([0, 1, 2, 1, 0, 1]))
        # Too many leaps
        self.assertFalse(composer.check_constraints([0, 5, 10, 3, 8]))
    
    def test_no_repeated_notes(self):
        """Test no repeated notes constraint."""
        composer = ConstraintBasedComposer()
        composer.no_repeated_notes()
        
        self.assertTrue(composer.check_constraints([0, 1, 2, 3]))
        self.assertFalse(composer.check_constraints([0, 1, 1, 2]))
    
    def test_ending_on_tonic(self):
        """Test ending on tonic constraint."""
        composer = ConstraintBasedComposer()
        composer.ending_on_tonic()
        
        self.assertTrue(composer.check_constraints([1, 2, 3, 0]))
        self.assertFalse(composer.check_constraints([0, 1, 2, 3]))
    
    def test_generate(self):
        """Test melody generation with constraints."""
        composer = ConstraintBasedComposer()
        composer.no_large_leaps(max_interval=4)
        composer.ending_on_tonic()
        
        motif = composer.generate(length=8, max_attempts=1000)
        # Should find a solution
        if motif is not None:
            self.assertIsInstance(motif, Motif)
            self.assertGreater(len(motif.intervals), 0)


class TestGeneticAlgorithmImproviser(unittest.TestCase):
    """Test genetic algorithm improviser."""
    
    def test_ga_creation(self):
        """Test GA initialization."""
        def fitness(melody):
            return len(melody)
        
        ga = GeneticAlgorithmImproviser(fitness)
        self.assertEqual(ga.population_size, 50)
        self.assertEqual(ga.mutation_rate, 0.1)
    
    def test_initialize_population(self):
        """Test population initialization."""
        def fitness(melody):
            return sum(melody)
        
        ga = GeneticAlgorithmImproviser(fitness, population_size=20)
        ga.initialize_population(length=8)
        
        self.assertEqual(len(ga.population), 20)
        self.assertEqual(len(ga.population[0]), 8)
    
    def test_evaluate_fitness(self):
        """Test fitness evaluation."""
        def fitness(melody):
            return len([x for x in melody if x > 0])
        
        ga = GeneticAlgorithmImproviser(fitness)
        score = ga.evaluate_fitness([1, 2, -1, 3, -2])
        self.assertEqual(score, 3)
    
    def test_evolve(self):
        """Test evolution process."""
        def fitness(melody):
            # Prefer ascending melodies
            return sum(1 for i in range(len(melody) - 1) if melody[i + 1] > melody[i])
        
        ga = GeneticAlgorithmImproviser(fitness, population_size=20)
        ga.initialize_population(length=6)
        
        motif = ga.evolve(generations=10)
        self.assertIsInstance(motif, Motif)
        self.assertGreater(len(motif.intervals), 0)
    
    def test_fitness_ascending(self):
        """Test ascending fitness function."""
        fitness = GeneticAlgorithmImproviser.fitness_ascending()
        
        ascending = [0, 1, 2, 3, 4]
        descending = [4, 3, 2, 1, 0]
        
        self.assertGreater(fitness(ascending), fitness(descending))
    
    def test_fitness_contour(self):
        """Test contour matching fitness function."""
        target = [1, 1, -1, 1]  # Up, up, down, up
        fitness = GeneticAlgorithmImproviser.fitness_contour(target)
        
        matching = [0, 1, 2, 1, 3]  # Matches contour
        not_matching = [0, -1, -2, -3, -4]  # Doesn't match
        
        self.assertGreater(fitness(matching), fitness(not_matching))


class TestOscilloscopeVisualizer(unittest.TestCase):
    """Test oscilloscope visualizer."""
    
    def test_oscilloscope_creation(self):
        """Test oscilloscope initialization."""
        viz = OscilloscopeVisualizer()
        self.assertEqual(viz.mode, 'waveform')
        self.assertEqual(viz.window_size, 1024)
    
    def test_waveform_mode(self):
        """Test waveform display mode."""
        viz = OscilloscopeVisualizer(mode='waveform')
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4410))
        data = viz.generate(signal)
        self.assertEqual(len(data), viz.window_size)
    
    def test_lissajous_mode(self):
        """Test Lissajous curve mode."""
        viz = OscilloscopeVisualizer(mode='lissajous')
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4410))
        data = viz.generate(signal)
        self.assertGreater(len(data), 0)
    
    def test_phase_mode(self):
        """Test phase correlation mode."""
        viz = OscilloscopeVisualizer(mode='phase')
        # Stereo signal
        signal = np.vstack([
            np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4410)),
            np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4410))
        ])
        data = viz.generate(signal)
        self.assertGreater(len(data), 0)
    
    def test_to_image_data(self):
        """Test image conversion."""
        viz = OscilloscopeVisualizer()
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4410))
        image = viz.to_image_data(signal, height=256, width=512)
        self.assertEqual(image.shape, (256, 512))


class TestPianoRollVisualizer(unittest.TestCase):
    """Test piano roll visualizer."""
    
    def test_piano_roll_creation(self):
        """Test piano roll initialization."""
        viz = PianoRollVisualizer()
        self.assertEqual(viz.time_resolution, 0.1)
        self.assertEqual(viz.pitch_range, (36, 84))
    
    def test_add_note(self):
        """Test adding notes to piano roll."""
        viz = PianoRollVisualizer()
        viz.add_note(midi_note=60, start_time=0.0, duration=0.5)
        self.assertEqual(len(viz.notes), 1)
    
    def test_clear_notes(self):
        """Test clearing notes."""
        viz = PianoRollVisualizer()
        viz.add_note(60, 0.0, 0.5)
        viz.clear_notes()
        self.assertEqual(len(viz.notes), 0)
    
    def test_to_grid(self):
        """Test grid generation."""
        viz = PianoRollVisualizer()
        viz.add_note(midi_note=60, start_time=0.0, duration=0.5)
        viz.add_note(midi_note=64, start_time=0.5, duration=0.5)
        
        grid = viz.to_grid(duration=1.0)
        self.assertGreater(grid.shape[0], 0)
        self.assertGreater(grid.shape[1], 0)
        # Should have some active cells
        self.assertGreater(np.sum(grid), 0)
    
    def test_to_image_data(self):
        """Test image conversion."""
        viz = PianoRollVisualizer()
        viz.add_note(midi_note=60, start_time=0.0, duration=0.5)
        
        image = viz.to_image_data(duration=1.0, height=480, width=640)
        self.assertEqual(image.shape[0], 480)
        self.assertEqual(image.shape[1], 640)


if __name__ == '__main__':
    unittest.main()
