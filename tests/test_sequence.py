"""Tests for audionaut.sequence module."""

import pytest
from audionaut.sequence import Scale, Motif, Rhythm, Arpeggiator


class TestScale:
    """Tests for Scale class."""
    
    def test_scale_creation(self):
        """Test creating a scale."""
        scale = Scale.major('C', octave=4)
        assert scale.root == 'C'
        assert scale.scale_type == 'major'
        assert scale.octave == 4
    
    def test_scale_types(self):
        """Test different scale types."""
        scales = [
            Scale.major('C'),
            Scale.minor('A'),
            Scale.pentatonic_major('G'),
            Scale.pentatonic_minor('E'),
            Scale.blues('D'),
        ]
        for scale in scales:
            assert scale.root in Scale.NOTE_NAMES
    
    def test_get_note(self):
        """Test getting notes from scale."""
        scale = Scale.major('C', octave=4)
        # Get first note (C4)
        note = scale.get_note(0)
        assert isinstance(note, int)
    
    def test_get_frequency(self):
        """Test getting frequencies from scale."""
        scale = Scale.major('A', octave=4)
        freq = scale.get_frequency(0)
        # A4 should be 440 Hz
        assert abs(freq - 440.0) < 1.0


class TestMotif:
    """Tests for Motif class."""
    
    def test_motif_creation(self):
        """Test creating a motif."""
        motif = Motif.from_intervals([0, 2, 4, 7], scale=Scale.major('C'))
        assert motif.intervals == [0, 2, 4, 7]
    
    def test_motif_get_frequencies(self):
        """Test getting frequencies from motif."""
        motif = Motif.from_intervals([0, 2, 4], scale=Scale.major('C'))
        freqs = motif.get_frequencies()
        assert len(freqs) == 3
        assert all(isinstance(f, float) for f in freqs)
    
    def test_motif_transpose(self):
        """Test transposing a motif."""
        motif = Motif.from_intervals([0, 2, 4], scale=Scale.major('C'))
        transposed = motif.transpose(semitones=5)
        assert transposed.intervals == [5, 7, 9]
    
    def test_motif_reverse(self):
        """Test reversing a motif."""
        motif = Motif.from_intervals([0, 2, 4, 7], scale=Scale.major('C'))
        reversed_motif = motif.reverse()
        assert reversed_motif.intervals == [7, 4, 2, 0]
    
    def test_motif_invert(self):
        """Test inverting a motif."""
        motif = Motif.from_intervals([0, 2, 4], scale=Scale.major('C'))
        inverted = motif.invert()
        assert inverted.intervals[0] == 0  # Root stays the same


class TestRhythm:
    """Tests for Rhythm class."""
    
    def test_rhythm_creation(self):
        """Test creating a rhythm."""
        rhythm = Rhythm(pattern=[1.0, 0.5, 0.5], time_signature=(4, 4))
        assert rhythm.pattern == [1.0, 0.5, 0.5]
        assert rhythm.time_signature == (4, 4)
    
    def test_rhythm_from_subdivision(self):
        """Test creating rhythm from subdivision."""
        rhythm = Rhythm.from_subdivision(subdivision=4, beats=4)
        assert len(rhythm.pattern) == 16  # 4 beats * 4 subdivisions


class TestArpeggiator:
    """Tests for Arpeggiator class."""
    
    def test_arpeggiator_creation(self):
        """Test creating an arpeggiator."""
        arp = Arpeggiator(pattern='up', octaves=1)
        assert arp.pattern == 'up'
        assert arp.octaves == 1
    
    def test_arpeggiator_patterns(self):
        """Test different arpeggio patterns."""
        motif = Motif.from_intervals([0, 2, 4], scale=Scale.major('C'))
        patterns = ['up', 'down', 'up-down']
        
        for pattern in patterns:
            arp = Arpeggiator(pattern=pattern, octaves=1)
            result = arp.arpeggiate(motif)
            assert isinstance(result, Motif)
    
    def test_arpeggiator_octaves(self):
        """Test arpeggiator with multiple octaves."""
        motif = Motif.from_intervals([0, 2, 4], scale=Scale.major('C'))
        arp = Arpeggiator(pattern='up', octaves=2)
        result = arp.arpeggiate(motif)
        # Should have more intervals than original
        assert len(result.intervals) > len(motif.intervals)
