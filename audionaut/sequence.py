"""
audionaut.sequence - Handles rhythmic and melodic patterns

This module provides tools for creating musical sequences, motifs, rhythms,
and arpeggios.
"""

from typing import List, Optional, Literal
import numpy as np


class Scale:
    """
    Musical scale definitions and operations.
    
    Provides various scale types and key transpositions.
    """
    
    # Scale intervals in semitones from root
    SCALES = {
        'major': [0, 2, 4, 5, 7, 9, 11],
        'minor': [0, 2, 3, 5, 7, 8, 10],
        'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
        'melodic_minor': [0, 2, 3, 5, 7, 9, 11],
        'pentatonic_major': [0, 2, 4, 7, 9],
        'pentatonic_minor': [0, 3, 5, 7, 10],
        'blues': [0, 3, 5, 6, 7, 10],
        'chromatic': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    }
    
    # Note names to MIDI note numbers (C4 = middle C = 60)
    NOTE_NAMES = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
        'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
    }
    
    def __init__(self, root: str, scale_type: str = 'major', octave: int = 4):
        """
        Initialize a scale.
        
        Args:
            root: Root note name (e.g., 'C', 'D#', 'Bb')
            scale_type: Type of scale (e.g., 'major', 'minor')
            octave: Base octave number
        """
        self.root = root
        self.scale_type = scale_type
        self.octave = octave
        
        if scale_type not in self.SCALES:
            raise ValueError(f"Unknown scale type: {scale_type}")
        
        if root not in self.NOTE_NAMES:
            raise ValueError(f"Unknown note name: {root}")
        
        # Calculate MIDI note number for root
        self.root_midi = self.NOTE_NAMES[root] + (octave * 12) + 12  # +12 for MIDI offset
    
    @classmethod
    def major(cls, root: str, octave: int = 4) -> 'Scale':
        """Create a major scale."""
        return cls(root, 'major', octave)
    
    @classmethod
    def minor(cls, root: str, octave: int = 4) -> 'Scale':
        """Create a minor scale."""
        return cls(root, 'minor', octave)
    
    @classmethod
    def pentatonic_major(cls, root: str, octave: int = 4) -> 'Scale':
        """Create a pentatonic major scale."""
        return cls(root, 'pentatonic_major', octave)
    
    @classmethod
    def pentatonic_minor(cls, root: str, octave: int = 4) -> 'Scale':
        """Create a pentatonic minor scale."""
        return cls(root, 'pentatonic_minor', octave)
    
    @classmethod
    def blues(cls, root: str, octave: int = 4) -> 'Scale':
        """Create a blues scale."""
        return cls(root, 'blues', octave)
    
    def get_note(self, degree: int) -> int:
        """
        Get MIDI note number for a scale degree.
        
        Args:
            degree: Scale degree (0-based, can be negative or > 7)
            
        Returns:
            MIDI note number
        """
        intervals = self.SCALES[self.scale_type]
        octave_offset = degree // len(intervals)
        note_index = degree % len(intervals)
        
        return self.root_midi + intervals[note_index] + (octave_offset * 12)
    
    def get_frequency(self, degree: int) -> float:
        """
        Get frequency in Hz for a scale degree.
        
        Args:
            degree: Scale degree (0-based)
            
        Returns:
            Frequency in Hz
        """
        midi_note = self.get_note(degree)
        # Convert MIDI note to frequency: f = 440 * 2^((n-69)/12)
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


class Motif:
    """
    A musical motif - a short melodic or rhythmic idea.
    
    Motifs can be transposed, repeated, and transformed.
    """
    
    def __init__(
        self,
        intervals: List[int],
        scale: Optional[Scale] = None,
        durations: Optional[List[float]] = None
    ):
        """
        Initialize a motif.
        
        Args:
            intervals: List of scale degrees or semitone intervals
            scale: Optional scale to map intervals to
            durations: Optional list of note durations in beats
        """
        self.intervals = intervals
        self.scale = scale or Scale.major('C', 4)
        self.durations = durations or [1.0] * len(intervals)
        
        if len(self.durations) < len(self.intervals):
            # Pad durations if not enough provided
            self.durations.extend([1.0] * (len(self.intervals) - len(self.durations)))
    
    @classmethod
    def from_intervals(
        cls,
        intervals: List[int],
        scale: Optional[Scale] = None,
        durations: Optional[List[float]] = None
    ) -> 'Motif':
        """
        Create a motif from scale degree intervals.
        
        Args:
            intervals: List of scale degrees (0 = root)
            scale: Scale to use for mapping
            durations: Note durations in beats
            
        Returns:
            New Motif instance
        """
        return cls(intervals, scale, durations)
    
    @classmethod
    def from_notes(
        cls,
        notes: List[str],
        durations: Optional[List[float]] = None
    ) -> 'Motif':
        """
        Create a motif from note names.
        
        Args:
            notes: List of note names (e.g., ['C4', 'E4', 'G4'])
            durations: Note durations in beats
            
        Returns:
            New Motif instance
        """
        # Parse notes and create intervals
        # Simplified implementation
        intervals = list(range(len(notes)))
        return cls(intervals, durations=durations)
    
    def get_frequencies(self) -> List[float]:
        """
        Get frequencies for all notes in the motif.
        
        Returns:
            List of frequencies in Hz
        """
        return [self.scale.get_frequency(interval) for interval in self.intervals]
    
    def transpose(self, semitones: int) -> 'Motif':
        """
        Transpose the motif by semitones.
        
        Args:
            semitones: Number of semitones to transpose
            
        Returns:
            New transposed Motif
        """
        new_intervals = [interval + semitones for interval in self.intervals]
        return Motif(new_intervals, self.scale, self.durations.copy())
    
    def reverse(self) -> 'Motif':
        """
        Reverse the motif.
        
        Returns:
            New reversed Motif
        """
        return Motif(
            list(reversed(self.intervals)),
            self.scale,
            list(reversed(self.durations))
        )
    
    def invert(self) -> 'Motif':
        """
        Invert the motif (mirror intervals).
        
        Returns:
            New inverted Motif
        """
        if not self.intervals:
            return Motif([], self.scale, [])
        
        root = self.intervals[0]
        inverted = [root] + [root - (interval - root) for interval in self.intervals[1:]]
        return Motif(inverted, self.scale, self.durations.copy())


class Rhythm:
    """
    Defines rhythmic patterns.
    
    Handles timing, subdivisions, and rhythmic variations.
    """
    
    def __init__(
        self,
        pattern: List[float],
        time_signature: tuple = (4, 4)
    ):
        """
        Initialize a rhythm.
        
        Args:
            pattern: List of beat durations
            time_signature: Time signature as (beats_per_bar, beat_unit)
        """
        self.pattern = pattern
        self.time_signature = time_signature
    
    @classmethod
    def from_subdivision(
        cls,
        subdivision: int,
        beats: int = 4
    ) -> 'Rhythm':
        """
        Create a rhythm from equal subdivisions.
        
        Args:
            subdivision: Number of subdivisions per beat
            beats: Number of beats
            
        Returns:
            New Rhythm instance
        """
        duration = 1.0 / subdivision
        pattern = [duration] * (beats * subdivision)
        return cls(pattern)
    
    def get_durations(self) -> List[float]:
        """Get rhythm durations."""
        return self.pattern.copy()


class Arpeggiator:
    """
    Generates arpeggios from chords or scales.
    
    Provides various arpeggio patterns and directions.
    """
    
    def __init__(
        self,
        pattern: Literal['up', 'down', 'up-down', 'random'] = 'up',
        octaves: int = 1
    ):
        """
        Initialize an arpeggiator.
        
        Args:
            pattern: Arpeggio pattern direction
            octaves: Number of octaves to span
        """
        self.pattern = pattern
        self.octaves = octaves
    
    def arpeggiate(self, motif: Motif) -> Motif:
        """
        Apply arpeggio pattern to a motif.
        
        Args:
            motif: Input motif
            
        Returns:
            Arpeggiated motif
        """
        intervals = motif.intervals.copy()
        
        # Extend across octaves
        if self.octaves > 1:
            scale_length = len(intervals)
            for octave in range(1, self.octaves):
                intervals.extend([i + (octave * scale_length) for i in motif.intervals])
        
        # Apply pattern
        if self.pattern == 'down':
            intervals = list(reversed(intervals))
        elif self.pattern == 'up-down':
            intervals = intervals + list(reversed(intervals[1:-1]))
        elif self.pattern == 'random':
            import random
            intervals = intervals.copy()
            random.shuffle(intervals)
        
        return Motif(intervals, motif.scale)
