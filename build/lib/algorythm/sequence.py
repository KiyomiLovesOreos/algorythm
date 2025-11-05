"""
algorythm.sequence - Handles rhythmic and melodic patterns

This module provides tools for creating musical sequences, motifs, rhythms,
and arpeggios.
"""

from typing import List, Optional, Literal, Union
import numpy as np


class Tuning:
    """
    Global tuning system for microtonal and alternative temperaments.
    
    Supports custom tunings, equal temperaments (e.g., 19-TET), and historical tunings.
    """
    
    # Pre-defined tuning systems (in cents, where 100 cents = 1 semitone)
    TUNINGS = {
        '12-TET': [i * 100 for i in range(12)],  # Standard 12-tone equal temperament
        '19-TET': [i * (1200 / 19) for i in range(19)],  # 19-tone equal temperament
        '24-TET': [i * 50 for i in range(24)],  # Quarter-tone system
        'just_intonation': [0, 111.73, 203.91, 315.64, 386.31, 498.04, 582.51, 701.96, 813.69, 884.36, 1017.60, 1088.27],
        'pythagorean': [0, 90.22, 203.91, 294.13, 407.82, 498.04, 588.27, 701.96, 792.18, 905.87, 996.09, 1109.78],
    }
    
    def __init__(
        self,
        tuning_system: Union[str, List[float]] = '12-TET',
        reference_frequency: float = 440.0,
        reference_note: int = 69  # A4 in MIDI
    ):
        """
        Initialize a tuning system.
        
        Args:
            tuning_system: Name of predefined tuning or list of cents values per octave
            reference_frequency: Reference frequency in Hz (default A4 = 440 Hz)
            reference_note: MIDI note number of reference frequency
        """
        self.reference_frequency = reference_frequency
        self.reference_note = reference_note
        
        if isinstance(tuning_system, str):
            if tuning_system not in self.TUNINGS:
                raise ValueError(f"Unknown tuning system: {tuning_system}")
            self.cents = self.TUNINGS[tuning_system]
            self.name = tuning_system
        else:
            self.cents = tuning_system
            self.name = 'custom'
        
        self.tones_per_octave = len(self.cents)
    
    @classmethod
    def equal_temperament(cls, divisions: int) -> 'Tuning':
        """
        Create an equal temperament tuning with N divisions per octave.
        
        Args:
            divisions: Number of equal divisions per octave
            
        Returns:
            Tuning instance
        """
        cents = [i * (1200 / divisions) for i in range(divisions)]
        return cls(tuning_system=cents)
    
    @classmethod
    def just_intonation(cls) -> 'Tuning':
        """Create a just intonation tuning."""
        return cls('just_intonation')
    
    @classmethod
    def pythagorean(cls) -> 'Tuning':
        """Create a Pythagorean tuning."""
        return cls('pythagorean')
    
    def get_frequency(self, degree: int) -> float:
        """
        Get frequency for a scale degree in this tuning.
        
        Args:
            degree: Scale degree (can span multiple octaves)
            
        Returns:
            Frequency in Hz
        """
        octave = degree // self.tones_per_octave
        tone = degree % self.tones_per_octave
        
        # Calculate cents from reference
        cents_from_ref = self.cents[tone] + (octave * 1200)
        
        # Convert cents to frequency ratio: 2^(cents/1200)
        ratio = 2 ** (cents_from_ref / 1200)
        
        return self.reference_frequency * ratio
    
    def midi_to_frequency(self, midi_note: int) -> float:
        """
        Convert MIDI note to frequency using this tuning.
        
        Args:
            midi_note: MIDI note number
            
        Returns:
            Frequency in Hz
        """
        # Calculate distance from reference note in scale degrees
        semitones_from_ref = midi_note - self.reference_note
        
        # For standard 12-TET, this is straightforward
        if self.tones_per_octave == 12:
            degree = semitones_from_ref
        else:
            # Map 12-TET semitones to this tuning's degrees
            # This is an approximation for non-12-TET systems
            degree = int(semitones_from_ref * self.tones_per_octave / 12)
        
        return self.get_frequency(degree)


class Scale:
    """
    Musical scale definitions and operations.
    
    Provides various scale types and key transpositions with microtonal support.
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
    
    def __init__(
        self,
        root: str,
        scale_type: str = 'major',
        octave: int = 4,
        tuning: Optional[Tuning] = None
    ):
        """
        Initialize a scale.
        
        Args:
            root: Root note name (e.g., 'C', 'D#', 'Bb')
            scale_type: Type of scale (e.g., 'major', 'minor')
            octave: Base octave number
            tuning: Optional custom tuning system (default: 12-TET)
        """
        self.root = root
        self.scale_type = scale_type
        self.octave = octave
        self.tuning = tuning or Tuning('12-TET')
        
        if scale_type not in self.SCALES:
            raise ValueError(f"Unknown scale type: {scale_type}")
        
        if root not in self.NOTE_NAMES:
            raise ValueError(f"Unknown note name: {root}")
        
        # Calculate MIDI note number for root
        self.root_midi = self.NOTE_NAMES[root] + (octave * 12) + 12  # +12 for MIDI offset
    
    @classmethod
    def major(cls, root: str, octave: int = 4, tuning: Optional[Tuning] = None) -> 'Scale':
        """Create a major scale."""
        return cls(root, 'major', octave, tuning)
    
    @classmethod
    def minor(cls, root: str, octave: int = 4, tuning: Optional[Tuning] = None) -> 'Scale':
        """Create a minor scale."""
        return cls(root, 'minor', octave, tuning)
    
    @classmethod
    def pentatonic_major(cls, root: str, octave: int = 4, tuning: Optional[Tuning] = None) -> 'Scale':
        """Create a pentatonic major scale."""
        return cls(root, 'pentatonic_major', octave, tuning)
    
    @classmethod
    def pentatonic_minor(cls, root: str, octave: int = 4, tuning: Optional[Tuning] = None) -> 'Scale':
        """Create a pentatonic minor scale."""
        return cls(root, 'pentatonic_minor', octave, tuning)
    
    @classmethod
    def blues(cls, root: str, octave: int = 4, tuning: Optional[Tuning] = None) -> 'Scale':
        """Create a blues scale."""
        return cls(root, 'blues', octave, tuning)
    
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
        # Use the tuning system to convert MIDI note to frequency
        return self.tuning.midi_to_frequency(midi_note)


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


class Chord:
    """
    Musical chord definitions and operations.
    
    Provides various chord types and voicings.
    """
    
    # Chord intervals in semitones from root
    CHORD_TYPES = {
        'major': [0, 4, 7],
        'minor': [0, 3, 7],
        'diminished': [0, 3, 6],
        'augmented': [0, 4, 8],
        'major7': [0, 4, 7, 11],
        'minor7': [0, 3, 7, 10],
        'dominant7': [0, 4, 7, 10],
        'diminished7': [0, 3, 6, 9],
        'sus2': [0, 2, 7],
        'sus4': [0, 5, 7],
        'major9': [0, 4, 7, 11, 14],
        'minor9': [0, 3, 7, 10, 14],
    }
    
    def __init__(self, root: str, chord_type: str = 'major', octave: int = 4):
        """
        Initialize a chord.
        
        Args:
            root: Root note name (e.g., 'C', 'D#', 'Bb')
            chord_type: Type of chord (e.g., 'major', 'minor', 'major7')
            octave: Base octave number
        """
        self.root = root
        self.chord_type = chord_type
        self.octave = octave
        
        if chord_type not in self.CHORD_TYPES:
            raise ValueError(f"Unknown chord type: {chord_type}")
        
        if root not in Scale.NOTE_NAMES:
            raise ValueError(f"Unknown note name: {root}")
        
        # Calculate MIDI note number for root
        self.root_midi = Scale.NOTE_NAMES[root] + (octave * 12) + 12  # +12 for MIDI offset
    
    @classmethod
    def major(cls, root: str, octave: int = 4) -> 'Chord':
        """Create a major chord."""
        return cls(root, 'major', octave)
    
    @classmethod
    def minor(cls, root: str, octave: int = 4) -> 'Chord':
        """Create a minor chord."""
        return cls(root, 'minor', octave)
    
    @classmethod
    def major7(cls, root: str, octave: int = 4) -> 'Chord':
        """Create a major 7th chord."""
        return cls(root, 'major7', octave)
    
    @classmethod
    def minor7(cls, root: str, octave: int = 4) -> 'Chord':
        """Create a minor 7th chord."""
        return cls(root, 'minor7', octave)
    
    @classmethod
    def dominant7(cls, root: str, octave: int = 4) -> 'Chord':
        """Create a dominant 7th chord."""
        return cls(root, 'dominant7', octave)
    
    def get_notes(self) -> List[int]:
        """
        Get MIDI note numbers for all notes in the chord.
        
        Returns:
            List of MIDI note numbers
        """
        intervals = self.CHORD_TYPES[self.chord_type]
        return [self.root_midi + interval for interval in intervals]
    
    def get_frequencies(self) -> List[float]:
        """
        Get frequencies in Hz for all notes in the chord.
        
        Returns:
            List of frequencies in Hz
        """
        midi_notes = self.get_notes()
        # Convert MIDI notes to frequencies: f = 440 * 2^((n-69)/12)
        return [440.0 * (2.0 ** ((note - 69) / 12.0)) for note in midi_notes]
    
    def to_motif(self, duration: float = 1.0) -> Motif:
        """
        Convert chord to a motif (for arpeggiation or sequencing).
        
        Args:
            duration: Duration per note in beats
            
        Returns:
            Motif representing the chord
        """
        # Create a temporary scale at the root
        temp_scale = Scale(self.root, 'chromatic', self.octave)
        
        # Get intervals relative to root
        intervals = self.CHORD_TYPES[self.chord_type]
        
        # Create motif with equal durations
        durations = [duration] * len(intervals)
        
        return Motif(intervals, temp_scale, durations)
    
    def invert(self, inversion: int = 1) -> 'Chord':
        """
        Create an inversion of the chord.
        
        Args:
            inversion: Inversion number (1 = first inversion, 2 = second, etc.)
            
        Returns:
            New Chord instance with the specified inversion
        """
        # Create a new chord with modified intervals
        intervals = self.CHORD_TYPES[self.chord_type].copy()
        
        # Apply inversions
        for _ in range(inversion):
            if intervals:
                # Move the lowest note up an octave
                intervals = intervals[1:] + [intervals[0] + 12]
        
        # Create a new chord type name for the inversion
        # Store the inverted intervals back (this is simplified)
        new_chord = Chord.__new__(Chord)
        new_chord.root = self.root
        new_chord.chord_type = self.chord_type
        new_chord.octave = self.octave
        new_chord.root_midi = self.root_midi
        
        return new_chord

