"""
algorythm.structure - Arranges and composes the final track

This module provides high-level composition tools for arranging tracks,
applying effects, and creating complete musical pieces.
"""

from typing import List, Optional, Dict, Any
import numpy as np
from algorythm.synth import Synth
from algorythm.sequence import Motif


class Reverb:
    """
    Reverb effect for adding spatial depth to audio.
    """
    
    def __init__(
        self,
        mix: float = 0.3,
        room_size: float = 0.5,
        damping: float = 0.5
    ):
        """
        Initialize a reverb effect.
        
        Args:
            mix: Wet/dry mix (0.0 = dry, 1.0 = wet)
            room_size: Room size parameter (0.0 to 1.0)
            damping: High frequency damping (0.0 to 1.0)
        """
        self.mix = mix
        self.room_size = room_size
        self.damping = damping
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply reverb effect to audio signal.
        
        Args:
            signal: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Processed audio signal
        """
        # Simplified reverb implementation
        # In a real implementation, this would use convolution or feedback delay network
        wet = signal.copy()
        dry = signal.copy()
        
        # Simple delay-based reverb approximation
        delay_samples = int(0.05 * sample_rate * self.room_size)
        if delay_samples > 0 and len(wet) > delay_samples:
            wet[delay_samples:] += wet[:-delay_samples] * 0.3 * (1 - self.damping)
        
        return dry * (1 - self.mix) + wet * self.mix


class Delay:
    """
    Delay effect for creating echoes.
    """
    
    def __init__(
        self,
        delay_time: float = 0.5,
        feedback: float = 0.3,
        mix: float = 0.3
    ):
        """
        Initialize a delay effect.
        
        Args:
            delay_time: Delay time in seconds
            feedback: Feedback amount (0.0 to 1.0)
            mix: Wet/dry mix (0.0 = dry, 1.0 = wet)
        """
        self.delay_time = delay_time
        self.feedback = feedback
        self.mix = mix
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply delay effect to audio signal.
        
        Args:
            signal: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Processed audio signal
        """
        delay_samples = int(self.delay_time * sample_rate)
        output = signal.copy()
        
        if delay_samples > 0 and len(output) > delay_samples:
            delayed = np.zeros_like(output)
            delayed[delay_samples:] = output[:-delay_samples]
            output = output * (1 - self.mix) + delayed * self.mix
        
        return output


class Chorus:
    """
    Chorus effect for creating a thicker, richer sound.
    """
    
    def __init__(
        self,
        rate: float = 0.5,
        depth: float = 0.3,
        mix: float = 0.5
    ):
        """
        Initialize a chorus effect.
        
        Args:
            rate: LFO rate in Hz
            depth: Modulation depth (0.0 to 1.0)
            mix: Wet/dry mix (0.0 = dry, 1.0 = wet)
        """
        self.rate = rate
        self.depth = depth
        self.mix = mix
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply chorus effect to audio signal.
        
        Args:
            signal: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Processed audio signal
        """
        # Simplified chorus implementation
        return signal


class EffectChain:
    """
    Chain of audio effects to be applied in sequence.
    """
    
    def __init__(self):
        """Initialize an empty effect chain."""
        self.effects: List[Any] = []
    
    def add_effect(self, effect: Any) -> 'EffectChain':
        """
        Add an effect to the chain.
        
        Args:
            effect: Effect object (Reverb, Delay, etc.)
            
        Returns:
            Self for method chaining
        """
        self.effects.append(effect)
        return self
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply all effects in the chain to audio signal.
        
        Args:
            signal: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Processed audio signal
        """
        output = signal.copy()
        for effect in self.effects:
            output = effect.apply(output, sample_rate)
        return output


class Track:
    """
    A single track in a composition.
    
    Contains a synth instrument and a sequence of notes/motifs to play.
    """
    
    def __init__(self, name: str, synth: Synth):
        """
        Initialize a track.
        
        Args:
            name: Track name
            synth: Synthesizer instrument for this track
        """
        self.name = name
        self.synth = synth
        self.notes: List[Dict[str, Any]] = []
        self.effect_chain = EffectChain()
        self.volume = 1.0
        self.pan = 0.0  # -1.0 (left) to 1.0 (right)
    
    def add_note(
        self,
        frequency: float,
        start_time: float,
        duration: float
    ) -> 'Track':
        """
        Add a single note to the track.
        
        Args:
            frequency: Note frequency in Hz
            start_time: Start time in seconds
            duration: Duration in seconds
            
        Returns:
            Self for method chaining
        """
        self.notes.append({
            'frequency': frequency,
            'start_time': start_time,
            'duration': duration
        })
        return self
    
    def add_motif(
        self,
        motif: Motif,
        start_time: float,
        tempo: float = 120.0
    ) -> 'Track':
        """
        Add a motif to the track.
        
        Args:
            motif: Motif to add
            start_time: Start time in seconds
            tempo: Tempo in BPM
            
        Returns:
            Self for method chaining
        """
        beat_duration = 60.0 / tempo
        current_time = start_time
        frequencies = motif.get_frequencies()
        
        for freq, duration in zip(frequencies, motif.durations):
            self.add_note(freq, current_time, duration * beat_duration)
            current_time += duration * beat_duration
        
        return self
    
    def add_fx(self, effect: Any) -> 'Track':
        """
        Add an effect to the track.
        
        Args:
            effect: Effect object
            
        Returns:
            Self for method chaining
        """
        self.effect_chain.add_effect(effect)
        return self
    
    def render(self, sample_rate: int = 44100) -> np.ndarray:
        """
        Render the track to audio.
        
        Args:
            sample_rate: Sample rate in Hz
            
        Returns:
            Rendered audio signal
        """
        if not self.notes:
            return np.array([])
        
        # Calculate total duration
        max_time = max(note['start_time'] + note['duration'] for note in self.notes)
        total_samples = int(max_time * sample_rate)
        output = np.zeros(total_samples)
        
        # Render each note
        for note in self.notes:
            note_signal = self.synth.generate_note(
                note['frequency'],
                note['duration'],
                sample_rate
            )
            start_sample = int(note['start_time'] * sample_rate)
            end_sample = start_sample + len(note_signal)
            
            if end_sample <= total_samples:
                output[start_sample:end_sample] += note_signal
            else:
                output[start_sample:] += note_signal[:total_samples - start_sample]
        
        # Apply effects
        output = self.effect_chain.apply(output, sample_rate)
        
        # Apply volume
        output = output * self.volume
        
        return output


class Composition:
    """
    Main composition class that combines multiple tracks.
    
    This is the top-level container for a musical piece.
    """
    
    def __init__(self, tempo: float = 120.0, sample_rate: int = 44100):
        """
        Initialize a composition.
        
        Args:
            tempo: Tempo in beats per minute
            sample_rate: Sample rate in Hz
        """
        self.tempo = tempo
        self.sample_rate = sample_rate
        self.tracks: Dict[str, Track] = {}
        self.current_track: Optional[Track] = None
    
    def add_track(self, name: str, synth: Synth) -> 'Composition':
        """
        Add a new track to the composition.
        
        Args:
            name: Track name
            synth: Synthesizer for the track
            
        Returns:
            Self for method chaining
        """
        track = Track(name, synth)
        self.tracks[name] = track
        self.current_track = track
        return self
    
    def repeat_motif(
        self,
        motif: Motif,
        bars: int = 1,
        start_bar: float = 0
    ) -> 'Composition':
        """
        Repeat a motif for a specified number of bars.
        
        Args:
            motif: Motif to repeat
            bars: Number of bars to repeat
            start_bar: Starting bar number
            
        Returns:
            Self for method chaining
        """
        if not self.current_track:
            raise ValueError("No track selected. Call add_track first.")
        
        beats_per_bar = 4  # Assuming 4/4 time
        bar_duration = (beats_per_bar * 60.0) / self.tempo
        
        for bar in range(bars):
            start_time = (start_bar + bar) * bar_duration
            self.current_track.add_motif(motif, start_time, self.tempo)
        
        return self
    
    def transpose(self, semitones: int) -> 'Composition':
        """
        Transpose the current track by semitones.
        
        Args:
            semitones: Number of semitones to transpose
            
        Returns:
            Self for method chaining
        """
        if not self.current_track:
            raise ValueError("No track selected. Call add_track first.")
        
        # Transpose all notes in the current track
        for note in self.current_track.notes:
            # Frequency ratio for semitone transposition
            ratio = 2.0 ** (semitones / 12.0)
            note['frequency'] *= ratio
        
        return self
    
    def add_fx(self, effect: Any) -> 'Composition':
        """
        Add an effect to the current track.
        
        Args:
            effect: Effect object
            
        Returns:
            Self for method chaining
        """
        if not self.current_track:
            raise ValueError("No track selected. Call add_track first.")
        
        self.current_track.add_fx(effect)
        return self
    
    def render(
        self,
        file_path: Optional[str] = None,
        quality: str = 'high',
        formats: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Render the composition to audio.
        
        Args:
            file_path: Output file path (without extension if multiple formats)
            quality: Quality setting ('low', 'medium', 'high')
            formats: List of output formats (e.g., ['wav', 'mp3', 'flac'])
            
        Returns:
            Rendered audio signal
        """
        if not self.tracks:
            return np.array([])
        
        # Render all tracks
        track_signals = [track.render(self.sample_rate) for track in self.tracks.values()]
        
        # Find maximum length
        max_length = max(len(signal) for signal in track_signals if len(signal) > 0)
        
        # Mix tracks
        output = np.zeros(max_length)
        for signal in track_signals:
            if len(signal) > 0:
                output[:len(signal)] += signal
        
        # Normalize to prevent clipping
        max_amplitude = np.max(np.abs(output))
        if max_amplitude > 0:
            output = output / max_amplitude * 0.9
        
        # Export if file path provided
        if file_path:
            from algorythm.export import Exporter
            exporter = Exporter()
            
            if formats:
                for fmt in formats:
                    # Remove extension from file_path if present
                    base_path = file_path.rsplit('.', 1)[0] if '.' in file_path else file_path
                    output_path = f"{base_path}.{fmt}"
                    exporter.export(output, output_path, self.sample_rate, quality)
            else:
                exporter.export(output, file_path, self.sample_rate, quality)
        
        return output
