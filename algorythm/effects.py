"""
algorythm.effects - Audio effects and signal processing

This module provides a comprehensive collection of audio effects including
reverb, delay, modulation, dynamics, and distortion effects.
"""

import numpy as np
from typing import Literal, Optional


class Effect:
    """Base class for audio effects."""
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply effect to audio signal.
        
        Args:
            signal: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Processed audio signal
        """
        return signal


class Reverb(Effect):
    """
    Reverb effect for adding space and depth.
    
    Simulates acoustic reflections in a room or hall.
    """
    
    def __init__(
        self,
        room_size: float = 0.5,
        damping: float = 0.5,
        wet_level: float = 0.3
    ):
        """
        Initialize reverb effect.
        
        Args:
            room_size: Size of the simulated room (0.0 to 1.0)
            damping: High-frequency damping (0.0 to 1.0)
            wet_level: Mix of reverb signal (0.0 to 1.0)
        """
        self.room_size = room_size
        self.damping = damping
        self.wet_level = wet_level
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        delay_times = [int(sample_rate * 0.037 * self.room_size),
                      int(sample_rate * 0.041 * self.room_size),
                      int(sample_rate * 0.043 * self.room_size),
                      int(sample_rate * 0.047 * self.room_size)]
        
        wet = np.zeros_like(signal)
        for delay_time in delay_times:
            if delay_time > 0 and delay_time < len(signal):
                delayed = np.concatenate([np.zeros(delay_time), signal[:-delay_time]])
                wet += delayed * self.damping
        
        wet = wet / len(delay_times)
        return signal * (1 - self.wet_level) + wet * self.wet_level


class Delay(Effect):
    """
    Delay effect for echoes and rhythmic patterns.
    
    Creates time-delayed copies of the input signal.
    """
    
    def __init__(
        self,
        delay_time: float = 0.5,
        feedback: float = 0.5,
        wet_level: float = 0.5
    ):
        """
        Initialize delay effect.
        
        Args:
            delay_time: Delay time in seconds
            feedback: Amount of feedback (0.0 to 1.0)
            wet_level: Mix of delayed signal (0.0 to 1.0)
        """
        self.delay_time = delay_time
        self.feedback = feedback
        self.wet_level = wet_level
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        delay_samples = int(self.delay_time * sample_rate)
        output = signal.copy()
        
        for i in range(delay_samples, len(signal)):
            output[i] += output[i - delay_samples] * self.feedback
        
        return signal * (1 - self.wet_level) + output * self.wet_level


class Chorus(Effect):
    """
    Chorus effect for thickening sounds.
    
    Creates multiple delayed copies with slight pitch modulation.
    """
    
    def __init__(
        self,
        depth: float = 0.003,
        rate: float = 1.5,
        mix: float = 0.5
    ):
        """
        Initialize chorus effect.
        
        Args:
            depth: Modulation depth in seconds
            rate: LFO rate in Hz
            mix: Mix of chorus signal (0.0 to 1.0)
        """
        self.depth = depth
        self.rate = rate
        self.mix = mix
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        num_samples = len(signal)
        t = np.arange(num_samples) / sample_rate
        
        lfo = np.sin(2 * np.pi * self.rate * t)
        delay_samples = (self.depth * sample_rate * (1 + lfo)).astype(int)
        
        wet = np.zeros_like(signal)
        for i in range(num_samples):
            delay = min(delay_samples[i], i)
            if delay > 0:
                wet[i] = signal[i - delay]
        
        return signal * (1 - self.mix) + wet * self.mix


class Flanger(Effect):
    """
    Flanger effect for sweeping comb filter sounds.
    
    Similar to chorus but with shorter delay times and feedback.
    """
    
    def __init__(
        self,
        depth: float = 0.002,
        rate: float = 0.25,
        feedback: float = 0.7,
        mix: float = 0.5
    ):
        self.depth = depth
        self.rate = rate
        self.feedback = feedback
        self.mix = mix
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        num_samples = len(signal)
        t = np.arange(num_samples) / sample_rate
        
        lfo = np.sin(2 * np.pi * self.rate * t)
        delay_samples = (self.depth * sample_rate * (1 + lfo)).astype(int)
        
        wet = signal.copy()
        for i in range(num_samples):
            delay = min(max(delay_samples[i], 1), i)
            if delay > 0:
                wet[i] = signal[i] + wet[i - delay] * self.feedback
        
        return signal * (1 - self.mix) + wet * self.mix


class Phaser(Effect):
    """
    Phaser effect for sweeping notch-filter sounds.
    
    Creates a series of notches in the frequency spectrum that move over time.
    """
    
    def __init__(
        self,
        rate: float = 0.5,
        depth: float = 1.0,
        feedback: float = 0.5,
        mix: float = 0.5
    ):
        self.rate = rate
        self.depth = depth
        self.feedback = feedback
        self.mix = mix
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        num_samples = len(signal)
        t = np.arange(num_samples) / sample_rate
        
        lfo = np.sin(2 * np.pi * self.rate * t)
        delay_samples = (2 + self.depth * lfo).astype(int)
        
        wet = np.zeros_like(signal)
        for i in range(num_samples):
            delay = min(delay_samples[i], i)
            if delay > 0:
                wet[i] = signal[i] - wet[i - delay] * self.feedback
        
        return signal * (1 - self.mix) + wet * self.mix


class Distortion(Effect):
    """
    Distortion effect for adding harmonics and grit.
    
    Uses waveshaping to add harmonic content.
    """
    
    def __init__(
        self,
        drive: float = 5.0,
        tone: float = 0.5,
        mix: float = 1.0
    ):
        """
        Initialize distortion effect.
        
        Args:
            drive: Amount of distortion (1.0 to 20.0)
            tone: Tone control (0.0 = dark, 1.0 = bright)
            mix: Mix of distorted signal (0.0 to 1.0)
        """
        self.drive = drive
        self.tone = tone
        self.mix = mix
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        wet = np.tanh(signal * self.drive) / np.tanh(self.drive)
        
        if self.tone < 1.0:
            from algorythm.synth import Filter
            cutoff = 500 + self.tone * 4500
            filter_obj = Filter.lowpass(cutoff=cutoff)
            wet = filter_obj.apply(wet, sample_rate)
        
        return signal * (1 - self.mix) + wet * self.mix


class Overdrive(Effect):
    """
    Overdrive effect for smooth tube-like distortion.
    """
    
    def __init__(
        self,
        drive: float = 2.0,
        tone: float = 0.7,
        mix: float = 1.0
    ):
        self.drive = drive
        self.tone = tone
        self.mix = mix
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        wet = signal * self.drive
        wet = np.where(wet > 1.0, 1.0 - np.exp(-(wet - 1.0)), wet)
        wet = np.where(wet < -1.0, -1.0 + np.exp(-(np.abs(wet) - 1.0)), wet)
        
        if self.tone < 1.0:
            from algorythm.synth import Filter
            cutoff = 1000 + self.tone * 4000
            filter_obj = Filter.lowpass(cutoff=cutoff)
            wet = filter_obj.apply(wet, sample_rate)
        
        return signal * (1 - self.mix) + wet * self.mix


class Fuzz(Effect):
    """
    Fuzz effect for extreme clipping distortion.
    """
    
    def __init__(
        self,
        gain: float = 10.0,
        mix: float = 1.0
    ):
        self.gain = gain
        self.mix = mix
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        wet = np.clip(signal * self.gain, -1.0, 1.0)
        return signal * (1 - self.mix) + wet * self.mix


class Compressor(Effect):
    """
    Dynamics compressor for controlling signal levels.
    
    Reduces the dynamic range of audio by attenuating loud signals.
    """
    
    def __init__(
        self,
        threshold: float = -20.0,
        ratio: float = 4.0,
        attack: float = 0.005,
        release: float = 0.1,
        makeup_gain: float = 1.0
    ):
        """
        Initialize compressor.
        
        Args:
            threshold: Threshold in dB
            ratio: Compression ratio (1.0 to 20.0)
            attack: Attack time in seconds
            release: Release time in seconds
            makeup_gain: Output gain multiplier
        """
        self.threshold = threshold
        self.ratio = ratio
        self.attack = attack
        self.release = release
        self.makeup_gain = makeup_gain
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        threshold_linear = 10 ** (self.threshold / 20)
        output = signal.copy()
        envelope = 0.0
        
        attack_coef = np.exp(-1 / (self.attack * sample_rate))
        release_coef = np.exp(-1 / (self.release * sample_rate))
        
        for i in range(len(signal)):
            input_level = abs(signal[i])
            
            if input_level > envelope:
                envelope = attack_coef * envelope + (1 - attack_coef) * input_level
            else:
                envelope = release_coef * envelope + (1 - release_coef) * input_level
            
            if envelope > threshold_linear:
                gain_reduction = (envelope / threshold_linear) ** (1 / self.ratio - 1)
                output[i] *= gain_reduction
        
        return output * self.makeup_gain


class Limiter(Effect):
    """
    Limiter for preventing signal clipping.
    """
    
    def __init__(
        self,
        threshold: float = -1.0,
        release: float = 0.05
    ):
        self.threshold = threshold
        self.release = release
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        threshold_linear = 10 ** (self.threshold / 20)
        output = signal.copy()
        envelope = 0.0
        
        release_coef = np.exp(-1 / (self.release * sample_rate))
        
        for i in range(len(signal)):
            input_level = abs(signal[i])
            
            if input_level > threshold_linear:
                envelope = input_level
            else:
                envelope = release_coef * envelope + (1 - release_coef) * input_level
            
            if envelope > threshold_linear:
                output[i] *= threshold_linear / envelope
        
        return output


class Gate(Effect):
    """
    Noise gate for removing low-level signals.
    """
    
    def __init__(
        self,
        threshold: float = -40.0,
        attack: float = 0.001,
        release: float = 0.1
    ):
        self.threshold = threshold
        self.attack = attack
        self.release = release
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        threshold_linear = 10 ** (self.threshold / 20)
        output = signal.copy()
        envelope = 0.0
        gate_open = False
        
        attack_coef = np.exp(-1 / (self.attack * sample_rate))
        release_coef = np.exp(-1 / (self.release * sample_rate))
        
        for i in range(len(signal)):
            input_level = abs(signal[i])
            
            if input_level > threshold_linear:
                gate_open = True
            
            if gate_open:
                envelope = attack_coef * envelope + (1 - attack_coef)
            else:
                envelope = release_coef * envelope
            
            if input_level < threshold_linear:
                gate_open = False
            
            output[i] *= envelope
        
        return output


class Tremolo(Effect):
    """
    Tremolo effect for amplitude modulation.
    """
    
    def __init__(
        self,
        rate: float = 5.0,
        depth: float = 0.5,
        waveform: Literal['sine', 'square', 'triangle'] = 'sine'
    ):
        self.rate = rate
        self.depth = depth
        self.waveform = waveform
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        num_samples = len(signal)
        t = np.arange(num_samples) / sample_rate
        
        if self.waveform == 'sine':
            lfo = np.sin(2 * np.pi * self.rate * t)
        elif self.waveform == 'square':
            lfo = np.sign(np.sin(2 * np.pi * self.rate * t))
        elif self.waveform == 'triangle':
            lfo = 2 * np.abs(2 * (t * self.rate - np.floor(0.5 + t * self.rate))) - 1
        else:
            lfo = np.sin(2 * np.pi * self.rate * t)
        
        modulation = 1.0 - self.depth * (1 - lfo) / 2
        return signal * modulation


class Vibrato(Effect):
    """
    Vibrato effect for pitch modulation.
    """
    
    def __init__(
        self,
        rate: float = 5.0,
        depth: float = 0.002
    ):
        self.rate = rate
        self.depth = depth
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        num_samples = len(signal)
        t = np.arange(num_samples) / sample_rate
        
        lfo = np.sin(2 * np.pi * self.rate * t)
        delay_samples = (self.depth * sample_rate * lfo).astype(int)
        
        output = np.zeros_like(signal)
        for i in range(num_samples):
            delay = delay_samples[i]
            read_pos = i - delay
            if 0 <= read_pos < num_samples:
                output[i] = signal[read_pos]
        
        return output


class BitCrusher(Effect):
    """
    Bit crusher for lo-fi digital distortion.
    
    Reduces bit depth and sample rate for retro digital sounds.
    """
    
    def __init__(
        self,
        bit_depth: int = 8,
        sample_rate_reduction: float = 1.0,
        mix: float = 1.0
    ):
        """
        Initialize bit crusher.
        
        Args:
            bit_depth: Target bit depth (1 to 16)
            sample_rate_reduction: Sample rate reduction factor (0.1 to 1.0)
            mix: Mix of crushed signal (0.0 to 1.0)
        """
        self.bit_depth = bit_depth
        self.sample_rate_reduction = sample_rate_reduction
        self.mix = mix
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        levels = 2 ** self.bit_depth
        wet = np.round(signal * levels / 2) / (levels / 2)
        
        if self.sample_rate_reduction < 1.0:
            reduced_rate = int(len(signal) * self.sample_rate_reduction)
            if reduced_rate > 0:
                indices = np.linspace(0, len(signal) - 1, reduced_rate).astype(int)
                reduced = wet[indices]
                wet = np.repeat(reduced, len(signal) // reduced_rate + 1)[:len(signal)]
        
        return signal * (1 - self.mix) + wet * self.mix


class AutoPan(Effect):
    """
    Auto-panning effect for stereo movement (when used with stereo signals).
    """
    
    def __init__(
        self,
        rate: float = 1.0,
        depth: float = 1.0
    ):
        self.rate = rate
        self.depth = depth
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        num_samples = len(signal)
        t = np.arange(num_samples) / sample_rate
        
        lfo = np.sin(2 * np.pi * self.rate * t)
        pan = (lfo * self.depth + 1) / 2
        
        return signal * pan


class RingModulator(Effect):
    """
    Ring modulator for metallic and bell-like effects.
    """
    
    def __init__(
        self,
        carrier_freq: float = 440.0,
        mix: float = 0.5
    ):
        self.carrier_freq = carrier_freq
        self.mix = mix
    
    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        num_samples = len(signal)
        t = np.arange(num_samples) / sample_rate
        
        carrier = np.sin(2 * np.pi * self.carrier_freq * t)
        wet = signal * carrier
        
        return signal * (1 - self.mix) + wet * self.mix


class Stutter(Effect):
    """
    Stutter/repeat effect for rhythmic glitching.

    Repeats small segments of audio at regular intervals.
    """

    def __init__(
        self,
        segment_duration: float = 0.125,
        repeat_count: int = 4,
        interval: float = 0.5,
        mix: float = 1.0
    ):
        """
        Initialize stutter effect.

        Args:
            segment_duration: Duration of segment to repeat in seconds
            repeat_count: Number of times to repeat the segment
            interval: Interval between stutters in seconds
            mix: Mix of stuttered signal (0.0 to 1.0)
        """
        self.segment_duration = segment_duration
        self.repeat_count = repeat_count
        self.interval = interval
        self.mix = mix

    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        segment_samples = int(self.segment_duration * sample_rate)
        interval_samples = int(self.interval * sample_rate)

        wet = signal.copy()
        position = 0

        while position + interval_samples < len(wet):
            # Capture segment
            segment_start = position + interval_samples
            segment_end = min(segment_start + segment_samples, len(wet))
            segment = wet[segment_start:segment_end].copy()

            # Repeat segment
            for i in range(1, self.repeat_count):
                repeat_start = segment_start + i * segment_samples
                repeat_end = min(repeat_start + len(segment), len(wet))
                if repeat_start < len(wet):
                    wet[repeat_start:repeat_end] = segment[:repeat_end - repeat_start]

            position += interval_samples + self.repeat_count * segment_samples

        return signal * (1 - self.mix) + wet * self.mix


class FilterSweep(Effect):
    """
    Filter sweep effect with time-varying cutoff frequency.

    Creates sweeping filter effects using LFO modulation.
    """

    def __init__(
        self,
        filter_type: Literal['lowpass', 'highpass', 'bandpass'] = 'lowpass',
        start_freq: float = 200.0,
        end_freq: float = 5000.0,
        duration: Optional[float] = None,
        rate: float = 0.5,
        resonance: float = 1.0,
        mix: float = 1.0
    ):
        """
        Initialize filter sweep effect.

        Args:
            filter_type: Type of filter to sweep
            start_freq: Starting cutoff frequency in Hz
            end_freq: Ending cutoff frequency in Hz
            duration: Duration of one sweep cycle in seconds (None = use rate)
            rate: Sweep rate in Hz (cycles per second, used if duration is None)
            resonance: Filter resonance (0.0 to 10.0)
            mix: Mix of filtered signal (0.0 to 1.0)
        """
        self.filter_type = filter_type
        self.start_freq = start_freq
        self.end_freq = end_freq
        self.duration = duration
        self.rate = rate
        self.resonance = resonance
        self.mix = mix

    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        try:
            from scipy import signal as scipy_signal
            has_scipy = True
        except ImportError:
            has_scipy = False

        num_samples = len(signal)
        t = np.arange(num_samples) / sample_rate

        # Create sweep envelope
        if self.duration:
            lfo = (t % self.duration) / self.duration
        else:
            lfo = (np.sin(2 * np.pi * self.rate * t) + 1) / 2

        # Calculate frequency sweep
        freqs = self.start_freq + (self.end_freq - self.start_freq) * lfo

        if not has_scipy:
            # Simple frequency-based filtering without scipy
            # Use FFT-based filtering as fallback
            wet = signal.copy()
            chunk_size = 2048

            for i in range(0, len(signal), chunk_size):
                chunk_end = min(i + chunk_size, len(signal))
                chunk = signal[i:chunk_end]

                if len(chunk) < 4:
                    wet[i:chunk_end] = chunk
                    continue

                # Use average frequency for this chunk
                freq = np.mean(freqs[i:chunk_end])

                # FFT-based filtering
                fft = np.fft.rfft(chunk)
                fft_freqs = np.fft.rfftfreq(len(chunk), 1/sample_rate)

                if self.filter_type == 'lowpass':
                    # Attenuate frequencies above cutoff
                    mask = fft_freqs > freq
                    fft[mask] *= np.exp(-((fft_freqs[mask] - freq) / (freq * 0.5)) ** 2)
                elif self.filter_type == 'highpass':
                    # Attenuate frequencies below cutoff
                    mask = fft_freqs < freq
                    fft[mask] *= np.exp(-((freq - fft_freqs[mask]) / (freq * 0.5)) ** 2)
                elif self.filter_type == 'bandpass':
                    # Keep frequencies around cutoff
                    center = freq
                    bandwidth = freq * 0.4
                    distances = np.abs(fft_freqs - center)
                    mask = distances > bandwidth
                    fft[mask] *= np.exp(-((distances[mask] - bandwidth) / bandwidth) ** 2)

                filtered_chunk = np.fft.irfft(fft, n=len(chunk))
                wet[i:chunk_end] = filtered_chunk

            return signal * (1 - self.mix) + wet * self.mix

        # Use scipy if available for better quality
        wet = np.zeros_like(signal)
        chunk_size = 1024

        for i in range(0, len(signal), chunk_size):
            chunk_end = min(i + chunk_size, len(signal))
            chunk = signal[i:chunk_end]

            # Use average frequency for this chunk
            freq = np.mean(freqs[i:chunk_end])
            nyquist = sample_rate / 2
            norm_freq = min(freq / nyquist, 0.99)

            # Design filter for this chunk
            try:
                if self.filter_type == 'lowpass':
                    b, a = scipy_signal.butter(4, norm_freq, btype='low')
                elif self.filter_type == 'highpass':
                    b, a = scipy_signal.butter(4, norm_freq, btype='high')
                elif self.filter_type == 'bandpass':
                    low = max(norm_freq * 0.8, 0.01)
                    high = min(norm_freq * 1.2, 0.99)
                    b, a = scipy_signal.butter(4, [low, high], btype='band')

                # Apply filter
                filtered_chunk = scipy_signal.lfilter(b, a, chunk)
                wet[i:chunk_end] = filtered_chunk
            except:
                wet[i:chunk_end] = chunk

        return signal * (1 - self.mix) + wet * self.mix


class Freeze(Effect):
    """
    Freeze effect that captures and holds a moment of audio.

    Creates sustained textures by repeating a small buffer.
    """

    def __init__(
        self,
        freeze_start: float = 0.5,
        freeze_duration: float = 0.1,
        hold_duration: float = 2.0,
        fade_time: float = 0.05,
        mix: float = 1.0
    ):
        """
        Initialize freeze effect.

        Args:
            freeze_start: Time position to freeze (0.0 to 1.0 = start to end)
            freeze_duration: Duration of frozen buffer in seconds
            hold_duration: How long to hold the frozen buffer in seconds
            fade_time: Crossfade time in seconds
            mix: Mix of frozen signal (0.0 to 1.0)
        """
        self.freeze_start = freeze_start
        self.freeze_duration = freeze_duration
        self.hold_duration = hold_duration
        self.fade_time = fade_time
        self.mix = mix

    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        # Find freeze position
        freeze_pos = int(self.freeze_start * len(signal))
        freeze_samples = int(self.freeze_duration * sample_rate)
        hold_samples = int(self.hold_duration * sample_rate)
        fade_samples = int(self.fade_time * sample_rate)

        # Extract freeze buffer
        freeze_start = min(freeze_pos, len(signal) - freeze_samples)
        freeze_end = freeze_start + freeze_samples
        freeze_buffer = signal[freeze_start:freeze_end].copy()

        if len(freeze_buffer) == 0:
            return signal

        # Create output
        wet = signal.copy()

        # Repeat freeze buffer for hold duration
        current_pos = freeze_start
        while current_pos < len(wet) and current_pos < freeze_start + hold_samples:
            # Loop through freeze buffer
            for i in range(len(freeze_buffer)):
                if current_pos >= len(wet):
                    break

                # Crossfade at loop boundaries
                fade_in = min(i / fade_samples, 1.0) if i < fade_samples else 1.0
                fade_out = min((len(freeze_buffer) - i) / fade_samples, 1.0) if i > len(freeze_buffer) - fade_samples else 1.0
                fade = min(fade_in, fade_out)

                wet[current_pos] = freeze_buffer[i] * fade + wet[current_pos] * (1 - fade)
                current_pos += 1

        return signal * (1 - self.mix) + wet * self.mix


class Reverse(Effect):
    """
    Reverse effect that reverses segments of audio.

    Can reverse the entire signal or create rhythmic reverse effects.
    """

    def __init__(
        self,
        segment_duration: Optional[float] = None,
        reverse_probability: float = 1.0,
        mix: float = 1.0
    ):
        """
        Initialize reverse effect.

        Args:
            segment_duration: Duration of segments to reverse in seconds (None = reverse all)
            reverse_probability: Probability of reversing each segment (0.0 to 1.0)
            mix: Mix of reversed signal (0.0 to 1.0)
        """
        self.segment_duration = segment_duration
        self.reverse_probability = reverse_probability
        self.mix = mix

    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        if self.segment_duration is None:
            # Reverse entire signal
            wet = signal[::-1].copy()
        else:
            # Reverse segments
            segment_samples = int(self.segment_duration * sample_rate)
            wet = signal.copy()

            for i in range(0, len(wet), segment_samples):
                if np.random.random() < self.reverse_probability:
                    segment_end = min(i + segment_samples, len(wet))
                    wet[i:segment_end] = wet[i:segment_end][::-1]

        return signal * (1 - self.mix) + wet * self.mix


class BeatRepeat(Effect):
    """
    Beat repeat effect for rhythmic repetition and glitching.

    Captures and repeats beats at musical intervals with variations.
    """

    def __init__(
        self,
        tempo: float = 120.0,
        beat_division: float = 0.25,
        repeat_count: int = 4,
        gate: float = 1.0,
        pitch_shift: float = 0.0,
        decay: float = 0.9,
        mix: float = 1.0
    ):
        """
        Initialize beat repeat effect.

        Args:
            tempo: Tempo in BPM for beat synchronization
            beat_division: Beat division (0.25 = 16th, 0.5 = 8th, 1.0 = quarter)
            repeat_count: Number of repetitions
            gate: Gate/volume envelope for repeats (0.0 to 1.0)
            pitch_shift: Pitch shift in semitones for repeats
            decay: Volume decay for each repeat (0.0 to 1.0)
            mix: Mix of repeated signal (0.0 to 1.0)
        """
        self.tempo = tempo
        self.beat_division = beat_division
        self.repeat_count = repeat_count
        self.gate = gate
        self.pitch_shift = pitch_shift
        self.decay = decay
        self.mix = mix

    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        # Calculate beat duration
        beat_duration = 60.0 / self.tempo * self.beat_division
        beat_samples = int(beat_duration * sample_rate)

        wet = signal.copy()

        # Process in beat-sized chunks
        for i in range(0, len(signal), beat_samples * (self.repeat_count + 1)):
            # Capture beat
            beat_end = min(i + beat_samples, len(signal))
            beat = signal[i:beat_end].copy()

            if len(beat) == 0:
                break

            # Apply pitch shift if specified
            if self.pitch_shift != 0.0:
                pitch_ratio = 2.0 ** (self.pitch_shift / 12.0)
                indices = np.arange(len(beat)) / pitch_ratio
                indices = np.clip(indices, 0, len(beat) - 1).astype(int)
                beat = beat[indices]

            # Repeat beat with decay
            for r in range(1, self.repeat_count + 1):
                repeat_start = i + r * beat_samples
                repeat_end = min(repeat_start + len(beat), len(wet))

                if repeat_start >= len(wet):
                    break

                # Apply decay and gate
                volume = (self.decay ** r) * self.gate
                repeat_length = repeat_end - repeat_start
                wet[repeat_start:repeat_end] = beat[:repeat_length] * volume + wet[repeat_start:repeat_end] * (1 - volume)

        return signal * (1 - self.mix) + wet * self.mix


class EffectChain:
    """
    Chain multiple effects together for complex signal processing.
    """

    def __init__(self):
        self.effects = []

    def add_effect(self, effect: Effect) -> 'EffectChain':
        """
        Add an effect to the chain.

        Args:
            effect: Effect instance to add

        Returns:
            Self for method chaining
        """
        self.effects.append(effect)
        return self

    def apply(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply all effects in the chain sequentially.

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

    def clear(self) -> 'EffectChain':
        """Clear all effects from the chain."""
        self.effects = []
        return self
