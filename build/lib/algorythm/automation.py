"""
algorythm.automation - Parameter automation and data sonification

This module provides tools for automating parameters over time and mapping
data to musical parameters.
"""

from typing import List, Callable, Optional, Literal, Any, Dict
import numpy as np


class Automation:
    """
    Parameter automation for smooth time-based changes.
    
    Supports various curve types including linear, exponential, and Bézier.
    """
    
    def __init__(
        self,
        start_value: float,
        end_value: float,
        duration: float,
        curve_type: Literal['linear', 'exponential', 'bezier', 'ease_in', 'ease_out'] = 'linear',
        control_points: Optional[List[float]] = None
    ):
        """
        Initialize a parameter automation.
        
        Args:
            start_value: Starting parameter value
            end_value: Ending parameter value
            duration: Duration in beats or seconds
            curve_type: Type of interpolation curve
            control_points: Control points for Bézier curves (0.0 to 1.0)
        """
        self.start_value = start_value
        self.end_value = end_value
        self.duration = duration
        self.curve_type = curve_type
        self.control_points = control_points or [0.5, 0.5]
    
    def get_value(self, time: float) -> float:
        """
        Get parameter value at a specific time.
        
        Args:
            time: Time position (0.0 to duration)
            
        Returns:
            Interpolated parameter value
        """
        # Normalize time to 0.0 - 1.0
        if self.duration == 0:
            return self.end_value
        
        t = np.clip(time / self.duration, 0.0, 1.0)
        
        if self.curve_type == 'linear':
            return self._linear(t)
        elif self.curve_type == 'exponential':
            return self._exponential(t)
        elif self.curve_type == 'bezier':
            return self._bezier(t)
        elif self.curve_type == 'ease_in':
            return self._ease_in(t)
        elif self.curve_type == 'ease_out':
            return self._ease_out(t)
        else:
            return self._linear(t)
    
    def _linear(self, t: float) -> float:
        """Linear interpolation."""
        return self.start_value + (self.end_value - self.start_value) * t
    
    def _exponential(self, t: float) -> float:
        """Exponential interpolation."""
        # Use exponential curve (base 2)
        exp_t = (np.power(2, t) - 1) / 1.0  # Normalized exponential
        return self.start_value + (self.end_value - self.start_value) * exp_t
    
    def _bezier(self, t: float) -> float:
        """Cubic Bézier interpolation."""
        # Simple cubic Bézier with two control points
        p0 = 0.0
        p1 = self.control_points[0]
        p2 = self.control_points[1]
        p3 = 1.0
        
        # Cubic Bézier formula
        bezier_t = (
            (1 - t)**3 * p0 +
            3 * (1 - t)**2 * t * p1 +
            3 * (1 - t) * t**2 * p2 +
            t**3 * p3
        )
        
        return self.start_value + (self.end_value - self.start_value) * bezier_t
    
    def _ease_in(self, t: float) -> float:
        """Ease-in curve (accelerating)."""
        ease_t = t * t
        return self.start_value + (self.end_value - self.start_value) * ease_t
    
    def _ease_out(self, t: float) -> float:
        """Ease-out curve (decelerating)."""
        ease_t = 1 - (1 - t) * (1 - t)
        return self.start_value + (self.end_value - self.start_value) * ease_t
    
    def generate_curve(self, num_points: int = 100) -> np.ndarray:
        """
        Generate the full automation curve.
        
        Args:
            num_points: Number of points to generate
            
        Returns:
            Array of parameter values
        """
        times = np.linspace(0, self.duration, num_points)
        return np.array([self.get_value(t) for t in times])
    
    @classmethod
    def fade_in(cls, duration: float, target_value: float = 1.0) -> 'Automation':
        """
        Create a fade-in automation.
        
        Args:
            duration: Fade duration
            target_value: Target value at end of fade
            
        Returns:
            Automation instance
        """
        return cls(0.0, target_value, duration, 'ease_in')
    
    @classmethod
    def fade_out(cls, duration: float, start_value: float = 1.0) -> 'Automation':
        """
        Create a fade-out automation.
        
        Args:
            duration: Fade duration
            start_value: Starting value
            
        Returns:
            Automation instance
        """
        return cls(start_value, 0.0, duration, 'ease_out')


class AutomationTrack:
    """
    Track for managing multiple automation segments.
    """
    
    def __init__(self):
        """Initialize an automation track."""
        self.segments: List[tuple[float, Automation]] = []  # (start_time, automation)
    
    def add_segment(self, start_time: float, automation: Automation) -> 'AutomationTrack':
        """
        Add an automation segment.
        
        Args:
            start_time: Start time of the automation
            automation: Automation instance
            
        Returns:
            Self for method chaining
        """
        self.segments.append((start_time, automation))
        # Sort by start time
        self.segments.sort(key=lambda x: x[0])
        return self
    
    def get_value(self, time: float, default_value: float = 0.0) -> float:
        """
        Get parameter value at a specific time.
        
        Args:
            time: Time position
            default_value: Default value if no automation at this time
            
        Returns:
            Parameter value
        """
        for start_time, automation in self.segments:
            end_time = start_time + automation.duration
            if start_time <= time <= end_time:
                return automation.get_value(time - start_time)
        
        return default_value


class DataSonification:
    """
    Engine for mapping arbitrary numeric datasets to musical parameters.
    """
    
    def __init__(
        self,
        data: List[float],
        param_range: tuple[float, float] = (0.0, 1.0),
        scaling: Literal['linear', 'logarithmic', 'exponential'] = 'linear'
    ):
        """
        Initialize a data sonification engine.
        
        Args:
            data: Numeric data to sonify
            param_range: Target parameter range (min, max)
            scaling: Scaling method for mapping data
        """
        self.data = np.array(data)
        self.param_range = param_range
        self.scaling = scaling
        
        # Normalize data to 0-1 range
        self.data_min = np.min(self.data)
        self.data_max = np.max(self.data)
        self.data_range = self.data_max - self.data_min
    
    def map_to_parameter(self, index: Optional[int] = None) -> float:
        """
        Map a data point to a parameter value.
        
        Args:
            index: Data point index (None returns all mapped values)
            
        Returns:
            Mapped parameter value
        """
        if index is not None:
            normalized = self._normalize(self.data[index])
            return self._scale(normalized)
        else:
            normalized = self._normalize(self.data)
            return self._scale(normalized)
    
    def _normalize(self, value: Any) -> Any:
        """Normalize data to 0-1 range."""
        if self.data_range == 0:
            return 0.5
        return (value - self.data_min) / self.data_range
    
    def _scale(self, normalized: Any) -> Any:
        """Scale normalized value to target parameter range."""
        if self.scaling == 'linear':
            scaled = normalized
        elif self.scaling == 'logarithmic':
            # Logarithmic scaling (avoid log(0))
            scaled = np.log1p(normalized) / np.log1p(1.0)
        elif self.scaling == 'exponential':
            # Exponential scaling
            scaled = (np.exp(normalized) - 1) / (np.exp(1.0) - 1)
        else:
            scaled = normalized
        
        # Map to target range
        param_min, param_max = self.param_range
        return param_min + scaled * (param_max - param_min)
    
    def to_pitch_sequence(self, scale: Optional[Any] = None) -> List[float]:
        """
        Convert data to a pitch sequence.
        
        Args:
            scale: Musical scale to use (algorythm.sequence.Scale)
            
        Returns:
            List of frequencies
        """
        if scale is None:
            # Import here to avoid circular dependency
            from algorythm.sequence import Scale
            scale = Scale.major('C', 4)
        
        # Map data to scale degrees (0-7 for major scale)
        num_degrees = 7  # Assuming major/minor scale
        degrees = np.round(self.map_to_parameter() * num_degrees).astype(int)
        
        # Convert to frequencies
        return [scale.get_frequency(int(degree)) for degree in degrees]
    
    def to_rhythm_pattern(self, min_duration: float = 0.25, max_duration: float = 2.0) -> List[float]:
        """
        Convert data to a rhythm pattern.
        
        Args:
            min_duration: Minimum note duration
            max_duration: Maximum note duration
            
        Returns:
            List of durations
        """
        old_range = self.param_range
        self.param_range = (min_duration, max_duration)
        durations = self.map_to_parameter()
        self.param_range = old_range
        return durations.tolist() if isinstance(durations, np.ndarray) else [durations]
    
    def to_volume_envelope(self, min_volume: float = 0.1, max_volume: float = 1.0) -> List[float]:
        """
        Convert data to a volume envelope.
        
        Args:
            min_volume: Minimum volume level
            max_volume: Maximum volume level
            
        Returns:
            List of volume values
        """
        old_range = self.param_range
        self.param_range = (min_volume, max_volume)
        volumes = self.map_to_parameter()
        self.param_range = old_range
        return volumes.tolist() if isinstance(volumes, np.ndarray) else [volumes]
    
    @classmethod
    def from_csv(cls, file_path: str, column: int = 0, **kwargs) -> 'DataSonification':
        """
        Create a data sonification from a CSV file.
        
        Args:
            file_path: Path to CSV file
            column: Column index to use
            **kwargs: Additional arguments for DataSonification
            
        Returns:
            DataSonification instance
        """
        import csv
        data = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    data.append(float(row[column]))
                except (ValueError, IndexError):
                    continue
        
        return cls(data, **kwargs)
    
    @classmethod
    def from_array(cls, data: np.ndarray, **kwargs) -> 'DataSonification':
        """
        Create a data sonification from a numpy array.
        
        Args:
            data: Numpy array of data
            **kwargs: Additional arguments for DataSonification
            
        Returns:
            DataSonification instance
        """
        return cls(data.flatten().tolist(), **kwargs)
