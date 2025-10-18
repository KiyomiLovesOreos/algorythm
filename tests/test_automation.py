"""
Tests for automation and data sonification.
"""

import pytest
import numpy as np
from algorythm.automation import Automation, AutomationTrack, DataSonification
from algorythm.sequence import Scale


class TestAutomation:
    """Tests for parameter automation."""
    
    def test_automation_creation(self):
        """Test automation creation."""
        auto = Automation(start_value=0.0, end_value=1.0, duration=1.0)
        assert auto.start_value == 0.0
        assert auto.end_value == 1.0
        assert auto.duration == 1.0
    
    def test_automation_linear(self):
        """Test linear automation."""
        auto = Automation(0.0, 1.0, 1.0, curve_type='linear')
        
        assert abs(auto.get_value(0.0) - 0.0) < 0.01
        assert abs(auto.get_value(0.5) - 0.5) < 0.01
        assert abs(auto.get_value(1.0) - 1.0) < 0.01
    
    def test_automation_exponential(self):
        """Test exponential automation."""
        auto = Automation(0.0, 1.0, 1.0, curve_type='exponential')
        
        assert abs(auto.get_value(0.0) - 0.0) < 0.01
        assert auto.get_value(1.0) > 0.99
    
    def test_automation_bezier(self):
        """Test Bézier automation."""
        auto = Automation(0.0, 1.0, 1.0, curve_type='bezier', control_points=[0.5, 0.5])
        
        assert abs(auto.get_value(0.0) - 0.0) < 0.01
        assert abs(auto.get_value(1.0) - 1.0) < 0.01
    
    def test_automation_ease_in(self):
        """Test ease-in automation."""
        auto = Automation(0.0, 1.0, 1.0, curve_type='ease_in')
        
        # Ease-in should start slow
        assert auto.get_value(0.25) < 0.25
    
    def test_automation_ease_out(self):
        """Test ease-out automation."""
        auto = Automation(0.0, 1.0, 1.0, curve_type='ease_out')
        
        # Ease-out should start fast
        assert auto.get_value(0.25) > 0.25
    
    def test_automation_generate_curve(self):
        """Test generating full automation curve."""
        auto = Automation(0.0, 1.0, 1.0)
        curve = auto.generate_curve(num_points=10)
        
        assert len(curve) == 10
        assert curve[0] < curve[-1]
    
    def test_automation_fade_in(self):
        """Test fade-in preset."""
        auto = Automation.fade_in(duration=1.0, target_value=1.0)
        
        assert auto.start_value == 0.0
        assert auto.end_value == 1.0
        assert auto.curve_type == 'ease_in'
    
    def test_automation_fade_out(self):
        """Test fade-out preset."""
        auto = Automation.fade_out(duration=1.0, start_value=1.0)
        
        assert auto.start_value == 1.0
        assert auto.end_value == 0.0
        assert auto.curve_type == 'ease_out'


class TestAutomationTrack:
    """Tests for automation track."""
    
    def test_automation_track_creation(self):
        """Test automation track creation."""
        track = AutomationTrack()
        assert len(track.segments) == 0
    
    def test_automation_track_add_segment(self):
        """Test adding automation segments."""
        track = AutomationTrack()
        auto = Automation(0.0, 1.0, 1.0)
        
        track.add_segment(0.0, auto)
        assert len(track.segments) == 1
    
    def test_automation_track_get_value(self):
        """Test getting value from track."""
        track = AutomationTrack()
        auto1 = Automation(0.0, 1.0, 1.0)
        auto2 = Automation(1.0, 0.0, 1.0)
        
        track.add_segment(0.0, auto1)
        track.add_segment(1.0, auto2)
        
        assert abs(track.get_value(0.5) - 0.5) < 0.01
        assert abs(track.get_value(1.5) - 0.5) < 0.01


class TestDataSonification:
    """Tests for data sonification."""
    
    def test_data_sonification_creation(self):
        """Test data sonification creation."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        ds = DataSonification(data)
        
        assert len(ds.data) == 5
    
    def test_data_sonification_map_to_parameter(self):
        """Test mapping data to parameter."""
        data = [0.0, 50.0, 100.0]
        ds = DataSonification(data, param_range=(0.0, 1.0))
        
        # Map all data points
        mapped = ds.map_to_parameter()
        
        assert len(mapped) == 3
        assert mapped[0] < mapped[1] < mapped[2]
    
    def test_data_sonification_linear_scaling(self):
        """Test linear scaling."""
        data = [0.0, 50.0, 100.0]
        ds = DataSonification(data, param_range=(0.0, 1.0), scaling='linear')
        
        mapped = ds.map_to_parameter()
        assert abs(mapped[0] - 0.0) < 0.01
        assert abs(mapped[1] - 0.5) < 0.01
        assert abs(mapped[2] - 1.0) < 0.01
    
    def test_data_sonification_to_pitch_sequence(self):
        """Test converting data to pitch sequence."""
        data = [1.0, 2.0, 3.0]
        ds = DataSonification(data)
        
        scale = Scale.major('C', 4)
        pitches = ds.to_pitch_sequence(scale)
        
        assert len(pitches) == 3
        assert all(isinstance(p, float) for p in pitches)
    
    def test_data_sonification_to_rhythm_pattern(self):
        """Test converting data to rhythm pattern."""
        data = [1.0, 5.0, 10.0]
        ds = DataSonification(data)
        
        rhythm = ds.to_rhythm_pattern(min_duration=0.5, max_duration=2.0)
        
        assert len(rhythm) == 3
        assert all(0.5 <= r <= 2.0 for r in rhythm)
    
    def test_data_sonification_to_volume_envelope(self):
        """Test converting data to volume envelope."""
        data = [1.0, 5.0, 10.0]
        ds = DataSonification(data)
        
        volumes = ds.to_volume_envelope(min_volume=0.0, max_volume=1.0)
        
        assert len(volumes) == 3
        assert all(0.0 <= v <= 1.0 for v in volumes)
    
    def test_data_sonification_from_array(self):
        """Test creating from numpy array."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ds = DataSonification.from_array(data)
        
        assert len(ds.data) == 5
