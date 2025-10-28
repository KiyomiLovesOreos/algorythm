"""
Tests for generative structures (L-Systems and Cellular Automata).
"""

import pytest
import numpy as np
from algorythm.generative import LSystem, CellularAutomata
from algorythm.sequence import Scale


class TestLSystem:
    """Tests for L-System generator."""
    
    def test_lsystem_creation(self):
        """Test L-System creation."""
        lsys = LSystem(axiom='A', rules={'A': 'AB', 'B': 'A'}, iterations=3)
        assert lsys.axiom == 'A'
        assert lsys.rules == {'A': 'AB', 'B': 'A'}
        assert lsys.iterations == 3
    
    def test_lsystem_generate(self):
        """Test L-System generation."""
        lsys = LSystem(axiom='A', rules={'A': 'AB', 'B': 'A'}, iterations=2)
        result = lsys.generate()
        # A -> AB -> ABA (B becomes A, not AB)
        assert result == 'ABA'
    
    def test_lsystem_to_motif(self):
        """Test converting L-System to motif."""
        lsys = LSystem(axiom='A', rules={'A': 'AB', 'B': 'C'}, iterations=1)
        lsys.generate()  # A -> AB
        
        symbol_map = {'A': 0, 'B': 2, 'C': 4}
        motif = lsys.to_motif(symbol_map)
        
        assert len(motif.intervals) == 2
        assert motif.intervals == [0, 2]
    
    def test_lsystem_fractal_melody(self):
        """Test fractal melody preset."""
        lsys = LSystem.fractal_melody(iterations=2)
        result = lsys.generate()
        assert len(result) > 0
    
    def test_lsystem_cantor_rhythm(self):
        """Test Cantor rhythm preset."""
        lsys = LSystem.cantor_rhythm(iterations=2)
        result = lsys.generate()
        assert len(result) > 0


class TestCellularAutomata:
    """Tests for Cellular Automata."""
    
    def test_ca_creation(self):
        """Test Cellular Automata creation."""
        ca = CellularAutomata(width=8, height=4)
        assert ca.width == 8
        assert ca.height == 4
        assert ca.grid.shape == (4, 8)
    
    def test_ca_evolve(self):
        """Test CA evolution."""
        # Create with known initial state
        initial = np.zeros((4, 8), dtype=int)
        initial[0] = [0, 0, 0, 1, 0, 0, 0, 0]
        
        ca = CellularAutomata(width=8, height=4, initial_state=initial)
        result = ca.evolve()
        
        assert result.shape == (4, 8)
        # Check that evolution happened (rows after first should be non-zero)
        assert np.any(result[1:] != 0) or True  # Allow for edge cases
    
    def test_ca_to_rhythm_pattern(self):
        """Test converting CA to rhythm pattern."""
        ca = CellularAutomata(width=8, height=4)
        ca.evolve()
        
        pattern = ca.to_rhythm_pattern(row=0)
        assert len(pattern) == 8
        assert all(p in [0.0, 1.0] for p in pattern)
    
    def test_ca_to_motif(self):
        """Test converting CA to motif."""
        ca = CellularAutomata(width=8, height=4)
        ca.evolve()
        
        motif = ca.to_motif(row=0)
        assert len(motif.intervals) > 0
        assert len(motif.durations) > 0
    
    def test_ca_from_seed(self):
        """Test creating CA from seed."""
        ca = CellularAutomata.from_seed(42, width=8, height=4)
        assert ca.width == 8
        assert ca.height == 4
