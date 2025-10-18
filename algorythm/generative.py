"""
algorythm.generative - Generative structure tools

This module provides advanced algorithmic methods for creating unique patterns
including L-Systems and Cellular Automata.
"""

from typing import Dict, List, Callable, Optional
import numpy as np
from algorythm.sequence import Motif, Scale


class LSystem:
    """
    L-System (Lindenmayer System) generator for creating fractal-like melodies and rhythms.
    
    Uses production rules to generate complex patterns from simple axioms.
    """
    
    def __init__(
        self,
        axiom: str,
        rules: Dict[str, str],
        iterations: int = 3
    ):
        """
        Initialize an L-System generator.
        
        Args:
            axiom: Starting string/axiom
            rules: Dictionary mapping symbols to replacement strings
            iterations: Number of iterations to apply
        """
        self.axiom = axiom
        self.rules = rules
        self.iterations = iterations
        self.current_string = axiom
    
    def generate(self) -> str:
        """
        Generate L-System string by applying production rules.
        
        Returns:
            Generated string after all iterations
        """
        self.current_string = self.axiom
        
        for _ in range(self.iterations):
            new_string = ""
            for char in self.current_string:
                # Apply rule if exists, otherwise keep the character
                new_string += self.rules.get(char, char)
            self.current_string = new_string
        
        return self.current_string
    
    def to_motif(
        self,
        symbol_map: Dict[str, int],
        scale: Optional[Scale] = None,
        default_duration: float = 1.0
    ) -> Motif:
        """
        Convert L-System string to a musical motif.
        
        Args:
            symbol_map: Dictionary mapping symbols to scale degrees or intervals
            scale: Musical scale to use
            default_duration: Default note duration
            
        Returns:
            Motif generated from the L-System
        """
        if not self.current_string:
            self.generate()
        
        intervals = []
        durations = []
        
        for char in self.current_string:
            if char in symbol_map:
                intervals.append(symbol_map[char])
                durations.append(default_duration)
        
        scale = scale or Scale.major('C', 4)
        
        return Motif(intervals, scale, durations)
    
    @classmethod
    def fractal_melody(cls, iterations: int = 3) -> 'LSystem':
        """
        Create a fractal melody L-System.
        
        Args:
            iterations: Number of iterations
            
        Returns:
            Configured L-System instance
        """
        return cls(
            axiom='A',
            rules={
                'A': 'AB',
                'B': 'AC',
                'C': 'A'
            },
            iterations=iterations
        )
    
    @classmethod
    def cantor_rhythm(cls, iterations: int = 3) -> 'LSystem':
        """
        Create a Cantor set rhythm L-System.
        
        Args:
            iterations: Number of iterations
            
        Returns:
            Configured L-System instance
        """
        return cls(
            axiom='X',
            rules={
                'X': 'XYX',
                'Y': 'YYY'
            },
            iterations=iterations
        )


class CellularAutomata:
    """
    Cellular Automata for generating evolving soundscapes and rhythmic textures.
    
    Based on grid-based rules (similar to Conway's Game of Life).
    """
    
    def __init__(
        self,
        width: int = 16,
        height: int = 8,
        rule: Optional[Callable] = None,
        initial_state: Optional[np.ndarray] = None
    ):
        """
        Initialize a cellular automaton.
        
        Args:
            width: Grid width
            height: Grid height (number of generations to compute)
            rule: Rule function for cell evolution
            initial_state: Initial grid state (if None, randomized)
        """
        self.width = width
        self.height = height
        self.rule = rule or self._rule_110
        
        # Initialize grid
        if initial_state is not None:
            self.grid = initial_state.copy()
        else:
            # Random initial state for first row
            self.grid = np.zeros((height, width), dtype=int)
            self.grid[0] = np.random.randint(0, 2, width)
    
    def _rule_110(self, left: int, center: int, right: int) -> int:
        """
        Elementary cellular automaton rule 110.
        
        Args:
            left: Left neighbor state
            center: Center cell state
            right: Right neighbor state
            
        Returns:
            New cell state
        """
        # Rule 110 lookup table
        lookup = {
            (1, 1, 1): 0,
            (1, 1, 0): 1,
            (1, 0, 1): 1,
            (1, 0, 0): 0,
            (0, 1, 1): 1,
            (0, 1, 0): 1,
            (0, 0, 1): 1,
            (0, 0, 0): 0,
        }
        return lookup.get((left, center, right), 0)
    
    def _conway_rule(self, grid: np.ndarray, row: int, col: int) -> int:
        """
        Conway's Game of Life rule.
        
        Args:
            grid: Current grid state
            row: Row index
            col: Column index
            
        Returns:
            New cell state
        """
        # Count live neighbors (with wrapping)
        neighbors = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr = (row + dr) % grid.shape[0]
                nc = (col + dc) % grid.shape[1]
                neighbors += grid[nr, nc]
        
        # Apply Conway's rules
        if grid[row, col] == 1:
            # Cell is alive
            return 1 if neighbors in [2, 3] else 0
        else:
            # Cell is dead
            return 1 if neighbors == 3 else 0
    
    def evolve(self) -> np.ndarray:
        """
        Evolve the cellular automaton through all generations.
        
        Returns:
            Complete grid after evolution
        """
        for row in range(1, self.height):
            for col in range(self.width):
                # Get neighbors (with wrapping)
                left = self.grid[row - 1, (col - 1) % self.width]
                center = self.grid[row - 1, col]
                right = self.grid[row - 1, (col + 1) % self.width]
                
                # Apply rule
                self.grid[row, col] = self.rule(left, center, right)
        
        return self.grid
    
    def to_rhythm_pattern(self, row: int = -1) -> List[float]:
        """
        Convert a row of the automaton to a rhythm pattern.
        
        Args:
            row: Row index to convert (-1 for last row)
            
        Returns:
            List of beat durations (1.0 for active, 0.0 for rest)
        """
        if row == -1:
            row = self.height - 1
        
        return [1.0 if cell == 1 else 0.0 for cell in self.grid[row]]
    
    def to_motif(
        self,
        row: int = -1,
        scale: Optional[Scale] = None,
        rest_value: int = -1
    ) -> Motif:
        """
        Convert a row of the automaton to a musical motif.
        
        Args:
            row: Row index to convert (-1 for last row)
            scale: Musical scale to use
            rest_value: Interval value to use for rests (inactive cells)
            
        Returns:
            Motif generated from the cellular automaton
        """
        if row == -1:
            row = self.height - 1
        
        scale = scale or Scale.major('C', 4)
        
        # Map cell states to scale degrees
        intervals = []
        durations = []
        
        for i, cell in enumerate(self.grid[row]):
            if cell == 1:
                # Active cell: map position to scale degree
                degree = i % 7  # Map to scale degree within one octave
                intervals.append(degree)
                durations.append(1.0)
            else:
                # Inactive cell: rest or silent note
                if rest_value >= 0:
                    intervals.append(rest_value)
                    durations.append(0.25)  # Short rest
        
        if not intervals:
            intervals = [0]
            durations = [1.0]
        
        return Motif(intervals, scale, durations)
    
    @classmethod
    def from_seed(cls, seed: int, width: int = 16, height: int = 8) -> 'CellularAutomata':
        """
        Create a cellular automaton from a random seed.
        
        Args:
            seed: Random seed for reproducibility
            width: Grid width
            height: Grid height
            
        Returns:
            New CellularAutomata instance
        """
        np.random.seed(seed)
        return cls(width=width, height=height)
