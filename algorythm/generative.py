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


class ConstraintBasedComposer:
    """
    Constraint-Based Composition (CBC) engine.
    
    Defines a formal set of musical rules and generates solutions that satisfy
    all constraints (e.g., voice-leading rules, rhythmic limitations).
    """
    
    def __init__(self, scale: Optional[Scale] = None):
        """
        Initialize a constraint-based composer.
        
        Args:
            scale: Musical scale to use for composition
        """
        self.scale = scale or Scale.major('C', 4)
        self.constraints: List[Callable[[List[int]], bool]] = []
    
    def add_constraint(self, constraint_func: Callable[[List[int]], bool]) -> 'ConstraintBasedComposer':
        """
        Add a constraint function that must be satisfied.
        
        Args:
            constraint_func: Function that takes a melody (list of intervals) and returns True if valid
            
        Returns:
            Self for method chaining
        """
        self.constraints.append(constraint_func)
        return self
    
    def no_large_leaps(self, max_interval: int = 5) -> 'ConstraintBasedComposer':
        """
        Add constraint: no melodic leaps larger than max_interval.
        
        Args:
            max_interval: Maximum allowed interval between consecutive notes
            
        Returns:
            Self for method chaining
        """
        def constraint(melody: List[int]) -> bool:
            for i in range(len(melody) - 1):
                if abs(melody[i + 1] - melody[i]) > max_interval:
                    return False
            return True
        
        return self.add_constraint(constraint)
    
    def prefer_stepwise_motion(self) -> 'ConstraintBasedComposer':
        """
        Add constraint: prefer stepwise motion (intervals of 1-2 semitones).
        
        Returns:
            Self for method chaining
        """
        def constraint(melody: List[int]) -> bool:
            stepwise_count = 0
            for i in range(len(melody) - 1):
                if abs(melody[i + 1] - melody[i]) <= 2:
                    stepwise_count += 1
            # At least 70% stepwise motion
            return stepwise_count >= len(melody) * 0.7
        
        return self.add_constraint(constraint)
    
    def no_repeated_notes(self) -> 'ConstraintBasedComposer':
        """
        Add constraint: no consecutive repeated notes.
        
        Returns:
            Self for method chaining
        """
        def constraint(melody: List[int]) -> bool:
            for i in range(len(melody) - 1):
                if melody[i] == melody[i + 1]:
                    return False
            return True
        
        return self.add_constraint(constraint)
    
    def ending_on_tonic(self) -> 'ConstraintBasedComposer':
        """
        Add constraint: melody must end on the tonic (scale degree 0).
        
        Returns:
            Self for method chaining
        """
        def constraint(melody: List[int]) -> bool:
            return len(melody) > 0 and melody[-1] == 0
        
        return self.add_constraint(constraint)
    
    def check_constraints(self, melody: List[int]) -> bool:
        """
        Check if a melody satisfies all constraints.
        
        Args:
            melody: List of scale degrees
            
        Returns:
            True if all constraints are satisfied
        """
        return all(constraint(melody) for constraint in self.constraints)
    
    def generate(
        self,
        length: int = 8,
        max_attempts: int = 1000
    ) -> Optional[Motif]:
        """
        Generate a melody that satisfies all constraints.
        
        Args:
            length: Desired melody length
            max_attempts: Maximum number of generation attempts
            
        Returns:
            Generated motif or None if no solution found
        """
        for _ in range(max_attempts):
            # Generate random melody
            melody = [np.random.randint(-7, 8) for _ in range(length)]
            
            # Check constraints
            if self.check_constraints(melody):
                return Motif(melody, self.scale)
        
        return None


class GeneticAlgorithmImproviser:
    """
    Genetic Algorithm (GA) improviser for evolutionary music generation.
    
    Defines a numerical "fitness function" and uses evolutionary processes
    (mutation, crossover) to iteratively generate motifs that optimize toward
    the defined musical goal.
    """
    
    def __init__(
        self,
        fitness_func: Callable[[List[int]], float],
        scale: Optional[Scale] = None,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7
    ):
        """
        Initialize a genetic algorithm improviser.
        
        Args:
            fitness_func: Function that scores a melody (higher is better)
            scale: Musical scale to use
            population_size: Number of individuals in population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
        """
        self.fitness_func = fitness_func
        self.scale = scale or Scale.major('C', 4)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population: List[List[int]] = []
    
    def initialize_population(self, length: int, range_min: int = -7, range_max: int = 8) -> None:
        """
        Initialize random population.
        
        Args:
            length: Length of each melody
            range_min: Minimum interval value
            range_max: Maximum interval value
        """
        self.population = [
            [np.random.randint(range_min, range_max) for _ in range(length)]
            for _ in range(self.population_size)
        ]
    
    def evaluate_fitness(self, individual: List[int]) -> float:
        """
        Evaluate fitness of an individual.
        
        Args:
            individual: Melody to evaluate
            
        Returns:
            Fitness score
        """
        return self.fitness_func(individual)
    
    def select_parents(self) -> tuple:
        """
        Select two parents using tournament selection.
        
        Returns:
            Tuple of two parent melodies
        """
        # Tournament selection
        tournament_size = 5
        
        def tournament() -> List[int]:
            competitors = [self.population[np.random.randint(0, len(self.population))]
                          for _ in range(tournament_size)]
            return max(competitors, key=self.evaluate_fitness)
        
        return tournament(), tournament()
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> tuple:
        """
        Perform single-point crossover between two parents.
        
        Args:
            parent1: First parent melody
            parent2: Second parent melody
            
        Returns:
            Tuple of two offspring melodies
        """
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        point = np.random.randint(1, len(parent1))
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        
        return child1, child2
    
    def mutate(self, individual: List[int]) -> List[int]:
        """
        Mutate an individual by randomly changing some notes.
        
        Args:
            individual: Melody to mutate
            
        Returns:
            Mutated melody
        """
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                mutated[i] += np.random.randint(-2, 3)
                mutated[i] = np.clip(mutated[i], -14, 14)
        
        return mutated
    
    def evolve(self, generations: int = 100) -> Motif:
        """
        Evolve the population over multiple generations.
        
        Args:
            generations: Number of generations to evolve
            
        Returns:
            Best motif found
        """
        for generation in range(generations):
            # Evaluate all individuals
            fitness_scores = [self.evaluate_fitness(ind) for ind in self.population]
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individual
            best_idx = np.argmax(fitness_scores)
            new_population.append(self.population[best_idx])
            
            # Generate rest of new population
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents()
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            self.population = new_population
        
        # Return best individual as motif
        fitness_scores = [self.evaluate_fitness(ind) for ind in self.population]
        best_idx = np.argmax(fitness_scores)
        best_melody = self.population[best_idx]
        
        return Motif(best_melody, self.scale)
    
    @staticmethod
    def fitness_ascending() -> Callable[[List[int]], float]:
        """
        Fitness function that prefers ascending melodies.
        
        Returns:
            Fitness function
        """
        def fitness(melody: List[int]) -> float:
            ascending_count = sum(1 for i in range(len(melody) - 1) if melody[i + 1] > melody[i])
            return ascending_count / max(len(melody) - 1, 1)
        
        return fitness
    
    @staticmethod
    def fitness_contour(target_contour: List[int]) -> Callable[[List[int]], float]:
        """
        Fitness function that matches a target melodic contour.
        
        Args:
            target_contour: List of +1 (ascending), -1 (descending), 0 (same)
            
        Returns:
            Fitness function
        """
        def fitness(melody: List[int]) -> float:
            if len(melody) < 2:
                return 0.0
            
            contour = []
            for i in range(len(melody) - 1):
                if melody[i + 1] > melody[i]:
                    contour.append(1)
                elif melody[i + 1] < melody[i]:
                    contour.append(-1)
                else:
                    contour.append(0)
            
            # Calculate similarity to target
            min_len = min(len(contour), len(target_contour))
            matches = sum(1 for i in range(min_len) if contour[i] == target_contour[i])
            
            return matches / max(len(target_contour), 1)
        
        return fitness
