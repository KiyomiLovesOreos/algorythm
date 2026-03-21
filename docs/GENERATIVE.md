# Generative Music Guide

Create algorithmic and generative music with Algorythm.

## What is Generative Music?

Generative music uses algorithms and rules to create musical patterns. Instead of composing every note, you define the system and let it generate music.

## L-Systems

L-Systems (Lindenmayer Systems) use string rewriting rules to generate patterns.

### Basic L-System

```python
from algorythm import LSystem

# Create L-system
lsystem = LSystem(
    axiom='A',              # Starting string
    rules={
        'A': 'AB',          # A becomes AB
        'B': 'A'            # B becomes A
    },
    iterations=4            # Number of generations
)

# Generate pattern
pattern = lsystem.generate()
print(pattern)  # 'ABAABABAABAAB'
```

### Musical L-System

Convert L-system output to music:

```python
from algorythm import LSystem, Scale, Motif, Composition, SynthPresets

# Define L-system
lsystem = LSystem(
    axiom='F',
    rules={
        'F': 'F+G',
        'G': 'F-G'
    },
    iterations=3
)

# Generate pattern
pattern = lsystem.generate()

# Map symbols to notes
scale = Scale.pentatonic('C')
note_map = {
    'F': 0,    # Root
    'G': 2,    # Second degree
    '+': 4,    # Up
    '-': -2    # Down
}

# Convert to melody
notes = []
current_degree = 0
for char in pattern:
    if char in note_map:
        if char in ['F', 'G']:
            notes.append(current_degree)
        else:
            current_degree += note_map[char]
            current_degree = max(-7, min(14, current_degree))  # Keep in range

# Create composition
song = Composition(tempo=120)
song.add_track('melody', SynthPresets.pluck())

melody = Motif.from_intervals(notes, scale=scale, duration=0.25)
song.play_motif(melody, start=0.0, track='melody')
song.render('lsystem_melody.wav')
```

### Pre-built Musical L-Systems

```python
# Fractal melody
lsystem = LSystem.fractal_melody()

# Growing pattern
lsystem = LSystem.growing_pattern()

# Branching structure
lsystem = LSystem.branching()
```

## Cellular Automata

Generate rhythmic patterns using cellular automata.

### Basic Cellular Automata

```python
from algorythm import CellularAutomata

# Create cellular automaton
ca = CellularAutomata(
    rule=30,           # Rule number (0-255)
    width=16,          # Pattern width
    generations=8      # Number of steps
)

# Generate pattern
pattern = ca.generate()

# Pattern is 2D array: pattern[time][position]
# Each cell is 0 (off) or 1 (on)
```

### Musical Cellular Automata

```python
from algorythm import CellularAutomata, Composition, SynthPresets

ca = CellularAutomata(rule=30, width=16, generations=16)
pattern = ca.generate()

song = Composition(tempo=120)

# Map to drum instruments
kick = song.add_track('kick', SynthPresets.kick())
snare = song.add_track('snare', SynthPresets.snare())
hihat = song.add_track('hihat', SynthPresets.hihat())

# Convert pattern to drum hits
for time_step, row in enumerate(pattern):
    time = time_step * 0.25  # Each step is 0.25 seconds
    
    if row[0]:  # Kick on position 0
        song.play_note(60, 0.2, start=time, track='kick')
    if row[4]:  # Snare on position 4
        song.play_note(60, 0.2, start=time, track='snare')
    if row[8]:  # Hi-hat on position 8
        song.play_note(60, 0.1, start=time, track='hihat')

song.render('ca_drums.wav')
```

### Interesting Rules

Try these rules for different patterns:
- Rule 30: Chaotic, random-seeming
- Rule 90: Fractal, symmetrical
- Rule 110: Complex, computational
- Rule 184: Traffic-like patterns

## Constraint-Based Composition

Generate music following musical rules.

### Creating Constraints

```python
from algorythm import ConstraintBasedComposer, Scale

composer = ConstraintBasedComposer(
    scale=Scale.major('C'),
    constraints={
        'min_interval': 1,      # Minimum step between notes
        'max_interval': 5,      # Maximum step between notes
        'prefer_steps': True,   # Prefer stepwise motion
        'avoid_leaps': True,    # Avoid large jumps
        'contour': 'arch'       # Overall shape: 'arch', 'ascending', 'descending'
    }
)

# Generate melody
melody = composer.compose(
    length=16,          # Number of notes
    start_degree=0      # Starting scale degree
)
```

### Contour Types

```python
# Arch: goes up then down
composer.constraints['contour'] = 'arch'

# Ascending: generally moves up
composer.constraints['contour'] = 'ascending'

# Descending: generally moves down
composer.constraints['contour'] = 'descending'

# Random: no specific contour
composer.constraints['contour'] = None
```

### Harmonic Constraints

```python
# Emphasize chord tones
composer.constraints['chord_tones'] = [0, 2, 4]  # Triad degrees
composer.constraints['chord_weight'] = 0.7       # Probability of chord tones

# Avoid certain intervals
composer.constraints['forbidden_intervals'] = [6]  # No tritones

# Preferred intervals
composer.constraints['preferred_intervals'] = [1, 2]  # Steps
```

## Genetic Algorithms

Evolve melodies through selection and mutation.

### Basic Genetic Algorithm

```python
from algorythm import GeneticAlgorithmImproviser, Scale

improviser = GeneticAlgorithmImproviser(
    scale=Scale.minor('A'),
    population_size=20,     # Number of melodies per generation
    mutation_rate=0.1,      # Probability of mutation
    generations=10          # Number of iterations
)

# Define fitness function (what makes a "good" melody)
def fitness(melody):
    score = 0
    
    # Prefer moderate interval sizes
    for i in range(len(melody) - 1):
        interval = abs(melody[i+1] - melody[i])
        if interval <= 3:
            score += 2
        elif interval <= 5:
            score += 1
    
    # Prefer melodies that return to root
    if melody[-1] == 0:
        score += 5
    
    return score

# Evolve melody
best_melody = improviser.evolve(
    length=8,
    fitness_function=fitness
)
```

### Fitness Functions

Different fitness functions create different styles:

```python
# Smooth, stepwise melody
def smooth_fitness(melody):
    score = 0
    for i in range(len(melody) - 1):
        interval = abs(melody[i+1] - melody[i])
        score += max(0, 5 - interval)
    return score

# Jumpy, wide intervals
def jumpy_fitness(melody):
    score = 0
    for i in range(len(melody) - 1):
        interval = abs(melody[i+1] - melody[i])
        if interval >= 4:
            score += 3
    return score

# Ascending melody
def ascending_fitness(melody):
    score = 0
    for i in range(len(melody) - 1):
        if melody[i+1] > melody[i]:
            score += 2
    return score
```

## Markov Chains

Generate patterns based on probability transitions.

### Simple Markov Chain

```python
import random

# Define transition probabilities
transitions = {
    0: [2, 4],      # From root, go to 2nd or 4th degree
    2: [0, 4, 5],   # From 2nd, go to root, 4th, or 5th
    4: [2, 5, 7],   # From 4th...
    5: [4, 7],
    7: [0, 5]
}

def generate_markov_melody(length, start_note=0):
    melody = [start_note]
    current = start_note
    
    for _ in range(length - 1):
        if current in transitions:
            next_note = random.choice(transitions[current])
            melody.append(next_note)
            current = next_note
        else:
            break
    
    return melody

# Generate
melody_notes = generate_markov_melody(16)
```

### Weighted Transitions

```python
# Transitions with probabilities
weighted_transitions = {
    0: [(2, 0.5), (4, 0.3), (5, 0.2)],  # (note, probability)
    2: [(0, 0.3), (4, 0.4), (5, 0.3)],
    4: [(2, 0.4), (5, 0.3), (7, 0.3)],
    5: [(4, 0.5), (7, 0.5)],
    7: [(0, 0.6), (5, 0.4)]
}

def generate_weighted_melody(length, start_note=0):
    melody = [start_note]
    current = start_note
    
    for _ in range(length - 1):
        if current in weighted_transitions:
            choices, weights = zip(*weighted_transitions[current])
            next_note = random.choices(choices, weights=weights)[0]
            melody.append(next_note)
            current = next_note
        else:
            break
    
    return melody
```

## Random with Constraints

Simple but effective random generation.

### Constrained Random

```python
import random
from algorythm import Scale, Motif, Composition, SynthPresets

scale = Scale.minor('D')

def random_melody_in_range(length, min_degree=-7, max_degree=7):
    return [random.randint(min_degree, max_degree) for _ in range(length)]

def random_stepwise_melody(length, max_step=2):
    melody = [0]  # Start at root
    for _ in range(length - 1):
        step = random.randint(-max_step, max_step)
        next_note = melody[-1] + step
        next_note = max(-7, min(7, next_note))  # Keep in range
        melody.append(next_note)
    return melody

# Generate
melody_notes = random_stepwise_melody(16, max_step=2)
melody = Motif.from_intervals(melody_notes, scale=scale, duration=0.5)

# Render
song = Composition(tempo=120)
song.add_track('melody', SynthPresets.pluck())
song.play_motif(melody, start=0.0, track='melody')
song.render('random_melody.wav')
```

## Data Sonification

Turn data into music.

### Basic Sonification

```python
from algorythm import DataSonification, Scale, Composition, SynthPresets

# Your data
data = [10, 15, 30, 25, 40, 35, 50, 45]

# Sonify it
sonifier = DataSonification(
    scale=Scale.major('C'),
    min_value=0,
    max_value=100,
    min_degree=-7,
    max_degree=7
)

# Map data to notes
melody_degrees = sonifier.map_to_scale(data)

# Create composition
song = Composition(tempo=120)
song.add_track('data', SynthPresets.bell())

melody = Motif.from_intervals(melody_degrees, scale=Scale.major('C'), duration=0.5)
song.play_motif(melody, start=0.0, track='data')
song.render('data_music.wav')
```

### Multi-Parameter Sonification

```python
# Map different data aspects to different parameters
temperature_data = [20, 22, 25, 28, 26, 23]
humidity_data = [60, 65, 70, 75, 72, 68]

# Temperature → pitch
temp_sonifier = DataSonification(Scale.major('C'), min_value=0, max_value=40)
pitches = temp_sonifier.map_to_scale(temperature_data)

# Humidity → duration
duration = [0.5 + (h / 100) for h in humidity_data]

# Create notes
melody = Motif.from_intervals(pitches, scale=Scale.major('C'), durations=duration)
```

## Practical Tips

1. Start simple - basic randomness can be musical
2. Constraints make random generation sound better
3. Use scales to ensure notes sound good together
4. Combine techniques (L-system structure + Markov melodies)
5. Add randomness to timing for human feel
6. Use generative techniques for variation, not entire songs
7. Fitness functions should reflect musical preferences
8. Test with different scales and tempos
9. Layer generated patterns for complexity
10. Add effects to make generated music more interesting

## Complete Generative Example

```python
from algorythm import (
    Composition, SynthPresets, Scale, Motif,
    LSystem, CellularAutomata, ReverbFX
)
import random

song = Composition(tempo=120)

# Track 1: L-system melody
melody_track = song.add_track('melody', SynthPresets.pluck())
melody_track.add_effect(ReverbFX(mix=0.3))

lsystem = LSystem(axiom='A', rules={'A': 'AB', 'B': 'A'}, iterations=4)
pattern = lsystem.generate()

scale = Scale.pentatonic('C')
notes = [0, 2, 4, 5, 7] * (len(pattern) // 5 + 1)
melody = Motif.from_intervals(notes[:len(pattern)], scale=scale, duration=0.25)
song.play_motif(melody, start=0.0, track='melody')

# Track 2: CA drums
ca = CellularAutomata(rule=30, width=8, generations=32)
drum_pattern = ca.generate()

kick_track = song.add_track('kick', SynthPresets.kick())
for t, row in enumerate(drum_pattern):
    if row[0]:
        song.play_note(60, 0.2, start=t*0.25, track='kick')

# Track 3: Random bass
bass_track = song.add_track('bass', SynthPresets.synth_bass())
bass_notes = [random.choice([0, 0, 0, 7, 5]) for _ in range(32)]
bass = Motif.from_intervals(bass_notes, scale=scale, octave=2, duration=0.5)
song.play_motif(bass, start=0.0, track='bass')

song.render('generative_track.wav')
```
