#!/usr/bin/env python3
"""
Test script to verify Live Coding functionality
"""

import sys
import numpy as np
from io import StringIO

def test_live_coding():
    """Test the live coding execution context"""
    print("Testing Live Coding functionality...")
    
    # Test code that would be executed in the Live Coding view
    test_code = """
from algorythm.synth import Synth, ADSR
from algorythm.sequence import Scale, Motif
from algorythm.structure import Composition, Reverb
import numpy as np

# Create composition
comp = Composition(tempo=120)

# Define synth
synth = Synth(
    waveform='saw',
    envelope=ADSR(attack=0.05, decay=0.2, sustain=0.6, release=0.4)
)

# Create melody
scale = Scale.minor('C', octave=4)
motif = Motif.from_intervals([0, 2, 3, 5], scale=scale)

# Add to composition
comp.add_track('melody', synth).repeat_motif(motif, bars=1).add_fx(Reverb(mix=0.3))

# Render audio
audio = comp.render()
print(f"Generated {len(audio)} samples")
print(f"Duration: {len(audio) / 44100:.2f}s")

# Store result for playback/export
result = audio
"""
    
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    try:
        # Execute code
        exec_globals = {
            '__name__': '__main__',
            'np': np,
        }
        
        exec(test_code, exec_globals)
        
        # Restore stdout
        sys.stdout = old_stdout
        
        # Get captured output
        output = captured_output.getvalue()
        print("Output from code execution:")
        print(output)
        
        # Check for result
        if 'result' in exec_globals and exec_globals['result'] is not None:
            result = exec_globals['result']
            if isinstance(result, np.ndarray):
                print(f"\n✓ Success! Audio generated: {len(result)} samples")
                print(f"  Sample rate: 44100 Hz")
                print(f"  Duration: {len(result) / 44100:.2f} seconds")
                print(f"  Shape: {result.shape}")
                print(f"  Data type: {result.dtype}")
                return True
            else:
                print(f"\n✗ Result is not a numpy array: {type(result)}")
                return False
        else:
            print("\n✗ No result found in execution context")
            return False
            
    except Exception as e:
        sys.stdout = old_stdout
        print(f"\n✗ Error executing code: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_live_coding()
    sys.exit(0 if success else 1)
