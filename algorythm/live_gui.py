"""
algorythm.live_gui - Live coding GUI for interactive composition

This module provides a graphical interface for live coding music with
real-time audio feedback and code editing.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from typing import Optional
import threading
import traceback
import sys
from io import StringIO


class LiveCodingGUI:
    """
    Live coding GUI with code editor and real-time playback.
    
    Provides an interactive environment for composing music with immediate
    audio feedback.
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize live coding GUI.
        
        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.window = tk.Tk()
        self.window.title("Algorythm Live Coding")
        self.window.geometry("1200x800")
        
        # Audio player
        self.player = None
        try:
            from algorythm.playback import AudioPlayer
            self.player = AudioPlayer(sample_rate=sample_rate)
        except:
            pass
        
        # Current audio
        self.current_audio = None
        self.is_playing = False
        
        # Setup UI
        self._setup_ui()
        self._setup_default_code()
    
    def _setup_ui(self):
        """Setup the user interface."""
        # Main container
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="Algorythm Live Coding Studio",
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Left panel - Code editor
        editor_frame = ttk.LabelFrame(main_frame, text="Code Editor", padding="5")
        editor_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        editor_frame.columnconfigure(0, weight=1)
        editor_frame.rowconfigure(0, weight=1)
        
        self.code_editor = scrolledtext.ScrolledText(
            editor_frame,
            wrap=tk.WORD,
            width=80,
            height=30,
            font=("Courier", 10)
        )
        self.code_editor.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Right panel - Controls and output
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        control_frame.columnconfigure(0, weight=1)
        
        # Control buttons
        button_frame = ttk.LabelFrame(control_frame, text="Controls", padding="5")
        button_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        button_frame.columnconfigure(0, weight=1)
        
        self.run_button = ttk.Button(
            button_frame,
            text="▶ Run (Ctrl+Enter)",
            command=self._run_code
        )
        self.run_button.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.play_button = ttk.Button(
            button_frame,
            text="🔊 Play",
            command=self._play_audio,
            state=tk.DISABLED
        )
        self.play_button.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.stop_button = ttk.Button(
            button_frame,
            text="⏹ Stop",
            command=self._stop_audio,
            state=tk.DISABLED
        )
        self.stop_button.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.clear_button = ttk.Button(
            button_frame,
            text="Clear Output",
            command=self._clear_output
        )
        self.clear_button.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.save_button = ttk.Button(
            button_frame,
            text="💾 Save Audio",
            command=self._save_audio,
            state=tk.DISABLED
        )
        self.save_button.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # Examples dropdown
        examples_frame = ttk.LabelFrame(control_frame, text="Examples", padding="5")
        examples_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        examples_frame.columnconfigure(0, weight=1)
        
        self.example_var = tk.StringVar()
        examples = [
            "Basic Synthesis",
            "FM Synthesis",
            "Wavetable",
            "Effects Chain",
            "Generative"
        ]
        
        example_combo = ttk.Combobox(
            examples_frame,
            textvariable=self.example_var,
            values=examples,
            state="readonly"
        )
        example_combo.grid(row=0, column=0, sticky=(tk.W, tk.E))
        example_combo.bind("<<ComboboxSelected>>", self._load_example)
        
        # Output console
        output_frame = ttk.LabelFrame(control_frame, text="Output", padding="5")
        output_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        
        self.output_console = scrolledtext.ScrolledText(
            output_frame,
            wrap=tk.WORD,
            width=40,
            height=20,
            font=("Courier", 9),
            state=tk.DISABLED
        )
        self.output_console.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        self.status_label = ttk.Label(
            main_frame,
            text="Ready",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Keyboard shortcuts
        self.window.bind("<Control-Return>", lambda e: self._run_code())
        self.window.bind("<Control-r>", lambda e: self._run_code())
    
    def _setup_default_code(self):
        """Setup default code in the editor."""
        default_code = """# Algorythm Live Coding
# Press Ctrl+Enter to run

from algorythm.synth import Synth, Filter, ADSR, FMSynth
from algorythm.sequence import Scale, Motif
from algorythm.structure import Composition, Reverb, Delay
import numpy as np

# Create a composition
comp = Composition(tempo=120)

# Define a synth
synth = Synth(
    waveform='saw',
    filter=Filter.lowpass(cutoff=2000, resonance=0.5),
    envelope=ADSR(attack=0.1, decay=0.3, sustain=0.6, release=0.5)
)

# Create a melody
scale = Scale.minor('C', octave=4)
motif = Motif.from_intervals([0, 2, 3, 5, 7], scale=scale)

# Add to composition
comp.add_track('melody', synth) \\
    .repeat_motif(motif, bars=4) \\
    .add_fx(Reverb(mix=0.3))

# Render the audio
audio = comp.render()
print(f"Generated {len(audio)} samples")
print(f"Duration: {len(audio) / 44100:.2f} seconds")

# Return audio for playback
result = audio
"""
        self.code_editor.insert("1.0", default_code)
    
    def _log_output(self, text: str, tag: Optional[str] = None):
        """Log text to output console."""
        self.output_console.config(state=tk.NORMAL)
        self.output_console.insert(tk.END, text + "\n")
        if tag:
            # You can add colored tags here
            pass
        self.output_console.see(tk.END)
        self.output_console.config(state=tk.DISABLED)
    
    def _clear_output(self):
        """Clear the output console."""
        self.output_console.config(state=tk.NORMAL)
        self.output_console.delete("1.0", tk.END)
        self.output_console.config(state=tk.DISABLED)
    
    def _run_code(self):
        """Execute the code in the editor."""
        code = self.code_editor.get("1.0", tk.END)
        
        self._log_output("=" * 50)
        self._log_output("Running code...")
        self.status_label.config(text="Running...")
        self.window.update()
        
        # Run in a thread to keep GUI responsive
        thread = threading.Thread(target=self._execute_code, args=(code,))
        thread.start()
    
    def _execute_code(self, code: str):
        """Execute code and capture output."""
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            # Create execution namespace with algorythm imports
            namespace = {
                '__name__': '__main__',
                'result': None
            }
            
            # Execute code
            exec(code, namespace)
            
            # Get captured output
            output = captured_output.getvalue()
            if output:
                self._log_output(output)
            
            # Check for result
            if 'result' in namespace and namespace['result'] is not None:
                import numpy as np
                result = namespace['result']
                
                if isinstance(result, np.ndarray):
                    self.current_audio = result
                    self._log_output(f"✓ Audio generated: {len(result)} samples")
                    self.status_label.config(text="Ready - Audio loaded")
                    self.play_button.config(state=tk.NORMAL)
                    self.save_button.config(state=tk.NORMAL)
                else:
                    self._log_output(f"✓ Code executed successfully")
                    self._log_output(f"Result type: {type(result)}")
                    self.status_label.config(text="Ready")
            else:
                self._log_output("✓ Code executed successfully")
                self.status_label.config(text="Ready")
        
        except Exception as e:
            self._log_output(f"✗ Error: {str(e)}")
            self._log_output(traceback.format_exc())
            self.status_label.config(text="Error - Check output")
        
        finally:
            sys.stdout = old_stdout
    
    def _play_audio(self):
        """Play the generated audio."""
        if self.current_audio is None:
            messagebox.showwarning("No Audio", "Please run code to generate audio first.")
            return
        
        if self.player is None:
            messagebox.showerror(
                "Playback Error",
                "Audio playback not available. Install pyaudio:\npip install pyaudio"
            )
            return
        
        self._log_output("Playing audio...")
        self.status_label.config(text="Playing...")
        self.stop_button.config(state=tk.NORMAL)
        self.is_playing = True
        
        # Play in thread
        thread = threading.Thread(target=self._play_worker)
        thread.start()
    
    def _play_worker(self):
        """Worker thread for audio playback."""
        try:
            self.player.play(self.current_audio, blocking=True)
            self._log_output("Playback finished")
            self.status_label.config(text="Ready")
            self.stop_button.config(state=tk.DISABLED)
            self.is_playing = False
        except Exception as e:
            self._log_output(f"Playback error: {e}")
            self.status_label.config(text="Playback error")
            self.is_playing = False
    
    def _stop_audio(self):
        """Stop audio playback."""
        if self.player:
            self.player.stop()
        self.is_playing = False
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Stopped")
        self._log_output("Playback stopped")
    
    def _save_audio(self):
        """Save the generated audio to file."""
        if self.current_audio is None:
            messagebox.showwarning("No Audio", "Please run code to generate audio first.")
            return
        
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[
                ("WAV files", "*.wav"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            try:
                from algorythm.export import Exporter
                exporter = Exporter(sample_rate=self.sample_rate)
                exporter.export(self.current_audio, filename)
                self._log_output(f"✓ Saved to: {filename}")
                self.status_label.config(text=f"Saved: {filename}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save audio:\n{e}")
                self._log_output(f"✗ Save error: {e}")
    
    def _load_example(self, event=None):
        """Load an example into the code editor."""
        example = self.example_var.get()
        
        examples = {
            "Basic Synthesis": """# Basic Synthesis Example
from algorythm.synth import Synth, Filter, ADSR
from algorythm.sequence import Scale, Motif
from algorythm.structure import Composition

# Simple melody
comp = Composition(tempo=120)
synth = Synth(
    waveform='sine',
    envelope=ADSR(attack=0.01, decay=0.1, sustain=0.5, release=0.3)
)

scale = Scale.major('C', octave=4)
motif = Motif.from_intervals([0, 2, 4, 5, 7, 5, 4, 2], scale=scale)

comp.add_track('melody', synth).repeat_motif(motif, bars=2)
result = comp.render()
""",
            "FM Synthesis": """# FM Synthesis Example
from algorythm.synth import FMSynth, ADSR
from algorythm.sequence import Scale
import numpy as np

# Create FM synth
fm = FMSynth(
    carrier_waveform='sine',
    modulator_waveform='sine',
    modulation_index=3.0,
    mod_freq_ratio=2.0,
    envelope=ADSR(attack=0.01, decay=0.2, sustain=0.3, release=0.5)
)

# Generate notes
scale = Scale.major('C', octave=4)
audio_parts = []

for i in range(8):
    note = fm.generate_note(
        frequency=scale.get_frequency(i % 7),
        duration=0.3
    )
    audio_parts.append(note)

result = np.concatenate(audio_parts)
""",
            "Wavetable": """# Wavetable Synthesis Example
from algorythm.synth import WavetableSynth, ADSR
from algorythm.sequence import Scale
import numpy as np

# Create wavetable synth
wt = WavetableSynth.from_waveforms(
    waveforms=['sine', 'triangle', 'saw', 'square'],
    envelope=ADSR(attack=0.05, decay=0.2, sustain=0.5, release=0.3)
)

# Create morphing automation
scale = Scale.minor('A', octave=3)
audio_parts = []

for i in range(8):
    # Morph through wavetable over time
    duration = 0.5
    morph = np.linspace(0, 1, int(duration * 44100))
    
    note = wt.generate_note(
        frequency=scale.get_frequency(i),
        duration=duration,
        morph_automation=morph
    )
    audio_parts.append(note)

result = np.concatenate(audio_parts)
""",
            "Effects Chain": """# Effects Chain Example
from algorythm.synth import Synth, ADSR
from algorythm.sequence import Scale, Motif
from algorythm.structure import *

# Create composition with effects
comp = Composition(tempo=130)

synth = Synth(
    waveform='saw',
    envelope=ADSR(attack=0.05, decay=0.2, sustain=0.7, release=0.4)
)

scale = Scale.pentatonic('E', octave=3)
motif = Motif.from_intervals([0, 2, 4, 5, 7], scale=scale)

# Add track with effects chain
comp.add_track('bass', synth) \\
    .repeat_motif(motif, bars=4) \\
    .add_fx(Distortion(drive=0.4)) \\
    .add_fx(Phaser(rate=0.3, depth=0.5)) \\
    .add_fx(Delay(delay_time=0.25, feedback=0.3)) \\
    .add_fx(Reverb(mix=0.2))

result = comp.render()
""",
            "Generative": """# Generative Music Example
from algorythm.generative import LSystem
from algorythm.synth import Synth, ADSR
from algorythm.sequence import Scale
from algorythm.structure import Composition

# Create L-System pattern
lsys = LSystem(
    axiom='A',
    rules={'A': 'AB', 'B': 'ABA'},
    iterations=3
)

# Convert to motif
scale = Scale.minor('D', octave=4)
motif = lsys.to_motif(
    symbol_map={'A': 0, 'B': 2},
    scale=scale
)

# Create composition
comp = Composition(tempo=140)
synth = Synth(
    waveform='triangle',
    envelope=ADSR(attack=0.01, decay=0.1, sustain=0.4, release=0.2)
)

comp.add_track('generative', synth).repeat_motif(motif, bars=2)
result = comp.render()
"""
        }
        
        if example in examples:
            self.code_editor.delete("1.0", tk.END)
            self.code_editor.insert("1.0", examples[example])
            self._log_output(f"Loaded example: {example}")
    
    def run(self):
        """Run the GUI main loop."""
        self._log_output("Algorythm Live Coding Studio")
        self._log_output("Press Ctrl+Enter to run code")
        self._log_output("=" * 50)
        self.window.mainloop()
    
    def close(self):
        """Close the GUI and clean up resources."""
        if self.player:
            self.player.close()
        self.window.destroy()


def launch():
    """Launch the live coding GUI."""
    gui = LiveCodingGUI()
    gui.run()


if __name__ == "__main__":
    launch()
