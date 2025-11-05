"""
Terminal-based DAW for Algorythm

A comprehensive terminal interface for music composition with multiple views:
- Tracker View: Classic vertical composition
- Piano Roll View: Grid-based note editing
- Arranger View: Pattern arrangement
- Mixer View: Track levels and effects
- Live Coding View: Python REPL for algorithmic composition
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Static, Button, Label, Input, TextArea
from textual.binding import Binding
from textual.reactive import reactive
from textual import events
import numpy as np
from typing import Optional, List, Dict, Any
import json
import yaml


class Note:
    """Represents a musical note in a pattern."""
    def __init__(self, pitch: int, instrument: int, velocity: int, 
                 fx_cmd: Optional[str] = None, fx_val: Optional[int] = None):
        self.pitch = pitch
        self.instrument = instrument
        self.velocity = velocity
        self.fx_cmd = fx_cmd
        self.fx_val = fx_val
    
    def to_dict(self) -> Dict:
        return {
            'pitch': self.pitch,
            'instrument': self.instrument,
            'velocity': self.velocity,
            'fx_cmd': self.fx_cmd,
            'fx_val': self.fx_val
        }


class Pattern:
    """Represents a pattern with steps."""
    def __init__(self, name: str, length: int = 16):
        self.name = name
        self.length = length
        self.steps: List[Optional[Note]] = [None] * length
    
    def set_note(self, step: int, note: Optional[Note]):
        if 0 <= step < self.length:
            self.steps[step] = note
    
    def get_note(self, step: int) -> Optional[Note]:
        if 0 <= step < self.length:
            return self.steps[step]
        return None


class Track:
    """Represents an audio track."""
    def __init__(self, name: str, track_id: int):
        self.name = name
        self.track_id = track_id
        self.volume = 0.0  # dB
        self.pan = 0  # -100 to 100
        self.muted = False
        self.solo = False
        self.instrument = "Synth"
        self.effects: List[str] = []
        self.patterns: Dict[str, Pattern] = {}
        self.level = 0.0  # Current audio level for metering
    
    def add_pattern(self, pattern: Pattern):
        self.patterns[pattern.name] = pattern


class Project:
    """Represents the entire DAW project."""
    def __init__(self, name: str = "Untitled"):
        self.name = name
        self.bpm = 120
        self.tracks: List[Track] = []
        self.current_pattern = "P01"
        self.arrangement: List[List[Optional[str]]] = []  # Track x Bar grid
        self.playing = False
        self.current_step = 0
    
    def add_track(self, track: Track):
        self.tracks.append(track)
    
    def get_track(self, track_id: int) -> Optional[Track]:
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None
    
    def save(self, filepath: str):
        """Save project to YAML file."""
        data = {
            'name': self.name,
            'bpm': self.bpm,
            'tracks': [
                {
                    'name': t.name,
                    'id': t.track_id,
                    'volume': t.volume,
                    'pan': t.pan,
                    'instrument': t.instrument,
                    'effects': t.effects,
                    'patterns': {
                        name: {
                            'length': p.length,
                            'steps': [n.to_dict() if n else None for n in p.steps]
                        }
                        for name, p in t.patterns.items()
                    }
                }
                for t in self.tracks
            ]
        }
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    @staticmethod
    def load(filepath: str) -> 'Project':
        """Load project from YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        project = Project(data['name'])
        project.bpm = data['bpm']
        
        for track_data in data['tracks']:
            track = Track(track_data['name'], track_data['id'])
            track.volume = track_data['volume']
            track.pan = track_data['pan']
            track.instrument = track_data['instrument']
            track.effects = track_data['effects']
            
            for pattern_name, pattern_data in track_data.get('patterns', {}).items():
                pattern = Pattern(pattern_name, pattern_data['length'])
                for i, step_data in enumerate(pattern_data['steps']):
                    if step_data:
                        note = Note(
                            pitch=step_data['pitch'],
                            instrument=step_data['instrument'],
                            velocity=step_data['velocity'],
                            fx_cmd=step_data.get('fx_cmd'),
                            fx_val=step_data.get('fx_val')
                        )
                        pattern.set_note(i, note)
                track.add_pattern(pattern)
            
            project.add_track(track)
        
        return project


class TrackerView(ScrollableContainer):
    """Classic tracker-style vertical composition view."""
    
    DEFAULT_CSS = """
    TrackerView {
        border: solid green;
        height: 100%;
    }
    """
    
    current_row = reactive(0)
    current_col = reactive(0)
    
    def __init__(self, project: Project, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project = project
        self.pattern = Pattern("P01", 16)
        self.note_names = ['C-', 'C#', 'D-', 'D#', 'E-', 'F-', 'F#', 'G-', 'G#', 'A-', 'A#', 'B-']
    
    def compose(self) -> ComposeResult:
        yield Static(self.render_tracker(), id="tracker-content")
    
    def render_tracker(self) -> str:
        """Render the tracker grid."""
        output = ["-- Track 01: Lead Synth | Pattern 01 | BPM: 120 --"]
        output.append("┌─────┬──────┬─────┬─────┬────────┬────────┐")
        output.append("│ Row │ Note │ Ins │ Vol │ FX Cmd │ FX Val │")
        output.append("├─────┼──────┼─────┼─────┼────────┼────────┤")
        
        for i in range(16):
            note = self.pattern.get_note(i)
            
            if note:
                note_name = f"{self.note_names[note.pitch % 12]}{note.pitch // 12}"
                ins = f"{note.instrument:02d}"
                vol = f"{note.velocity:02d}"
                fx_cmd = note.fx_cmd if note.fx_cmd else "  "
                fx_val = f"{note.fx_val:02X}" if note.fx_val else "  "
            else:
                note_name = "---"
                ins = "--"
                vol = "--"
                fx_cmd = "  "
                fx_val = "  "
            
            cursor = ">" if i == self.current_row else " "
            output.append(f"│{cursor}{i:02d}  │ {note_name}  │ {ins}  │ {vol}  │   {fx_cmd}   │   {fx_val}   │")
        
        output.append("└─────┴──────┴─────┴─────┴────────┴────────┘")
        output.append("\n[↑↓] Move | [Space] Add Note | [Del] Delete | [Tab] Next View")
        
        return "\n".join(output)
    
    def on_key(self, event: events.Key) -> None:
        """Handle key presses."""
        if event.key == "up" and self.current_row > 0:
            self.current_row -= 1
            self.refresh_display()
        elif event.key == "down" and self.current_row < 15:
            self.current_row += 1
            self.refresh_display()
        elif event.key == "space":
            # Add a note at current position
            note = Note(pitch=60, instrument=1, velocity=64)
            self.pattern.set_note(self.current_row, note)
            self.refresh_display()
        elif event.key == "delete":
            # Delete note at current position
            self.pattern.set_note(self.current_row, None)
            self.refresh_display()
    
    def refresh_display(self):
        """Refresh the tracker display."""
        content = self.query_one("#tracker-content", Static)
        content.update(self.render_tracker())


class PianoRollView(ScrollableContainer):
    """Grid-based piano roll for note editing."""
    
    DEFAULT_CSS = """
    PianoRollView {
        border: solid blue;
        height: 100%;
    }
    """
    
    def __init__(self, project: Project, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project = project
        self.cursor_x = 0
        self.cursor_y = 4  # Middle note
        self.notes: List[List[bool]] = [[False for _ in range(16)] for _ in range(12)]
    
    def compose(self) -> ComposeResult:
        yield Static(self.render_piano_roll(), id="piano-roll-content")
    
    def render_piano_roll(self) -> str:
        """Render the piano roll grid."""
        output = ["-- Pattern 01: Bassline | Track 01: Bass | BPM: 120 --"]
        output.append("┌──────┬" + "─────┬" * 8)
        output.append("│ Step │ 1   │ 2   │ 3   │ 4   │ 5   │ 6   │ 7   │ 8   │")
        output.append("├──────┼" + "─────┼" * 8)
        
        note_names = ['G-4', 'F-4', 'E-4', 'D-4', 'C-4', 'B-3', 'A-3', 'G-3', 'F-3', 'E-3', 'D-3', 'C-3']
        
        for i, note_name in enumerate(note_names):
            row = [f"│ {note_name}  │"]
            for j in range(8):
                cursor = ">" if i == self.cursor_y and j == self.cursor_x else " "
                cell = "[X]" if self.notes[i][j] else "   "
                row.append(f" {cursor}{cell}│")
            output.append("".join(row))
        
        output.append("└──────┴" + "─────┴" * 8)
        output.append("\n[Arrows] Move | [Space] Toggle Note | [Tab] Next View")
        
        return "\n".join(output)
    
    def on_key(self, event: events.Key) -> None:
        """Handle key presses."""
        if event.key == "up" and self.cursor_y > 0:
            self.cursor_y -= 1
            self.refresh_display()
        elif event.key == "down" and self.cursor_y < 11:
            self.cursor_y += 1
            self.refresh_display()
        elif event.key == "left" and self.cursor_x > 0:
            self.cursor_x -= 1
            self.refresh_display()
        elif event.key == "right" and self.cursor_x < 7:
            self.cursor_x += 1
            self.refresh_display()
        elif event.key == "space":
            self.notes[self.cursor_y][self.cursor_x] = not self.notes[self.cursor_y][self.cursor_x]
            self.refresh_display()
    
    def refresh_display(self):
        """Refresh the piano roll display."""
        content = self.query_one("#piano-roll-content", Static)
        content.update(self.render_piano_roll())


class ArrangerView(ScrollableContainer):
    """Pattern arrangement view."""
    
    DEFAULT_CSS = """
    ArrangerView {
        border: solid yellow;
        height: 100%;
    }
    """
    
    def __init__(self, project: Project, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project = project
        self.cursor_track = 0
        self.cursor_bar = 0
        # Initialize arrangement grid
        self.arrangement = [
            ["P01", "P01", "P02", "P02", "P01", "P01"],
            ["---", "---", "P03", "P03", "P04", "P04"],
            ["---", "---", "---", "---", "P05", "P05"]
        ]
    
    def compose(self) -> ComposeResult:
        yield Static(self.render_arranger(), id="arranger-content")
    
    def render_arranger(self) -> str:
        """Render the arranger grid."""
        output = ["-- Arranger | Song Length: 6 bars | BPM: 120 --"]
        output.append("┌──────────────┬───────┬───────┬───────┬───────┬───────┬───────┐")
        output.append("│ Track        │ Bar 1 │ Bar 2 │ Bar 3 │ Bar 4 │ Bar 5 │ Bar 6 │")
        output.append("├──────────────┼───────┼───────┼───────┼───────┼───────┼───────┤")
        
        track_names = ["01: Drums", "02: Bass", "03: Lead"]
        
        for i, track_name in enumerate(track_names):
            cursor = ">" if i == self.cursor_track else " "
            row = [f"│{cursor}{track_name:12} │"]
            for j in range(6):
                pattern = self.arrangement[i][j]
                highlight = " * " if i == self.cursor_track and j == self.cursor_bar else "   "
                row.append(f"{highlight}{pattern}{highlight}│")
            output.append("".join(row))
        
        output.append("└──────────────┴───────┴───────┴───────┴───────┴───────┴───────┘")
        output.append("\n[Arrows] Move | [Enter] Edit Pattern | [S] Solo | [Tab] Next View")
        
        return "\n".join(output)
    
    def on_key(self, event: events.Key) -> None:
        """Handle key presses."""
        if event.key == "up" and self.cursor_track > 0:
            self.cursor_track -= 1
            self.refresh_display()
        elif event.key == "down" and self.cursor_track < 2:
            self.cursor_track += 1
            self.refresh_display()
        elif event.key == "left" and self.cursor_bar > 0:
            self.cursor_bar -= 1
            self.refresh_display()
        elif event.key == "right" and self.cursor_bar < 5:
            self.cursor_bar += 1
            self.refresh_display()
    
    def refresh_display(self):
        """Refresh the arranger display."""
        content = self.query_one("#arranger-content", Static)
        content.update(self.render_arranger())


class MixerView(ScrollableContainer):
    """Track mixer with volume, pan, and effects."""
    
    DEFAULT_CSS = """
    MixerView {
        border: solid red;
        height: 100%;
    }
    """
    
    def __init__(self, project: Project, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project = project
        self.cursor_track = 0
        # Initialize some tracks
        self.tracks_data = [
            {"name": "Drums", "vol": -6.0, "pan": "C", "mute": False, "solo": False, "fx": ["Compressor"], "level": 0.7},
            {"name": "Bass", "vol": -3.5, "pan": "C", "mute": False, "solo": True, "fx": ["EQ", "Saturation"], "level": 0.85},
            {"name": "Lead", "vol": -8.0, "pan": "L15", "mute": False, "solo": False, "fx": ["Delay", "Reverb"], "level": 0.4},
            {"name": "Pads", "vol": -12.0, "pan": "R20", "mute": True, "solo": False, "fx": ["Reverb"], "level": 0.2}
        ]
    
    def compose(self) -> ComposeResult:
        yield Static(self.render_mixer(), id="mixer-content")
    
    def render_mixer(self) -> str:
        """Render the mixer display."""
        output = ["-- Mixer | CPU: 15% --"]
        output.append("┌───┬────────────┬──────┬───────────────┬─────┬─────┬────────────────────┐")
        output.append("│ # │ Track Name │ Vol  │ Meter         │ Pan │ M/S │ FX                 │")
        output.append("├───┼────────────┼──────┼───────────────┼─────┼─────┼────────────────────┤")
        
        for i, track in enumerate(self.tracks_data):
            cursor = ">" if i == self.cursor_track else " "
            
            # Create meter visualization
            meter_bars = int(track["level"] * 10)
            meter = "[" + "|" * meter_bars + " " * (10 - meter_bars) + "]"
            
            # M/S status
            ms = ""
            if track["mute"]:
                ms = "M"
            elif track["solo"]:
                ms = "S"
            else:
                ms = "---"
            
            fx_str = ", ".join(track["fx"])
            
            output.append(
                f"│{cursor}{i+1} │ {track['name']:10} │{track['vol']:5.1f} │ {meter} │ {track['pan']:3} │ {ms:3} │ {fx_str:18} │"
            )
        
        output.append("└───┴────────────┴──────┴───────────────┴─────┴─────┴────────────────────┘")
        output.append("\n[↑↓] Select Track | [S] Solo | [M] Mute | [E] Edit FX | [Tab] Next View")
        
        return "\n".join(output)
    
    def on_key(self, event: events.Key) -> None:
        """Handle key presses."""
        if event.key == "up" and self.cursor_track > 0:
            self.cursor_track -= 1
            self.refresh_display()
        elif event.key == "down" and self.cursor_track < 3:
            self.cursor_track += 1
            self.refresh_display()
        elif event.key == "s":
            # Toggle solo
            track = self.tracks_data[self.cursor_track]
            track["solo"] = not track["solo"]
            self.refresh_display()
        elif event.key == "m":
            # Toggle mute
            track = self.tracks_data[self.cursor_track]
            track["mute"] = not track["mute"]
            self.refresh_display()
    
    def refresh_display(self):
        """Refresh the mixer display."""
        content = self.query_one("#mixer-content", Static)
        content.update(self.render_mixer())


class LiveCodingView(ScrollableContainer):
    """Python REPL for live coding."""
    
    DEFAULT_CSS = """
    LiveCodingView {
        border: solid magenta;
        height: 100%;
    }
    """
    
    def __init__(self, project: Project, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project = project
        self.code_history: List[str] = []
        self.output_history: List[str] = []
    
    def compose(self) -> ComposeResult:
        yield Static(self.render_repl(), id="repl-content")
    
    def render_repl(self) -> str:
        """Render the REPL interface."""
        output = ["-- Live Coding --"]
        output.append("Python REPL with access to 'project' object")
        output.append("─" * 60)
        
        # Show history
        for i, (code, result) in enumerate(zip(self.code_history[-5:], self.output_history[-5:])):
            output.append(f">> {code}")
            output.append(result)
        
        output.append(">> _")
        output.append("─" * 60)
        output.append("\n[Ctrl+Enter] Run Code | [Tab] Next View")
        output.append("\nExample:")
        output.append("  from algorythm.generative import euclid")
        output.append("  seq = euclid(hits=5, steps=16)")
        output.append("  print(f'Generated {len(seq)} steps')")
        
        return "\n".join(output)
    
    def refresh_display(self):
        """Refresh the REPL display."""
        content = self.query_one("#repl-content", Static)
        content.update(self.render_repl())


class InstrumentEditorView(ScrollableContainer):
    """Dedicated view for editing instrument and effect parameters."""
    
    DEFAULT_CSS = """
    InstrumentEditorView {
        border: solid cyan;
        height: 100%;
    }
    """
    
    def __init__(self, project: Project, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project = project
        self.current_param = 0
        self.editing_instrument = True  # True for instrument, False for effect
        self.current_effect_index = 0
        
        # Sample parameters for demonstration
        self.instrument_params = [
            {"name": "Oscillator Type", "value": "Saw", "options": ["Sine", "Saw", "Square", "Triangle"]},
            {"name": "Attack", "value": 0.1, "min": 0.0, "max": 2.0, "step": 0.01},
            {"name": "Decay", "value": 0.2, "min": 0.0, "max": 2.0, "step": 0.01},
            {"name": "Sustain", "value": 0.7, "min": 0.0, "max": 1.0, "step": 0.01},
            {"name": "Release", "value": 0.5, "min": 0.0, "max": 2.0, "step": 0.01},
            {"name": "Filter Cutoff", "value": 2000, "min": 20, "max": 20000, "step": 10},
            {"name": "Filter Resonance", "value": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
            {"name": "LFO Rate", "value": 2.0, "min": 0.1, "max": 20.0, "step": 0.1},
        ]
        
        self.effect_params = [
            {"name": "Delay Time", "value": 0.25, "min": 0.01, "max": 2.0, "step": 0.01},
            {"name": "Delay Feedback", "value": 0.3, "min": 0.0, "max": 0.95, "step": 0.01},
            {"name": "Delay Mix", "value": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
        ]
    
    def compose(self) -> ComposeResult:
        yield Static(self.render_editor(), id="editor-content")
    
    def render_editor(self) -> str:
        """Render the instrument/effect editor."""
        device_type = "Instrument" if self.editing_instrument else f"Effect: Delay"
        params = self.instrument_params if self.editing_instrument else self.effect_params
        
        output = [f"-- {device_type} Editor | Track 02: Bass --"]
        output.append("┌────────────────────────────┬──────────────────────────────┐")
        output.append("│ Parameter                  │ Value                        │")
        output.append("├────────────────────────────┼──────────────────────────────┤")
        
        for i, param in enumerate(params):
            cursor = ">" if i == self.current_param else " "
            param_name = param["name"]
            
            if "options" in param:
                # Discrete parameter
                value_str = param["value"]
            else:
                # Continuous parameter
                value = param["value"]
                min_val = param.get("min", 0)
                max_val = param.get("max", 1)
                
                # Create visual bar
                normalized = (value - min_val) / (max_val - min_val)
                bar_length = 15
                filled = int(normalized * bar_length)
                bar = "[" + "=" * filled + " " * (bar_length - filled) + "]"
                value_str = f"{value:.2f} {bar}"
            
            output.append(f"│{cursor}{param_name:27} │ {value_str:28} │")
        
        output.append("└────────────────────────────┴──────────────────────────────┘")
        output.append("\n[↑↓] Select Param | [+/-] Adjust | [I] Instrument | [E] Effect | [Tab] Next View")
        
        return "\n".join(output)
    
    def on_key(self, event: events.Key) -> None:
        """Handle key presses."""
        params = self.instrument_params if self.editing_instrument else self.effect_params
        
        if event.key == "up" and self.current_param > 0:
            self.current_param -= 1
            self.refresh_display()
        elif event.key == "down" and self.current_param < len(params) - 1:
            self.current_param += 1
            self.refresh_display()
        elif event.key == "plus" or event.key == "equals":
            # Increase parameter value
            param = params[self.current_param]
            if "options" in param:
                options = param["options"]
                current_idx = options.index(param["value"])
                if current_idx < len(options) - 1:
                    param["value"] = options[current_idx + 1]
            else:
                new_value = param["value"] + param.get("step", 0.1)
                param["value"] = min(new_value, param.get("max", 1.0))
            self.refresh_display()
        elif event.key == "minus" or event.key == "underscore":
            # Decrease parameter value
            param = params[self.current_param]
            if "options" in param:
                options = param["options"]
                current_idx = options.index(param["value"])
                if current_idx > 0:
                    param["value"] = options[current_idx - 1]
            else:
                new_value = param["value"] - param.get("step", 0.1)
                param["value"] = max(new_value, param.get("min", 0.0))
            self.refresh_display()
        elif event.key == "i":
            # Switch to instrument editing
            self.editing_instrument = True
            self.current_param = 0
            self.refresh_display()
        elif event.key == "e":
            # Switch to effect editing
            self.editing_instrument = False
            self.current_param = 0
            self.refresh_display()
    
    def refresh_display(self):
        """Refresh the editor display."""
        content = self.query_one("#editor-content", Static)
        content.update(self.render_editor())


class TerminalDAW(App):
    """Main Terminal DAW Application."""
    
    CSS = """
    Screen {
        layout: vertical;
    }
    
    #view-container {
        height: 1fr;
    }
    
    #status-bar {
        background: $boost;
        color: $text;
        height: 3;
        padding: 1;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("tab", "next_view", "Next View"),
        Binding("shift+tab", "prev_view", "Prev View"),
        Binding("ctrl+p", "command_palette", "Commands"),
        Binding("ctrl+s", "save_project", "Save"),
        Binding("ctrl+o", "open_project", "Open"),
        Binding("space", "toggle_playback", "Play/Stop"),
        ("1", "view_tracker", "Tracker"),
        ("2", "view_piano_roll", "Piano Roll"),
        ("3", "view_arranger", "Arranger"),
        ("4", "view_mixer", "Mixer"),
        ("5", "view_live_coding", "Live Coding"),
        ("6", "view_instrument_editor", "Instrument/FX"),
    ]
    
    def __init__(self):
        super().__init__()
        self.project = Project("New Project")
        self.current_view_index = 0
        self.views = []
        self.project_file = None
    
    def compose(self) -> ComposeResult:
        """Create the UI layout."""
        yield Header()
        
        # Create all views
        with Container(id="view-container"):
            self.views = [
                TrackerView(self.project, id="tracker-view"),
                PianoRollView(self.project, id="piano-roll-view"),
                ArrangerView(self.project, id="arranger-view"),
                MixerView(self.project, id="mixer-view"),
                LiveCodingView(self.project, id="live-coding-view"),
                InstrumentEditorView(self.project, id="instrument-editor-view")
            ]
            
            # Start with tracker view visible
            for i, view in enumerate(self.views):
                view.display = (i == 0)
                yield view
        
        yield Static(
            "🎵 Terminal DAW | Project: New Project | BPM: 120 | ⏸️ Stopped",
            id="status-bar"
        )
        yield Footer()
    
    def action_next_view(self) -> None:
        """Switch to next view."""
        self.views[self.current_view_index].display = False
        self.current_view_index = (self.current_view_index + 1) % len(self.views)
        self.views[self.current_view_index].display = True
        self.update_status()
    
    def action_prev_view(self) -> None:
        """Switch to previous view."""
        self.views[self.current_view_index].display = False
        self.current_view_index = (self.current_view_index - 1) % len(self.views)
        self.views[self.current_view_index].display = True
        self.update_status()
    
    def action_view_tracker(self) -> None:
        """Switch to tracker view."""
        self.switch_to_view(0)
    
    def action_view_piano_roll(self) -> None:
        """Switch to piano roll view."""
        self.switch_to_view(1)
    
    def action_view_arranger(self) -> None:
        """Switch to arranger view."""
        self.switch_to_view(2)
    
    def action_view_mixer(self) -> None:
        """Switch to mixer view."""
        self.switch_to_view(3)
    
    def action_view_live_coding(self) -> None:
        """Switch to live coding view."""
        self.switch_to_view(4)
    
    def action_view_instrument_editor(self) -> None:
        """Switch to instrument/effect editor view."""
        self.switch_to_view(5)
    
    def switch_to_view(self, index: int) -> None:
        """Switch to a specific view by index."""
        if 0 <= index < len(self.views):
            self.views[self.current_view_index].display = False
            self.current_view_index = index
            self.views[self.current_view_index].display = True
            self.update_status()
    
    def action_toggle_playback(self) -> None:
        """Toggle playback."""
        self.project.playing = not self.project.playing
        self.update_status()
    
    def action_save_project(self) -> None:
        """Save the project."""
        try:
            filename = self.project_file or "project.agp"
            self.project.save(filename)
            self.notify(f"Project saved to {filename}")
        except Exception as e:
            self.notify(f"Error saving: {e}", severity="error")
    
    def action_open_project(self) -> None:
        """Open a project (placeholder for file dialog)."""
        self.notify("Open Project: Please use 'algorythm studio <file.agp>' to load a project")
    
    def action_command_palette(self) -> None:
        """Show command palette with available actions."""
        commands = [
            "Save Project (Ctrl+S)",
            "Open Project (Ctrl+O)",
            "Toggle Playback (Space)",
            "Switch to Tracker View (1)",
            "Switch to Piano Roll View (2)",
            "Switch to Arranger View (3)",
            "Switch to Mixer View (4)",
            "Switch to Live Coding View (5)",
            "Switch to Instrument/FX Editor (6)",
            "Next View (Tab)",
            "Previous View (Shift+Tab)",
            "Quit (Q)"
        ]
        message = "Available Commands:\n\n" + "\n".join(f"• {cmd}" for cmd in commands)
        self.notify(message, timeout=10)
    
    def update_status(self) -> None:
        """Update status bar."""
        view_names = ["Tracker", "Piano Roll", "Arranger", "Mixer", "Live Coding", "Instrument/FX"]
        status_icon = "▶️" if self.project.playing else "⏸️"
        playback_status = "Playing" if self.project.playing else "Stopped"
        
        status = (
            f"🎵 Terminal DAW | View: {view_names[self.current_view_index]} | "
            f"Project: {self.project.name} | BPM: {self.project.bpm} | "
            f"{status_icon} {playback_status}"
        )
        
        status_bar = self.query_one("#status-bar", Static)
        status_bar.update(status)
    
    def load_project(self, filepath: str) -> None:
        """Load a project from file."""
        try:
            self.project = Project.load(filepath)
            self.project_file = filepath
            self.notify(f"Project loaded from {filepath}")
            self.update_status()
        except Exception as e:
            self.notify(f"Error loading project: {e}", severity="error")


def main():
    """Launch the Terminal DAW."""
    app = TerminalDAW()
    app.run()


if __name__ == "__main__":
    main()
