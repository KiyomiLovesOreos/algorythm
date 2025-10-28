
# Design Doc: Terminal-Based DAW for Algorythm

**Version:** 0.2
**Status:** Draft

## 1. Overview

This document outlines the design for a terminal-based Digital Audio Workstation (DAW) experience for the Algorythm library. The goal is to create an interactive, keyboard-driven interface for music composition, arrangement, and playback directly within the terminal, leveraging Algorythm's existing powerful backend.

This will extend the current CLI from a set of one-off commands into a cohesive, session-based application. It will offer multiple composition paradigms, including a traditional piano roll, a classic **tracker** interface, and a **live coding** panel. It aims to appeal to developers, chiptune artists, and users who prefer keyboard-centric, resource-light workflows.

## 2. Goals

*   **Multiple Composition Workflows:** Support different methods of composing (piano roll, tracker, live coding).
*   **Keyboard-First:** All actions should be accessible via keyboard shortcuts for a fast and efficient workflow.
*   **Text-Based UI:** Create a clear, organized, and responsive terminal user interface (TUI).
*   **Advanced Sequencing:** Allow for complex patterns using features like parameter locks and conditional triggers.
*   **Project-Based:** Work within the context of a project file that saves all tracks, patterns, and settings.
*   **Leverage Existing Code:** Utilize Algorythm's core components (Synth, Sampler, Sequence, Effects, etc.) as the audio engine.

## 3. Non-Goals

*   **Full GUI Replacement:** This is not intended to replace a graphical DAW.
*   **Mouse Support:** The primary interface will be the keyboard.
*   **Real-time Audio Input:** Initial versions will not support recording from a microphone or line-in.

## 4. Proposed User Interface & Workflow

The TUI will be built using a library like `textual` and will be launched with a new command, e.g., `algorythm studio`.

### 4.1. Main Views

The interface will be divided into several "views" or "panes" that the user can switch between.

#### a. Piano Roll View

A grid-based editor for a single pattern, familiar to users of modern DAWs.

*   **Layout:** Rows for pitches, columns for time steps.
*   **Interaction:** Arrow keys to move, `Spacebar` to add/remove notes.

```
-- Pattern 01: Bassline | Track 01: Bass Synth | BPM: 120 --
| Step | 1...| 2...| 3...| 4...| 5...| 6...| 7...| 8...|
|------|-----------------|-----------------|-----------------|
| G-4  |     |     |     |     |     |     |     |     |
| F-4  |     |     |     |     |     |     |     |     |
| E-4  | >[X] | --- | --- | --- | [X] | --- | --- | --- |
| D-4  |     |     | [X] | --- |     |     | [X] | --- |
| C-4  | [X] | --- | --- | --- | --- | --- | --- | --- |
-----------------------------------------------------------
Help: [Arrow Keys] Move | [Space] Add/Del Note | [Tab] Next View
```

#### b. Tracker View

A classic, data-dense vertical composition view inspired by software like Renoise and ProTracker. This is a highly efficient, keyboard-native way to compose.

*   **Layout:** A vertically scrolling list of rows (time steps). Each row has columns for `Note`, `Instrument`, `Volume`, and `Effect` commands for a given track.
*   **Interaction:** `Up/Down` to move, alphanumeric keys to enter data directly.

```
-- Track 01: Lead Synth | Pattern 01 | BPM: 120 --
| Row | Note | Ins | Vol | FX Cmd | FX Val |
|-----|------|-----|-----|--------|--------|
|>00  | C-4  | 01  | 64  |        |        |
| 01  | ---  | --  | --  |        |        |
| 02  | E-4  | 01  | 64  | F0     | 48     | <-- Set Filter Cutoff
| 03  | ---  | --  | --  |        |        |
| 04  | G-4  | 01  | 60  | D0     | 08     | <-- Trigger Delay
| 05  | ---  | --  | --  |        |        |
| 06  | ---  | --  | --  |        |        |
| 07  | ---  | --  | --  |        |        |
| 08  | C-4  | 01  | 64  | 70     | 50     | <-- Set Probability to 50%
-------------------------------------------------
Help: [Arrows] Move | [A-G] Enter Note | [Tab] Next View
```

#### c. Arranger View

This view is for arranging patterns into a full song structure.

*   **Layout:** Rows for tracks, columns for bars. Cells contain a pattern number.

```
-- Arranger | Song Length: 8 bars | BPM: 120 --
| Track        | Bar 1 | Bar 2 | Bar 3 | Bar 4 | Bar 5 | Bar 6 |
|--------------|-------|-------|-------|-------|-------|-------|
| 01: Drums    | P01   | P01   | P02   | P02   | P01   | P01   |
| 02: Bass     | ---   | ---   | P03   | P03   | P04   | P04   |
| 03: Lead     | ---   | ---   | ---   | ---   | P05   | P05   |
-----------------------------------------------------------------
Help: [Arrow Keys] Move | [Enter] Select Pattern | [S] Solo Track
```

#### d. Mixer/Track View

A view for adjusting track-level parameters, now with live metering.

*   **Layout:** A list of all tracks with their parameters.
*   **Live Feedback:** Volume meters show real-time audio levels for each track.
*   **Parameters:** Volume, Pan, Mute/Solo, Instrument, Effects, and live audio meters.

```
-- Mixer | CPU: 15% --
| # | Track Name | Vol | Meter         | Pan | M/S | FX
|---|------------|-----|---------------|-----|-----|----------------
| 1 | Drums      |-6.0 | [|||||    ]    | C   | --- | [Compressor]
|>2 | Bass       |-3.5 | [|||||||  ]    | C   | S   | [EQ, Saturation]
| 3 | Lead       |-8.0 | [|||      ]    | L15 | --- | [Delay, Reverb]
| 4 | Pads       |-12.0| [||       ]    | R20 | M   | [Reverb]
--------------------------------------------------------------------
Help: [Tab] Next View | [S] Solo | [M] Mute | [E] Edit FX
```

#### e. Live Coding View

A view that provides a Python REPL or text editor with direct access to the running project. This embraces the 'algorithmic' nature of the library.

*   **Interaction:** Write `algorythm` Python code to generate or manipulate sequences, instruments, and parameters on the fly.

```
-- Live Coding --
>> from algorythm.generative import euclid
>>
>> # Create a euclidean rhythm for the kick drum
>> seq = euclid(hits=5, steps=16)
>> project.get_track("Drums").get_pattern("P01").set_sequence(seq)
>>
>> print("Kick pattern updated.")
Kick pattern updated.
>>
>>
>> |
--------------------------------------------------------------------
Help: [Ctrl+Enter] Run Code | [Tab] Next View
```

#### f. Instrument/Effect Editor View

A dedicated view to "dive into" a device on a track and edit its parameters live.

*   **Interaction:** From the Mixer view, pressing `E` on a track's FX slot or `I` on its instrument would open this view. Use `Up/Down` and `+/-` to select and modify parameters and hear the results in real-time.

### 4.2. Advanced Sequencing

These features will be available in both the Piano Roll and Tracker views.

*   **Parameter Locks (P-Locks):** The ability to "lock" any automatable parameter to a specific value on a single step of a sequence. In the Tracker, this is done with FX commands. In the Piano Roll, this could be a separate lane below the notes.
*   **Conditional Triggers & Probability:** Logic for each step, such as setting the probability of a note playing, or making it play only on certain loops of the pattern.

### 4.3. Command Palette

A global command palette (invoked by `Ctrl+P`) remains the central hub for all actions.

### 5. Core Architecture

### 5.1. Project File

The `.agp` project file (YAML or JSON) will be expanded to store the state for these new features, including tracker-style effect commands.

```yaml
# ... (bpm, master_volume, etc.)
patterns:
  - id: "P03"
    steps:
      # Step 0: Note, Instrument, Volume, FX Cmd, FX Val
      - [60, 1, 64, null, null]
      # Step 1: No note
      - [null, null, null, null, null]
      # Step 2: Note with a filter command
      - [62, 1, 64, "F0", 48]
```

### 5.2. State Management & Audio Engine

The state management and audio engine design remains the same, running in separate threads and translating the state object into audio. The engine will need to be enhanced to handle per-step commands in real-time.

## 6. Command Line Interface Integration

The `algorythm studio [project_file]` command remains the entry point.

## 7. Implementation Plan (High-Level)

1.  **Spike: TUI Library:** Evaluate and choose a TUI library (`textual` remains the top candidate).
2.  **Phase 1: Core Data Structures:** Expand the project format to support advanced sequencing commands.
3.  **Phase 2: Playback Engine:** Enhance the audio engine to handle real-time P-Locks and conditional triggers.
4.  **Phase 3: UI - Core Views:** Build the foundational views: **Tracker**, **Arranger**, and **Mixer** with live metering.
5.  **Phase 4: UI - Additional Views:** Implement the **Piano Roll** and **Live Coding** views.
6.  **Phase 5: Editors & I/O:** Build the **Instrument/Effect Editor** and implement project saving/loading.
7.  **Phase 6: Integration and Refinement:** Integrate with the main CLI, add export functions, and conduct thorough testing.

## 8. Open Questions

1.  **Performance:** How will the audio engine handle rapid, per-step parameter changes (P-Locks) at scale?
2.  **Real-time Effects:** What is the most efficient way to apply and modify effect parameters in the real-time audio thread without causing glitches?
3.  **UI/UX:** Should the Piano Roll or Tracker be the default view? How do we ensure the UX is intuitive for both?
4.  **Live Coding Security:** What sandboxing is needed for the Live Coding view to prevent malicious or system-breaking code?
