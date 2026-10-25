# The Sound Of Mathematics 🎵📐

A scientific-calculator formula visualizer with synchronized real-time sonification built with Python, PyQt6, SymPy, NumPy, Matplotlib, and Sounddevice.

## Features

- **Strictly 2D Animated Graph**: Zero-centered axes, glowing neon progressive curves, moving tip dot, and vertical drop lines.
- **1, 2, and 3 Variables Supported**:
  - **1 Variable ($x$)**: Progressive 2D curves ($y = f(x)$ like `sin(4x)/x`, `sin(x^2)`).
  - **2 Variables ($x, y$)**: 2D implicit geometry and contours (like `x^2 + y^2 - 1 = 0` for the unit circle, ellipses, and curves).
  - **3 Variables ($x, y, z$)**: Dynamic animated 2D cross-sections evolving over time as $z$ sweeps through space (e.g. `x^2 + y^2 + z^2 - 1 = 0` sphere).
- **Draggable & Minimizable Keypad**:
  - Full scientific calculator keypad with digits, math functions, and variables ($x, y, z$).
  - Freely draggable anywhere on the screen.
  - Minimizes into a compact 38px bar or hides with a single click, leaving the full screen for the animated graph.
- **Real-Time Synchronized Audio**:
  - Phase-continuous synthesis driven by the animation clock (220 Hz to 880 Hz).
  - Sound modes: Pitch, Pitch + Harmonics, and Stereo Pan by $x$.
- **Video Export with Native Audio**:
  - Export MP4 animations with the synthesized audio track embedded (H.264 + AAC) using FFmpeg.

## Installation

```bash
pip install PyQt6 numpy sympy matplotlib sounddevice
```

## Running the App

```bash
python main.py
```

## Controls & Shortcuts

- **Enter**: Plot formula
- **Space**: Pause / Resume animation
- **R**: Replay from start
- **Esc**: Close window
