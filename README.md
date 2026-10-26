# The Sound Of Mathematics 🎵📐

A scientific-calculator formula visualizer with synchronized real-time sonification built with Python, PyQt6, SymPy, NumPy, Matplotlib, and Sounddevice.

## Features

- **Strictly 2D Animated Graph**: Zero-centered axes, glowing neon progressive curves, moving tip dot, and vertical drop lines.
- **Polar Flower Equations & Parametric Curves**:
  - **Polar Coordinates ($\theta$, $r$)**: Mathematical flowers and roses like $r = 5\sin(12\theta) + 2\sin(24\theta)$ and $r = 8\cos(8\theta)$ with symbolic ranges like `0` to `2*pi`.
  - **Parametric Curves ($t$, $x(t)$, $y(t)$)**: Intricate parametric floral blooms separated by commas, semicolons, or newlines.
- **1, 2, and 3 Variables Supported**:
  - **1 Variable ($x$)**: Progressive 2D curves ($y = f(x)$ like `sin(4x)/x`, `sin(x^2)`).
  - **2 Variables ($x, y$)**: 2D implicit geometry and contours (like `(x*x + y*y)^3 = 4*x*x*y*y` 4-petal flower, and `x^2 + y^2 - 1 = 0` circle).
  - **3 Variables ($x, y, z$)**: Dynamic animated 2D cross-sections evolving over time as $z$ sweeps through space (e.g. `x^2 + y^2 + z^2 - 1 = 0` sphere).
- **Draggable & Minimizable Keypad**:
  - Full scientific calculator keypad with digits, math functions, operators, and variables ($x, y, z, \theta, t, r$).
  - Quick macro keys: `r =`, `2π`, `x =`, `y =`, `t =`, `=`, `,`.
  - Freely draggable anywhere on the screen.
  - Minimizes into a compact 30px bar or hides with a single click, leaving the full screen for the animated graph.
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
