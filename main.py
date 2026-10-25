# Requirements: PyQt6 numpy sympy matplotlib sounddevice
"""
The Sound Of Mathematics: Animated function visualizer with synchronized real-time audio.
Strictly 2D graphing with a draggable and minimizable scientific calculator overlay.
"""
from __future__ import annotations
import os, re, sys, time, warnings
from typing import Callable, Optional
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.animation as animation
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QFont, QKeyEvent
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QPushButton, QSlider, QVBoxLayout, QWidget
)
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor, implicit_multiplication_application, parse_expr, standard_transformations
)

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.animation")
try:
    import sounddevice as sd
except Exception:
    sd = None


class FormulaParser:
    """Preprocesses and parses math expressions supporting x, y, and z."""
    FUNCS = {
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "asin": sp.asin, "acos": sp.acos,
        "atan": sp.atan, "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh, "ln": sp.log,
        "log": sp.log, "exp": sp.exp, "sqrt": sp.sqrt, "Abs": sp.Abs, "abs": sp.Abs,
        "floor": sp.floor, "ceil": sp.ceiling, "ceiling": sp.ceiling, "sign": sp.sign, "sinc": sp.sinc
    }
    X, Y, Z = sp.Symbol("x", real=True), sp.Symbol("y", real=True), sp.Symbol("z", real=True)
    WORDS = {"x", "y", "z", "pi", "e", "E", *FUNCS.keys()}
    TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
    LOCAL_DICT = {"x": X, "y": Y, "z": Z, "pi": sp.pi, "e": sp.E, "E": sp.E, **FUNCS}

    @classmethod
    def preprocess(cls, text: str) -> str:
        """Sanitizes text, converting equations, LaTeX, Unicode, and powers into valid syntax."""
        s = text.strip()
        if "=" in s:
            left, right = s.split("=", 1)
            left, right = left.strip(), right.strip()
            if left in ["y", "f(x)", "z", "f(x,y)", "f(x, y)", "f"]:
                s = right
            elif re.match(r"^[a-zA-Z](\([a-zA-Z,\s]*\))?$", left):
                s = right
            else:
                s = f"(({left}) - ({right}))"
        s = s.replace("×", "*").replace("÷", "/").replace("π", "pi").replace("−", "-").replace("²", "**2").replace("³", "**3")
        s = re.sub(r"√\s*\(([^)]+)\)", r"sqrt(\1)", s)
        s = re.sub(r"√\s*([a-zA-Z0-9_]+)", r"sqrt(\1)", s)
        while r"\frac" in s:
            ns = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)
            if ns == s: break
            s = ns
        s = s.replace(r"\left", "").replace(r"\right", "").replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\pi", "pi")
        s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
        s = re.sub(r"\\([a-zA-Z]+)", r"\1", s).replace("^", "**")
        s = re.sub(r"\|([^|]+)\|", r"Abs(\1)", s)
        s = re.sub(r"\be\s*\*\*\s*\(([^)]+)\)", r"exp(\1)", s)
        return re.sub(r"\be\s*\*\*\s*([a-zA-Z0-9_]+)", r"exp(\1)", s)

    @classmethod
    def parse(cls, formula_str: str) -> tuple[sp.Expr, str, Callable, list[str]]:
        """Compiles expression into sympy object, display latex, and numpy evaluation function."""
        cleaned = cls.preprocess(formula_str)
        if not cleaned: raise ValueError("Formula cannot be empty")
        if any(bad in cleaned for bad in [";", ":", "__"]): raise ValueError("Forbidden syntax (; : __)")
        for w in re.findall(r"[a-zA-Z_]\w*", cleaned):
            if w not in cls.WORDS: raise ValueError(f"Unknown identifier '{w}'")
        expr = parse_expr(cleaned, local_dict=cls.LOCAL_DICT, transformations=cls.TRANSFORMS)
        allowed_syms = {cls.X, cls.Y, cls.Z}
        for sym in expr.free_symbols:
            if sym not in allowed_syms: raise ValueError(f"Unknown variable '{sym}', only x, y, z allowed")
        for atom in expr.atoms(sp.Function):
            if atom.func not in set(cls.FUNCS.values()): raise ValueError(f"Disallowed function '{atom.func}'")

        var_order = [cls.X, cls.Y, cls.Z]
        used_vars = [s for s in var_order if s in expr.free_symbols]
        if not used_vars: used_vars = [cls.X]
        var_names = [s.name for s in used_vars]

        lam = sp.lambdify(tuple(used_vars), expr, modules=["numpy"])
        raw_latex = sp.latex(expr).replace(r"\left", "").replace(r"\right", "")
        if len(var_names) == 1: latex_str = f"f({var_names[0]}) = {raw_latex}"
        elif len(var_names) == 2: latex_str = f"f(x, y) = {raw_latex} = 0"
        else: latex_str = f"f(x, y, z) = {raw_latex} = 0"
        return expr, latex_str, lam, var_names


class AudioEngine:
    """Real-time audio sonification engine with phase continuity and clock synchronization."""
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.stream: Optional[sd.OutputStream] = None
        self.is_playing = self.is_muted = False
        self.volume, self.mode, self.duration = 0.75, "Pitch", 8.0
        self.ys: Optional[np.ndarray] = None
        self.y_min, self.y_max = -1.0, 1.0
        self.sample_idx, self.phase, self.curr_freq = 0, 0.0, 440.0
        self.total_samples = int(self.duration * self.sample_rate)
        if sd is not None:
            try:
                self.stream = sd.OutputStream(samplerate=self.sample_rate, channels=2, blocksize=512, callback=self._audio_cb)
                self.stream.start()
            except Exception: self.stream = None

    def prepare(self, ys: np.ndarray, y_min: float, y_max: float, duration: float) -> None:
        """Sets curve data and resets playback sample head to origin."""
        self.ys, self.y_min, self.y_max, self.duration = ys, y_min, y_max, max(0.5, duration)
        self.total_samples = max(100, int(self.duration * self.sample_rate))
        self.sample_idx, self.is_playing = 0, True

    def get_progress(self) -> float:
        """Returns normalized playback progress in [0.0, 1.0]."""
        return min(1.0, self.sample_idx / float(self.total_samples)) if self.total_samples > 0 else 1.0

    def set_progress(self, progress: float) -> None:
        self.sample_idx = int(np.clip(progress, 0.0, 1.0) * self.total_samples)

    def _audio_cb(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        if not self.is_playing or self.is_muted or self.ys is None:
            outdata.fill(0); return
        end_idx = min(self.sample_idx + frames, self.total_samples)
        n = end_idx - self.sample_idx
        if n <= 0: outdata.fill(0); self.is_playing = False; return

        indices = np.arange(self.sample_idx, end_idx)
        progs = indices / max(1, self.total_samples - 1)
        y_vals = np.interp(progs, np.linspace(0.0, 1.0, len(self.ys)), self.ys)
        norm_y = np.clip((y_vals - self.y_min) / max(1e-6, self.y_max - self.y_min), 0.0, 1.0)
        target_f = 220.0 * (2.0 ** (2.0 * norm_y))
        start_f = self.curr_freq
        end_f = target_f[-1] if np.isfinite(target_f[-1]) else start_f
        dt = 1.0 / self.sample_rate
        phases = self.phase + np.cumsum((2.0 * np.pi * dt) * np.linspace(start_f, end_f, n, dtype=np.float32))

        mono = (np.sin(phases) + 0.3 * np.sin(2.0 * phases) + 0.1 * np.sin(3.0 * phases)) / 1.4 if self.mode == "Pitch+Harmonics" else np.sin(phases)
        vol_curve = 0.7 + 0.3 * np.abs(norm_y - 0.5) * 2.0
        fade_len, fades = int(0.03 * self.sample_rate), np.ones(n, dtype=np.float32)
        m_s, m_e = indices < fade_len, indices > (self.total_samples - fade_len)
        if np.any(m_s): fades[m_s] = indices[m_s] / max(1, fade_len)
        if np.any(m_e): fades[m_e] = np.maximum(0.0, (self.total_samples - indices[m_e]) / max(1, fade_len))
        mono = np.where(np.isnan(y_vals), 0.0, mono * vol_curve * fades * self.volume)

        out = np.zeros((frames, 2), dtype=np.float32)
        if self.mode == "Stereo pan by x":
            out[:n, 0], out[:n, 1] = mono * np.cos(0.5 * np.pi * progs), mono * np.sin(0.5 * np.pi * progs)
        else: out[:n, 0] = out[:n, 1] = mono * 0.707
        self.curr_freq, self.phase = float(end_f), float(phases[-1] % (2.0 * np.pi))
        self.sample_idx += n
        outdata[:] = out

    def render_audio_buffer(self, duration: Optional[float] = None) -> np.ndarray:
        """Renders offline stereo audio corresponding to the current curve and sound settings."""
        if self.ys is None: return np.zeros((100, 2), dtype=np.float32)
        dur = self.duration if duration is None else duration
        total_samples = max(100, int(dur * self.sample_rate))
        indices = np.arange(total_samples)
        progs = indices / max(1, total_samples - 1)
        y_vals = np.interp(progs, np.linspace(0.0, 1.0, len(self.ys)), self.ys)
        norm_y = np.clip((y_vals - self.y_min) / max(1e-6, self.y_max - self.y_min), 0.0, 1.0)
        target_f = 220.0 * (2.0 ** (2.0 * norm_y))

        blocksize, curr_freq, phase = 512, 440.0, 0.0
        dt = 1.0 / self.sample_rate
        mono = np.zeros(total_samples, dtype=np.float32)

        for start in range(0, total_samples, blocksize):
            end = min(start + blocksize, total_samples)
            n = end - start
            end_f = target_f[end - 1] if np.isfinite(target_f[end - 1]) else curr_freq
            f_sweep = np.linspace(curr_freq, end_f, n, dtype=np.float32)
            dphases = (2.0 * np.pi * dt) * f_sweep
            phases = phase + np.cumsum(dphases)
            if self.mode == "Pitch+Harmonics":
                mono[start:end] = (np.sin(phases) + 0.3 * np.sin(2.0 * phases) + 0.1 * np.sin(3.0 * phases)) / 1.4
            else:
                mono[start:end] = np.sin(phases)
            curr_freq = float(end_f)
            phase = float(phases[-1] % (2.0 * np.pi))

        vol_curve = 0.7 + 0.3 * np.abs(norm_y - 0.5) * 2.0
        fade_len = int(0.03 * self.sample_rate)
        fades = np.ones(total_samples, dtype=np.float32)
        m_s, m_e = indices < fade_len, indices > (total_samples - fade_len)
        if np.any(m_s): fades[m_s] = indices[m_s] / max(1, fade_len)
        if np.any(m_e): fades[m_e] = np.maximum(0.0, (total_samples - indices[m_e]) / max(1, fade_len))

        mono = np.where(np.isnan(y_vals), 0.0, mono * vol_curve * fades * self.volume)
        out = np.zeros((total_samples, 2), dtype=np.float32)
        if self.mode == "Stereo pan by x":
            out[:, 0] = mono * np.cos(0.5 * np.pi * progs)
            out[:, 1] = mono * np.sin(0.5 * np.pi * progs)
        else:
            out[:, 0] = out[:, 1] = mono * 0.707
        return out


class GraphAnimator:
    """Manages strictly 2D origin-centered axes, glowing curve styling, and animated rendering."""
    PALETTE = ["#00FFFF", "#FF1493", "#39FF14", "#FF8C00", "#FFFF00", "#BF00FF"]

    def __init__(self, canvas: FigureCanvasQTAgg, fig: Figure):
        self.canvas, self.fig = canvas, fig
        self.color_idx, self.duration, self.color = 0, 8.0, self.PALETTE[0]
        self.anim: Optional[animation.FuncAnimation] = None
        self.is_paused = False
        self.get_progress_cb: Optional[Callable[[], float]] = None
        self.on_finish_cb: Optional[Callable[[], None]] = None
        self.xs, self.ys = np.zeros(10), np.zeros(10)
        self.slices_3d: list[tuple[Optional[np.ndarray], Optional[np.ndarray]]] = []
        self.is_3d_slice_mode = False

    def setup_2d_axes(self, xmin: float, xmax: float, ymin: float, ymax: float,
                      latex_str: str, equal_aspect: bool = False) -> None:
        """Sets up 2D coordinate system with black background, origin axes, and arrowheads."""
        self.color = self.PALETTE[self.color_idx % len(self.PALETTE)]; self.color_idx += 1
        self.fig.clf()
        self.ax = self.fig.add_subplot(111, facecolor="black")
        self.fig.patch.set_facecolor("black")
        self.fig.subplots_adjust(left=0.08, right=0.92, top=0.94, bottom=0.12)

        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        if equal_aspect: self.ax.set_aspect("equal", adjustable="box")

        for sp_name in ["left", "bottom"]:
            self.ax.spines[sp_name].set_position("zero"); self.ax.spines[sp_name].set_color("white"); self.ax.spines[sp_name].set_linewidth(1.3)
        for sp_name in ["right", "top"]: self.ax.spines[sp_name].set_color("none")
        self.ax.tick_params(colors="white", labelsize=8)
        self.ax.set_xticks([t for t in self.ax.get_xticks() if abs(t) > 1e-4 and xmin <= t <= xmax])
        self.ax.set_yticks([t for t in self.ax.get_yticks() if abs(t) > 1e-4 and ymin <= t <= ymax])
        self.ax.plot(xmax, 0, marker=">", color="white", markersize=7, clip_on=False)
        self.ax.plot(0, ymax, marker="^", color="white", markersize=7, clip_on=False)

        self.glow1, = self.ax.plot([], [], color=self.color, lw=7.0, alpha=0.25)
        self.glow2, = self.ax.plot([], [], color=self.color, lw=4.2, alpha=0.45)
        self.main_line, = self.ax.plot([], [], color=self.color, lw=2.2, alpha=1.0)
        self.dot_glow, = self.ax.plot([], [], marker="o", markersize=14, color=self.color, alpha=0.35, ls="")
        self.dot_main, = self.ax.plot([], [], marker="o", markersize=6, color="white", alpha=1.0, ls="")
        self.drop_line, = self.ax.plot([], [], color="white", ls="--", lw=1.0, alpha=0.35)
        self.formula_text = self.fig.text(0.5, 0.03, f"${latex_str}$", color=self.color, fontsize=13, ha="center", va="bottom", weight="bold")
        self.canvas.draw_idle()

    def start_animation_trace(self, xs: np.ndarray, ys: np.ndarray, duration: float,
                              get_progress: Callable[[], float], on_finish: Callable[[], None]) -> None:
        """Starts 2D progressive curve tracing loop (for 1D explicit curves & 2D implicit circles)."""
        if self.anim is not None and self.anim.event_source is not None: self.anim.event_source.stop()
        self.xs, self.ys, self.duration = xs, ys, duration
        self.is_3d_slice_mode = False
        self.get_progress_cb, self.on_finish_cb, self.is_paused = get_progress, on_finish, False
        self.anim = animation.FuncAnimation(self.fig, self._update_frame, interval=33, blit=False, cache_frame_data=False)

    def start_animation_slices(self, slices: list[tuple[Optional[np.ndarray], Optional[np.ndarray]]], duration: float,
                               get_progress: Callable[[], float], on_finish: Callable[[], None]) -> None:
        """Starts 2D animated slice evolution loop (for 3-variable dynamic slices)."""
        if self.anim is not None and self.anim.event_source is not None: self.anim.event_source.stop()
        self.slices_3d, self.duration = slices, duration
        self.is_3d_slice_mode = True
        self.get_progress_cb, self.on_finish_cb, self.is_paused = get_progress, on_finish, False
        self.anim = animation.FuncAnimation(self.fig, self._update_frame, interval=33, blit=False, cache_frame_data=False)

    def _update_frame(self, frame_num: int) -> None:
        if self.is_paused or self.get_progress_cb is None: return
        prog = self.get_progress_cb()

        if not self.is_3d_slice_mode:
            idx = int(np.clip(prog * (len(self.xs) - 1), 0, len(self.xs) - 1))
            x_curr, y_curr = self.xs[:idx + 1], self.ys[:idx + 1]
            self.glow1.set_data(x_curr, y_curr); self.glow2.set_data(x_curr, y_curr); self.main_line.set_data(x_curr, y_curr)
            tx, ty = self.xs[idx], self.ys[idx]
            if np.isfinite(ty) and np.isfinite(tx):
                self.dot_glow.set_data([tx], [ty]); self.dot_main.set_data([tx], [ty]); self.drop_line.set_data([tx, tx], [0, ty])
            else:
                self.dot_glow.set_data([], []); self.dot_main.set_data([], []); self.drop_line.set_data([], [])
        else:
            if not self.slices_3d: return
            s_idx = int(np.clip(prog * (len(self.slices_3d) - 1), 0, len(self.slices_3d) - 1))
            sx, sy = self.slices_3d[s_idx]
            if sx is not None and len(sx) > 0:
                self.glow1.set_data(sx, sy); self.glow2.set_data(sx, sy); self.main_line.set_data(sx, sy)
                tx, ty = sx[0], sy[0]
                self.dot_glow.set_data([tx], [ty]); self.dot_main.set_data([tx], [ty]); self.drop_line.set_data([tx, tx], [0, ty])
            else:
                self.glow1.set_data([], []); self.glow2.set_data([], []); self.main_line.set_data([], [])
                self.dot_glow.set_data([], []); self.dot_main.set_data([], []); self.drop_line.set_data([], [])

        self.canvas.draw_idle()
        if prog >= 1.0 and self.on_finish_cb: self.on_finish_cb()


def extract_2d_contour(F: Callable[[np.ndarray, np.ndarray], np.ndarray],
                       xmin: float, xmax: float, ymin: float, ymax: float,
                       total_points: int = 2400) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extracts ordered 2D contour points of F(x, y) = 0 using matplotlib's marching contour."""
    fig, ax = Figure(), Figure().add_subplot(111)
    grid_res = 220
    X, Y = np.meshgrid(np.linspace(xmin, xmax, grid_res), np.linspace(ymin, ymax, grid_res))
    with np.errstate(all="ignore"):
        Z = F(X, Y)
        if np.iscomplexobj(Z): Z = np.where(np.abs(np.imag(Z)) < 1e-8, np.real(Z), np.nan)
        Z = np.asarray(Z, dtype=float)
    cs = ax.contour(X, Y, Z, levels=[0])
    paths = cs.get_paths()
    if not paths: return None, None

    all_pts = []
    for p in paths:
        if len(p.vertices) == 0: continue
        if p.codes is None:
            all_pts.append(p.vertices)
            all_pts.append(np.array([[np.nan, np.nan]]))
        else:
            seg = []
            for pt, code in zip(p.vertices, p.codes):
                if code == 1 and seg:
                    all_pts.append(np.array(seg))
                    all_pts.append(np.array([[np.nan, np.nan]]))
                    seg = [pt]
                else: seg.append(pt)
            if seg:
                all_pts.append(np.array(seg))
                all_pts.append(np.array([[np.nan, np.nan]]))
    if not all_pts: return None, None
    pts = np.vstack(all_pts)[:-1]
    if len(pts) == 0: return None, None
    idx = np.linspace(0, len(pts) - 1, total_points).astype(int)
    return pts[idx, 0], pts[idx, 1]


class DraggableKeypad(QFrame):
    """Floating, draggable, and minimizable scientific calculator keypad panel."""

    def __init__(self, parent: QWidget, on_insert: Callable[[str], None], on_plot: Callable[[], None]):
        super().__init__(parent)
        self.on_insert, self.on_plot = on_insert, on_plot
        self.dragging = False
        self.offset = QPoint()
        self.setStyleSheet("""
            QFrame { background-color: rgba(16, 16, 16, 0.96); border: 1.5px solid #FFD700; border-radius: 8px; }
            QLabel { color: #FFD700; font-weight: bold; font-size: 11px; }
            QPushButton { background-color: #242424; color: #fff; border: 1px solid #3c3c3c; border-radius: 4px; font-weight: bold; font-size: 11px; }
            QPushButton:hover { background-color: #3a3a3a; color: #FFD700; border-color: #FFD700; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(4)

        # Draggable Header Bar
        hdr = QHBoxLayout()
        hdr.setContentsMargins(2, 0, 2, 2)
        lbl = QLabel("⠿ Calculator (Drag anywhere)")
        lbl.setStyleSheet("color: #FFD700; font-size: 11px;")
        hdr.addWidget(lbl)
        hdr.addStretch()

        self.btn_min = QPushButton("−")
        self.btn_min.setToolTip("Minimize / Collapse Keypad")
        self.btn_min.setFixedSize(22, 20)
        self.btn_min.clicked.connect(self.toggle_minimize)
        hdr.addWidget(self.btn_min)

        self.btn_close = QPushButton("✕")
        self.btn_close.setToolTip("Close / Hide Keypad")
        self.btn_close.setFixedSize(22, 20)
        self.btn_close.clicked.connect(self.hide)
        hdr.addWidget(self.btn_close)
        layout.addLayout(hdr)

        # Keypad Grid Body
        self.body = QWidget()
        grid = QGridLayout(self.body)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)

        buttons = [
            ["sin", "cos", "tan", "asin", "acos", "atan", "DEL"],
            ["sinh", "cosh", "tanh", "ln", "log", "exp", "CLEAR"],
            ["√", "|x|", "x²", "1/x", "a/b", "sinc", "sign"],
            ["floor", "ceil", "(", ")", "^", "π", "e"],
            ["7", "8", "9", "÷", "×", "−", "+"],
            ["4", "5", "6", "1", "2", "3", "."],
            ["0", "x", "y", "z", "Plot ▶"]
        ]
        for r, row in enumerate(buttons):
            for c, text in enumerate(row):
                btn = QPushButton(text, focusPolicy=Qt.FocusPolicy.NoFocus)
                btn.setFixedHeight(26)
                if text == "Plot ▶":
                    btn.setStyleSheet("background-color: #FFD700; color: #000; font-weight: bold;")
                    btn.clicked.connect(self.on_plot)
                    grid.addWidget(btn, r, c, 1, 3)
                elif text in ["DEL", "CLEAR"]:
                    btn.setStyleSheet("background-color: #700000; color: #fff;")
                    btn.clicked.connect(lambda _, t=text: self.on_insert(t))
                    grid.addWidget(btn, r, c)
                else:
                    btn.clicked.connect(lambda _, t=text: self.on_insert(t))
                    grid.addWidget(btn, r, c)
        layout.addWidget(self.body)

    def toggle_minimize(self) -> None:
        """Minimizes keypad down to a compact drag strip at the side or restores it."""
        show = not self.body.isVisible()
        self.body.setVisible(show)
        self.btn_min.setText("−" if show else "＋")
        self.adjustSize()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.offset = event.pos()
            self.raise_()

    def mouseMoveEvent(self, event) -> None:
        if self.dragging and self.parent():
            new_pos = self.mapToParent(event.pos() - self.offset)
            p_rect = self.parent().rect()
            x = max(6, min(new_pos.x(), p_rect.width() - self.width() - 6))
            y = max(50, min(new_pos.y(), p_rect.height() - self.height() - 6))
            self.move(x, y)

    def mouseReleaseEvent(self, event) -> None:
        self.dragging = False


class MainWindow(QMainWindow):
    """Main window with dark theme, floating draggable keypad, and unobstructed 2D visualizer."""
    PRESETS = [
        "x^2+y^2-1", "sin(4x)/x", "x^2+y^2+z^2-1", "sin(x^2)",
        "cos(3x)+sin(8x)", "x^2/4+y^2/9-1", "e^(x/4)*sin(6x)", "x*sin(1/x)"
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("The Sound Of Mathematics")
        self.resize(760, 960); self.setMinimumSize(640, 720)
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #000000; color: #d0d0d0; font-family: 'Segoe UI', Arial; }
            QLineEdit, QComboBox { background-color: #141414; color: #fff; border: 1px solid #333; border-radius: 4px; padding: 4px 6px; }
            QLineEdit:focus { border: 1px solid #FFD700; }
            QPushButton { background-color: #1a1a1a; color: #fff; border: 1px solid #333; border-radius: 4px; padding: 5px; font-weight: bold; }
            QPushButton:hover { background-color: #333; color: #FFD700; border-color: #FFD700; }
            QSlider::groove:horizontal { height: 4px; background: #333; }
            QSlider::handle:horizontal { background: #FFD700; width: 14px; margin: -5px 0; border-radius: 7px; }
        """)
        self.audio_engine = AudioEngine()
        self.wall_clock_start, self.accumulated_pause, self.pause_timestamp = time.perf_counter(), 0.0, 0.0
        self.history: list[str] = []

        self._build_ui()
        # Initialize floating draggable keypad overlay
        self.keypad = DraggableKeypad(self, on_insert=self._on_keypad_press, on_plot=self.plot_formula)
        self.keypad.move(self.width() - 370, 115)
        self.keypad.show()

        self.load_preset("x^2+y^2-1")

    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setSpacing(6)
        layout.setContentsMargins(14, 8, 14, 8)
        self.setCentralWidget(central)

        title = QLabel("THE SOUND OF MATHEMATICS")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter); title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #FFD700; letter-spacing: 2px;")
        layout.addWidget(title)

        inp_box = QHBoxLayout()
        self.fx_lbl = QLabel("f ="); self.fx_lbl.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold)); self.fx_lbl.setStyleSheet("color: #FFD700;")
        inp_box.addWidget(self.fx_lbl)
        self.formula_input = QLineEdit(); self.formula_input.setPlaceholderText("e.g. x^2+y^2-1, sin(4x)/x, x^2+y^2+z^2-1...")
        self.formula_input.returnPressed.connect(self.plot_formula); inp_box.addWidget(self.formula_input)

        self.btn_plot = QPushButton("Plot ▶"); self.btn_plot.setStyleSheet("background-color: #FFD700; color: #000; font-weight: bold; padding: 6px 14px;")
        self.btn_plot.clicked.connect(self.plot_formula); inp_box.addWidget(self.btn_plot)

        self.btn_toggle_keypad = QPushButton("⌨ Keypad")
        self.btn_toggle_keypad.setToolTip("Show / Hide Floating Keypad")
        self.btn_toggle_keypad.setStyleSheet("background-color: #242424; color: #FFD700; border: 1px solid #FFD700; padding: 6px 10px;")
        self.btn_toggle_keypad.clicked.connect(self.toggle_keypad_visibility)
        inp_box.addWidget(self.btn_toggle_keypad)
        layout.addLayout(inp_box)

        self.error_label = QLabel(""); self.error_label.setStyleSheet("color: #FF4D4D; font-size: 11px; font-weight: bold;")
        self.error_label.setVisible(False); layout.addWidget(self.error_label)

        ctrl_bar = QHBoxLayout()
        for lbl, dval, w, attr in [("x min:", "-3.0", 44, "inp_xmin"), ("x max:", "3.0", 44, "inp_xmax"), ("Duration (s):", "8.0", 40, "inp_dur")]:
            ctrl_bar.addWidget(QLabel(lbl)); le = QLineEdit(dval); le.setMaximumWidth(w); setattr(self, attr, le); ctrl_bar.addWidget(le)
        ctrl_bar.addWidget(QLabel("Sound:"))
        self.combo_mode = QComboBox(); self.combo_mode.addItems(["Pitch", "Pitch+Harmonics", "Stereo pan by x"])
        self.combo_mode.currentTextChanged.connect(lambda m: setattr(self.audio_engine, "mode", m)); ctrl_bar.addWidget(self.combo_mode)
        ctrl_bar.addWidget(QLabel("Vol:"))
        self.slider_vol = QSlider(Qt.Orientation.Horizontal); self.slider_vol.setRange(0, 100); self.slider_vol.setValue(75); self.slider_vol.setMaximumWidth(70)
        self.slider_vol.valueChanged.connect(lambda v: setattr(self.audio_engine, "volume", v / 100.0)); ctrl_bar.addWidget(self.slider_vol)
        layout.addLayout(ctrl_bar)

        act_bar = QHBoxLayout()
        for name, handler in [("Pause", self.toggle_pause), ("Replay", self.replay), ("Mute 🔇", self.toggle_mute), ("Save MP4/GIF", self.save_animation)]:
            btn = QPushButton(name); btn.clicked.connect(handler); act_bar.addWidget(btn)
            if "Pause" in name: self.btn_pause = btn
            elif "Mute" in name: self.btn_mute = btn
        self.combo_history = QComboBox(); self.combo_history.setPlaceholderText("History...")
        self.combo_history.currentTextChanged.connect(lambda t: self.load_preset(t) if t else None); act_bar.addWidget(self.combo_history)
        layout.addLayout(act_bar)

        pre_bar = QHBoxLayout(); pre_bar.addWidget(QLabel("Presets:", styleSheet="color: #FFD700; font-weight: bold;"))
        for p in self.PRESETS:
            b = QPushButton(p, focusPolicy=Qt.FocusPolicy.NoFocus, styleSheet="padding: 3px 6px; font-size: 11px;")
            b.clicked.connect(lambda _, f=p: self.load_preset(f)); pre_bar.addWidget(b)
        layout.addLayout(pre_bar)

        # Main 2D Graph Canvas - Takes maximum window space
        self.fig = Figure(figsize=(6, 5.4), facecolor="black")
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas, stretch=1)
        self.animator = GraphAnimator(self.canvas, self.fig)

        badge = QLabel("Wear Headphones 🎧", alignment=Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet("color: #FFD700; border: 1.5px solid #FFD700; border-radius: 12px; padding: 4px 18px; margin: 2px auto; background-color: #101010; font-weight: bold;")
        layout.addWidget(badge, alignment=Qt.AlignmentFlag.AlignCenter)

    def toggle_keypad_visibility(self) -> None:
        """Toggles the floating draggable keypad on or off."""
        if self.keypad.isHidden():
            self.keypad.show()
            self.keypad.raise_()
            if not self.keypad.body.isVisible():
                self.keypad.toggle_minimize()
        else:
            self.keypad.hide()

    def _on_keypad_press(self, text: str) -> None:
        inp = self.formula_input
        inserts = {"DEL": None, "CLEAR": None, "a/b": "()/()", "|x|": "|x|", "x²": "^2", "1/x": "1/x", "√": "sqrt("}
        if text == "DEL": inp.backspace()
        elif text == "CLEAR": inp.clear()
        elif text in inserts: inp.insert(inserts[text])
        elif text in ["sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "ln", "log", "exp", "floor", "ceil", "sign", "sinc"]:
            inp.insert(f"{text}(")
        else: inp.insert(text)
        inp.setFocus()

    def load_preset(self, formula: str) -> None:
        self.formula_input.setText(formula); self.plot_formula()

    def plot_formula(self) -> None:
        """Parses formula, plots 2D curves/slices, and synchronizes real-time sound."""
        text = self.formula_input.text().strip()
        if not text: return
        try:
            expr, latex_str, lam, var_names = FormulaParser.parse(text)
            self.error_label.setVisible(False)
            self.fx_lbl.setText(f"f({','.join(var_names)}) =")
        except Exception as e:
            self.error_label.setText(f"Error: {e}"); self.error_label.setVisible(True); return

        try:
            xmin, xmax, dur = float(self.inp_xmin.text()), float(self.inp_xmax.text()), max(0.5, float(self.inp_dur.text()))
        except ValueError:
            self.error_label.setText("Error: Parameters must be numbers"); self.error_label.setVisible(True); return
        if xmin >= xmax:
            self.error_label.setText("Error: x min must be less than x max"); self.error_label.setVisible(True); return

        if text not in self.history:
            self.history.insert(0, text); self.combo_history.blockSignals(True)
            self.combo_history.clear(); self.combo_history.addItems(self.history); self.combo_history.blockSignals(False)

        # 1-VARIABLE 2D GRAPH
        if len(var_names) == 1:
            xs = np.linspace(xmin, xmax, 2400)
            with np.errstate(all="ignore"):
                ys = lam(np.where(xs == 0.0, 1e-12, xs))
                if np.isscalar(ys): ys = np.full_like(xs, float(ys))
                if np.iscomplexobj(ys): ys = np.where(np.abs(np.imag(ys)) < 1e-8, np.real(ys), np.nan)
                ys = np.asarray(ys, dtype=float); ys[~np.isfinite(ys)] = np.nan

            valid = ys[np.isfinite(ys)]
            if len(valid) == 0: ymin, ymax = -1.0, 1.0
            else:
                p2, p98 = np.percentile(valid, [2, 98])
                if p98 - p2 < 1e-4: p2, p98 = p2 - 1.0, p98 + 1.0
                pad = (p98 - p2) * 0.15; ymin, ymax = float(p2 - pad), float(p98 + pad)

            ys_plot = ys.copy(); span = ymax - ymin
            ys_plot[(ys_plot < ymin - span * 0.5) | (ys_plot > ymax + span * 0.5)] = np.nan
            flips = (ys_plot[:-1] * ys_plot[1:]) < 0
            ys_plot[:-1][flips & (np.abs(np.diff(ys_plot)) > span * 0.6)] = np.nan

            self.animator.setup_2d_axes(xmin, xmax, ymin, ymax, latex_str, equal_aspect=False)
            self.audio_engine.prepare(ys, ymin, ymax, dur)
            self.animator.start_animation_trace(xs, ys_plot, dur, self._unified_progress, self._on_anim_finish)

        # 2-VARIABLE 2D CONTOUR (Circle, Ellipse, Implicit Curves)
        elif len(var_names) == 2:
            F_wrap = lambda X, Y: lam(X, Y)
            cx, cy = extract_2d_contour(F_wrap, xmin, xmax, xmin, xmax, 2400)
            if cx is None:
                self.error_label.setText("No real 2D contour found in this range. Try widening x min/max."); self.error_label.setVisible(True); return

            valid_x, valid_y = cx[np.isfinite(cx)], cy[np.isfinite(cy)]
            min_x, max_x = np.nanmin(valid_x), np.nanmax(valid_x)
            min_y, max_y = np.nanmin(valid_y), np.nanmax(valid_y)
            pad_x = max(0.5, (max_x - min_x) * 0.2)
            pad_y = max(0.5, (max_y - min_y) * 0.2)
            lim = max(max(abs(min_x - pad_x), abs(max_x + pad_x)), max(abs(min_y - pad_y), abs(max_y + pad_y)))

            self.animator.setup_2d_axes(-lim, lim, -lim, lim, latex_str, equal_aspect=True)
            self.audio_engine.prepare(cy, -lim, lim, dur)
            self.animator.start_animation_trace(cx, cy, dur, self._unified_progress, self._on_anim_finish)

        # 3-VARIABLE ANIMATED 2D SLICES (Evolving cross-sections over z)
        else:
            slices: list[tuple[Optional[np.ndarray], Optional[np.ndarray]]] = []
            z_levels = np.linspace(xmin, xmax, 45)
            sound_values: list[float] = []

            for zv in z_levels:
                F_slice = lambda X, Y, z_fixed=zv: lam(X, Y, z_fixed)
                scx, scy = extract_2d_contour(F_slice, xmin, xmax, xmin, xmax, 300)
                if scx is not None:
                    slices.append((scx, scy))
                    r_val = float(np.nanmean(np.sqrt(scx**2 + scy**2)))
                    sound_values.append(r_val)
                else:
                    slices.append((None, None))
                    sound_values.append(0.0)

            lim = max(abs(xmin), abs(xmax)) * 1.2
            self.animator.setup_2d_axes(-lim, lim, -lim, lim, latex_str, equal_aspect=True)
            sound_arr = np.interp(np.linspace(0, 1, 2400), np.linspace(0, 1, len(sound_values)), sound_values)
            self.audio_engine.prepare(sound_arr, 0.0, float(max(1e-4, max(sound_values))), dur)
            self.animator.start_animation_slices(slices, dur, self._unified_progress, self._on_anim_finish)

        self.wall_clock_start, self.accumulated_pause = time.perf_counter(), 0.0
        self.btn_pause.setText("Pause")

    def _unified_progress(self) -> float:
        if self.audio_engine.stream and self.audio_engine.stream.active and self.audio_engine.is_playing:
            return self.audio_engine.get_progress()
        elapsed = (time.perf_counter() - self.wall_clock_start) - self.accumulated_pause
        return min(1.0, max(0.0, elapsed / self.animator.duration))

    def _on_anim_finish(self) -> None:
        self.audio_engine.is_playing = False; self.btn_pause.setText("Replay")

    def toggle_pause(self) -> None:
        if self._unified_progress() >= 1.0: self.replay(); return
        if not self.animator.is_paused:
            self.animator.is_paused, self.audio_engine.is_playing = True, False
            self.pause_timestamp = time.perf_counter(); self.btn_pause.setText("Resume")
        else:
            self.animator.is_paused, self.audio_engine.is_playing = False, True
            self.accumulated_pause += time.perf_counter() - self.pause_timestamp; self.btn_pause.setText("Pause")

    def replay(self) -> None:
        self.audio_engine.set_progress(0.0); self.audio_engine.is_playing = True
        self.wall_clock_start, self.accumulated_pause = time.perf_counter(), 0.0
        self.animator.is_paused = False; self.btn_pause.setText("Pause")

    def toggle_mute(self) -> None:
        self.audio_engine.is_muted = not self.audio_engine.is_muted
        self.btn_mute.setText("Unmute 🔊" if self.audio_engine.is_muted else "Mute 🔇")

    def save_animation(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Video with Audio", "sound_of_math.mp4",
            "MP4 Video with Sound (*.mp4);;GIF Animation (*.gif)"
        )
        if not path: return
        self.audio_engine.is_playing = False; fps, total_frames = 30, int(self.animator.duration * 30)

        self.error_label.setStyleSheet("color: #FFD700; font-size: 11px; font-weight: bold;")
        self.error_label.setText("Exporting video & synthesizing synchronized audio... Please wait.")
        self.error_label.setVisible(True)
        QApplication.processEvents()

        def make_frame(frame: int):
            prog = frame / float(total_frames)
            if not self.animator.is_3d_slice_mode:
                idx = int(prog * (len(self.animator.xs) - 1))
                self.animator.glow1.set_data(self.animator.xs[:idx + 1], self.animator.ys[:idx + 1])
                self.animator.glow2.set_data(self.animator.xs[:idx + 1], self.animator.ys[:idx + 1])
                self.animator.main_line.set_data(self.animator.xs[:idx + 1], self.animator.ys[:idx + 1])
                if np.isfinite(self.animator.ys[idx]):
                    self.animator.dot_main.set_data([self.animator.xs[idx]], [self.animator.ys[idx]])
                    self.animator.dot_glow.set_data([self.animator.xs[idx]], [self.animator.ys[idx]])
                    self.animator.drop_line.set_data([self.animator.xs[idx], self.animator.xs[idx]], [0, self.animator.ys[idx]])
            else:
                s_idx = int(prog * (len(self.animator.slices_3d) - 1))
                sx, sy = self.animator.slices_3d[s_idx]
                if sx is not None:
                    self.animator.main_line.set_data(sx, sy)
                    self.animator.glow1.set_data(sx, sy)
                    self.animator.dot_main.set_data([sx[0]], [sy[0]])
                    self.animator.dot_glow.set_data([sx[0]], [sy[0]])

        save_anim = animation.FuncAnimation(self.fig, make_frame, frames=total_frames, interval=1000 // fps)
        try:
            if path.endswith(".gif"):
                save_anim.save(path, writer="pillow", fps=fps)
            else:
                temp_video = path + ".raw_video.mp4"
                temp_audio = path + ".raw_audio.wav"
                save_anim.save(temp_video, writer="ffmpeg", fps=fps)

                audio_stereo = self.audio_engine.render_audio_buffer(self.animator.duration)
                int16_samples = (np.clip(audio_stereo, -1.0, 1.0) * 32767.0).astype(np.int16)

                import wave, subprocess
                with wave.open(temp_audio, "wb") as wf:
                    wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(self.audio_engine.sample_rate)
                    wf.writeframes(int16_samples.tobytes())

                cmd = ["ffmpeg", "-y", "-i", temp_video, "-i", temp_audio, "-c:v", "copy", "-c:a", "aac", "-shortest", path]
                proc = subprocess.run(cmd, capture_output=True, text=True)

                if os.path.exists(temp_video): os.remove(temp_video)
                if os.path.exists(temp_audio): os.remove(temp_audio)

                if proc.returncode != 0: raise RuntimeError(f"FFmpeg muxing failed: {proc.stderr}")

            self.error_label.setStyleSheet("color: #00FF66; font-size: 11px; font-weight: bold;")
            self.error_label.setText(f"✓ Video with sound successfully saved to {os.path.basename(path)}!")
        except Exception as e:
            self.error_label.setStyleSheet("color: #FF4D4D; font-size: 11px; font-weight: bold;")
            self.error_label.setText(f"Export failed: {e}"); self.error_label.setVisible(True)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "keypad"):
            # Keep keypad bounded inside the window during window resize
            pw, ph = self.width(), self.height()
            kx = min(self.keypad.x(), max(6, pw - self.keypad.width() - 8))
            ky = min(self.keypad.y(), max(50, ph - self.keypad.height() - 8))
            self.keypad.move(max(6, kx), max(50, ky))

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in [Qt.Key.Key_Return, Qt.Key.Key_Enter]: self.plot_formula()
        elif key == Qt.Key.Key_Space: self.toggle_pause()
        elif key == Qt.Key.Key_R: self.replay()
        elif key == Qt.Key.Key_Escape: self.close()
        else: super().keyPressEvent(event)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
