"""Cellular Studio, an interactive viewer for neural cellular automata."""

import argparse
import tkinter as tk
from collections.abc import Callable
from math import ceil, isfinite
from pathlib import Path
from time import perf_counter
from tkinter import filedialog, messagebox, simpledialog, ttk

import torch
from PIL import Image
from torch import Tensor

from compute import configure_cpu, thread_count
from grid_view import GridView
from nca import (
    MAX_EXPORT_SIZE,
    ModelSettings,
    NeuralCA,
    PatternCA,
    TextureCA,
    load_model,
    place_seed,
    seed_grid,
)
from rendering import sprite_image
from training_dialog import TrainingDialog
from ui_help import info_button
from ui_style import configure_style

PROJECT = Path(__file__).resolve().parents[1]
MODELS = PROJECT / "models"
PAPER = (241, 239, 229)
PREVIEW_PLAYING_LIMIT = 384
PREVIEW_PAUSED_LIMIT = 768


class Explorer:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.model: NeuralCA | PatternCA | None = None
        self.settings: ModelSettings | None = None
        self.state: Tensor | None = None
        self.rng = torch.Generator()
        self.playing = False
        self.step_number = 0
        self.design = 0
        self.growth_angle = 0.0
        self.orientation = tk.StringVar(root, "0")
        self.angle_slider_value = tk.DoubleVar(root, 0)
        self._tick_job: str | None = None
        self._model_directory = MODELS
        self.training_dialog: TrainingDialog | None = None
        self._dragging = False
        self._last_cell: tuple[int, int] | None = None
        self._axis = torch.arange(0, dtype=torch.float32)
        self._paper = torch.tensor(PAPER).reshape(3, 1, 1) / 255

        self.speed = tk.DoubleVar(root, 24)
        self.brush = tk.DoubleVar(root, 1)
        self.speed_label = tk.StringVar(root, "24 steps/s")
        self.brush_label = tk.StringVar(root, "3 cells across")
        self.hover_enabled = tk.BooleanVar(root)
        self.grid_enabled = tk.BooleanVar(root)
        self.synchronous = tk.BooleanVar(root)
        self.neural_detail = tk.BooleanVar(root)
        self.crisp_edges = tk.BooleanVar(root)
        self.angle_label = tk.StringVar(root, "Growth angle (°)")
        self.seed_hint = tk.StringVar(root, "Double-click to plant a seed.")
        self.live_morph = tk.BooleanVar(root)
        self.grid_label = tk.StringVar(root)
        self.model_name = tk.StringVar(root, "No model loaded")
        self.status = tk.StringVar(root)
        self.counter = tk.StringVar(root)

        root.title("Cellular Studio")
        root.geometry("1160x790")
        root.protocol("WM_DELETE_WINDOW", self.close)
        # Keep the named Tk fonts alive for the lifetime of the viewer.
        self._fonts = configure_style(root)
        self._build_layout()
        self._bind_events()
        self._sync_controls()
        root.update_idletasks()
        root.minsize(max(760, root.winfo_reqwidth()), max(640, root.winfo_reqheight()))

    def _build_layout(self) -> None:
        layout = ttk.Frame(self.root, padding=(22, 19, 22, 14))
        layout.pack(fill="both", expand=True)
        layout.columnconfigure(0, weight=1)
        layout.rowconfigure(1, weight=1)

        header = ttk.Frame(layout)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 20))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Cellular Studio", style="Heading.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header, text="Grow and steer neural cellular automata.", style="Muted.TLabel"
        ).grid(row=1, column=0, sticky="w", pady=(3, 0))
        ttk.Button(header, text="Open model…", command=self.open_model).grid(
            row=0, column=1, rowspan=2, sticky="e", padx=(8, 0)
        )
        ttk.Button(header, text="Train model…", command=self.train_from_image).grid(
            row=0, column=2, rowspan=2, sticky="e", padx=(8, 0)
        )
        self.export_button = ttk.Button(header, text="Export PNG…", command=self.export_png)
        self.export_button.grid(row=0, column=3, rowspan=2, sticky="e", padx=(8, 0))

        simulation = ttk.Frame(layout, style="Panel.TFrame", padding=14)
        simulation.grid(row=1, column=0, sticky="nsew")
        simulation.columnconfigure(0, weight=1)
        simulation.rowconfigure(2, weight=1)
        title = ttk.Frame(simulation, style="Panel.TFrame")
        title.grid(row=0, column=0, sticky="ew")
        ttk.Label(title, text="Simulation", style="Section.Panel.TLabel").pack(side="left")
        ttk.Label(title, textvariable=self.grid_label, style="Value.Panel.TLabel").pack(
            side="right"
        )

        tools = ttk.Frame(simulation, style="Panel.TFrame")
        tools.grid(row=1, column=0, sticky="ew", pady=(8, 10))
        ttk.Label(tools, text="Design", style="Panel.TLabel").pack(side="left", padx=(0, 8))
        self.design_box = ttk.Combobox(tools, width=14, state="readonly")
        self.design_box.pack(side="left", padx=(0, 18))
        ttk.Label(tools, textvariable=self.angle_label, style="Panel.TLabel").pack(
            side="left", padx=(0, 8)
        )
        self.angle_slider = ttk.Scale(
            tools,
            from_=0,
            to=360,
            length=120,
            variable=self.angle_slider_value,
            command=self._slide_angle,
        )
        self.angle_slider.pack(side="left", padx=(0, 6))
        self.orientation_box = ttk.Spinbox(
            tools,
            width=7,
            from_=0,
            to=359,
            wrap=True,
            textvariable=self.orientation,
            command=self._select_orientation,
        )
        self.orientation_box.pack(side="left")
        info_button(
            tools,
            "Angle",
            "Sprite angle restarts growth. Texture angle steers the local rule on the current canvas. "
            "For animated textures it also rotates the reference motion vectors. Gray–Scott has no preferred direction.",
        ).pack(side="left", padx=(4, 0))
        self.morph_check = ttk.Checkbutton(tools, text="Switch in place", variable=self.live_morph)
        self.morph_check.pack(side="left", padx=(12, 0))
        info_button(
            tools,
            "Switch in place",
            "When enabled, changing Design gives the existing cells a new shape to grow toward. "
            "When disabled, the new design grows from a fresh seed. Available on models trained for live morphing.",
        ).pack(side="left", padx=(4, 0))

        self.view = GridView(simulation)
        self.view.grid(row=2, column=0, sticky="nsew")
        sidebar = ttk.Frame(layout, style="Panel.TFrame", padding=(18, 16))
        sidebar.grid(row=1, column=1, sticky="ns", padx=(16, 0))
        ttk.Label(sidebar, text="Controls", style="Section.Panel.TLabel").pack(
            anchor="w", pady=(0, 12)
        )
        controls = ttk.Frame(sidebar, style="Panel.TFrame")
        controls.pack(fill="x")
        controls.columnconfigure(0, weight=1, minsize=220)
        self._build_controls(controls)

        footer = ttk.Frame(layout)
        footer.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(13, 0))
        ttk.Label(footer, textvariable=self.status, style="Status.TLabel").pack(side="left")
        ttk.Label(footer, textvariable=self.model_name, style="Muted.TLabel", width=36).pack(
            side="left", padx=(16, 10)
        )
        ttk.Label(footer, textvariable=self.counter, style="Counter.TLabel").pack(side="right")

    def _build_controls(self, panel: ttk.Frame) -> None:
        ttk.Label(panel, text="Erase brush", style="Section.Panel.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 13)
        )
        self.brush_slider = self._slider(
            panel, 1, "Size", self.brush_label, self.brush, 0, 6, self._brush_changed
        )
        self.hover_check = ttk.Checkbutton(
            panel, text="Hover to erase", variable=self.hover_enabled, command=self._end_drag
        )
        self.hover_check.grid(row=3, column=0, sticky="w")
        ttk.Label(panel, text="Click and drag over the image.", style="Muted.Panel.TLabel").grid(
            row=4, column=0, sticky="w", pady=(1, 7)
        )
        self.grid_check = ttk.Checkbutton(
            panel,
            text="Show cell grid",
            variable=self.grid_enabled,
            command=lambda: self.view.show_grid(self.grid_enabled.get()),
        )
        self.grid_check.grid(row=5, column=0, sticky="w")
        ttk.Separator(panel).grid(row=6, column=0, sticky="ew", pady=12)

        ttk.Label(panel, text="Playback", style="Section.Panel.TLabel").grid(
            row=7, column=0, sticky="w", pady=(0, 13)
        )
        self.speed_slider = self._slider(
            panel, 8, "Speed", self.speed_label, self.speed, 1, 60, self._speed_changed
        )
        actions = ttk.Frame(panel, style="Panel.TFrame")
        actions.grid(row=10, column=0, sticky="ew")
        actions.columnconfigure((0, 1), weight=1, uniform="playback")
        self.play_button = ttk.Button(actions, style="Accent.TButton", command=self.toggle_play)
        self.play_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.step_button = ttk.Button(actions, text="Step", command=self.step)
        self.step_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.reset_button = ttk.Button(panel, text="Reset to seed", command=self.reset)
        self.reset_button.grid(row=11, column=0, sticky="ew", pady=(7, 8))
        ttk.Label(
            panel,
            textvariable=self.seed_hint,
            style="Muted.Panel.TLabel",
            justify="left",
            wraplength=214,
        ).grid(row=12, column=0, sticky="w", pady=(0, 5))
        self.sync_check = self._explained_check(
            panel,
            13,
            "Synchronous updates",
            self.synchronous,
            "Updates every cell on each step. With this off, sprite and texture models update a random subset. "
            "Reference motion still moves the whole texture. Gray–Scott uses synchronous steps.",
        )
        self.detail_check = self._explained_check(
            panel,
            14,
            "Neural detail",
            self.neural_detail,
            "Uses the model’s learned renderer to draw detail between cells in the preview and exported PNG. "
            "Growth and repair still happen on the cell grid. Available on models with a detail decoder.",
            self._refresh_image,
        )
        self.crisp_check = self._explained_check(
            panel,
            15,
            "Crisp edges",
            self.crisp_edges,
            "Makes pixels at least 50% opaque fully opaque and the others transparent. "
            "This gives transparent artwork hard edges in the preview and exported PNG. "
            "Fully opaque textures already have no transparency edges.",
            self._refresh_image,
        )

    @staticmethod
    def _explained_check(
        panel: ttk.Frame,
        row: int,
        label: str,
        value: tk.BooleanVar,
        description: str,
        command: Callable[[], None] | None = None,
    ) -> ttk.Checkbutton:
        line = ttk.Frame(panel, style="Panel.TFrame")
        line.grid(row=row, column=0, sticky="w")
        check = ttk.Checkbutton(line, text=label, variable=value, command=command)
        check.pack(side="left")
        info_button(line, label, description).pack(side="left", padx=(4, 0))
        return check

    @staticmethod
    def _slider(
        parent: ttk.Frame,
        row: int,
        name: str,
        label: tk.StringVar,
        variable: tk.DoubleVar,
        minimum: int,
        maximum: int,
        on_change: Callable[[str], None],
    ) -> ttk.Scale:
        caption = ttk.Frame(parent, style="Panel.TFrame")
        caption.grid(row=row, column=0, sticky="ew")
        ttk.Label(caption, text=name, style="Panel.TLabel").pack(side="left")
        ttk.Label(caption, textvariable=label, style="Value.Panel.TLabel").pack(side="right")
        scale = ttk.Scale(
            parent, from_=minimum, to=maximum, variable=variable, command=on_change, takefocus=True
        )
        scale.grid(row=row + 1, column=0, sticky="ew", pady=(4, 9))
        return scale

    def _bind_events(self) -> None:
        self.root.bind("<Control-o>", self.open_model)
        if self.root.tk.call("tk", "windowingsystem") == "aqua":
            self.root.bind("<Command-o>", self.open_model)
        self.design_box.bind("<<ComboboxSelected>>", self._select_design)
        self.orientation_box.bind("<Return>", self._select_orientation)
        self.orientation_box.bind("<FocusOut>", self._select_orientation)
        self.view.bind("<space>", lambda _event: self.toggle_play())
        self.view.bind("<Configure>", self._resize_detail_preview, add="+")
        self.view.bind("<Button-1>", self._start_drag)
        self.view.bind("<B1-Motion>", self._drag)
        self.view.bind("<Enter>", self._preview_brush)
        self.view.bind("<Motion>", self._hover)
        self.view.bind("<ButtonRelease-1>", self._end_drag)
        self.view.bind("<Leave>", self._leave_grid)
        self.view.bind("<Double-Button-1>", self._plant_seed)

    def _sync_controls(self) -> None:
        loaded = self.model is not None
        for widget in (
            self.export_button,
            self.play_button,
            self.reset_button,
            self.speed_slider,
            self.brush_slider,
            self.hover_check,
            self.grid_check,
        ):
            widget.state(("!disabled",) if loaded else ("disabled",))
        for widget in (self.sync_check, self.orientation_box, self.angle_slider):
            widget.state(("!disabled",) if isinstance(self.model, NeuralCA) else ("disabled",))
        self.morph_check.state(
            ("!disabled",) if self.settings and self.settings.live_morph else ("disabled",)
        )
        multiple_designs = self.settings is not None and len(self.settings.design_names) > 1
        self.design_box.state(("!disabled",) if multiple_designs else ("disabled",))
        has_decoder = self.model is not None and self.model.decoder is not None
        self.detail_check.state(("!disabled",) if has_decoder else ("disabled",))
        has_alpha = isinstance(self.model, NeuralCA) and (
            not isinstance(self.model, TextureCA) or self.model.has_alpha
        )
        self.crisp_check.state(("!disabled",) if has_alpha else ("disabled",))
        self.step_button.state(("!disabled",) if loaded and not self.playing else ("disabled",))
        self.play_button.configure(text="Pause" if self.playing else "Play")
        self._sync_status()

    def _sync_status(self) -> None:
        self.status.set(("Running" if self.playing else "Paused") if self.model else "Ready")
        self.counter.set(f"Step {self.step_number:,}" if self.model else "")

    def train_from_image(self) -> None:
        if self.training_dialog and self.training_dialog.window.winfo_exists():
            self.training_dialog.window.lift()
            return
        self.training_dialog = TrainingDialog(
            self.root, PROJECT / "assets", MODELS, self._training_complete
        )

    def _training_complete(self, path: Path) -> None:
        self._model_directory = path.resolve().parent
        messagebox.showinfo(
            "Model saved", f"{path}\n\nUse Open model to load it.", parent=self.root
        )

    def open_model(self, _event: tk.Event | None = None) -> None:
        was_playing = self.playing
        self.pause()
        filename = filedialog.askopenfilename(
            parent=self.root,
            title="Open NCA model",
            initialdir=self._model_directory,
            filetypes=(("NCA checkpoint", "*.pt"), ("All files", "*")),
        )
        if filename and self.load_checkpoint(Path(filename)):
            return
        if was_playing:
            self.toggle_play()

    def load_checkpoint(self, path: Path) -> bool:
        try:
            model, settings = load_model(path)
        except (OSError, RuntimeError, ValueError) as error:
            detail = str(error).splitlines()[0] if str(error) else type(error).__name__
            messagebox.showerror(
                "Could not open model", f"{path.name}\n\n{detail}", parent=self.root
            )
            return False
        self.model, self.settings = model, settings
        self.live_morph.set(settings.live_morph)
        pattern = isinstance(model, PatternCA)
        texture = isinstance(model, TextureCA)
        self.angle_label.set("Direction (°)" if texture else "Growth angle (°)")
        self.seed_hint.set(
            "Double-click to reseed a patch." if texture else "Double-click to plant a seed."
        )
        self.reset_button.configure(text="Reset field" if texture or pattern else "Reset to seed")
        self.speed_slider.configure(to=240 if pattern else 60)
        self.speed.set(120 if pattern else min(self.speed.get(), 60))
        self._speed_changed(str(self.speed.get()))
        self._model_directory = path.resolve().parent
        self.design = 0
        self.growth_angle = 0.0
        self.orientation.set("0")
        self.angle_slider_value.set(0)
        self.synchronous.set(pattern)
        self.neural_detail.set(settings.detail_scale is not None and not settings.pixel_art)
        self.crisp_edges.set(settings.pixel_art)
        self.design_box.configure(values=settings.design_names)
        self.design_box.current(0)
        self.model_name.set(f"{path.parent.name}/{path.name}" if path.parent.name else path.name)
        size = settings.grid_size
        self.grid_label.set(f"{size} × {size} cells")
        self._axis = torch.arange(size, dtype=torch.float32)
        self.reset()
        self.view.focus_set()
        return True

    def _select_design(self, _event: tk.Event) -> None:
        selection = self.design_box.current()
        if selection >= 0 and selection != self.design:
            self.design = selection
            if not self.live_morph.get():
                self._restart_growth()

    def _select_orientation(self, _event: tk.Event | None = None) -> None:
        try:
            degrees = float(self.orientation.get())
        except ValueError:
            degrees = self.growth_angle
        angle = degrees % 360 if isfinite(degrees) else self.growth_angle
        self.angle_slider_value.set(angle)
        self._apply_growth_angle(angle)

    def _slide_angle(self, value: str) -> None:
        self._apply_growth_angle(float(round(float(value))) % 360)

    def _apply_growth_angle(self, angle: float) -> None:
        self.orientation.set(str(int(angle)) if angle.is_integer() else str(angle))
        if angle != self.growth_angle:
            self.growth_angle = angle
            if not isinstance(self.model, TextureCA):
                self._restart_growth()

    def pause(self) -> None:
        self.playing = False
        if self._tick_job is not None:
            self.root.after_cancel(self._tick_job)
            self._tick_job = None
        self._sync_controls()

    def _resize_detail_preview(self, _event: tk.Event | None = None) -> None:
        if not self.playing and self.neural_detail.get():
            self._refresh_image()

    def toggle_play(self) -> None:
        if self.model is None:
            return
        if self.playing:
            self.pause()
            self._resize_detail_preview()
        else:
            self.playing = True
            self._sync_controls()
            self._tick_job = self.root.after(0, self._tick)

    def _tick(self) -> None:
        self._tick_job = None
        if not self.playing:
            return
        started = perf_counter()
        rate = round(self.speed.get())
        updates = ceil(rate / 60)
        self.step(updates)
        if self.playing:
            delay = updates / rate - (perf_counter() - started)
            self._tick_job = self.root.after(max(1, round(delay * 1_000)), self._tick)

    @torch.no_grad()
    def step(self, updates: int = 1) -> None:
        if self.model is None or self.state is None:
            return
        state = self.state
        if isinstance(self.model, PatternCA):
            for _ in range(updates):
                state = self.model(state)
        else:
            texture_step = {"step": self.step_number} if isinstance(self.model, TextureCA) else {}
            state = self.model(
                state,
                self.design,
                synchronous=self.synchronous.get(),
                angle=self.growth_angle,
                generator=self.rng,
                **texture_step,
            )
            updates = 1
        if not torch.isfinite(state).all():
            self.pause()
            self.status.set("Unstable")
            return
        self.state = state
        self.step_number += updates
        self._refresh_image()
        self._sync_status()

    def reset(self) -> None:
        self.pause()
        self._restart_growth()

    def _restart_growth(self) -> None:
        if self.settings is None or self.model is None:
            return
        self._end_drag()
        self.rng.manual_seed(0)
        size = self.settings.grid_size
        if isinstance(self.model, PatternCA):
            self.state = self.model.initial_field(size, self.rng)
        elif isinstance(self.model, TextureCA):
            self.state = self.model.initial_field(self.rng)
        else:
            self.state = seed_grid(size, structured=self.model.isotropic, angle=self.growth_angle)
        self.step_number = 0
        self._refresh_image()
        self._sync_status()

    def _speed_changed(self, value: str) -> None:
        self.speed_label.set(f"{round(float(value))} steps/s")

    def _brush_changed(self, value: str) -> None:
        radius = round(float(value))
        diameter = 2 * radius + 1
        self.brush_label.set("1 cell across" if diameter == 1 else f"{diameter} cells across")
        self.view.resize_brush(radius)

    def _start_drag(self, event: tk.Event) -> None:
        self.view.focus_set()
        self._last_cell = self.view.cell_at(event.x, event.y)
        self._dragging = self._last_cell is not None
        self._preview_brush(event)

    def _end_drag(self, _event: tk.Event | None = None) -> None:
        self._dragging = False
        self._last_cell = None

    def _leave_grid(self, _event: tk.Event) -> None:
        self._last_cell = None
        self.view.hide_brush()

    def _drag(self, event: tk.Event) -> None:
        self._preview_brush(event)
        if self._dragging:
            self._erase(event)

    def _preview_brush(self, event: tk.Event) -> None:
        self.view.show_brush(event.x, event.y, round(self.brush.get()))

    def _hover(self, event: tk.Event) -> None:
        self._preview_brush(event)
        if self.hover_enabled.get():
            self._erase(event)

    def _erase(self, event: tk.Event) -> None:
        if self.state is None:
            return
        cell = self.view.cell_at(event.x, event.y)
        previous, self._last_cell = self._last_cell, cell
        if cell is None or cell == previous:
            return
        start_x, start_y = previous or cell
        dx, dy = cell[0] - start_x, cell[1] - start_y
        offset_x, offset_y = self._axis - start_x, self._axis[:, None] - start_y
        projection = ((offset_x * dx + offset_y * dy) / max(1, dx * dx + dy * dy)).clamp(0, 1)
        distance = (offset_x - projection * dx).square() + (offset_y - projection * dy).square()
        radius = max(0.5, round(self.brush.get()))
        mask = distance <= radius**2
        if isinstance(self.model, PatternCA):
            quiescent = self.state.new_tensor([1, 0]).view(1, 2, 1, 1)
            self.state = torch.where(mask, quiescent, self.state)
        else:
            self.state = self.state.masked_fill(mask, 0)
            if isinstance(self.model, TextureCA) and not self.model.has_alpha:
                self.state[:, 3] = 1
        self._refresh_image()

    def _plant_seed(self, event: tk.Event) -> None:
        self._end_drag()
        cell = self.view.cell_at(event.x, event.y)
        if self.state is not None and self.model is not None and cell is not None:
            if isinstance(self.model, (PatternCA, TextureCA)):
                size = self.state.shape[-1]
                dx = (self._axis - cell[0] + size // 2) % size - size // 2
                dy = (self._axis - cell[1] + size // 2) % size - size // 2
                mask = dy[:, None].square() + dx.square() <= (size // 16) ** 2
                if isinstance(self.model, TextureCA):
                    self.state = torch.where(mask, self.model.initial_field(self.rng), self.state)
                else:
                    self.state[0, 0, mask] = 0.52
                    self.state[0, 1, mask] = 0.25
            else:
                place_seed(
                    self.state, *cell, structured=self.model.isotropic, angle=self.growth_angle
                )
            self._refresh_image()

    def _appearance(self, state: Tensor) -> Tensor:
        if isinstance(self.model, PatternCA):
            return self.model.colorize(state)[0]
        return state[0, :4]

    @torch.no_grad()
    def _refresh_image(self) -> None:
        if self.state is None or self.model is None:
            return
        rgba = self._appearance(self.state)
        if self.neural_detail.get() and self.model.decoder is not None:
            limit = PREVIEW_PLAYING_LIMIT if self.playing else PREVIEW_PAUSED_LIMIT
            size = min(limit, max(2 * rgba.shape[-1], self.view.display_size()))
            rgba = self.model.decoder.render(self.state, size, size)[0]
        crisp = self.crisp_edges.get()
        image = sprite_image(rgba, crisp=crisp, background=self._paper)
        self.view.set_image(image, self.state.shape[-1], crisp=crisp)

    @torch.no_grad()
    def export_png(self) -> None:
        if self.state is None or self.model is None or self.settings is None:
            return
        state = self.state.clone()
        decoder = self.model.decoder if self.neural_detail.get() else None
        coarse = self._appearance(state)
        crisp = self.crisp_edges.get()
        design = self.settings.design_names[self.design]
        scale = (self.settings.detail_scale or 4) if decoder is not None else 1
        size = simpledialog.askinteger(
            "Export PNG",
            "Image width and height in pixels",
            parent=self.root,
            initialvalue=self.settings.grid_size * scale,
            minvalue=1,
            maxvalue=MAX_EXPORT_SIZE,
        )
        if size is None:
            return
        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export PNG",
            defaultextension=".png",
            filetypes=(("PNG image", "*.png"),),
            initialfile=f"{design}.png",
        )
        if not filename:
            return
        try:
            rgba = decoder.render(state, size, size)[0] if decoder is not None else coarse
            image = sprite_image(rgba, crisp=crisp)
            if decoder is None:
                image = image.resize((size, size), Image.Resampling.NEAREST)
            image.save(filename, format="PNG")
        except (OSError, RuntimeError, ValueError) as error:
            messagebox.showerror("Could not export PNG", str(error), parent=self.root)

    def close(self) -> None:
        if (
            self.training_dialog
            and self.training_dialog.window.winfo_exists()
            and not self.training_dialog.close()
        ):
            return
        self.pause()
        self.view.cancel_redraw()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, nargs="?", help="Checkpoint to open at launch")
    parser.add_argument(
        "--cpu-threads", type=thread_count, default=0, help="CPU playback workers (default: 1)"
    )
    args = parser.parse_args()
    configure_cpu(args.cpu_threads, automatic=1)
    root = tk.Tk()
    explorer = Explorer(root)
    if args.model is not None:
        explorer.load_checkpoint(args.model)
    root.mainloop()


if __name__ == "__main__":
    main()
