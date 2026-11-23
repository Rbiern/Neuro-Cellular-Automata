"""Configure and run the image trainer without blocking the Studio viewer."""

import tkinter as tk
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import torch
from PIL import Image

from nca import (
    MAX_DESIGNS,
    MAX_DETAIL_SCALE,
    MAX_GRID_SIZE,
    MIN_GRID_SIZE,
    MODES,
    PERCEPTIONS,
    load_model,
)
from train import training_settings, validate_training_options
from training_job import TrainingJob
from ui_help import info_button

TRAINER = Path(__file__).resolve().with_name("train.py")
PATTERN_TRAINER = TRAINER.with_name("train_patterns.py")
TEXTURE_TRAINER = TRAINER.with_name("train_textures.py")
SPRITE = "Sprite from images"
PATTERN = "Living patterns (Gray–Scott)"
TEXTURE = "Texture from image"
ANIMATION = "Animated texture"
IMAGE_TYPES = (("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"), ("All files", "*"))
OPTION_HELP = {
    "Model type": "Sprite training grows a target shape. Living patterns learns Gray–Scott dynamics. "
    "Texture training fills the canvas with the appearance of a reference image. "
    "Animated texture also uses a motion reference.",
    "Target images": "Add up to 32 images. Their filenames become the design names. Move targets up or down "
    "to change the order, especially when reusing a checkpoint.",
    "Training mode": "Growing learns to reach a target from seed. Persistent also learns to hold the shape. "
    "Regenerating adds damaged examples so the model can learn repair.",
    "Grid width in cells": "Sets the size of the simulation grid. Larger grids can represent finer structures "
    "and use more training time and memory.",
    "Training iterations": "The number of optimizer updates. More iterations take longer and give the "
    "model more opportunities to learn.",
    "Preserve hard pixel edges in targets": "Uses nearest-neighbor sampling for target images to keep pixel "
    "edges crisp when resizing.",
    "Train neural detail renderer": "Trains a decoder to render between simulation cells. It adds training "
    "work and can preserve fine shading from higher-resolution sources.",
    "Remove flat background": "Removes a uniform background connected to the image border. "
    "Images with transparency keep their existing alpha.",
    "Train live morphing between targets": "Trains transitions between grown designs without restarting. "
    "Requires two or more targets, a batch of at least 3, "
    "and persistent or regenerating mode.",
    "Perception": "Sobel reads directional gradients. Isotropic reads local changes without a preferred "
    "direction and uses a structured seed to orient the sprite.",
    "Training device": "Auto uses CUDA if PyTorch can access it and falls back to CPU. "
    "Choose CPU to train on the processor or CUDA to require a compatible NVIDIA GPU.",
    "CPU threads": "Use 0 for automatic or enter a worker count. Larger workloads can benefit from more "
    "threads. CUDA training leaves this setting unused.",
    "Batch size": "How many simulation grids the model trains on per optimizer update. "
    "Larger batches generally use more memory.",
    "Training seed": "Controls randomized training operations such as initial weights and generated "
    "examples. Reuse it to repeat the training setup.",
    "Scale rollout length with grid size": "Sets the minimum to the grid width and the maximum to 1.5 times "
    "that width, rounded down. Turn this off to enter the range yourself.",
    "Minimum rollout steps": "The shortest simulation run used for a training example.",
    "Maximum rollout steps": "The longest simulation run used for a training example. "
    "Longer rollouts give growth more time and increase memory and compute.",
    "Detail pixels per cell": "The resolution of the decoder's training image per simulation cell. "
    "This increases image resolution while keeping the simulation grid the same.",
    "Detail samples per iteration": "How many pixel positions supervise the decoder per optimizer update. "
    "More samples increase training work.",
    "Warm-start checkpoint": "Starts from learned weights. Match target names and order, grid width, "
    "perception, and decoder configuration. Save to a separate checkpoint.",
    "Living patterns": "Uses built-in Gray–Scott trajectories and runs on CPU. The training seed changes the "
    "examples and initialization. The simulator uses the built-in parameter regime.",
    "Texture image": "Choose one example of the colors and local structure you want. "
    "Textures work best with a repeated subject that can fill the canvas.",
    "Motion reference": "Choose an animated GIF, APNG, WebP, a folder of numbered frames, or a NumPy flow field. "
    "Standard video files require FFmpeg. The reference supplies movement for the texture image.",
    "Rollout steps": "Simulation updates between training comparisons. Longer rollouts cost more memory and time.",
}


class TrainingDialog:
    def __init__(
        self,
        parent: tk.Tk,
        image_directory: Path,
        model_directory: Path,
        on_complete: Callable[[Path], None],
    ) -> None:
        self.image_directory = image_directory
        self.model_directory = model_directory
        self.on_complete = on_complete
        self.targets: list[Path] = []
        self.first_selection = True
        self.init_model: Path | None = None
        self.window = tk.Toplevel(parent)
        self.window.title("Train model")
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.job: TrainingJob | None = None
        self.poll_job: str | None = None
        self.inputs: list[ttk.Widget | tk.Listbox] = []
        self.required_markers: dict[ttk.Spinbox, ttk.Label] = {}

        root = self.window
        self.model_type = tk.StringVar(root, SPRITE)
        self.cpu_threads = tk.StringVar(root, "0")
        self.pattern_iterations = tk.StringVar(root, "6000")
        self.pattern_seed = tk.StringVar(root, "0")
        self.texture_image: Path | None = None
        self.motion_reference: Path | None = None
        self.texture_image_label = tk.StringVar(root, "Choose image…")
        self.motion_label = tk.StringVar(root, "Choose file…")
        self.texture_size = tk.StringVar(root, "32")
        self.texture_iterations = tk.StringVar(root, "800")
        self.texture_steps = tk.StringVar(root, "12")
        self.texture_batch = tk.StringVar(root, "2")
        self.texture_device = tk.StringVar(root, "auto")
        self.texture_seed = tk.StringVar(root, "0")
        self.texture_detail = tk.BooleanVar(root)
        self.texture_scale = tk.StringVar(root, "4")
        self.size = tk.StringVar(root, "48")
        self.iterations = tk.StringVar(root, "1600")
        self.mode = tk.StringVar(root, "regenerating")
        self.perception = tk.StringVar(root, "sobel")
        self.device = tk.StringVar(root, "auto")
        self.batch_size = tk.StringVar(root, "3")
        self.seed = tk.StringVar(root, "0")
        self.min_steps = tk.StringVar(root, "48")
        self.max_steps = tk.StringVar(root, "72")
        self.detail_scale = tk.StringVar(root, "8")
        self.detail_samples = tk.StringVar(root, "4096")
        self.auto_steps = tk.BooleanVar(root, True)
        self.pixel_art = tk.BooleanVar(root)
        self.neural_detail = tk.BooleanVar(root, True)
        self.remove_background = tk.BooleanVar(root)
        self.live_morph = tk.BooleanVar(root)
        self.checkpoint_label = tk.StringVar(root, "Train from scratch")
        self.target_count = tk.StringVar(root)
        self.status = tk.StringVar(root, "Add images to set up training.")

        panel = ttk.Frame(root, padding=16)
        panel.pack(fill="both", expand=True)
        header = ttk.Frame(panel)
        header.pack(fill="x", pady=(0, 8))
        header.columnconfigure(1, weight=1)
        self._choice(
            header,
            0,
            "Model type",
            self.model_type,
            (SPRITE, PATTERN, TEXTURE, ANIMATION),
            self._select_type,
        )
        self._number(header, 1, "CPU threads", self.cpu_threads, 0, 64)
        content = ttk.Frame(panel)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)
        self.notebook = ttk.Notebook(content, style="Training.TNotebook")
        self.notebook.grid(row=0, column=0, sticky="nsew")
        basic = ttk.Frame(self.notebook, padding=12)
        advanced = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(basic, text="Setup")
        self.notebook.add(advanced, text="Advanced")
        for tab in (basic, advanced):
            tab.columnconfigure(1, weight=1)
        self._build_setup(basic)
        self._build_advanced(advanced)
        self.pattern_panel = ttk.Frame(content, padding=12)
        self.pattern_panel.columnconfigure(1, weight=1)
        self.pattern_panel.grid(row=0, column=0, sticky="nsew")
        self._help_label(self.pattern_panel, 0, "Living patterns")
        ttk.Label(
            self.pattern_panel,
            text="Train on simulated Gray–Scott trajectories using CPU.",
            wraplength=460,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self._number(
            self.pattern_panel, 2, "Training iterations", self.pattern_iterations, 1, 10000
        )
        self._number(self.pattern_panel, 3, "Training seed", self.pattern_seed, 0, 2**32 - 1)
        self.pattern_panel.grid_remove()
        self.texture_panel = ttk.Frame(content, padding=12)
        self.texture_panel.columnconfigure(1, weight=1)
        self.texture_panel.grid(row=0, column=0, sticky="nsew")
        self._build_texture(self.texture_panel)
        self.texture_panel.grid_remove()

        self.progress = ttk.Progressbar(panel, mode="determinate")
        self.progress.pack(fill="x", pady=(12, 5))
        ttk.Label(panel, textvariable=self.status, wraplength=510, justify="left").pack(anchor="w")
        actions = ttk.Frame(panel)
        actions.pack(fill="x", pady=(10, 0))
        self.close_button = ttk.Button(actions, text="Close", command=self.close)
        self.close_button.pack(side="left")
        self.start_button = ttk.Button(actions, text="Save and train…", command=self.start)
        self.start_button.pack(side="right")
        self.inputs.append(self.start_button)
        self.size.trace_add("write", self._refresh_options)
        self._refresh_targets()
        root.update_idletasks()
        root.minsize(root.winfo_reqwidth(), root.winfo_reqheight())

    def _build_setup(self, tab: ttk.Frame) -> None:
        self._help_label(tab, 0, "Target images", self.target_count, required=True)
        targets = ttk.Frame(tab)
        targets.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(5, 0))
        tab.rowconfigure(1, weight=1)
        self.target_list = tk.Listbox(
            targets,
            width=52,
            height=5,
            exportselection=False,
            activestyle="none",
            background="#181f1c",
            foreground="#f2f0e6",
            selectbackground="#506348",
        )
        self.target_list.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(targets, orient="vertical", command=self.target_list.yview)
        scrollbar.pack(side="right", fill="y")
        self.target_list.configure(yscrollcommand=scrollbar.set)
        self.inputs.append(self.target_list)
        self.target_list.bind("<<ListboxSelect>>", self._refresh_options)
        buttons = ttk.Frame(tab)
        buttons.grid(row=2, column=0, columnspan=2, sticky="w", pady=(5, 10))
        for label, action in (
            ("Add images…", self.add_targets),
            ("Remove", self.remove_target),
            ("Move up", lambda: self.move_target(-1)),
            ("Move down", lambda: self.move_target(1)),
        ):
            button = ttk.Button(buttons, text=label, command=action)
            button.pack(side="left", padx=(0, 4))
            self.inputs.append(button)
        self.add_button, self.remove_button, self.up_button, self.down_button = (
            buttons.winfo_children()
        )

        self._choice(tab, 3, "Training mode", self.mode, MODES)
        self._number(tab, 4, "Grid width in cells", self.size, MIN_GRID_SIZE, MAX_GRID_SIZE)
        self._number(tab, 5, "Training iterations", self.iterations, 1, 10000)
        self._check(tab, 6, "Preserve hard pixel edges in targets", self.pixel_art)
        self._check(tab, 7, "Train neural detail renderer", self.neural_detail)
        self._check(tab, 8, "Remove flat background", self.remove_background)
        self.morph_check = self._check(
            tab, 9, "Train live morphing between targets", self.live_morph
        )
        ttk.Label(
            tab,
            text="Live morphing uses two or more targets and a batch size of at least 3.",
            wraplength=460,
            justify="left",
        ).grid(row=10, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def _build_advanced(self, tab: ttk.Frame) -> None:
        self._choice(tab, 0, "Perception", self.perception, PERCEPTIONS)
        self._choice(tab, 1, "Training device", self.device, ("auto", "cpu", "cuda"))
        self._number(tab, 2, "Batch size", self.batch_size, 1, 128)
        self._number(tab, 3, "Training seed", self.seed, 0, 2**32 - 1)
        self._check(tab, 4, "Scale rollout length with grid size", self.auto_steps)
        self.minimum_field = self._number(tab, 5, "Minimum rollout steps", self.min_steps, 1, 2048)
        self.maximum_field = self._number(tab, 6, "Maximum rollout steps", self.max_steps, 1, 2048)
        self.scale_field = self._number(
            tab, 7, "Detail pixels per cell", self.detail_scale, 2, MAX_DETAIL_SCALE
        )
        self.samples_field = self._number(
            tab, 8, "Detail samples per iteration", self.detail_samples, 1, 32768
        )
        ttk.Separator(tab).grid(row=9, column=0, columnspan=2, sticky="ew", pady=10)
        self._help_label(tab, 10, "Warm-start checkpoint", self.checkpoint_label, columnspan=2)
        choose = ttk.Button(tab, text="Use existing checkpoint…", command=self.choose_checkpoint)
        choose.grid(row=11, column=0, sticky="w", pady=(6, 0))
        self.clear_checkpoint = ttk.Button(
            tab, text="Clear checkpoint", command=self.reset_checkpoint
        )
        self.clear_checkpoint.grid(row=11, column=1, sticky="e", pady=(6, 0))
        self.inputs.extend((choose, self.clear_checkpoint))
        ttk.Label(
            tab,
            text="The warm-start model must match the target names and order. "
            "Choosing a checkpoint copies its grid and model settings.",
            wraplength=460,
            justify="left",
        ).grid(row=12, column=0, columnspan=2, sticky="w", pady=(7, 0))

    def _build_texture(self, tab: ttk.Frame) -> None:
        self._help_label(tab, 0, "Texture image", required=True)
        image = ttk.Button(
            tab, textvariable=self.texture_image_label, command=self._pick_texture_image
        )
        image.grid(row=0, column=1, sticky="ew", pady=3)
        self.motion_row = ttk.Frame(tab)
        self.motion_row.columnconfigure(1, weight=1)
        self.motion_row.grid(row=1, column=0, columnspan=2, sticky="ew")
        self._help_label(self.motion_row, 0, "Motion reference", required=True)
        choices = ttk.Frame(self.motion_row)
        choices.grid(row=0, column=1, sticky="e", pady=3)
        motion = ttk.Button(choices, textvariable=self.motion_label, command=self._pick_motion)
        motion.pack(side="left", padx=(0, 4))
        frames = ttk.Button(
            choices, text="Frame folder…", command=lambda: self._pick_motion(folder=True)
        )
        frames.pack(side="left")
        self.inputs.extend((image, motion, frames))
        self._number(tab, 2, "Grid width in cells", self.texture_size, MIN_GRID_SIZE, MAX_GRID_SIZE)
        self._number(tab, 3, "Training iterations", self.texture_iterations, 1, 10000)
        self._number(tab, 4, "Rollout steps", self.texture_steps, 1, 128)
        self._number(tab, 5, "Batch size", self.texture_batch, 1, 32)
        self._choice(tab, 6, "Training device", self.texture_device, ("auto", "cpu", "cuda"))
        self._number(tab, 7, "Training seed", self.texture_seed, 0, 2**32 - 1)
        self._check(tab, 8, "Train neural detail renderer", self.texture_detail)
        self.texture_scale_field = self._number(
            tab, 9, "Detail pixels per cell", self.texture_scale, 2, MAX_DETAIL_SCALE
        )

    def _pick_texture_image(self) -> None:
        filename = filedialog.askopenfilename(
            parent=self.window,
            title="Choose texture image",
            initialdir=self.image_directory,
            filetypes=IMAGE_TYPES,
        )
        if filename:
            self.texture_image = Path(filename).resolve()
            self.texture_image_label.set(self._short_filename(self.texture_image))
            self.image_directory = self.texture_image.parent
            self._refresh_options()

    def _pick_motion(self, *, folder: bool = False) -> None:
        if folder:
            filename = filedialog.askdirectory(
                parent=self.window,
                title="Choose numbered frame folder",
                initialdir=self.image_directory,
                mustexist=True,
            )
        else:
            filename = filedialog.askopenfilename(
                parent=self.window,
                title="Choose motion reference",
                initialdir=self.image_directory,
                filetypes=(
                    (
                        "Motion references",
                        "*.gif *.png *.apng *.webp *.mp4 *.mov *.webm *.mkv *.avi *.npy",
                    ),
                    ("All files", "*"),
                ),
            )
        if filename:
            self.motion_reference = Path(filename).resolve()
            self.motion_label.set(self._short_filename(self.motion_reference))
            self.image_directory = self.motion_reference.parent
            self._refresh_options()

    @staticmethod
    def _short_filename(path: Path) -> str:
        return path.name if len(path.name) <= 26 else f"{path.name[:23]}…"

    def _help_icon(self, frame: ttk.Frame, label: str) -> ttk.Button:
        return info_button(frame, label, OPTION_HELP[label], style="Training.Info.TButton")

    def _help_label(
        self,
        tab: ttk.Frame,
        row: int,
        label: str,
        value: tk.StringVar | None = None,
        *,
        columnspan: int = 1,
        required: bool = False,
    ) -> ttk.Label | None:
        line = ttk.Frame(tab)
        line.grid(row=row, column=0, columnspan=columnspan, sticky="w", padx=(0, 12), pady=3)
        ttk.Label(line, text=label, textvariable=value, wraplength=440).pack(side="left")
        marker = None
        if required:
            marker = ttk.Label(line, text="*", foreground="#ff7777", width=1)
            marker.pack(side="left", padx=(3, 0))
        self._help_icon(line, label).pack(side="left", padx=(4, 0))
        return marker

    def _choice(
        self,
        tab: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        values: tuple[str, ...],
        on_select: Callable[[tk.Event], None] | None = None,
    ) -> None:
        self._help_label(tab, row, label, required=True)
        field = ttk.Combobox(tab, textvariable=variable, values=values, state="readonly", width=19)
        field.grid(row=row, column=1, sticky="ew", pady=3)
        field.bind("<<ComboboxSelected>>", on_select or self._refresh_options)
        self.inputs.append(field)

    def _number(
        self,
        tab: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        minimum: int,
        maximum: int,
    ) -> ttk.Spinbox:
        marker = self._help_label(tab, row, label, required=True)
        field = ttk.Spinbox(tab, from_=minimum, to=maximum, textvariable=variable, width=19)
        field.grid(row=row, column=1, sticky="ew", pady=3)
        self.inputs.append(field)
        if marker is not None:
            self.required_markers[field] = marker
        return field

    def _check(
        self, tab: ttk.Frame, row: int, label: str, variable: tk.BooleanVar
    ) -> ttk.Checkbutton:
        line = ttk.Frame(tab)
        line.grid(row=row, column=0, columnspan=2, sticky="w")
        field = ttk.Checkbutton(line, text=label, variable=variable, command=self._refresh_options)
        field.pack(side="left")
        self._help_icon(line, label).pack(side="left", padx=(4, 0))
        self.inputs.append(field)
        return field

    def _select_type(self, _event: tk.Event) -> None:
        kind = self.model_type.get()
        self.notebook.grid_remove()
        self.pattern_panel.grid_remove()
        self.texture_panel.grid_remove()
        if kind == PATTERN:
            self.pattern_panel.grid()
            self.status.set("Choose the training iterations and seed, then save and train.")
        elif kind in (TEXTURE, ANIMATION):
            self.texture_panel.grid()
            if kind == ANIMATION:
                self.motion_row.grid()
                self.status.set("Choose a texture image and a motion reference.")
            else:
                self.motion_row.grid_remove()
                self.status.set("Choose an image with the pattern you want to synthesize.")
        else:
            self.notebook.grid()
            self.status.set(
                "Choose the sprite settings, then save and train."
                if self.targets
                else "Add images to set up training."
            )
        self.progress.configure(value=0)
        self._refresh_options()

    def _refresh_options(self, *_event: object) -> None:
        if self.job is not None:
            return
        selection = self.target_list.curselection()
        index = selection[0] if selection else -1
        auto = self.auto_steps.get()
        if auto:
            try:
                size = int(self.size.get())
            except ValueError:
                pass
            else:
                self.min_steps.set(str(size))
                self.max_steps.set(str(size + size // 2))
        can_morph = len(self.targets) >= 2 and self.mode.get() != "growing"
        if not can_morph:
            self.live_morph.set(False)
        detail = self.neural_detail.get()
        texture_ready = self.texture_image is not None and (
            self.model_type.get() == TEXTURE or self.motion_reference is not None
        )
        ready = {
            SPRITE: bool(self.targets),
            PATTERN: True,
            TEXTURE: texture_ready,
            ANIMATION: texture_ready,
        }[self.model_type.get()]
        for widget, enabled in (
            (self.add_button, len(self.targets) < MAX_DESIGNS),
            (self.remove_button, index >= 0),
            (self.up_button, index > 0),
            (self.down_button, 0 <= index < len(self.targets) - 1),
            (self.minimum_field, not auto),
            (self.maximum_field, not auto),
            (self.scale_field, detail),
            (self.samples_field, detail),
            (self.morph_check, can_morph),
            (self.clear_checkpoint, self.init_model is not None),
            (self.texture_scale_field, self.texture_detail.get()),
            (self.start_button, ready),
        ):
            widget.state(("!disabled",) if enabled else ("disabled",))
            marker = self.required_markers.get(widget)
            if marker is not None:
                marker.configure(text="*" if enabled else "")

    def _refresh_targets(self, selection: int | None = None) -> None:
        self.target_list.delete(0, tk.END)
        for target in self.targets:
            self.target_list.insert(tk.END, target.name)
        if selection is not None:
            self.target_list.selection_set(selection)
            self.target_list.see(selection)
        self.target_count.set(f"Target images ({len(self.targets)}/{MAX_DESIGNS})")
        self._refresh_options()

    def add_targets(self) -> None:
        filenames = filedialog.askopenfilenames(
            parent=self.window,
            title="Add target images",
            initialdir=self.image_directory,
            filetypes=IMAGE_TYPES,
        )
        targets = list(dict.fromkeys(Path(filename).resolve() for filename in filenames))
        added = [target for target in targets if target not in self.targets]
        if not added:
            return
        if len(self.targets) + len(added) > MAX_DESIGNS:
            self.status.set(f"Use at most {MAX_DESIGNS} target images.")
            return
        extents = []
        try:
            for target in added:
                with Image.open(target) as image:
                    extents.append(max(image.size))
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            self.status.set(str(error))
            return
        if (
            self.first_selection
            and self.init_model is None
            and max(extents) <= 64
            and self.size.get() == "48"
            and self.neural_detail.get()
            and not self.pixel_art.get()
        ):
            self.size.set(str(min(64, max(28, (max(extents) * 8 + 4) // 5))))
            self.pixel_art.set(True)
            self.neural_detail.set(False)
        self.first_selection = False
        self.targets.extend(added)
        self.image_directory = added[0].parent
        self.status.set("Target order determines the order of designs in the saved model.")
        self._refresh_targets(len(self.targets) - len(added))

    def remove_target(self) -> None:
        (index,) = self.target_list.curselection()
        del self.targets[index]
        self._refresh_targets(min(index, len(self.targets) - 1) if self.targets else None)

    def move_target(self, offset: int) -> None:
        (index,) = self.target_list.curselection()
        destination = index + offset
        self.targets[destination], self.targets[index] = (
            self.targets[index],
            self.targets[destination],
        )
        self._refresh_targets(destination)

    def choose_checkpoint(self) -> None:
        filename = filedialog.askopenfilename(
            parent=self.window,
            title="Choose warm-start model",
            initialdir=self.model_directory,
            filetypes=(("Studio model", "*.pt"),),
        )
        if not filename:
            return
        try:
            checkpoint = Path(filename).resolve()
            _, settings = load_model(checkpoint)
            if settings.mode not in MODES or settings.fire_rate != 0.5:
                raise ValueError("Choose a compatible sprite checkpoint.")
        except (OSError, RuntimeError, ValueError) as error:
            self.status.set(str(error))
            return
        self.init_model = checkpoint
        self.checkpoint_label.set(f"{checkpoint.name} · {', '.join(settings.design_names)}")
        self.size.set(str(settings.grid_size))
        self.perception.set(settings.perception)
        self.neural_detail.set(settings.neural_detail)
        self.pixel_art.set(settings.pixel_art)
        if settings.detail_scale is not None:
            self.detail_scale.set(str(settings.detail_scale))
        self.status.set("Add the checkpoint’s target images in the order shown above.")
        self._refresh_options()

    def reset_checkpoint(self) -> None:
        self.init_model = None
        self.checkpoint_label.set("Train from scratch")
        self._refresh_options()

    def _options(self) -> Namespace:
        values = {}
        for name in (
            "size",
            "iterations",
            "batch_size",
            "seed",
            "min_steps",
            "max_steps",
            "detail_scale",
            "detail_samples",
        ):
            if not self.neural_detail.get() and name in ("detail_scale", "detail_samples"):
                values[name] = 8 if name == "detail_scale" else 4096
                continue
            try:
                values[name] = int(getattr(self, name).get())
            except ValueError as error:
                raise ValueError(
                    f"{name.replace('_', ' ').capitalize()} must be a whole number."
                ) from error
        options = Namespace(
            target=self.targets,
            output=None,
            init_model=self.init_model,
            **values,
            mode=self.mode.get(),
            perception=self.perception.get(),
            device=self.device.get(),
            pixel_art=self.pixel_art.get(),
            neural_detail=self.neural_detail.get(),
            remove_background=self.remove_background.get(),
            live_morph=self.live_morph.get(),
        )
        validate_training_options(options)
        if options.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Choose auto or CPU.")
        initial = load_model(self.init_model)[1] if self.init_model else None
        training_settings(options, initial)
        return options

    def start(self) -> None:
        kind = self.model_type.get()
        try:
            threads = int(self.cpu_threads.get())
            if threads < 0:
                raise ValueError("CPU threads must be 0 (automatic) or a positive integer.")
            if kind == PATTERN:
                iterations = int(self.pattern_iterations.get())
                seed = int(self.pattern_seed.get())
                if iterations < 1:
                    raise ValueError("Training iterations must be a positive integer.")
                options = Namespace(output=None, iterations=iterations, seed=seed)
                trainer, initialfile = PATTERN_TRAINER, "my-living-patterns.pt"
            elif kind in (TEXTURE, ANIMATION):
                options = self._texture_options(animate=kind == ANIMATION)
                label = "animated" if kind == ANIMATION else "texture"
                trainer, initialfile = TEXTURE_TRAINER, f"{options.target.stem}-{label}.pt"
            else:
                options = self._options()
                trainer, initialfile = TRAINER, f"{options.target[0].stem}.pt"
            if not -(2**63) <= options.seed <= 2**64 - 1:
                raise ValueError("Training seed is outside the supported range.")
            options.cpu_threads = threads
            self.model_directory.mkdir(parents=True, exist_ok=True)
        except (OSError, RuntimeError, ValueError) as error:
            self.status.set(str(error))
            return
        filename = filedialog.asksaveasfilename(
            parent=self.window,
            title="Save model",
            initialfile=initialfile,
            initialdir=self.model_directory,
            defaultextension=".pt",
            filetypes=(("Studio model", "*.pt"),),
        )
        if not filename:
            return
        destination = Path(filename)
        if destination.suffix.lower() != ".pt":
            self.status.set("Use a .pt filename for the model.")
            return
        options.output = destination
        try:
            if kind == SPRITE:
                validate_training_options(options)
            if kind in (TEXTURE, ANIMATION) and destination.resolve() in (
                options.target,
                options.motion,
            ):
                raise ValueError(
                    "Choose an output different from the texture and motion references."
                )
            self.job = TrainingJob(trainer, options, destination)
        except (OSError, ValueError) as error:
            self.status.set(str(error))
            return

        self.status.set("Starting training…")
        self.progress.configure(maximum=options.iterations, value=0)
        for widget in self.inputs:
            if isinstance(widget, ttk.Widget):
                widget.state(("disabled",))
            else:
                widget.configure(state="disabled")
        self.close_button.configure(text="Cancel training")
        self.poll_job = self.window.after(500, self.poll)

    def _texture_options(self, *, animate: bool) -> Namespace:
        if self.texture_image is None or (animate and self.motion_reference is None):
            raise ValueError("Choose the required texture and motion references.")
        try:
            size, iterations, steps, batch_size, seed = (
                int(value.get())
                for value in (
                    self.texture_size,
                    self.texture_iterations,
                    self.texture_steps,
                    self.texture_batch,
                    self.texture_seed,
                )
            )
            scale = int(self.texture_scale.get()) if self.texture_detail.get() else 4
        except ValueError as error:
            raise ValueError("Texture settings must use whole numbers.") from error
        if not MIN_GRID_SIZE <= size <= MAX_GRID_SIZE or min(iterations, steps, batch_size) < 1:
            raise ValueError(
                f"Use a {MIN_GRID_SIZE}–{MAX_GRID_SIZE} cell grid and positive iterations, steps and batch size."
            )
        if not 2 <= scale <= MAX_DETAIL_SCALE:
            raise ValueError(f"Detail pixels per cell must be between 2 and {MAX_DETAIL_SCALE}.")
        device = self.texture_device.get()
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Choose auto or CPU.")
        return Namespace(
            target=self.texture_image,
            motion=self.motion_reference if animate else None,
            output=None,
            size=size,
            iterations=iterations,
            steps=steps,
            batch_size=batch_size,
            seed=seed,
            device=device,
            neural_detail=self.texture_detail.get(),
            detail_scale=scale,
        )

    def poll(self) -> None:
        self.poll_job = None
        job = self.job
        if job is None:
            return
        update = job.poll()
        if update.message:
            self.status.set(update.message)
        if update.iteration is not None:
            self.progress.configure(value=update.iteration)
        if not update.finished:
            self.poll_job = self.window.after(500, self.poll)
            return

        self.job = None
        if update.error:
            for widget in self.inputs:
                if isinstance(widget, ttk.Widget):
                    widget.state(("!disabled",))
                else:
                    widget.configure(state="normal")
            self._refresh_options()
            self.close_button.configure(text="Close")
            return
        self.window.destroy()
        self.on_complete(job.destination)

    def close(self) -> bool:
        if self.job is not None:
            if self.poll_job is not None:
                self.window.after_cancel(self.poll_job)
                self.poll_job = None
            if not messagebox.askyesno(
                "Cancel training", "Stop this training run?", parent=self.window
            ):
                self.poll_job = self.window.after(0, self.poll)
                return False
            try:
                self.job.cancel()
            except OSError as error:
                self.status.set(str(error))
                self.poll_job = self.window.after(0, self.poll)
                return False
            self.job = None
        self.window.destroy()
        return True
