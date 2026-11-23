"""Shared neural cellular automaton, grid operations, and checkpoint format."""

import os
import tempfile
from dataclasses import asdict, dataclass
from math import cos, radians, sin, tau
from pathlib import Path
from pickle import UnpicklingError

import torch
from torch import Tensor, nn
from torch.nn import functional as F

CHANNELS = 16
MODES = ("growing", "persistent", "regenerating")
PATTERN_MODE = "reaction-diffusion"
TEXTURE_MODE = "texture"
ANIMATED_TEXTURE_MODE = "animated-texture"
TEXTURE_MODES = (TEXTURE_MODE, ANIMATED_TEXTURE_MODE)
MAX_MOTION_FRAMES = 32
PERCEPTIONS = ("sobel", "isotropic")
MIN_GRID_SIZE = 16
MAX_GRID_SIZE = 96
MAX_DESIGNS = 32
CHECKPOINT_VERSION = 6
MAX_DETAIL_SCALE = 16
MAX_EXPORT_SIZE = 2048


@dataclass(frozen=True)
class ModelSettings:
    grid_size: int
    mode: str
    target_name: str
    fire_rate: float = 0.5
    perception: str = "sobel"
    designs: tuple[str, ...] = ()
    neural_detail: bool = False
    detail_scale: int | None = None
    live_morph: bool = False
    pixel_art: bool = False
    texture_alpha: bool = False
    motion_frames: int = 0

    def __post_init__(self) -> None:
        if type(self.grid_size) is not int or not MIN_GRID_SIZE <= self.grid_size <= MAX_GRID_SIZE:
            raise ValueError(f"Grid size must be between {MIN_GRID_SIZE} and {MAX_GRID_SIZE}.")
        if self.mode not in (*MODES, PATTERN_MODE, *TEXTURE_MODES):
            raise ValueError("Unknown training mode.")
        if not isinstance(self.target_name, str) or not self.target_name.strip():
            raise ValueError("Missing target name.")
        if not isinstance(self.fire_rate, (int, float)) or not 0 < self.fire_rate <= 1:
            raise ValueError("Cell update probability must be in (0, 1].")
        if self.perception not in PERCEPTIONS:
            raise ValueError("Unknown perception rule.")
        if (
            not isinstance(self.designs, tuple)
            or len(self.designs) > MAX_DESIGNS
            or any(not isinstance(name, str) or not name.strip() for name in self.designs)
        ):
            raise ValueError("Invalid design names.")
        if len(set(self.designs)) != len(self.designs) or type(self.neural_detail) is not bool:
            raise ValueError("Invalid model options.")
        if self.detail_scale is not None and (
            not self.neural_detail
            or type(self.detail_scale) is not int
            or not 2 <= self.detail_scale <= MAX_DETAIL_SCALE
        ):
            raise ValueError("Invalid detail-training scale.")
        if type(self.live_morph) is not bool or (
            self.live_morph
            and (len(self.designs) < 2 or self.mode not in ("persistent", "regenerating"))
        ):
            raise ValueError(
                "Live morphing requires multiple designs and a persistent or regenerating model."
            )
        if type(self.pixel_art) is not bool:
            raise ValueError("Invalid pixel art setting.")
        if self.mode == PATTERN_MODE and (
            self.perception != "isotropic"
            or self.neural_detail
            or self.designs
            or self.fire_rate != 1
            or self.pixel_art
        ):
            raise ValueError(
                "Reaction–diffusion requires isotropic, synchronous updates on a single field."
            )
        if type(self.texture_alpha) is not bool or (
            self.texture_alpha and self.mode not in TEXTURE_MODES
        ):
            raise ValueError("Invalid texture transparency setting.")
        if (
            type(self.motion_frames) is not int
            or not 0 <= self.motion_frames <= MAX_MOTION_FRAMES
            or (bool(self.motion_frames) != (self.mode == ANIMATED_TEXTURE_MODE))
        ):
            raise ValueError("Invalid motion frame count.")
        if self.mode in TEXTURE_MODES and (self.designs or self.perception != "sobel"):
            raise ValueError("Texture models use one target and directional perception.")

    @property
    def design_names(self) -> tuple[str, ...]:
        if self.designs:
            return self.designs
        return (Path(self.target_name).stem.replace("-", " ").replace("_", " ").title(),)

    def make_model(self) -> "NeuralCA | PatternCA":
        if self.mode == PATTERN_MODE:
            return PatternCA()
        if self.mode in TEXTURE_MODES:
            return TextureCA(self)
        return NeuralCA(
            self.fire_rate,
            self.perception,
            len(self.designs),
            self.neural_detail,
            periodic_detail=self.detail_scale is not None,
        )


class NeuralCA(nn.Module):
    def __init__(
        self,
        fire_rate: float = 0.5,
        perception: str = "sobel",
        designs: int = 0,
        neural_detail: bool = False,
        *,
        periodic_detail: bool = False,
    ) -> None:
        super().__init__()
        self.fire_rate = fire_rate
        self.isotropic = perception == "isotropic"
        self.design_count = designs if designs > 1 else 0
        identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8
        laplacian = torch.tensor([[1, 2, 1], [2, -12, 2], [1, 2, 1]], dtype=torch.float32) / 12
        kernels = (identity, laplacian) if self.isotropic else (identity, sobel_x, sobel_x.T)
        filters = torch.stack(kernels).unsqueeze(1).repeat(CHANNELS, 1, 1, 1)
        self.register_buffer("perception", filters, persistent=False)
        self.register_buffer("design_codes", torch.eye(self.design_count), persistent=False)
        self.update = nn.Sequential(
            nn.Conv2d(len(kernels) * CHANNELS + self.design_count, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, CHANNELS, kernel_size=1),
        )
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)
        self.decoder = DetailDecoder(periodic_detail) if neural_detail else None

    @staticmethod
    def living_cells(state: Tensor) -> Tensor:
        return F.max_pool2d(state[:, 3:4], kernel_size=3, stride=1, padding=1) > 0.1

    def forward(
        self,
        state: Tensor,
        design: int = 0,
        turns: int = 0,
        synchronous: bool = False,
        *,
        angle: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        delta = self.cell_delta(
            state,
            angle=(turns * 90 + angle) % 360,
            design=design,
            synchronous=synchronous,
            generator=generator,
        )
        updated = state + delta
        return updated * (self.living_cells(state) & self.living_cells(updated))

    def cell_delta(
        self,
        state: Tensor,
        *,
        angle: float,
        synchronous: bool,
        generator: torch.Generator | None,
        design: int = 0,
        wrap: bool = False,
    ) -> Tensor:
        filters = self.perception
        direction = angle
        if direction and not self.isotropic:
            if direction % 90 == 0:
                filters = filters.rot90(-int(direction // 90), (-2, -1)).contiguous()
            else:
                theta = radians(direction)
                c, s = cos(theta), sin(theta)
                identity, dx, dy = filters.reshape(CHANNELS, 3, 1, 3, 3).unbind(dim=1)
                filters = torch.stack((identity, c * dx + s * dy, -s * dx + c * dy), dim=1)
                filters = filters.flatten(0, 1)
        neighbors = F.pad(state, (1, 1, 1, 1), mode="circular") if wrap else state
        local_state = F.conv2d(neighbors, filters, padding=0 if wrap else 1, groups=CHANNELS)
        if self.design_count:
            code = self.design_codes[design, :, None, None].expand(
                state.shape[0], -1, *state.shape[-2:]
            )
            local_state = torch.cat((local_state, code), dim=1)
        delta = self.update(local_state)
        if not synchronous and self.fire_rate < 1:
            fires = torch.rand(
                state[:, :1].shape, device=state.device, dtype=state.dtype, generator=generator
            )
            delta = delta * (fires < self.fire_rate)
        return delta


class TextureCA(NeuralCA):
    """Synthesize a texture on a periodic grid, optionally transported by reference motion."""

    def __init__(self, settings: ModelSettings) -> None:
        super().__init__(settings.fire_rate)
        self.has_alpha = settings.texture_alpha
        self.decoder = (
            DetailDecoder(periodic=True, tileable=True, opaque=not self.has_alpha)
            if settings.neural_detail
            else None
        )
        self.size = settings.grid_size
        self.register_buffer(
            "motion",
            torch.zeros(settings.motion_frames, 2, self.size, self.size),
            persistent=bool(settings.motion_frames),
        )
        axis = torch.arange(self.size, dtype=torch.float32) + 0.5
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("centers", torch.stack((x, y), dim=-1), persistent=False)

    def initial_field(self, generator: torch.Generator, batch: int = 1) -> Tensor:
        state = torch.rand(
            (batch, CHANNELS, self.size, self.size), device=self.centers.device, generator=generator
        )
        state[:, :3] *= state[:, 3:4] if self.has_alpha else 1
        if not self.has_alpha:
            state[:, 3] = 1
        state[:, 4:] -= 0.5
        return state

    def forward(
        self,
        state: Tensor,
        design: int = 0,
        turns: int = 0,
        synchronous: bool = False,
        *,
        angle: float = 0.0,
        generator: torch.Generator | None = None,
        step: int = 0,
    ) -> Tensor:
        if design != 0:
            raise ValueError("Texture models contain one design.")
        direction = (angle + turns * 90) % 360
        if self.motion.numel():
            dx, dy = self.motion[step % len(self.motion)].unbind()
            c, s = cos(radians(direction)), sin(radians(direction))
            flow = torch.stack((c * dx - s * dy, s * dx + c * dy), dim=-1)
            coords = (self.centers - flow).remainder(self.size)
            coords = (coords + 1) * (2 / (self.size + 2)) - 1
            state = F.grid_sample(
                F.pad(state, (1, 1, 1, 1), mode="circular"),
                coords.expand(state.shape[0], -1, -1, -1),
                align_corners=False,
            )
        delta = self.cell_delta(
            state, angle=direction, synchronous=synchronous, generator=generator, wrap=True
        )
        updated = state + delta
        alpha = updated[:, 3:4].clamp(0, 1) if self.has_alpha else torch.ones_like(updated[:, 3:4])
        rgba = torch.cat((updated[:, :3].clamp_min(0).minimum(alpha), alpha), dim=1)
        return torch.cat((rgba, updated[:, 4:].clamp(-3, 3)), dim=1)


class PatternCA(nn.Module):
    """Learn a two-channel reaction–diffusion rule on a periodic grid."""

    decoder = None
    isotropic = True

    def __init__(self) -> None:
        super().__init__()
        laplacian = torch.tensor([[1, 4, 1], [4, -20, 4], [1, 4, 1]], dtype=torch.float32) / 20
        self.register_buffer(
            "laplacian", laplacian.expand(2, 1, 3, 3).contiguous(), persistent=False
        )
        self.rates = nn.Parameter(torch.zeros(4))
        self.reaction = nn.Sequential(nn.Conv2d(2, 16, 1), nn.SiLU(), nn.Conv2d(16, 1, 1))

    def perceive(self, field: Tensor) -> Tensor:
        neighbors = F.pad(field, (1, 1, 1, 1), mode="circular")
        diffusion = F.conv2d(neighbors, self.laplacian, groups=2)
        return torch.cat((field, diffusion), dim=1)

    def predict_delta(self, local_state: Tensor) -> Tensor:
        u, v, lap_u, lap_v = local_state.unbind(dim=1)
        diffuse_u, diffuse_v, feed, removal = self.rates * 0.1
        # The two concentrations exchange mass through a shared learned reaction rate.
        # Its mass-action factors preserve the quiescent state exactly.
        reaction = u * v.square() * self.reaction(local_state[:, :2]).squeeze(1)
        return torch.stack(
            (
                diffuse_u * lap_u - reaction + feed * (1 - u),
                diffuse_v * lap_v + reaction - removal * v,
            ),
            dim=1,
        )

    def forward(self, field: Tensor) -> Tensor:
        return (field + self.predict_delta(self.perceive(field))).clamp(0, 1)

    @staticmethod
    def initial_field(size: int, generator: torch.Generator, batch: int = 1) -> Tensor:
        """Create seeded patches with small random variations on a quiescent field."""
        noise = torch.rand((batch, 2, size, size), generator=generator)
        field = torch.zeros_like(noise)
        field[:, 0] = 1
        axis = torch.arange(size)
        for index in range(batch):
            centers = torch.randint(size // 4, 3 * size // 4, (4, 2), generator=generator)
            for x, y in centers:
                mask = (axis - x).square() + (axis[:, None] - y).square() <= (size // 12) ** 2
                field[index, 0, mask] = 0.48 + 0.12 * noise[index, 0, mask]
                field[index, 1, mask] = 0.20 + 0.10 * noise[index, 1, mask]
        return field

    @staticmethod
    def colorize(field: Tensor) -> Tensor:
        strength = (field[:, 1:2] * 2.4).clamp(0, 1)
        rgb = torch.cat(
            (
                0.08 + 0.92 * strength.square(),
                0.13 + 0.73 * strength,
                0.20 + 0.42 * (2 * strength).clamp_max(1),
            ),
            dim=1,
        )
        return torch.cat((rgb, torch.ones_like(strength)), dim=1)


class DetailDecoder(nn.Module):
    """Query premultiplied RGBA at continuous positions on the cell grid."""

    def __init__(
        self, periodic: bool = False, *, tileable: bool = False, opaque: bool = False
    ) -> None:
        super().__init__()
        self.periodic = periodic
        self.tileable = tileable
        self.opaque = opaque
        self.network = nn.Sequential(
            nn.Conv2d(CHANNELS + (4 if periodic else 2), 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 4, kernel_size=1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def sample(self, state: Tensor, positions: Tensor, living: Tensor | None = None) -> Tensor:
        """Sample [x, y] coordinates in [0, 1], returning [batch, 4, points]."""
        grid = (positions * 2 - 1)[None, None].expand(state.shape[0], -1, -1, -1)
        extent = positions.new_tensor((state.shape[-1], state.shape[-2]))
        cells = positions * extent
        if self.tileable:
            grid = ((cells.remainder(extent) + 1) * 2 / (extent + 2) - 1)[None, None].expand_as(
                grid
            )
            features = F.grid_sample(
                F.pad(state, (1, 1, 1, 1), mode="circular"), grid, align_corners=False
            )
        else:
            features = F.grid_sample(state, grid, padding_mode="border", align_corners=False)
        phase = cells.remainder(1) - 0.5
        coordinates = (
            torch.cat(((phase * tau).sin(), (phase * tau).cos()), dim=-1)
            if self.periodic
            else phase
        )
        coordinates = coordinates.T[None, :, None].expand(state.shape[0], -1, -1, -1)
        residual = self.network(torch.cat((features, coordinates), dim=1))
        if self.tileable:
            rgba = (features[:, :4] + residual).squeeze(2)
            return (
                torch.cat((rgba[:, :3], torch.ones_like(rgba[:, 3:4])), dim=1)
                if self.opaque
                else rgba
            )
        if living is None:
            living = NeuralCA.living_cells(state).to(state.dtype)
        local = F.grid_sample(
            living,
            grid,
            mode="bilinear" if self.periodic else "nearest",
            padding_mode="border",
            align_corners=False,
        )
        return (features[:, :4] + residual * local).squeeze(2)

    def render(self, state: Tensor, width: int, height: int) -> Tensor:
        living = None if self.tileable else NeuralCA.living_cells(state).to(state.dtype)
        chunks = []
        for start in range(0, width * height, 16_384):
            indices = torch.arange(start, min(start + 16_384, width * height), device=state.device)
            positions = pixel_positions(indices, width, height, state.dtype)
            chunks.append(self.sample(state, positions, living))
        return torch.cat(chunks, dim=-1).reshape(state.shape[0], 4, height, width)

    def forward(self, state: Tensor, scale: int = 2) -> Tensor:
        return self.render(state, state.shape[-1] * scale, state.shape[-2] * scale)


def pixel_positions(indices: Tensor, width: int, height: int, dtype: torch.dtype) -> Tensor:
    x = (indices.remainder(width).to(dtype) + 0.5) / width
    y = (indices.div(width, rounding_mode="floor").to(dtype) + 0.5) / height
    return torch.stack((x, y), dim=-1)


def place_seed(
    state: Tensor, x: int, y: int, structured: bool = False, turns: int = 0, *, angle: float = 0.0
) -> None:
    points = ((0, 0), (2, 0), (0, -2)) if structured else ((0, 0),)
    theta = radians(turns * 90 + angle)
    c, s = cos(theta), sin(theta)
    for marker, (dx, dy) in enumerate(points):
        px, py = x + round(c * dx - s * dy), y + round(s * dx + c * dy)
        if 0 <= px < state.shape[-1] and 0 <= py < state.shape[-2]:
            state[:, :3, py, px] = 0
            state[:, 3:, py, px] = 1
            if structured:
                state[:, 4:7, py, px] = 0
                state[:, 4 + marker, py, px] = 1


def seed_grid(
    size: int,
    device: torch.device | str = "cpu",
    *,
    structured: bool = False,
    turns: int = 0,
    angle: float = 0.0,
) -> Tensor:
    state = torch.zeros((1, CHANNELS, size, size), device=device)
    place_seed(state, size // 2, size // 2, structured, angle=angle)
    return state.rot90(-turns, (-2, -1)).contiguous() if turns else state


def save_model(path: Path, model: NeuralCA | PatternCA, settings: ModelSettings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "settings": asdict(settings),
        "weights": {name: weight.detach().cpu() for name, weight in model.state_dict().items()},
    }
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_model(path: Path) -> tuple[NeuralCA | PatternCA, ModelSettings]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (EOFError, IndexError, KeyError, RuntimeError, TypeError, UnpicklingError) as error:
        raise ValueError("The file is not a readable Cellular Studio checkpoint.") from error
    if not isinstance(checkpoint, dict) or checkpoint.get("version") not in (
        1,
        2,
        3,
        4,
        5,
        CHECKPOINT_VERSION,
    ):
        raise ValueError("Unsupported model file. Open a checkpoint created by train.py.")
    try:
        settings = ModelSettings(**checkpoint["settings"])
        weights = checkpoint["weights"]
    except (KeyError, TypeError) as error:
        raise ValueError("The checkpoint is missing model settings or weights.") from error
    model = settings.make_model()
    expected = model.state_dict()
    if (
        not isinstance(weights, dict)
        or weights.keys() != expected.keys()
        or not all(
            isinstance(weight, Tensor)
            and weight.layout == torch.strided
            and weight.shape == expected[name].shape
            and weight.dtype == expected[name].dtype
            and torch.isfinite(weight).all()
            for name, weight in weights.items()
        )
    ):
        raise ValueError("The checkpoint contains invalid model weights.")
    model.load_state_dict(weights)
    return model.eval(), settings
