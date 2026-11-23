"""Train a neural cellular automaton to grow one or more target images."""

import argparse
import random
import sys
from math import isfinite
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torch import Tensor
from torch.nn import functional as F

from compute import positive_int, thread_count, training_device, training_layout
from nca import (
    MAX_DESIGNS,
    MAX_DETAIL_SCALE,
    MAX_GRID_SIZE,
    MIN_GRID_SIZE,
    MODES,
    PERCEPTIONS,
    ModelSettings,
    load_model,
    pixel_positions,
    save_model,
    seed_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        action="append",
        required=True,
        help="Target image. Repeat to train multiple designs.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination checkpoint (.pt)")
    parser.add_argument("--mode", choices=MODES, default="growing")
    parser.add_argument(
        "--iterations", type=positive_int, default=2000, help="Optimizer updates (default: 2000)"
    )
    parser.add_argument(
        "--size", type=positive_int, default=64, help="Square grid width in cells (default: 64)"
    )
    parser.add_argument(
        "--batch-size", type=positive_int, default=4, help="Grids per batch (default: 4)"
    )
    parser.add_argument(
        "--min-steps", type=positive_int, default=64, help="Shortest rollout (default: 64)"
    )
    parser.add_argument(
        "--max-steps", type=positive_int, default=96, help="Longest rollout (default: 96)"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--cpu-threads", type=thread_count, default=0, help="CPU workers (0 chooses automatically)"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument(
        "--perception",
        choices=PERCEPTIONS,
        default="sobel",
        help="Use directional Sobel filters or isotropic Laplacian perception with a structured seed",
    )
    parser.add_argument(
        "--neural-detail", action="store_true", help="Train a continuous high-resolution decoder"
    )
    parser.add_argument(
        "--pixel-art",
        action="store_true",
        help="Use nearest-neighbor target sampling to preserve hard pixel edges",
    )
    parser.add_argument(
        "--detail-scale", type=positive_int, default=8, help="Target pixels per cell (default: 8)"
    )
    parser.add_argument(
        "--detail-samples",
        type=positive_int,
        default=4096,
        help="High-resolution target pixels sampled per training iteration (default: 4096)",
    )
    parser.add_argument(
        "--remove-background",
        action="store_true",
        help="Remove a flat background that matches the image corners",
    )
    parser.add_argument(
        "--live-morph",
        action="store_true",
        help="Train transitions between grown designs by changing the goal in place",
    )
    parser.add_argument("--init-model", type=Path, help="Warm-start from a compatible checkpoint")
    args = parser.parse_args()
    try:
        validate_training_options(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def validate_training_options(args: argparse.Namespace) -> None:
    for name in ("iterations", "batch_size", "min_steps", "max_steps", "detail_samples"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be a positive integer")
    if not MIN_GRID_SIZE <= args.size <= MAX_GRID_SIZE:
        raise ValueError(f"--size must be between {MIN_GRID_SIZE} and {MAX_GRID_SIZE}")
    if not 2 <= args.detail_scale <= MAX_DETAIL_SCALE:
        raise ValueError(f"--detail-scale must be between 2 and {MAX_DETAIL_SCALE}")
    if args.max_steps < args.min_steps:
        raise ValueError("--max-steps must be at least --min-steps")
    if args.mode != "growing" and args.batch_size < 2:
        raise ValueError("Persistent and regenerating modes require a batch size of at least 2.")
    if args.live_morph and (args.mode == "growing" or len(args.target) < 2 or args.batch_size < 3):
        raise ValueError(
            "Live morphing requires persistent or regenerating mode, at least two targets, and a batch size of at least 3."
        )
    if not args.target:
        raise ValueError("Add at least one target image.")
    if len(args.target) > MAX_DESIGNS:
        raise ValueError(f"Use at most {MAX_DESIGNS} target images.")
    if args.iterations < len(args.target):
        raise ValueError("Training iterations must be at least the number of targets.")
    if len({design_name(path).casefold() for path in args.target}) != len(args.target):
        raise ValueError("Targets must have distinct filenames.")
    if args.output is not None:
        output = args.output.resolve()
        if any(path.resolve() == output for path in args.target):
            raise ValueError("Choose an output different from the target images.")
        if args.init_model and args.init_model.resolve() == output:
            raise ValueError("Choose an output different from the warm-start checkpoint.")


def training_settings(args: argparse.Namespace, initial: ModelSettings | None) -> ModelSettings:
    designs = tuple(map(design_name, args.target))
    if initial and (
        initial.mode not in MODES
        or args.size != initial.grid_size
        or args.perception != initial.perception
        or designs != initial.design_names
        or args.neural_detail != initial.neural_detail
        or initial.fire_rate != 0.5
    ):
        raise ValueError(
            "Warm-start requires a sprite checkpoint with the same grid size, perception, "
            "ordered designs, and decoder configuration."
        )
    return ModelSettings(
        args.size,
        args.mode,
        args.target[0].name,
        perception=args.perception,
        designs=designs,
        neural_detail=args.neural_detail,
        live_morph=args.live_morph,
        pixel_art=args.pixel_art,
        detail_scale=args.detail_scale
        if args.neural_detail and (initial is None or initial.detail_scale is not None)
        else None,
    )


def design_name(path: Path) -> str:
    return path.stem.replace("-", " ").replace("_", " ").strip().title()


def remove_flat_background(sprite: Image.Image) -> Image.Image:
    if sprite.getchannel("A").getextrema()[0] < 255:
        return sprite
    rgba = np.array(sprite)
    rgb = rgba[:, :, :3].astype(np.int16)
    tolerance = 24
    patch = max(1, min(rgb.shape[:2]) // 32)
    corners = np.array(
        [
            np.median(rgb[rows, cols], axis=(0, 1))
            for rows in (slice(0, patch), slice(-patch, None))
            for cols in (slice(0, patch), slice(-patch, None))
        ]
    )
    background = np.median(corners, axis=0)
    if np.abs(corners - background).max() > tolerance:
        raise ValueError(
            "The image corners have different colors. Use a PNG with a transparent background."
        )
    matches = np.abs(rgb - background).max(axis=2) <= tolerance
    # A flat one-pixel frame can surround the background in exported artwork.
    for image, mask in ((rgb, matches), (rgb.swapaxes(0, 1), matches.T)):
        for edge, inside in ((0, 1), (-1, -2)):
            if (
                len(image) > 1
                and mask[inside].mean() >= 0.98
                and np.ptp(image[edge], axis=0).max() <= tolerance
            ):
                mask[edge] = True
    # The padding connects all of the border pixels to a single flood-fill origin.
    width, height = sprite.size
    mask = Image.new("L", (width + 2, height + 2))
    mask.paste(Image.fromarray(np.where(matches, 0, 255).astype(np.uint8)), (1, 1))
    ImageDraw.floodfill(mask, (0, 0), 128)
    rgba[np.array(mask)[1:-1, 1:-1] == 128, 3] = 0
    return Image.fromarray(rgba)


def load_target(
    path: Path,
    size: int,
    device: torch.device,
    scale: int = 1,
    *,
    remove_background: bool = False,
    pixel_art: bool = False,
) -> Tensor:
    with Image.open(path) as source:
        sprite = ImageOps.exif_transpose(source).convert("RGBA")
    if remove_background:
        sprite = remove_flat_background(sprite)
    bounds = sprite.getchannel("A").getbbox()
    if bounds is None:
        raise ValueError("The target image is fully transparent.")
    sprite = sprite.crop(bounds)
    # Leave space around the sprite for growth, damage, and stray cells.
    resampling = Image.Resampling.NEAREST if pixel_art else Image.Resampling.LANCZOS
    coarse = sprite.copy()
    coarse.thumbnail((size * 5 // 8, size * 5 // 8), resampling)
    x, y = (size - coarse.width + 1) // 2, (size - coarse.height + 1) // 2
    sprite = (
        coarse
        if scale == 1
        else sprite.resize((coarse.width * scale, coarse.height * scale), resampling)
    )
    canvas = Image.new("RGBA", (size * scale, size * scale))
    canvas.paste(sprite, (x * scale, y * scale))
    rgba = torch.from_numpy(np.array(canvas)).to(device=device, dtype=torch.float32) / 255
    rgba[:, :, :3] *= rgba[:, :, 3:4]
    return rgba.permute(2, 0, 1).unsqueeze(0)


def image_loss(states: Tensor, target: Tensor) -> Tensor:
    return (states[:, :4] - target).square().mean(dim=(1, 2, 3))


class StatePool:
    """Recycle completed states so the model learns longer-term behavior."""

    def __init__(self, seed: Tensor, batch_size: int, damage: bool, rng: random.Random) -> None:
        self.seed = seed
        self.batch_size = batch_size
        self.states = seed.repeat(max(128, batch_size), 1, 1, 1)
        self.damage_count = min(3, batch_size - 1) if damage else 0
        self.rng = rng
        self.size = seed.shape[-1]
        self.axis = torch.arange(self.size, device=seed.device)

    def sample(self, target: Tensor) -> tuple[Tensor, Tensor]:
        indices = torch.tensor(
            self.rng.sample(range(len(self.states)), self.batch_size), device=self.states.device
        )
        indices = indices[image_loss(self.states[indices], target).argsort(descending=True)]
        batch = self.states[indices]
        batch[0] = self.seed[0]
        for offset in range(1, self.damage_count + 1):
            center_x = self.rng.randint(self.size // 4, self.size * 3 // 4)
            center_y = self.rng.randint(self.size // 4, self.size * 3 // 4)
            radius = self.rng.randint(max(1, self.size // 16), self.size // 6)
            distance = (self.axis - center_x).square() + (self.axis[:, None] - center_y).square()
            batch[-offset].masked_fill_(distance <= radius * radius, 0)
        return indices, batch

    def store(self, indices: Tensor, states: Tensor) -> None:
        self.states[indices] = states.detach()

    def established(self, target: Tensor) -> Tensor:
        """Choose a good rollout from a random sample, avoiding the empty seed when possible."""
        sample = self.rng.sample(range(len(self.states)), 8)
        candidates = self.states[sample]
        return candidates[image_loss(candidates, target).argmin()]


def train(args: argparse.Namespace) -> None:
    device = training_device(args.device)
    memory_format, device_name = training_layout(
        device, args.size, args.batch_size, getattr(args, "cpu_threads", 0)
    )
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model, initial = load_model(args.init_model) if args.init_model else (None, None)
    settings = training_settings(args, initial)
    target_options = {"remove_background": args.remove_background, "pixel_art": args.pixel_art}
    targets = [load_target(path, args.size, device, **target_options) for path in args.target]
    model = (
        (model if model is not None else settings.make_model())
        .to(device, memory_format=memory_format)
        .train()
    )
    # Keep the high-resolution rasters in host memory and transfer only sampled pixels.
    detail_targets = (
        [
            load_target(
                path, args.size, torch.device("cpu"), scale=args.detail_scale, **target_options
            )
            for path in args.target
        ]
        if model.decoder
        else []
    )
    # Sample around the silhouette as well as the entire canvas, where most
    # pixels can be transparent. This gives thin features and edges supervision.
    detail_regions = [
        F.max_pool2d((target[:, 3:4] > 0.1).float(), kernel_size=5, stride=1, padding=2)
        .flatten()
        .nonzero()
        .flatten()
        for target in detail_targets
    ]
    seed = seed_grid(args.size, device, structured=model.isotropic)
    pools = []
    if args.mode != "growing":
        pools = [
            StatePool(seed, args.batch_size, args.mode == "regenerating", rng) for _ in targets
        ]
    if initial and args.live_morph:
        with torch.no_grad():
            for design, pool in enumerate(pools):
                state = pool.states[:32].contiguous(memory_format=memory_format)
                for _ in range(2 * args.max_steps):
                    state = model(state, design)
                pool.states[:32] = state

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    schedule = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [max(1, args.iterations * 3 // 4)], gamma=0.2
    )
    names = ", ".join(settings.design_names)
    transition_start = 1 if initial else max(len(targets), args.iterations // 4)
    print(
        f"Training {args.mode} ({args.perception}) on {device_name} | {args.size}x{args.size} cells | {names}",
        flush=True,
    )
    start = perf_counter()
    log_every = max(1, min(100, args.iterations // 10))
    for iteration in range(1, args.iterations + 1):
        design = (iteration - 1) % len(targets)
        target = targets[design]
        pool = pools[design] if pools else None
        if pool is None:
            state = seed.expand(args.batch_size, -1, -1, -1)
        else:
            indices, state = pool.sample(target)
            # GoalNCA (Sudhakaran et al., 2022): supervise a different goal
            # starting from a mature rollout, while retaining seed and repair examples.
            if args.live_morph and iteration >= transition_start and rng.random() < 0.75:
                source = (design + rng.randrange(1, len(targets))) % len(targets)
                state[1] = pools[source].established(targets[source])

        state = state.contiguous(memory_format=memory_format)
        optimizer.zero_grad(set_to_none=True)
        for _ in range(rng.randint(args.min_steps, args.max_steps)):
            state = model(state, design)
        loss = image_loss(state, target).mean()
        if model.decoder is not None:
            detail_target = detail_targets[design]
            height, width = detail_target.shape[-2:]
            count = min(args.detail_samples, width * height)
            region = detail_regions[design]
            focused = count // 2 if region.numel() else 0
            pixels = torch.randint(width * height, (count - focused,))
            if focused:
                pixels = torch.cat((pixels, region[torch.randint(region.numel(), (focused,))]))
            positions = pixel_positions(pixels, width, height, state.dtype).to(device)
            expected = detail_target.flatten(2)[:, :, pixels].to(device)
            residual = model.decoder.sample(state, positions) - expected
            silhouette = (state[:, 3:4] - target[:, 3:4]).abs().mean()
            loss = (
                loss + 0.25 * silhouette + 0.5 * (residual.abs().mean() + residual.square().mean())
            )
        loss_value = loss.item()
        if not isfinite(loss_value):
            raise ValueError(f"Training diverged at iteration {iteration}.")
        loss.backward()
        for parameter in model.parameters():
            gradient = parameter.grad
            gradient.div_(gradient.norm().clamp_min(1e-8))
        optimizer.step()
        schedule.step()
        if pool is not None:
            pool.store(indices, state)
        if iteration == 1 or iteration % log_every == 0 or iteration == args.iterations:
            print(
                f"{iteration:>{len(str(args.iterations))}}/{args.iterations}  {settings.design_names[design]}  loss {loss_value:.6f}",
                flush=True,
            )

    save_model(args.output, model, settings)
    print(f"Saved {args.output} ({perf_counter() - start:.1f}s)")


def main() -> int:
    try:
        train(parse_args())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Training error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Training cancelled.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
