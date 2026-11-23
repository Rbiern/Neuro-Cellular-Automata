"""Train image textures with optional motion from an animation or a flow field."""

import argparse
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from compute import positive_int, thread_count, training_device, training_layout
from nca import (
    ANIMATED_TEXTURE_MODE,
    MAX_DETAIL_SCALE,
    MAX_GRID_SIZE,
    MIN_GRID_SIZE,
    TEXTURE_MODE,
    DetailDecoder,
    ModelSettings,
    TextureCA,
    pixel_positions,
    save_model,
)
from texture_inputs import image_tensor, load_motion

PATCH_OFFSETS = ((0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (0, 4), (4, 0))


def texture_samples(image: Tensor, positions: Tensor) -> Tensor:
    """Circular local color patches at several neighborhood radii."""
    height, width = image.shape[-2:]
    y, x = positions.div(width, rounding_mode="floor"), positions.remainder(width)
    return torch.cat(
        [image[:, :, (y + dy) % height, (x + dx) % width] for dy, dx in PATCH_OFFSETS], dim=1
    )


class TextureLoss:
    """Match distributions of linear projections of local premultiplied color patches."""

    def __init__(self, target: Tensor, generator: torch.Generator) -> None:
        self.generator = generator
        self.target = target
        self.projection = F.normalize(
            torch.randn(4 * len(PATCH_OFFSETS), 48, device=target.device, generator=generator),
            dim=0,
        )
        self.reference = self.distribution(target[None])

    def distribution(self, image: Tensor) -> Tensor:
        positions = torch.randint(
            image.shape[-2] * image.shape[-1], (512,), device=image.device, generator=self.generator
        )
        samples = texture_samples(image, positions)
        return self.project(samples)

    def project(self, patches: Tensor) -> Tensor:
        return (patches.transpose(1, 2) @ self.projection).sort(dim=1).values

    def __call__(self, image: Tensor) -> Tensor:
        return (self.distribution(image) - self.reference).abs().mean()

    def decoded(self, decoder: DetailDecoder, state: Tensor) -> Tensor:
        height, width = self.target.shape[-2:]
        indices = torch.randint(
            width * height, (512,), device=state.device, generator=self.generator
        )
        y, x = indices.div(width, rounding_mode="floor"), indices.remainder(width)
        patches = torch.cat(
            [((y + dy) % height) * width + (x + dx) % width for dy, dx in PATCH_OFFSETS]
        )
        rgba = decoder.sample(state, pixel_positions(patches, width, height, state.dtype))
        samples = (
            rgba.reshape(state.shape[0], 4, len(PATCH_OFFSETS), -1).transpose(1, 2).flatten(1, 2)
        )
        return (self.project(samples) - self.reference).abs().mean()


def train_texture(args: argparse.Namespace) -> None:
    device = training_device(args.device)
    memory_format, device_name = training_layout(
        device, args.size, args.batch_size, getattr(args, "cpu_threads", 0)
    )
    torch.manual_seed(args.seed)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    with Image.open(args.target) as image:
        coarse = image_tensor(image, args.size).to(device)
        detail = (
            image_tensor(image, args.size * args.detail_scale).to(device)
            if args.neural_detail
            else None
        )
    if not (coarse[3] > 0).any() and (detail is None or not (detail[3] > 0).any()):
        raise ValueError("The texture target is fully transparent in the training crop.")
    motion = load_motion(args.motion, args.size) if args.motion else None
    settings = ModelSettings(
        args.size,
        ANIMATED_TEXTURE_MODE if motion is not None else TEXTURE_MODE,
        args.target.name,
        neural_detail=args.neural_detail,
        detail_scale=args.detail_scale if args.neural_detail else None,
        texture_alpha=bool((coarse[3] < 1).any())
        or (detail is not None and bool((detail[3] < 1).any())),
        motion_frames=len(motion) if motion is not None else 0,
    )
    model = TextureCA(settings).to(device, memory_format=memory_format)
    if motion is not None:
        model.motion.copy_(motion)
    coarse_loss = TextureLoss(coarse, rng)
    detail_loss = TextureLoss(detail, rng) if detail is not None else None
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.iterations, eta_min=0.0002
    )
    pool_size = max(24, args.batch_size)
    pool = model.initial_field(rng, pool_size)
    axis = torch.arange(args.size, device=device)
    print(f"Training {settings.mode} on {device_name} | {args.size}×{args.size} cells", flush=True)
    for iteration in range(1, args.iterations + 1):
        indices = torch.randperm(pool_size, generator=rng, device=device)[: args.batch_size]
        state = pool[indices]
        if iteration % 3 == 1:
            state[0] = model.initial_field(rng)[0]
        elif iteration % 3 == 2:
            x, y = torch.randint(args.size, (2,), device=device, generator=rng)
            dx = (axis - x + args.size // 2).remainder(args.size) - args.size // 2
            dy = (axis - y + args.size // 2).remainder(args.size) - args.size // 2
            state[-1].masked_fill_(dx.square() + dy[:, None].square() <= (args.size // 8) ** 2, 0)
        state = state.contiguous(memory_format=memory_format)
        start = (iteration - 1) * args.steps
        for step in range(args.steps):
            state = model(state, generator=rng, step=start + step)
        loss = coarse_loss(state[:, :4])
        if detail_loss is not None and model.decoder is not None:
            loss = loss + detail_loss.decoded(model.decoder, state)
        if not torch.isfinite(loss):
            raise ValueError(f"Texture training diverged at iteration {iteration}.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()
        pool[indices] = state.detach()
        if iteration == 1 or iteration % 25 == 0 or iteration == args.iterations:
            print(f"{iteration}/{args.iterations}  texture loss {loss.item():.5f}", flush=True)
    save_model(args.output, model, settings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument(
        "--motion", type=Path, help="Animated image, video, frame folder, or NumPy flow"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", type=positive_int, default=32)
    parser.add_argument("--iterations", type=positive_int, default=800)
    parser.add_argument("--steps", type=positive_int, default=12)
    parser.add_argument("--batch-size", type=positive_int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--cpu-threads", type=thread_count, default=0, help="CPU workers (0 chooses automatically)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--neural-detail", action="store_true")
    parser.add_argument("--detail-scale", type=positive_int, default=4)
    args = parser.parse_args()
    if not MIN_GRID_SIZE <= args.size <= MAX_GRID_SIZE:
        parser.error(f"--size must be between {MIN_GRID_SIZE} and {MAX_GRID_SIZE}")
    if not 2 <= args.detail_scale <= MAX_DETAIL_SCALE:
        parser.error(f"--detail-scale must be between 2 and {MAX_DETAIL_SCALE}")
    if args.output.resolve() in {
        path.resolve() for path in (args.target, args.motion) if path is not None
    }:
        parser.error("Choose an output different from the texture and motion references.")
    try:
        train_texture(args)
    except (
        OSError,
        RuntimeError,
        ValueError,
        subprocess.SubprocessError,
        Image.DecompressionBombError,
    ) as error:
        print(f"Texture training error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
