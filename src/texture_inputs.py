"""Load texture images and estimate periodic motion from reference frames."""

import re
import shutil
import subprocess
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.nn import functional as F

from nca import MAX_MOTION_FRAMES

VIDEO_TYPES = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
FRAME_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def frame_order(path: Path) -> list[str | int]:
    return [
        int(part) if index % 2 else part
        for index, part in enumerate(re.split(r"(\d+)", path.name.casefold()))
    ]


def image_tensor(image: Image.Image, size: int) -> Tensor:
    fitted = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGBA"), (size, size))
    rgba = torch.from_numpy(np.array(fitted)).permute(2, 0, 1).float() / 255
    rgba[:3] *= rgba[3:4]
    return rgba


def read_motion_frames(path: Path, size: int) -> Tensor:
    limit = MAX_MOTION_FRAMES + 1
    if path.is_dir():
        frames = []
        images = (
            item for item in path.iterdir() if item.is_file() and item.suffix.lower() in FRAME_TYPES
        )
        for file in sorted(images, key=frame_order)[:limit]:
            with Image.open(file) as image:
                frames.append(image_tensor(image, size)[:3])
    elif path.suffix.lower() in VIDEO_TYPES:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise ValueError(
                "FFmpeg is required for video files. Use an animated GIF, APNG, WebP or a frame folder instead."
            )
        result = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-nostdin",
                "-protocol_whitelist",
                "file,pipe",
                "-i",
                str(path),
                "-vf",
                f"scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}",
                "-frames:v",
                str(limit),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            capture_output=True,
            timeout=60,
            check=True,
        )
        colors = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, size, size, 3).copy()
        frames = list(torch.from_numpy(colors).permute(0, 3, 1, 2).float() / 255)
    else:
        with Image.open(path) as image:
            frames = []
            for index in range(min(limit, getattr(image, "n_frames", 1))):
                image.seek(index)
                frames.append(image_tensor(image, size)[:3])
    if len(frames) < 2:
        raise ValueError("A motion reference needs at least two frames.")
    return torch.stack(frames).mean(dim=1)


def estimate_flow(before: Tensor, after: Tensor) -> Tensor:
    """Estimate bulk translation, then a local Lucas–Kanade residual in cell units."""
    size = before.shape[-1]
    first, last = torch.fft.rfft2(before - before.mean()), torch.fft.rfft2(after - after.mean())
    spectrum = last * first.conj()
    correlation = torch.fft.irfft2(spectrum / spectrum.abs().clamp_min(1e-6), s=before.shape)
    row, column = divmod(int(correlation.argmax()), size)
    dy, dx = (row + size // 2) % size - size // 2, (column + size // 2) % size - size // 2
    aligned = before.roll((dy, dx), dims=(0, 1))
    average = (aligned + after) * 0.5
    gx = (average.roll(-1, 1) - average.roll(1, 1)) * 0.5
    gy = (average.roll(-1, 0) - average.roll(1, 0)) * 0.5
    dt = aligned - after
    terms = torch.stack((gx.square(), gy.square(), gx * gy, gx * dt, gy * dt))[None]
    xx, yy, xy, xt, yt = F.avg_pool2d(F.pad(terms, (2, 2, 2, 2), mode="circular"), 5, stride=1)[0]
    xx, yy = xx + 0.002, yy + 0.002
    determinant = xx * yy - xy.square()
    residual = torch.stack(((yy * xt - xy * yt) / determinant, (xx * yt - xy * xt) / determinant))
    return residual.clamp(-1, 1) + before.new_tensor((dx, dy)).view(2, 1, 1)


def load_motion(path: Path, size: int) -> Tensor:
    if path.suffix.lower() == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if not isinstance(array, np.ndarray):
            array.close()
            raise ValueError("Supply a single NumPy .npy flow array.")
        if array.dtype.kind not in "fiu":
            raise ValueError("Motion vectors must use real numeric values.")
        if array.ndim == 3:
            array = array[None]
        if array.ndim != 4 or array.shape[-1] != 2 or not 1 <= array.shape[0] <= MAX_MOTION_FRAMES:
            raise ValueError("Flow must have shape H×W×2 or F×H×W×2 (up to 32 frames).")
        if min(array.shape[1:3]) < 1:
            raise ValueError("Motion fields must have positive width and height.")
        flow = torch.from_numpy(np.array(array, dtype=np.float32)).permute(0, 3, 1, 2)
        scale = flow.new_tensor((size / array.shape[2], size / array.shape[1])).view(1, 2, 1, 1)
        flow = F.interpolate(flow, size=(size, size), mode="bilinear", align_corners=False) * scale
    else:
        frames = read_motion_frames(path, size)
        flow = torch.stack([estimate_flow(first, last) for first, last in pairwise(frames)])
    if not torch.isfinite(flow).all() or flow.abs().max() < 1e-5:
        raise ValueError("The reference must contain finite, nonzero motion.")
    # Bound displacement per update for playback and training on the cell grid.
    return flow / (flow.norm(dim=1, keepdim=True) / 2).clamp_min(1)
