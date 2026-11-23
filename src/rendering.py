"""Convert the simulation's premultiplied colors to display and export images."""

import torch
from PIL import Image
from torch import Tensor


def to_image(channels: Tensor) -> Image.Image:
    pixels = (channels.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).contiguous()
    return Image.fromarray(pixels.numpy())


def sprite_image(
    rgba: Tensor, *, crisp: bool = False, background: Tensor | None = None
) -> Image.Image:
    alpha = rgba[3:4].clamp(0, 1)
    premultiplied = rgba[:3].clamp_min(0).minimum(alpha)
    if background is not None and not crisp:
        return to_image(premultiplied + background * (1 - alpha))

    # Recover straight color before replacing alpha so hard edges retain their color.
    rgb = premultiplied / alpha.clamp_min(1e-8)
    if crisp:
        alpha = (alpha >= 0.5).to(alpha.dtype)
    if background is not None:
        return to_image(rgb * alpha + background * (1 - alpha))
    rgb = rgb.masked_fill(alpha < 0.5 / 255, 0)
    return to_image(torch.cat((rgb, alpha)))
