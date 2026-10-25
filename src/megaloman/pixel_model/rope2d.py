"""Zero-based axial RoPE, FP32 trigonometry, metadata prefixes unrotated.

Each head is split equally into X and Y channels, with adjacent scalar pairs.
The position-provider interface accepts depth for a later color-depth axis; v0
only implements spatial axes and does not mix palette depth into token IDs.
"""
import torch


def pixel_coordinates(height, width, *, device, num_colors=4):
    if num_colors != 4:
        raise ValueError("a future color-depth axis needs a versioned position provider")
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    return torch.stack((x.flatten(), y.flatten()), -1)


def apply_axial_rope(tensor, coordinates, prefix_tokens=1):
    """[B,heads,prefix+HW,head_dim] -> same shape; never applied to V."""
    if tensor.shape[-1] % 4:
        raise ValueError("head dimension must be divisible by four")
    if tensor.shape[-2] != prefix_tokens + len(coordinates):
        raise ValueError("coordinate/token count mismatch")
    half = tensor.shape[-1] // 2
    frequencies = 10000.0 ** (-torch.arange(0, half, 2, device=tensor.device, dtype=torch.float32) / half)
    pixels = tensor[..., prefix_tokens:, :].float()
    axes = []
    for axis in range(2):
        pairs = pixels[..., axis * half:(axis + 1) * half].reshape(*pixels.shape[:-1], half // 2, 2)
        angles = coordinates[:, axis].float()[:, None] * frequencies[None, :]
        c, s = angles.cos(), angles.sin()
        a, b = pairs.unbind(-1)
        axes.append(torch.stack((a * c - b * s, a * s + b * c), -1).flatten(-2))
    return torch.cat((tensor[..., :prefix_tokens, :], torch.cat(axes, -1).to(tensor.dtype)), -2)
