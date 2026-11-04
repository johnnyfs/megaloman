"""Text -> high-contrast grayscale concept on a solid white background (SDXL-Turbo, CPU bf16).

Only used when no image is given. The prompt asks for exactly the constrained input the model is built for.
"""
import os

import numpy as np

MODEL = os.environ.get('NESGRAY_T2I', 'stabilityai/sdxl-turbo')
STYLE = ('{desc}, black and white, high contrast, bold thick black ink outlines, flat white and grey fills, '
         'simple chunky shapes, one single character, full body, centered, isolated on a solid white background, '
         'video game sprite design, no shading gradients, no ground, no shadow')
_pipe = None


def imagine(desc, seeds=range(4), steps=1, size=512, style=STYLE):
    """List of grayscale PIL images (mode 'L'), one per seed."""
    global _pipe
    import torch
    from diffusers import AutoPipelineForText2Image
    if _pipe is None:
        torch.set_num_threads(os.cpu_count() or 4)
        _pipe = AutoPipelineForText2Image.from_pretrained(MODEL, variant='fp16', torch_dtype=torch.bfloat16)
        _pipe.set_progress_bar_config(disable=True)
    out = []
    for s in seeds:
        g = torch.Generator().manual_seed(int(s))
        im = _pipe(prompt=style.format(desc=desc), num_inference_steps=steps, guidance_scale=0.0,
                   height=size, width=size, generator=g).images[0]
        out.append(im.convert('L'))
    return out


def concept_score(img):
    """Higher = better constrained input: solid border, one subject not touching the edge, strong contrast."""
    from .model import load_gray, matte
    g = load_gray(img)
    try:
        fg, bg = matte(g)
    except ValueError:
        return -9.0
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    s = -4 * np.concatenate([fg[0], fg[-1], fg[:, 0], fg[:, -1]]).mean()     # subject cropped by the edge
    s -= 5 * border.std()                                                   # background not solid
    cov = fg.mean()
    s -= 4 * abs(np.clip(cov, 0.12, 0.6) - cov)                             # too small / too big
    v = g[fg]
    s += 1.5 * min(1.0, ((v < 0.2).mean() + (v > 0.8).mean()))              # high contrast
    from scipy import ndimage as ndi
    _, n = ndi.label(fg)
    s -= 0.3 * max(0, n - 1)                                                # clutter
    return float(s)
