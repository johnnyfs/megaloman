"""Text -> concept image ("imagination" stage). Pretrained SD-Turbo, steered to NES-friendly concepts.

The concept is not the output: it is fed to the hand-weighted renderer (`render.py`), which owns every
pixel-art decision. So the prompt asks for what the renderer converts best: one subject, centred, full
body, flat cel colours, bold dark outlines, plain white background.
"""
import os

import numpy as np

MODEL = os.environ.get('NESPIX_T2I', 'stabilityai/sdxl-turbo')

STYLE = ('{desc}, cute cartoon mascot character, full body, centered, isolated on a plain white background, '
         'flat colors, thick black outline, simple bold shapes, sticker style vector illustration')

NEG = 'text, watermark, frame, border, scenery, background pattern, multiple objects, cropped, blurry, photo, gradient'

_pipes = {}


def _dtype():
    import torch
    # SDXL in fp32 needs ~14 GB; bfloat16 halves that and runs well on modern x86 CPUs
    return torch.bfloat16 if 'xl' in MODEL else torch.float32


def _pipe(kind='t2i'):
    if kind in _pipes:
        return _pipes[kind]
    import torch
    from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
    torch.set_num_threads(os.cpu_count() or 4)
    if 't2i' not in _pipes:
        _pipes['t2i'] = AutoPipelineForText2Image.from_pretrained(MODEL, variant='fp16', torch_dtype=_dtype())
        _pipes['t2i'].set_progress_bar_config(disable=True)
    if kind == 'i2i':
        _pipes['i2i'] = AutoPipelineForImage2Image.from_pipe(_pipes['t2i'])
    return _pipes[kind]


def imagine(desc, seeds=(0, 1, 2, 3), steps=2, size=512, style=STYLE):
    """Return a list of PIL concept images, one per seed."""
    import torch
    pipe = _pipe('t2i')
    out = []
    for s in seeds:
        g = torch.Generator().manual_seed(int(s))
        im = pipe(prompt=style.format(desc=desc), num_inference_steps=steps, guidance_scale=0.0,
                  height=size, width=size, generator=g).images[0]
        out.append(im)
    return out


def restyle(image, desc='', seeds=(0,), strength=0.5, steps=2, size=512, style=STYLE):
    """Image (+ optional text) -> flattened concept: SD-Turbo img2img toward the NES concept style.

    Use for photos or painterly references the renderer can't flatten on its own."""
    import torch
    from PIL import Image
    pipe = _pipe('i2i')
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    if image.mode == 'RGBA':
        bg = Image.new('RGBA', image.size, (255, 255, 255, 255))
        bg.alpha_composite(image)
        image = bg
    image = image.convert('RGB').resize((size, size), Image.LANCZOS)
    out = []
    for s in seeds:
        g = torch.Generator().manual_seed(int(s))
        steps_eff = max(steps, int(np.ceil(1 / max(strength, 1e-3))))
        out.append(pipe(prompt=style.format(desc=desc or 'the character'), image=image, strength=strength,
                        num_inference_steps=steps_eff, guidance_scale=0.0, generator=g).images[0])
    return out
