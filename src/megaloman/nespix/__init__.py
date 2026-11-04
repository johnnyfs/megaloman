"""nespix: text and/or image -> NES-quality pixel art (4 colours, 8x8 to 256x256).

    from megaloman.nespix import generate, pixelate
    res = generate('a robot squid enemy', sizes=(16, 32))       # text -> sprites
    sp = pixelate('concept.png', 24)                            # image -> best sprite
    sp.rgba().save('squid_24.png')
"""
from .render import Concept, Params, Sprite, render
from .model import generate, pixelate

__all__ = ['Concept', 'Params', 'Sprite', 'render', 'generate', 'pixelate']
