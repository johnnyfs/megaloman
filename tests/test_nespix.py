"""Hard guarantees of the nespix renderer (no network, no torch)."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from megaloman.nespix.palette import NES
from megaloman.nespix.render import Concept, Params, render, tile_colours


def _creature(sym=True):
    im = Image.new('RGB', (512, 512), 'white')
    d = ImageDraw.Draw(im)
    d.ellipse((96, 80, 416, 400), fill=(248, 120, 88), outline='black', width=10)       # body
    d.rectangle((150, 330, 200, 470), fill=(240, 208, 176), outline='black', width=8)   # legs
    d.rectangle((312, 330, 362, 470), fill=(240, 208, 176), outline='black', width=8)
    d.ellipse((176, 170, 236, 250), fill='white', outline='black', width=8)              # eyes
    d.ellipse((276, 170, 336, 250), fill='white', outline='black', width=8)
    d.ellipse((196, 196, 220, 232), fill='black')
    d.ellipse((292, 196, 316, 232), fill='black')
    if not sym:
        d.polygon([(400, 120), (500, 60), (470, 180)], fill=(60, 120, 250), outline='black')
    return im


@pytest.fixture(scope='module')
def concept():
    return Concept(_creature())


@pytest.mark.parametrize('n', [8, 16, 24, 32, 64, 128, 256])
def test_nes_legal(concept, n):
    sp = render(concept, n)
    g = sp.grid
    assert g.shape == (n, n)
    assert len(sp.codes) <= 4 and all(c in NES for c in sp.codes)
    used = np.unique(g[g >= 0])
    assert len(used) <= 4
    assert tile_colours(g) <= 3                       # NES: <= 3 opaque colours per 8x8 tile
    a = np.array(sp.rgba())
    assert set(map(tuple, a[a[..., 3] > 0][:, :3])) <= {tuple(int(NES[c][i:i + 2], 16) for i in (0, 2, 4)) for c in sp.codes}


@pytest.mark.parametrize('n', [16, 32, 64])
def test_outlined(concept, n):
    """Every fill pixel is separated from transparency by black (4-connected)."""
    g = render(concept, n).grid
    P = np.pad(g, 1, constant_values=-1)
    fill = g > 0
    for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
        assert not (fill & (P[dy:dy + n, dx:dx + n] == -1)).any()


@pytest.mark.parametrize('n', [8, 15, 16, 24, 31, 32])
def test_exact_mirror(concept, n):
    sp = render(concept, n)
    assert sp.meta['symmetric']
    assert (sp.grid == sp.grid[:, ::-1]).all()


def test_asymmetric_left_alone():
    sp = render(Concept(_creature(sym=False)), 32)
    assert not sp.meta['symmetric']


def test_deterministic(concept):
    a, b = render(concept, 24), render(Concept(_creature()), 24)
    assert (a.grid == b.grid).all() and a.codes == b.codes


def test_palette_choice(concept):
    sp = render(concept, 32, Params(colors=3))
    assert sp.codes[0] == '0F' and len(sp.codes) == 3
    assert '26' in sp.codes or '16' in sp.codes or '27' in sp.codes       # salmon body found


def test_forced_palette(concept):
    sp = render(concept, 16, palette=('0F', '12', '30'), colors=3)
    assert sp.codes == ['0F', '12', '20']                                  # 30 is an alias of 20


def test_eyes_survive_16(concept):
    """The white eyes must still be there at 16 px (detail-preserving downsampler)."""
    sp = render(concept, 16, Params(colors=4))
    light = [i for i, c in enumerate(sp.codes) if c in ('30', '20', '3D', '36', '37')]
    assert light and np.isin(sp.grid, light).sum() >= 2
