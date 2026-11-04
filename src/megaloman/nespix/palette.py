"""NES 2C02 palette and perceptual colour maths (OKLab)."""
import numpy as np

# Canonical 2C02 palette, one entry per distinct colour (duplicates such as 0D/1D/2D/0E... removed).
NES = {
    '00': '7C7C7C', '01': '0000FC', '02': '0000BC', '03': '4428BC', '04': '940084', '05': 'A80020', '06': 'A81000',
    '07': '881400', '08': '503000', '09': '007800', '0A': '006800', '0B': '005800', '0C': '004058', '0F': '000000',
    '10': 'BCBCBC', '11': '0078F8', '12': '0058F8', '13': '6844FC', '14': 'D800CC', '15': 'E40058', '16': 'F83800',
    '17': 'E45C10', '18': 'AC7C00', '19': '00B800', '1A': '00A800', '1B': '00A844', '1C': '008888',
    '20': 'F8F8F8', '21': '3CBCFC', '22': '6888FC', '23': '9878F8', '24': 'F878F8', '25': 'F85898', '26': 'F87858',
    '27': 'FCA044', '28': 'F8B800', '29': 'B8F818', '2A': '58D854', '2B': '58F898', '2C': '00E8D8', '2D': '787878',
    '31': 'A4E4FC', '32': 'B8B8F8', '33': 'D8B8F8', '34': 'F8B8F8', '35': 'F8A4C0', '36': 'F0D0B0',
    '37': 'FCE0A8', '38': 'F8D878', '39': 'D8F878', '3A': 'B8F8B8', '3B': 'B8F8D8', '3C': '00FCFC', '3D': 'F8D8F8',
}
# 2D (787878) is visually identical to 00; keep one grey per level.
NES.pop('2D')
CODES = list(NES)
BLACK = '0F'


def hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


NES_RGB = np.array([hex_rgb(NES[c]) for c in CODES], float)


def oklab(rgb):
    """sRGB 0..255 (..., 3) -> OKLab (..., 3), L in 0..1."""
    c = np.asarray(rgb, float) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    lms = np.cbrt(c @ m1.T)
    return lms @ m2.T


NES_LAB = oklab(NES_RGB)


def dist2(a, b, wl=1.0):
    """Squared OKLab distance with lightness weight wl. a (..., 3), b (K, 3) -> (..., K)."""
    d = a[..., None, :] - b
    d[..., 0] *= wl
    return (d ** 2).sum(-1)


def code_index(code):
    return CODES.index(code)
