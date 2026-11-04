"""NES 2C02 colours, OKLab, and mapping the source's flat tones to distinct NES colours."""
import numpy as np

NES = {
    '00': '7C7C7C', '01': '0000FC', '02': '0000BC', '03': '4428BC', '04': '940084', '05': 'A80020', '06': 'A81000',
    '07': '881400', '08': '503000', '09': '007800', '0A': '006800', '0B': '005800', '0C': '004058', '0F': '000000',
    '10': 'BCBCBC', '11': '0078F8', '12': '0058F8', '13': '6844FC', '14': 'D800CC', '15': 'E40058', '16': 'F83800',
    '17': 'E45C10', '18': 'AC7C00', '19': '00B800', '1A': '00A800', '1B': '00A844', '1C': '008888',
    '20': 'F8F8F8', '21': '3CBCFC', '22': '6888FC', '23': '9878F8', '24': 'F878F8', '25': 'F85898', '26': 'F87858',
    '27': 'FCA044', '28': 'F8B800', '29': 'B8F818', '2A': '58D854', '2B': '58F898', '2C': '00E8D8',
    '31': 'A4E4FC', '32': 'B8B8F8', '33': 'D8B8F8', '34': 'F8B8F8', '35': 'F8A4C0', '36': 'F0D0B0',
    '37': 'FCE0A8', '38': 'F8D878', '39': 'D8F878', '3A': 'B8F8B8', '3B': 'B8F8D8', '3C': '00FCFC', '3D': 'F8D8F8',
}
ALIAS = {'30': '20', '2D': '00', '0D': '0F', '1D': '0F', '1F': '0F', '2F': '0F', '3F': '0F', '0E': '0F', '1E': '0F'}


def rgb(code):
    code = ALIAS.get(code.upper(), code.upper())
    h = NES[code]
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def oklab(rgb_arr):
    c = np.asarray(rgb_arr, float) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    return np.cbrt(c @ m1.T) @ m2.T


CODES = [c for c in NES if c != '0F']
LAB = oklab(np.array([rgb(c) for c in CODES], float))


def nearest_ramp(tone_lab):
    """Black for tone 0, then for the other tones the jointly nearest *distinct* NES colours (exhaustive)."""
    tl = np.asarray(tone_lab)[1:]
    D = ((tl[:, None] - LAB[None]) ** 2).sum(-1)       # (k-1, K)
    order = np.argsort(D, 1)[:, :8]
    best, be = None, np.inf

    def rec(i, used, cost):
        nonlocal best, be
        if cost >= be:
            return
        if i == len(tl):
            best, be = list(used), cost
            return
        for j in order[i]:
            if j not in used:
                rec(i + 1, used + [j], cost + D[i, j])

    rec(0, [], 0.0)
    return ['0F'] + [CODES[j] for j in best]
