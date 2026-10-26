"""Discrete denoising (D3PM / MaskGIT-style) for 4-role sprites.

Forward process q(x_t | x_0) at level t in [0, 1]: optional structured sprite damage (tile
shifts/swaps, outline breaks, blur-then-requantise patches; count ~ Poisson(t * rate)), then
each pixel is independently corrupted with probability t, as either an absorbing MASK token
or a uniform random role. The model sees x_t and t and predicts x_0 at EVERY position, so
every pixel stays revisable during sampling (unlike masked decoding, which locks commits).
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch.nn import functional as F

from .palette import MASK, PAD

# role -> "luminance" for blur-then-requantise (transparent, ink, tone1, tone2)
LUMA = torch.tensor([0.0, 1.0, 2.0, 3.0])


@dataclass
class NoiseConfig:
    mask_share: float = 0.5        # of per-pixel corruptions: MASK token (rest: uniform random role)
    uniform: bool = True           # False: per-pixel corruption is MASK only (pure absorbing)
    tile_shift: bool = True        # move a 2-6 px tile by 1-2 px, or swap two tiles
    outline_break: bool = True     # delete / recolour a run of silhouette-border INK
    blur_patch: bool = True        # 3x3 box-blur a patch in luminance, requantise to roles
    structured_rate: float = 4.0   # expected structured ops at t = 1
    corrupted_weight: float = 2.0  # loss weight on corrupted cells (1 on clean cells)
    t_min: float = 0.02
    full_noise_probability: float = 0.1


def _border_ink(g):
    fg = g != 0
    p = F.pad(fg, (1, 1, 1, 1), value=False)
    edge = fg & (~p[:-2, 1:-1] | ~p[2:, 1:-1] | ~p[1:-1, :-2] | ~p[1:-1, 2:])
    return edge & (g == 1)


def _randint(lo, hi, gen):
    return int(torch.randint(lo, hi + 1, (), generator=gen))


def structured_damage(g, ops, cfg, gen):
    """Apply `ops` random structured degradations to one LongTensor[H,W] grid (copy)."""
    g = g.clone()
    h, w = g.shape
    kinds = [k for k, on in (("tile", cfg.tile_shift), ("outline", cfg.outline_break), ("blur", cfg.blur_patch)) if on]
    if not kinds:
        return g
    fgc = (g != 0).nonzero()
    for _ in range(ops):
        kind = kinds[_randint(0, len(kinds) - 1, gen)]
        cy, cx = (fgc[_randint(0, len(fgc) - 1, gen)].tolist() if len(fgc) else (h // 2, w // 2))
        if kind == "tile":
            s = _randint(2, max(2, min(6, h // 4)), gen)
            y0, x0 = min(max(0, cy - s // 2), h - s), min(max(0, cx - s // 2), w - s)
            tile = g[y0:y0 + s, x0:x0 + s].clone()
            if torch.rand((), generator=gen) < .5:  # shift by 1-2 px
                dy, dx = _randint(-2, 2, gen), _randint(-2, 2, gen)
                y1, x1 = min(max(0, y0 + dy), h - s), min(max(0, x0 + dx), w - s)
            else:  # swap with another tile
                y1, x1 = _randint(0, h - s, gen), _randint(0, w - s, gen)
                g[y0:y0 + s, x0:x0 + s] = g[y1:y1 + s, x1:x1 + s].clone()
            g[y1:y1 + s, x1:x1 + s] = tile
        elif kind == "outline":
            edge = _border_ink(g).nonzero()
            if not len(edge):
                continue
            ey, ex = edge[_randint(0, len(edge) - 1, gen)].tolist()
            n = _randint(2, 5, gen)
            near = edge[((edge[:, 0] - ey).abs() + (edge[:, 1] - ex).abs()).argsort()[:n]]
            new = _randint(0, 3, gen) if torch.rand((), generator=gen) < .5 else 0
            g[near[:, 0], near[:, 1]] = new if new != 1 else 2
        else:  # blur then requantise
            s = _randint(3, max(3, min(8, h // 3)), gen)
            y0, x0 = min(max(0, cy - s // 2), h - s), min(max(0, cx - s // 2), w - s)
            lum = LUMA[g].float()[None, None]
            blur = F.avg_pool2d(F.pad(lum, (1, 1, 1, 1), mode="replicate"), 3, 1)[0, 0]
            g[y0:y0 + s, x0:x0 + s] = blur[y0:y0 + s, x0:x0 + s].round().clamp(0, 3).long()
    return g


def corrupt(x0, t, cfg, gen, *, structured=True):
    """x0 LongTensor[B,H,W] on CPU, t FloatTensor[B]. Returns x_t, corrupted BoolTensor."""
    out = []
    for b in range(len(x0)):
        g = x0[b]
        if structured:
            ops = int(torch.poisson(torch.tensor(float(t[b]) * cfg.structured_rate), generator=gen))
            if ops:
                g = structured_damage(g, ops, cfg, gen)
        out.append(g)
    xt = torch.stack(out)
    hit = torch.rand(x0.shape, generator=gen) < t[:, None, None]
    as_mask = (torch.rand(x0.shape, generator=gen) < cfg.mask_share) | (not cfg.uniform)
    rand = torch.randint(0, 4, x0.shape, generator=gen)
    xt = torch.where(hit & as_mask, torch.full_like(xt, MASK), torch.where(hit, rand, xt))
    return xt, xt != x0


def sample_t(n, cfg, gen):
    t = torch.rand(n, generator=gen) * (1 - cfg.t_min) + cfg.t_min
    full = torch.rand(n, generator=gen) < cfg.full_noise_probability
    return torch.where(full, torch.ones_like(t), t)


def denoise_loss(logits, x0, corrupted, cfg):
    ce = F.cross_entropy(logits.float().movedim(-1, 1), x0, reduction="none")
    weight = 1 + (cfg.corrupted_weight - 1) * corrupted.float()
    return ((ce * weight).flatten(1).sum(1) / weight.flatten(1).sum(1)).mean()


@dataclass
class DenoiseSampleConfig:
    steps: int = 32
    temperature: float = 1.0
    schedule: str = "cosine"       # t_k = cos(pi/2 * k/T) or linear
    renoise: str = "random"        # random: D3PM-style; confidence: least confident cells; none: refeed prediction (cleanup)
    ranking_temperature: float = 1.0  # Gumbel noise on confidence ranking (scaled by t)
    final_argmax: bool = True


def _level(k, steps, schedule, t0=1.0):
    u = k / steps
    return t0 * (math.cos(math.pi / 2 * u) if schedule == "cosine" else 1 - u)


@torch.inference_mode()
def denoise_sample(model, *, size=None, count=1, seed=0, cfg=None, noise=None, start=None, t0=1.0, trace=False):
    """Iterative denoising. start: LongTensor[B,H,W] rough sprite (cleanup) run from level t0."""
    cfg = cfg or DenoiseSampleConfig()
    noise = noise or NoiseConfig()
    device = next(model.parameters()).device
    gen = torch.Generator().manual_seed(seed)
    if start is None:
        shape = (count, size, size)
        hit_mask = (torch.rand(shape, generator=gen) < noise.mask_share) | (not noise.uniform)
        x = torch.where(hit_mask, torch.full(shape, MASK), torch.randint(0, 4, shape, generator=gen))
    else:
        x = start.clone().cpu()
        shape = tuple(x.shape)
    frames = [x.clone()] if trace else None
    was = model.training
    model.eval()
    try:
        for k in range(cfg.steps):
            t, t_next = _level(k, cfg.steps, cfg.schedule, t0), _level(k + 1, cfg.steps, cfg.schedule, t0)
            logits = model(x.to(device), noise_level=t).float().cpu()
            if cfg.temperature > 0 and not (cfg.final_argmax and k == cfg.steps - 1):
                probs = (logits / cfg.temperature).softmax(-1)
                pred = torch.multinomial(probs.reshape(-1, 4), 1, generator=gen).reshape(shape)
            else:
                probs = logits.softmax(-1)
                pred = logits.argmax(-1)
            if k == cfg.steps - 1 or t_next <= 0:
                x = pred
                break
            n = x[0].numel()
            keep = torch.zeros(shape, dtype=torch.bool)
            if cfg.renoise == "none":  # cleanup: feed the prediction back at the lower level
                x = pred
                if trace:
                    frames.append(x.clone())
                continue
            if cfg.renoise == "random":
                renoise = torch.rand(shape, generator=gen) < t_next
            else:
                conf = probs.gather(-1, pred[..., None])[..., 0].clamp_min(1e-9).log()
                u = torch.rand(shape, generator=gen).clamp(1e-7, 1 - 1e-7)
                conf = conf + cfg.ranking_temperature * t * -(-u.log()).log()
                cut = round(t_next * n)
                order = conf.flatten(1).argsort(1)[:, :cut]
                renoise = keep.flatten(1).scatter(1, order, True).reshape(shape)
            as_mask = (torch.rand(shape, generator=gen) < noise.mask_share) | (not noise.uniform)
            rand = torch.randint(0, 4, shape, generator=gen)
            x = torch.where(renoise & as_mask, torch.full(shape, MASK), torch.where(renoise, rand, pred))
            if trace:
                frames.append(x.clone())
    finally:
        model.train(was)
    return (x, frames) if trace else x


class MaskLevelWrapper(torch.nn.Module):
    """Lets masked-model evaluation (conditional_report) drive a denoiser: t = MASK fraction * scale
    (scale 1/mask_share: the level at which training would show that many MASK tokens)."""

    def __init__(self, model, scale=1.0):
        super().__init__()
        self.inner, self.resolutions, self.vocab, self.scale = model, model.resolutions, {}, scale

    def forward(self, pixels, labels=None, **_):
        valid = pixels != PAD
        t = ((pixels == MASK) & valid).flatten(1).sum(1).float() / valid.flatten(1).sum(1).clamp_min(1)
        return torch.cat([self.inner(pixels[i:i + 1], noise_level=min(1.0, float(t[i]) * self.scale)) for i in range(len(pixels))])


def noise_dict(cfg):
    return asdict(cfg)
