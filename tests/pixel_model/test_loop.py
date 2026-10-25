"""Loop architecture options: masking happens before any conv/bias, so hidden pixels cannot leak."""
import torch

from megaloman.pixel_model.config import ModelConfig
from megaloman.pixel_model.masking import corrupt, make_mask
from megaloman.pixel_model.model import PixelTransformer
from megaloman.pixel_model.sample import sample
from megaloman.pixel_model.config import SampleConfig


def _model(**kw):
    torch.manual_seed(0)
    model = PixelTransformer(ModelConfig(d_model=32, layers=2, heads=2, dropout=0.0, **kw), (16,)).eval()
    for p in model.parameters():  # break zero-inits so conv/bias paths are active
        torch.nn.init.normal_(p, std=.1) if p.ndim > 1 else None
    return model


def test_hidden_pixels_do_not_leak_through_stem_decoder_or_bias():
    g = torch.Generator().manual_seed(3)
    a = torch.randint(0, 4, (2, 16, 16), generator=g)
    mask = make_mask(torch.ones_like(a, dtype=torch.bool), ratio=.5, generator=g)
    b = torch.where(mask, (a + 1) % 4, a)  # differs from a only at masked cells
    for kw in ({"conv_stem": 2}, {"conv_decoder": 2}, {"rel_bias": True},
               {"conv_stem": 2, "conv_decoder": 2, "rel_bias": True}):
        model = _model(**kw)
        with torch.no_grad():
            la, lb = model(corrupt(a, mask)), model(corrupt(b, mask))
            assert torch.equal(la, lb), kw
            # and the option really is active: visible changes do change outputs
            c = torch.where(~mask, (a + 1) % 4, a)
            assert not torch.allclose(la, model(corrupt(c, mask))), kw


def test_remask_sampler_completes():
    model = _model(conv_stem=1)
    out = sample(model, size=16, count=2, seed=1, cfg=SampleConfig(steps=8, remask=.1, ranking_temperature=1.0))
    assert out.shape == (2, 16, 16) and int(out.max()) < 4
