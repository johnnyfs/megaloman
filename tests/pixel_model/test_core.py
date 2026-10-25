import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from megaloman.pixel_model.config import Config, ModelConfig
from megaloman.pixel_model.data import ApprovedDataset, BalancedBatchStream, create_fixture
from megaloman.pixel_model.masking import corrupt, make_mask, masked_loss
from megaloman.pixel_model.model import PixelTransformer
from megaloman.pixel_model.palette import MASK, PAD, read_indexed, write_indexed
from megaloman.pixel_model.rope2d import apply_axial_rope, pixel_coordinates


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_rejects_unknown_and_future_depth(self):
        with self.assertRaises(ValueError):
            Config.from_dict({"model": {"num_colors": 16}})
        with self.assertRaises(ValueError):
            Config.from_dict({"model": {"heads": 5}})
        with self.assertRaises(ValueError):
            Config.from_dict({"train": {"typo": 1}})
        with self.assertRaises(ValueError):
            Config.from_dict({"future": {}})

    def test_indexed_round_trip_and_alpha(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sprite.png"
            grid = np.tile(np.arange(4), (16, 4))
            write_indexed(p, grid)
            np.testing.assert_array_equal(read_indexed(p, 16), grid)
            with self.assertRaises(ValueError):
                read_indexed(p, 24)
            write_indexed(p, grid, palette=((0, 0, 0), (0, 0, 0), (100, 100, 100), (255, 255, 255)))
            with Image.open(p) as im:
                rgba = np.array(im.convert("RGBA"))
                self.assertEqual(rgba[0, 0, 3], 0)
                self.assertEqual(rgba[0, 1, 3], 255)
                np.testing.assert_array_equal(rgba[0, 0, :3], rgba[0, 1, :3])
                im.save(p, transparency=bytes([0, 128, 255, 255]))
            with self.assertRaisesRegex(ValueError, "fully opaque"):
                read_indexed(p)
            Image.new("RGBA", (16, 16)).save(p)
            with self.assertRaisesRegex(ValueError, "indexed"):
                read_indexed(p)

    def test_contract_and_balanced_stream_state(self):
        with tempfile.TemporaryDirectory() as d:
            p = create_fixture(d)
            ds = ApprovedDataset(p)
            self.assertEqual(ds.summary()["counts"]["train/16"], 16)
            stream = BalancedBatchStream(ds, 2, 42)
            state = stream.state_dict()
            a = stream.next()
            stream.load_state_dict(state)
            torch.testing.assert_close(a, stream.next())
            raw = json.loads(p.read_text())
            raw["records"][16]["identity"] = raw["records"][0]["identity"]
            p.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "leaks"):
                ApprovedDataset(p)
            raw["records"][16]["identity"] = "new"
            raw["records"][16]["path"] = raw["records"][0]["path"]
            raw["records"][16]["sha256"] = raw["records"][0]["sha256"]
            p.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                ApprovedDataset(p)
            raw["records"][0]["sha256"] = "changed"
            p.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                ApprovedDataset(p)

    def test_masks_and_per_example_objective(self):
        target = torch.zeros(3, 16, 16, dtype=torch.long)
        target[:, 0, 0] = PAD
        valid = target != PAD
        mask = make_mask(valid, generator=torch.Generator().manual_seed(9))
        self.assertTrue((mask.flatten(1).sum(1) >= 1).all())
        self.assertFalse(mask[:, 0, 0].any())
        self.assertTrue(torch.equal(corrupt(target, mask)[mask], torch.full_like(target[mask], MASK)))
        for ratio in (0, .25, 1):
            m = make_mask(valid, ratio=ratio, contiguous=True, generator=torch.Generator().manual_seed(0))
            self.assertEqual(m[0].sum().item(), max(1, int(np.ceil(255 * ratio))))
        logits = torch.randn(3, 16, 16, 4, requires_grad=True)
        loss = masked_loss(logits, target, mask)
        expected = torch.stack([torch.nn.functional.cross_entropy(logits[i][mask[i]], target[i][mask[i]])
                                for i in range(3)]).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(logits.grad[~mask].abs().sum().item(), 0)
        changed = target.clone()
        changed[~mask] = MASK  # never selected; CE must not inspect these values
        torch.testing.assert_close(masked_loss(logits.detach(), changed, mask), loss.detach())

    def test_full_mask_frequency_and_target_isolation(self):
        from megaloman.pixel_model.config import MaskConfig
        valid = torch.ones(1000, 16, 16, dtype=torch.bool)
        masks = make_mask(valid, MaskConfig(full_probability=.2), generator=torch.Generator().manual_seed(77))
        frequency = masks.flatten(1).all(1).float().mean().item()
        self.assertGreater(frequency, .17)
        self.assertLess(frequency, .28)  # ceil(cosine fraction * n) can also select every cell
        a, b = torch.zeros(1, 16, 16, dtype=torch.long), torch.ones(1, 16, 16, dtype=torch.long)
        torch.testing.assert_close(corrupt(a, valid[:1]), corrupt(b, valid[:1]))

    def test_rope_axes_prefix_and_relative_displacement(self):
        coords = pixel_coordinates(2, 2, device="cpu")
        x = torch.randn(1, 2, 5, 8)
        out = apply_axial_rope(x, coords)
        torch.testing.assert_close(out[:, :, 0], x[:, :, 0])
        torch.testing.assert_close(out[:, :, 2, 4:], x[:, :, 2, 4:])  # x=1, y=0
        torch.testing.assert_close(out[:, :, 3, :4], x[:, :, 3, :4])  # x=0, y=1
        q, k = torch.randn_like(x), torch.randn_like(x)
        before = apply_axial_rope(q, coords)[..., 1:, :] @ apply_axial_rope(k, coords)[..., 1:, :].transpose(-2, -1)
        after = apply_axial_rope(q, coords + 3)[..., 1:, :] @ apply_axial_rope(k, coords + 3)[..., 1:, :].transpose(-2, -1)
        torch.testing.assert_close(before, after, atol=2e-6, rtol=2e-6)

    def test_forward_backward_and_pad_exclusion(self):
        model = PixelTransformer(ModelConfig(d_model=32, layers=1, heads=4, dropout=0), (16, 24))
        for n in (16, 24):
            x = torch.randint(4, (2, n, n))
            mask = make_mask(x != PAD, ratio=.5)
            logits = model(corrupt(x, mask), height=n, width=n)
            self.assertEqual(logits.shape, (2, n, n, 4))
            masked_loss(logits, x, mask).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        model.eval()
        valid = torch.ones(2, 16, 16, dtype=torch.bool)
        valid[:, 0] = False
        a = torch.zeros(2, 16, 16, dtype=torch.long)
        b = a.clone(); b[:, 0] = 3
        torch.testing.assert_close(model(a, valid_mask=valid)[valid], model(b, valid_mask=valid)[valid])
        with self.assertRaisesRegex(ValueError, "unsupported"):
            model(torch.zeros(1, 32, 32, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
