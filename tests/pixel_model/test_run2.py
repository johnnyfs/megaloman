"""Run-2 additions: 8px buckets, crop/pad, character splits, weighted stream, balanced
loss, ranking decay, global prior, best checkpoint and early-stop resume."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from megaloman.pixel_model.checkpoint import load_model
from megaloman.pixel_model.config import ALLOWED_SIZES, Config, SampleConfig
from megaloman.pixel_model.data import ApprovedDataset, BalancedBatchStream, create_fixture
from megaloman.pixel_model.evaluate import conditional_report, global_prior, summarize
from megaloman.pixel_model.masking import balanced_loss, masked_loss
from megaloman.pixel_model.model import PixelTransformer
from megaloman.pixel_model.sample import sample
from megaloman.pixel_model.snapshot_megaman import bucket, character, place
from megaloman.pixel_model.train import train
from tests.pixel_model.test_workflow import tiny_config


class Run2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_eight_pixel_sizes(self):
        self.assertEqual(ALLOWED_SIZES, (16, 24, 32, 40, 48, 56, 64))
        Config.from_dict({"dataset": {"resolutions": [40, 48, 56]}})
        with self.assertRaises(ValueError):
            Config.from_dict({"dataset": {"resolutions": [20]}})
        model = PixelTransformer(Config().model, (40,))
        self.assertEqual(model(torch.zeros(1, 40, 40, dtype=torch.long)).shape, (1, 40, 40, 4))

    def test_crop_bucket_and_bottom_centre_pad(self):
        self.assertEqual([bucket(5, 3), bucket(17, 9), bucket(24, 24), bucket(25, 1), bucket(65, 2)],
                         [16, 24, 24, 32, None])
        c = np.random.default_rng(0).integers(1, 4, size=(19, 10))
        grid, (x, y) = place(c)
        self.assertEqual(grid.shape, (24, 24))
        self.assertEqual((x, y), (7, 5))
        np.testing.assert_array_equal(grid[y:, x:x + 10], c)
        self.assertEqual((grid != 0).sum(), (c != 0).sum())

    def test_character_groups_games_and_derived(self):
        self.assertEqual(character("megaman/mm1/robot-masters/fire-man/walk"), "fireman")
        self.assertEqual(character("_derived/megaman/mm-rock-and-roll/robot-masters/fireman/sheet-run-trim8"), "fireman")
        self.assertEqual(character("megaman/mm2/enemies/batton/default"), "batton")
        key = "_derived/megaman/mm-rock-and-roll/robot-masters/polarwoman-whiteskates/sheet-skate-throw"
        self.assertEqual(character(key, {"polarwoman"}), "polarwoman")
        self.assertEqual(character("megaman/mm1/robot-masters/fire-man/walk", {"fire"}), "fire")
        self.assertEqual(character("megaman/mmx/enemies/beetleladyplus/a", {"beetlelady"}), "beetlelady")

    def test_weighted_stream_resume_and_shift(self):
        with tempfile.TemporaryDirectory() as d:
            ds = ApprovedDataset(create_fixture(d, (16, 24)), (16, 24))
            a = BalancedBatchStream(ds, 3, 7, bucket_sampling="sqrt_count", hshift=1)
            self.assertAlmostEqual(sum(a.policy["weights"].values()), 1)
            a.next()
            state = a.state_dict()
            b = BalancedBatchStream(ds, 3, 99, bucket_sampling="sqrt_count", hshift=1)
            b.load_state_dict(state)
            for _ in range(5):
                torch.testing.assert_close(a.next(), b.next(), rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "bucket policy"):
                BalancedBatchStream(ds, 3, 7).load_state_dict(state)
            grid = torch.zeros(16, 16, dtype=torch.long)
            grid[10:, 0:4] = 2  # touches the left edge: may only shift right or stay
            for _ in range(20):
                out = a._shift(grid)
                self.assertEqual((out != 0).sum(), (grid != 0).sum())
                self.assertTrue(torch.equal(out[:10], grid[:10]))

    def test_balanced_loss(self):
        logits = torch.randn(2, 4, 4, 4)
        targets = torch.zeros(2, 4, 4, dtype=torch.long)
        targets[0, :1] = 1  # 4 fg vs 12 bg, so balancing changes the weight
        mask = torch.ones_like(targets, dtype=torch.bool)
        ce = torch.nn.functional.cross_entropy(logits.movedim(-1, 1), targets, reduction="none")
        fg, bg = ce[0][targets[0] != 0].mean(), ce[0][targets[0] == 0].mean()
        expected = ((.5 * fg + .5 * bg) + ce[1].mean()) / 2  # example 1 is background only
        torch.testing.assert_close(balanced_loss(logits, targets, mask), expected)
        self.assertNotAlmostEqual(balanced_loss(logits, targets, mask).item(), masked_loss(logits, targets, mask).item())

    def test_linear_ranking_decay_keeps_locks(self):
        model = PixelTransformer(Config.from_dict({"model": {"d_model": 16, "layers": 1, "heads": 2}}).model, (16,))
        cfg = SampleConfig(steps=8, temperature=.7, ranking_temperature=1.0, ranking_schedule="linear")
        a = sample(model, size=16, count=2, seed=3, cfg=cfg)
        torch.testing.assert_close(a, sample(model, size=16, count=2, seed=3, cfg=cfg), rtol=0, atol=0)
        initial = torch.randint(4, (1, 16, 16))
        editable = torch.zeros_like(initial, dtype=torch.bool)
        editable[0, 4:9, 4:9] = True
        out = sample(model, size=16, initial=initial, editable=editable, cfg=cfg)
        self.assertTrue(torch.equal(out[~editable], initial[~editable]))
        with self.assertRaises(ValueError):
            Config.from_dict({"sample": {"ranking_schedule": "cosine"}})

    def test_global_prior_and_summary(self):
        with tempfile.TemporaryDirectory() as d:
            ds = ApprovedDataset(create_fixture(d, (16, 24)), (16, 24))
            prior = global_prior(ds)
            counts = torch.bincount(torch.cat([ds.grids[i].flatten() for i in ds.indices("train")]), minlength=4)
            torch.testing.assert_close(prior, ((counts + 1) / (counts + 1).sum()).float())
            model = PixelTransformer(Config.from_dict({"model": {"d_model": 16, "layers": 1, "heads": 2}}).model, (16, 24))
            report = conditional_report(model, ds, ratios=(.5, 1.0))
            self.assertEqual(len(report["rows"]), 2 * 3)  # random .5/1, contiguous .5, per size
            s = summarize(report)
            self.assertIn("bucket_frequency", s["priors"])
            self.assertIsNotNone(s["model"]["foreground_nll"])
            self.assertIsNotNone(s["balanced_nll"])

    def test_best_checkpoint_and_early_stop_resume(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            raw = tiny_config(directory).to_dict()
            raw["dataset"].update(loss="fg_bg_balanced", bucket_sampling="sqrt_count", hshift=1)
            raw["model"]["attention_dropout"] = 0.0
            raw["train"].update(select_metric="val_balanced_nll", patience=50)
            cfg = Config.from_dict(raw)
            create_fixture(Path(directory) / "data")
            full = train(cfg)
            self.assertTrue((Path(cfg.train.run_dir) / "best/model.safetensors").exists())
            _, _, meta = load_model(Path(cfg.train.run_dir) / "best")
            self.assertEqual(meta["best"]["step"], full["best_val_step"])
            rows = [json.loads(l) for l in (Path(cfg.train.run_dir) / "metrics.jsonl").read_text().splitlines()]
            self.assertTrue(all("loss_ce" in r for r in rows))
            self.assertIn("val_balanced_nll", rows[1])
            cfg.train.run_dir = str(Path(directory) / "resumed")
            partial = train(cfg, stop_after=2)
            resumed = train(cfg, resume=partial["checkpoint"])
            self.assertEqual((resumed["best_val_loss"], resumed["best_val_step"]),
                             (full["best_val_loss"], full["best_val_step"]))


if __name__ == "__main__":
    unittest.main()
