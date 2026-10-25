from contextlib import redirect_stdout
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from megaloman.pixel_model.checkpoint import load_model
from megaloman.pixel_model.config import Config, SampleConfig
from megaloman.pixel_model.data import ApprovedDataset, create_fixture
from megaloman.pixel_model.evaluate import evaluate, foreground_distance, novelty_report
from megaloman.pixel_model.model import PixelTransformer
from megaloman.pixel_model.palette import MASK, read_indexed
from megaloman.pixel_model.sample import sample
from megaloman.pixel_model.train import train


def tiny_config(root):
    return Config.from_dict({
        "dataset": {"manifest": str(Path(root) / "data/manifest.json")},
        "model": {"d_model": 16, "layers": 1, "heads": 2, "mlp_ratio": 2, "dropout": .1},
        "train": {"device": "cpu", "precision": "fp32", "cpu_threads": 2,
                  "run_dir": str(Path(root) / "run"), "max_steps": 4, "warmup_steps": 1,
                  "batch_size": 2, "grad_accum_steps": 2, "eval_every": 2, "sample_every": 2,
                  "checkpoint_every": 2, "max_seconds": 120},
        "sample": {"steps": 3}})


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_sampler_seed_schedule_locks_and_small_regions(self):
        cfg = tiny_config("unused")
        torch.manual_seed(0)
        model = PixelTransformer(cfg.model)
        a, trace = sample(model, size=16, count=2, seed=5, cfg=cfg.sample, return_trace=True)
        b = sample(model, size=16, count=2, seed=5, cfg=cfg.sample)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(((a >= 0) & (a < 4)).all())
        self.assertEqual(len(trace), cfg.sample.steps + 1)
        for t, current in enumerate(trace[1:], 1):
            committed = 256 if t == cfg.sample.steps else 256 - math.floor(256 * math.cos(math.pi * t / (2 * cfg.sample.steps)))
            self.assertTrue(((current != MASK).flatten(1).sum(1) == committed).all())
        for previous, current in zip(trace, trace[1:]):
            committed = previous != MASK
            torch.testing.assert_close(previous[committed], current[committed])
        initial = torch.randint(4, (1, 16, 16))
        for n in (0, 1, 3):
            editable = torch.zeros_like(initial, dtype=torch.bool)
            editable.view(-1)[:n] = True
            result = sample(model, size=16, initial=initial, editable=editable, cfg=SampleConfig(steps=16))
            torch.testing.assert_close(result[~editable], initial[~editable], rtol=0, atol=0)
            self.assertFalse((result >= 4).any())

    def test_checkpoint_resume_exact_and_evaluation_exports(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            cfg = tiny_config(directory)
            create_fixture(Path(directory) / "data")
            result = train(cfg)
            continuous, _, _ = load_model(result["checkpoint"])
            cfg.train.run_dir = str(Path(directory) / "resumed")
            partial = train(cfg, stop_after=2)
            restored, _, meta = load_model(partial["checkpoint"])
            self.assertEqual(meta["step"], 2)
            final = train(cfg, resume=partial["checkpoint"])
            resumed, _, _ = load_model(final["checkpoint"])
            for key, value in continuous.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
            lines = [json.loads(line) for line in (Path(cfg.train.run_dir) / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([line["step"] for line in lines], [1, 2, 3, 4])
            self.assertTrue(all(np.isfinite(row["loss"]) for row in lines))
            report = evaluate(restored, cfg, output=Path(directory) / "evaluation", count=2)
            self.assertEqual(len(report["conditional"]["rows"]), 9)  # contiguous 100% counted once
            self.assertTrue((Path(directory) / "evaluation/16/blind-sheet.png").exists())
            self.assertTrue((Path(directory) / "evaluation/16/human-ratings.csv").exists())
            png = read_indexed(Path(directory) / "evaluation/16/samples/0000.png", 16)
            self.assertEqual(png.shape, (16, 16))
            self.assertEqual(list(Path(final["checkpoint"]).glob("*.pt")), [])
            with self.assertRaisesRegex(ValueError, "config mismatch"):
                cfg.train.lr *= 2
                train(cfg, resume=partial["checkpoint"])

    def test_dataset_mismatch_blocks_resume(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            cfg = tiny_config(directory)
            create_fixture(Path(directory) / "data")
            result = train(cfg, stop_after=2)
            manifest = Path(cfg.dataset.manifest)
            raw = json.loads(manifest.read_text())
            raw["approval_id"] = "new-snapshot"
            manifest.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "dataset/split/encoding mismatch"):
                train(cfg, resume=result["checkpoint"])

    def test_baselines_train_only_and_duplicate_controls(self):
        from megaloman.pixel_model.evaluate import fit_baselines
        with tempfile.TemporaryDirectory() as directory:
            ds = ApprovedDataset(create_fixture(directory))
            before = fit_baselines(ds, 16)["position"].clone()
            for i in ds.indices("val"):
                ds.grids[i].fill_(3)
            torch.testing.assert_close(before, fit_baselines(ds, 16)["position"])
            training = torch.stack([ds.grids[i] for i in ds.indices("train")])
            novelty = novelty_report(training[:2], training)
            self.assertEqual(novelty["exact_training_copy_rate"], 1)
            torch.testing.assert_close(foreground_distance(training[0], training[0]), torch.tensor(0.))

    def test_cli_registration_without_optional_dependencies(self):
        script = '''
import importlib.abc, sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'modal', 'safetensors'):
            raise ImportError('optional package unavailable')
sys.meta_path.insert(0, Reject())
from megaloman.cli import main
'''
        for args in (["ml", "--help"], ["ml", "pixel", "modal", "train", "-c", "configs/pixel_model/smoke.yaml", "--dry-run"]):
            process = subprocess.run([sys.executable, "-c", script + f"\nsys.argv = {args!r}\nmain()\n"],
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("pixel", process.stdout)

    def test_modal_validation_and_import_are_offline(self):
        from megaloman.pixel_model.modal_app import create_app, execution_plan, require_budget, safe_remote_path
        cfg = tiny_config("unused")
        plan = execution_plan(cfg, "train")
        self.assertEqual(plan["retries"], 0)
        with self.assertRaisesRegex(ValueError, "refreshed pricing"):
            require_budget(plan)
        for path in ("/absolute", "../escape", "runs/../../escape"):
            with self.assertRaises(ValueError):
                safe_remote_path(path)
        if importlib.util.find_spec("modal"):
            import modal
            # Any accidental hydration or launch fails the test before networking.
            with mock.patch.object(modal.App, "run", side_effect=AssertionError("remote launch")), \
                 mock.patch.object(modal.Volume, "hydrate", side_effect=AssertionError("volume hydration")):
                app, remote_train, remote_sample = create_app(cfg)
                self.assertEqual(app.name, "megaloman-pixel-transformer")
                self.assertIsNotNone(remote_train)
                self.assertIsNotNone(remote_sample)


if __name__ == "__main__":
    unittest.main()
