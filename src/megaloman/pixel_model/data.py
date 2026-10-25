"""Read-only approved snapshot contract and a clearly labeled synthetic fixture."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .config import canonical_json
from .palette import ENCODING, read_indexed, write_indexed


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ApprovedDataset:
    """Eager tiny/native grids; immutable path-independent snapshot fingerprint.

    No approval, splitting, normalization, or role assignment is performed here.
    External preparation supplies those decisions and the exact PNG byte hashes.
    """

    def __init__(self, manifest, resolutions=(16,), skip_other_sizes=False):
        self.path = Path(manifest).resolve()
        raw = json.loads(self.path.read_text())
        if raw.get("schema_version") != 1 or raw.get("encoding") != ENCODING:
            raise ValueError("unsupported manifest schema or encoding policy")
        if not raw.get("approval_id") or not raw.get("records"):
            raise ValueError("manifest requires an external approval_id and nonempty records")
        self.raw = raw
        self.records, self.grids = [], []
        groups, duplicates, ids = {}, {}, set()
        self.skipped = Counter()
        for row in raw["records"]:
            required = {"id", "path", "sha256", "identity", "animation", "split", "size"}
            if not required <= row.keys() or row["split"] not in ("train", "val", "test"):
                raise ValueError("each record needs id/path/sha256/identity/animation/split/size")
            if not row["identity"] or not row["animation"] or row["id"] in ids:
                raise ValueError("record IDs must be unique and identity/animation nonempty")
            ids.add(row["id"])
            if row["size"] not in resolutions and skip_other_sizes:
                self.skipped[row["size"]] += 1
                continue
            if row["size"] not in resolutions:
                raise ValueError(f"manifest resolution {row['size']} is not configured")
            rel = Path(row["path"])
            path = (self.path.parent / rel).resolve()
            if rel.is_absolute() or not path.is_relative_to(self.path.parent):
                raise ValueError("frame paths must stay within the manifest directory")
            if sha256(path) != row["sha256"]:
                raise ValueError(f"approved content hash mismatch: {row['id']}")
            grid = read_indexed(path, row["size"])
            for key, seen in ((row["identity"], groups),
                              ((grid.shape, grid.tobytes()), duplicates)):
                if key in seen and seen[key] != row["split"]:
                    raise ValueError("identity or exact categorical duplicate leaks across splits")
                seen[key] = row["split"]
            self.records.append(row)
            self.grids.append(torch.from_numpy(grid))
        for size in resolutions:
            for split in ("train", "val", "test"):
                if not self.indices(split, size):
                    raise ValueError(f"no {split} examples at {size}; external prep must supply all splits")
        self.fingerprint = hashlib.sha256(canonical_json(
            raw if not self.skipped else {"manifest": raw, "resolutions": sorted(resolutions)}).encode()).hexdigest()
        self.split_hash = hashlib.sha256(canonical_json(
            [(r["id"], r["identity"], r["split"]) for r in self.records]).encode()).hexdigest()
        self.encoding_hash = hashlib.sha256(canonical_json(ENCODING).encode()).hexdigest()

    def indices(self, split, size=None):
        return [i for i, r in enumerate(self.records)
                if r["split"] == split and (size is None or r["size"] == size)]

    def summary(self):
        return {"snapshot_sha256": self.fingerprint, "split_sha256": self.split_hash,
                "encoding_sha256": self.encoding_hash, "approval_id": self.raw["approval_id"],
                "synthetic": self.raw.get("synthetic", False), "frames": len(self.records),
                "counts": dict(Counter(f"{r['split']}/{r['size']}" for r in self.records)),
                "skipped_other_sizes": {str(k): v for k, v in sorted(self.skipped.items())},
                "families": {s: len({r['identity'] for r in self.records if r['split'] == s})
                             for s in ("train", "val", "test")},
                "class_counts": torch.bincount(torch.cat([g.flatten() for g in self.grids]), minlength=4).tolist()}


class BalancedBatchStream:
    """Size bucket, uniform identity, uniform animation, uniform frame, with replacement.

    The dedicated generator is checkpointed, so no loader workers/prefetch state leak.
    Bucket weights: uniform, or sqrt(train frames) per size. Optional training-only
    horizontal shift of up to `hshift` px (zero fill, no wrap, foreground kept in bounds).
    """

    def __init__(self, dataset, batch_size, seed, *, bucket_sampling="uniform", hshift=0, hflip=False):
        self.dataset, self.batch_size, self.hshift, self.hflip = dataset, batch_size, hshift, hflip
        self.generator = torch.Generator().manual_seed(seed)
        self.buckets = {}
        for size in sorted({r["size"] for r in dataset.records}):
            groups = defaultdict(lambda: defaultdict(list))
            for i in dataset.indices("train", size):
                r = dataset.records[i]
                groups[r["identity"]][r["animation"]].append(i)
            self.buckets[size] = groups
        sizes = list(self.buckets)
        counts = [len(dataset.indices("train", size)) for size in sizes]
        raw = [1.0] * len(sizes) if bucket_sampling == "uniform" else [c ** .5 for c in counts]
        self.policy = {"bucket_sampling": bucket_sampling, "hshift": hshift,
                       "weights": {str(z): w / sum(raw) for z, w in zip(sizes, raw)},
                       "train_counts": {str(z): c for z, c in zip(sizes, counts)}}
        self.weights = torch.tensor(raw, dtype=torch.float64)
        self.position = 0
        self.exposure = Counter()

    def _choice(self, seq):
        return seq[torch.randint(len(seq), (), generator=self.generator).item()]

    def _shift(self, grid):
        if self.hflip and torch.rand((), generator=self.generator).item() < .5:
            grid = grid.flip(-1)
        return self._hshift(grid)

    def _hshift(self, grid):
        if not self.hshift:
            return grid
        d = torch.randint(-self.hshift, self.hshift + 1, (), generator=self.generator).item()
        cols = (grid != 0).any(0).nonzero().flatten()
        if not d or not len(cols) or cols.min() + d < 0 or cols.max() + d >= grid.shape[1]:
            return grid
        out = torch.zeros_like(grid)
        if d > 0:
            out[:, d:] = grid[:, :-d]
        else:
            out[:, :d] = grid[:, -d:]
        return out

    def next(self):
        size = list(self.buckets)[torch.multinomial(self.weights, 1, generator=self.generator).item()]
        groups = self.buckets[size]
        indices = []
        for _ in range(self.batch_size):
            animations = groups[self._choice(list(groups))]
            indices.append(self._choice(animations[self._choice(list(animations))]))
        self.position += self.batch_size
        for i in indices:
            self.exposure[self.dataset.records[i]["identity"]] += 1
        return torch.stack([self._shift(self.dataset.grids[i]) for i in indices])

    def state_dict(self):
        return {"generator": self.generator.get_state(), "position": self.position,
                "policy": json.dumps(self.policy, sort_keys=True), "exposure": json.dumps(dict(self.exposure))}

    def load_state_dict(self, state):
        if "policy" in state and json.loads(state["policy"]) != self.policy:
            raise ValueError("resume bucket policy/counts mismatch")
        self.generator.set_state(state["generator"])
        self.position = state["position"]
        self.exposure = Counter(json.loads(state.get("exposure", "{}")))


def create_fixture(root, resolutions=(16,)):
    """Procedural blobs for plumbing only, with explicitly frozen synthetic splits."""
    root = Path(root)
    if (root / "manifest.json").exists():
        raise ValueError("fixture manifest already exists; choose a fresh directory")
    records = []
    for size in resolutions:
        y, x = np.mgrid[:size, :size]
        for family in range(12):
            split = "train" if family < 8 else "val" if family < 10 else "test"
            for frame in range(2):
                grid = np.zeros((size, size), dtype=np.int64)
                cx, cy = size // 2 + family % 3 - 1, size // 2 + frame - 1
                rx, ry = 3 + family % 4, 3 + family // 4
                inside = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1
                inner = ((x - cx) / max(1, rx - 1)) ** 2 + ((y - cy) / max(1, ry - 1)) ** 2 <= 1
                grid[inside] = 1
                grid[inner] = 2
                grid[inner & (x >= cx) & (y < cy)] = 3
                grid[cy - 1, cx - 1] = 1
                name = f"frames/{size}-{family}-{frame}.png"
                write_indexed(root / name, grid)
                records.append({"id": f"{size}-{family}-{frame}", "path": name,
                                "sha256": sha256(root / name), "size": size,
                                "identity": f"synthetic-{family}", "animation": "wobble", "split": split})
    raw = {"schema_version": 1, "approval_id": "synthetic-fixture-not-human-approved",
           "synthetic": True, "encoding": ENCODING, "records": records}
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(raw, indent=2) + "\n")
    return root / "manifest.json"
