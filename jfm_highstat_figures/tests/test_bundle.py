#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CASES = {
    ("HS", 20.0, 0.2, "Figure5"),
    ("BGK", 20.0, 0.2, "Figure5"),
    ("SHAKHOV", 20.0, 0.2, "Figure5"),
    ("SHAKHOV", 20.0, 0.5, "Figure6"),
    ("HS", 30.0, 0.5, "Figure2d"),
}
EXPECTED_SHA = {
    "solver/JFM_hs_dsmc_quarter.py": "d234a93910403bcf4429ab2b20767a2a7d35349b27dab3016d257abfa5881f6f",
    "solver/JFM_bgk_shakhov_quarter.py": "c20d1a5fac0ab7e019e695a68e7ddecc64571f63102a8a957f61d766d6b4db55",
}


def read(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


for relative, expected in EXPECTED_SHA.items():
    actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    assert actual == expected, (relative, actual)

tables = {
    "high48_40m.csv": 3,
    "high80_40m.csv": 3,
    "low_40m.csv": 3,
    "low_20m_split.csv": 6,
}
all_seed_sets = []
for name, repetitions in tables.items():
    rows = read(ROOT / "cases" / name)
    assert len(rows) == 5 * repetitions, (name, len(rows))
    counts = Counter(
        (row["model"], float(row["kn"]), float(row["rt"]), row["figure"])
        for row in rows
    )
    assert set(counts) == EXPECTED_CASES
    assert set(counts.values()) == {repetitions}
    seeds = {int(row["seed"]) for row in rows}
    assert len(seeds) == repetitions
    all_seed_sets.append(seeds)

assert all_seed_sets[0].isdisjoint(all_seed_sets[1])
assert all_seed_sets[0].isdisjoint(all_seed_sets[2])
assert all_seed_sets[1].isdisjoint(all_seed_sets[2])

high_particle_time = 3 * 40_000_000 * 2_000_000
low_split_particle_time = 6 * 20_000_000 * 2_000_000
assert high_particle_time == low_split_particle_time

spec = importlib.util.spec_from_file_location(
    "summarize_highstat", ROOT / "tools" / "summarize_highstat.py"
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
stack = np.array([[[1.0]], [[3.0]], [[5.0]]])
mean, sd, se, n_eff = module.weighted_stats(stack, np.ones(3))
assert np.allclose(mean, 3.0)
assert np.allclose(sd, 2.0)
assert np.allclose(se, 2.0 / np.sqrt(3.0))
assert n_eff == 3.0
print("[OK] bundle tables, solver provenance, and weighted statistics validated")
