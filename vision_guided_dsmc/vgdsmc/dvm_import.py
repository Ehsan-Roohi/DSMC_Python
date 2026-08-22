from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
import numpy as np

ALIASES = {
    "x": {"x", "xc", "xcoord"},
    "y": {"y", "yc", "ycoord"},
    "rho": {"rho", "density", "r"},
    "u": {"u", "ux", "velx"},
    "v": {"v", "uy", "vely"},
    "T": {"t", "temp", "temperature"},
}


@dataclass(frozen=True)
class ImportedGrid:
    x: np.ndarray
    y: np.ndarray
    fields: dict[str, np.ndarray]


def _canonical(name: str) -> str | None:
    clean = re.sub(r"[^A-Za-z0-9_]", "", name).lower()
    for target, aliases in ALIASES.items():
        if clean in aliases:
            return target
    return None


def _numeric_rows(path: Path) -> tuple[list[str] | None, np.ndarray]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header: list[str] | None = None
    rows: list[list[float]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        upper = stripped.upper()
        if upper.startswith(("TITLE", "ZONE")):
            continue
        if upper.startswith("VARIABLES"):
            quoted = re.findall(r'"([^"]+)"', stripped)
            if quoted:
                header = quoted
            continue
        parts = re.split(r"[\s,]+", stripped)
        try:
            values = [float(item) for item in parts if item]
        except ValueError:
            if header is None:
                names = [item for item in parts if item]
                if len(names) >= 6:
                    header = names
            continue
        if values:
            rows.append(values)
    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("Inconsistent numeric column count")
    return header, np.asarray(rows, dtype=np.float64)


def import_dvm_table(path: str | Path, columns: str | None = None) -> ImportedGrid:
    path = Path(path)
    header, data = _numeric_rows(path)
    if columns:
        names = [item.strip() for item in columns.split(",")]
    elif header:
        names = header
    else:
        names = ["x", "y", "rho", "u", "v", "T"] + [f"extra_{i}" for i in range(data.shape[1] - 6)]
    if len(names) != data.shape[1]:
        raise ValueError(f"Column specification has {len(names)} names but data has {data.shape[1]} columns")
    mapping: dict[str, int] = {}
    for index, name in enumerate(names):
        canonical = _canonical(name)
        if canonical and canonical not in mapping:
            mapping[canonical] = index
    missing = [key for key in ("x", "y", "rho", "u", "v", "T") if key not in mapping]
    if missing:
        raise ValueError(f"Missing required columns: {missing}; use --columns")
    x_raw, y_raw = data[:, mapping["x"]], data[:, mapping["y"]]
    x_unique, y_unique = np.unique(x_raw), np.unique(y_raw)
    expected = len(x_unique) * len(y_unique)
    if expected != len(data):
        raise ValueError(f"Rows do not form a complete tensor grid: {len(data)} rows vs {expected} expected")
    ix = np.searchsorted(x_unique, x_raw)
    iy = np.searchsorted(y_unique, y_raw)
    fields: dict[str, np.ndarray] = {}
    for key in ("rho", "u", "v", "T"):
        field = np.full((len(y_unique), len(x_unique)), np.nan)
        field[iy, ix] = data[:, mapping[key]]
        if not np.isfinite(field).all():
            raise ValueError(f"Incomplete or non-finite field {key}")
        fields[key] = field
    if np.any(fields["rho"] <= 0.0) or np.any(fields["T"] <= 0.0):
        raise ValueError("rho and T must be positive")
    return ImportedGrid(x_unique, y_unique, fields)


def write_reference_npz(grid: ImportedGrid, output: str | Path, metadata: dict | None = None) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"x": grid.x, "y": grid.y, **grid.fields}
    np.savez_compressed(output, **payload)
    meta = {"shape": list(grid.fields["T"].shape), "fields": sorted(grid.fields), **(metadata or {})}
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return output
