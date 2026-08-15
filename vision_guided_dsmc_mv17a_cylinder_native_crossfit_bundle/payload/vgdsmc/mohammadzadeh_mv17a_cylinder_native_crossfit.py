"""MV17A cylinder-native polar cross-fit for kinetic heat-flux denoising.

MV16B established that the cavity-frozen spectral map does not transfer
zero-shot to the Mach-10 cylinder.  MV17A follows the predeclared second path:
it learns a geometry-native, strongly regularised 2x2 Wiener residual operator
in polar normal/tangential coordinates.  It uses only the four already
completed MV11 trajectories and performs a strict 2+1+1 double cross-fit:

* two seeds train the prior and residual operator;
* one disjoint seed provides the B=3 observation;
* one disjoint seed provides the B=10 evaluation reference.

All twelve ordered observation/reference assignments are evaluated.  No DSMC
is rerun, no cavity label is used, and no fold may use its observation or
reference seed during fitting.  The result is explicitly retrospective and can
only authorise a separately frozen fresh-seed confirmation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


STAGE = "MV17A_Mohammadzadeh_cylinder_native_polar_crossfit"
STATUS = "locked_retrospective_cylinder_native_double_crossfit"
PROTOCOL_FILE = "mv17a_cylinder_native_crossfit_protocol.json"
RESULT_POINTER = "LAST_MOHAMMADZADEH_MV17A_CYLINDER_NATIVE_RESULT.env"
SEEDS = (20260813, 32452843, 49979687, 67867967)
DOMAIN = (-0.2, 0.65, 0.0, 0.4)
CYLINDER_CENTER = (0.1524, 0.0)
CYLINDER_RADIUS = 0.1524
POLAR_SHAPE = (128, 96)
TRANSFER_BINS = (4, 4)
RIDGE_FRACTION = 1.0
PHASE_CONTROL_SEED = 17012026
NEAR_WALL_THICKNESS_DIAMETERS = 0.05
EPS = 1.0e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialise {type(value).__name__}")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=_json_default)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_json_dumps(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(root: Path, name: str, files: Sequence[Path]) -> dict[str, Any]:
    root = Path(root).resolve()
    records = []
    for path in files:
        resolved = Path(path).resolve()
        records.append(
            {
                "path": str(resolved.relative_to(root)),
                "size_bytes": resolved.stat().st_size,
                "sha256": _sha256(resolved),
            }
        )
    value = {
        "stage": STAGE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": records,
    }
    _atomic_json(root / name, value)
    return value


def _verify_manifest(root: Path, name: str) -> dict[str, Any]:
    root = Path(root).resolve()
    path = root / name
    value = json.loads(path.read_text(encoding="utf-8"))
    raw_records = value["files"]
    if isinstance(raw_records, Mapping):
        records = [
            {"path": relative, **record}
            for relative, record in raw_records.items()
        ]
    else:
        records = raw_records
    for record in records:
        candidate = root / record["path"]
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        if candidate.stat().st_size != int(record["size_bytes"]):
            raise ValueError(f"size mismatch for {candidate}")
        if _sha256(candidate) != record["sha256"]:
            raise ValueError(f"hash mismatch for {candidate}")
    return value


def protocol_path() -> Path:
    value = (
        Path(__file__).resolve().parents[1]
        / "reference_data"
        / "mohammadzadeh_2012"
        / PROTOCOL_FILE
    )
    if not value.is_file():
        raise FileNotFoundError(value)
    return value


def locked_protocol() -> dict[str, Any]:
    value = json.loads(protocol_path().read_text(encoding="utf-8"))
    if value.get("stage") != STAGE or value.get("status") != STATUS:
        raise ValueError("MV17A protocol is absent or unlocked")
    contract = value["cylinder_native_contract"]
    if tuple(int(seed) for seed in contract["seeds"]) != SEEDS:
        raise ValueError("MV17A seed contract changed")
    if tuple(int(v) for v in contract["polar_shape"]) != POLAR_SHAPE:
        raise ValueError("MV17A polar shape changed")
    if tuple(int(v) for v in contract["transfer_bins"]) != TRANSFER_BINS:
        raise ValueError("MV17A transfer bins changed")
    if tuple(float(v) for v in contract["cylinder_center_m"]) != CYLINDER_CENTER:
        raise ValueError("MV17A cylinder centre changed")
    return value


def verify_contract() -> dict[str, Any]:
    value = locked_protocol()
    return {
        "stage": STAGE,
        "status": "MV17A_contract_verified",
        "protocol_sha256": _sha256(protocol_path()),
        "scientific_classification": value["scientific_classification"],
        "DSMC_rerun": False,
        "neural_training": False,
        "seed_count": len(SEEDS),
        "ordered_double_crossfit_fold_count": len(SEEDS) * (len(SEEDS) - 1),
        "cylinder_center_m": list(CYLINDER_CENTER),
        "cylinder_radius_m": CYLINDER_RADIUS,
        "corrects_MV16B_near_wall_origin_error": True,
    }


def double_crossfit_roles(count: int = 4) -> list[tuple[int, int, tuple[int, int]]]:
    if count != 4:
        raise ValueError("the locked double cross-fit requires exactly four seeds")
    folds = []
    for observation in range(count):
        for reference in range(count):
            if reference == observation:
                continue
            training = tuple(index for index in range(count) if index not in (observation, reference))
            if len(training) != 2:
                raise AssertionError("invalid 2+1+1 split")
            folds.append((observation, reference, training))
    return folds


def polar_geometry(
    x: np.ndarray,
    y: np.ndarray,
    *,
    domain: Sequence[float] = DOMAIN,
    center: Sequence[float] = CYLINDER_CENTER,
    radius: float = CYLINDER_RADIUS,
) -> dict[str, np.ndarray | float | int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("native coordinates must be equal one-dimensional arrays")
    xmin, xmax, ymin, ymax = (float(v) for v in domain)
    xc, yc = (float(v) for v in center)
    if abs(ymin - yc) > 1.0e-12:
        raise ValueError("MV17A expects the cylinder centre on the lower symmetry boundary")
    dx, dy = x - xc, y - yc
    radial = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    hit_x = np.where(
        cos_theta > 1.0e-12,
        (xmax - xc) / cos_theta,
        np.where(cos_theta < -1.0e-12, (xmin - xc) / cos_theta, np.inf),
    )
    hit_y = np.where(sin_theta > 1.0e-12, (ymax - yc) / sin_theta, np.inf)
    radial_max = np.minimum(hit_x, hit_y)
    rho_raw = (radial - radius) / np.maximum(radial_max - radius, EPS)
    if np.min(theta) < -1.0e-8 or np.max(theta) > math.pi + 1.0e-8:
        raise ValueError("native cells are outside the upper half-cylinder domain")
    if np.min(radial) < radius - 2.0e-3:
        raise ValueError("native mesh contains cells materially inside the cylinder")
    rho = np.clip(rho_raw, 0.0, 1.0)
    thickness = NEAR_WALL_THICKNESS_DIAMETERS * (2.0 * radius)
    near_wall = (radial >= radius) & (radial - radius <= thickness)
    return {
        "theta": theta,
        "rho": rho,
        "radial_m": radial,
        "radial_max_m": radial_max,
        "cos_theta": cos_theta,
        "sin_theta": sin_theta,
        "near_wall_mask": near_wall,
        "minimum_radius_m": float(np.min(radial)),
        "minimum_raw_rho": float(np.min(rho_raw)),
        "maximum_raw_rho": float(np.max(rho_raw)),
        "near_wall_cell_count": int(np.count_nonzero(near_wall)),
    }


class PolarMapper:
    """Reusable linear interpolation between native cells and a polar raster."""

    def __init__(self, theta: np.ndarray, rho: np.ndarray, shape: Sequence[int] = POLAR_SHAPE):
        from scipy.spatial import Delaunay, cKDTree

        self.theta = np.asarray(theta, dtype=np.float64)
        self.rho = np.asarray(rho, dtype=np.float64)
        self.shape = tuple(int(v) for v in shape)
        if self.shape[0] < 8 or self.shape[1] < 8:
            raise ValueError("polar raster is too small")
        self.theta_grid = np.linspace(0.0, math.pi, self.shape[0])
        self.rho_grid = np.linspace(0.0, 1.0, self.shape[1])
        tt, rr = np.meshgrid(self.theta_grid, self.rho_grid, indexing="ij")
        self.native_points = np.column_stack((self.theta, self.rho))
        self.grid_points = np.column_stack((tt.ravel(), rr.ravel()))
        self.triangulation = Delaunay(self.native_points)
        self.nearest = cKDTree(self.native_points).query(self.grid_points, k=1)[1]

    def to_grid(self, values: np.ndarray) -> np.ndarray:
        from scipy.interpolate import LinearNDInterpolator

        values = np.asarray(values, dtype=np.float64)
        if values.shape != self.theta.shape:
            raise ValueError("native field shape changed")
        result = np.asarray(
            LinearNDInterpolator(self.triangulation, values, fill_value=np.nan)(self.grid_points),
            dtype=np.float64,
        )
        missing = ~np.isfinite(result)
        result[missing] = values[self.nearest[missing]]
        return result.reshape(self.shape)

    def to_native(self, grid: np.ndarray) -> np.ndarray:
        from scipy.interpolate import RegularGridInterpolator

        grid = np.asarray(grid, dtype=np.float64)
        if grid.shape != self.shape:
            raise ValueError("polar grid shape changed")
        interpolator = RegularGridInterpolator(
            (self.theta_grid, self.rho_grid),
            grid,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        return np.asarray(interpolator(self.native_points), dtype=np.float64)


def cartesian_to_normal_tangential(
    qx: np.ndarray, qy: np.ndarray, cos_theta: np.ndarray, sin_theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    qx, qy = np.asarray(qx), np.asarray(qy)
    return qx * cos_theta + qy * sin_theta, -qx * sin_theta + qy * cos_theta


def normal_tangential_to_cartesian(
    qn: np.ndarray, qt: np.ndarray, cos_theta: np.ndarray, sin_theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    qn, qt = np.asarray(qn), np.asarray(qt)
    return qn * cos_theta - qt * sin_theta, qn * sin_theta + qt * cos_theta


def _dct(array: np.ndarray) -> np.ndarray:
    from scipy.fft import dctn

    return dctn(np.asarray(array, dtype=np.float64), axes=(-2, -1), norm="ortho")


def _idct(array: np.ndarray) -> np.ndarray:
    from scipy.fft import idctn

    return idctn(np.asarray(array, dtype=np.float64), axes=(-2, -1), norm="ortho")


def fit_binned_transfer(
    input_residual: np.ndarray,
    target_residual: np.ndarray,
    *,
    bins: Sequence[int] = TRANSFER_BINS,
    ridge_fraction: float = RIDGE_FRACTION,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit a regularised 2x2 residual map in fixed polar-frequency bins."""

    source = np.asarray(input_residual, dtype=np.float64)
    target = np.asarray(target_residual, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 4 or source.shape[1] != 2:
        raise ValueError("residual training arrays must have shape (cases,2,n_theta,n_rho)")
    _, _, ntheta, nrho = source.shape
    ntheta_bins, nrho_bins = (int(v) for v in bins)
    theta_edges = np.linspace(0, ntheta, ntheta_bins + 1, dtype=int)
    rho_edges = np.linspace(0, nrho, nrho_bins + 1, dtype=int)
    expanded = np.empty((ntheta, nrho, 2, 2), dtype=np.float64)
    blocks = np.empty((ntheta_bins, nrho_bins, 2, 2), dtype=np.float64)
    singular_values = []
    for ti, (left, right) in enumerate(zip(theta_edges[:-1], theta_edges[1:], strict=True)):
        for ri, (bottom, top) in enumerate(zip(rho_edges[:-1], rho_edges[1:], strict=True)):
            x = source[:, :, left:right, bottom:top].transpose(0, 2, 3, 1).reshape(-1, 2)
            y = target[:, :, left:right, bottom:top].transpose(0, 2, 3, 1).reshape(-1, 2)
            gram = x.T @ x
            ridge = float(ridge_fraction) * float(np.trace(gram)) / 2.0 + EPS
            matrix = np.linalg.solve(gram + ridge * np.eye(2), x.T @ y)
            u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
            singular = np.clip(singular, 0.0, 1.0)
            matrix = (u * singular) @ vt
            blocks[ti, ri] = matrix
            expanded[left:right, bottom:top] = matrix
            singular_values.extend(singular.tolist())
    blocks[0, 0] = np.eye(2)
    expanded[0, 0] = np.eye(2)
    audit = {
        "ridge_fraction": float(ridge_fraction),
        "bins": [ntheta_bins, nrho_bins],
        "minimum_singular_value": float(np.min(singular_values)),
        "maximum_singular_value": float(np.max(singular_values)),
        "mean_singular_value": float(np.mean(singular_values)),
        "DC_transfer": expanded[0, 0].tolist(),
    }
    return expanded, blocks, audit


def apply_transfer(prior: np.ndarray, raw: np.ndarray, transfer: np.ndarray) -> np.ndarray:
    prior = np.asarray(prior, dtype=np.float64)
    raw = np.asarray(raw, dtype=np.float64)
    if prior.shape != raw.shape or prior.ndim != 3 or prior.shape[0] != 2:
        raise ValueError("prior/raw coefficient arrays must have shape (2,n_theta,n_rho)")
    residual = (raw - prior).transpose(1, 2, 0)
    correction = np.einsum("...i,...ij->...j", residual, transfer)
    return prior + correction.transpose(2, 0, 1)


def phase_scramble_residual(residual: np.ndarray, seed: int = PHASE_CONTROL_SEED) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float64)
    if residual.ndim != 3 or residual.shape[0] != 2:
        raise ValueError("residual must have shape (2,n_theta,n_rho)")
    rng = np.random.default_rng(int(seed))
    signs = rng.choice((-1.0, 1.0), size=residual.shape[1:])
    signs[0, 0] = 1.0
    return residual * signs[None]


def area_weighted_nrmse(
    candidate: np.ndarray,
    target: np.ndarray,
    area: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    candidate, target, area = (np.asarray(v, dtype=np.float64) for v in (candidate, target, area))
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        candidate, target, area = candidate[selected], target[selected], area[selected]
    numerator = math.sqrt(float(np.sum(area * (candidate - target) ** 2) / np.sum(area)))
    denominator = math.sqrt(float(np.sum(area * target**2) / np.sum(area)))
    return numerator / max(denominator, EPS)


def preserve_cartesian_dc(
    qx: np.ndarray, qy: np.ndarray, raw_qx: np.ndarray, raw_qy: np.ndarray, area: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    qx, qy = np.asarray(qx, dtype=np.float64).copy(), np.asarray(qy, dtype=np.float64).copy()
    area = np.asarray(area, dtype=np.float64)
    total = float(np.sum(area))
    qx += float(np.sum(area * (raw_qx - qx)) / total)
    qy += float(np.sum(area * (raw_qy - qy)) / total)
    error_x = float(abs(np.sum(area * (qx - raw_qx)) / total))
    error_y = float(abs(np.sum(area * (qy - raw_qy)) / total))
    return qx, qy, {"qx_DC_absolute_error": error_x, "qy_DC_absolute_error": error_y}


def paired_statistics(method: Sequence[float], baseline: Sequence[float]) -> dict[str, Any]:
    from scipy.stats import t

    method = np.asarray(method, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    logs = np.log(np.maximum(method, EPS) / np.maximum(baseline, EPS))
    count = len(logs)
    improved = int(np.count_nonzero(logs < 0.0))
    one_sided = sum(math.comb(count, k) for k in range(improved, count + 1)) / (2**count)
    two_sided = min(1.0, 2.0 * one_sided)
    mean = float(np.mean(logs))
    if count > 1:
        half = float(t.ppf(0.975, count - 1) * np.std(logs, ddof=1) / math.sqrt(count))
    else:
        half = math.inf
    return {
        "n_independent_observation_seeds": count,
        "improved_seed_count": improved,
        "geometric_mean_ratio": float(math.exp(mean)),
        "t95_geometric_CI": [float(math.exp(mean - half)), float(math.exp(mean + half))],
        "exact_sign_test_one_sided_p": float(one_sided),
        "exact_sign_test_two_sided_p": float(two_sided),
        "minimum_attainable_one_sided_p_at_n": float(1.0 / (2**count)),
    }


def _load_native_sources(root: Path) -> tuple[list[dict[str, np.ndarray]], list[Path]]:
    paths = [root / f"cylinder_native_fields_seed_{seed}.npz" for seed in SEEDS]
    sources = []
    required = {
        "seed",
        "x_m",
        "y_m",
        "area_m2",
        "target_qx",
        "target_qy",
        "raw_b3_qx",
        "raw_b3_qy",
        "raw_b10_qx",
        "raw_b10_qy",
        "selected_qx",
        "selected_qy",
    }
    for seed, path in zip(SEEDS, paths, strict=True):
        with np.load(path, allow_pickle=False) as data:
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(f"{path} lacks {missing}")
            source = {name: np.asarray(data[name]) for name in data.files}
        if int(source["seed"]) != seed:
            raise ValueError(f"seed identity mismatch in {path}")
        sources.append(source)
    x0, y0, a0 = sources[0]["x_m"], sources[0]["y_m"], sources[0]["area_m2"]
    for source in sources[1:]:
        if not (
            np.array_equal(source["x_m"], x0)
            and np.array_equal(source["y_m"], y0)
            and np.array_equal(source["area_m2"], a0)
        ):
            raise ValueError("MV17A requires the common locked native DS2V mesh")
    return sources, paths


def _native_from_coefficients(
    coefficients: np.ndarray,
    mapper: PolarMapper,
    cosine: np.ndarray,
    sine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    normal, tangential = [mapper.to_native(_idct(coefficients[index])) for index in range(2)]
    return normal_tangential_to_cartesian(normal, tangential, cosine, sine)


def _cluster_means(rows: Sequence[Mapping[str, Any]], method: str, endpoint: str) -> tuple[list[float], list[float], dict[str, float]]:
    selected, baseline, ratios = [], [], {}
    for seed in SEEDS:
        group = [row for row in rows if int(row["observation_seed"]) == seed]
        method_mean = float(np.mean([float(row[f"{method}_{endpoint}_nrmse"]) for row in group]))
        baseline_mean = float(np.mean([float(row[f"raw_b10_{endpoint}_nrmse"]) for row in group]))
        selected.append(method_mean)
        baseline.append(baseline_mean)
        ratios[str(seed)] = method_mean / max(baseline_mean, EPS)
    return selected, baseline, ratios


def _masked_triangulation(x: np.ndarray, y: np.ndarray):
    import matplotlib.tri as mtri

    triangulation = mtri.Triangulation(x, y)
    triangles = triangulation.triangles
    cx = np.mean(x[triangles], axis=1) - CYLINDER_CENTER[0]
    cy = np.mean(y[triangles], axis=1) - CYLINDER_CENTER[1]
    long_edge = np.max(
        np.stack(
            [
                np.hypot(x[triangles[:, 0]] - x[triangles[:, 1]], y[triangles[:, 0]] - y[triangles[:, 1]]),
                np.hypot(x[triangles[:, 1]] - x[triangles[:, 2]], y[triangles[:, 1]] - y[triangles[:, 2]]),
                np.hypot(x[triangles[:, 2]] - x[triangles[:, 0]], y[triangles[:, 2]] - y[triangles[:, 0]]),
            ]
        ),
        axis=0,
    )
    triangulation.set_mask((np.hypot(cx, cy) < CYLINDER_RADIUS) | (long_edge > 0.03))
    return triangulation


def _plot_representative(
    output: Path,
    sources: Sequence[Mapping[str, np.ndarray]],
    fields: Mapping[str, np.ndarray],
    geometry: Mapping[str, Any],
    fold_index: int,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 9.4,
            "axes.linewidth": 0.85,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    test_index = int(fields["observation_index"][fold_index])
    x, y = sources[test_index]["x_m"], sources[test_index]["y_m"]
    target = fields["target_qy"][fold_index]
    order = ("target", "raw_b3", "cavity_zero_shot", "prior", "selected", "raw_b10")
    titles = (
        "Independent B10 reference",
        "Raw DSMC\n$B=3$",
        "Cavity-frozen transfer\n$B=3$",
        "Cylinder peer prior",
        "Cylinder-native cross-fit\n$B=3$",
        "Raw DSMC\n$B=10$",
    )
    values = {
        "target": target,
        "raw_b3": fields["raw_b3_qy"][fold_index],
        "cavity_zero_shot": fields["cavity_zero_shot_qy"][fold_index],
        "prior": fields["prior_qy"][fold_index],
        "selected": fields["selected_qy"][fold_index],
        "raw_b10": fields["raw_b10_qy"][fold_index],
    }
    field_limit = max(float(np.quantile(np.abs(np.concatenate([v.ravel() for v in values.values()])), 0.995)), EPS)
    target_rms = max(math.sqrt(float(np.mean(target**2))), EPS)
    errors = {name: 100.0 * (value - target) / target_rms for name, value in values.items()}
    error_limit = max(
        float(np.quantile(np.abs(np.concatenate([errors[name].ravel() for name in order[1:]])), 0.995)),
        1.0,
    )
    field_norm = TwoSlopeNorm(vmin=-field_limit, vcenter=0.0, vmax=field_limit)
    error_norm = TwoSlopeNorm(vmin=-error_limit, vcenter=0.0, vmax=error_limit)
    triangulation = _masked_triangulation(x, y)
    figure, axes = plt.subplots(2, 6, figsize=(18.0, 6.25), constrained_layout=True, sharex=True, sharey=True)
    field_artist = error_artist = None
    for column, (name, title) in enumerate(zip(order, titles, strict=True)):
        field_artist = axes[0, column].tricontourf(
            triangulation,
            values[name],
            levels=np.linspace(-field_limit, field_limit, 51),
            cmap="RdBu_r",
            norm=field_norm,
            extend="both",
        )
        axes[0, column].set_title(title)
        if name == "target":
            axes[1, column].set_facecolor("0.94")
            axes[1, column].text(0.5, 0.5, "Reference", transform=axes[1, column].transAxes, ha="center", va="center", color="0.45")
        else:
            error_artist = axes[1, column].tricontourf(
                triangulation,
                errors[name],
                levels=np.linspace(-error_limit, error_limit, 51),
                cmap="RdBu_r",
                norm=error_norm,
                extend="both",
            )
        axes[1, column].set_xlabel("$x$ [m]")
        for row in range(2):
            axes[row, column].set_aspect("equal")
    axes[0, 0].set_ylabel("$y$ [m]")
    axes[1, 0].set_ylabel("$y$ [m]")
    if field_artist is None or error_artist is None:
        raise RuntimeError("representative plot is incomplete")
    figure.colorbar(field_artist, ax=axes[0, :], shrink=0.86, label="normalised $q_y$")
    figure.colorbar(error_artist, ax=axes[1, :], shrink=0.86, label=r"$100\Delta q_y/\mathrm{RMS}(q_{y,ref})$ [\%]")
    figure.suptitle(
        f"MV17A strict cross-fit: observation seed {int(fields['observation_seed'][fold_index])}, "
        f"independent reference seed {int(fields['reference_seed'][fold_index])}"
    )
    names = []
    for suffix in ("png", "pdf"):
        path = output / f"mv17a_cylinder_native_qy_six_panel.{suffix}"
        figure.savefig(path, dpi=500 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
        names.append(path.name)
    plt.close(figure)

    theta = np.asarray(geometry["theta"])
    near = np.asarray(geometry["near_wall_mask"], dtype=bool)
    area = sources[test_index]["area_m2"]
    cosine, sine = np.asarray(geometry["cos_theta"]), np.asarray(geometry["sin_theta"])
    qx_fields = {
        "Reference": fields["target_qx"][fold_index],
        "Raw B3": fields["raw_b3_qx"][fold_index],
        "Peer prior": fields["prior_qx"][fold_index],
        "Cylinder-native B3": fields["selected_qx"][fold_index],
        "Raw B10": fields["raw_b10_qx"][fold_index],
    }
    qy_fields = {
        "Reference": fields["target_qy"][fold_index],
        "Raw B3": fields["raw_b3_qy"][fold_index],
        "Peer prior": fields["prior_qy"][fold_index],
        "Cylinder-native B3": fields["selected_qy"][fold_index],
        "Raw B10": fields["raw_b10_qy"][fold_index],
    }
    edges = np.linspace(0.0, math.pi, 61)
    centres = 0.5 * (edges[:-1] + edges[1:]) * 180.0 / math.pi
    figure, axis = plt.subplots(figsize=(9.2, 4.4), constrained_layout=True)
    for name in qx_fields:
        qn = qx_fields[name] * cosine + qy_fields[name] * sine
        profile = []
        for left, right in zip(edges[:-1], edges[1:], strict=True):
            mask = near & (theta >= left) & (theta < right)
            profile.append(float(np.sum(area[mask] * qn[mask]) / max(np.sum(area[mask]), EPS)))
        axis.plot(centres, profile, label=name)
    axis.set_xlabel(r"cylinder angle $\theta$ [deg]")
    axis.set_ylabel("near-wall normal heat flux $q_n$ (normalised)")
    axis.set_xlim(0.0, 180.0)
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, ncol=3)
    for suffix in ("png", "pdf"):
        path = output / f"mv17a_cylinder_native_near_wall_qn.{suffix}"
        figure.savefig(path, dpi=500 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
        names.append(path.name)
    plt.close(figure)
    return names


def analyze(mv16b_root: Path, output_root: Path) -> dict[str, Any]:
    source_root = Path(mv16b_root).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite MV17A output: {output}")
    verify_contract()
    _verify_manifest(source_root, "artifact_manifest.json")
    sources, native_paths = _load_native_sources(source_root)
    output.mkdir(parents=True)
    copied_protocol = output / PROTOCOL_FILE
    copied_protocol.write_bytes(protocol_path().read_bytes())
    lock = {
        "stage": STAGE,
        "status": "MV17A_sources_locked_before_cylinder_native_crossfit",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": _sha256(copied_protocol),
        "mv16b_output_root": str(source_root),
        "mv16b_summary_sha256": _sha256(source_root / "summary.json"),
        "mv16b_artifact_manifest_sha256": _sha256(source_root / "artifact_manifest.json"),
        "native_field_sha256": {path.name: _sha256(path) for path in native_paths},
        "DSMC_rerun": False,
        "neural_training": False,
        "cavity_model_reused": False,
        "heldout_reference_used_for_fitting": False,
        "scientific_classification": locked_protocol()["scientific_classification"],
        "corrected_cylinder_center_m": list(CYLINDER_CENTER),
        "MV16B_near_wall_origin_error_not_reused": True,
    }
    _atomic_json(output / "source_lock.json", lock)
    _write_manifest(output, "source_lock_manifest.json", [copied_protocol, output / "source_lock.json"])

    x, y, area = (np.asarray(sources[0][name], dtype=np.float64) for name in ("x_m", "y_m", "area_m2"))
    geometry = polar_geometry(x, y)
    mapper = PolarMapper(np.asarray(geometry["theta"]), np.asarray(geometry["rho"]))
    cosine, sine = np.asarray(geometry["cos_theta"]), np.asarray(geometry["sin_theta"])
    raw3_grid, raw10_grid = [], []
    for source in sources:
        raw3_nt = cartesian_to_normal_tangential(source["raw_b3_qx"], source["raw_b3_qy"], cosine, sine)
        raw10_nt = cartesian_to_normal_tangential(source["raw_b10_qx"], source["raw_b10_qy"], cosine, sine)
        raw3_grid.append([mapper.to_grid(component) for component in raw3_nt])
        raw10_grid.append([mapper.to_grid(component) for component in raw10_nt])
    raw3_coeff = _dct(np.asarray(raw3_grid))
    raw10_coeff = _dct(np.asarray(raw10_grid))

    records: list[dict[str, Any]] = []
    field_lists = {
        name: []
        for name in (
            "observation_index",
            "reference_index",
            "observation_seed",
            "reference_seed",
            "training_seeds",
            "target_qx",
            "target_qy",
            "raw_b3_qx",
            "raw_b3_qy",
            "raw_b10_qx",
            "raw_b10_qy",
            "cavity_zero_shot_qx",
            "cavity_zero_shot_qy",
            "prior_qx",
            "prior_qy",
            "selected_qx",
            "selected_qy",
            "phase_qx",
            "phase_qy",
            "transfer_blocks",
        )
    }
    dc_errors, transfer_audits = [], []
    for fold_index, (observation, reference, training) in enumerate(double_crossfit_roles()):
        left, right = training
        prior_coeff = np.mean(raw10_coeff[list(training)], axis=0, dtype=np.float64)
        input_residual = np.asarray(
            [raw3_coeff[left] - raw10_coeff[right], raw3_coeff[right] - raw10_coeff[left]]
        )
        target_residual = np.asarray(
            [raw10_coeff[left] - raw10_coeff[right], raw10_coeff[right] - raw10_coeff[left]]
        )
        transfer, blocks, transfer_audit = fit_binned_transfer(input_residual, target_residual)
        selected_coeff = apply_transfer(prior_coeff, raw3_coeff[observation], transfer)
        phase_coeff = prior_coeff + np.einsum(
            "...i,...ij->...j",
            phase_scramble_residual(raw3_coeff[observation] - prior_coeff).transpose(1, 2, 0),
            transfer,
        ).transpose(2, 0, 1)
        prior_qx, prior_qy = _native_from_coefficients(prior_coeff, mapper, cosine, sine)
        selected_qx, selected_qy = _native_from_coefficients(selected_coeff, mapper, cosine, sine)
        phase_qx, phase_qy = _native_from_coefficients(phase_coeff, mapper, cosine, sine)
        observed = sources[observation]
        target_source = sources[reference]
        selected_qx, selected_qy, selected_dc = preserve_cartesian_dc(
            selected_qx, selected_qy, observed["raw_b3_qx"], observed["raw_b3_qy"], area
        )
        phase_qx, phase_qy, phase_dc = preserve_cartesian_dc(
            phase_qx, phase_qy, observed["raw_b3_qx"], observed["raw_b3_qy"], area
        )
        dc_errors.extend(selected_dc.values())
        dc_errors.extend(phase_dc.values())
        transfer_audits.append({"fold": fold_index, **transfer_audit})
        target_qx, target_qy = target_source["raw_b10_qx"], target_source["raw_b10_qy"]
        target_qn = target_qx * cosine + target_qy * sine
        methods = {
            "raw_b3": (observed["raw_b3_qx"], observed["raw_b3_qy"]),
            "raw_b10": (observed["raw_b10_qx"], observed["raw_b10_qy"]),
            "cavity_zero_shot": (observed["selected_qx"], observed["selected_qy"]),
            "prior": (prior_qx, prior_qy),
            "selected": (selected_qx, selected_qy),
            "phase": (phase_qx, phase_qy),
        }
        row: dict[str, Any] = {
            "fold": fold_index,
            "observation_seed": int(SEEDS[observation]),
            "reference_seed": int(SEEDS[reference]),
            "training_seed_1": int(SEEDS[left]),
            "training_seed_2": int(SEEDS[right]),
        }
        for name, (qx_value, qy_value) in methods.items():
            qn_value = qx_value * cosine + qy_value * sine
            row[f"{name}_global_qy_nrmse"] = area_weighted_nrmse(qy_value, target_qy, area)
            row[f"{name}_near_wall_qn_nrmse"] = area_weighted_nrmse(
                qn_value, target_qn, area, np.asarray(geometry["near_wall_mask"])
            )
        records.append(row)
        values = {
            "observation_index": observation,
            "reference_index": reference,
            "observation_seed": SEEDS[observation],
            "reference_seed": SEEDS[reference],
            "training_seeds": np.asarray([SEEDS[left], SEEDS[right]], dtype=np.int64),
            "target_qx": target_qx,
            "target_qy": target_qy,
            "raw_b3_qx": observed["raw_b3_qx"],
            "raw_b3_qy": observed["raw_b3_qy"],
            "raw_b10_qx": observed["raw_b10_qx"],
            "raw_b10_qy": observed["raw_b10_qy"],
            "cavity_zero_shot_qx": observed["selected_qx"],
            "cavity_zero_shot_qy": observed["selected_qy"],
            "prior_qx": prior_qx,
            "prior_qy": prior_qy,
            "selected_qx": selected_qx,
            "selected_qy": selected_qy,
            "phase_qx": phase_qx,
            "phase_qy": phase_qy,
            "transfer_blocks": blocks,
        }
        for name, value in values.items():
            field_lists[name].append(value)

    method_names = ("raw_b3", "raw_b10", "cavity_zero_shot", "prior", "selected", "phase")
    endpoints = ("global_qy", "near_wall_qn")
    mean_nrmse = {
        method: {
            endpoint: float(np.mean([row[f"{method}_{endpoint}_nrmse"] for row in records]))
            for endpoint in endpoints
        }
        for method in method_names
    }
    ratios = {
        method: {
            endpoint: mean_nrmse[method][endpoint] / max(mean_nrmse["raw_b10"][endpoint], EPS)
            for endpoint in endpoints
        }
        for method in method_names
    }
    selected_seed_values, selected_baseline_values, per_seed_qy = _cluster_means(records, "selected", "global_qy")
    selected_seed_qn, selected_baseline_qn, per_seed_qn = _cluster_means(records, "selected", "near_wall_qn")
    statistics = {
        "global_qy": paired_statistics(selected_seed_values, selected_baseline_values),
        "near_wall_qn": paired_statistics(selected_seed_qn, selected_baseline_qn),
    }
    gates = {
        "all_twelve_disjoint_2_plus_1_plus_1_folds_present": len(records) == 12,
        "no_observation_or_reference_seed_used_for_fitting": all(
            row["observation_seed"] not in (row["training_seed_1"], row["training_seed_2"])
            and row["reference_seed"] not in (row["training_seed_1"], row["training_seed_2"])
            for row in records
        ),
        "correct_cylinder_center_verified_from_native_mesh": float(geometry["minimum_radius_m"]) >= CYLINDER_RADIUS,
        "corrected_near_wall_region_contains_at_least_400_cells": int(geometry["near_wall_cell_count"]) >= 400,
        "cartesian_DC_preserved": max(dc_errors) <= 1.0e-10,
        "selected_global_qy_mean_better_than_independent_Raw_B10": ratios["selected"]["global_qy"] < 1.0,
        "selected_near_wall_qn_mean_better_than_independent_Raw_B10": ratios["selected"]["near_wall_qn"] < 1.0,
        "selected_global_qy_better_for_every_observation_seed": all(value < 1.0 for value in per_seed_qy.values()),
        "selected_near_wall_qn_better_for_every_observation_seed": all(value < 1.0 for value in per_seed_qn.values()),
        "selected_beats_peer_prior_global_qy": ratios["selected"]["global_qy"] < ratios["prior"]["global_qy"],
        "selected_beats_peer_prior_near_wall_qn": ratios["selected"]["near_wall_qn"] < ratios["prior"]["near_wall_qn"],
        "phase_scramble_degrades_global_qy_by_at_least_one_percent": ratios["phase"]["global_qy"] >= 1.01 * ratios["selected"]["global_qy"],
        "phase_scramble_degrades_near_wall_qn_by_at_least_one_percent": ratios["phase"]["near_wall_qn"] >= 1.01 * ratios["selected"]["near_wall_qn"],
        "no_DSMC_rerun": True,
        "no_neural_training": True,
        "original_tU_over_D_30_warning_preserved": True,
        "not_reclassified_as_fresh_confirmation": True,
    }
    all_gates = all(gates.values())
    arrays = {name: np.asarray(values) for name, values in field_lists.items()}
    prediction_path = output / "mv17a_crossfit_fields.npz"
    np.savez_compressed(prediction_path, x_m=x, y_m=y, area_m2=area, **arrays)
    figures = _plot_representative(output, sources, arrays, geometry, fold_index=0)
    _write_rows(output / "mv17a_crossfit_metrics.csv", list(records[0]), records)
    _atomic_json(output / "mv17a_transfer_audit.json", {"geometry": geometry, "folds": transfer_audits})
    summary = {
        "stage": STAGE,
        "status": "complete_MV17A_cylinder_native_double_crossfit",
        "decision": (
            "MV17A_retrospective_cylinder_native_crossfit_supports_freezing_for_fresh_confirmation"
            if all_gates
            else "MV17A_cylinder_native_crossfit_does_not_authorize_fresh_confirmation"
        ),
        "scientific_classification": locked_protocol()["scientific_classification"],
        "all_gates_pass": all_gates,
        "gates": gates,
        "mean_nrmse": mean_nrmse,
        "ratios_to_independent_Raw_B10": ratios,
        "selected_per_observation_seed_ratios_to_Raw_B10": {
            "global_qy": per_seed_qy,
            "near_wall_qn": per_seed_qn,
        },
        "clustered_paired_statistics": statistics,
        "geometry_audit": {
            "cylinder_center_m": list(CYLINDER_CENTER),
            "cylinder_radius_m": CYLINDER_RADIUS,
            "minimum_native_radius_m": geometry["minimum_radius_m"],
            "near_wall_cell_count": geometry["near_wall_cell_count"],
            "MV16B_origin_based_qn_not_reused": True,
        },
        "fold_count": len(records),
        "training_seeds_per_fold": 2,
        "independent_observation_seeds": len(SEEDS),
        "DSMC_rerun": False,
        "neural_training": False,
        "fresh_confirmation_required": True,
        "minimum_fresh_seed_count_recommended": 5,
        "formal_p_less_than_0p05_claim_authorized": False,
        "original_tU_over_D_30_gate_pass": False,
        "figures": figures,
        "prediction_file": prediction_path.name,
    }
    _atomic_json(output / "summary.json", summary)
    return summary


def package_results(output_root: Path, return_directory: Path) -> dict[str, Any]:
    output = Path(output_root).resolve()
    returned = Path(return_directory).resolve()
    _verify_manifest(output, "source_lock_manifest.json")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    names = [
        PROTOCOL_FILE,
        "source_lock.json",
        "source_lock_manifest.json",
        "summary.json",
        "mv17a_crossfit_metrics.csv",
        "mv17a_transfer_audit.json",
        "mv17a_crossfit_fields.npz",
        *summary["figures"],
    ]
    accounting = output / "mv17a_slurm_accounting.psv"
    if accounting.is_file():
        names.append(accounting.name)
    files = [output / name for name in names]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    _write_manifest(output, "artifact_manifest.json", files)
    _verify_manifest(output, "artifact_manifest.json")
    returned.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = returned / f"MV17A_CYLINDER_NATIVE_CROSSFIT_BUNDLE_{timestamp}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as stream:
        for path in (*files, output / "artifact_manifest.json"):
            stream.write(path, arcname=path.name)
    digest = _sha256(archive)
    result = {
        "stage": STAGE,
        "decision": summary["decision"],
        "archive": str(archive),
        "archive_sha256": digest,
        "DSMC_rerun": False,
        "neural_training": False,
    }
    _atomic_json(output / "return.json", result)
    pointer = returned / RESULT_POINTER
    temporary = pointer.with_suffix(pointer.suffix + ".tmp")
    temporary.write_text(
        "\n".join(
            (
                f"MV17A_OUTPUT_ROOT={output}",
                f"MV17A_RESULT_ARCHIVE={archive}",
                f"MV17A_RESULT_ARCHIVE_SHA256={digest}",
                f"MV17A_DECISION={summary['decision']}",
                "MV17A_DSMC_RERUN=false",
                "MV17A_NEURAL_TRAINING=false",
                "",
            )
        ),
        encoding="utf-8",
    )
    temporary.replace(pointer)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify")
    run = subparsers.add_parser("run")
    run.add_argument("--mv16b-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    package = subparsers.add_parser("package")
    package.add_argument("--output-root", type=Path, required=True)
    package.add_argument("--return-directory", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        value = verify_contract()
    elif args.command == "run":
        value = analyze(args.mv16b_root, args.output_root)
    else:
        value = package_results(args.output_root, args.return_directory)
    print(_json_dumps(value))


if __name__ == "__main__":
    main()
