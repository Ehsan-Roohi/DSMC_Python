"""Prospective fresh-seed confirmation of the frozen MV17A cylinder estimator.

MV17B freezes one final cylinder-native polar/DCT estimator using only the four
historical development trajectories that passed MV17A.  It then evaluates the
unchanged estimator on six predeclared, independent observation/reference
pairs (twelve new Bird/DS2V trajectories).  The reference member of a pair is
never used for prediction.  It supplies only the independent Raw-B10 target.

The two co-primary endpoints are native-cell area-weighted global q_y and the
cylinder-centred near-wall q_n.  Six all-improved pairs give an exact one-sided
sign-test p=1/64; Holm correction across the two endpoints gives 0.03125.
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


STAGE = "MV17B_Mohammadzadeh_fresh_cylinder_native_confirmation"
STATUS = "locked_before_fresh_cylinder_DSMC"
PROTOCOL_FILE = "mv17b_fresh_cylinder_confirmation_protocol.json"
RESULT_POINTER = "LAST_MOHAMMADZADEH_MV17B_FRESH_CYLINDER_RESULT.env"
DEVELOPMENT_SEEDS = (20260813, 32452843, 49979687, 67867967)
PAIRS = (
    ("pair_01", 171701, 171702),
    ("pair_02", 171703, 171704),
    ("pair_03", 171705, 171706),
    ("pair_04", 171707, 171708),
    ("pair_05", 171709, 171710),
    ("pair_06", 171711, 171712),
)
B3_NOUT = (100, 108, 116)
B10_NOUT = (101, 102, 103, 104, 105, 109, 110, 111, 112, 113)
GUARD_NOUT = (106, 107, 114, 115)
LOCKED_NOUT = tuple(range(100, 117))
QX_INDEX = 2
QY_INDEX = 3
PHASE_SEEDS = (171721, 171722, 171723, 171724, 171725, 171726)
EPS = 1.0e-12


def _mv16a_module():
    from . import mohammadzadeh_mv16a_frozen_cylinder_transfer as mv16a

    return mv16a


def _mv17a_module():
    from . import mohammadzadeh_mv17a_cylinder_native_crossfit as mv17a

    return mv17a


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        allow_nan=False,
        default=_json_default,
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_json_dumps(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_rows(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(root: Path, name: str, files: Sequence[Path]) -> dict[str, Any]:
    root = Path(root).resolve()
    records = {}
    for candidate in files:
        path = Path(candidate).resolve()
        records[str(path.relative_to(root))] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    value = {"stage": STAGE, "files": records}
    _atomic_json(root / name, value)
    return value


def _verify_manifest(root: Path, name: str) -> dict[str, Any]:
    root = Path(root).resolve()
    path = root / name
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for relative, record in manifest["files"].items():
        candidate = root / relative
        if (
            not candidate.is_file()
            or candidate.stat().st_size != int(record["size_bytes"])
            or _sha256(candidate) != record["sha256"]
        ):
            raise ValueError(f"MV17B manifest verification failed: {candidate}")
    return manifest


def protocol_path() -> Path:
    path = (
        Path(__file__).resolve().parents[1]
        / "reference_data"
        / "mohammadzadeh_2012"
        / PROTOCOL_FILE
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def locked_protocol() -> dict[str, Any]:
    protocol = json.loads(protocol_path().read_text(encoding="utf-8"))
    if protocol.get("stage") != STAGE or protocol.get("status") != STATUS:
        raise ValueError("MV17B protocol is absent or unlocked")
    contract = protocol["fresh_pair_contract"]
    pairs = tuple(
        (str(row["pair_id"]), int(row["observation_seed"]), int(row["reference_seed"]))
        for row in contract["pairs"]
    )
    if pairs != PAIRS or int(contract["pair_count"]) != len(PAIRS):
        raise ValueError("MV17B pair identities differ from the locked protocol")
    if tuple(int(v) for v in contract["observation_B3_nout"]) != B3_NOUT:
        raise ValueError("MV17B B3 split changed")
    if tuple(int(v) for v in contract["observation_Raw_B10_nout"]) != B10_NOUT:
        raise ValueError("MV17B observation B10 split changed")
    if tuple(int(v) for v in contract["independent_reference_B10_nout"]) != B10_NOUT:
        raise ValueError("MV17B reference B10 split changed")
    if tuple(int(v) for v in contract["unused_guard_nout"]) != GUARD_NOUT:
        raise ValueError("MV17B guard split changed")
    if set(B3_NOUT) & set(B10_NOUT) or set(B3_NOUT) & set(GUARD_NOUT):
        raise ValueError("MV17B locked block sets overlap")
    if set(B10_NOUT) & set(GUARD_NOUT):
        raise ValueError("MV17B locked block sets overlap")
    if set(B3_NOUT + B10_NOUT + GUARD_NOUT) != set(LOCKED_NOUT):
        raise ValueError("MV17B locked block partition is incomplete")
    seeds = [seed for _, observation, reference in PAIRS for seed in (observation, reference)]
    if len(seeds) != len(set(seeds)) or set(seeds) & set(DEVELOPMENT_SEEDS):
        raise ValueError("MV17B fresh seeds are not unique and disjoint")
    return protocol


def verify_contract() -> dict[str, Any]:
    protocol = locked_protocol()
    return {
        "stage": STAGE,
        "status": "MV17B_contract_verified",
        "protocol_sha256": _sha256(protocol_path()),
        "pair_count": len(PAIRS),
        "trajectory_count": 2 * len(PAIRS),
        "fresh_seeds": [seed for _, left, right in PAIRS for seed in (left, right)],
        "B3_nout": list(B3_NOUT),
        "B10_nout": list(B10_NOUT),
        "required_last_nout": max(LOCKED_NOUT),
        "minimum_unadjusted_one_sided_p": 1.0 / 2**len(PAIRS),
        "minimum_two_endpoint_Holm_p": 2.0 / 2**len(PAIRS),
        "neural_training": False,
        "fresh_parameter_selection": False,
        "classification": protocol["scientific_classification"],
    }


def _verify_external_artifact(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_files = manifest["files"]
    # MV16B used a path-keyed mapping; MV17A deliberately switched to a
    # list-of-records manifest.  Both are immutable published formats.
    if isinstance(raw_files, Mapping):
        records = [(relative, record) for relative, record in raw_files.items()]
    elif isinstance(raw_files, list):
        records = [(str(record["path"]), record) for record in raw_files]
    else:
        raise ValueError(f"unsupported external artifact manifest: {manifest_path}")
    for relative, record in records:
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or _sha256(path) != record["sha256"]
        ):
            raise ValueError(f"development artifact changed: {path}")
    return manifest


def freeze_model(mv16b_root: Path, mv17a_root: Path, output_root: Path) -> dict[str, Any]:
    """Fit and hash the final estimator before any fresh trajectory can start."""

    mv16b_root = Path(mv16b_root).resolve()
    mv17a_root = Path(mv17a_root).resolve()
    output_root = Path(output_root).resolve()
    frozen = output_root / "frozen"
    if frozen.exists():
        raise FileExistsError(f"refusing to overwrite an MV17B freeze: {frozen}")
    verify_contract()
    _verify_external_artifact(mv16b_root)
    _verify_external_artifact(mv17a_root)
    mv17a_summary = json.loads((mv17a_root / "summary.json").read_text(encoding="utf-8"))
    required_decision = locked_protocol()["development_lock"]["required_MV17A_decision"]
    if (
        mv17a_summary.get("decision") != required_decision
        or not bool(mv17a_summary.get("all_gates_pass"))
    ):
        raise ValueError("MV17B requires the successful, all-gates-passed MV17A result")

    mv17a = _mv17a_module()
    sources, native_paths = mv17a._load_native_sources(mv16b_root)
    if tuple(int(np.asarray(source["seed"]).item()) for source in sources) != DEVELOPMENT_SEEDS:
        raise ValueError("MV17B development seed identity changed")
    x = np.asarray(sources[0]["x_m"], dtype=np.float64)
    y = np.asarray(sources[0]["y_m"], dtype=np.float64)
    area = np.asarray(sources[0]["area_m2"], dtype=np.float64)
    geometry = mv17a.polar_geometry(x, y)
    mapper = mv17a.PolarMapper(np.asarray(geometry["theta"]), np.asarray(geometry["rho"]))
    cosine = np.asarray(geometry["cos_theta"])
    sine = np.asarray(geometry["sin_theta"])

    raw3_grid, raw10_grid = [], []
    for source in sources:
        raw3_nt = mv17a.cartesian_to_normal_tangential(
            source["raw_b3_qx"], source["raw_b3_qy"], cosine, sine
        )
        raw10_nt = mv17a.cartesian_to_normal_tangential(
            source["raw_b10_qx"], source["raw_b10_qy"], cosine, sine
        )
        raw3_grid.append([mapper.to_grid(component) for component in raw3_nt])
        raw10_grid.append([mapper.to_grid(component) for component in raw10_nt])
    raw3_coeff = mv17a._dct(np.asarray(raw3_grid))
    raw10_coeff = mv17a._dct(np.asarray(raw10_grid))
    prior_coeff = np.mean(raw10_coeff, axis=0, dtype=np.float64)
    ordered = [(i, j) for i in range(4) for j in range(4) if i != j]
    input_residual = np.asarray([raw3_coeff[i] - raw10_coeff[j] for i, j in ordered])
    target_residual = np.asarray([raw10_coeff[i] - raw10_coeff[j] for i, j in ordered])
    transfer, blocks, audit = mv17a.fit_binned_transfer(input_residual, target_residual)

    frozen.mkdir(parents=True)
    copied_protocol = frozen / PROTOCOL_FILE
    copied_protocol.write_bytes(protocol_path().read_bytes())
    model_path = frozen / "mv17b_frozen_cylinder_model.npz"
    np.savez_compressed(
        model_path,
        x_m=x,
        y_m=y,
        area_m2=area,
        prior_coeff=prior_coeff,
        transfer=transfer,
        transfer_blocks=blocks,
        development_seeds=np.asarray(DEVELOPMENT_SEEDS, dtype=np.int64),
        ordered_training_pairs=np.asarray(ordered, dtype=np.int64),
    )
    model_hash = _sha256(model_path)
    lock = {
        "stage": STAGE,
        "status": "MV17B_model_frozen_before_fresh_DSMC",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": _sha256(copied_protocol),
        "model_file": model_path.name,
        "model_sha256": model_hash,
        "mv17a_output_root": str(mv17a_root),
        "mv17a_summary_sha256": _sha256(mv17a_root / "summary.json"),
        "mv17a_artifact_manifest_sha256": _sha256(mv17a_root / "artifact_manifest.json"),
        "mv16b_output_root": str(mv16b_root),
        "mv16b_artifact_manifest_sha256": _sha256(mv16b_root / "artifact_manifest.json"),
        "development_native_field_sha256": {path.name: _sha256(path) for path in native_paths},
        "development_seed_count": len(DEVELOPMENT_SEEDS),
        "ordered_transfer_training_pair_count": len(ordered),
        "transfer_audit": audit,
        "geometry_audit": {
            "cylinder_center_m": list(mv17a.CYLINDER_CENTER),
            "cylinder_radius_m": mv17a.CYLINDER_RADIUS,
            "minimum_native_radius_m": geometry["minimum_radius_m"],
            "near_wall_cell_count": geometry["near_wall_cell_count"],
        },
        "fresh_DSMC_files_present_at_freeze": False,
        "fresh_parameter_selection": False,
        "neural_training": False,
    }
    _atomic_json(frozen / "model_lock.json", lock)
    _write_manifest(
        output_root,
        "frozen/source_lock_manifest.json",
        [copied_protocol, model_path, frozen / "model_lock.json"],
    )
    (frozen / "FROZEN_MODEL_PASS").write_text(
        f"MV17B frozen before fresh DSMC\nmodel_sha256={model_hash}\n",
        encoding="utf-8",
    )
    return lock


def verify_frozen_model(output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    _verify_manifest(output_root, "frozen/source_lock_manifest.json")
    lock = json.loads((output_root / "frozen/model_lock.json").read_text(encoding="utf-8"))
    model = output_root / "frozen" / lock["model_file"]
    if lock.get("status") != "MV17B_model_frozen_before_fresh_DSMC":
        raise ValueError("invalid MV17B model lock")
    if _sha256(model) != lock["model_sha256"]:
        raise ValueError("MV17B frozen model hash changed")
    return lock


def _case_id(pair_id: str, role: str) -> str:
    return f"{pair_id}_{role}"


def _moment_path(campaign: Path, pair_id: str, role: str, nout: int) -> Path:
    return (
        Path(campaign)
        / "cases"
        / _case_id(pair_id, role)
        / "results"
        / "moments"
        / f"MV11_MOMENTS_NOUT{nout:04d}.DAT"
    )


def _load_case_fields(
    campaign: Path, pair_id: str, role: str, nouts: Sequence[int]
) -> tuple[dict[str, np.ndarray], list[Path]]:
    mv16a = _mv16a_module()
    paths = [_moment_path(campaign, pair_id, role, nout) for nout in nouts]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata, additive = mv16a.aggregate_moment_files(paths)
    return mv16a.reconstruct_fields(metadata, additive), paths


def _assert_same_mesh(reference: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray]) -> None:
    for name in ("x_m", "y_m", "area_m2"):
        if not np.allclose(reference[name], candidate[name], rtol=0.0, atol=2.0e-10):
            raise ValueError(f"MV17B native mesh mismatch in {name}")


def _apply_frozen(
    fields: Mapping[str, np.ndarray], model: Mapping[str, np.ndarray], phase_seed: int
) -> dict[str, np.ndarray | dict[str, float]]:
    mv17a = _mv17a_module()
    x, y, area = (
        np.asarray(fields[name], dtype=np.float64) for name in ("x_m", "y_m", "area_m2")
    )
    if not (
        np.allclose(x, model["x_m"], rtol=0.0, atol=2.0e-10)
        and np.allclose(y, model["y_m"], rtol=0.0, atol=2.0e-10)
        and np.allclose(area, model["area_m2"], rtol=0.0, atol=2.0e-10)
    ):
        raise ValueError("fresh MV17B mesh differs from the frozen development mesh")
    geometry = mv17a.polar_geometry(x, y)
    mapper = mv17a.PolarMapper(np.asarray(geometry["theta"]), np.asarray(geometry["rho"]))
    cosine = np.asarray(geometry["cos_theta"])
    sine = np.asarray(geometry["sin_theta"])
    raw_qx = np.asarray(fields["outputs"][:, QX_INDEX], dtype=np.float64)
    raw_qy = np.asarray(fields["outputs"][:, QY_INDEX], dtype=np.float64)
    raw_nt = mv17a.cartesian_to_normal_tangential(raw_qx, raw_qy, cosine, sine)
    raw_coeff = mv17a._dct(
        np.asarray([[mapper.to_grid(component) for component in raw_nt]], dtype=np.float64)
    )[0]
    prior_coeff = np.asarray(model["prior_coeff"], dtype=np.float64)
    transfer = np.asarray(model["transfer"], dtype=np.float64)
    selected_coeff = mv17a.apply_transfer(prior_coeff, raw_coeff, transfer)
    phase_coeff = prior_coeff + np.einsum(
        "...i,...ij->...j",
        mv17a.phase_scramble_residual(raw_coeff - prior_coeff, seed=phase_seed).transpose(1, 2, 0),
        transfer,
    ).transpose(2, 0, 1)
    prior_qx, prior_qy = mv17a._native_from_coefficients(
        prior_coeff, mapper, cosine, sine
    )
    selected_qx, selected_qy = mv17a._native_from_coefficients(
        selected_coeff, mapper, cosine, sine
    )
    phase_qx, phase_qy = mv17a._native_from_coefficients(
        phase_coeff, mapper, cosine, sine
    )
    selected_qx, selected_qy, selected_dc = mv17a.preserve_cartesian_dc(
        selected_qx, selected_qy, raw_qx, raw_qy, area
    )
    phase_qx, phase_qy, phase_dc = mv17a.preserve_cartesian_dc(
        phase_qx, phase_qy, raw_qx, raw_qy, area
    )
    return {
        "prior_qx": prior_qx,
        "prior_qy": prior_qy,
        "selected_qx": selected_qx,
        "selected_qy": selected_qy,
        "phase_qx": phase_qx,
        "phase_qy": phase_qy,
        "selected_dc": selected_dc,
        "phase_dc": phase_dc,
        "near_wall_mask": np.asarray(geometry["near_wall_mask"], dtype=bool),
        "cos_theta": cosine,
        "sin_theta": sine,
        "theta": np.asarray(geometry["theta"]),
    }


def _holm_adjust(raw_p: Mapping[str, float]) -> dict[str, float]:
    names = list(raw_p)
    order = sorted(range(len(names)), key=lambda index: float(raw_p[names[index]]))
    adjusted = [0.0] * len(names)
    running = 0.0
    total = len(names)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * float(raw_p[names[index]]))
        adjusted[index] = min(1.0, running)
    return {name: float(adjusted[index]) for index, name in enumerate(names)}


def _plot_results(
    output: Path, arrays: Mapping[str, np.ndarray], records: Sequence[Mapping[str, Any]]
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    mv17a = _mv17a_module()
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
    index = 0
    order = ("target", "raw_b3", "prior", "selected", "phase", "raw_b10")
    titles = (
        "Independent Raw DSMC\n$B=10$ reference",
        "Raw DSMC\n$B=3$",
        "Frozen cylinder prior",
        "Frozen MV17B\n$B=3$",
        "Phase-scrambled control",
        "Raw DSMC\n$B=10$",
    )
    values = {name: np.asarray(arrays[f"{name}_qy"])[index] for name in order}
    x, y = np.asarray(arrays["x_m"]), np.asarray(arrays["y_m"])
    triangulation = mv17a._masked_triangulation(x, y)
    field_limit = max(
        float(np.quantile(np.abs(np.concatenate([value.ravel() for value in values.values()])), 0.995)),
        EPS,
    )
    target_rms = max(math.sqrt(float(np.mean(values["target"] ** 2))), EPS)
    errors = {
        name: 100.0 * (value - values["target"]) / target_rms for name, value in values.items()
    }
    error_limit = max(
        float(np.quantile(np.abs(np.concatenate([errors[name].ravel() for name in order[1:]])), 0.995)),
        1.0,
    )
    field_norm = TwoSlopeNorm(vmin=-field_limit, vcenter=0.0, vmax=field_limit)
    error_norm = TwoSlopeNorm(vmin=-error_limit, vcenter=0.0, vmax=error_limit)
    figure, axes = plt.subplots(
        2, 6, figsize=(18.2, 6.3), constrained_layout=True, sharex=True, sharey=True
    )
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
            axes[1, column].text(
                0.5,
                0.5,
                "Reference",
                transform=axes[1, column].transAxes,
                ha="center",
                va="center",
                color="0.45",
            )
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
        raise RuntimeError("MV17B representative figure is incomplete")
    figure.colorbar(field_artist, ax=axes[0, :], shrink=0.86, label="normalised $q_y$")
    figure.colorbar(
        error_artist,
        ax=axes[1, :],
        shrink=0.86,
        label=r"$100\Delta q_y/\mathrm{RMS}(q_{y,ref})$ [\%]",
    )
    figure.suptitle(
        f"MV17B fresh confirmation: {records[index]['pair_id']}, "
        f"observation seed {records[index]['observation_seed']}, "
        f"reference seed {records[index]['reference_seed']}"
    )
    names: list[str] = []
    for suffix in ("png", "pdf"):
        path = output / f"mv17b_fresh_cylinder_qy_six_panel.{suffix}"
        figure.savefig(path, dpi=500 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
        names.append(path.name)
    plt.close(figure)

    labels = [str(row["pair_id"]).replace("pair_", "P") for row in records]
    xindex = np.arange(len(labels), dtype=float)
    width = 0.36
    qy_ratio = [float(row["selected_global_qy_ratio_to_Raw_B10"]) for row in records]
    qn_ratio = [float(row["selected_near_wall_qn_ratio_to_Raw_B10"]) for row in records]
    figure, axis = plt.subplots(figsize=(8.4, 4.5), constrained_layout=True)
    axis.bar(xindex - width / 2, qy_ratio, width, label="global $q_y$")
    axis.bar(xindex + width / 2, qn_ratio, width, label="near-wall $q_n$")
    axis.axhline(1.0, color="black", linewidth=1.1, linestyle="--", label="Raw $B=10$")
    axis.set_xticks(xindex, labels)
    axis.set_ylabel("NRMSE ratio to Raw DSMC $B=10$")
    axis.set_xlabel("independent fresh observation/reference pair")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=3)
    for suffix in ("png", "pdf"):
        path = output / f"mv17b_fresh_pair_error_ratios.{suffix}"
        figure.savefig(path, dpi=500 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
        names.append(path.name)
    plt.close(figure)

    theta = np.asarray(arrays["theta"])[index]
    near = np.asarray(arrays["near_wall_mask"], dtype=bool)[index]
    area = np.asarray(arrays["area_m2"])
    cosine = np.asarray(arrays["cos_theta"])[index]
    sine = np.asarray(arrays["sin_theta"])[index]
    edges = np.linspace(0.0, math.pi, 61)
    centres = 0.5 * (edges[:-1] + edges[1:]) * 180.0 / math.pi
    figure, axis = plt.subplots(figsize=(9.2, 4.4), constrained_layout=True)
    profile_names = (
        ("Independent B10 reference", "target"),
        ("Raw B3", "raw_b3"),
        ("Frozen prior", "prior"),
        ("Frozen MV17B B3", "selected"),
        ("Raw B10", "raw_b10"),
    )
    for label, name in profile_names:
        qx = np.asarray(arrays[f"{name}_qx"])[index]
        qy = np.asarray(arrays[f"{name}_qy"])[index]
        qn = qx * cosine + qy * sine
        profile = []
        for left, right in zip(edges[:-1], edges[1:], strict=True):
            mask = near & (theta >= left) & (theta < right)
            profile.append(float(np.sum(area[mask] * qn[mask]) / max(np.sum(area[mask]), EPS)))
        axis.plot(centres, profile, label=label)
    axis.set_xlabel(r"cylinder angle $\theta$ [deg]")
    axis.set_ylabel("near-wall normal heat flux $q_n$ (normalised)")
    axis.set_xlim(0.0, 180.0)
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, ncol=3)
    for suffix in ("png", "pdf"):
        path = output / f"mv17b_fresh_near_wall_qn.{suffix}"
        figure.savefig(path, dpi=500 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
        names.append(path.name)
    plt.close(figure)
    return names


def analyze(campaign_root: Path, output_root: Path) -> dict[str, Any]:
    campaign = Path(campaign_root).resolve()
    output_root = Path(output_root).resolve()
    analysis = output_root / "analysis"
    if analysis.exists():
        raise FileExistsError(f"refusing to overwrite MV17B analysis: {analysis}")
    lock = verify_frozen_model(output_root)
    with np.load(
        output_root / "frozen" / lock["model_file"], allow_pickle=False
    ) as source:
        model = {name: np.asarray(source[name]) for name in source.files}
    analysis.mkdir(parents=True)

    mv17a = _mv17a_module()
    records: list[dict[str, Any]] = []
    fresh_paths: list[Path] = []
    field_lists: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "target_qx",
            "target_qy",
            "raw_b3_qx",
            "raw_b3_qy",
            "raw_b10_qx",
            "raw_b10_qy",
            "prior_qx",
            "prior_qy",
            "selected_qx",
            "selected_qy",
            "phase_qx",
            "phase_qy",
            "near_wall_mask",
            "cos_theta",
            "sin_theta",
            "theta",
        )
    }
    selected_dc_errors: list[float] = []
    last_tud_values: list[float] = []
    for pair_index, (pair_id, observation_seed, reference_seed) in enumerate(PAIRS):
        for role, expected_seed in (("observation", observation_seed), ("reference", reference_seed)):
            case = campaign / "cases" / _case_id(pair_id, role)
            metadata = json.loads((case / "CASE_METADATA.json").read_text(encoding="utf-8"))
            if (
                metadata.get("pair_id") != pair_id
                or metadata.get("role") != role
                or int(metadata.get("seed", -1)) != expected_seed
            ):
                raise ValueError(f"MV17B case identity mismatch: {case}")
            status_values = {}
            for line in (case / "results" / "RUN_STATUS.env").read_text(encoding="utf-8").splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    status_values[key] = value
            if status_values.get("STATUS") != "LOCKED_WINDOW_COMPLETE":
                raise ValueError(f"MV17B trajectory incomplete: {case}")
            if int(status_values.get("LAST_NOUT", "0")) < max(LOCKED_NOUT):
                raise ValueError(f"MV17B trajectory lacks NOUT116: {case}")
            last_tud_values.append(float(status_values.get("LAST_TUD", "nan")))

        observation3, paths3 = _load_case_fields(campaign, pair_id, "observation", B3_NOUT)
        observation10, paths10 = _load_case_fields(campaign, pair_id, "observation", B10_NOUT)
        target10, target_paths = _load_case_fields(campaign, pair_id, "reference", B10_NOUT)
        fresh_paths.extend((*paths3, *paths10, *target_paths))
        _assert_same_mesh(observation3, observation10)
        _assert_same_mesh(observation3, target10)
        applied = _apply_frozen(observation3, model, PHASE_SEEDS[pair_index])
        x = np.asarray(observation3["x_m"], dtype=np.float64)
        y = np.asarray(observation3["y_m"], dtype=np.float64)
        area = np.asarray(observation3["area_m2"], dtype=np.float64)
        near = np.asarray(applied["near_wall_mask"], dtype=bool)
        cosine = np.asarray(applied["cos_theta"])
        sine = np.asarray(applied["sin_theta"])
        methods = {
            "raw_b3": (
                observation3["outputs"][:, QX_INDEX],
                observation3["outputs"][:, QY_INDEX],
            ),
            "raw_b10": (
                observation10["outputs"][:, QX_INDEX],
                observation10["outputs"][:, QY_INDEX],
            ),
            "prior": (applied["prior_qx"], applied["prior_qy"]),
            "selected": (applied["selected_qx"], applied["selected_qy"]),
            "phase": (applied["phase_qx"], applied["phase_qy"]),
        }
        target_qx = np.asarray(target10["outputs"][:, QX_INDEX], dtype=np.float64)
        target_qy = np.asarray(target10["outputs"][:, QY_INDEX], dtype=np.float64)
        target_qn = target_qx * cosine + target_qy * sine
        row: dict[str, Any] = {
            "pair_id": pair_id,
            "observation_seed": observation_seed,
            "reference_seed": reference_seed,
        }
        for name, (qx_value, qy_value) in methods.items():
            qx_value = np.asarray(qx_value, dtype=np.float64)
            qy_value = np.asarray(qy_value, dtype=np.float64)
            qn_value = qx_value * cosine + qy_value * sine
            row[f"{name}_global_qy_nrmse"] = mv17a.area_weighted_nrmse(
                qy_value, target_qy, area
            )
            row[f"{name}_near_wall_qn_nrmse"] = mv17a.area_weighted_nrmse(
                qn_value, target_qn, area, near
            )
        for endpoint in ("global_qy", "near_wall_qn"):
            row[f"selected_{endpoint}_ratio_to_Raw_B10"] = row[
                f"selected_{endpoint}_nrmse"
            ] / max(row[f"raw_b10_{endpoint}_nrmse"], EPS)
        records.append(row)
        for name, value in {
            "target_qx": target_qx,
            "target_qy": target_qy,
            "raw_b3_qx": methods["raw_b3"][0],
            "raw_b3_qy": methods["raw_b3"][1],
            "raw_b10_qx": methods["raw_b10"][0],
            "raw_b10_qy": methods["raw_b10"][1],
            "prior_qx": methods["prior"][0],
            "prior_qy": methods["prior"][1],
            "selected_qx": methods["selected"][0],
            "selected_qy": methods["selected"][1],
            "phase_qx": methods["phase"][0],
            "phase_qy": methods["phase"][1],
            "near_wall_mask": near,
            "cos_theta": cosine,
            "sin_theta": sine,
            "theta": applied["theta"],
        }.items():
            field_lists[name].append(np.asarray(value))
        selected_dc_errors.extend(float(v) for v in applied["selected_dc"].values())

    _write_manifest(output_root, "analysis/fresh_source_manifest.json", fresh_paths)
    _verify_manifest(output_root, "analysis/fresh_source_manifest.json")
    methods = ("raw_b3", "raw_b10", "prior", "selected", "phase")
    endpoints = ("global_qy", "near_wall_qn")
    mean_nrmse = {
        method: {
            endpoint: float(np.mean([row[f"{method}_{endpoint}_nrmse"] for row in records]))
            for endpoint in endpoints
        }
        for method in methods
    }
    ratios = {
        method: {
            endpoint: mean_nrmse[method][endpoint]
            / max(mean_nrmse["raw_b10"][endpoint], EPS)
            for endpoint in endpoints
        }
        for method in methods
    }
    statistics = {
        endpoint: mv17a.paired_statistics(
            [row[f"selected_{endpoint}_nrmse"] for row in records],
            [row[f"raw_b10_{endpoint}_nrmse"] for row in records],
        )
        for endpoint in endpoints
    }
    adjusted = _holm_adjust(
        {endpoint: statistics[endpoint]["exact_sign_test_one_sided_p"] for endpoint in endpoints}
    )
    for endpoint in endpoints:
        statistics[endpoint]["Holm_two_endpoint_adjusted_one_sided_p"] = adjusted[endpoint]
    per_pair = {
        endpoint: {
            str(row["pair_id"]): float(row[f"selected_{endpoint}_ratio_to_Raw_B10"])
            for row in records
        }
        for endpoint in endpoints
    }
    primary_gates = {
        "all_six_disjoint_fresh_pairs_present": len(records) == len(PAIRS),
        "all_twelve_trajectories_reached_locked_NOUT116": len(last_tud_values) == 2 * len(PAIRS),
        "frozen_model_hash_verified": _sha256(output_root / "frozen" / lock["model_file"])
        == lock["model_sha256"],
        "no_fresh_parameter_selection_or_retraining": True,
        "area_weighted_cartesian_DC_preserved": max(selected_dc_errors) <= 1.0e-10,
        "selected_mean_global_qy_better_than_Raw_B10": ratios["selected"]["global_qy"] < 1.0,
        "selected_mean_near_wall_qn_better_than_Raw_B10": ratios["selected"]["near_wall_qn"] < 1.0,
        "selected_global_qy_better_in_every_pair": all(
            value < 1.0 for value in per_pair["global_qy"].values()
        ),
        "selected_near_wall_qn_better_in_every_pair": all(
            value < 1.0 for value in per_pair["near_wall_qn"].values()
        ),
        "Holm_adjusted_global_qy_one_sided_p_below_0p05": adjusted["global_qy"] < 0.05,
        "Holm_adjusted_near_wall_qn_one_sided_p_below_0p05": adjusted["near_wall_qn"] < 0.05,
    }
    diagnostic_gates = {
        "selected_beats_frozen_prior_global_qy": ratios["selected"]["global_qy"]
        < ratios["prior"]["global_qy"],
        "selected_beats_frozen_prior_near_wall_qn": ratios["selected"]["near_wall_qn"]
        < ratios["prior"]["near_wall_qn"],
        "phase_scramble_degrades_global_qy": ratios["phase"]["global_qy"]
        > ratios["selected"]["global_qy"],
        "phase_scramble_degrades_near_wall_qn": ratios["phase"]["near_wall_qn"]
        > ratios["selected"]["near_wall_qn"],
    }
    all_primary = all(primary_gates.values())
    arrays = {name: np.asarray(values) for name, values in field_lists.items()}
    np.savez_compressed(
        analysis / "mv17b_fresh_confirmation_fields.npz",
        x_m=np.asarray(model["x_m"]),
        y_m=np.asarray(model["y_m"]),
        area_m2=np.asarray(model["area_m2"]),
        pair_ids=np.asarray([row["pair_id"] for row in records]),
        observation_seeds=np.asarray([row["observation_seed"] for row in records], dtype=np.int64),
        reference_seeds=np.asarray([row["reference_seed"] for row in records], dtype=np.int64),
        **arrays,
    )
    _write_rows(analysis / "mv17b_fresh_pair_metrics.csv", list(records[0]), records)
    figures = _plot_results(
        analysis,
        {"x_m": model["x_m"], "y_m": model["y_m"], "area_m2": model["area_m2"], **arrays},
        records,
    )
    summary = {
        "stage": STAGE,
        "status": "complete_MV17B_fresh_cylinder_confirmation",
        "decision": (
            "MV17B_fresh_independent_cylinder_confirmation_passes"
            if all_primary
            else "MV17B_fresh_independent_cylinder_confirmation_fails"
        ),
        "scientific_classification": locked_protocol()["scientific_classification"],
        "all_primary_confirmation_gates_pass": all_primary,
        "primary_confirmation_gates": primary_gates,
        "secondary_diagnostic_gates": diagnostic_gates,
        "mean_nrmse": mean_nrmse,
        "ratios_to_independent_Raw_B10": ratios,
        "selected_per_pair_ratios_to_Raw_B10": per_pair,
        "clustered_paired_statistics": statistics,
        "pair_count": len(records),
        "fresh_trajectory_count": 2 * len(records),
        "frozen_model_sha256": lock["model_sha256"],
        "minimum_last_tU_over_D": float(np.min(last_tud_values)),
        "maximum_last_tU_over_D": float(np.max(last_tud_values)),
        "tU_over_D_30_stationarity_claim_authorized": False,
        "original_tU_over_D_30_warning_preserved": True,
        "neural_training": False,
        "fresh_parameter_selection": False,
        "figures": figures,
        "prediction_file": "mv17b_fresh_confirmation_fields.npz",
    }
    _atomic_json(analysis / "summary.json", summary)
    return summary


def package_results(output_root: Path, return_directory: Path) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    returned = Path(return_directory).resolve()
    verify_frozen_model(output_root)
    _verify_manifest(output_root, "analysis/fresh_source_manifest.json")
    analysis = output_root / "analysis"
    summary = json.loads((analysis / "summary.json").read_text(encoding="utf-8"))
    files: list[Path] = [
        output_root / "frozen" / PROTOCOL_FILE,
        output_root / "frozen" / "model_lock.json",
        output_root / "frozen" / "mv17b_frozen_cylinder_model.npz",
        output_root / "frozen" / "source_lock_manifest.json",
        analysis / "fresh_source_manifest.json",
        analysis / "summary.json",
        analysis / "mv17b_fresh_pair_metrics.csv",
        analysis / "mv17b_fresh_confirmation_fields.npz",
        *[analysis / name for name in summary["figures"]],
        output_root / "seed_table.tsv",
        output_root / "SUBMISSION.env",
    ]
    accounting = analysis / "mv17b_slurm_accounting.psv"
    if accounting.is_file():
        files.append(accounting)
    for pair_id, _, _ in PAIRS:
        for role in ("observation", "reference"):
            case = output_root / "campaign" / "cases" / _case_id(pair_id, role)
            for name in ("CASE_METADATA.json", "CASE_SHA256SUMS.txt"):
                path = case / name
                if path.is_file():
                    files.append(path)
            for name in ("RUN_STATUS.env", "RNG_SEED_USED.txt", "DIAG.TXT", "FINAL TIME.dat"):
                path = case / "results" / name
                if path.is_file():
                    files.append(path)
    # Include every raw moment that directly enters a primary metric.  Guard
    # blocks and large solver field dumps remain on Unity and are not packaged.
    source_manifest = json.loads((analysis / "fresh_source_manifest.json").read_text(encoding="utf-8"))
    files.extend(output_root / relative for relative in source_manifest["files"])
    files = list(dict.fromkeys(path.resolve() for path in files))
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    _write_manifest(output_root, "analysis/artifact_manifest.json", files)
    _verify_manifest(output_root, "analysis/artifact_manifest.json")
    files.append(analysis / "artifact_manifest.json")

    returned.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = returned / f"MV17B_FRESH_CYLINDER_CONFIRMATION_BUNDLE_{timestamp}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as stream:
        for path in files:
            stream.write(path, arcname=str(path.relative_to(output_root)))
    digest = _sha256(archive)
    (archive.with_suffix(archive.suffix + ".sha256")).write_text(
        f"{digest}  {archive.name}\n", encoding="utf-8"
    )
    result = {
        "stage": STAGE,
        "decision": summary["decision"],
        "archive": str(archive),
        "archive_sha256": digest,
        "fresh_DSMC_trajectory_count": 2 * len(PAIRS),
        "neural_training": False,
        "fresh_parameter_selection": False,
    }
    _atomic_json(analysis / "return.json", result)
    pointer = returned / RESULT_POINTER
    temporary = pointer.with_suffix(pointer.suffix + ".tmp")
    temporary.write_text(
        "\n".join(
            (
                f"MV17B_OUTPUT_ROOT={output_root}",
                f"MV17B_RESULT_ARCHIVE={archive}",
                f"MV17B_RESULT_ARCHIVE_SHA256={digest}",
                f"MV17B_DECISION={summary['decision']}",
                "MV17B_FRESH_DSMC_TRAJECTORIES=12",
                "MV17B_NEURAL_TRAINING=false",
                "MV17B_FRESH_PARAMETER_SELECTION=false",
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
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--mv16b-root", type=Path, required=True)
    freeze.add_argument("--mv17a-root", type=Path, required=True)
    freeze.add_argument("--output-root", type=Path, required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--campaign-root", type=Path, required=True)
    analyze_parser.add_argument("--output-root", type=Path, required=True)
    package = subparsers.add_parser("package")
    package.add_argument("--output-root", type=Path, required=True)
    package.add_argument("--return-directory", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        value = verify_contract()
    elif args.command == "freeze":
        value = freeze_model(args.mv16b_root, args.mv17a_root, args.output_root)
    elif args.command == "analyze":
        value = analyze(args.campaign_root, args.output_root)
    else:
        value = package_results(args.output_root, args.return_directory)
    print(_json_dumps(value))


if __name__ == "__main__":
    main()
