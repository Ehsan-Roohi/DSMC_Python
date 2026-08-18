#!/usr/bin/env python3
"""Compare the released common-grid pure-Maxwell cavity calculations.

The script consumes ``data/pure_maxwell/common_grid_fields.npz`` at
``Kn_Gu = 0.05`` and ``0.20``. It validates the packaged field shapes,
positivity and case metadata, evaluates field and anti-Fourier diagnostics,
and regenerates the manuscript's six numerical figures. Raw-state validation
is performed separately by ``analysis/validate_release.py``; the excluded
legacy R13 wall-completion source is neither imported nor required here.

No model field is fitted to, or used to smooth, the DSMC reference.  The only
filter applied to the anti-Fourier diagnostic is the declared uniform spatial
filter, applied independently to every method.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-pure-maxwell-jfm")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import uniform_filter, uniform_filter1d


HERE = Path(__file__).resolve()
RELEASE = HERE.parents[2]
DATA_ROOT = RELEASE / "data"
DATA_OUT = DATA_ROOT / "pure_maxwell"
FIG_OUT = RELEASE / "figures"
FIG_PM = FIG_OUT / "pure_maxwell"

TARGET_N = 160
SMOOTHING = 7
ACTIVITY = 0.05
CORNER_EPS = 0.05
METHODS = ("DSMC", "R13", "R26")
FIELD_ORDER = (
    "rho", "u", "v", "theta", "qx", "qy", "sigma_xx", "sigma_xy",
    "sigma_yy", "R_xx", "R_xy", "R_yy", "m_xxx", "m_xxy", "m_xyy",
    "m_yyy", "Delta",
)
COLORS = {"DSMC": "#2E75B6", "R13": "#D46A1F", "R26": "#3A9D78"}
LIGHT_DSMC = "#A8CBE8"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(a: float, b: float, *, rtol: float = 2e-11, atol: float = 2e-14) -> bool:
    return math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    names: list[str] = []
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "stixsans",
            "font.size": 12.4,
            "axes.labelsize": 13.0,
            "axes.titlesize": 13.0,
            "xtick.labelsize": 11.2,
            "ytick.labelsize": 11.2,
            "legend.fontsize": 11.3,
            "axes.linewidth": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, path: Path, *, dpi: int = 600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"Creator": HERE.name, "CreationDate": None, "ModDate": None}
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04, metadata=meta)
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, *, light: bool = True) -> None:
    ax.text(
        0.975,
        0.975,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontweight="bold",
        fontsize=12.4,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78 if light else 1.0, "pad": 1.0},
        zorder=50,
    )


def method_label(ax: plt.Axes, method: str) -> None:
    ax.text(
        0.025,
        0.975,
        method,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.0,
        fontweight="semibold",
        color=COLORS[method],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
        zorder=50,
    )


def cavity_axes(ax: plt.Axes, row: int, col: int, nrows: int) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks((0, 0.5, 1))
    ax.set_yticks((0, 0.5, 1))
    if row == nrows - 1:
        ax.set_xlabel(r"$x/L$")
    else:
        ax.tick_params(labelbottom=False)
    if col == 0:
        ax.set_ylabel(r"$y/L$")
    else:
        ax.tick_params(labelleft=False)


def light_cmap(name: str, low: float = 0.08, high: float = 0.82) -> LinearSegmentedColormap:
    base = plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(name + "_print", base(np.linspace(low, high, 256)))



def vector_metrics(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    p = pred[mask]
    r = ref[mask]
    nr = float(np.linalg.norm(r))
    npred = float(np.linalg.norm(p))
    require(nr > 0 and npred > 0, "nonzero vector norm")
    mp = np.linalg.norm(p, axis=-1)
    mr = np.linalg.norm(r, axis=-1)
    valid = (mp > 1e-14) & (mr > 1e-14)
    cosine = np.clip(np.sum(p[valid] * r[valid], axis=-1) / (mp[valid] * mr[valid]), -1, 1)
    angle = np.degrees(np.arccos(cosine))
    weights = mp[valid] * mr[valid]
    return {
        "E": float(np.linalg.norm(p - r) / nr),
        "C": float(np.sum(p * r) / (npred * nr)),
        "G": npred / nr,
        "angle_weighted_deg": float(np.average(angle, weights=weights)),
        "angle_median_deg": float(np.median(angle)),
        "angle_p90_deg": float(np.percentile(angle, 90)),
    }


def scalar_metrics(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    p = pred[mask]
    r = ref[mask]
    pp = p - 1.0
    rr = r - 1.0
    return {
        "E_full": float(np.linalg.norm(p - r) / np.linalg.norm(r)),
        "E_perturbation": float(np.linalg.norm(p - r) / np.linalg.norm(rr)),
        "C_perturbation": float(np.corrcoef(pp, rr)[0, 1]),
        "G_perturbation": float(np.linalg.norm(pp) / np.linalg.norm(rr)),
        "mean_bias": float(np.mean(p - r)),
    }


def anti_fourier(
    fields: dict[str, np.ndarray], eligible: np.ndarray, *, smoothing: int, threshold: float
) -> dict[str, Any]:
    x, y = fields["x"], fields["y"]
    theta = uniform_filter(fields["theta"], size=smoothing, mode="nearest")
    qx = uniform_filter(fields["qx"], size=smoothing, mode="nearest")
    qy = uniform_filter(fields["qy"], size=smoothing, mode="nearest")
    dtdy, dtdx = np.gradient(theta, y, x, edge_order=2)
    qmag = np.hypot(qx, qy)
    gmag = np.hypot(dtdx, dtdy)
    qcut = threshold * float(np.max(qmag[eligible]))
    gcut = threshold * float(np.max(gmag[eligible]))
    active = eligible & (qmag > qcut) & (gmag > gcut)
    iaf = np.full(theta.shape, np.nan)
    iaf[active] = (qx[active] * dtdx[active] + qy[active] * dtdy[active]) / (
        qmag[active] * gmag[active]
    )
    af = active & (iaf > 0)
    return {
        "theta": theta,
        "qx": qx,
        "qy": qy,
        "qmag": qmag,
        "dTdx": dtdx,
        "dTdy": dtdy,
        "gmag": gmag,
        "active": active,
        "iaf": iaf,
        "af": af,
        "f_active_domain": float(np.mean(active)),
        "f_AF_domain": float(np.mean(af)),
        "f_AF_active": float(np.count_nonzero(af) / max(np.count_nonzero(active), 1)),
        "mean_IAF_AF": float(np.mean(iaf[af])) if np.any(af) else math.nan,
    }


def overlap(pred: np.ndarray, ref: np.ndarray, common: np.ndarray) -> dict[str, float]:
    p = pred & common
    r = ref & common
    inter = int(np.count_nonzero(p & r))
    union = int(np.count_nonzero(p | r))
    pc = int(np.count_nonzero(p))
    rc = int(np.count_nonzero(r))
    return {
        "Jaccard": inter / max(union, 1),
        "Dice": 2 * inter / max(pc + rc, 1),
        "precision": inter / max(pc, 1),
        "recall": inter / max(rc, 1),
    }


def eligible_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    X, Y = np.meshgrid(x, y)
    return ~(((X < CORNER_EPS) | (X > 1.0 - CORNER_EPS)) & (Y > 1.0 - CORNER_EPS))



def build_released_case(
    tag: str, kn: float
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Load the released common-grid fields without the excluded R13 solver.

    The raw-to-common-grid conversion was performed by the source-locked
    campaign analysis and is preserved in ``common_grid_fields.npz``.  This
    public path regenerates every metric and figure from those six primary
    fields while the independent release validator checks the corresponding
    raw DSMC and accepted R13/R26 states.
    """

    archive_path = DATA_OUT / "common_grid_fields.npz"
    report_path = DATA_OUT / "audit_metrics.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metadata = json.loads(
        (DATA_ROOT / "dsmc" / f"kn0{tag}" / "case_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    require(close(metadata["kn_gu"], kn), "released DSMC Kn value")
    c0 = math.sqrt(
        1.380649e-23
        * float(metadata["wall_temperature_K"])
        / float(metadata["argon_mass_kg"])
    )
    grid_path = DATA_ROOT / "dsmc" / f"kn0{tag}" / "grid.final.00200000"
    cell_coordinates = np.loadtxt(
        grid_path, skiprows=9, usecols=(1, 2), dtype=float
    )
    x_coordinates = np.sort(np.unique(cell_coordinates[:, 0])) / float(
        metadata["length_m"]
    )
    y_coordinates = np.sort(np.unique(cell_coordinates[:, 1])) / float(
        metadata["length_m"]
    )
    require(
        x_coordinates.size == TARGET_N and y_coordinates.size == TARGET_N,
        "released DSMC tensor grid",
    )
    fields: dict[str, dict[str, np.ndarray]] = {}
    with np.load(archive_path, allow_pickle=False) as archive:
        for method in METHODS:
            prefix = f"k{tag}_{method.lower()}_"
            fields[method] = {
                name: np.asarray(archive[prefix + name], dtype=float)
                for name in ("rho", "u", "v", "theta", "qx", "qy")
            }
            for name in ("rho", "u", "v", "theta", "qx", "qy"):
                require(
                    fields[method][name].shape == (TARGET_N, TARGET_N),
                    f"released {tag}/{method}/{name} shape",
                )
                require(
                    bool(np.all(np.isfinite(fields[method][name]))),
                    f"released {tag}/{method}/{name} finite",
                )
            require(float(np.min(fields[method]["rho"])) > 0.0, "positive density")
            require(float(np.min(fields[method]["theta"])) > 0.0, "positive temperature")
            fields[method].update(
                {
                    "x": x_coordinates.copy(),
                    "y": y_coordinates.copy(),
                    "c0": np.asarray(c0),
                }
            )

    eligible = eligible_mask(x_coordinates, y_coordinates)
    analysis = {
        method: anti_fourier(
            fields[method], eligible, smoothing=SMOOTHING, threshold=ACTIVITY
        )
        for method in METHODS
    }
    integrity = report.get("integrity", {}).get(tag, {})
    require(set(integrity) == set(METHODS), f"released {tag} integrity record")
    return fields, {
        "kn_gu": kn,
        "tag": tag,
        "eligible": eligible,
        "analysis": analysis,
        "audit": integrity,
        "processed_fields_sha256": sha256(archive_path),
    }


def evaluate_case(
    fields: dict[str, dict[str, np.ndarray]], context: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    kn = context["kn_gu"]
    eligible = context["eligible"]
    analyses = context["analysis"]
    common = analyses["DSMC"]["active"]
    ref_af = analyses["DSMC"]["af"] & common
    field_rows: list[dict[str, Any]] = []
    af_rows: list[dict[str, Any]] = []
    for method in ("R13", "R26"):
        rho = scalar_metrics(fields[method]["rho"], fields["DSMC"]["rho"], eligible)
        temp = scalar_metrics(fields[method]["theta"], fields["DSMC"]["theta"], eligible)
        vel = vector_metrics(
            np.stack((fields[method]["u"], fields[method]["v"]), axis=-1),
            np.stack((fields["DSMC"]["u"], fields["DSMC"]["v"]), axis=-1),
            eligible,
        )
        qraw = vector_metrics(
            np.stack((fields[method]["qx"], fields[method]["qy"]), axis=-1),
            np.stack((fields["DSMC"]["qx"], fields["DSMC"]["qy"]), axis=-1),
            eligible,
        )
        qsm = vector_metrics(
            np.stack((analyses[method]["qx"], analyses[method]["qy"]), axis=-1),
            np.stack((analyses["DSMC"]["qx"], analyses["DSMC"]["qy"]), axis=-1),
            eligible,
        )
        field_rows.append(
            {
                "kn_gu": kn,
                "model": method,
                "comparison_cells": int(np.count_nonzero(eligible)),
                **{f"rho_{k}": v for k, v in rho.items()},
                **{f"temperature_{k}": v for k, v in temp.items()},
                **{f"velocity_{k}": v for k, v in vel.items()},
                **{f"q_raw_{k}": v for k, v in qraw.items()},
                **{f"q_smoothed_{k}": v for k, v in qsm.items()},
            }
        )
        ov = overlap(analyses[method]["af"], ref_af, common)
        qcommon = vector_metrics(
            np.stack((analyses[method]["qx"], analyses[method]["qy"]), axis=-1),
            np.stack((analyses["DSMC"]["qx"], analyses["DSMC"]["qy"]), axis=-1),
            common,
        )
        af_rows.append(
            {
                "kn_gu": kn,
                "model": method,
                "smoothing_cells": SMOOTHING,
                "activity_threshold": ACTIVITY,
                "common_mask_cells": int(np.count_nonzero(common)),
                "DSMC_f_active_domain": analyses["DSMC"]["f_active_domain"],
                "DSMC_f_AF_domain": analyses["DSMC"]["f_AF_domain"],
                "DSMC_f_AF_active": analyses["DSMC"]["f_AF_active"],
                "model_f_active_domain": analyses[method]["f_active_domain"],
                "model_f_AF_domain": analyses[method]["f_AF_domain"],
                "model_f_AF_active": analyses[method]["f_AF_active"],
                **ov,
                **{f"q_common_{k}": v for k, v in qcommon.items()},
            }
        )
    sensitivity: list[dict[str, Any]] = []
    for window in (3, 5, 7, 9, 11):
        for cut in (0.03, 0.05, 0.07, 0.10):
            local = {m: anti_fourier(fields[m], eligible, smoothing=window, threshold=cut) for m in METHODS}
            c = local["DSMC"]["active"]
            r = local["DSMC"]["af"] & c
            if np.count_nonzero(c) < 10:
                continue
            for method in ("R13", "R26"):
                ov = overlap(local[method]["af"], r, c)
                q = vector_metrics(
                    np.stack((local[method]["qx"], local[method]["qy"]), axis=-1),
                    np.stack((local["DSMC"]["qx"], local["DSMC"]["qy"]), axis=-1),
                    c,
                )
                sensitivity.append(
                    {
                        "kn_gu": kn,
                        "model": method,
                        "smoothing_cells": window,
                        "activity_threshold": cut,
                        "common_mask_cells": int(np.count_nonzero(c)),
                        **ov,
                        **{f"q_common_{k}": v for k, v in q.items()},
                    }
                )
    return field_rows, af_rows, sensitivity


def primary_figure(fields: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    configure_plotting()
    X, Y = np.meshgrid(fields["DSMC"]["x"], fields["DSMC"]["y"])
    scalar_sets = [
        [fields[m]["rho"] for m in METHODS],
        [fields[m]["theta"] for m in METHODS],
        [np.hypot(fields[m]["u"], fields[m]["v"]) for m in METHODS],
        [np.hypot(fields[m]["qx"], fields[m]["qy"]) for m in METHODS],
    ]
    norms = [
        Normalize(np.percentile(np.concatenate([v.ravel() for v in scalar_sets[0]]), 0.2), np.percentile(np.concatenate([v.ravel() for v in scalar_sets[0]]), 99.8)),
        Normalize(np.percentile(np.concatenate([v.ravel() for v in scalar_sets[1]]), 0.2), np.percentile(np.concatenate([v.ravel() for v in scalar_sets[1]]), 99.8)),
        Normalize(0, np.percentile(np.concatenate([v.ravel() for v in scalar_sets[2]]), 99.8)),
        Normalize(0, np.percentile(np.concatenate([v.ravel() for v in scalar_sets[3]]), 99.5)),
    ]
    cmaps = (light_cmap("Blues", 0.12, 0.78), light_cmap("YlOrRd", 0.08, 0.75), light_cmap("BuGn", 0.08, 0.75), light_cmap("YlGnBu", 0.06, 0.76))
    cblabels = (r"$\rho/\rho_0$", r"$T/T_w$", r"$|\mathbf{u}|/c_0$", r"$|\mathbf{q}|/(\rho_0c_0^3)$")
    fig, axes = plt.subplots(3, 4, figsize=(14.8, 9.4), layout="constrained")
    images: list[Any] = [None] * 4
    panel = 0
    for row, method in enumerate(METHODS):
        values = [scalar_sets[col][row] for col in range(4)]
        for col, value in enumerate(values):
            ax = axes[row, col]
            images[col] = ax.imshow(value, origin="lower", extent=(0, 1, 0, 1), interpolation="nearest", cmap=cmaps[col], norm=norms[col], rasterized=True)
            sample = np.s_[6::12, 6::12]
            if col == 2:
                ax.quiver(X[sample], Y[sample], fields[method]["u"][sample], fields[method]["v"][sample], color="#155F75", scale=1.35, width=0.0034, headwidth=3.4)
            elif col == 3:
                ax.quiver(X[sample], Y[sample], fields[method]["qx"][sample], fields[method]["qy"][sample], color="#A24A22", scale=0.34, width=0.0034, headwidth=3.4)
            cavity_axes(ax, row, col, 3)
            panel_label(ax, f"({chr(97 + panel)})")
            if col == 0:
                method_label(ax, method)
            panel += 1
    for col, image in enumerate(images):
        cb = fig.colorbar(image, ax=axes[:, col], shrink=0.90, pad=0.012, extend="both" if col < 2 else "max")
        cb.set_label(cblabels[col], fontsize=12.2)
        cb.ax.tick_params(labelsize=10.4)
    save_figure(fig, path)


def center_average(values: np.ndarray, axis: int) -> np.ndarray:
    n = values.shape[axis]
    return 0.5 * (np.take(values, n // 2 - 1, axis=axis) + np.take(values, n // 2, axis=axis))


def centerline_figure(fields: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    configure_plotting()
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.0), layout="constrained")
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(rect=(0.0, 0.0, 1.0, 0.93))
    uw = 100.0 / float(fields["DSMC"]["c0"])
    d = fields["DSMC"]
    profiles = (
        (center_average(d["u"], 1) / uw, d["y"], "xy"),
        (d["x"], center_average(d["v"], 0) / uw, "yx"),
        (center_average(d["rho"], 1), d["y"], "xy"),
        (center_average(d["theta"], 1), d["y"], "xy"),
        (center_average(d["qx"], 1), d["y"], "xy"),
        (d["x"], center_average(d["qy"], 0), "yx"),
    )
    for ax, (first, second, orient) in zip(axes.flat, profiles):
        if orient == "xy":
            ax.plot(first, second, color=LIGHT_DSMC, lw=0.9, alpha=0.75)
            ax.plot(uniform_filter1d(first, 7, mode="nearest"), second, color=COLORS["DSMC"], lw=2.8, label="DSMC")
        else:
            ax.plot(first, second, color=LIGHT_DSMC, lw=0.9, alpha=0.75)
            ax.plot(first, uniform_filter1d(second, 7, mode="nearest"), color=COLORS["DSMC"], lw=2.8, label="DSMC")
    styles = {"R13": ("--", COLORS["R13"]), "R26": ("-.", COLORS["R26"])}
    for method in ("R13", "R26"):
        f = fields[method]
        ls, color = styles[method]
        style = {"ls": ls, "color": color, "lw": 2.5, "label": method}
        axes[0, 0].plot(center_average(f["u"], 1) / uw, f["y"], **style)
        axes[0, 1].plot(f["x"], center_average(f["v"], 0) / uw, **style)
        axes[0, 2].plot(center_average(f["rho"], 1), f["y"], **style)
        axes[1, 0].plot(center_average(f["theta"], 1), f["y"], **style)
        axes[1, 1].plot(center_average(f["qx"], 1), f["y"], **style)
        axes[1, 2].plot(f["x"], center_average(f["qy"], 0), **style)
    labels = (
        (r"$u(x=0.5,y)/U_w$", r"$y/L$"),
        (r"$x/L$", r"$v(x,y=0.5)/U_w$"),
        (r"$\rho(x=0.5,y)/\rho_0$", r"$y/L$"),
        (r"$T(x=0.5,y)/T_w$", r"$y/L$"),
        (r"$q_x(x=0.5,y)/(\rho_0c_0^3)$", r"$y/L$"),
        (r"$x/L$", r"$q_y(x,y=0.5)/(\rho_0c_0^3)$"),
    )
    for i, (ax, (xlabel, ylabel)) in enumerate(zip(axes.flat, labels)):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#D9E2EC", lw=0.7)
        ax.margins(x=0.035, y=0.025)
        panel_label(ax, f"({chr(97 + i)})")
    handles = [Line2D([0], [0], color=COLORS[m], lw=2.7, ls="-" if m == "DSMC" else ("--" if m == "R13" else "-."), label=m) for m in METHODS]
    fig.legend(handles=handles, loc="upper center", frameon=False, ncol=3, bbox_to_anchor=(0.5, 1.01))
    save_figure(fig, path)


def anti_fourier_figure(
    fields: dict[str, dict[str, np.ndarray]], context: dict[str, Any], af_rows: list[dict[str, Any]], path: Path
) -> None:
    configure_plotting()
    analyses = context["analysis"]
    eligible = context["eligible"]
    X, Y = np.meshgrid(fields["DSMC"]["x"], fields["DSMC"]["y"])
    common = analyses["DSMC"]["active"]
    ref_af = analyses["DSMC"]["af"] & common
    qref = np.stack((analyses["DSMC"]["qx"], analyses["DSMC"]["qy"]), axis=-1)
    qrms = float(np.sqrt(np.mean(np.sum(qref[common] ** 2, axis=-1))))
    vmax = float(np.percentile(np.concatenate([analyses[m]["qmag"][eligible] / qrms for m in METHODS]), 99.4))
    fig, axes = plt.subplots(2, 3, figsize=(14.6, 8.8), layout="constrained")
    sample = np.s_[6::12, 6::12]
    image = None
    for col, method in enumerate(METHODS):
        a = analyses[method]
        image = axes[0, col].imshow(np.ma.masked_where(~eligible, a["qmag"] / qrms), origin="lower", extent=(0, 1, 0, 1), interpolation="nearest", cmap=light_cmap("YlGnBu", 0.06, 0.76), norm=Normalize(0, vmax, clip=True), rasterized=True)
        axes[0, col].quiver(X[sample], Y[sample], a["qx"][sample] / qrms, a["qy"][sample] / qrms, color="#9C4A24", scale=36, width=0.0034, headwidth=3.5)
        method_label(axes[0, col], method)
        cavity_axes(axes[0, col], 0, col, 2)
        panel_label(axes[0, col], f"({chr(97 + col)})")
    require(image is not None, "anti-Fourier magnitude image")
    cb = fig.colorbar(image, ax=axes[0, :], shrink=0.90, pad=0.012, extend="max")
    cb.set_label(r"$|\mathbf{q}|/q_{\mathrm{DSMC,rms}}$")

    codes = np.zeros(common.shape, dtype=int)
    codes[common & ~ref_af] = 1
    codes[ref_af] = 2
    support_cmap = ListedColormap(("#E8EBEF", "#BBD8EC", "#EEAA7B"))
    axes[1, 0].imshow(codes, origin="lower", extent=(0, 1, 0, 1), interpolation="nearest", cmap=support_cmap, norm=BoundaryNorm((-0.5, 0.5, 1.5, 2.5), 3), rasterized=True)
    for x0 in (0.0, 1.0 - CORNER_EPS):
        axes[1, 0].add_patch(Rectangle((x0, 1.0 - CORNER_EPS), CORNER_EPS, CORNER_EPS, facecolor="none", edgecolor="#586069", hatch="////", lw=0.7))
    axes[1, 0].legend(handles=(Patch(facecolor="#BBD8EC", label="Fourier"), Patch(facecolor="#EEAA7B", label="anti-Fourier"), Patch(facecolor="white", edgecolor="#586069", hatch="////", label="excluded corner")), loc="lower left", fontsize=9.3, frameon=True, framealpha=0.94)
    method_label(axes[1, 0], "DSMC")
    cavity_axes(axes[1, 0], 1, 0, 2)
    panel_label(axes[1, 0], "(d)")

    bounds = np.asarray((0, 5, 10, 20, 40, 90, 180), dtype=float)
    angle_cmap = ListedColormap(plt.get_cmap("YlOrBr")(np.linspace(0.18, 0.88, 6)))
    angle_cmap.set_bad("#E8EBEF")
    angle_norm = BoundaryNorm(bounds, angle_cmap.N)
    angle_image = None
    by_method = {row["model"]: row for row in af_rows}
    for col, method in ((1, "R13"), (2, "R26")):
        q = np.stack((analyses[method]["qx"], analyses[method]["qy"]), axis=-1)
        mr = np.linalg.norm(qref, axis=-1)
        mp = np.linalg.norm(q, axis=-1)
        valid = common & (mr > 1e-14) & (mp > 1e-14)
        angle = np.full(common.shape, np.nan)
        angle[valid] = np.degrees(np.arccos(np.clip(np.sum(qref[valid] * q[valid], axis=-1) / (mr[valid] * mp[valid]), -1, 1)))
        angle_image = axes[1, col].imshow(np.ma.masked_invalid(angle), origin="lower", extent=(0, 1, 0, 1), interpolation="nearest", cmap=angle_cmap, norm=angle_norm, rasterized=True)
        for mask, color, ls in ((ref_af, COLORS["DSMC"], "-"), (analyses[method]["af"] & common, COLORS[method], "--")):
            if np.any(mask) and np.any(~mask):
                axes[1, col].contour(X, Y, mask.astype(float), levels=(0.5,), colors=color, linestyles=ls, linewidths=1.15)
        row = by_method[method]
        axes[1, col].text(0.025, 0.025, rf"$E_q={row['q_common_E']:.3f}$, $C_q={row['q_common_C']:.3f}$" + "\n" + rf"$\bar\vartheta_q={row['q_common_angle_weighted_deg']:.1f}^\circ$, $J_{{AF}}={row['Jaccard']:.3f}$", transform=axes[1, col].transAxes, ha="left", va="bottom", fontsize=10.2, bbox={"facecolor": "white", "edgecolor": "#87929C", "alpha": 0.94, "pad": 2.4})
        method_label(axes[1, col], method)
        cavity_axes(axes[1, col], 1, col, 2)
        panel_label(axes[1, col], f"({chr(100 + col)})")
    require(angle_image is not None, "anti-Fourier angle image")
    cb2 = fig.colorbar(angle_image, ax=axes[1, 1:], shrink=0.90, pad=0.012, ticks=(0, 10, 20, 40, 90, 180), spacing="proportional")
    cb2.set_label(r"heat-flux angular difference, $\vartheta_q$ (deg)")
    save_figure(fig, path)


def topology_atlas(cases: dict[str, tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]], path: Path) -> None:
    configure_plotting()
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.2), layout="constrained")
    cmap = LinearSegmentedColormap.from_list(
        "light_diverging", plt.get_cmap("RdYlBu_r")(np.linspace(0.12, 0.88, 256))
    )
    cmap.set_bad("#E8EBEF")
    image = None
    panel = 0
    for row, tag in enumerate(("05", "20")):
        fields, context = cases[tag]
        X, Y = np.meshgrid(fields["DSMC"]["x"], fields["DSMC"]["y"])
        for col, method in enumerate(METHODS):
            analysis = context["analysis"][method]
            image = axes[row, col].imshow(np.ma.masked_where(~analysis["active"], analysis["iaf"]), origin="lower", extent=(0, 1, 0, 1), interpolation="nearest", cmap=cmap, norm=Normalize(-1, 1), rasterized=True)
            if np.any(analysis["af"]) and np.any(~analysis["af"]):
                axes[row, col].contour(X, Y, analysis["af"].astype(float), levels=(0.5,), colors=COLORS[method], linewidths=1.15)
            method_label(axes[row, col], method)
            axes[row, col].text(0.025, 0.055, rf"$Kn_{{\rm Gu}}={context['kn_gu']:.2f}$", transform=axes[row, col].transAxes, ha="left", va="bottom", fontsize=11.0, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.1})
            cavity_axes(axes[row, col], row, col, 2)
            panel_label(axes[row, col], f"({chr(97 + panel)})")
            panel += 1
    require(image is not None, "topology atlas")
    cb = fig.colorbar(image, ax=axes, shrink=0.88, pad=0.014, extend="both")
    cb.set_label(r"$I_{AF}=\mathbf{q}\!\cdot\!\nabla T/(|\mathbf{q}|\,|\nabla T|)$")
    save_figure(fig, path)


def metrics_summary(field_rows: list[dict[str, Any]], af_rows: list[dict[str, Any]], path: Path) -> None:
    configure_plotting()
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.8), layout="constrained")
    kns = (0.05, 0.20)
    x = np.arange(2)
    width = 0.35
    lookup = {(float(r["kn_gu"]), r["model"]): r for r in field_rows}
    aflookup = {(float(r["kn_gu"]), r["model"]): r for r in af_rows}
    for offset, method in ((-width / 2, "R13"), (width / 2, "R26")):
        axes[0].bar(x + offset, [lookup[(k, method)]["velocity_E"] for k in kns], width, color=COLORS[method], label=method, alpha=0.90)
        axes[1].bar(x + offset, [lookup[(k, method)]["q_smoothed_E"] for k in kns], width, color=COLORS[method], alpha=0.90)
        axes[2].bar(x + offset, [aflookup[(k, method)]["q_common_angle_weighted_deg"] for k in kns], width, color=COLORS[method], alpha=0.90)
        axes[3].bar(x + offset, [aflookup[(k, method)]["Jaccard"] for k in kns], width, color=COLORS[method], alpha=0.90)
    ylabels = (r"velocity error $E_u$", r"smoothed heat-flux error $E_q$", r"weighted heat-flux angle (deg)", r"anti-Fourier Jaccard $J_{AF}$")
    for i, (ax, ylabel) in enumerate(zip(axes, ylabels)):
        ax.set_xticks(x, (r"$0.05$", r"$0.20$"))
        ax.set_xlabel(r"$Kn_{\rm Gu}$")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#D9E2EC", lw=0.7)
        ax.set_axisbelow(True)
        panel_label(ax, f"({chr(97 + i)})", light=False)
    axes[0].legend(frameon=False, loc="upper left")
    axes[3].set_ylim(0, 1)
    save_figure(fig, path)


def write_centerline_data(cases: dict[str, tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]]) -> None:
    rows: list[dict[str, Any]] = []
    for tag in ("05", "20"):
        fields, context = cases[tag]
        for method in METHODS:
            f = fields[method]
            uw = 100.0 / float(f["c0"])
            for index, coordinate in enumerate(f["x"]):
                rows.append(
                    {
                        "kn_gu": context["kn_gu"],
                        "method": method,
                        "index": index,
                        "coordinate": float(coordinate),
                        "vertical_u_over_Uw": float((center_average(f["u"], 1) / uw)[index]),
                        "vertical_rho": float(center_average(f["rho"], 1)[index]),
                        "vertical_temperature": float(center_average(f["theta"], 1)[index]),
                        "vertical_qx": float(center_average(f["qx"], 1)[index]),
                        "horizontal_v_over_Uw": float((center_average(f["v"], 0) / uw)[index]),
                        "horizontal_qy": float(center_average(f["qy"], 0)[index]),
                    }
                )
    write_csv(DATA_OUT / "centerline_profiles.csv", rows)


def main() -> int:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    FIG_PM.mkdir(parents=True, exist_ok=True)
    cases = {
        "05": build_released_case("05", 0.05),
        "20": build_released_case("20", 0.20),
    }
    field_rows: list[dict[str, Any]] = []
    af_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    for tag in ("05", "20"):
        fields, context = cases[tag]
        frows, arows, srows = evaluate_case(fields, context)
        field_rows.extend(frows)
        af_rows.extend(arows)
        sensitivity_rows.extend(srows)
    write_csv(DATA_OUT / "field_metrics.csv", field_rows)
    write_csv(DATA_OUT / "anti_fourier_metrics.csv", af_rows)
    write_csv(DATA_OUT / "processing_sensitivity.csv", sensitivity_rows)
    write_centerline_data(cases)

    # New, collision-matched assets live under a distinct directory so an
    # historical build step cannot silently restore the earlier figures.
    primary_figure(cases["20"][0], FIG_PM / "primary_k20")
    centerline_figure(cases["20"][0], FIG_PM / "centerlines_k20")
    anti_fourier_figure(cases["20"][0], cases["20"][1], [r for r in af_rows if close(r["kn_gu"], 0.20)], FIG_PM / "antifourier_k20")
    anti_fourier_figure(cases["05"][0], cases["05"][1], [r for r in af_rows if close(r["kn_gu"], 0.05)], FIG_PM / "antifourier_k05")
    topology_atlas(cases, FIG_PM / "antifourier_atlas")
    metrics_summary(field_rows, af_rows, FIG_PM / "model_metrics")

    sensitivity_summary: dict[str, Any] = {}
    for kn in (0.05, 0.20):
        for method in ("R13", "R26"):
            rows = [r for r in sensitivity_rows if close(r["kn_gu"], kn) and r["model"] == method]
            sensitivity_summary[f"Kn{kn:.2f}_{method}"] = {
                key: [float(min(r[key] for r in rows)), float(max(r[key] for r in rows))]
                for key in ("Jaccard", "Dice", "q_common_E", "q_common_C", "q_common_angle_weighted_deg")
            }
    report = {
        "schema_version": 1,
        "contract": {
            "knudsen_numbers": [0.05, 0.20],
            "knudsen_convention": "Gu equilibrium mean free path divided by cavity length",
            "wall_temperature_K": 300.0,
            "lid_velocity_m_per_s": 100.0,
            "wall_accommodation": 1.0,
            "DSMC_collision_model": "SPARTA VSS transport approximation to IPL Maxwell molecules (omega=1, alpha=2.14)",
            "R13_collision_model": "Maxwell-molecule coefficients and mu proportional to T",
            "R26_collision_model": "Gu--Emerson nonlinear R26 Maxwell-molecule coefficients and mu proportional to T",
        },
        "integrity": {tag: context["audit"] for tag, (_, context) in cases.items()},
        "field_metrics": field_rows,
        "anti_fourier_metrics": af_rows,
        "processing_sensitivity_ranges": sensitivity_summary,
        "interpretation": {
            "primary": "Both moment models reproduce lower-order circulation more closely than the anti-Fourier heat-flux field. R26 improves heat-flux direction and topology relative to R13, but does not reproduce the DSMC cold-to-hot set completely.",
            "limitations": [
                "Each SPARTA operating point is represented by one statistically averaged realization; no between-seed confidence interval is available.",
                "R13 states satisfy the archived algebraic and physical gates, but their own reports label external validation as not completed and publication_grade as false.",
                "R26 Kn_Gu=0.20 is an accepted single-grid state; no transition-grid extrapolation is available.",
                "SPARTA sonine/grid fourth moments are diagnostic only and no independent rank-three m_ijk is available in the pure-Maxwell packages; higher-moment certification is therefore not claimed from these runs.",
                "Binary anti-Fourier overlap is processing-dependent; the full smoothing/threshold envelope is retained in processing_sensitivity.csv.",
            ],
        },
    }
    write_json(DATA_OUT / "audit_metrics.json", report)
    hashes = [
        f"{sha256(path)}  {path.name}"
        for path in sorted(DATA_OUT.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.txt"
    ]
    (DATA_OUT / "SHA256SUMS.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
