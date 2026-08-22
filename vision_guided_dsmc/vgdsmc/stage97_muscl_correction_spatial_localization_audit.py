from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

GRID = (64, 64)
WALL_BAND_CELLS = 4
TOP_FRACTION = 0.10
DOMINANT_SHARE = 2.0 / 3.0
MATERIAL_SHARE_SHIFT = 0.10
STAGE96_RUN_ID = 31338220298
STAGE96_DECISION = "stage96_material_persistent_muscl_correction_stage97_spatial_localization_audit"


def validate_stage97_design(**overrides: object) -> None:
    frozen = {
        "grid": GRID,
        "wall_band_cells": WALL_BAND_CELLS,
        "top_fraction": TOP_FRACTION,
        "dominant_share": DOMINANT_SHARE,
        "material_share_shift": MATERIAL_SHARE_SHIFT,
        "stage96_run_id": STAGE96_RUN_ID,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 97 is a fixed artifact-only spatial-localization audit of the exact "
            "Stage-96 first/final MUSCL-correction maps. The four-cell wall band is "
            "retained from the earlier cavity diagnostics and may not be optimized. "
            "No physics, collision/source treatment, clipping or positivity floor, "
            "relaxation, transport parameter, wall model, limiter, quadrature, "
            "tolerance, diagnostic window, or solver endpoint may be retuned."
        )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1.0e-300))


def _region_masks(shape: tuple[int, int], wall_band_cells: int = WALL_BAND_CELLS) -> dict[str, np.ndarray]:
    nx, ny = shape
    if (nx, ny) != GRID:
        raise ValueError(f"Stage 97 requires the exact {GRID[0]}x{GRID[1]} Stage-96 maps")
    if wall_band_cells != WALL_BAND_CELLS:
        raise ValueError("Stage 97 wall-band thickness is frozen at four cells")
    ix = np.arange(nx)[:, None]
    iy = np.arange(ny)[None, :]
    x_wall = (ix < wall_band_cells) | (ix >= nx - wall_band_cells)
    y_wall = (iy < wall_band_cells) | (iy >= ny - wall_band_cells)
    corners = x_wall & y_wall
    vertical_sidewalls = x_wall & ~y_wall
    horizontal_walls = y_wall & ~x_wall
    interior = ~(x_wall | y_wall)
    wall_band = ~interior
    return {
        "corners": corners,
        "vertical_sidewalls": vertical_sidewalls,
        "horizontal_walls": horizontal_walls,
        "wall_band": wall_band,
        "interior": interior,
    }


def _top_fraction_share(field: np.ndarray, fraction: float = TOP_FRACTION) -> float:
    values = np.asarray(field, dtype=np.float64).ravel()
    total = float(np.sum(values))
    if total <= 0.0:
        return 0.0
    k = max(1, int(math.ceil(fraction * values.size)))
    return float(np.partition(values, values.size - k)[-k:].sum() / total)


def _map_metrics(field: np.ndarray) -> dict[str, object]:
    values = np.asarray(field, dtype=np.float64)
    if values.shape != GRID:
        raise ValueError(f"Stage-96 correction map shape {values.shape} does not match {GRID}")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("Stage-96 correction maps must be finite and nonnegative")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("Stage-96 correction map has zero total magnitude")
    masks = _region_masks(values.shape)
    shares = {name: float(np.sum(values[mask]) / total) for name, mask in masks.items()}
    return {
        "total_magnitude": total,
        "region_shares": shares,
        "top_decile_share": _top_fraction_share(values),
    }


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 1.0e-300 or sy <= 1.0e-300:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    return _safe_ratio(float(np.dot(x, y)), float(np.linalg.norm(x) * np.linalg.norm(y)))


def _pair_metrics(first: np.ndarray, final: np.ndarray) -> dict[str, object]:
    first_metrics = _map_metrics(first)
    final_metrics = _map_metrics(final)
    first_total = float(first_metrics["total_magnitude"])
    final_total = float(final_metrics["total_magnitude"])
    p = np.asarray(first, dtype=np.float64) / first_total
    q = np.asarray(final, dtype=np.float64) / final_total
    first_shares = first_metrics["region_shares"]
    final_shares = final_metrics["region_shares"]
    return {
        "first": first_metrics,
        "final": final_metrics,
        "final_to_first_total_magnitude_ratio": _safe_ratio(final_total, first_total),
        "wall_band_share_change": float(final_shares["wall_band"] - first_shares["wall_band"]),
        "interior_share_change": float(final_shares["interior"] - first_shares["interior"]),
        "normalized_map_total_variation": float(0.5 * np.sum(np.abs(q - p))),
        "first_final_cosine_similarity": _cosine(first, final),
        "first_final_pearson": _pearson(first, final),
    }


def stage97_decision(phi: dict[str, object], psi: dict[str, object]) -> str:
    final_interior = min(
        float(phi["final"]["region_shares"]["interior"]),
        float(psi["final"]["region_shares"]["interior"]),
    )
    final_wall = min(
        float(phi["final"]["region_shares"]["wall_band"]),
        float(psi["final"]["region_shares"]["wall_band"]),
    )
    interior_shift = min(float(phi["interior_share_change"]), float(psi["interior_share_change"]))
    if final_interior >= DOMINANT_SHARE and interior_shift >= MATERIAL_SHARE_SHIFT:
        return "stage97_interior_dominant_redistribution_stage98_directional_operator_growth_audit"
    if final_wall >= DOMINANT_SHARE:
        return "stage97_wall_dominant_persistence_stage98_wall_orientation_operator_audit"
    return "stage97_mixed_spatial_persistence_stage98_signed_directional_balance_audit"


def run_stage97(stage96_artifact_dir: str | Path, output_dir: str | Path, **design: object) -> dict[str, object]:
    validate_stage97_design(**design)
    root = Path(stage96_artifact_dir)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 96:
        raise ValueError("Stage 97 requires the exact completed Stage-96 artifact")
    if summary.get("decision") != STAGE96_DECISION:
        raise ValueError("Stage-96 decision does not authorize the Stage-97 localization audit")
    cfg96 = summary.get("configuration", {})
    if cfg96.get("grid") != list(GRID) or cfg96.get("diagnostic_steps") != 25:
        raise ValueError("Stage-96 artifact does not match the frozen Stage-97 parent design")

    with np.load(root / "muscl_correction_growth_histories.npz") as data:
        maps = {}
        for distribution in ("phi", "psi"):
            for when in ("first", "final"):
                key = f"{when}_{distribution}_cell_correction_m0"
                if key not in data:
                    raise ValueError(f"Stage-96 artifact is missing {key}")
                maps[key] = np.asarray(data[key], dtype=np.float64)

    phi = _pair_metrics(maps["first_phi_cell_correction_m0"], maps["final_phi_cell_correction_m0"])
    psi = _pair_metrics(maps["first_psi_cell_correction_m0"], maps["final_psi_cell_correction_m0"])
    decision = stage97_decision(phi, psi)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    masks = _region_masks(GRID)
    np.savez_compressed(
        out / "spatial_localization_maps.npz",
        first_phi=maps["first_phi_cell_correction_m0"],
        final_phi=maps["final_phi_cell_correction_m0"],
        first_psi=maps["first_psi_cell_correction_m0"],
        final_psi=maps["final_psi_cell_correction_m0"],
        wall_band=masks["wall_band"],
        interior=masks["interior"],
        corners=masks["corners"],
        vertical_sidewalls=masks["vertical_sidewalls"],
        horizontal_walls=masks["horizontal_walls"],
    )

    result: dict[str, object] = {
        "stage": 97,
        "description": (
            "Artifact-only spatial localization of the exact Stage-96 first/final weighted-absolute "
            "MUSCL-correction m0 maps, using the previously established fixed four-cell wall band."
        ),
        "retained_stage96_decision": summary["decision"],
        "configuration": {
            "grid": list(GRID),
            "wall_band_cells": WALL_BAND_CELLS,
            "top_fraction": TOP_FRACTION,
            "dominant_share": DOMINANT_SHARE,
            "material_share_shift": MATERIAL_SHARE_SHIFT,
            "stage96_run_id": STAGE96_RUN_ID,
            "input_maps": "exact Stage-96 first/final weighted-absolute correction m0 maps",
            "solver_rerun": False,
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False,
            "correction_floor_retuning": False,
            "positivity_floor_retuning": False,
            "source_relaxation_retuning": False,
            "transport_parameter_retuning": False,
            "wall_model_retuning": False,
            "normalization_retuning": False,
            "limiter_retuning": False,
            "velocity_quadrature_retuning": False,
            "failed_muscl_endpoint_rehabilitated": False,
            "one_sided_boundary_slope_promoted": False,
            "cross_knudsen_extension_permitted": False,
            "validation_claim_permitted": False,
            "solver_endpoint_claim_permitted": False,
        },
        "phi": phi,
        "psi": psi,
        "decision": decision,
        "scientific_conclusion": (
            "Stage 97 localizes where the material retained MUSCL correction magnitude resides and "
            "how that normalized spatial distribution shifts over the fixed Stage-96 diagnostic window. "
            "Because the parent maps contain weighted absolute correction magnitude only, this stage "
            "cannot determine signed x/y cancellation, causality, stability, or accuracy; any directional "
            "operator follow-up must preserve the frozen physics and numerical design."
        ),
        "negative_result_guard": (
            "Stage 90 remains nonconverged in both reconstruction arms, Stage 28 remains a failed MUSCL "
            "endpoint, and the Stage-89 one-sided boundary slope is not promoted. The Stage-97 spatial "
            "shares are diagnostics rather than proof that a wall or interior treatment is defective. "
            "No failed parameter is retuned, no cross-Knudsen extension is allowed, and no stable-solver, "
            "accuracy, benchmark, or validation claim is authorized."
        ),
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage96-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_stage97(args.stage96_artifact_dir, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
