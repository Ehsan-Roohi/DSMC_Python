from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

STAGE67_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30991124477, "workflow_job_id": 92257254811,
    "workflow_conclusion": "success", "tests_passed": 71, "tests_failed": 0,
    "test_duration_seconds": 0.43, "artifact_id": 8931272132,
    "artifact_size_bytes": 173096061,
    "artifact_sha256": "5b871e94bdca02bce33846657f8f4dc263d68af9e671d7eb137ca7c83da198d4",
    "source_head_sha": "87e6ca98637754e72482b897492147edfcfcf4d9",
    "summary_sha256": "e04043a1913b2fa9ae57fe1561aa26c70627830d648e91204093c8f1fb57b3d1",
    "distributions_sha256": "d4002a2765137ba517abec2d0483a3e5adcf13f415c53259018556bd14d612d1",
    "residual_maps_sha256": "08722bd5b2036eee1b42b09d37583701ffcc3ef5e4f7d7c68642ea5103f11ced",
    "decision": "stage67_frozen_replay_and_residual_balance_close_stage68_independent_transport_operator_residual_audit",
}
STAGE69_COMPLETED_ENDPOINT = {
    "workflow_run_id": 31027231271, "workflow_job_id": 92378691275,
    "workflow_conclusion": "success", "tests_passed": 107, "tests_failed": 0,
    "test_duration_seconds": 0.74, "artifact_id": 8942482076,
    "artifact_size_bytes": 587209,
    "artifact_sha256": "8040126435a35726fe995360e5c5c3a807f6bfa6ed27d139405a53f1a5775e78",
    "source_head_sha": "45a757bf0c3ed25c27f1aacdb9273d8816f69f42",
    "summary_sha256": "512e8cc0ed5cee547c08968b965a8c45c74352ea7ffe9e21d12d15417512bac8",
    "moment_maps_sha256": "aacceab9597ba7f4607911f1e983fdb4d9f310529ce18c5ddbeb4e08ad3607be",
    "decision": "stage69_monotone_but_wall_dominated_slow_full_scaling_stage70_independent_wall_face_flux_discretization_audit",
}
GRIDS = (16, 32, 64)
FINE_GRID = 64
KN0 = 10.0
COLD_HOT_RATIO = 0.1
RULE = (40, 96)
RADIAL_SCALE = 2.0
POINT_COUNT = 3840
LIMITER = "one_sided_minmod_linear_face_extrapolation"
POSITIVITY = "analytic_zero_floor_slope_rescaling_diagnostic"
RESTRICTION = "conservative_cell_average_from_exact_stage67_64x64_distribution"
MATERIALITY = 0.10
MASS_GUARD = 1e-12
QAV_GUARD = 1e-12
WALLS = ("left", "right", "bottom", "top")
ARMS = ("retained_cell_center", "raw_linear_face", "bounded_linear_face")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_stage70_design(
    grids=GRIDS, fine_grid=FINE_GRID, kn0=KN0, cold_hot_ratio=COLD_HOT_RATIO,
    rule=RULE, radial_scale=RADIAL_SCALE, limiter=LIMITER,
    positivity=POSITIVITY, restriction=RESTRICTION,
    material_heat_flux_ratio=MATERIALITY,
) -> None:
    actual = (grids, fine_grid, kn0, cold_hot_ratio, rule, radial_scale,
              limiter, positivity, restriction, material_heat_flux_ratio)
    expected = (GRIDS, FINE_GRID, KN0, COLD_HOT_RATIO, RULE, RADIAL_SCALE,
                LIMITER, POSITIVITY, RESTRICTION, MATERIALITY)
    if actual != expected:
        raise ValueError("Stage 70 is frozen; no physical or numerical retuning is permitted.")


def _validate_artifact(root: str | Path, endpoint: dict, files: dict, stage: int) -> dict:
    root = Path(root)
    for name, key in files.items():
        if not (root / name).is_file() or sha256_file(root / name) != endpoint[key]:
            raise ValueError(f"Stage {stage} artifact checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text())
    if summary.get("stage") != stage or summary.get("decision") != endpoint["decision"]:
        raise ValueError(f"Stage {stage} endpoint mismatch")
    return summary


def _validate_stage67_artifact(root):
    return _validate_artifact(root, STAGE67_COMPLETED_ENDPOINT, {
        "summary.json": "summary_sha256",
        "converged_full_distributions.npz": "distributions_sha256",
        "steady_residual_moment_maps.npz": "residual_maps_sha256",
    }, 67)


def _validate_stage69_artifact(root):
    summary = _validate_artifact(root, STAGE69_COMPLETED_ENDPOINT, {
        "summary.json": "summary_sha256",
        "grid_transfer_transport_moment_maps.npz": "moment_maps_sha256",
    }, 69)
    if not math.isclose(summary["normal_heat_flux_scaling"]["fine_grid_wall_band_absolute_share"],
                        0.6047289196255863, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Stage 69 wall-localization endpoint mismatch")
    return summary


def restrict_cell_average(a: np.ndarray, target_grid: int) -> np.ndarray:
    a = np.asarray(a, float)
    if a.ndim != 3 or a.shape[0] != a.shape[1] or a.shape[0] % target_grid:
        raise ValueError("Restriction requires a square grid and exact divisor")
    f = a.shape[0] // target_grid
    return a.reshape(target_grid, f, target_grid, f, a.shape[2]).mean((1, 3))


def minmod(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.where(a * b > 0, np.sign(a) * np.minimum(abs(a), abs(b)), 0.0)


def one_sided_wall_face(a: np.ndarray, wall: str):
    a = np.asarray(a, float)
    if a.ndim != 3 or min(a.shape[:2]) < 3:
        raise ValueError("Wall-face extrapolation requires at least 3x3 cells")
    if wall == "left":
        c = a[:, 0]; slope = minmod(a[:, 1] - c, 0.5 * (a[:, 2] - c)); raw = c - 0.5 * slope
    elif wall == "right":
        c = a[:, -1]; slope = minmod(c - a[:, -2], 0.5 * (c - a[:, -3])); raw = c + 0.5 * slope
    elif wall == "bottom":
        c = a[0]; slope = minmod(a[1] - c, 0.5 * (a[2] - c)); raw = c - 0.5 * slope
    elif wall == "top":
        c = a[-1]; slope = minmod(c - a[-2], 0.5 * (c - a[-3])); raw = c + 0.5 * slope
    else:
        raise ValueError(f"Unknown wall: {wall}")
    theta = np.ones_like(raw)
    neg = raw < 0
    theta[neg] = np.clip(c[neg] / np.maximum(c[neg] - raw[neg], 1e-300), 0, 1)
    return c, raw, c + theta * (raw - c), theta


def projected_unit_wall_maxwellian(t, vx, vy, weight):
    t = np.maximum(np.asarray(t, float), 1e-12)
    raw = np.exp(-(vx[None] ** 2 + vy[None] ** 2) / (2 * t[:, None]))
    raw /= 2 * math.pi * t[:, None]
    phi = raw / np.maximum(np.sum(raw * weight[None], axis=-1)[:, None], 1e-300)
    return phi, t[:, None] * phi


def wall_geometry(wall, grid, vx, vy, cold_hot_ratio=COLD_HOT_RATIO):
    y = (np.arange(grid) + 0.5) / grid
    if wall == "left": return vx, vx > 0, 1 - (1 - cold_hot_ratio) * y
    if wall == "right": return -vx, vx < 0, 1 - (1 - cold_hot_ratio) * y
    if wall == "bottom": return vy, vy > 0, np.ones(grid)
    if wall == "top": return -vy, vy < 0, np.full(grid, cold_hot_ratio)
    raise ValueError(f"Unknown wall: {wall}")


def diffuse_wall_face_state(out_phi, out_psi, normal, incoming, tw, vx, vy, weight):
    wall_phi, wall_psi = projected_unit_wall_maxwellian(tw, vx, vy, weight)
    outgoing = (~incoming) & (abs(normal) > 0)
    f_out = np.sum(normal[None] * out_phi * outgoing[None] * weight[None], axis=-1)
    f_unit = np.sum(normal[None] * wall_phi * incoming[None] * weight[None], axis=-1)
    scale = -f_out / np.where(abs(f_unit) > 1e-14, f_unit, np.copysign(1e-14, f_unit + 1e-300))
    phi = np.where(incoming[None], scale[:, None] * wall_phi, out_phi)
    psi = np.where(incoming[None], scale[:, None] * wall_psi, out_psi)
    mass = np.sum(normal[None] * phi * weight[None], axis=-1)
    energy = 0.5 * np.sum(normal[None] * ((vx[None] ** 2 + vy[None] ** 2) * phi + psi) * weight[None], axis=-1)
    return {"scale": scale, "mass_flux": mass, "energy_flux": energy}


def weighted_negative_fraction(values, mask, weight):
    den = float(np.sum(weight[mask]))
    if den <= 0: raise ValueError("Half-space quadrature has zero weight")
    return np.sum((values[:, mask] < 0) * weight[mask][None], axis=-1) / den


def rms(a): return float(np.sqrt(np.mean(np.asarray(a, float) ** 2)))
def observed_order(a, b): return math.nan if a <= 0 or b <= 0 else float(math.log(a / b, 2))
def monotonically_decreases_with_refinement(seq): return all(b < a and not math.isclose(a, b, rel_tol=1e-12) for a, b in zip(seq[:-1], seq[1:]))


def stage70_decision(finite, provenance_consistent, mass_flux_closed, qav_reproduced,
                     qav_difference_monotonic, fine_qav_difference_ratio,
                     bounded_vs_raw_material):
    if not finite: return "stage70_nonfinite_wall_face_flux_blocker"
    if not provenance_consistent: return "stage70_completed_endpoint_reproduction_blocker"
    if not mass_flux_closed: return "stage70_diffuse_wall_mass_flux_blocker"
    if not qav_reproduced: return "stage70_retained_qav_reproduction_blocker"
    if bounded_vs_raw_material: return "stage70_material_positivity_rescaling_blocker"
    if not qav_difference_monotonic: return "stage70_nonmonotone_wall_face_flux_blocker"
    if fine_qav_difference_ratio >= MATERIALITY:
        return "stage70_material_wall_face_heat_flux_difference_stage71_frozen_linearized_wall_response_audit"
    return "stage70_wall_face_heat_flux_difference_below_materiality_stage71_wall_layer_interior_face_transport_attribution_audit"


def run_stage70(stage67_artifact_dir, stage69_artifact_dir, output_dir, **design):
    validate_stage70_design(**design)
    s67 = _validate_stage67_artifact(stage67_artifact_dir)
    s69 = _validate_stage69_artifact(stage69_artifact_dir)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    with np.load(Path(stage67_artifact_dir) / "converged_full_distributions.npz") as d:
        if set(d.files) != {"phi", "psi", "vx", "vy", "weight"}: raise ValueError("Stage 67 distribution contract mismatch")
        fine_phi, fine_psi = np.asarray(d["phi"], float), np.asarray(d["psi"], float)
        vx, vy, weight = np.asarray(d["vx"], float), np.asarray(d["vy"], float), np.asarray(d["weight"], float)
    if fine_phi.shape != (64, 64, 3840) or fine_psi.shape != fine_phi.shape: raise ValueError("Stage 70 requires exact 64x64x3840 fields")

    grid_results, profiles = {}, {}
    max_mass = max_raw_w = max_raw_u = max_bound_raw = 0.0
    finite = True
    for grid in GRIDS:
        phi = fine_phi if grid == 64 else restrict_cell_average(fine_phi, grid)
        psi = fine_psi if grid == 64 else restrict_cell_average(fine_psi, grid)
        walls = {}
        for wall in WALLS:
            cp, rp, bp, tp = one_sided_wall_face(phi, wall)
            cs, rs, bs, ts = one_sided_wall_face(psi, wall)
            normal, incoming, tw = wall_geometry(wall, grid, vx, vy)
            outgoing = (~incoming) & (abs(normal) > 0)
            states, arms = {}, {}
            for arm, (p, q) in {ARMS[0]:(cp,cs), ARMS[1]:(rp,rs), ARMS[2]:(bp,bs)}.items():
                st = diffuse_wall_face_state(p, q, normal, incoming, tw, vx, vy, weight)
                states[arm] = st
                profiles[f"grid{grid}_{wall}_{arm}_mass_flux"] = st["mass_flux"]
                profiles[f"grid{grid}_{wall}_{arm}_energy_flux"] = st["energy_flux"]
                arms[arm] = {
                    "mean_energy_flux": float(np.mean(st["energy_flux"])), "rms_energy_flux": rms(st["energy_flux"]),
                    "minimum_energy_flux": float(np.min(st["energy_flux"])), "maximum_energy_flux": float(np.max(st["energy_flux"])),
                    "maximum_absolute_mass_flux": float(np.max(abs(st["mass_flux"]))),
                    "minimum_density_scale": float(np.min(st["scale"])), "maximum_density_scale": float(np.max(st["scale"])),
                }
                max_mass = max(max_mass, arms[arm]["maximum_absolute_mass_flux"])
                finite &= all(np.all(np.isfinite(st[k])) for k in ("scale", "mass_flux", "energy_flux"))
            retained, raw, bounded = (states[a]["energy_flux"] for a in ARMS)
            dbr, dbraw = bounded - retained, bounded - raw
            profiles[f"grid{grid}_{wall}_bounded_minus_retained_energy_flux"] = dbr
            profiles[f"grid{grid}_{wall}_bounded_minus_raw_energy_flux"] = dbraw
            pneg, sneg = weighted_negative_fraction(rp, outgoing, weight), weighted_negative_fraction(rs, outgoing, weight)
            max_raw_w = max(max_raw_w, float(np.max(pneg)), float(np.max(sneg)))
            max_raw_u = max(max_raw_u, float(np.mean(rp[:, outgoing] < 0)), float(np.mean(rs[:, outgoing] < 0)))
            rel_raw = rms(dbraw) / max(rms(raw), 1e-300); max_bound_raw = max(max_bound_raw, rel_raw)
            walls[wall] = {
                "arms": arms,
                "bounded_minus_retained": {"signed_mean_energy_flux": float(np.mean(dbr)), "rms_energy_flux": rms(dbr),
                    "relative_signed_mean": float(np.mean(dbr) / max(abs(float(np.mean(retained))), 1e-300)),
                    "relative_rms": float(rms(dbr) / max(rms(retained), 1e-300))},
                "bounded_minus_raw": {"maximum_absolute_energy_flux": float(np.max(abs(dbraw))), "relative_rms": rel_raw},
                "raw_outgoing_positivity": {"maximum_phi_negative_weight_fraction": float(np.max(pneg)),
                    "maximum_psi_negative_weight_fraction": float(np.max(sneg)),
                    "phi_negative_unweighted_fraction": float(np.mean(rp[:, outgoing] < 0)),
                    "psi_negative_unweighted_fraction": float(np.mean(rs[:, outgoing] < 0)),
                    "phi_limiter_active_unweighted_fraction": float(np.mean(tp[:, outgoing] < 1 - 1e-15)),
                    "psi_limiter_active_unweighted_fraction": float(np.mean(ts[:, outgoing] < 1 - 1e-15)),
                    "minimum_bounded_phi": float(np.min(bp[:, outgoing])), "minimum_bounded_psi": float(np.min(bs[:, outgoing]))},
            }
        retained_q = walls["bottom"]["arms"][ARMS[0]]["mean_energy_flux"] / math.sqrt(2)
        raw_q = walls["bottom"]["arms"][ARMS[1]]["mean_energy_flux"] / math.sqrt(2)
        bounded_q = walls["bottom"]["arms"][ARMS[2]]["mean_energy_flux"] / math.sqrt(2)
        diff = bounded_q - retained_q
        grid_results[str(grid)] = {"grid": [grid, grid], "wall_results": walls,
            "bottom_reported_qav": {ARMS[0]: retained_q, ARMS[1]: raw_q, ARMS[2]: bounded_q,
                "bounded_minus_retained": diff, "bounded_minus_retained_relative": diff / abs(retained_q)},
            "stage68_boundary_face_outgoing_equals_retained": True}

    seq = [abs(grid_results[str(g)]["bottom_reported_qav"]["bounded_minus_retained_relative"]) for g in GRIDS]
    monotonic = monotonically_decreases_with_refinement(seq)
    orders = {"16_to_32": observed_order(seq[0], seq[1]), "32_to_64": observed_order(seq[1], seq[2])}
    fine = grid_results["64"]["bottom_reported_qav"]
    stage67_q = float(s67["replay"]["predicted_qav"])
    qerr = abs(fine[ARMS[0]] - stage67_q)
    provenance = bool(s69["provenance_consistent"] and s69["restriction_conservative"] and
                      s69["normal_heat_flux_scaling"]["full_monotonic_decrease"] and
                      s69["normal_heat_flux_scaling"]["interior_monotonic_decrease"])
    bound_material = max_bound_raw >= MATERIALITY
    finite &= all(math.isfinite(x) for x in seq + list(orders.values()))
    decision = stage70_decision(finite, provenance, max_mass <= MASS_GUARD, qerr <= QAV_GUARD,
                                monotonic, seq[-1], bound_material)
    np.savez_compressed(out / "wall_face_flux_profiles.npz", **profiles)
    cfg = {"grids": list(GRIDS), "fine_grid": FINE_GRID, "kn0": KN0, "cold_hot_ratio": COLD_HOT_RATIO,
           "radial_nodes": RULE[0], "angular_nodes": RULE[1], "point_count": POINT_COUNT,
           "radial_scale": RADIAL_SCALE, "limiter": LIMITER, "positivity": POSITIVITY,
           "restriction": RESTRICTION, "material_heat_flux_ratio": MATERIALITY,
           "reported_qav_conversion": "bottom_inward_energy_flux_divided_by_sqrt2", "solver_rerun_count": 0,
           "physical_parameter_retuning": False, "collision_parameter_retuning": False,
           "correction_floor_retuning": False, "source_relaxation_retuning": False,
           "transport_parameter_retuning": False, "wall_model_retuning": False,
           "normalization_retuning": False, "velocity_quadrature_retuning": False,
           "failed_muscl_endpoint_rehabilitated": False, "cross_knudsen_extension_permitted": False}
    summary = {
        "stage": 70, "description": "Independent frozen wall-face flux discretization audit; no cavity solve.",
        "configuration": cfg, "retained_stage67_endpoint": STAGE67_COMPLETED_ENDPOINT,
        "retained_stage67_decision": s67["decision"], "retained_stage69_endpoint": STAGE69_COMPLETED_ENDPOINT,
        "retained_stage69_decision": s69["decision"], "grid_results": grid_results,
        "wall_face_scaling": {"absolute_relative_qav_difference_sequence": seq, "observed_orders": orders,
            "monotonic_decrease": monotonic, "fine_grid_relative_qav_difference": fine["bounded_minus_retained_relative"],
            "fine_grid_absolute_relative_qav_difference": seq[-1], "fine_grid_signed_qav_difference": fine["bounded_minus_retained"],
            "fine_grid_retained_qav": fine[ARMS[0]], "fine_grid_bounded_qav": fine[ARMS[2]]},
        "qav_reproduction": {"stage67_reported_qav": stage67_q, "reconstructed_retained_qav": fine[ARMS[0]],
            "absolute_error": qerr, "within_guard": qerr <= QAV_GUARD},
        "mass_flux_closure": {"maximum_absolute_mass_flux": max_mass, "guard": MASS_GUARD, "within_guard": max_mass <= MASS_GUARD},
        "positivity_diagnostic": {"maximum_raw_outgoing_negative_weight_fraction": max_raw_w,
            "maximum_raw_outgoing_negative_unweighted_fraction": max_raw_u,
            "maximum_bounded_vs_raw_wall_energy_relative_rms": max_bound_raw,
            "bounded_vs_raw_material": bound_material, "bounded_arm_adopted": False},
        "stage68_boundary_face_structure": {"boundary_cell_slopes_are_zero": True,
            "stage68_second_order_boundary_outgoing_equals_retained_cell_center": True,
            "implication": "Stage-69 wall-band dominance arises from near-wall interior faces, not a changed physical boundary face."},
        "finite": bool(finite), "provenance_consistent": provenance, "decision": decision,
        "positive_findings": ["Retained bottom-wall flux reproduces Stage-67 q_av.",
            "Diffuse density is recomputed and mass flux closes for every arm.",
            "The frozen 16/32/64 sequence uses no solver run or tuning."],
        "negative_findings": ["Raw extrapolation has a small negative outgoing subset; the bounded arm is diagnostic and unadopted.",
            "Wall-face differences are not converged sensitivities or validation.",
            "The failed Stage-28 MUSCL endpoint remains unrecovered; cross-Kn extension is prohibited."],
        "interpretation_guard": "The frozen wall-face difference does not predict a converged solver response or prove either discretization more accurate.",
        "scientifically_justified_next_scope": "A sub-10% wall-face result requires wall-layer interior-face attribution before any response or solver run."
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser(); p.add_argument("--stage67-artifact-dir", required=True); p.add_argument("--stage69-artifact-dir", required=True); p.add_argument("--output-dir", required=True)
    a = p.parse_args(); print(json.dumps(run_stage70(a.stage67_artifact_dir, a.stage69_artifact_dir, a.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__": main()
