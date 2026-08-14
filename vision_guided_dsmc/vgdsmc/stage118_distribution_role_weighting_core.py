from __future__ import annotations

import numpy as np

BANDS = ("near_1_4", "mid_5_14", "inner_15_28")
COMMON_COSINE_MIN = 0.95
COMMON_OVERLAP_MIN = 0.90

ALIGNED = "stage118_reduced_distribution_roles_align_stage119_role_weighted_spatial_colocation_audit"
INCOMPLETE = "stage118_energy_role_weighting_incomplete_stage119_exact_directional_moment_kernel_audit"
NONFINITE = "stage118_nonfinite_role_weighting_blocker_without_retuning"


def _normalize(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=float)
    s = float(x.sum())
    if (
        x.ndim != 1
        or x.size != 10
        or not np.isfinite(x).all()
        or np.any(x < 0)
        or not np.isfinite(s)
        or s <= 0
    ):
        raise ValueError("invalid radial profile")
    return x / s


def _metrics(p: np.ndarray, q: np.ndarray) -> dict[str, float | list[int]]:
    p = _normalize(p)
    q = _normalize(q)
    d = p - q
    den = max(float(np.linalg.norm(p) * np.linalg.norm(q)), 1e-300)
    return {
        "profile_cosine": float(np.dot(p, q) / den),
        "overlap_coefficient": float(np.minimum(p, q).sum()),
        "total_variation_distance": float(0.5 * np.abs(d).sum()),
        "transition_boundaries": [
            int(j)
            for j in range(9)
            if d[j] != 0 and d[j + 1] != 0 and d[j] * d[j + 1] < 0
        ],
        "phi_centroid_node": float(np.dot(p, np.arange(10, dtype=float))),
        "psi_centroid_node": float(np.dot(q, np.arange(10, dtype=float))),
    }


def analyze(phi: np.ndarray, psi: np.ndarray, speed: np.ndarray):
    p = np.asarray(phi, dtype=float)
    q = np.asarray(psi, dtype=float)
    r = np.asarray(speed, dtype=float)
    if (
        p.shape != (3, 10)
        or q.shape != (3, 10)
        or r.shape != (10,)
        or not np.isfinite(r).all()
        or np.any(r <= 0)
        or np.any(np.diff(r) <= 0)
    ):
        raise ValueError(
            "Stage 118 requires three ten-node profiles and strictly increasing positive node speeds"
        )

    role = []
    role_phi = []
    role_psi = []
    for i in range(3):
        pn = _normalize(p[i])
        qn = _normalize(q[i])
        raw = _metrics(pn, qn)

        # Fixed reduced-distribution energy roles, up to the common factor 1/2:
        # phi carries in-plane kinetic energy c_perp^2; psi already represents
        # the integrated transverse kinetic-energy moment and receives no
        # additional speed power. These powers are not fitted to Stage 117.
        pe = _normalize(pn * r * r)
        qe = qn.copy()
        weighted = _metrics(pe, qe)
        weighted["raw_profile_cosine"] = raw["profile_cosine"]
        weighted["raw_total_variation_distance"] = raw[
            "total_variation_distance"
        ]
        weighted["cosine_gain_from_role_weighting"] = float(
            weighted["profile_cosine"] - raw["profile_cosine"]
        )
        weighted["tv_reduction_fraction_from_role_weighting"] = float(
            (
                raw["total_variation_distance"]
                - weighted["total_variation_distance"]
            )
            / max(raw["total_variation_distance"], 1e-300)
        )
        role.append(weighted)
        role_phi.append(pe)
        role_psi.append(qe)

    finite = all(
        np.isfinite(
            [
                float(m[k])
                for k in (
                    "profile_cosine",
                    "overlap_coefficient",
                    "total_variation_distance",
                    "cosine_gain_from_role_weighting",
                    "tv_reduction_fraction_from_role_weighting",
                )
            ]
        ).all()
        for m in role
    )
    aligned = finite and all(
        float(m["profile_cosine"]) >= COMMON_COSINE_MIN
        and float(m["overlap_coefficient"]) >= COMMON_OVERLAP_MIN
        for m in role
    )
    decision = ALIGNED if aligned else (INCOMPLETE if finite else NONFINITE)
    aggregate = {
        "minimum_role_weighted_profile_cosine": float(
            min(m["profile_cosine"] for m in role)
        ),
        "minimum_role_weighted_overlap": float(
            min(m["overlap_coefficient"] for m in role)
        ),
        "maximum_role_weighted_total_variation": float(
            max(m["total_variation_distance"] for m in role)
        ),
        "minimum_cosine_gain_from_role_weighting": float(
            min(m["cosine_gain_from_role_weighting"] for m in role)
        ),
        "minimum_tv_reduction_fraction_from_role_weighting": float(
            min(m["tv_reduction_fraction_from_role_weighting"] for m in role)
        ),
    }
    metrics = {BANDS[i]: role[i] for i in range(3)}
    return metrics, aggregate, decision, np.asarray(role_phi), np.asarray(role_psi)
