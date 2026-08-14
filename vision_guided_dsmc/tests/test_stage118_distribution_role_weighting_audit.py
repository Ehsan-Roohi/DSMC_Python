import json

import numpy as np
import pytest

from vgdsmc.stage118_distribution_role_weighting_audit import (
    run,
    validate_stage118_design,
)
from vgdsmc.stage118_distribution_role_weighting_core import (
    ALIGNED,
    INCOMPLETE,
    analyze,
)


def observed_like():
    phi = np.tile(
        np.array([0.21, 0.18, 0.11, 0.065, 0.10, 0.115, 0.096, 0.064, 0.036, 0.019]),
        (3, 1),
    )
    psi = np.tile(
        np.array([0.02, 0.04, 0.086, 0.135, 0.165, 0.166, 0.145, 0.112, 0.079, 0.052]),
        (3, 1),
    )
    return phi, psi, np.linspace(0.39, 1.85, 10)


def test_fixed_energy_role_weighting_moves_phi_to_higher_speed_nodes():
    phi, psi, speed = observed_like()
    metrics, _, decision, phi_role, psi_role = analyze(phi, psi, speed)
    assert decision == INCOMPLETE
    assert all(metrics[band]["profile_cosine"] > 0.85 for band in metrics)
    assert np.allclose(psi_role, psi / psi.sum(axis=1)[:, None])
    assert np.all(np.argmax(phi_role, axis=1) >= 4)


def test_exact_role_alignment_branch():
    speed = np.linspace(0.4, 1.8, 10)
    psi = np.tile(np.linspace(1.0, 2.0, 10), (3, 1))
    phi = psi / (speed * speed)[None, :]
    _, aggregate, decision, _, _ = analyze(phi, psi, speed)
    assert decision == ALIGNED
    assert aggregate["minimum_role_weighted_profile_cosine"] > 0.999999
    assert aggregate["minimum_role_weighted_overlap"] > 0.999999


def test_runner_and_artifact(tmp_path):
    phi, psi, speed = observed_like()
    parent = tmp_path / "parent"
    output = tmp_path / "output"
    parent.mkdir()
    (parent / "summary.json").write_text(
        json.dumps(
            {
                "stage": 117,
                "finite": True,
                "decision": "stage117_stable_single_radial_transition_stage118_distribution_role_weighting_audit",
            }
        )
    )
    np.savez_compressed(
        parent / "radial_transition_profiles.npz",
        phi=phi,
        psi=psi,
        node_speed_mean=speed,
    )
    result = run(parent, output)
    assert result["stage"] == 118
    assert result["decision"] == INCOMPLETE
    assert (output / "distribution_role_weighted_profiles.npz").exists()


def test_design_rejects_role_power_retuning():
    with pytest.raises(ValueError):
        validate_stage118_design(phi_role_speed_power=3)
