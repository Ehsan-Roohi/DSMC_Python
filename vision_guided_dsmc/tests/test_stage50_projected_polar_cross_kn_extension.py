from __future__ import annotations

import copy
import pytest

from vgdsmc.stage50_projected_polar_cross_kn_extension import (
    STAGE49_COMPLETED_ENDPOINT,
    STAGE49_RETAINED_0P1_CASE,
    STAGE50_GRID,
    STAGE50_KNUDSEN_NUMBERS,
    STAGE50_MAX_ITERATIONS,
    STAGE50_QAV_RELATIVE_ERROR_SCREEN,
    STAGE50_RATIO,
    STAGE50_RULE,
    STAGE50_SIGN_AGREEMENT_SCREEN,
    STAGE50_SOURCE_RELAXATION,
    STAGE50_TOLERANCE,
    STAGE50_VELOCITY_RELATIVE_RMS_SCREEN,
    build_stage50_config,
    stage50_decision,
    validate_stage50_design,
)


def test_stage50_design_is_frozen() -> None:
    validate_stage50_design(
        STAGE50_GRID,
        STAGE50_RULE,
        STAGE50_KNUDSEN_NUMBERS,
        STAGE50_RATIO,
        STAGE50_MAX_ITERATIONS,
        STAGE50_TOLERANCE,
        STAGE50_SOURCE_RELAXATION,
    )
    with pytest.raises(ValueError):
        validate_stage50_design(
            (56, 56), STAGE50_RULE, STAGE50_KNUDSEN_NUMBERS,
            STAGE50_RATIO, STAGE50_MAX_ITERATIONS, STAGE50_TOLERANCE,
            STAGE50_SOURCE_RELAXATION,
        )
    with pytest.raises(ValueError):
        validate_stage50_design(
            STAGE50_GRID, (40, 96), STAGE50_KNUDSEN_NUMBERS,
            STAGE50_RATIO, STAGE50_MAX_ITERATIONS, STAGE50_TOLERANCE,
            STAGE50_SOURCE_RELAXATION,
        )
    with pytest.raises(ValueError):
        validate_stage50_design(
            STAGE50_GRID, STAGE50_RULE, (0.1, 1.0, 10.0),
            STAGE50_RATIO, STAGE50_MAX_ITERATIONS, STAGE50_TOLERANCE,
            STAGE50_SOURCE_RELAXATION,
        )


def test_stage50_configs_change_only_knudsen_number() -> None:
    cfg1 = build_stage50_config(1.0)
    cfg10 = build_stage50_config(10.0)
    for cfg, kn0 in ((cfg1, 1.0), (cfg10, 10.0)):
        assert (cfg.nx, cfg.ny) == (64, 64)
        assert cfg.kn0 == kn0
        assert cfg.cold_hot_ratio == 0.1
        assert cfg.viscosity_exponent == 0.5
        assert cfg.prandtl == 2.0 / 3.0
        assert cfg.max_steps == 3000
        assert cfg.cfl == 0.2
        assert cfg.tolerance == 2.0e-5
        assert cfg.check_interval == 25
        assert cfg.minimum_steps == 500
        assert cfg.positivity_floor == 1.0e-30
    with pytest.raises(ValueError):
        build_stage50_config(0.1)


def test_stage49_provenance_is_exactly_retained() -> None:
    endpoint = STAGE49_COMPLETED_ENDPOINT
    assert endpoint["workflow_run_id"] == 30767671512
    assert endpoint["workflow_job_id"] == 91549183689
    assert endpoint["workflow_conclusion"] == "success"
    assert endpoint["tests_passed"] == 57
    assert endpoint["tests_failed"] == 0
    assert endpoint["test_duration_seconds"] == 0.25
    assert endpoint["artifact_id"] == 8841966863
    assert endpoint["artifact_size_bytes"] == 136455
    assert endpoint["artifact_sha256"] == (
        "1955c345571e08dd338c324aef9febf2098ddc728395fd4ab08b77420e91023c"
    )
    assert endpoint["source_head_sha"] == (
        "2bd8ae07fd8aaf007e724bba1fc254ca5cd8166b"
    )
    assert endpoint["decision"] == (
        "projected_polar_64x64_converging_stage50_cross_kn_extension"
    )


def test_stage49_kn0_0p1_endpoint_is_exactly_retained() -> None:
    retained = STAGE49_RETAINED_0P1_CASE
    assert retained["kn0"] == 0.1
    assert retained["grid"] == [64, 64]
    assert retained["iterations"] == 1550
    assert retained["converged"] is True
    assert retained["final_change"] == 1.673947904379247e-05
    assert retained["predicted_qav"] == 0.07583282640214684
    assert retained["literature_qav"] == 0.072
    assert retained["qav_relative_error"] == 0.05323370002981724
    assert retained["velocity_metrics"]["relative_rms"] == 0.34777209032506057
    assert retained["velocity_metrics"]["relative_l1"] == 0.36774617638417995
    assert retained["velocity_metrics"]["sign_agreement"] == 0.8
    assert retained["wall_mass_balance_relative_error"] == 2.051199878905107e-16
    assert retained["maximum_phi_clipped_weight_fraction"] == 0.003122144949217257
    assert retained["maximum_psi_clipped_weight_fraction"] == 0.005236378288354882


def _case(
    kn0: float,
    qerr: float,
    vrms: float,
    sign: float,
    *,
    converged: bool = True,
    stable: bool = True,
) -> dict[str, object]:
    case = copy.deepcopy(STAGE49_RETAINED_0P1_CASE)
    case["kn0"] = kn0
    case["qav_relative_error"] = qerr
    case["velocity_metrics"]["relative_rms"] = vrms
    case["velocity_metrics"]["sign_agreement"] = sign
    case["converged"] = converged
    if not stable:
        case["finite"] = False
    return case


def test_stage50_decision_retains_positive_mixed_and_negative_endpoints() -> None:
    baseline = copy.deepcopy(STAGE49_RETAINED_0P1_CASE)

    consistent = [baseline, _case(1.0, 0.04, 0.30, 1.0), _case(10.0, 0.06, 0.40, 0.8)]
    assert stage50_decision(consistent) == (
        "projected_polar_cross_kn_consistent_"
        "stage51_velocity_resolution_confirmation"
    )

    heat_only = [baseline, _case(1.0, 0.04, 0.70, 1.0), _case(10.0, 0.06, 0.40, 0.8)]
    assert stage50_decision(heat_only) == (
        "projected_polar_cross_kn_heat_flux_consistent_velocity_unresolved_"
        "stage51_wall_profile_audit"
    )

    velocity_only = [baseline, _case(1.0, 0.14, 0.30, 1.0), _case(10.0, 0.06, 0.40, 0.8)]
    assert stage50_decision(velocity_only) == (
        "projected_polar_cross_kn_velocity_consistent_heat_flux_unresolved_"
        "stage51_heat_flux_definition_audit"
    )

    mixed = [baseline, _case(1.0, 0.14, 0.70, 0.6), _case(10.0, 0.06, 0.40, 0.8)]
    assert stage50_decision(mixed) == (
        "projected_polar_cross_kn_mixed_or_negative_"
        "stage51_space_velocity_coupling_audit"
    )

    nonconverged = [baseline, _case(1.0, 0.04, 0.30, 1.0, converged=False), _case(10.0, 0.06, 0.40, 0.8)]
    assert stage50_decision(nonconverged) == (
        "stage50_cross_kn_stable_nonconverged_"
        "stage51_fixed_point_convergence_audit"
    )

    blocker = [baseline, _case(1.0, 0.04, 0.30, 1.0, stable=False), _case(10.0, 0.06, 0.40, 0.8)]
    assert stage50_decision(blocker) == (
        "stage50_cross_kn_numerical_blocker_"
        "stage51_projected_operator_stability_audit"
    )


def test_stage50_consistency_screens_are_preregistered() -> None:
    assert STAGE50_KNUDSEN_NUMBERS == (1.0, 10.0)
    assert STAGE50_QAV_RELATIVE_ERROR_SCREEN == 0.10
    assert STAGE50_VELOCITY_RELATIVE_RMS_SCREEN == 0.50
    assert STAGE50_SIGN_AGREEMENT_SCREEN == 0.80
