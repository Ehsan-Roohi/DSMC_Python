from vgdsmc.collision_validation import (
    FrequencyValidationConfig,
    RelaxationValidationConfig,
    run_frequency_validation,
    run_relaxation_validation,
)


def test_sbt_frequency_matches_exact_pair_expectation():
    result = run_frequency_validation(
        FrequencyValidationConfig(
            particles=60,
            trials=1500,
            fnum=450.0,
            seed=5,
        )
    )
    assert result["maximum_initial_candidate_probability"] < 0.08
    assert result["relative_error"] < 0.15


def test_anisotropic_distribution_relaxes_and_conserves_energy():
    result = run_relaxation_validation(
        RelaxationValidationConfig(
            particles=160,
            sweeps=80,
            seed=9,
        )
    )
    assert result["anisotropy_ratio"] < 0.35
    assert result["total_temperature_relative_change"] < 1.0e-12
    assert result["velocity_energy_relative_change"] < 1.0e-12
    assert result["mean_velocity_absolute_change"] < 1.0e-10
