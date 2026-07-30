import numpy as np

from vgdsmc.dvm_bgk import DVMReferenceConfig, save_dvm_reference, solve_dvm_reference
from vgdsmc.reference_adapter import build_supervised_reference_case


def test_isothermal_reference_is_nearly_uniform():
    cfg = DVMReferenceConfig(
        nx=8,
        ny=8,
        nv=10,
        t_left=300.0,
        t_right=300.0,
        t_top=300.0,
        t_bottom=300.0,
        max_steps=700,
        tolerance=3.0e-6,
    )
    result = solve_dvm_reference(cfg)
    assert abs(float(np.mean(result["T"])) - 300.0) < 1.0
    assert float(np.max(np.hypot(result["u"], result["v"]))) < 0.1
    assert float(np.max(np.abs(result["rho"] - 1.0))) < 2.0e-3


def test_hot_left_cold_right_ordering_and_convergence():
    cfg = DVMReferenceConfig(
        nx=10,
        ny=8,
        nv=10,
        t_left=330.0,
        t_right=270.0,
        t_top=300.0,
        t_bottom=300.0,
        max_steps=1200,
        tolerance=3.0e-6,
    )
    result = solve_dvm_reference(cfg)
    assert float(np.mean(result["T"][:, 0])) > float(np.mean(result["T"][:, -1])) + 10.0
    assert result["iterations"] < cfg.max_steps
    assert float(result["residual_history"][-1]) < 3.0e-6


def test_saved_contract_and_supervised_integration(tmp_path):
    cfg = DVMReferenceConfig(nx=6, ny=6, nv=8, max_steps=600, tolerance=5.0e-6)
    reference = save_dvm_reference(tmp_path / "reference.npz", cfg)
    with np.load(reference) as data:
        assert {"T", "rho", "u", "v"}.issubset(data.files)
        phase = np.sin(np.arange(36).reshape(6, 6))
        coarse = {
            "T": data["T"] * (1.0 + 0.02 * phase),
            "rho": data["rho"] * (1.0 + 0.02 * phase),
            "u": data["u"] + 0.2,
            "v": data["v"] - 0.1,
        }
        x = np.stack(
            [coarse["T"], coarse["u"], coarse["v"], np.full((6, 6), 0.1)]
        ).astype(np.float32)
        np.savez_compressed(
            tmp_path / "coarse.npz",
            x=x,
            **{f"coarse_{key}": value for key, value in coarse.items()},
        )
    output = build_supervised_reference_case(
        tmp_path / "coarse.npz", reference, tmp_path / "supervised.npz"
    )
    with np.load(output) as data:
        assert data["score"].shape == (6, 6)
        assert set(np.unique(data["label"])) == {0, 1, 2}
        assert np.all(np.isfinite(data["score"]))
