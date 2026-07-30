import json
import numpy as np
import pytest

from vgdsmc.reference_adapter import (
    build_supervised_reference_case,
    deterministic_error_map,
    load_reference_npz,
    quantile_labels,
)


def synthetic_fields(shape=(6, 6)):
    y, x = np.indices(shape)
    return {
        "T": 300.0 + 2.0 * x,
        "rho": 1.0 + 0.01 * y,
        "u": 10.0 + 0.2 * x,
        "v": -2.0 + 0.1 * y,
    }


def test_reference_contract_and_error_map(tmp_path):
    reference = synthetic_fields()
    path = tmp_path / "dvm_reference.npz"
    np.savez_compressed(path, **reference)
    loaded = load_reference_npz(path, expected_shape=(6, 6))
    score = deterministic_error_map(loaded, loaded)
    assert np.allclose(score, 0.0)


def test_reference_contract_rejects_missing_field(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, T=np.ones((2, 2)), rho=np.ones((2, 2)))
    with pytest.raises(ValueError, match="missing fields"):
        load_reference_npz(path)


def test_quantile_labels_are_balanced_for_unique_scores():
    score = np.arange(36, dtype=float).reshape(6, 6)
    label, thresholds = quantile_labels(score)
    assert np.bincount(label.ravel(), minlength=3).tolist() == [12, 12, 12]
    assert thresholds[0] < thresholds[1]


def test_build_supervised_reference_case(tmp_path):
    reference = synthetic_fields()
    coarse = {key: value.copy() for key, value in reference.items()}
    coarse["T"][:, 3:] += 20.0
    x = np.stack(
        [coarse["T"], coarse["u"], coarse["v"], np.ones((6, 6))],
        axis=0,
    ).astype(np.float32)
    coarse_path = tmp_path / "coarse.npz"
    np.savez_compressed(
        coarse_path,
        x=x,
        **{f"coarse_{key}": value for key, value in coarse.items()},
    )
    reference_path = tmp_path / "reference.npz"
    np.savez_compressed(reference_path, **reference)
    output = build_supervised_reference_case(
        coarse_path,
        reference_path,
        tmp_path / "supervised.npz",
    )
    with np.load(output) as data:
        assert data["x"].shape == (4, 6, 6)
        assert data["score"].shape == (6, 6)
        assert set(np.unique(data["label"])) == {0, 1, 2}
        assert float(data["score"][:, 4:].mean()) > float(data["score"][:, :2].mean())
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert sum(metadata["class_counts"]) == 36
