import json
from pathlib import Path

import numpy as np
import pytest

from vgdsmc.stage116_pair_resolved_radial_node_audit import (
    RADIAL_NODES_PER_SHELL,
    PAIR_SECTORS,
    _load_stage115_record,
    _profile_metrics,
    radial_node_indices_within_shell,
    stage116_decision,
    validate_stage116_design,
)


def test_stage116_frozen_design_rejects_retuning():
    validate_stage116_design()
    with pytest.raises(ValueError):
        validate_stage116_design(kn0=9.0)
    with pytest.raises(ValueError):
        validate_stage116_design(pair_sectors=(4, 5))


def test_radial_node_grouping_has_exact_96_points_per_node():
    radii = np.linspace(0.4, 1.8, RADIAL_NODES_PER_SHELL)
    theta = np.arange(96, dtype=float) * (2.0 * np.pi / 96.0)
    vx = np.concatenate([r * np.cos(theta) for r in radii])
    vy = np.concatenate([r * np.sin(theta) for r in radii])
    labels = radial_node_indices_within_shell(vx, vy)
    assert labels.shape == (960,)
    assert [int(np.count_nonzero(labels == j)) for j in range(10)] == [96] * 10


def test_profile_metrics_identify_common_top_two_nodes():
    phi = np.array([0.03, 0.04, 0.05, 0.07, 0.08, 0.24, 0.29, 0.08, 0.07, 0.05])
    psi = np.array([0.04, 0.04, 0.05, 0.07, 0.08, 0.23, 0.28, 0.08, 0.07, 0.06])
    block = _profile_metrics(phi, psi)
    assert block["top2_sets_match"] is True
    assert set(block["joint_top2_node_index"]) == {5, 6}
    assert block["profile_cosine"] > 0.99


def _decision_block(profile, cosine=0.99, overlap=0.95):
    p = np.asarray(profile, dtype=float); p = p / p.sum(); top = np.argsort(p)[-2:][::-1]
    return {"profile_cosine": cosine, "overlap_coefficient": overlap, "top2_sets_match": True,
            "phi_top2_share": float(p[top].sum()), "psi_top2_share": float(p[top].sum()),
            "joint_top2_node_index": [int(v) for v in top],
            "phi_maximum_node_share": float(p.max()), "psi_maximum_node_share": float(p.max()),
            "phi_effective_node_count": float(1.0 / np.sum(p*p)), "psi_effective_node_count": float(1.0 / np.sum(p*p))}


def test_decision_routes():
    p = [0.03, 0.04, 0.05, 0.06, 0.07, 0.27, 0.29, 0.07, 0.06, 0.06]
    common = {"near_1_4": _decision_block(p), "mid_5_14": _decision_block(p)}
    assert stage116_decision(common, True, 0.0).startswith("stage116_common_pair_radial_nodes")
    diffuse = {"near_1_4": _decision_block(np.ones(10)), "mid_5_14": _decision_block(np.ones(10))}
    assert stage116_decision(diffuse, True, 0.0).startswith("stage116_pair_support_radially_diffuse")
    assert stage116_decision(diffuse, False, 0.0).startswith("stage116_nonfinite")
    assert stage116_decision(diffuse, True, 1e-6).startswith("stage116_stage115_pair_share_reconstruction")


def test_load_stage115_record_requires_exact_completed_provenance(tmp_path: Path):
    record = {"stage":115,"decision":"stage115_common_adjacent_pair_support_stage116_pair_resolved_radial_node_audit","finite":True,
      "source_head":"2d189102c0bca08c1f9d4a5a56daedd482fcd914","workflow_status":"completed","workflow_conclusion":"success",
      "workflow_run_id":31690647300,"workflow_job_id":94416925424,"artifact_id":9181151145,
      "artifact_sha256":"8245baedf9ad29db6c8b0d290e8507069479c867f449fc67d2b215a65edef3a1",
      "summary_sha256":"9c36830cc56cae1745075f57a0a0992656e09486bf5c90a93eb93e7f251b30e6",
      "distribution_specific_sector_profiles_sha256":"932a643786f94b08c6156fbf985d9c39b094f98532a2480190a4ded791bcf02a",
      "tests":{"passed":9,"failed":0},"metrics":{
        "near_1_4":{"joint_top2_sector_index":[5,6],"phi_top2_sector_index":[5,6],"psi_top2_sector_index":[5,6],"phi_top2_share":0.65,"psi_top2_share":0.59},
        "mid_5_14":{"joint_top2_sector_index":[5,6],"phi_top2_sector_index":[5,6],"psi_top2_sector_index":[5,6],"phi_top2_share":0.58,"psi_top2_share":0.54}}}
    path = tmp_path / "record.json"; path.write_text(json.dumps(record))
    assert _load_stage115_record(path)["stage"] == 115


def test_pair_sectors_are_frozen_to_stage115_common_pair():
    assert PAIR_SECTORS == (5, 6)
