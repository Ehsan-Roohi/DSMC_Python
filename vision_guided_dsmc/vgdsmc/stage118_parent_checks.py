from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

STAGE117_RUN_ID = 31748875063
STAGE117_JOB_ID = 94609793461
STAGE117_ARTIFACT_ID = 9211647861
STAGE117_ARTIFACT_SHA256 = "742226adf7b15d0d4e22ad1124fb39eac26deeaac9bac9d610af015b6f56aba4"
STAGE117_SUMMARY_SHA256 = "d3176700d7a03e5c0c10ac9429c4330802a4f6ef11c411739ae6bcd86f10b1f6"
STAGE117_PROFILES_SHA256 = "8e73a6afff8021bd15e26dc9de6bd122ae6e2d8b0494015d07f6aad46c37b86e"
STAGE117_SOURCE_HEAD = "d61f5eb677f1580b563fe62691a11fb535d707df"
STAGE117_DECISION = "stage117_stable_single_radial_transition_stage118_distribution_role_weighting_audit"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check(parent_dir: str | Path, record_path: str | Path) -> None:
    parent = Path(parent_dir)
    record = json.loads(Path(record_path).read_text())
    required = (
        record.get("stage") == 117,
        record.get("source_head") == STAGE117_SOURCE_HEAD,
        record.get("workflow_status") == "completed",
        record.get("workflow_conclusion") == "success",
        record.get("workflow_run_id") == STAGE117_RUN_ID,
        record.get("workflow_job_id") == STAGE117_JOB_ID,
        record.get("artifact_id") == STAGE117_ARTIFACT_ID,
        record.get("artifact_sha256") == STAGE117_ARTIFACT_SHA256,
        record.get("summary_sha256") == STAGE117_SUMMARY_SHA256,
        record.get("profiles_sha256") == STAGE117_PROFILES_SHA256,
        record.get("decision") == STAGE117_DECISION,
        record.get("tests", {}).get("passed") == 3,
        record.get("tests", {}).get("failed") == 0,
    )
    if not all(required):
        raise ValueError("Committed Stage-117 result record does not authorize Stage 118")

    summary_path = parent / "summary.json"
    profiles_path = parent / "radial_transition_profiles.npz"
    if _sha256(summary_path) != STAGE117_SUMMARY_SHA256:
        raise ValueError("Stage-117 summary checksum mismatch")
    if _sha256(profiles_path) != STAGE117_PROFILES_SHA256:
        raise ValueError("Stage-117 profile checksum mismatch")

    summary = json.loads(summary_path.read_text())
    if (
        summary.get("stage") != 117
        or summary.get("finite") is not True
        or summary.get("decision") != STAGE117_DECISION
    ):
        raise ValueError("Stage-117 artifact decision mismatch")

    with np.load(profiles_path) as data:
        phi = np.asarray(data["phi"], dtype=float)
        psi = np.asarray(data["psi"], dtype=float)
        speed = np.asarray(data["node_speed_mean"], dtype=float)
    if (
        phi.shape != (3, 10)
        or psi.shape != (3, 10)
        or speed.shape != (10,)
        or not np.isfinite(phi).all()
        or not np.isfinite(psi).all()
        or not np.isfinite(speed).all()
        or np.any(speed <= 0)
        or np.any(np.diff(speed) <= 0)
    ):
        raise ValueError("Stage-117 artifact shape/finite guard failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_dir")
    parser.add_argument("record_path")
    args = parser.parse_args()
    check(args.parent_dir, args.record_path)


if __name__ == "__main__":
    main()
