import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import monitor  # noqa: E402


def base_config(tmp: Path):
    return {
        "schema_version": 1,
        "cluster_name": "Unity test",
        "unity_user": "roohie_umass_edu",
        "project_root": "/project/pi_roohie_umass_edu",
        "work_root": "/work/pi_roohie_umass_edu/roohie_umass_edu",
        "runtime_dir": str(tmp / "runtime"),
        "history_event_limit": 100,
        "scan": {"lookback_days": 14, "max_jobs": 100, "max_artifact_matches": 5},
        "github": {
            "status_repo": str(tmp / "status"),
            "branch": "main",
            "push": False,
            "heartbeat_hours": 6,
        },
        "projects": [
            {
                "id": "sparta",
                "name": "SPARTA",
                "roots": ["/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK"],
                "job_name_patterns": ["sparta-*"],
                "artifact_checks": [],
            }
        ],
    }


class ParserTests(unittest.TestCase):
    def test_merge_prefers_live_queue_state(self):
        accounting = [
            {
                "job_id": "100",
                "job_name": "sparta-build",
                "state": "PENDING",
                "elapsed": "00:00:00",
                "source": "sacct",
            }
        ]
        live = [
            {
                "job_id": "100",
                "job_name": "sparta-build",
                "state": "RUNNING",
                "elapsed": "00:01:00",
                "reason": "cpu050",
                "source": "squeue",
            }
        ]
        jobs = monitor.merge_jobs(accounting, live)
        self.assertEqual(jobs[0]["state"], "RUNNING")
        self.assertEqual(jobs[0]["elapsed"], "00:01:00")

    def test_dependency_ids(self):
        self.assertEqual(monitor.dependency_ids("afterok:62606387,afterany:62606388_2"), ["62606387", "62606388"])

    def test_signature_ignores_elapsed_and_generation_time(self):
        report = {
            "generated_at": "2026-08-06T10:00:00Z",
            "warnings": [],
            "alerts": [],
            "projects": [{"id": "p", "status": "running", "counts": {"running": 1}, "validation": {}}],
            "jobs": [{"job_id": "1", "state": "RUNNING", "elapsed": "00:10", "project_id": "p"}],
        }
        changed_clock = copy.deepcopy(report)
        changed_clock["generated_at"] = "2026-08-06T10:15:00Z"
        changed_clock["jobs"][0]["elapsed"] = "00:25"
        self.assertEqual(monitor.report_signature(report), monitor.report_signature(changed_clock))
        changed_state = copy.deepcopy(report)
        changed_state["jobs"][0]["state"] = "COMPLETED"
        self.assertNotEqual(monitor.report_signature(report), monitor.report_signature(changed_state))


class ClassificationTests(unittest.TestCase):
    def test_later_success_resolves_same_job_name_failure(self):
        with tempfile.TemporaryDirectory() as td:
            config = base_config(Path(td))
            jobs = [
                {
                    "job_id": "102",
                    "job_name": "sparta-build",
                    "state": "COMPLETED",
                    "submit": "2026-08-06T11:00:00",
                    "work_dir": "/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK",
                },
                {
                    "job_id": "101",
                    "job_name": "sparta-build",
                    "state": "FAILED",
                    "submit": "2026-08-06T10:00:00",
                    "work_dir": "/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK",
                },
            ]
            projects, alerts = monitor.summarize_projects(jobs, config["projects"], config)
            self.assertEqual(projects[0]["status"], "completed_unverified")
            self.assertEqual(alerts, [])

    def test_successful_array_task_does_not_hide_failed_sibling(self):
        with tempfile.TemporaryDirectory() as td:
            config = base_config(Path(td))
            jobs = [
                {
                    "job_id": "500_0",
                    "job_name": "sparta-matrix",
                    "state": "COMPLETED",
                    "submit": "2026-08-06T11:00:00",
                    "work_dir": "/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK",
                },
                {
                    "job_id": "500_1",
                    "job_name": "sparta-matrix",
                    "state": "FAILED",
                    "submit": "2026-08-06T11:00:00",
                    "work_dir": "/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK",
                },
            ]
            projects, alerts = monitor.summarize_projects(jobs, config["projects"], config)
            self.assertEqual(projects[0]["status"], "attention")
            self.assertEqual([item["job_id"] for item in alerts], ["500_1"])

    def test_failed_dependency_is_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            config = base_config(Path(td))
            jobs = [
                {
                    "job_id": "200",
                    "job_name": "sparta-build",
                    "state": "FAILED",
                    "submit": "2026-08-06T10:00:00",
                    "work_dir": "/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK",
                },
                {
                    "job_id": "201",
                    "job_name": "sparta-pack",
                    "state": "PENDING",
                    "dependencies": "afterok:200",
                    "submit": "2026-08-06T10:01:00",
                    "work_dir": "/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK",
                },
            ]
            projects, alerts = monitor.summarize_projects(jobs, config["projects"], config)
            blocked = next(job for job in jobs if job["job_id"] == "201")
            self.assertEqual(blocked["category"], "blocked")
            self.assertEqual(blocked["failed_dependencies"], ["200"])
            self.assertEqual(projects[0]["status"], "attention")
            self.assertEqual(len(alerts), 2)

    def test_sanitization_removes_paths_and_tokens(self):
        with tempfile.TemporaryDirectory() as td:
            config = base_config(Path(td))
            source = "/project/pi_roohie_umass_edu/run github_pat_abcdefghijklmnopqrstuvwxyz123456"
            cleaned = monitor.sanitize(source, config)
            self.assertEqual(cleaned, "$PROJECT/run [REDACTED]")


class EndToEndTests(unittest.TestCase):
    def test_build_render_and_local_commit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = base_config(root)
            status = root / "status"
            status.mkdir()
            subprocess.run(["git", "init", "-b", "main", str(status)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(status), "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", str(status), "config", "user.email", "test@example.com"], check=True)
            sacct = [
                {
                    "job_id": "300",
                    "job_name": "sparta-run",
                    "state": "COMPLETED",
                    "exit_code": "0:0",
                    "submit": "2026-08-06T10:00:00",
                    "work_dir": "/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK",
                }
            ]
            with mock.patch.object(monitor, "collect_squeue", return_value=[]), mock.patch.object(
                monitor, "collect_sacct", return_value=(sacct, None)
            ):
                report = monitor.build_report(config)
            events = monitor.transition_events({}, report)
            result = monitor.publish_report(report, events, config, do_push=False)
            self.assertTrue(result["published"])
            self.assertFalse(result["pushed"])
            self.assertTrue((status / "README.md").is_file())
            self.assertTrue((status / "reports" / "status.json").is_file())
            self.assertTrue((status / "reports" / "dashboard.html").is_file())
            loaded = json.loads((status / "reports" / "status.json").read_text())
            self.assertEqual(loaded["projects"][0]["status"], "completed_unverified")


if __name__ == "__main__":
    unittest.main()
