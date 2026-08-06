#!/usr/bin/env python3
"""Read-only Slurm monitor for Ehsan Roohi's Unity research campaigns.

The collector intentionally uses only Python's standard library.  It never
submits, cancels, requeues, or modifies scientific jobs.  Its only write
targets are the configured runtime directory and the dedicated GitHub status
checkout.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import glob
import hashlib
import html
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


VERSION = "0.1.0"
UTC = dt.timezone.utc
FAILURE_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}
RUNNING_STATES = {"COMPLETING", "CONFIGURING", "RUNNING", "SUSPENDED"}
PENDING_STATES = {"PENDING", "REQUEUED", "REQUEUE_FED", "RESIZING"}
SUCCESS_STATES = {"COMPLETED"}
ERROR_RE = re.compile(
    r"(?i)(traceback|fatal|segmentation|segfault|out of memory|oom|cuda error|"
    r"exception|permission denied|no such file|invalid depend|killed|nan detected|"
    r"error(?:\s*:|\b))"
)
SECRET_PATTERNS = (
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)(authorization\s*:\s*(?:bearer|token)\s+)[^\s]+"),
    re.compile(r"(?i)(password\s*[=:]\s*)[^\s]+"),
    re.compile(r"https://[^/@\s:]+:[^/@\s]+@github\.com"),
)


class MonitorError(RuntimeError):
    pass


def utcnow() -> dt.datetime:
    return dt.datetime.now(tz=UTC)


def iso(value: Optional[dt.datetime] = None) -> str:
    return (value or utcnow()).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_time(value: str) -> Optional[dt.datetime]:
    if not value or value in {"Unknown", "N/A", "None"}:
        return None
    candidate = value.strip().replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(candidate)
    except ValueError:
        return None
    return parsed.replace(tzinfo=parsed.tzinfo or UTC).astimezone(UTC)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def load_json(path: Path, default: Any = None) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError as exc:
        raise MonitorError(f"Invalid JSON in {path}: {exc}") from exc


def dump_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def expand(value: str) -> str:
    return os.path.expanduser(os.path.expandvars(value))


def normalize_state(value: str) -> str:
    value = (value or "UNKNOWN").strip().upper()
    value = value.rstrip("+")
    if " " in value:
        value = value.split()[0]
    return value or "UNKNOWN"


def run_command(args: Sequence[str], timeout: int = 60, check: bool = True) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            list(args),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise MonitorError(f"Required command is missing: {args[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise MonitorError(f"Command timed out after {timeout}s: {args[0]}") from exc
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-1000:]
        raise MonitorError(f"{args[0]} failed ({result.returncode}): {detail}")
    return result


def parse_delimited(text: str, fields: Sequence[str], delimiter: str = "|") -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        parts = raw.rstrip("\n").split(delimiter)
        if parts and parts[-1] == "":
            parts.pop()
        if len(parts) < len(fields):
            parts.extend([""] * (len(fields) - len(parts)))
        if len(parts) > len(fields):
            parts = parts[: len(fields) - 1] + [delimiter.join(parts[len(fields) - 1 :])]
        rows.append(dict(zip(fields, parts)))
    return rows


def collect_squeue(user: str) -> List[Dict[str, str]]:
    fields = [
        "job_id",
        "array_job_id",
        "array_task_id",
        "job_name",
        "state",
        "elapsed",
        "time_limit",
        "partition",
        "reason",
        "nodes",
        "cpus",
        "min_memory",
        "tres_per_node",
        "work_dir",
        "dependencies",
    ]
    fmt = "%i|%A|%a|%j|%T|%M|%l|%P|%R|%D|%C|%m|%b|%Z|%E"
    result = run_command(["squeue", "-h", "-u", user, "-o", fmt])
    rows = parse_delimited(result.stdout, fields)
    for row in rows:
        row["state"] = normalize_state(row["state"])
        row["source"] = "squeue"
    return rows


def collect_sacct(user: str, lookback_days: int) -> Tuple[List[Dict[str, str]], Optional[str]]:
    start = (utcnow() - dt.timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    rich_fields = [
        ("JobIDRaw", "job_id_raw"),
        ("JobID", "job_id"),
        ("JobName%100", "job_name"),
        ("State", "state"),
        ("ExitCode", "exit_code"),
        ("Elapsed", "elapsed"),
        ("Start", "start"),
        ("End", "end"),
        ("Submit", "submit"),
        ("Partition", "partition"),
        ("NodeList", "node_list"),
        ("AllocTRES%160", "alloc_tres"),
        ("ReqMem", "req_mem"),
        ("MaxRSS", "max_rss"),
        ("WorkDir%300", "work_dir"),
        ("StdOut%300", "stdout_path"),
    ]
    basic_fields = rich_fields[:11]
    warning: Optional[str] = None
    for field_set in (rich_fields, basic_fields):
        command = [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-u",
            user,
            "-S",
            start,
            "--format=" + ",".join(item[0] for item in field_set),
        ]
        result = run_command(command, timeout=120, check=False)
        if result.returncode == 0:
            rows = parse_delimited(result.stdout, [item[1] for item in field_set])
            for row in rows:
                row["state"] = normalize_state(row.get("state", ""))
                row["source"] = "sacct"
            if field_set is basic_fields:
                warning = "sacct rich fields were unavailable; work directories and log excerpts are omitted"
            return rows, warning
    detail = (result.stderr or result.stdout).strip()[-1000:]
    raise MonitorError(f"sacct failed: {detail}")


def merge_jobs(sacct_rows: Sequence[Mapping[str, str]], squeue_rows: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for row in sacct_rows:
        job_id = row.get("job_id") or row.get("job_id_raw") or ""
        if not job_id or "." in job_id:
            continue
        current = dict(row)
        current["job_id"] = job_id
        merged[job_id] = current
    for row in squeue_rows:
        job_id = row.get("job_id", "")
        if not job_id:
            continue
        existing = merged.get(job_id, {})
        for key, value in row.items():
            if value or key in {"state", "reason"}:
                existing[key] = value
        existing["job_id"] = job_id
        existing["active"] = True
        merged[job_id] = existing
    jobs = list(merged.values())
    for job in jobs:
        job["state"] = normalize_state(str(job.get("state", "UNKNOWN")))
        job.setdefault("active", job["state"] in RUNNING_STATES | PENDING_STATES)
        job.setdefault("job_name", "unnamed")
        job.setdefault("reason", "")
        job.setdefault("exit_code", "")
    jobs.sort(key=job_sort_key, reverse=True)
    return jobs


def numeric_job_id(value: str) -> int:
    match = re.match(r"(\d+)", value or "")
    return int(match.group(1)) if match else -1


def job_sort_key(job: Mapping[str, Any]) -> Tuple[str, int]:
    return (
        str(job.get("submit") or job.get("start") or job.get("end") or ""),
        numeric_job_id(str(job.get("job_id", ""))),
    )


def canonical_job_name(value: str) -> str:
    value = re.sub(r"_\[.*\]$", "", value or "")
    value = re.sub(r"_\d+$", "", value)
    return value.lower()


def state_category(state: str) -> str:
    state = normalize_state(state)
    if state in FAILURE_STATES:
        return "attention"
    if state in RUNNING_STATES:
        return "running"
    if state in PENDING_STATES:
        return "pending"
    if state in SUCCESS_STATES:
        return "completed"
    return "other"


def dependency_ids(value: str) -> List[str]:
    return re.findall(r"(?<!\d)(\d+)(?:_[0-9]+)?", value or "")


def match_project(job: Mapping[str, Any], projects: Sequence[Mapping[str, Any]]) -> str:
    work_dir = str(job.get("work_dir") or "")
    job_name = str(job.get("job_name") or "")
    for project in projects:
        for root in project.get("roots", []):
            expanded = expand(str(root)).rstrip("/")
            if expanded and (work_dir == expanded or work_dir.startswith(expanded + "/")):
                return str(project["id"])
    for project in projects:
        for pattern in project.get("job_name_patterns", []):
            if fnmatch.fnmatch(job_name.lower(), str(pattern).lower()):
                return str(project["id"])
    return "unclassified"


def sanitize(text: Any, config: Mapping[str, Any]) -> str:
    value = str(text or "")
    replacements = {
        expand(str(config.get("project_root", ""))): "$PROJECT",
        expand(str(config.get("work_root", ""))): "$WORK",
        str(Path.home()): "$HOME",
    }
    for source, target in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        if source:
            value = value.replace(source, target)
    for pattern in SECRET_PATTERNS:
        if "github.com" in pattern.pattern:
            value = pattern.sub("https://***:***@github.com", value)
        else:
            value = pattern.sub(lambda m: (m.group(1) if m.lastindex else "") + "[REDACTED]", value)
    return value


def resolve_log_path(job: Mapping[str, Any]) -> Optional[Path]:
    value = str(job.get("stdout_path") or "")
    if not value:
        work_dir = str(job.get("work_dir") or "")
        if work_dir:
            value = str(Path(work_dir) / f"slurm-{job.get('job_id', '')}.out")
    substitutions = {
        "%j": str(job.get("job_id", "")),
        "%A": str(job.get("array_job_id") or job.get("job_id", "")),
        "%a": str(job.get("array_task_id") or ""),
        "%x": str(job.get("job_name") or ""),
    }
    for marker, replacement in substitutions.items():
        value = value.replace(marker, replacement)
    return Path(value) if value else None


def error_excerpt(job: Mapping[str, Any], config: Mapping[str, Any]) -> List[str]:
    if state_category(str(job.get("state", ""))) != "attention":
        return []
    path = resolve_log_path(job)
    if not path or not path.is_file():
        return []
    max_bytes = int(config.get("scan", {}).get("max_log_bytes", 262144))
    max_lines = int(config.get("scan", {}).get("max_error_lines", 12))
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    matches = [sanitize(line.strip(), config) for line in text.splitlines() if ERROR_RE.search(line)]
    return matches[-max_lines:]


def expand_artifact_pattern(pattern: str, project: Mapping[str, Any]) -> List[str]:
    roots = [expand(str(item)) for item in project.get("roots", [])]
    if "{root}" in pattern:
        return [expand(pattern.replace("{root}", root)) for root in roots]
    return [expand(pattern)]


def collect_artifacts(project: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    max_matches = int(config.get("scan", {}).get("max_artifact_matches", 20))
    for check in project.get("artifact_checks", []):
        matches: List[Path] = []
        for raw_pattern in check.get("paths", []):
            for pattern in expand_artifact_pattern(str(raw_pattern), project):
                for item in glob.iglob(pattern, recursive=True):
                    path = Path(item)
                    if path.is_file():
                        matches.append(path)
                    if len(matches) >= max_matches:
                        break
                if len(matches) >= max_matches:
                    break
            if len(matches) >= max_matches:
                break
        matches = sorted(set(matches), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        records = []
        for path in matches[:max_matches]:
            try:
                info = path.stat()
            except OSError:
                continue
            records.append(
                {
                    "path": sanitize(path, config),
                    "size_bytes": info.st_size,
                    "modified_at": iso(dt.datetime.fromtimestamp(info.st_mtime, tz=UTC)),
                }
            )
        checks.append(
            {
                "label": str(check.get("label", "artifact")),
                "required": bool(check.get("required", False)),
                "found": bool(records),
                "matches": records,
            }
        )
    required = [item for item in checks if item["required"]]
    validation_mode = str(project.get("validation_mode", "all")).lower()
    if not required:
        validated = False
    elif validation_mode == "any":
        validated = any(item["found"] for item in required)
    else:
        validated = all(item["found"] for item in required)
    return {"validated": validated, "checks": checks}


def job_attempt_id(job: Mapping[str, Any]) -> str:
    array_id = str(job.get("array_job_id") or "")
    if array_id and array_id not in {"N/A", "None"}:
        return array_id
    job_id = str(job.get("job_id") or "")
    return re.split(r"_", job_id, maxsplit=1)[0]


def most_recent_attempts_by_name(jobs: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """Return every task from the newest Slurm attempt for each job name.

    Keeping the whole newest array is important: one successful array task must
    not hide a failed sibling task.  A later array/job ID with the same name is
    treated as a retry and supersedes the older attempt.
    """
    families: Dict[str, Dict[str, List[Mapping[str, Any]]]] = {}
    for job in jobs:
        name = canonical_job_name(str(job.get("job_name", "")))
        attempt = job_attempt_id(job)
        families.setdefault(name, {}).setdefault(attempt, []).append(job)
    selected: List[Mapping[str, Any]] = []
    for attempts in families.values():
        newest_id = max(
            attempts,
            key=lambda attempt: max((job_sort_key(job) for job in attempts[attempt]), default=("", -1)),
        )
        selected.extend(attempts[newest_id])
    return selected


def summarize_projects(
    jobs: List[Dict[str, Any]], projects: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    known_ids = {str(project["id"]) for project in projects}
    project_map = {str(project["id"]): project for project in projects}
    project_map["unclassified"] = {
        "id": "unclassified",
        "name": "Unclassified Unity jobs",
        "description": "Active or recent jobs that do not yet match a configured campaign.",
    }
    alerts: List[Dict[str, Any]] = []
    job_index = {str(job.get("job_id", "")): job for job in jobs}
    for job in jobs:
        project_id = match_project(job, projects)
        job["project_id"] = project_id if project_id in known_ids else "unclassified"
        category = state_category(str(job.get("state", "")))
        dependencies = dependency_ids(str(job.get("dependencies", "")))
        failed_dependencies = [
            dep for dep in dependencies if dep in job_index and state_category(str(job_index[dep].get("state"))) == "attention"
        ]
        if category == "pending" and failed_dependencies:
            category = "blocked"
            job["failed_dependencies"] = failed_dependencies
        reason = str(job.get("reason", ""))
        if category == "pending" and re.search(r"(?i)(dependencynever|invaliddepend)", reason):
            category = "blocked"
        job["category"] = category
        job["error_excerpt"] = error_excerpt(job, config)

    summaries: List[Dict[str, Any]] = []
    for project_id, project in project_map.items():
        project_jobs = [job for job in jobs if job.get("project_id") == project_id]
        if not project_jobs and project_id == "unclassified":
            continue
        artifacts = collect_artifacts(project, config) if project_id != "unclassified" else {"validated": False, "checks": []}
        recent = most_recent_attempts_by_name(project_jobs)
        unresolved = [job for job in recent if job.get("category") in {"attention", "blocked"}]
        active = [job for job in project_jobs if job.get("category") in {"running", "pending", "blocked"}]
        counts: Dict[str, int] = {}
        for job in project_jobs:
            key = str(job.get("category", "other"))
            counts[key] = counts.get(key, 0) + 1
        if any(job.get("category") == "attention" for job in unresolved):
            status = "attention"
        elif any(job.get("category") == "blocked" for job in unresolved):
            status = "blocked"
        elif any(job.get("category") == "running" for job in active):
            status = "running"
        elif any(job.get("category") == "pending" for job in active):
            status = "pending"
        elif artifacts["validated"]:
            status = "validated"
        elif any(job.get("category") == "completed" for job in project_jobs):
            status = "completed_unverified"
        else:
            status = "no_jobs"
        summary = {
            "id": project_id,
            "name": str(project.get("name", project_id)),
            "description": str(project.get("description", "")),
            "status": status,
            "counts": counts,
            "job_count": len(project_jobs),
            "latest_job_id": project_jobs[0]["job_id"] if project_jobs else None,
            "validation": artifacts,
        }
        summaries.append(summary)
        max_alerts = int(config.get("scan", {}).get("max_alerts_per_project", 30))
        for job in unresolved[:max_alerts]:
            alert = {
                "severity": "critical" if job.get("category") == "attention" else "warning",
                "project_id": project_id,
                "project": summary["name"],
                "job_id": str(job.get("job_id", "")),
                "job_name": str(job.get("job_name", "")),
                "state": str(job.get("state", "")),
                "reason": sanitize(job.get("reason", ""), config),
                "exit_code": str(job.get("exit_code", "")),
                "error_excerpt": job.get("error_excerpt", []),
            }
            alerts.append(alert)
        if len(unresolved) > max_alerts:
            alerts.append(
                {
                    "severity": "warning",
                    "project_id": project_id,
                    "project": summary["name"],
                    "job_id": "multiple",
                    "job_name": "additional unresolved array tasks",
                    "state": "SUPPRESSED",
                    "reason": f"{len(unresolved) - max_alerts} additional unresolved tasks are in reports/status.json",
                    "exit_code": "",
                    "error_excerpt": [],
                }
            )
    order = {"attention": 0, "blocked": 1, "running": 2, "pending": 3, "completed_unverified": 4, "validated": 5, "no_jobs": 6}
    summaries.sort(key=lambda item: (order.get(str(item["status"]), 9), str(item["name"])))
    alerts.sort(key=lambda item: (0 if item["severity"] == "critical" else 1, item["project"], -numeric_job_id(item["job_id"])))
    return summaries, alerts


def public_job(job: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    allowed = [
        "job_id",
        "array_job_id",
        "array_task_id",
        "job_name",
        "state",
        "category",
        "exit_code",
        "elapsed",
        "time_limit",
        "partition",
        "reason",
        "nodes",
        "cpus",
        "min_memory",
        "tres_per_node",
        "node_list",
        "alloc_tres",
        "req_mem",
        "max_rss",
        "submit",
        "start",
        "end",
        "work_dir",
        "dependencies",
        "failed_dependencies",
        "project_id",
        "error_excerpt",
    ]
    result = {
        key: job[key]
        for key in allowed
        if key in job and job[key] is not None and job[key] != "" and job[key] != []
    }
    for key in ("reason", "work_dir", "node_list", "tres_per_node", "alloc_tres"):
        if key in result:
            result[key] = sanitize(result[key], config)
    return result


def build_report(config: Mapping[str, Any]) -> Dict[str, Any]:
    user = str(config.get("unity_user") or os.environ.get("USER") or "")
    if not user:
        raise MonitorError("unity_user is not configured")
    lookback = int(config.get("scan", {}).get("lookback_days", 14))
    warnings: List[str] = []
    squeue_rows: List[Dict[str, str]] = []
    sacct_rows: List[Dict[str, str]] = []
    try:
        squeue_rows = collect_squeue(user)
    except MonitorError as exc:
        warnings.append(str(exc))
    try:
        sacct_rows, sacct_warning = collect_sacct(user, lookback)
        if sacct_warning:
            warnings.append(sacct_warning)
    except MonitorError as exc:
        warnings.append(str(exc))
    if not squeue_rows and not sacct_rows and len(warnings) == 2:
        raise MonitorError("Neither squeue nor sacct could be collected: " + " | ".join(warnings))
    jobs = merge_jobs(sacct_rows, squeue_rows)
    projects, alerts = summarize_projects(jobs, config.get("projects", []), config)
    category_counts: Dict[str, int] = {}
    for job in jobs:
        category = str(job.get("category", "other"))
        category_counts[category] = category_counts.get(category, 0) + 1
    report = {
        "schema_version": 1,
        "generated_at": iso(),
        "monitor_version": VERSION,
        "cluster": str(config.get("cluster_name", "Unity")),
        "unity_user": user,
        "lookback_days": lookback,
        "summary": {
            "jobs": category_counts,
            "projects": {status: sum(1 for p in projects if p["status"] == status) for status in sorted({p["status"] for p in projects})},
            "alert_count": len(alerts),
        },
        "warnings": [sanitize(item, config) for item in warnings],
        "alerts": alerts,
        "projects": projects,
        "jobs": [public_job(job, config) for job in jobs[: int(config.get("scan", {}).get("max_jobs", 2000))]],
    }
    return report


def report_signature(report: Mapping[str, Any]) -> str:
    meaningful = {
        "warnings": report.get("warnings", []),
        "alerts": report.get("alerts", []),
        "projects": [
            {
                "id": item.get("id"),
                "status": item.get("status"),
                "counts": item.get("counts"),
                "validation": item.get("validation"),
            }
            for item in report.get("projects", [])
        ],
        "jobs": [
            {
                "job_id": item.get("job_id"),
                "state": item.get("state"),
                "reason": item.get("reason"),
                "exit_code": item.get("exit_code"),
                "project_id": item.get("project_id"),
            }
            for item in report.get("jobs", [])
        ],
    }
    encoded = json.dumps(meaningful, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def status_label(status: str) -> str:
    return {
        "attention": "🔴 Attention",
        "blocked": "🟠 Blocked",
        "running": "🔵 Running",
        "pending": "🟡 Pending",
        "completed_unverified": "🟣 Completed—unverified",
        "validated": "🟢 Validated",
        "no_jobs": "⚪ No recent jobs",
    }.get(status, status)


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Unity Watchtower",
        "",
        f"Last collection: **{report['generated_at']}**  ",
        f"Cluster: **{report['cluster']}** · Accounting window: **{report['lookback_days']} days** · Alerts: **{report['summary']['alert_count']}**",
        "",
        "This repository contains sanitized, read-only monitoring output. It does not contain GitHub tokens, Unity credentials, raw environment files, or job-control commands.",
        "",
        "## Projects",
        "",
        "| Project | Status | Running | Pending | Blocked | Failed | Completed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for project in report.get("projects", []):
        counts = project.get("counts", {})
        lines.append(
            "| {name} | {status} | {running} | {pending} | {blocked} | {attention} | {completed} |".format(
                name=str(project["name"]).replace("|", "\\|"),
                status=status_label(str(project["status"])),
                running=counts.get("running", 0),
                pending=counts.get("pending", 0),
                blocked=counts.get("blocked", 0),
                attention=counts.get("attention", 0),
                completed=counts.get("completed", 0),
            )
        )
    lines.extend(["", "## Items requiring attention", ""])
    if not report.get("alerts"):
        lines.append("No unresolved failed or dependency-blocked job families were found.")
    else:
        lines.extend(["| Project | Job | State | Reason |", "|---|---|---|---|"])
        for alert in report["alerts"]:
            reason = str(alert.get("reason") or alert.get("exit_code") or "—").replace("|", "\\|")
            lines.append(f"| {alert['project']} | `{alert['job_name']}` (`{alert['job_id']}`) | {alert['state']} | {reason} |")
    if report.get("warnings"):
        lines.extend(["", "## Collector warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `reports/status.json`: machine-readable source for Codex/ChatGPT",
            "- `reports/dashboard.html`: standalone dashboard",
            "- `reports/events.json`: recent job-state transitions",
            "",
        ]
    )
    return "\n".join(lines)


def render_html(report: Mapping[str, Any]) -> str:
    colors = {
        "attention": "#b42318",
        "blocked": "#b54708",
        "running": "#175cd3",
        "pending": "#a15c00",
        "completed_unverified": "#6941c6",
        "validated": "#067647",
        "no_jobs": "#667085",
    }
    project_rows = []
    for project in report.get("projects", []):
        counts = project.get("counts", {})
        color = colors.get(str(project["status"]), "#344054")
        project_rows.append(
            "<tr><td>{}</td><td><span class='pill' style='background:{}'>{}</span></td>"
            "<td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                html.escape(str(project["name"])),
                color,
                html.escape(str(project["status"])),
                counts.get("running", 0),
                counts.get("pending", 0),
                counts.get("blocked", 0),
                counts.get("attention", 0),
                counts.get("completed", 0),
            )
        )
    alert_rows = []
    for alert in report.get("alerts", []):
        excerpts = "<br>".join(html.escape(str(line)) for line in alert.get("error_excerpt", []))
        detail = html.escape(str(alert.get("reason") or alert.get("exit_code") or "—"))
        if excerpts:
            detail += "<details><summary>log excerpt</summary><code>" + excerpts + "</code></details>"
        alert_rows.append(
            "<tr><td>{}</td><td><code>{}</code><br>{}</td><td>{}</td><td>{}</td></tr>".format(
                html.escape(str(alert["project"])),
                html.escape(str(alert["job_id"])),
                html.escape(str(alert["job_name"])),
                html.escape(str(alert["state"])),
                detail,
            )
        )
    warning_html = "".join(f"<li>{html.escape(str(item))}</li>" for item in report.get("warnings", []))
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Unity Watchtower</title>
<style>
:root{color-scheme:light dark;font-family:Inter,system-ui,sans-serif}body{max-width:1200px;margin:0 auto;padding:28px;background:#f8fafc;color:#101828}h1{margin-bottom:4px}.sub{color:#475467;margin-top:0}.card{background:white;border:1px solid #e4e7ec;border-radius:12px;padding:20px;margin:18px 0;box-shadow:0 1px 2px #1018280d}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;border-bottom:1px solid #eaecf0;padding:10px;vertical-align:top}.pill{color:white;border-radius:999px;padding:4px 9px;font-size:12px;font-weight:700}code{white-space:pre-wrap;word-break:break-word}details{margin-top:8px}@media(max-width:700px){body{padding:10px}.card{overflow:auto}}
</style></head><body>
<h1>Unity Watchtower</h1><p class="sub">Read-only Slurm monitoring · generated __GENERATED__</p>
<section class="card"><h2>Projects</h2><table><thead><tr><th>Project</th><th>Status</th><th>Running</th><th>Pending</th><th>Blocked</th><th>Failed</th><th>Completed</th></tr></thead><tbody>__PROJECT_ROWS__</tbody></table></section>
<section class="card"><h2>Attention (__ALERT_COUNT__)</h2><table><thead><tr><th>Project</th><th>Job</th><th>State</th><th>Detail</th></tr></thead><tbody>__ALERT_ROWS__</tbody></table></section>
<section class="card"><h2>Collector warnings</h2><ul>__WARNINGS__</ul></section>
</body></html>
""".replace("__GENERATED__", html.escape(str(report["generated_at"]))).replace("__PROJECT_ROWS__", "".join(project_rows)).replace(
        "__ALERT_COUNT__", str(report["summary"]["alert_count"])
    ).replace("__ALERT_ROWS__", "".join(alert_rows) or "<tr><td colspan='4'>No unresolved alerts.</td></tr>").replace(
        "__WARNINGS__", warning_html or "<li>None</li>"
    )


def transition_events(previous: Mapping[str, Any], report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    before = {str(item.get("job_id")): item for item in previous.get("jobs", [])}
    events = []
    for job in report.get("jobs", []):
        job_id = str(job.get("job_id", ""))
        old = before.get(job_id)
        old_state = str(old.get("state")) if old else None
        new_state = str(job.get("state"))
        if old_state != new_state:
            events.append(
                {
                    "at": report["generated_at"],
                    "job_id": job_id,
                    "job_name": job.get("job_name"),
                    "project_id": job.get("project_id"),
                    "from": old_state,
                    "to": new_state,
                    "reason": job.get("reason", ""),
                }
            )
    return events


def git_environment(config: Mapping[str, Any]) -> Dict[str, str]:
    env = dict(os.environ)
    token_file = Path(expand(str(config.get("github", {}).get("token_file", "~/.config/unity-watchtower/github.token"))))
    askpass = Path(expand(str(config.get("github", {}).get("askpass", "~/.config/unity-watchtower/git-askpass.sh"))))
    env["GIT_TERMINAL_PROMPT"] = "0"
    if token_file.is_file() and askpass.is_file():
        env["UNITY_MONITOR_TOKEN_FILE"] = str(token_file)
        env["GIT_ASKPASS"] = str(askpass)
    return env


def run_git(repo: Path, args: Sequence[str], config: Mapping[str, Any], check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=git_environment(config),
        timeout=120,
        check=False,
    )
    if check and result.returncode != 0:
        detail = sanitize((result.stderr or result.stdout).strip()[-1200:], config)
        raise MonitorError(f"git {' '.join(args[:2])} failed: {detail}")
    return result


def publish_report(report: Mapping[str, Any], events: List[Dict[str, Any]], config: Mapping[str, Any], do_push: bool) -> Dict[str, Any]:
    runtime = Path(expand(str(config.get("runtime_dir", "~/.local/state/unity-watchtower"))))
    runtime.mkdir(parents=True, exist_ok=True)
    previous = load_json(runtime / "latest.json", {}) or {}
    atomic_write(runtime / "latest.json", dump_json(report))
    existing_events = load_json(runtime / "events.json", []) or []
    all_events = (existing_events + events)[-int(config.get("history_event_limit", 2000)) :]
    atomic_write(runtime / "events.json", dump_json(all_events))

    github = config.get("github", {})
    status_repo = Path(expand(str(github.get("status_repo", ""))))
    if not status_repo:
        return {"published": False, "pushed": False, "reason": "status_repo is not configured"}
    signature = report_signature(report)
    state = load_json(runtime / "publish-state.json", {}) or {}
    last_push = parse_time(str(state.get("last_push_at", "")))
    heartbeat_hours = float(github.get("heartbeat_hours", 6))
    heartbeat_due = not last_push or utcnow() - last_push >= dt.timedelta(hours=heartbeat_hours)
    changed = signature != state.get("last_signature")
    should_publish = changed or heartbeat_due or bool(github.get("publish_every_run", False))
    if not should_publish:
        return {"published": False, "pushed": False, "reason": "no semantic change and heartbeat not due"}
    if not (status_repo / ".git").is_dir():
        raise MonitorError(f"GitHub status checkout is not initialized: {status_repo}")
    reports_dir = status_repo / "reports"
    atomic_write(status_repo / "README.md", render_markdown(report))
    atomic_write(reports_dir / "status.json", dump_json(report))
    atomic_write(reports_dir / "dashboard.html", render_html(report))
    atomic_write(reports_dir / "events.json", dump_json(all_events))
    run_git(status_repo, ["add", "--", "README.md", "reports/status.json", "reports/dashboard.html", "reports/events.json"], config)
    diff = run_git(status_repo, ["diff", "--cached", "--quiet"], config, check=False)
    if diff.returncode == 0:
        state.update({"last_signature": signature, "last_publish_at": report["generated_at"]})
        atomic_write(runtime / "publish-state.json", dump_json(state))
        return {"published": True, "pushed": False, "reason": "generated files were unchanged"}
    summary = report.get("summary", {}).get("jobs", {})
    message = "Unity status: {running} running, {pending} pending, {attention} attention".format(
        running=summary.get("running", 0), pending=summary.get("pending", 0), attention=summary.get("attention", 0)
    )
    run_git(status_repo, ["commit", "-m", message], config)
    pushed = False
    if do_push and bool(github.get("push", True)):
        branch = str(github.get("branch", "main"))
        run_git(status_repo, ["push", "origin", f"HEAD:{branch}"], config)
        pushed = True
        state["last_push_at"] = report["generated_at"]
    state.update({"last_signature": signature, "last_publish_at": report["generated_at"]})
    atomic_write(runtime / "publish-state.json", dump_json(state))
    return {"published": True, "pushed": pushed, "reason": "semantic change" if changed else "heartbeat"}


def load_config(path: Path) -> Dict[str, Any]:
    config = load_json(path)
    if not isinstance(config, dict):
        raise MonitorError(f"Configuration not found or invalid: {path}")
    if int(config.get("schema_version", 0)) != 1:
        raise MonitorError("Unsupported config schema_version; expected 1")
    return config


def print_summary(report: Mapping[str, Any]) -> None:
    print(f"Unity Watchtower {report['generated_at']} — {report['summary']['alert_count']} alert(s)")
    print(f"{'PROJECT':34} {'STATUS':22} {'RUN':>4} {'PEND':>4} {'FAIL':>4} {'DONE':>4}")
    for project in report.get("projects", []):
        counts = project.get("counts", {})
        print(
            f"{str(project['name'])[:34]:34} {str(project['status'])[:22]:22} "
            f"{counts.get('running', 0):4} {counts.get('pending', 0):4} "
            f"{counts.get('attention', 0):4} {counts.get('completed', 0):4}"
        )


def doctor(config: Mapping[str, Any]) -> int:
    checks: List[Tuple[str, bool, str]] = []
    for command in ("squeue", "sacct", "git", "sbatch", "scontrol", "curl"):
        checks.append((command, shutil.which(command) is not None, shutil.which(command) or "missing"))
    github = config.get("github", {})
    repo = Path(expand(str(github.get("status_repo", ""))))
    checks.append(("status repo", (repo / ".git").is_dir(), str(repo)))
    token = Path(expand(str(github.get("token_file", "~/.config/unity-watchtower/github.token"))))
    token_ok = token.is_file() and stat.S_IMODE(token.stat().st_mode) & 0o077 == 0
    checks.append(("token permissions", token_ok, f"{token} (must be mode 600)"))
    config_roots = sum((list(item.get("roots", [])) for item in config.get("projects", [])), [])
    present = sum(1 for item in config_roots if Path(expand(str(item))).exists())
    checks.append(("project roots", present > 0, f"{present}/{len(config_roots)} currently exist"))
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL':4}  {name:20} {detail}")
    return 0 if all(ok for _, ok, _ in checks) else 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only Unity Slurm monitor with sanitized GitHub reporting")
    parser.add_argument(
        "--config",
        default=os.environ.get("UNITY_WATCHTOWER_CONFIG", "~/.config/unity-watchtower/config.json"),
        help="configuration JSON path",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="collect, render, and optionally push one snapshot")
    run_parser.add_argument("--push", action="store_true", help="push a changed report to GitHub")
    run_parser.add_argument("--json", action="store_true", help="print the full report")
    sub.add_parser("show", help="show the last locally collected snapshot")
    sub.add_parser("doctor", help="verify Unity commands, paths, repository, and token permissions")
    args = parser.parse_args(argv)
    config_path = Path(expand(args.config))
    try:
        config = load_config(config_path)
        if args.command == "doctor":
            return doctor(config)
        runtime = Path(expand(str(config.get("runtime_dir", "~/.local/state/unity-watchtower"))))
        if args.command == "show":
            report = load_json(runtime / "latest.json")
            if not report:
                raise MonitorError("No local snapshot exists; run `unity-watch run` first")
            print_summary(report)
            return 0
        previous = load_json(runtime / "latest.json", {}) or {}
        report = build_report(config)
        events = transition_events(previous, report)
        result = publish_report(report, events, config, do_push=bool(args.push))
        if args.json:
            print(dump_json(report), end="")
        else:
            print_summary(report)
            print(f"Publish: {result['reason']}; pushed={result['pushed']}")
        return 0
    except MonitorError as exc:
        print(f"unity-watch: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
