# Unity Watchtower

Unity Watchtower is a read-only Slurm monitor for Ehsan Roohi's research
campaigns on the Unity cluster. It collects active and recent jobs, resolves
project membership, follows failed dependencies, checks selected validation
artifacts, sanitizes the report, and pushes meaningful changes to the private
`Ehsan-Roohi/UnityMonitor` GitHub repository.

It sends **no email** and contains no `scancel`, requeue, retry, or scientific
job-submission logic. The only Slurm job it submits is its own low-priority,
five-minute collector job.

## What the dashboard distinguishes

- `Running` and `Pending`, including Slurm's reason;
- `Blocked` by a failed or invalid dependency;
- `Attention` for `FAILED`, `OUT_OF_MEMORY`, `TIMEOUT`, `NODE_FAIL`,
  `CANCELLED`, and related states;
- `Completed—unverified` when Slurm finished but no configured acceptance
  artifact proves the scientific result;
- `Validated` when required project-specific artifacts exist.

A historical failure is no longer considered unresolved when a newer job with
the same canonical job name succeeds or is currently running.

## Default project registry

The supplied configuration recognizes:

- SPARTA / Mohammadzadeh cavity;
- JFM R13/R26 cavity campaign;
- Fokker–Planck PINN;
- KineticGaussian / BGK shock;
- Q–K combustion gates;
- Lekzian bulk-to-wall gate;
- Lennard–Jones Gate 2 and strong-shock suites;
- Ab-initio Mach-10 shock;
- an `Unclassified` bucket for every other recent Unity job.

## One-time installation on Unity

Run from a Unity login terminal:

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/agent/unity-watchtower/unity_watchtower/bootstrap_unity.sh)
```

The installer securely prompts for a GitHub token with hidden input. It never
puts the token in a command-line argument, Git remote, report, or repository.
The local token file is:

```text
~/.config/unity-watchtower/github.token
```

and is created with mode `600`. For least privilege, use a fine-grained token
with repository `Contents: Read and write` access only to `UnityMonitor`. If
the token cannot create a repository, create an empty **private** repository
named `Ehsan-Roohi/UnityMonitor` and rerun the installer. The installer refuses
to publish to a public repository.

The installer then:

1. checks out this source under
   `/project/pi_roohie_umass_edu/UNITY_MONITOR/source`;
2. installs the editable project registry at
   `~/.config/unity-watchtower/config.json`;
3. creates or connects the private status repository at
   `/project/pi_roohie_umass_edu/UNITY_MONITOR/status_repo`;
4. performs and pushes the first collection;
5. starts a 15-minute low-priority Slurm collector chain.

## Daily commands

```bash
# Compact local dashboard
unity-watch show

# Fresh one-shot collection and GitHub push
unity-watch run --push

# Full diagnostic
unity-watch doctor

# Scheduler state and latest collector job
/project/pi_roohie_umass_edu/UNITY_MONITOR/source/unity_watchtower/scripts/watcher_status.sh

# Stop all future automatic collections
/project/pi_roohie_umass_edu/UNITY_MONITOR/source/unity_watchtower/scripts/stop_watcher.sh

# Restart automatic collection
/project/pi_roohie_umass_edu/UNITY_MONITOR/source/unity_watchtower/scripts/start_watcher.sh
```

To change the collection interval for a new chain:

```bash
export UNITY_WATCHTOWER_INTERVAL_MINUTES=30
/project/pi_roohie_umass_edu/UNITY_MONITOR/source/unity_watchtower/scripts/stop_watcher.sh
/project/pi_roohie_umass_edu/UNITY_MONITOR/source/unity_watchtower/scripts/start_watcher.sh
```

## GitHub output

The private status repository contains:

```text
README.md                  GitHub-native project summary
reports/status.json        structured source for Codex/ChatGPT
reports/dashboard.html     standalone HTML dashboard
reports/events.json        recent state transitions
```

The collector commits only when a job state, reason, alert, project status, or
validation artifact changes. A six-hour heartbeat refreshes stale reports.
Elapsed time alone does not create a commit.

## Security and privacy

Before publication, the monitor:

- replaces the PI project root with `$PROJECT`;
- replaces the work root with `$WORK`;
- replaces the home directory with `$HOME`;
- redacts GitHub token formats, authorization headers, and password-like text;
- excludes job commands, environment variables, token files, and raw state
  files;
- reads only the tail of failed-job logs and publishes only lines matching
  error signatures.

The status repository is required to be private. Scientific data, `.npz`
files, raw logs, and result bundles are never uploaded.

## Add or edit a project

Edit:

```text
~/.config/unity-watchtower/config.json
```

A project can match jobs by working-directory root and/or job-name pattern:

```json
{
  "id": "new-campaign",
  "name": "New campaign",
  "roots": ["/project/pi_roohie_umass_edu/NEW_CAMPAIGN"],
  "job_name_patterns": ["new-*"],
  "validation_mode": "all",
  "artifact_checks": [
    {
      "label": "Acceptance result",
      "required": true,
      "paths": ["{root}/results/*/final_status.json"]
    }
  ]
}
```

Prefer narrow artifact patterns. Avoid a recursive `**` scan at a very large
data root such as the full Fokker–Planck dataset.

## Validation

The repository has standard-library unit and end-to-end tests:

```bash
python3 -m unittest discover -s unity_watchtower/tests -v
bash -n unity_watchtower/bootstrap_unity.sh unity_watchtower/scripts/*.sh unity_watchtower/hpc/*.slurm
```

No third-party Python package is required.
