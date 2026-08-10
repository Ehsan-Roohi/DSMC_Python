# Gate-5 reacting-nozzle physics prescreen

This stage analyzes the completed 12-case Gate-5 DSMC screen before more
expensive DSMC cases are launched.  It is intended to answer three different
questions without conflating them:

1. Was there a resolved internal shock in a numerically acceptable case?
2. Is the post-shock residence time long enough relative to homogeneous
   induction (`Da_ind = t_res/tau_ind`)?
3. Is the incoming velocity even high enough for a stationary normal
   detonation, based on a preliminary equilibrium-CJ speed screen?

The program reads
`QK_GATE5_SHOCK_IGNITION_SCREEN_REPORT.json` and each
`<case>_centerline.csv` from a completed screen directory.  It measures
window-averaged pre/post-shock states, integrates axial residence times,
estimates the 10--90% shock thickness, runs adiabatic constant-volume Cantera
reactors with `h2o2.yaml`, and writes a small proposed `(p0,T0,pb/p0)` matrix.

## Unity run

Run the submitter as a child Bash process on a Unity login node.  Do **not**
source it; its strict-shell options should not modify or terminate the
interactive shell.

```bash
bash <(curl -fsSL "https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/70e0af2e1c20c02626a3f74453008e09329e3e83/qk_gate5_physics_prescreen/submit_qk_gate5_physics_prescreen_unity.sh")
```

The default source is the completed screen:

```text
/project/pi_roohie_umass_edu/Combustion/QK_GATE5_COUPLED/runs/screen_62734194
```

Override it only when analyzing another completed screen:

```bash
RESULT_DIR=/absolute/screen/path bash <(curl -fsSL "https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/70e0af2e1c20c02626a3f74453008e09329e3e83/qk_gate5_physics_prescreen/submit_qk_gate5_physics_prescreen_unity.sh")
```

The submitter downloads analyzer and Slurm files from an immutable commit,
verifies both SHA-256 values, and submits one lightweight CPU job.  If the
existing DSMC Python environment lacks Cantera, the job creates the shared
environment `.venvs/gate5-physics-cantera-3.1` below the project directory.

## Outputs

Results are written to
`$BASE/runs/physics_prescreen_<JOB_ID>/`:

- `QK_GATE5_PHYSICS_PRESCREEN.json`: complete states, ignition histories,
  Damkohler numbers, CJ screen, assumptions, and limitations;
- `QK_GATE5_PHYSICS_CASES.csv`: flat case comparison;
- `QK_GATE5_NEXT_MATRIX.csv`: proposed next DSMC conditions;
- `QK_GATE5_PHYSICS_PRESCREEN.txt`: concise human-readable summary;
- `RUN_PROVENANCE.txt`: job, source result, Python, and analyzer hash.

## Interpretation limits

`Da_post >= 1` is only a homogeneous-kinetics feasibility result.  It does not
prove ignition in a rarefied nozzle with gradients, walls, diffusion, and a
finite-thickness shock.  The reported CJ speed is the minimum of an
equilibrium Hugoniot/Rayleigh scan and must be independently checked with a
validated CJ implementation before publication.  Every recommended DSMC case
still needs grid/particle sensitivity, repeat statistics, chemistry-off
controls, and mass/energy/species conservation audits.
