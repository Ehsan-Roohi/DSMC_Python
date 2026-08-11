# Gate-5 closed-duct geometry preflight

The completed Mach--residence scan identified a primary-mixture baseline at
`M1=3.5`, `T0=2500 K`, `p0=5 MPa`, `A/A*=6.1098556`, and a `0.10 mm`
post-shock residence length.  Its screening values were `Da_post≈1.09`,
`Kn≈0.0224`, and `u1/DCJ≈1.065`.

This stage does **not** run reacting DSMC.  It first makes the proposed
residence geometry physically explicit and locates the nonreacting shock.

## Geometry and boundary correction

The legacy downstream buffer had an open lower receiver boundary (`IB(6)=1`).
It was therefore an open plenum, not a constant-area residence duct.  This
stage:

- changes boundary 6 to a specular wall and adds the missing `K=6` reflection
  branch in `MOVE2`;
- sets the throat height to `15 um` and the nozzle/duct height to
  `91.647834 um`, giving `A/A*=6.1098556`;
- places the throat at `x=50 um`, the nozzle-to-duct junction at `x=250 um`,
  and the outlet at `x=350 um`;
- removes the former buffer-height step by setting `NBY=NCY`;
- uses specular walls to isolate inviscid shock siting from wall heat loss;
- fixes chemistry OFF and brackets `pb/p0={0.16,0.20,0.24}` around the frozen
  normal-shock prediction `p2/p0=0.200032`;
- uses the predicted downstream reservoir temperature `2395.254 K`.

The submitter first runs a 30-cycle compile/boundary/particle smoke test.  Only
if it passes are the three 300-cycle shock-siting cases launched.

## Unity run

Run the immutable submitter as a child Bash process on a Unity login node.
Do not `source` it.

```bash
bash <(curl -fsSL "https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/b9f677c37a86e95ee72b451787933e2b046276ed/qk_gate5_geometry_preflight/submit_qk_gate5_geometry_preflight_unity.sh")
```

The job IDs and output directory are written to:

```text
/project/pi_roohie_umass_edu/Combustion/QK_GATE5_COUPLED/LAST_GATE5_GEOMETRY_PREFLIGHT.env
```

The first Unity smoke attempt reached cycle 24/30 with about 353,000
particles and no fatal, boundary, or reflection errors before its 50-minute
internal timeout.  The pinned command above keeps the particle weight and all
physics unchanged, extends the smoke timeout to 90 minutes, and raises only
the smoke Slurm wall time to two hours.

## Outputs

The summary directory contains:

- `QK_GATE5_GEOMETRY_PREFLIGHT.json`;
- `QK_GATE5_GEOMETRY_PREFLIGHT.txt`;
- `QK_GATE5_GEOMETRY_RANKING.csv`;
- one `*_centerline.csv` per back-pressure case;
- per-task logs and complete DSMC fields.

## Mandatory interpretation limit

The `100 x 30 + 40 x 30` grid preserves the legacy fixed-array envelope and is
only a geometry/topology and coarse shock-siting screen.  Its largest cell is
larger than the design mean free path; the validator therefore always reports
`publication_ready=false`.  A refined chemistry-OFF cell/particle/time-step
study is mandatory before the matched OFF/ON reacting pair.  No ignition or
detonation conclusion may be drawn from this stage.
