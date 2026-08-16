# SPARTA steady normal shock

This package produces a shock-fixed, one-dimensional argon normal shock for a
solver-to-solver comparison between SPARTA, dsmcFoam, and a deterministic
discrete-velocity method (DVM).  It is intentionally not the transient SPARTA
`shock_tube` example.

The left and right reservoirs are fixed upstream and downstream
Rankine-Hugoniot Maxwellians for a monatomic gas (`gamma = 5/3`).  Both
boundaries are open.  At the right boundary SPARTA samples the incoming tail
of the downstream Maxwellian while outgoing particles leave normally; its
mean stream velocity therefore points out of the box and SPARTA reports an
expected informational warning.  The transverse direction is periodic and
the initial discontinuity is placed at the middle of the domain.  Production
jobs cover Mach 2.5, 3, and 5 with three independent seeds per Mach number.

## Observable contract

Every run writes number density, stream velocity, translational temperature,
pressure, the three normal pressure-tensor components, and streamwise heat
flux.  Post-processing reports

- `n/n1`, `u/u1`, `T/T1`, `Tx/T1`, and `Tperp/T1`;
- normalized `Pxx` and `qx`;
- the measured shock center and 10--90% density thickness;
- upstream/downstream Rankine-Hugoniot errors;
- mass-, momentum-, and total-energy-flux conservation errors.

Each realization is translated to its measured density midpoint before the
ensemble is formed.  No spatial filter or smoothing operation is applied.

## Local verification

```bash
python3 -m unittest discover -s tests -v
bash scripts/run_smoke.sh /absolute/path/to/spa_serial
```

## Unity campaign

The public bootstrap creates or updates a dedicated checkout, builds pinned
SPARTA source, runs the real smoke case, submits a 9-member production array,
post-processes the completed members, and makes a compact return bundle.

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/agent/sparta-normal-shock-book/sparta_normal_shock/hpc/bootstrap_unity_sparta_normal_shock.sh)
```

The bootstrap prints job IDs and writes
`/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK/LAST_SPARTA_NORMAL_SHOCK_JOBS.env`.

## Scientific scope

The default upstream reference mean free path is `1e-7 m`, evaluated from the
same reference-diameter convention used in the cavity comparison.  This is a
reproducible solver-comparison convention; the metadata records it explicitly
so a later viscosity-based mean-free-path definition can be added without
silently changing the benchmark.
