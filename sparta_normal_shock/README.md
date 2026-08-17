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

Version 2 uses a common `x/lambda1 = [-30, 30]` domain with 1200 cells,
preserving `dx/lambda1 = 0.05`.  Each production realization uses 64 upstream
particles per cell, 80,000 warm-up steps, and a 320,000-step cumulative
average sampled every 10 steps.  This longer domain is required because the
Mach-5 thermal precursor contaminated the upstream plateau in the earlier
`[-15, 15]` campaign.

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
The upstream and downstream validation windows remain in the original
physical coordinate at `[-28, -24]` and `[24, 28]`; they are not translated
with the shock.

A realization is accepted only when all of the following pass:

- every upstream/downstream `n`, `u`, and `T` Rankine-Hugoniot error is at
  most 3%;
- the maximum mass-, momentum-, and energy-flux error relative to the
  upstream invariant is at most 0.5%;
- the maximum relative plateau slope is at most 0.5% per `lambda1`;
- the final cumulative average agrees with the preceding checkpoint in shock
  position, thickness, far-field state, and aligned profiles of every stored
  observable.  The profile limits are 2% RMS and 5% maximum relative change.

The last two averages are nested and correlated, so this is deliberately
reported as a **checkpoint-stability diagnostic**, not as an independent-
sample convergence test.

Ensembles require exactly three distinct, individually validated seeds.
Pointwise 95% uncertainty uses the Student-t multiplier for two degrees of
freedom (`t=4.3026527`), not the large-sample multiplier 1.96.  The confidence
interval is evaluated after density-midpoint alignment and is not a
simultaneous confidence band.

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
bash <(curl -fsSL https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/agent/sparta-normal-shock-v2/sparta_normal_shock/hpc/bootstrap_unity_sparta_normal_shock.sh)
```

The bootstrap prints job IDs and writes
`/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK/LAST_SPARTA_NORMAL_SHOCK_V2_JOBS.env`.

The collector always creates a diagnostic bundle after the array finishes.
If any realization fails a physics gate, that array task exits with status 9
and the collector exits with status 6 after packaging the evidence.  Only a
manifest containing `validated_member_count=9` is publication-ready; a tarball
by itself is not proof of physical validation.

## Scientific scope

The default upstream reference mean free path is `1e-7 m`, evaluated from the
same reference-diameter convention used in the cavity comparison.  This is a
reproducible solver-comparison convention; the metadata records it explicitly
so a later viscosity-based mean-free-path definition can be added without
silently changing the benchmark.
