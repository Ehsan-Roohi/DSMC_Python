# Matched Maxwell-VSS DSMC campaign for the JFM revision

This campaign generates one statistically long SPARTA realization at each of
`Kn_Gu = 0.05` and `Kn_Gu = 0.20`.  Both cases use the same molecular
collision class as the Maxwell-molecule moment formulation: the standard VSS
transport approximation to the inverse-power-law Maxwell interaction,
`omega = 1` and `alpha = 2.140`.

## Collision and Knudsen-number contract

The viscosity-equivalent diameter and the diameter read by SPARTA are not
interchanged:

```text
d_eq  = 4.6326659042208623e-10 m
d_VSS = d_eq sqrt[(1+alpha)(2+alpha)/(6 alpha)]
      = 4.6613687882519160e-10 m
```

`d_eq` enters the Gu viscosity-based mean free path,

```text
lambda_Gu = 15 (T/Tref)^(omega-1/2)
            / [2 sqrt(2) (5-2 omega) (7-2 omega) d_eq^2 n],
Kn_Gu = lambda_Gu/L.
```

The generator reconstructs `Kn_Gu` from the dimensional input and fails before
writing a case if the contract does not close to floating-point tolerance.
For `L = 1e-6 m`, `N = 160`, and 256 initial simulator particles per cell:

| `Kn_Gu` | number density (`m^-3`) | `fnum` | `dx/lambda_Gu` |
|---:|---:|---:|---:|
| 0.05 | 3.4538410244383913e25 | 5.270143164731432e6 | 0.125 |
| 0.20 | 8.634602561095979e24 | 1.317535791182858e6 | 0.03125 |

The VSS model is a transport-matched approximation to the IPL angular kernel;
it should be described that way rather than as an exact event-by-event IPL
collision implementation.

## Production settings

Each case uses:

- one deterministic seed, `104729` by default;
- a `160 x 160` grid and 256 initial particles per cell;
- 40,000 warm-up steps;
- 200,000 sampling steps, sampled every 10 steps;
- 20,000 accumulated samples per cell;
- diffuse walls with full thermal accommodation at 300 K;
- a 100 m/s isothermal lid;
- pinned upstream SPARTA commit
  `912c9e163c38ea5c3562d039e65215f6e2a4f3f8`.

The two Knudsen numbers are the two tasks of a two-element job array.  Each
task is one realization; there is no hidden seed ensemble.

## Fifteen-field output schema

`grid.final.00200000` has the following fixed column order after `id xc yc`:

| Index | Name | SPARTA source | Role |
|---:|---|---|---|
| 1--4 | `nrho,u,v,w` | `grid` | density and velocity |
| 5 | `T` | `thermal/grid` | COM-subtracted translational temperature |
| 6--7 | `qx,qy` | `eflux/grid` | COM-subtracted heat-flux density |
| 8--11 | `Pxx,Pxy,Pyy,Pzz` | `pflux/grid` | pooled-COM momentum-flux density |
| 12--15 | `B1xx,B1xy,B1yy,B1zz` | `sonine/grid b ... 1` | diagnostic raw fourth moments |

`Pxy` is a direct off-diagonal stress measurement.  The four `B1` columns are
the standard SPARTA raw moments `B1ij = <C_i C_j C^2>`.  Upstream SPARTA
subtracts the cell COM separately at each sampled timestep for `sonine/grid`,
not once from the pooled particle population.  These columns are therefore
explicitly marked `diagnostic_only` in `case_metadata.json`.  They are not, by
themselves, a quantitatively unbiased measurement of the R26 `R_ij` or
`Delta`.  A pooled-COM custom sampler or an independently validated finite-N
bias correction is required before making that higher-moment claim.  Upstream
SPARTA also has no direct complete rank-three `m_ijk` grid sampler.  The
temperature, heat flux, and direct momentum-flux columns do not share this
specific `sonine/grid` limitation.

## Unity one-line submission

After the branch is published, this avoids the raw-GitHub `curl` rate limit by
updating a persistent clone and then running the bootstrap:

```bash
bash -lc 'D=/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK/DSMC_Python_sparta_maxwell_kngu005_020_jfm; B=agent/maxwell-matched-antifourier; if [[ -d "$D/.git" ]]; then git -C "$D" fetch origin "+refs/heads/$B:refs/remotes/origin/$B" && git -C "$D" switch "$B" && git -C "$D" pull --ff-only origin "$B"; else git clone --branch "$B" --single-branch https://github.com/Ehsan-Roohi/DSMC_Python.git "$D"; fi && DSMC_MAXWELL_BRANCH="$B" bash "$D/sparta_cavity_mohammadzadeh/hpc/bootstrap_unity_sparta_maxwell_kngu005_020_jfm.sh"'
```

The bootstrap submits a pinned build/test job, the two one-realization case
tasks, and a collector.  It prints a state file and the final archive path:

```text
SPARTA_MAXWELL_KNGU005_020_JFM_<job>_SEED104729_TO_ANALYZE.zip
```

The ZIP contains final grids, logs, dimensional metadata, exact source decks,
tests, scheduler logs, a per-file SHA-256 manifest, and no large checkpoints or
restart files.  A sibling `.sha256.txt` verifies the archive itself.

## Local QA

```bash
python3 -m unittest discover -s tests -v
python3 scripts/generate_jfm_maxwell_kngu020_case.py \
  --seed 104729 --kn-gu 0.20 --output /tmp/maxwell-kngu020-contract
python3 scripts/validate_jfm_maxwell_kngu_case.py \
  /tmp/maxwell-kngu020-contract --kn-gu 0.20
```

The build job additionally runs both generated decks through the pinned SPARTA
parser and requires the exact 15-field dump header.
