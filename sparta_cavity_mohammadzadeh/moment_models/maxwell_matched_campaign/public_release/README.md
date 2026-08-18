# Matched Maxwell-molecule cavity release

This directory is the reproducibility release for **“What does anti-Fourier
heat-flux agreement validate? DSMC sensitivity and R13/R26 diagnostics in a
rarefied cavity.”**  It contains the redistributable source, run contracts,
validators, processed states and reduced tables used for the matched
comparisons at `Kn_Gu = 0.05` and `0.20`.

## Physical contract

All matched comparisons use the same equilibrium mean-free-path definition,
`Kn_Gu = lambda_Gu/L`, a 300 K isothermal cavity, a 100 m/s lid and fully
diffuse walls. R13 and R26 use the Maxwell-molecule viscosity law
`mu/mu_0 = T/T_0`. SPARTA uses VSS parameters `omega = 1` and `alpha = 2.14`:
this is the transport-matched VSS representation of the inverse-power-law
Maxwell class, not an assertion that the stochastic angular kernel is
identical event by event to the continuum collision operator.

The comparisons distinguish three levels of evidence: lower-order fields,
the heat-flux vector, and higher moments/topology. Agreement of the first two
does not by itself validate every R13 or R26 wall and closure channel.

## Contents

| Path | Contents |
| --- | --- |
| `dsmc/` | SPARTA input generator, completed-case validator and Slurm driver |
| `r13/equations/` | exact-rational transcription of the published Appendix-A production matrix |
| `r13/tests/` | public equation tests and completed-state validator |
| `r26/source/` | source-locked independent R26 implementation and unit tests |
| `r26/hpc/` | source-locked Maxwell run drivers |
| `data/dsmc/` | final 15-field SPARTA cell averages and their input decks |
| `data/r13/`, `data/r26/` | accepted numerical states and run records |
| `data/pure_maxwell/` | common-grid fields and every table behind the six matched-model figures |
| `data/reduced/` | compact eight-realization DSMC sensitivity tables |
| `analysis/pure_maxwell/` | self-contained matched-field reducer and six-figure generator |
| `figures/` | deterministic plotting scripts and generated publication assets |
| `docs/` | provenance, exclusions, manuscript-ready statements and audit |

No SPARTA source is vendored. Build the official SPARTA revision listed in
`THIRD_PARTY_NOTICES.md`, then point `SPARTA_BIN` at its MPI executable.

## Reproduce the checks

From this directory, with Python 3.12 and the packages in `requirements.txt`:

```bash
python analysis/validate_release.py
```

The individual checks are also directly callable:

```bash
python dsmc/scripts/validate_jfm_maxwell_kngu_case.py data/dsmc/kn005 --kn-gu 0.05 --require-final
python dsmc/scripts/validate_jfm_maxwell_kngu_case.py data/dsmc/kn020 --kn-gu 0.20 --require-final
python r13/tests/test_r13_equation_contract.py
python r13/tests/validate_r13_maxwell_run.py data/r13/kn005_N60 --expected-kn-rana 0.039894228040143274 --expected-nodes 60
python r13/tests/validate_r13_maxwell_run.py data/r13/kn020_N60 --expected-kn-rana 0.1595769121605731 --expected-nodes 60
(cd r26/source/r26/code && python run_tests.py)
python r26/tests/test_maxwell_contract.py
python r26/tests/validate_r26_maxwell_run.py data/r26/kn005_N40 --expected-kn 0.05 --expected-nodes 40
python r26/tests/validate_r26_maxwell_run.py data/r26/kn020_N20 --expected-kn 0.20 --expected-nodes 20
python analysis/pure_maxwell/analyze_pure_maxwell.py
python figures/scripts/make_dsmc_sensitivity_figure.py
sha256sum -c MANIFEST.sha256
```

The released R26 states may be used as initial guesses by
`r26/hpc/submit_maxwell_pair.sh`; a continuation still has to satisfy the raw
residual and positivity gates. The R13 legacy solver and its coefficient
adapter are not redistributed because permission to republish the supplied
code was not established. The released R13 states therefore reproduce the
paper analysis, while the public equation module and tests document the
collision operator without republishing that legacy solver.

## Scientific status

- DSMC: completed, schema-validated single realizations for the two matched
  cases. The low-Knudsen sensitivity table uses eight independent
  realizations per grid/particle case.
- R13: converged numerical states for the explicitly identified
  Appendix-A-coefficient formulation. They are diagnostic results, not a
  claim of bitwise reproduction of the original authors' solver.
- R26: accepted nonlinear states from the independent Gu--Emerson
  implementation. Unit/contract checks establish algebraic and numerical
  consistency; they do not convert model--DSMC disagreement into numerical
  error or prove universal R26 accuracy.

Use an immutable tagged release (recommended tag:
`jfm-antifourier-v1.0.0`) in citations. The branch path is
`sparta_cavity_mohammadzadeh/moment_models/maxwell_matched_campaign/public_release`.

Software in this directory is BSD-3-Clause unless a file or notice says
otherwise. Numerical data are CC BY 4.0. See `LICENSE`, `DATA_LICENSE.md` and
`THIRD_PARTY_NOTICES.md`.
