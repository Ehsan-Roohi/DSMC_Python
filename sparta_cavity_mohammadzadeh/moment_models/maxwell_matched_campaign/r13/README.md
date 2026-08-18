# R13 Maxwell production audit and candidate patch

## Status

This directory is a **candidate, not a run-ready or publication-ready R13
solver**.  No nonlinear cavity state has been recomputed with the candidate,
and no external numerical benchmark has yet been passed.  Nothing under
`work/article_final_v2/manuscript` was edited.

## Audited inputs

- Archived coefficients:
  `work/article_final_v2/data/kn020_models/r13_N60/rana_original_coefficients.py`
  (`SHA-256 08caba3895db19c72cc69fe8c9be4b41fb5676b47e140ce123798df548a0b6fd`)
- Archived solver:
  `work/article_final_v2/data/kn020_models/r13_N60/rana_original_reference_solver.py`
  (`SHA-256 9b10862a3582ae59e91303292865ef142eabb022c103a4b18abbe0956f7f2e24`)
- Primary source: A. S. Rana, M. Torrilhon & H. Struchtrup, *Journal of
  Computational Physics* **236** (2013), 169-186,
  doi:10.1016/j.jcp.2012.11.023.  The audited author PDF is
  `https://www.engr.uvic.ca/~struchtr/2013_JCP_Lidcavity.pdf`.

The paper states “Maxwell molecules” in section 2.1; equation (29) defines
`Kn_Rana = mu0/(rho0 sqrt(theta0) L)`; and journal page 184 prints the reduced
production matrix.

## Proven findings

1. The archived prefactor is
   `rho*sqrt(theta)/Kn_Rana`.  Algebraically this is the factor obtained from
   `rho*theta/(Kn_Rana*mu*)` with `mu*=sqrt(theta)`, not with the Maxwell law
   `mu*=theta`.
2. With `rb=ra=ma=1`, the archived unscaled matrix differs from Appendix A in
   20 nonzero entries at a generic non-equilibrium state:
   two in `R_xy`, four in `m_xxx`, five in `m_xxy`, five in `m_xyy`, and four
   in `m_yyy`.  The two `R_xy` heat-flux couplings printed in Appendix A are
   zero in the archive.
3. There is a typesetting/order inconsistency inside the paper.  The state
   list below Eq. (11) prints `m_xxx,m_xyy,m_xxy,m_yyy`.  However, the
   Appendix-A x-flux matrix has `A[7,13]=1`, which identifies slot 13 as
   `m_xxy`; the x-wall matrix applies the `m_ssn=m_xyy` condition in slot 14.
   The archived flux and wall matrices use this latter conventional order.
   Therefore a production-only 13/14 permutation would be wrong.  The adapter
   proves these invariants before installing the candidate.
4. The common Knudsen-number contract is
   `Kn_Rana = sqrt(2/pi) Kn_Gu`:
   - `Kn_Gu=0.05 -> Kn_Rana=0.039894228040143274`
   - `Kn_Gu=0.20 -> Kn_Rana=0.1595769121605731`

The machine-readable details, including every differing matrix entry, are in
`audit_report.json`.

## What the candidate implements

`r13_maxwell_production.py` contains an exact-rational transcription of the
reduced Appendix-A production coefficients in the executable flux/wall order.
Its main candidate uses the collision-consistent Maxwell prefactor
`rho/Kn_Rana`, derived from Eqs. (3)-(4), Eq. (29), and `mu*=theta`.

The paper's literal Eq. (11)+Appendix-A display instead reads `P(U)/Kn` with
no density prefactor.  Because these two readings differ away from `rho=1`,
the module exposes the literal branch separately and does **not** call either
one an externally validated reproduction of the authors' results.

`r13_maxwell_adapter.py` is fail-closed: it checks both archived hashes,
checks `STATE_ORDER`, proves the flux and wall slot semantics, and only then
installs the candidate.  There is intentionally no automatic drop-in alias and
no one-line production run command.

## Verification completed

The local test suite contains 13 passing tests:

- exact `fractions.Fraction` checks of the `R_xy` and all four third-moment
  rows;
- numeric checks of the Maxwell and archived prefactors;
- explicit detection of the archived coefficient differences;
- exact `Kn_Gu`/`Kn_Rana` conversion at 0.05 and 0.20;
- source-hash, flux-order and wall-order integration gates.

Run only the audit tests (not a cavity solve) with:

```bash
python3 -m unittest -v work/r13_maxwell_fix/test_r13_maxwell_production.py
python3 work/r13_maxwell_fix/audit_r13_production.py
```

## Required gates before any R13 result is used

1. Apply `CANDIDATE_INTEGRATION.patch` in a separate solver branch and place
   `r13_maxwell_production.py` beside the archived coefficient module.
2. Add the companion module hash to every run report, as shown in the patch.
3. Run assembly-level finite-difference/Jacobian tests on a small grid and
   verify that the unbordered residual decreases under Newton updates.
4. Reproduce at least one published independent cavity scalar (`D` or `G`)
   at a tabulated `Kn_Rana`, with the same wall accommodation and lid speed.
   This is the external gate that is currently missing.
5. Only after that gate, solve `Kn_Gu=0.05` and `0.20` using the exact converted
   `Kn_Rana` values, with independent initialisation and grid convergence.
6. Regenerate centreline and anti-Fourier figures only from those accepted
   states and collision-model-matched DSMC data.

Until steps 3-5 pass, the existing R13 curves must not be represented as a
paper-exact Maxwell comparison.

