# Provenance and implementation lineage

This record separates the mathematical source of each model, the software
actually used to generate the released states, and the material that is
legally redistributed.

## R13

### Starting point

A legacy 17-field lid-driven-cavity implementation supplied by A. S. Rana was
the numerical starting point. Its mathematical reference is Rana, Torrilhon
and Struchtrup, *J. Comput. Phys.* **236** (2013),
DOI `10.1016/j.jcp.2012.11.023`. Permission to republish the supplied source
was not established. The supplied source and the two Python files derived
from it are therefore deliberately absent from `public_release`.

### Changes used for the released states

The executed numerical implementation differed substantially from the
starting code. The recorded changes were:

1. shared-face conservative finite-volume continuity;
2. an explicit compatible total-mass constraint;
3. defect-Newton/Jacobian-free Newton--Krylov nonlinear iteration with
   residual backtracking and positivity gates;
4. two-point paper-linear wall extrapolation;
5. tangential effective-pressure evaluation at the wall;
6. the Appendix-A Maxwell-molecule production matrix with
   `mu/mu_0 = T/T_0` and the explicit `rho/Kn_Rana` relaxation prefactor;
7. exact conversion `Kn_Rana = sqrt(2/pi) Kn_Gu`; and
8. source hashing, local-balance, global-mass and positivity validators.

The public file `r13/equations/r13_maxwell_production.py` is an
exact-rational transcription of the printed production matrix. Its SHA-256
in the executed runs was
`3f2e6c142f1271712f0e45bbdb4b174f6dfa9f21675aa0ddeed5bd7cf2871e42`.
It contains both the collision-consistent Maxwell prefactor and a separately
named literal Appendix-A branch; no coefficient is fit to DSMC.

For audit only, the executed but excluded files had hashes:

- derived coefficient adapter:
  `321f409386a1de770c0fed21b7ffd65af9c383f4710be37818020f456e22dd0e`;
- derived reference solver:
  `813d908f0bfb5332c61ff5ca2065af74652e365d1e8b24727bab7e7373577221`.

These hashes identify the calculation but do not grant redistribution
permission. The R13 reports correctly retain `publication_grade: false` and
`external_validation_status: not completed`; in the article the states are
used as diagnostic model results, not as a claimed bitwise reproduction of
the authors' original solver.

## R26

The R26 source in `r26/source/r26/` is an independent Python tensor
implementation of the Gu--Emerson 2009 equations, closure coefficients and
wall conditions. It was not derived from author-supplied R26 source. The
production states use the `jfm2009` closure coefficients, Maxwell viscosity
exponent one, fully diffuse walls and the Gu equilibrium-mean-free-path
Knudsen convention.

The three collision/wall-critical files are source locked by the public
contract test:

- `r26_bulk_equations.py`:
  `9abe3943ce541e6c5243a61893c1428daea30cf8fae42ab3e90c140eb7ba6a06`;
- `r26_tensor_closures.py`:
  `13037256b49de8ce0737136c56ab31fa5b1641545a79a65e77c761c25bcbbbea`;
- `r26_wall_conditions.py`:
  `b3a7bf0bc4be58f3e0c42928c87f4b01802ae88d50055b58410e485e7bbcdd49`.

Some exact source-locked files retain old internal names for tensor packing
and development-stage access gates. Those names are not software lineage and
do not identify an external source dependency. They were left unchanged so
the released code hashes match the source manifest embedded in each R26 run
record.

The R26 record describes the states as algebraically accepted predictions for
which external validation and grid convergence remain separate questions.
This is intentionally narrower than a universal validation claim.

## DSMC

The released DSMC data were computed with official SPARTA commit
`912c9e163c38ea5c3562d039e65215f6e2a4f3f8`. The physical lock is:

- `Kn_Gu = lambda_Gu/L`;
- argon at `T_w = 300 K`;
- lid speed `100 m/s`;
- fully diffuse walls;
- VSS `omega = 1`, `alpha = 2.14`;
- `160 x 160` cells and nominally 256 particles per cell;
- 40,000 warm-up steps and 200,000 sampling steps at stride 10;
- 20,000 accumulated samples per cell; and
- one released realization with seed 104729 at each matched Knudsen number.

This VSS choice matches the Maxwell-molecule viscosity exponent and transport
class. It is not an exact identity between SPARTA's stochastic scattering
kernel and either moment model's closed collision operator. The final dump
has 15 sampled fields and is schema checked. SPARTA's Sonine/grid columns are
retained as diagnostics only; the metadata explicitly prohibits treating
them as a complete direct sample of every R13/R26 higher moment.

## Knudsen-number match

No difference is hidden by redefining Knudsen number. The matched cases are:

| `Kn_Gu` | `Kn_Rana = sqrt(2/pi) Kn_Gu` | R26/SPARTA convention |
| ---: | ---: | --- |
| 0.05 | 0.039894228040143274 | 0.05 |
| 0.20 | 0.1595769121605731 | 0.20 |

