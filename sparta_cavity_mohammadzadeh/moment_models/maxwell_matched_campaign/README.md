# Four matched-Maxwell moment-model runs

This bundle submits the four deterministic moment-model targets needed to
pair with the Maxwell--VSS SPARTA cases:

| model | Kn_Gu | production grid |
|---|---:|---:|
| R13 | 0.05 | N60 |
| R13 | 0.20 | N60 |
| R26 | 0.05 | N40 |
| R26 | 0.20 | N20 |

R13 uses the Appendix-A production coefficients and the Maxwell viscosity
law `mu/mu0=theta/theta0`.  R26 uses the Gu--Emerson `jfm2009` Maxwell
closures with the same viscosity law.  Both use full diffuse accommodation.
Archived accepted states are treated only as nonlinear initial guesses; each
job resolves the modified equations and must pass its numerical validator.

The R13 implementation has passed exact coefficient tests and a complete N4
nonlinear smoke solve at both assembly and numerical gates.  These four jobs
are production candidates.  Final scientific use still requires comparison
with the new DSMC fields and the planned grid checks.

Run on Unity through the repository bootstrap.  A collector writes a compact
ZIP and SHA-256 sidecar under
`JFM_Maxwell_Matched_Campaign_20260817` after all four tasks finish.
