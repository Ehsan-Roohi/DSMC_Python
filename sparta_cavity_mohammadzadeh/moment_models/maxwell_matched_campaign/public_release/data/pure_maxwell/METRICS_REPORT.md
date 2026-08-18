# Pure-Maxwell matched comparison

## Validation result

All six numerical fields used in the two three-method comparisons passed their
archived validators:

- SPARTA: `MAXWELL_KNGU_CASE_VALIDATION_PASS` at both
  \(Kn_{\mathrm{Gu}}=0.05\) and 0.20;
- R13: `R13_MAXWELL_VALIDATION_PASS` at both operating points;
- R26: `R26_MAXWELL_VALIDATION_PASS` on the 40-node
  \(Kn_{\mathrm{Gu}}=0.05\) state and the 20-node
  \(Kn_{\mathrm{Gu}}=0.20\) state.

The SPARTA inputs use the VSS transport representation of the Maxwell-molecule
class (\(\omega=1\), \(\alpha=2.14\)); this matches the viscosity exponent and
transport class of the R13/R26 Maxwell-molecule calculations but is not an
assertion that the VSS angular kernel is identical to the exact IPL kernel.
All calculations use \(T_w=300\,\mathrm{K}\), \(U_w=100\,\mathrm{m\,s^{-1}}\),
fully diffuse walls, and the Gu equilibrium mean-free-path convention.

## Model contract

| method | collision/transport contract | wall contract | native grids |
|---|---|---|---|
| DSMC | SPARTA VSS Maxwell transport class, \(\omega=1\), \(\alpha=2.14\) | fully diffuse, full thermal accommodation | \(160^2\), 256 particles/cell and 20,000 accumulated samples/cell at both Kn |
| R13 | Maxwell-molecule Appendix-A coefficients, \(\mu/\mu_0=T/T_0\), with \(Kn_{\rm Rana}=\sqrt{2/\pi}\,Kn_{\rm Gu}\) | accommodation 1, paper-linear reconstruction, paper-tangential effective pressure | \(60^2\) at both Kn |
| R26 | nonlinear Gu--Emerson JFM-2009 closure, Maxwell molecules, \(\mu/\mu_0=T/T_0\) | accommodation 1, smooth-wall R26 conditions | \(40^2\) at 0.05; \(20^2\) at 0.20 |

## Field metrics relative to DSMC

The heat-flux values below use the declared seven-cell spatial filter.  The
velocity metrics use the unsmoothed fields.  Errors are relative RMS norms and
\(C\) is the vector cosine.

| \(Kn_{\rm Gu}\) | model | \(E_u\) | \(C_u\) | \(E_q\), whole field | \(C_q\), whole field | \(E_{\rho'}\) | \(E_{T'}\) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.05 | R13 | 0.0498 | 0.9988 | 0.856 | 0.792 | 0.0801 | 0.261 |
| 0.05 | R26 | 0.0194 | 0.9998 | 0.215 | 0.9766 | 0.0310 | 0.0887 |
| 0.20 | R13 | 0.1629 | 0.9876 | 0.646 | 0.814 | 0.2028 | 0.803 |
| 0.20 | R26 | 0.1131 | 0.9940 | 0.366 | 0.9345 | 0.0987 | 0.265 |

## Anti-Fourier comparison on the DSMC-defined active mask

The primary diagnostic uses seven-cell smoothing, 5% heat-flux and temperature-
gradient activity thresholds, and excludes the two top-corner squares of side
\(0.05L\).

| \(Kn_{\rm Gu}\) | model | \(E_q\) | \(C_q\) | weighted angle | Jaccard | Dice |
|---:|---|---:|---:|---:|---:|---:|
| 0.05 | R13 | 0.847 | 0.792 | 22.5 deg | 0.247 | 0.396 |
| 0.05 | R26 | 0.212 | 0.977 | 6.09 deg | 0.344 | 0.511 |
| 0.20 | R13 | 0.620 | 0.823 | 27.4 deg | 0.100 | 0.181 |
| 0.20 | R26 | 0.359 | 0.936 | 11.7 deg | 0.569 | 0.726 |

The quantitative result supports the manuscript's central hierarchy.  Both
moment systems reproduce the lower-order circulation more closely than the
heat-flux structure.  R26 is consistently closer than R13 in heat-flux
magnitude/direction and binary topology, but its Jaccard indices of 0.344 and
0.569 remain far below complete spatial reproduction.  Consequently, R26
captures the dominant cold-to-hot signature without fully identifying its
DSMC spatial support; R13 is less accurate still.  This preserves the central
claim when stated as incomplete spatial/closure reproduction, not as failure
to generate any anti-Fourier heat flux.

Across smoothing windows 3--11 and activity cuts 3--10%, R26 retains the
vector ranking: its weighted angle spans 5.81--7.06 deg at
\(Kn_{\rm Gu}=0.05\) and 11.4--13.1 deg at 0.20, compared with 21.7--25.4 deg
and 26.7--35.2 deg for R13.  Binary topology is more processing-sensitive;
the full ranges are retained in `processing_sensitivity.csv`.

## Scientific scope

- Each DSMC operating point is a single statistically averaged realization;
  the comparison does not provide between-seed confidence intervals.
- The R13 reports mark the roots as accepted private physical solutions, but
  also record `external_validation_status = not completed` and
  `publication_grade = false`; these flags must not be silently upgraded.
- The R26 transition result is an accepted single-grid state.  No
  \(Kn_{\rm Gu}=0.20\) grid extrapolation is available.
- SPARTA `sonine/grid` fourth moments in these packages are diagnostic only,
  and no independent rank-three \(m_{ijk}\) is available.  The pure-Maxwell
  comparison therefore supports lower-order and heat-flux/anti-Fourier claims,
  not independent full R26-state certification.

Machine-readable values and hashes are in `field_metrics.csv`,
`anti_fourier_metrics.csv`, `processing_sensitivity.csv`, `audit_metrics.json`
and `SHA256SUMS.txt`.
