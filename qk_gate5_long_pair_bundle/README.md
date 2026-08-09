# Gate 5 long OFF/ON production pair

This release keeps the validated production conditions unchanged and doubles only the integration and sampling duration.

- one pressure ratio: `pb/p0 = 0.27`
- `p0 = 500 kPa`, `pb = 135 kPa`, `T0 = 4000 K`
- chemistry OFF and chemistry ON with identical mesh, boundary conditions, and seed
- 240,000 time steps per case
- 120,000-step burn-in followed by 120,000 sampled steps
- compile/smoke preflight before the dependent full run
- immutable payload SHA-256 verification

The standalone comparison script writes matched-scale OFF/ON contours, signed `ON-OFF` contours, centerline overlays, a per-cell delta CSV, and a JSON summary.
