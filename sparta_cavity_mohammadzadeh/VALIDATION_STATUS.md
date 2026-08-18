# Validation status

## What has been verified

- The generated input deck was parsed and executed end-to-end with the official
  SPARTA source at commit `912c9e163c38ea5c3562d039e65215f6e2a4f3f8`
  (the executable reports SPARTA 24 Sep 2025).
- The serial smoke case completed particle creation, VHS/NTC gas collisions,
  diffuse wall collisions, grid averaging, dump output, and Python
  post-processing.
- Unit tests check case generation and the time-step/collision-time ratio.

This is **syntax and workflow evidence**, not a Mohammadzadeh validation result.
The smoke grid and sample window are intentionally too small for quantitative
agreement.

## Before calling the SPARTA case validated

1. Run the 200 x 200, 32-particles-per-cell production case for at least three
   independent seeds.
2. Report grid, particles-per-cell, time-step, warm-up, and sampling-window
   sensitivity separately.
3. Compare the raw centerline lid-adjacent slip and temperature data against the
   digitized PRE reference.
4. Apply the paper's five-neighbor filter only as a second, clearly labelled
   result; retain the unfiltered values.
5. Report confidence intervals across independent seeds.
6. Cross-check against the repository's Python NTC-PreScan implementation.

The current automated gates (slip RMSE <= 0.08 and temperature RMSE <= 2 K)
are regression targets. They do not replace the sensitivity and uncertainty
study above.
