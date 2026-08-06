# JFM Kn=0.01 A100-80GB high-statistics reruns

This targeted package reruns only the two endpoint models that remained
provisional after the completed 22M-particle audit: BGK and Shakhov at
`R_T=0.5`, `Kn=0.01`. The converged HS integral endpoint is not repeated.

Each model has six independent seeds (12 GPU tasks total). Each task uses
80,000,000 particles and 5,000,000 steps. Sampling starts at step 100,000,
occurs every second step, and is retained in 49 contiguous time blocks of
approximately 100,000 simulation steps each. Quantitative fields remain raw:
no spatial filter, smoothing, or velocity projection is applied.

Resources are requested from Unity's non-preemptible `gpu` partition with the
exact `a100-80g` constraint. No obsolete `qos=long` request is used. Production
uses an array concurrency of four and the current general 14-day time limit.

The dependent CPU summary calculates the six-seed field mean/SD/SE, all
independent three-seed cross products, leave-one-seed-out jackknife
uncertainty, and blockwise convergence diagnostics. New outputs are written
below:

`/project/pi_roohie_umass_edu/JFM_revision_2026/JFM_RT05_KN001_A100_80M_S100K/run_output`
