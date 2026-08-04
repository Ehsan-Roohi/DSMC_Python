# JFM high-statistics Figure 2(d), Figure 5, and Figure 6 runs

This package reruns the five lowest-signal publication cases:

1. HS, BGK, and Shakhov at `RT=0.2`, `Kn=20` (Figure 5);
2. Shakhov at `RT=0.5`, `Kn=20` (Figure 6);
3. HS at `RT=0.5`, `Kn=30` (Figure 2(d)).

Each production run uses 5,000,000 time steps. Sampling spans steps
1,000,000 through 4,999,999 at every second step (2,000,000 sampled states),
with eight time blocks for convergence diagnostics.

## Submission modes

- `low`: requests RTX 2080 Ti. A GPU preflight first tries 40M particles. If
  both HS and Shakhov fit, it runs 3 x 40M per physical case. Otherwise it
  automatically runs 6 x 20M, preserving the same total particle-time.
- `vram48`: requests any Unity GPU carrying the cumulative `vram48` feature;
  this includes suitable 48, 80, and 143 GB devices. Runs 3 x 40M.
- `vram80`: requests the cumulative `vram80` feature. Runs 3 x 40M.

All long production jobs use the non-preemptible `gpu` partition and
`--qos=long`. See the current Unity documentation:

- https://docs.unity.rc.umass.edu/documentation/tools/gpus/
- https://docs.unity.rc.umass.edu/documentation/cluster_specs/features/
- https://docs.unity.rc.umass.edu/documentation/cluster_specs/gpu_summary/

The three routes use disjoint seeds, so independently completed routes can be
combined later rather than being duplicate realizations.

## Outputs

Outputs are stored below:

`/project/pi_roohie_umass_edu/JFM_revision_2026/JFM_HIGHSTAT_FIGURES/run_output/<route>`

The dependent CPU summary job writes raw ensemble NPZ files, Tecplot POINT
DAT files, `y/L=0.25` profiles with uncertainty, JSON/CSV diagnostics, and
unfiltered diagnostic figures.
