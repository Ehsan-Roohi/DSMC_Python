# Reference data provenance

`mohammadzadeh_2012_lid_profiles.csv` is a conservative manual digitization
of the **macroscopic DSMC** open-circle series in Figs. 4 and 5 of:

> A. Mohammadzadeh, E. Roohi, H. Niazmand, S. Stefanov, and R. S. Myong,
> "Thermal and second-law analysis of a micro- or nanocavity using direct-
> simulation Monte Carlo," *Physical Review E* 85, 056310 (2012),
> DOI: 10.1103/PhysRevE.85.056310.

Only the interior interval `0.1 <= x/L <= 0.9` is tabulated.  This avoids
claiming pointwise accuracy in the corner singularity and avoids marker/axis
overlap in the published raster.  The uncertainty columns describe the
plot-resolution digitization uncertainty, not the DSMC sampling uncertainty.

The publication reports monatomic argon, `m=6.63e-26 kg`, `d=4.17e-10 m`,
isothermal diffuse walls at 300 K, a 100 m/s lid, VHS collisions, 32 initial
particles per cell, and NTC pair selection.  It reports 200x200 as the selected
grid after a 100/200/400 grid study.  Reproducing the publication therefore
requires the production configuration and long ensemble averaging; the quick
configuration is only a software smoke test.
