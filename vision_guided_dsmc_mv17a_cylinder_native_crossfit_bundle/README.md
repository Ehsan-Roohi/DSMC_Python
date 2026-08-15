# MV17A cylinder-native polar cross-fit

MV17A implements the second, geometry-specific cylinder path after MV16B
rejected zero-shot transfer of the cavity-frozen map.  It does **not** rerun
DSMC.  It reuses the four hash-locked MV11/MV16B native-cell datasets.

The audit fixes an important geometry issue in MV16B: the cylinder centre is
`(0.1524, 0)` m, not the coordinate origin.  Consequently MV17A reconstructs
the normal/tangential components about the correct centre and rebuilds the
near-wall mask within `0.05D` of the actual surface.

The method maps the common native DS2V mesh to a cylinder-centred polar raster,
learns a strongly regularised 2x2 residual Wiener map in fixed DCT bins, and
uses a strict ordered 2+1+1 double cross-fit.  In each of 12 folds, two seeds
train the geometry-native prior/operator, a third supplies the B=3 observation,
and the fourth supplies an independent B=10 reference.  Observation and
reference seeds are never used for fitting.

Primary endpoints are global native-cell area-weighted `q_y` NRMSE and the
correct near-wall normal heat flux `q_n`.  Controls include peer-prior-only,
fixed DCT phase scrambling, Raw B3, independent Raw B10, and the failed
cavity-frozen MV16B transfer.  Cartesian area-weighted DC is preserved exactly.

A positive result is retrospective development evidence only.  It authorises
freezing the cylinder-native method and running at least five new independent
confirmation seeds; it does not create a zero-shot cross-geometry or `p<0.05`
claim from the existing four seeds.  The original `tU/D=30` warning is retained.

The installer verifies the prior MV16B result, installs the module and locked
protocol, runs deterministic tests, then submits a CPU analysis job followed by
a packaging job.  It returns a SHA256-verified ZIP and a result pointer.

