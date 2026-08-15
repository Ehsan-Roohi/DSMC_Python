# MV17B — fresh cylinder-native confirmation

MV17B is the prospective confirmation stage authorised by MV17A.  It freezes
one final cylinder-native polar/DCT residual estimator using only the four
historical development trajectories, then runs six wholly new and disjoint
observation/reference pairs (twelve Bird/DS2V trajectories).

The observation member supplies locked Raw-B3 and Raw-B10 fields.  The paired
reference member supplies only an independent Raw-B10 target and never enters
prediction.  Co-primary endpoints are global native-cell area-weighted `q_y`
and cylinder-centred near-wall `q_n`.  Six all-improved pairs support Holm-
adjusted one-sided significance below 0.05 for both endpoints.

The acquisition deliberately reproduces the locked NOUT 100–116 late window.
It does not claim `tU/D=30` stationarity, universal zero-shot geometry transfer,
or direct wall-collision heat-flux validation.

Unity installation requires sparse-checking out the MV11, MV16A, MV17A, and
MV17B bundles.  The installer copies prerequisite files without invoking their
job-submission scripts, verifies the protocol and tests, and submits the
prepare → 12-task array → analysis → package dependency chain.

