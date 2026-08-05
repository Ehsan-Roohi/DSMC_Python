# Gate 5: coupled 2-D multispecies Q-K integration pilot

This gate compiles the repaired legacy two-zone nozzle with eight particle
species (`H2,H,O2,O,OH,H2O,HO2,Ar`) and the Gate-3B-validated Q-K kernel inside
the accepted-collision loop. Four Gate-4F-selected physical brackets are run
with chemistry on and with an otherwise identical Q-K chemistry-off control.

Each case advances 6000 DSMC steps (`dt=2e-11 s`) for `1.2e-7 s`. Local atom
and total-energy conservation are audited around every accepted Q-K event.
This remains an integration pilot: a PASS licenses longer convergence and
particle/grid studies; it is not itself a publication-production field.
