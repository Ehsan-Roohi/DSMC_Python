# Q-K Gate 4D + reactive-nozzle pilot bundle

Gate 4D bounds the legacy `COLLMR` loop to the 4200 physically active cells (`100*30 + 30*40`) instead of the array capacity `MNC=5000`. It audits every active `CC(N)` before NTC collision selection and fails closed on nonpositive cell area. This fixes the Gate 4C divide-by-zero at Fortran line 2524/2525.

Gate 4C hotfix restores `FNUM=5.E13` after Gate 4B initialized zero molecules with `2.E14`. It retains the continuous piecewise-wall bisection fix, adds a fail-closed `NM <= 0` guard, removes Windows-only cleanup commands, and compiles a symbolic debug executable for actionable Unity backtraces.

One Unity submission advances both the chemistry-off nozzle recovery and the
first chemistry-in-nozzle screen.

The job runs restored VHS and GHS transport cases in parallel while a live
Bird-QK kernel screens nine combinations of back pressure and effective
thermal forcing.  The screen uses the restored 205 micrometre nozzle geometry
and downstream buffer.

The bundle includes a repair to an inherited `ENTER2` defect: the local-row
locator previously received `(x,x)` instead of `(x,y)` for newly injected
particles.

Submit on Unity:

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/main/qk_gate4_nozzle_pilot_bundle/submit_qk_gate4_nozzle_unity.sh)
```

Then:

```bash
source /project/pi_roohie_umass_edu/Combustion/QK_GATE4_NOZZLE/LAST_GATE4_NOZZLE_JOB.env
squeue -j "$JOB_ID"
tail -f "$OUT"
```

The chemistry lane is a fast Lagrangian Q-K pilot tied to the nozzle
area/pressure path.  It is not labelled as a fully coupled two-dimensional
reactive DSMC result; selected regimes are the inputs for that expensive step.
