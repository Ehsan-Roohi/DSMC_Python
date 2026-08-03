# Q-K Gate 4 + reactive-nozzle pilot bundle

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
