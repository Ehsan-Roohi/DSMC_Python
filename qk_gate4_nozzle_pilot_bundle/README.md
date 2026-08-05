# Gate 4F dual-transport + replicated Q-K chemistry audit

Gate 4F requires both VHS and GHS chemistry-off transport smokes. Its Q-K
screen uses four independent seeds and 6000 particles per seed at twenty
pressure/temperature points. A Gate-3B post-shock positive control runs in
the same job and must observe exchange, dissociation, and statistically
adequate recombination while conserving atoms and energy. Zero nozzle events
are reported with a 95% upper probability bound per accepted collision; they
are not automatically called proof of frozen chemistry.

Gate 4E stabilizes the first 100 samples of `ENTER2`. During startup, inlet
particles use the prescribed `PIN, FTMP, VFX, VFY`; outlet and buffer backflow
use the prescribed `POUT, FTMP` reservoir. After 100 samples the legacy
characteristic `PROPERTIES` treatment resumes. This prevents sparse one-particle
cells from generating unbounded corrected temperature and velocity.

Gate 4D fixes the collision-loop extent exposed by the symbolic Gate 4C run.
The legacy array capacity is `MNC=5000`, while the restored mesh has exactly
`100*30 + 30*40 = 4200` active cells. `COLLMR` now iterates only over those
4200 cells and fails closed if any active cell has nonpositive area.

Gate 4C restores `FNUM=5.E13`, the last setting known to initialize a
populated legacy nozzle. Gate 4B used `2.E14`, produced `NM=0` on Unity and
then raised SIGFPE. The continuous piecewise-wall bisection repair is retained.
This release also adds a fail-closed zero-particle guard, Linux-safe cleanup,
and a symbolic diagnostic build so any remaining crash reports source lines.

This bundle advances two lanes in one Unity job.

1. Gate 4F compiles the restored two-zone DSMC nozzle with bounds checking and
   runs parallel four-cycle chemistry-off VHS and GHS code-path smokes.
2. The validated Bird-QK kernel covers the complete physical residence time
   and screens the accepted geometry over 1, 2 and 5 bar stagnation pressure,
   pressure ratios 0.15--0.33, and effective temperatures 2200--4000 K.

The chemistry lane uses the restored 205 micrometre nozzle length, 15
micrometre throat, 69 micrometre exit and 100 kPa inlet stagnation pressure.
It locates an internal normal shock through quasi-one-dimensional area/pressure
matching and then follows stochastic Q-K chemistry along the remaining nozzle
residence path.

The chemistry event audit is physically tied to the nozzle
but is not represented as a fully coupled 2-D reactive DSMC field. Its purpose
is to identify the first useful reactive window, verify the reaction path with
a positive control, and quantify zero-event evidence before expensive
multispecies 2-D coupling.

On Unity the job invokes the `dsmc-gpu` environment's Python interpreter by
absolute path. It does not require `mamba init`, `mamba shell hook`, or shell
activation, which keeps the batch environment compatible with Unity's older
`mamba` command.

The restored legacy mover uses a bisection intersection against the exact
piecewise-linear lower wall and defines the wall height continuously at every
segment vertex.  The transport validator treats any out-of-domain coordinate,
invalid wall intersection, or inlet-cell mismatch as a fatal geometry error.

Key outputs:

- `gate4_transport.json`
- `chemistry_pilot/qk_nozzle_pilot.csv`
- `chemistry_pilot/qk_nozzle_pilot.json`
- `QK_GATE4_NOZZLE_PIPELINE_REPORT.txt`
- `QK_GATE4_NOZZLE_PIPELINE_REPORT.json`
