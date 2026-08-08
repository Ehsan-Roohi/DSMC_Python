# Gate 5 production OFF/ON pair

This bundle runs one physical nozzle condition only:

- `p0 = 5 bar`, `pb/p0 = 0.27`, `pb = 1.35 bar`;
- `T0 = 4000 K`, wall `Tw = 300 K`;
- `2 H2 + O2 + 7 Ar`;
- matched chemistry-OFF and chemistry-ON cases with seed `73001`;
- 120,000 steps per case: 60,000 burn-in plus 60,000 sampling.
- fixed pressure-reservoir injection at `PIN` and `POUT` during both phases;
- a 30-cycle transition preflight and a post-burn-in particle-runaway guard.

The one-line submitter first schedules a compile/unit/smoke preflight.  The
production pair is submitted at the same time with an `afterok` dependency and
starts automatically only when preflight passes.

The physical report requires steady through-flow, an interior compression
shock in both cases, zero OFF reactions, and ON-case OH/H2O production with
positive chemical heat release.  Flow, species, reaction, heat-release,
monitor and centerline fields are preserved even if a physical criterion
fails.
