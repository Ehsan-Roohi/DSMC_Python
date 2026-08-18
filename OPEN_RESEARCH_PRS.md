# Open research pull requests

Inventory date: 2026-08-18.  These 39 PRs are preserved research records.  A
listed PR is not automatically accepted or merge-ready; its own body, result
files, and gates remain authoritative.  R13/R26/JFM material is governed by
`JFM_R13_R26_PROTECTION.md`.

## Current synthesis and reference campaigns

- [#42](https://github.com/Ehsan-Roohi/DSMC_Python/pull/42) — gated Phase-0 workflow for JCP redesign
- [#41](https://github.com/Ehsan-Roohi/DSMC_Python/pull/41) — Maxwell-transport-class DSMC, R13 and R26 evidence
- [#40](https://github.com/Ehsan-Roohi/DSMC_Python/pull/40) — hardened SPARTA normal-shock v2 validation
- [#39](https://github.com/Ehsan-Roohi/DSMC_Python/pull/39) — two-run high-statistics SPARTA DSMC Kn=0.20 JFM ensemble
- [#35](https://github.com/Ehsan-Roohi/DSMC_Python/pull/35) — SPARTA thermal-temperature correction and Kn=0.1 HQ rerun
- [#34](https://github.com/Ehsan-Roohi/DSMC_Python/pull/34) — accepted R13-seeded R26 Kn=0.2 routes
- [#27](https://github.com/Ehsan-Roohi/DSMC_Python/pull/27) — MV11 DS2V cylinder campaign
- [#26](https://github.com/Ehsan-Roohi/DSMC_Python/pull/26) — fail-closed R26 Kn=0.2 recovery ensemble
- [#23](https://github.com/Ehsan-Roohi/DSMC_Python/pull/23) — standalone SPARTA lid-driven-cavity tutorial
- [#21](https://github.com/Ehsan-Roohi/DSMC_Python/pull/21) — recovered R13/R26 reviewer solvers
- [#16](https://github.com/Ehsan-Roohi/DSMC_Python/pull/16) — restartable HS Kn=0.05 Figure 8(b) workflow
- [#14](https://github.com/Ehsan-Roohi/DSMC_Python/pull/14) — exact restart continuation for JFM 80M runs
- [#13](https://github.com/Ehsan-Roohi/DSMC_Python/pull/13) — checkpoint-fast 80M JFM replacement workflow
- [#12](https://github.com/Ehsan-Roohi/DSMC_Python/pull/12) — fast single-seed L40S JFM figure completion
- [#10](https://github.com/Ehsan-Roohi/DSMC_Python/pull/10) — six-seed A100 Kn=0.01 endpoint reruns
- [#7](https://github.com/Ehsan-Roohi/DSMC_Python/pull/7) — validated DSMC cavity solvers and SPARTA teaching case
- [#6](https://github.com/Ehsan-Roohi/DSMC_Python/pull/6) — JFM endpoint and 80M high-statistics reruns
- [#5](https://github.com/Ehsan-Roohi/DSMC_Python/pull/5) — hosted Bird-QK Gate-2 validation
- [#1](https://github.com/Ehsan-Roohi/DSMC_Python/pull/1) — audited DSMC/DVM pipeline, negative allocation results, and Shakhov convergence

## Mohammadzadeh/vision and recovery sequence

- [#38](https://github.com/Ehsan-Roohi/DSMC_Python/pull/38) — MV17B-A3 fixed-endpoint recovery
- [#37](https://github.com/Ehsan-Roohi/DSMC_Python/pull/37) — MV17B-A2 locked-window recovery
- [#36](https://github.com/Ehsan-Roohi/DSMC_Python/pull/36) — MV17B-A1 incomplete same-seed trajectory recovery
- [#33](https://github.com/Ehsan-Roohi/DSMC_Python/pull/33) — frozen MV15C q_y after reference-QC warning
- [#32](https://github.com/Ehsan-Roohi/DSMC_Python/pull/32) — prospectively locked MV15C B3 confirmation
- [#31](https://github.com/Ehsan-Roohi/DSMC_Python/pull/31) — MV15B data-consistent q_y budget ladder
- [#30](https://github.com/Ehsan-Roohi/DSMC_Python/pull/30) — MV15A q_y information audit
- [#29](https://github.com/Ehsan-Roohi/DSMC_Python/pull/29) — MV14 kinetic-conservation cavity reconstruction
- [#28](https://github.com/Ehsan-Roohi/DSMC_Python/pull/28) — MV12 safety-aware gated q_y ensemble
- [#25](https://github.com/Ehsan-Roohi/DSMC_Python/pull/25) — MV10 multiscale q_y repair pilot
- [#24](https://github.com/Ehsan-Roohi/DSMC_Python/pull/24) — MV9 heat-flux Noise2Noise pilot
- [#22](https://github.com/Ehsan-Roohi/DSMC_Python/pull/22) — locked MV5 repairs and MV7 budget matrix
- [#20](https://github.com/Ehsan-Roohi/DSMC_Python/pull/20) — MV6 architecture screen
- [#19](https://github.com/Ehsan-Roohi/DSMC_Python/pull/19) — MV5 preregistered selector benchmark
- [#18](https://github.com/Ehsan-Roohi/DSMC_Python/pull/18) — MV4 bounded stability repair
- [#17](https://github.com/Ehsan-Roohi/DSMC_Python/pull/17) — Gate-5 shock-triggered ignition screen
- [#15](https://github.com/Ehsan-Roohi/DSMC_Python/pull/15) — locked cross-condition MV3 benchmark
- [#9](https://github.com/Ehsan-Roohi/DSMC_Python/pull/9) — leakage-safe validated-field vision stage
- [#8](https://github.com/Ehsan-Roohi/DSMC_Python/pull/8) — locked R100 heat-flux precision stage

## Operations

- [#11](https://github.com/Ehsan-Roohi/DSMC_Python/pull/11) — Unity Watchtower for automated Slurm monitoring

## Triage rule

Close a PR only when a documented successor preserves its evidence and branch.
Merge only after its declared execution/scientific gate passes; otherwise keep
it explicitly draft, held, failed, or diagnostic.
