# Task: verify the SPARTA cavity workflow safely

Work only inside this repository. Read `AGENTS.md`, `README.md`, and
`VALIDATION_STATUS.md` before acting.

1. Run `python3 -m unittest discover -s tests -v`.
2. Locate `third_party/sparta/src/spa_serial`. If it is absent, stop and report
   the exact documented install command. Do not use `sudo` and do not install
   system packages.
3. Run `bash scripts/run_case.sh smoke serial`.
4. Inspect `runs/smoke_kn01/log.cavity`, `case_metadata.json`, and
   `validation_metrics.json`.
5. Explain that a smoke run verifies syntax and data flow, not agreement with
   the reference benchmark. Do not label the case validated.
6. If all technical checks pass, give the student the next command
   `bash scripts/run_case.sh student serial`. Do not start the production case
   unless the user explicitly asks for it.

Finish with a concise PASS/FAIL report, the evidence inspected, and any exact
next command. Do not change benchmark constants or reference data.
