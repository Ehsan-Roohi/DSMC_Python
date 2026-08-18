# R26 component

`source/r26` is the source-locked independent Python implementation of the
Gu--Emerson 2009 nonlinear R26 equations, closure coefficients and diffuse
wall conditions used for the released states. Its source hashes are embedded
in both `run_summary.json` records and tested by `tests/test_maxwell_contract.py`.

Run the unit and molecular-model contracts with:

```bash
(cd source/r26/code && python run_tests.py)
python tests/test_maxwell_contract.py
```

The accepted states in `../data/r26` can be supplied as continuation initial
guesses. To submit both production targets:

```bash
export R26_MAXWELL_SEED_KN005=$PWD/../data/r26/kn005_N40/last_accepted_state.npz
export R26_MAXWELL_SEED_KN020=$PWD/../data/r26/kn020_N20/last_accepted_state.npz
bash hpc/submit_maxwell_pair.sh
```

Acceptance requires the raw residual, positivity, wall-pressure and global
balance gates. A successful gate is numerical evidence for that
implementation and case; it is not a universal R26-vs-DSMC validation.

