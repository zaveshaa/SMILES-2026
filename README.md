# SMILES-2026

Solution for the SMILES-2026 interference suppression task. The solver fits a nonlinear TX model, removes the dominant spatial component with SVD, and blends the result with the provided baseline.

## Run

    pip install numpy scipy gdown
    python application_solution.py

The script downloads `challenge.mat`, runs the solver and writes `results.json`.

## Files

- `application_solution.py` — solver
- `task_and_baseline.py` — task helpers and baseline
- `SOLUTION.md` — notes
