# CoReMol Stage-1 Gate Design

## Goal

Build and run the first CoReMol validity gate inside `CORMOL/`: a two-dataset, three-seed molecular property experiment that requires both task metrics and interpretability metrics to improve.

## Scope

The gate uses only:

- `BBBP` for binary classification, measured by ROC-AUC.
- `ESOL` for regression, measured by RMSE and MAE.
- `AttentiveFP` as the base molecular backbone.
- `Base AttentiveFP` and `Full CoReMol` as the first comparison pair.
- Scaffold split with train/valid/test = 8/1/1.
- Seeds `0`, `1`, and `2`.

All code, downloaded data, reference repositories, configs, logs, and reports are kept under `CORMOL/`.

## Passing Criteria

The stage-1 gate passes only when both model behavior and mechanism behavior improve:

- BBBP ROC-AUC improves for Full CoReMol over Base AttentiveFP.
- ESOL RMSE decreases for Full CoReMol over Base AttentiveFP.
- TCM@K improves on both BBBP and ESOL.
- At least two of three seeds show same-direction improvement for task metric and TCM@K.
- If the task metric improves but TCM@K does not, the method is treated as mechanically unvalidated.
- If TCM@K improves but the task metric does not, the mechanism is treated as partially valid but not yet useful for prediction.

## Main Mechanism

CoReMol keeps the molecular covalent graph fixed and adds an internal signed residual communication calibration step:

1. Encode atoms with AttentiveFP on the original bond graph.
2. Build candidate atom pairs with topological distance `1 <= d(i,j) <= d_max`.
3. Compute base communication support `C(i,j)` from finite-hop propagation over the covalent graph.
4. Predict task-conditioned demand `D(i,j | c_mol)` from atom states, molecular context, and pair structure features.
5. Compute signed residual score `S(i,j) = D(i,j | c_mol) - C(i,j)`.
6. Convert `S` to bounded logit calibration `delta`.
7. Update atom states with `sum_j (alpha_cal - alpha_base) W h_j`.
8. Predict from calibrated atom states with the same readout head.

## Failure Analysis Policy

If the gate fails, changes must preserve the main mechanism. Allowed adjustments are limited to:

- Candidate pair distance and filtering.
- Base support normalization.
- Demand feature scaling and context usage.
- Signed residual temperature and magnitude.
- Residual update gate initialization.
- TCM probe stability and top-K sensitivity settings.

The first failure response must diagnose why the current mechanism failed before making a change. Additional unrelated modules are out of scope for this gate.

## Outputs

The run writes:

- Per-seed raw metrics to `CORMOL/results/stage1_gate/raw_metrics.csv`.
- Aggregated metrics to `CORMOL/results/stage1_gate/summary_metrics.csv`.
- TCM and mechanism diagnostics to `CORMOL/results/stage1_gate/mechanism_metrics.csv`.
- A final analysis report to `CORMOL/reports/stage1_gate_report.md`.

## Constraints

The local BORF directory is not a Git repository, so this work cannot be committed from this workspace. The design and implementation artifacts are still written to deterministic paths under `CORMOL/`.
