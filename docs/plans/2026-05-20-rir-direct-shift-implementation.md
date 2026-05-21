# RIR Direct-Shift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the latest CoReMol-RIR formulation as a direct signed residual interaction rewiring branch, then verify task and normalized Full TCM improvements under paired baseline-vs-RIR experiments.

**Architecture:** Keep the existing CoReMol adapter as the backbone-agnostic insertion point. Add `residual_score_space="rir"` as the main method, where the pair MLP directly predicts signed rewiring logits relative to `C_ref`; retain `intensity` and `distribution` as legacy ablations. Keep diagnostics compatible but avoid invalid “demand mismatch” metrics in RIR mode.

**Tech Stack:** PyTorch, PyTorch Geometric, MoleculeNet loaders, existing `run_stage1_gate.py`, existing TCM probe scripts, pytest.

---

## Design Decisions

- `C_ref` is the structure-induced reference support used only inside the residual branch.
- `alpha_ref = softmax(log(C_ref + eps))`; this is not claimed to be the backbone's true communication.
- RIR predicts `r_ij = MLP(z_ij)` without sigmoid and without subtracting `C_ref`.
- The bounded logit shift is `delta_ij = beta * tanh(r_ij / tau)`. This preserves the current tunable temperature while matching the latest “bounded signed residual shift” design.
- `alpha_rew = softmax(log(C_ref + eps) + delta_ij)`.
- The injected update is `sum_j (alpha_rew - alpha_ref) * m_ij`.
- The default RIR message for main experiments is `m_ij = W h_j - W h_i`.
- Main mechanism reporting uses paired normalized Full TCM: `TCM_norm(q_RIR) - TCM_norm(q_base)`.

## Files

- Modify: `CORMOL/coremol/modules/coremol_adapter.py`
- Modify: `CORMOL/coremol/metrics/mechanism.py`
- Modify: `CORMOL/coremol/training/trainer.py`
- Modify: `CORMOL/scripts/run_stage1_gate.py`
- Modify: `CORMOL/scripts/compute_tcm_variants.py`
- Modify: `CORMOL/scripts/pretrain_zinc_coremol_residual.py`
- Modify: `CORMOL/tests/test_coremol_adapter.py`
- Modify: `CORMOL/tests/test_metrics.py`
- Create: `CORMOL/results/graphformer_classification_sweep/_logs/run_graphformer_rir_smoke_2026_05_20.sh`

## Task 1: Add RIR Semantics Tests

- [ ] Add a test proving `residual_score_space="rir"` accepts signed raw pair scores and does not apply sigmoid, support subtraction, or demand softmax.
- [ ] Add a test proving RIR diagnostics contain `alpha_ref`, `alpha_rew`, `shift_raw`, and `rewiring_delta`, while preserving `alpha_base` and `alpha_cal` aliases.
- [ ] Add a test proving `mechanism_summary` returns `mismatch_reduction=nan` when no demand target exists.
- [ ] Run:

```bash
PYTHONPATH=CORMOL conda run --no-capture-output -n GNRF pytest -q \
  CORMOL/tests/test_coremol_adapter.py::test_rir_residual_score_uses_signed_shift_directly \
  CORMOL/tests/test_coremol_adapter.py::test_rir_diagnostics_use_reference_and_rewired_names \
  CORMOL/tests/test_metrics.py::test_mechanism_summary_skips_demand_mismatch_for_rir
```

Expected: fail before implementation because `rir` is not supported and diagnostics do not expose the new names.

## Task 2: Implement RIR Direct Shift

- [ ] Extend `CoReMolConfig.residual_score_space` validation to include `rir`.
- [ ] Refactor adapter pair scoring so legacy modes use `sigmoid(demand_net(z))`, but RIR uses raw `demand_net(z)` as `shift_raw`.
- [ ] In `compute_residual_scores`, return direct `shift_raw` for RIR, `demand_alpha - alpha_ref` for distribution, and `demand - C_ref` for intensity.
- [ ] Use `rewiring_delta = beta * tanh(residual_score / tau)` for all modes.
- [ ] Add diagnostic aliases:

```text
support / c_ref
alpha_base / alpha_ref
alpha_cal / alpha_rew
residual_score
shift_raw
rewiring_delta
residual_score_space
```

- [ ] Keep old keys so existing TCM and result scripts continue to work.

## Task 3: Fix RIR Mechanism Metrics

- [ ] Update `mechanism_summary` to compute enhance/suppress from `residual_score`, `alpha_ref`, and `alpha_rew`.
- [ ] Compute `mismatch_reduction` only when a valid `demand_alpha` or `demand` target exists.
- [ ] Keep update/gate diagnostics unchanged.
- [ ] Update `residual_demand_alignment_loss` to skip RIR diagnostics, because RIR has no demand target.

## Task 4: Expose RIR in CLI

- [ ] Add `rir` to `--residual_score_space` choices in stage1, TCM variant, and ZINC residual pretrain scripts.
- [ ] Run:

```bash
PYTHONPATH=CORMOL conda run --no-capture-output -n GNRF pytest -q \
  CORMOL/tests/test_coremol_adapter.py CORMOL/tests/test_metrics.py \
  CORMOL/tests/test_graphformer_coremol.py CORMOL/tests/test_compute_tcm_variants.py
```

Expected: all pass.

## Task 5: RIR Smoke Experiments

- [ ] Run Graphformer scaffold smoke on BBBP and BACE with paired seeds, using `residual_score_space=rir`, `residual_message=delta`, true `layerwise` placement, and moderate residual strength.
- [ ] Recompute normalized Full TCM for the generated checkpoints.
- [ ] Compare against the same seeds' baseline metrics and legacy distribution results.
- [ ] If task delta or normalized Full TCM is negative, change one mechanism variable at a time: `beta/tau`, gate scale, placement, or support radius.

## Task 6: Adaptive Optimization Logic

- [ ] If `update_atom_norm_ratio > 0.05` and task drops, reduce gate or beta.
- [ ] If `update_atom_norm_ratio < 0.005` and TCM is flat, increase gate or beta.
- [ ] If Full TCM positive but task flat, keep mechanism and tune training regularization/backbone LR.
- [ ] If task positive but normalized Full TCM negative, inspect benefit/harm components before accepting.
- [ ] If RIR fails while legacy `D-C` works, report the mismatch as an ablation finding and inspect whether raw shift collapses or saturates.

## Self-Review

- No claim treats `C_ref` as actual backbone communication.
- No claim treats an MLP output as true task demand in the RIR main method.
- The scale issue is explicit: RIR avoids `D-C_ref` same-scale assumptions by learning signed logit shifts directly.
- Main TCM is paired and normalized; internal `alpha_rew-alpha_ref` diagnostics are not substituted for baseline-vs-RIR TCM.
- The adapter remains backbone-agnostic because it only consumes atom embeddings, graph edges, batch IDs, and structural support.
