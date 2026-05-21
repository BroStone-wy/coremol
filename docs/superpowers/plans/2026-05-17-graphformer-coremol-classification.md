# Graphformer CoReMol Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Graphformer-like molecular backbone to the existing CORMOL pipeline and run classification baseline plus CoReMol experiments.

**Architecture:** Implement a lightweight PyG-compatible Graphformer backbone inside `CORMOL/coremol/models`, using node feature projection, shortest-path attention bias, Transformer blocks, and graph-level readout. Reuse the existing CoReMol residual adapter on post-encoder atom states so the mechanism remains backbone-agnostic.

**Tech Stack:** PyTorch, PyTorch Geometric, existing CORMOL trainer/split/TCM scripts.

---

### Task 1: Add Graphformer-Lite Model

**Files:**
- Create: `CORMOL/coremol/models/graphformer_coremol.py`
- Test: `CORMOL/tests/test_graphformer_coremol.py`

- [ ] Implement `CoReMolGraphformer` with the same public methods as `CoReMolAttentiveFP`: `forward`, `apply_coremol`, and `_apply_single_graph`.
- [ ] Use per-graph shortest-path distance bias in multi-head attention.
- [ ] Add smoke tests for base forward shape, CoReMol diagnostics, and non-empty gradients.

### Task 2: Wire Backbone Selection Into Runner

**Files:**
- Modify: `CORMOL/scripts/run_stage1_gate.py`
- Modify: `CORMOL/scripts/run_curvflow_classification_sweep.py`
- Test: `CORMOL/tests/test_graphformer_coremol.py`

- [ ] Add `--backbone attentivefp|graphformer`.
- [ ] Preserve existing AttentiveFP behavior as default.
- [ ] Forward the `backbone` scalar argument from sweep config.

### Task 3: Verify And Run Classification

**Files:**
- Create results under: `CORMOL/results/graphformer_classification_sweep`

- [ ] Run unit tests with `PYTHONPATH=CORMOL conda run --no-capture-output -n GNRF pytest -q CORMOL/tests`.
- [ ] Smoke-run BBBP seed 0 baseline/coremol to verify training and TCM output.
- [ ] Run 7 classification datasets with saved configs and checkpoints.
- [ ] Summarize mean/std, paired wins, and Full TCM wins into a markdown report.
