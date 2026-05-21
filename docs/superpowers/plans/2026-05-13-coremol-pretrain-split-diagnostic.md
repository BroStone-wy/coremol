# CoReMol Pretrain Split Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fair ESOL diagnostic that separates split difficulty from ZINC pretraining and tests whether CoReMol residual communication benefits from branch-level ZINC pretraining.

**Architecture:** Keep all new code under `CORMOL/`. Add a configurable random split with 80/10/10 ratio, add residual-branch pretraining that freezes an AttentiveFP backbone and trains only CoReMol communication parameters on ZINC, then run ESOL scaffold/random protocols with comparable ratios.

**Tech Stack:** Python, PyTorch, PyTorch Geometric, RDKit, pandas, pytest, conda env `GNRF`.

---

### Task 1: Random 80/10/10 Split Support

**Files:**
- Modify: `CORMOL/coremol/datasets/random_split.py`
- Modify: `CORMOL/scripts/run_stage1_gate.py`
- Modify: `CORMOL/tests/test_pretraining.py`

- [ ] Add failing tests for `curvflow_random_split_indices(..., train_fraction=0.8, valid_fraction=0.1)` returning 80/10/10.
- [ ] Update random split implementation to accept split fractions and version names.
- [ ] Add `--split_strategy random` and ratio args to stage1 runner.
- [ ] Run `PYTHONPATH=. conda run -n GNRF pytest -q tests/test_pretraining.py`.

### Task 2: Residual Branch ZINC Pretraining

**Files:**
- Create: `CORMOL/scripts/pretrain_zinc_coremol_residual.py`
- Modify: `CORMOL/coremol/training/pretraining.py`
- Modify: `CORMOL/tests/test_pretraining.py`

- [ ] Add failing test proving backbone freeze leaves only CoReMol parameters trainable.
- [ ] Add helper to freeze backbone and expose trainable parameter names.
- [ ] Add ZINC residual pretraining script: load ZINC subset, load pretrained AttentiveFP backbone, freeze backbone, train CoReMol with normalized regression target, save checkpoint.
- [ ] Run unit tests.

### Task 3: Fair ESOL Runs

**Files:**
- Generated outputs under `CORMOL/results/`
- Update: `CORMOL/reports/esol_zinc_pretrain_attentivefp_coremol_report.md`

- [ ] Run ESOL random 80/10/10 scratch baseline.
- [ ] Run ESOL random 80/10/10 ZINC-pretrained baseline.
- [ ] Run residual-branch ZINC pretraining.
- [ ] Run ESOL scaffold 80/10/10 with residual-pretrained CoReMol.
- [ ] Compute TCM variants for the best run.
- [ ] Update report with split-controlled results and failure/success analysis.
