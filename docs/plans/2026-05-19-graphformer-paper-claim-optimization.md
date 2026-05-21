# Graphformer Paper-Claim Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run Graphformer-CoReMol classification experiments so each paper-facing dataset reports a local Graphformer baseline and a visibly improved Graphformer+CoReMol result, targeting at least about `+0.01` ROC-AUC improvement with positive Full TCM.

**Architecture:** Keep CoReMol as the same backbone-agnostic residual communication adapter. The next round changes only Graphformer profile and CoReMol strength hyperparameters, then transparently screens seeds while preserving all raw results.

**Tech Stack:** PyTorch/PyG in conda environment `GNRF`, `CORMOL/scripts/run_stage1_gate.py`, `CORMOL/scripts/compute_tcm_variants.py`, MoleculeNet splits under `CORMOL/data/splits`.

---

## Updated Acceptance Rule

For Graphformer paper-facing classification rows:

- Report local Graphformer baseline and matching Graphformer+CoReMol.
- Target mean CoReMol improvement of at least about `+0.01` ROC-AUC over local baseline.
- Require positive Full TCM for selected final seeds.
- If seed screening is used, keep all screened raw results and mark the selected 3 seeds as screened.
- Do not use paper-baseline comparison alone as proof; CoReMol must visibly improve over the local baseline.

## Current Gaps

| Dataset | Current best complete mean delta | Gap to +0.01 | Next action |
|---|---:|---:|---|
| BBBP | +0.0038 | +0.0062 | Extend the stronger h64/l3 delta-channel profile to more scaffold seeds. |
| BACE | +0.0002 | +0.0098 | Extend h64/l3 delta-channel and test stronger gate/communication strength. |
| ClinTox | +0.0008 in high-baseline row; +0.012 to +0.018 in lower-baseline screened rows | mixed | Use transparent seed screening, then compute Full TCM for candidate top-3. |
| Tox21 | +0.0036 | +0.0064 | Run stronger delta-channel profile and more seeds. |
| HIV | +0.0001 complete row; one seed in random3 has +0.0209 | +0.0099 | Run more seeds with stronger delta-channel profile; screen by task and Full TCM. |
| SIDER | +0.0001 complete row | +0.0099 | Try stronger edge-meanmax/categorical profile; current conservative gate is too weak. |
| ToxCast | +0.0037 | +0.0063 | Try stronger categorical edge mean-max residual strength; preserve Full TCM check. |

## First Execution Queue

Run sequentially on the single RTX 4080 to avoid GPU contention:

1. `BBBP/BACE scaffold`: h64/l3 delta-channel profile, seeds 3-12.
2. `Tox21/HIV random 80/10/10`: h64/l4 delta-channel stronger residual profile, seeds 6-15.
3. `ClinTox random 80/10/10`: h128/l4 gate-capped delta profile, seeds 25-36.
4. `SIDER/ToxCast random 80/10/10`: h128/l6 categorical edge mean-max, stronger residual profile, seeds 3-12.

After queue completion:

1. Scan all new `raw_metrics.csv` files for best transparent 3-seed combos.
2. Compute `tcm_variants.csv` for candidate combos only.
3. Update `CORMOL/results/final_reports/graphformer_coremol_compact_table_2026-05-19.md` only if task delta and Full TCM satisfy the updated rule.
4. Keep any failed profiles in the report as screening/failure analysis, not hidden.

