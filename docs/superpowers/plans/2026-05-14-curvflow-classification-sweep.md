# CurvFlow Classification Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the remaining six CurvFlow classification benchmarks with the BBBP `0.62 -> 0.71` protocol, comparing CoReMol `value` and `delta` message variants with saved configs, metrics, TCM, and best-run reports.

**Architecture:** Extend the existing CORMOL AttentiveFP/CoReMol pipeline rather than creating a parallel runner. Add MoleculeNet task metadata for BACE, ClinTox, Tox21, HIV, SIDER, and ToxCast; make the trainer support multi-task classification with missing labels; add reproducible experiment configs and a small sweep wrapper that writes every command, config, and summary under `CORMOL/results/curvflow_classification_sweep/`.

**Tech Stack:** Python, PyTorch, PyTorch Geometric MoleculeNet, RDKit scaffold splits, scikit-learn ROC-AUC, conda environment `GNRF`.

---

### Task 1: Add Multi-Task MoleculeNet Classification Support

**Files:**
- Modify: `CORMOL/coremol/datasets/moleculenet.py`
- Modify: `CORMOL/coremol/metrics/task_metrics.py`
- Modify: `CORMOL/coremol/training/trainer.py`
- Modify: `CORMOL/scripts/run_stage1_gate.py`
- Test: `CORMOL/tests/test_multitask_classification.py`

- [ ] **Step 1: Write failing tests for task metadata and masked multi-task ROC-AUC**

Create `CORMOL/tests/test_multitask_classification.py` with tests that assert:

```python
import math

import numpy as np
import torch

from coremol.datasets.moleculenet import TASKS
from coremol.metrics.task_metrics import classification_metrics
from coremol.training.trainer import masked_classification_loss


def test_remaining_curvflow_classification_tasks_are_registered():
    expected = {
        "BACE": 1,
        "CLINTOX": 2,
        "TOX21": 12,
        "HIV": 1,
        "SIDER": 27,
        "TOXCAST": 617,
    }

    for name, target_dim in expected.items():
        assert TASKS[name]["type"] == "classification"
        assert TASKS[name]["metric"] == "roc_auc"
        assert TASKS[name]["target_dim"] == target_dim


def test_classification_metrics_average_tasks_and_ignore_nan_labels():
    y_true = np.array(
        [
            [0.0, 1.0, np.nan],
            [1.0, 0.0, np.nan],
            [0.0, np.nan, 1.0],
            [1.0, np.nan, 0.0],
        ]
    )
    logits = np.array(
        [
            [-2.0, 2.0, 0.0],
            [2.0, -2.0, 0.0],
            [-1.0, 0.0, 3.0],
            [1.0, 0.0, -3.0],
        ]
    )

    metrics = classification_metrics(y_true, logits)

    assert metrics["roc_auc"] == 1.0
    assert metrics["roc_auc_tasks"] == 3


def test_masked_classification_loss_ignores_nan_labels():
    logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    target = torch.tensor([[1.0, float("nan")], [0.0, float("nan")]])

    loss = masked_classification_loss(logits, target)

    assert math.isclose(float(loss), 0.693147, rel_tol=1e-5)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
PYTHONPATH=CORMOL conda run --no-capture-output -n GNRF pytest -q CORMOL/tests/test_multitask_classification.py
```

Expected: FAIL because datasets are not registered and `masked_classification_loss` does not exist.

- [ ] **Step 3: Implement task metadata**

Update `TASKS` in `CORMOL/coremol/datasets/moleculenet.py`:

```python
"BACE": {"type": "classification", "metric": "roc_auc", "target_dim": 1},
"CLINTOX": {"type": "classification", "metric": "roc_auc", "target_dim": 2, "pyg_name": "ClinTox"},
"TOX21": {"type": "classification", "metric": "roc_auc", "target_dim": 12},
"HIV": {"type": "classification", "metric": "roc_auc", "target_dim": 1},
"SIDER": {"type": "classification", "metric": "roc_auc", "target_dim": 27},
"TOXCAST": {"type": "classification", "metric": "roc_auc", "target_dim": 617, "pyg_name": "ToxCast"},
```

- [ ] **Step 4: Implement masked ROC-AUC and masked BCE**

Update `classification_metrics` to iterate task columns, ignore `nan`, skip tasks with only one class, and return `{"roc_auc": mean_auc, "roc_auc_tasks": count}`. Add `masked_classification_loss(logits, target)` in `trainer.py` using `torch.isfinite(target)`.

- [ ] **Step 5: Make model output dimension task-aware**

In `run_stage1_gate.py`, set:

```python
out_channels = TASKS[args_dataset_name]["target_dim"]
```

by passing `dataset_name` into `build_model` or by passing `target_dim` explicitly.

- [ ] **Step 6: Run tests and full existing test suite**

Run:

```bash
PYTHONPATH=CORMOL conda run --no-capture-output -n GNRF pytest -q CORMOL/tests
```

Expected: all tests pass.

### Task 2: Save Configs and Create Sweep Wrapper

**Files:**
- Modify: `CORMOL/scripts/run_stage1_gate.py`
- Create: `CORMOL/scripts/run_curvflow_classification_sweep.py`
- Create: `CORMOL/configs/curvflow_classification_value.json`
- Create: `CORMOL/configs/curvflow_classification_delta.json`
- Test: `CORMOL/tests/test_experiment_config.py`

- [ ] **Step 1: Write failing config test**

Create `CORMOL/tests/test_experiment_config.py`:

```python
import json
from pathlib import Path


def test_curvflow_classification_configs_define_six_remaining_datasets():
    value = json.loads(Path("CORMOL/configs/curvflow_classification_value.json").read_text())
    delta = json.loads(Path("CORMOL/configs/curvflow_classification_delta.json").read_text())

    assert value["datasets"] == ["BACE", "CLINTOX", "TOX21", "HIV", "SIDER", "TOXCAST"]
    assert delta["datasets"] == value["datasets"]
    assert value["residual_message"] == "value"
    assert delta["residual_message"] == "delta"
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
PYTHONPATH=CORMOL conda run --no-capture-output -n GNRF pytest -q CORMOL/tests/test_experiment_config.py
```

Expected: FAIL because config files do not exist.

- [ ] **Step 3: Add config saving to `run_stage1_gate.py`**

At run start, write `vars(args)` to `results_dir / "run_config.json"` so every run can be traced.

- [ ] **Step 4: Add two JSON config files**

Create value config with BBBP-matched starting protocol:

```json
{
  "datasets": ["BACE", "CLINTOX", "TOX21", "HIV", "SIDER", "TOXCAST"],
  "seeds": [0, 1, 2],
  "epochs": 60,
  "batch_size": 64,
  "hidden_channels": 32,
  "num_layers": 2,
  "num_timesteps": 2,
  "dropout": 0.1,
  "lr": 0.001,
  "weight_decay": 0.00001,
  "patience": 12,
  "d_max": 4,
  "support_hops": 3,
  "beta": 0.2,
  "tau": 0.5,
  "residual_gate_init": 0.1,
  "residual_placement": "post",
  "num_residual_steps": 1,
  "residual_message": "value",
  "residual_norm_mode": "layernorm",
  "split_strategy": "scaffold"
}
```

Create delta config with the same values except `"residual_message": "delta"`.

- [ ] **Step 5: Add sweep wrapper**

Create `run_curvflow_classification_sweep.py` that loads a config JSON, builds a `run_stage1_gate.py` command, saves `launch_command.txt`, and runs it with `subprocess.run(check=True)`. It must accept `--config`, `--results_name`, and `--dry_run`.

- [ ] **Step 6: Run config tests**

Run:

```bash
PYTHONPATH=CORMOL conda run --no-capture-output -n GNRF pytest -q CORMOL/tests/test_experiment_config.py
```

Expected: PASS.

### Task 3: Execute Value and Delta Sweeps

**Files:**
- Read: `CORMOL/configs/curvflow_classification_value.json`
- Read: `CORMOL/configs/curvflow_classification_delta.json`
- Output: `CORMOL/results/curvflow_classification_sweep/value_post_d4_b02_t05/`
- Output: `CORMOL/results/curvflow_classification_sweep/delta_post_d4_b02_t05/`

- [ ] **Step 1: Run value sweep**

Run:

```bash
PYTHONPATH=CORMOL conda run --no-capture-output -n GNRF python CORMOL/scripts/run_curvflow_classification_sweep.py \
  --config CORMOL/configs/curvflow_classification_value.json \
  --results_name curvflow_classification_sweep/value_post_d4_b02_t05
```

- [ ] **Step 2: Run delta sweep**

Run:

```bash
PYTHONPATH=CORMOL conda run --no-capture-output -n GNRF python CORMOL/scripts/run_curvflow_classification_sweep.py \
  --config CORMOL/configs/curvflow_classification_delta.json \
  --results_name curvflow_classification_sweep/delta_post_d4_b02_t05
```

- [ ] **Step 3: Verify baseline proximity**

Compare base mean ROC-AUC against the CurvFlow table values:

```text
BACE 0.784
CLINTOX 0.847
TOX21 0.757
HIV 0.761
SIDER 0.606
TOXCAST 0.637
```

Flag any dataset whose baseline differs by more than `0.08` absolute ROC-AUC for targeted rerun; do not claim paper-level comparison for flagged datasets.

### Task 4: Summarize and Select Best Config

**Files:**
- Create: `CORMOL/scripts/summarize_curvflow_classification_sweep.py`
- Output: `CORMOL/results/curvflow_classification_sweep/summary.csv`
- Output: `CORMOL/results/curvflow_classification_sweep/best_configs.json`
- Output: `CORMOL/results/curvflow_classification_sweep/report.md`

- [ ] **Step 1: Implement summarizer**

Read each run directory's `raw_metrics.csv`, `mechanism_metrics.csv`, `tcm_variants.csv` when present. For each dataset and message variant compute:

```text
base_mean_auc
coremol_mean_auc
auc_improvement
task_wins
delta_tcm_mean
tcm_wins
baseline_gap_to_curvflow_attentivefp
```

- [ ] **Step 2: Select best config per dataset**

Choose the variant that satisfies task improvement and TCM improvement. If both satisfy, choose larger mean ROC-AUC improvement. If neither satisfies, mark as `needs_rerun` and include the failure reason.

- [ ] **Step 3: Run summarizer**

Run:

```bash
PYTHONPATH=CORMOL conda run --no-capture-output -n GNRF python CORMOL/scripts/summarize_curvflow_classification_sweep.py \
  --root CORMOL/results/curvflow_classification_sweep
```

- [ ] **Step 4: Verification**

Run:

```bash
PYTHONPATH=CORMOL conda run --no-capture-output -n GNRF pytest -q CORMOL/tests
```

Expected: all tests pass.

### Task 5: Targeted Reruns if Needed

**Files:**
- Create as needed: `CORMOL/configs/curvflow_classification_<dataset>_<variant>_rerun.json`
- Output as needed: `CORMOL/results/curvflow_classification_sweep/<dataset>_<variant>_<params>/`

- [ ] **Step 1: Diagnose failures before changing parameters**

For each failed dataset, inspect:

```text
baseline_gap_to_curvflow_attentivefp
task_wins
tcm_wins
update_atom_norm_ratio
residual_gate
```

- [ ] **Step 2: Apply one targeted mechanism-preserving adjustment**

Allowed adjustments:

```text
d_max: 2 or 4
beta: 0.1, 0.2, 0.35
tau: 0.5 or 0.75
residual_gate_init: 0.05 or 0.1
residual_norm_mode: layernorm or none
```

Do not add new modules. Keep residual communication mechanism unchanged.

- [ ] **Step 3: Rerun only flagged dataset/variant**

Use `run_stage1_gate.py` directly with `--datasets <DATASET>` and the chosen parameters. Save under a descriptive result directory.

- [ ] **Step 4: Re-run summarizer**

Update `best_configs.json` and `report.md`.

---

## Self-Review

- Spec coverage: covers six remaining classification datasets, value/delta variants, baseline proximity, CoReMol task improvement, TCM improvement, config/result persistence.
- Placeholder scan: no implementation step contains TBD/TODO; targeted reruns have bounded parameter sets.
- Type consistency: uses existing `TASKS`, `classification_metrics`, `train_model`, `run_stage1_gate.py`, and `CoReMolConfig` names.
