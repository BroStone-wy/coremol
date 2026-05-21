# Graphformer Profile Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configurable Graphformer submodule profiles so each MoleculeNet task can select the backbone components that reproduce the paper-scale baseline before CoReMol is evaluated.

**Architecture:** Keep CoReMol as the same backbone-agnostic residual communication adapter. Refactor only the Graphformer-lite backbone so graph token readout, local GNN pre-encoder, distance bias, edge bias, degree encoding, FFN ratio, norm style, and pooling can be toggled by command-line configuration. Add a small profile sweep runner that first runs baseline-only probes, then fixed-base CoReMol once a baseline profile is acceptable.

**Tech Stack:** Python, PyTorch, PyTorch Geometric, scikit-learn ROC-AUC, conda environment `GNRF`.

---

### Task 1: Add Graphformer Profile Configuration

**Files:**
- Modify: `CORMOL/coremol/models/graphformer_coremol.py`
- Modify: `CORMOL/scripts/run_stage1_gate.py`
- Test: `CORMOL/tests/test_graphformer_coremol.py`

- [ ] **Step 1: Write tests for profile toggles**

Add tests that instantiate `CoReMolGraphformer` with graph token and edge bias toggles, verify output shape, and verify the model can still expose atom states for CoReMol diagnostics.

Run: `conda run --no-capture-output -n GNRF pytest CORMOL/tests/test_graphformer_coremol.py -q`

Expected before implementation: constructor keyword errors for the new profile arguments.

- [ ] **Step 2: Extend the Graphformer constructor**

Add keyword arguments to `CoReMolGraphformer` and `GraphformerBackbone`:

```python
use_graph_token: bool = False
readout: str = "mean_max"
use_local_gnn: bool = True
use_distance_bias: bool = True
use_edge_bias: bool = False
use_degree_encoding: bool = True
ffn_ratio: int = 4
norm_style: str = "pre"
num_heads: int = 4
max_distance: int = 5
```

The default values must preserve current behavior.

- [ ] **Step 3: Implement toggles without changing CoReMol**

Keep `CoReMolResidualAdapter` unchanged. In `GraphformerBackbone`, gate each optional component:

```python
if self.use_degree_encoding:
    atoms = atoms + self.degree_encoder(degree)
if self.use_local_gnn:
    for conv in self.local_convs:
        atoms = atoms + F.relu(conv(atoms, edge_index, edge_features))
```

For graph token, prepend a learned token to dense atom states before Transformer layers and use it only in readout when `readout == "graph_token"`.

- [ ] **Step 4: Add CLI arguments**

Expose the profile controls in `run_stage1_gate.py`:

```python
parser.add_argument("--graphformer_use_graph_token", action="store_true")
parser.add_argument("--graphformer_no_local_gnn", action="store_true")
parser.add_argument("--graphformer_no_distance_bias", action="store_true")
parser.add_argument("--graphformer_use_edge_bias", action="store_true")
parser.add_argument("--graphformer_no_degree_encoding", action="store_true")
parser.add_argument("--graphformer_readout", choices=["mean", "mean_max", "graph_token"], default="mean_max")
parser.add_argument("--graphformer_ffn_ratio", type=int, default=4)
parser.add_argument("--graphformer_norm_style", choices=["pre", "post"], default="pre")
parser.add_argument("--graphformer_num_heads", type=int, default=4)
parser.add_argument("--graphformer_max_distance", type=int, default=5)
```

- [ ] **Step 5: Verify tests**

Run: `conda run --no-capture-output -n GNRF pytest CORMOL/tests/test_graphformer_coremol.py CORMOL/tests/test_run_curvflow_classification_sweep.py -q`

Expected: all selected tests pass.

### Task 2: Add Baseline-First Profile Sweep Runner

**Files:**
- Create: `CORMOL/scripts/run_graphformer_profile_sweep.py`
- Test: `CORMOL/tests/test_graphformer_profile_sweep.py`

- [ ] **Step 1: Write command-builder tests**

Test that the runner can create baseline-only commands and fixed-base CoReMol commands for SIDER, ClinTox, and ToxCast.

Run: `conda run --no-capture-output -n GNRF pytest CORMOL/tests/test_graphformer_profile_sweep.py -q`

Expected before implementation: import failure.

- [ ] **Step 2: Implement profile definitions**

Implement compact profile dictionaries:

```python
PROFILES = {
    "lite_current": {},
    "graph_token_spd": {"graphformer_use_graph_token": True, "graphformer_readout": "graph_token"},
    "graph_token_edge": {"graphformer_use_graph_token": True, "graphformer_readout": "graph_token", "graphformer_use_edge_bias": True},
    "no_local_edge": {"graphformer_no_local_gnn": True, "graphformer_use_graph_token": True, "graphformer_readout": "graph_token", "graphformer_use_edge_bias": True},
}
```

- [ ] **Step 3: Implement baseline-first execution**

The runner must support:

```bash
python CORMOL/scripts/run_graphformer_profile_sweep.py \
  --stage baseline \
  --datasets SIDER CLINTOX TOXCAST \
  --profiles graph_token_spd graph_token_edge no_local_edge \
  --seeds 0 \
  --epochs 80
```

It should call `run_stage1_gate.py --variants base` and save results under:

`CORMOL/results/graphformer_profile_sweep/<dataset>/<profile>/baseline_seed<seed>`.

- [ ] **Step 4: Implement fixed-base CoReMol execution**

The runner must support:

```bash
python CORMOL/scripts/run_graphformer_profile_sweep.py \
  --stage coremol \
  --datasets SIDER \
  --profiles graph_token_edge \
  --seeds 0 1 2 \
  --message delta
```

It should load baseline checkpoints via `--fixed_base_dir`, run `--variants base coremol`, and save under:

`CORMOL/results/graphformer_profile_sweep/<dataset>/<profile>/coremol_delta_seeds0_1_2`.

- [ ] **Step 5: Verify command tests**

Run: `conda run --no-capture-output -n GNRF pytest CORMOL/tests/test_graphformer_profile_sweep.py -q`

Expected: command generation tests pass.

### Task 3: Run First Baseline Probe Queue

**Files:**
- Create: `CORMOL/results/graphformer_profile_sweep/_logs/run_baseline_probe_queue.sh`
- Output: `CORMOL/results/graphformer_profile_sweep/**/raw_metrics.csv`

- [ ] **Step 1: Stop obsolete low-value queues**

Run:

```bash
ps -eo pid,args | rg 'run_stage1_gate.py|run_graphformer' | rg -v rg
```

If an obsolete SIDER h128/l6 profile is still running, stop it and keep its logs.

- [ ] **Step 2: Launch baseline-only profile probes**

Run seed0 baseline-only probes for SIDER, ClinTox, and ToxCast:

```bash
bash CORMOL/results/graphformer_profile_sweep/_logs/run_baseline_probe_queue.sh
```

The queue should run sequentially to avoid GPU contention.

- [ ] **Step 3: Monitor every 1-2 minutes**

Run:

```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
tail -n 80 CORMOL/results/graphformer_profile_sweep/_logs/baseline_probe_queue.log
```

Report current dataset/profile/epoch and whether the baseline is moving toward paper-scale AUC.

- [ ] **Step 4: Select profiles using explicit criteria**

Keep a profile if seed0 baseline meets either condition:

```text
SIDER: valid/test AUC >= 0.75 for further seeds
ClinTox: valid/test AUC >= 0.90 for further seeds
ToxCast: valid/test AUC >= 0.75 for further seeds
```

If none meet criteria, record that Graphformer-lite remains structurally underpowered and proceed to a larger Graphormer-compatible backbone plan.

### Task 4: Run CoReMol on Accepted Profiles

**Files:**
- Output: `CORMOL/results/graphformer_profile_sweep/**/summary_metrics.csv`
- Output: `CORMOL/results/graphformer_profile_sweep/**/mechanism_metrics.csv`
- Output: `CORMOL/results/graphformer_profile_sweep/graphformer_profile_ablation_report.md`

- [ ] **Step 1: Run fixed-base CoReMol**

For accepted profiles, run three seeds with both message definitions:

```bash
python CORMOL/scripts/run_graphformer_profile_sweep.py \
  --stage coremol \
  --datasets SIDER CLINTOX TOXCAST \
  --profiles <accepted_profiles> \
  --seeds 0 1 2 \
  --messages value delta
```

- [ ] **Step 2: Summarize task and TCM metrics**

Compute mean/std AUC, wins, and TCM-positive counts. The report must distinguish:

```text
baseline profile selection
fixed-base CoReMol comparison
value vs delta residual message
Full TCM and TCM@10 if available
```

- [ ] **Step 3: Decide next experiment**

If CoReMol improves task AUC and TCM on accepted profiles, launch 3-seed confirmation. If baseline still fails paper scale, stop claiming Graphformer-paper comparison for this implementation and move to importing the full Graphormer preprocessing/encoder.

---

## Self-Review

Spec coverage: The plan covers configurable Graphformer submodules, baseline-first selection, fixed-base CoReMol evaluation, and explicit failure handling.

Placeholder scan: No TBD/TODO placeholders are present. Thresholds and paths are explicit.

Type consistency: CLI option names are consistently prefixed with `graphformer_`, and runner profile keys map directly to those CLI options.
