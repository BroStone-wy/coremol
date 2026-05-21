# D-MPNN CoReMol Backbone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a D-MPNN-style backbone to CoReMol and run paired D-MPNN baseline versus D-MPNN+CoReMol experiments under the same reporting discipline used for AttentiveFP and Graphformer.

**Architecture:** Implement D-MPNN as directed bond-centered message passing following the Yang et al. 2019 design principle: hidden states live on directed bonds during message passing, then aggregate into atom states for molecular readout. CoReMol remains the same backbone-agnostic residual communication adapter over atom states, inserted post-encoder by default and optionally layerwise after atom-state reconstruction.

**Tech Stack:** Python, PyTorch, PyTorch Geometric, RDKit/MoleculeNet loaders already in this repo, pytest, existing `run_stage1_gate.py`, existing `compute_tcm_variants.py`.

---

## Requirements From Project Memory

- Final claim requires both task metric improvement and true Full TCM improvement.
- `mechanism_metrics.delta_tcm` is `TCM@10`, diagnostic only.
- Use `scripts/compute_tcm_variants.py` for true Full TCM.
- Report local D-MPNN baseline and matching D-MPNN+CoReMol, not paper-value comparison alone.
- Prefer 3-seed mean/std; seed screening is allowed only if explicitly marked and all raw results are retained.
- CoReMol must stay plug-and-play: residual calculation uses atom states, graph connectivity, and batch; do not hard-code to D-MPNN private internals.

## Paper Anchor

The D-MPNN row in the Graph Curvature Flow-Based Masked Attention table corresponds to the directed message passing neural network from Yang et al. 2019, where messages are centered on directed bonds rather than atoms. We will implement this principle locally rather than importing Chemprop wholesale, because the project needs the same PyG `Data` interface and CoReMol diagnostics as existing backbones.

Paper table anchors to track:

- Classification table D-MPNN anchors: BBBP `0.710`, BACE `0.809`, ClinTox `0.905`, Tox21 `0.759`, HIV `0.771`, SIDER `0.570`, ToxCast `0.655`.
- Regression table D-MPNN anchors: ESOL `1.050`, FreeSolv `2.082`, Lipo `0.683`.

These values are reference anchors. The final report must emphasize local paired baseline versus local D-MPNN+CoReMol.

## Target Datasets

Primary classification batch:

```text
BBBP BACE CLINTOX TOX21 HIV SIDER TOXCAST
```

Primary regression batch:

```text
ESOL FREESOLV LIPO
```

Protocol starts from the same practical project standard used in the latest Graphformer/AttentiveFP runs:

- classification: use the split protocol already used for the matching dataset family in current configs, with paper table values as anchors;
- regression: use random 80/10/10 no-pretrain first, because this is the latest saved Graphformer regression protocol;
- if strict CurvFlow/ZINC-pretrain comparison is needed later, add it as a separate protocol, not mixed into the first D-MPNN run.

## Task 1: Add D-MPNN Model Wrapper

**Files:**
- Create: `coremol/models/dmpnn_coremol.py`
- Modify: `coremol/models/__init__.py`
- Test: `tests/test_dmpnn_coremol.py`

- [ ] **Step 1: Write model shape tests**

Create `tests/test_dmpnn_coremol.py` with tests that instantiate `CoReMolDMPNN`, run base/coremol forwards on a small PyG `Data` batch, and verify:

```text
base output shape == [num_graphs, out_channels]
coremol output shape == [num_graphs, out_channels]
return_diagnostics=True returns non-empty CoReMol diagnostics when enabled
```

- [ ] **Step 2: Implement directed edge indexing**

In `coremol/models/dmpnn_coremol.py`, implement helpers:

```text
directed_edge_index = original directed PyG edges
reverse_edge_index[e] = index of edge dst->src when present
incoming_edge_index for each directed edge dst->src receives messages from k->dst except src->dst reverse
```

This keeps D-MPNN bond-centered and prevents immediate reverse-edge echo.

- [ ] **Step 3: Implement `DMPNNBackbone`**

Implement:

```text
atom_proj: Linear(in_channels -> hidden)
bond_proj: Linear(edge_dim -> hidden)
edge_init: Linear(atom_src || bond_attr -> hidden)
message_layers: num_layers rounds of directed bond update
atom_update: aggregate incoming directed bond states into atom states
readout: mean or mean_max graph pooling + MLP head
```

Initial update rule:

```text
m_e = ReLU(W_i([x_src, e_attr]))
for t in range(num_layers):
    incoming = sum_{k->src(e), k != dst(e)} m_{k->src(e)}
    m_e = ReLU(m_e_initial + W_h(incoming))
atom_state_i = ReLU(W_o([atom_proj(x_i), sum_{e: dst(e)=i} m_e]))
graph_repr = global_mean_pool(atom_state, batch)
prediction = graph_head(graph_repr)
```

- [ ] **Step 4: Implement `CoReMolDMPNN`**

Match the public API of existing wrappers:

```text
forward(data, return_diagnostics=False)
encode_atoms(x, edge_index, edge_attr, batch, return_diagnostics=False)
apply_coremol(atoms, edge_index, batch, return_diagnostics=True)
_apply_single_graph(atoms, edge_index)
```

Default CoReMol placement:

```text
post: after D-MPNN atom_state construction, before graph readout
layerwise: optional after each atom-state reconstruction checkpoint if implemented
```

Initial implementation may support `post` and `both` as post-equivalent, then add true layerwise only if task/TCM diagnosis needs it.

- [ ] **Step 5: Run D-MPNN tests**

Run:

```bash
PYTHONPATH=. conda run --no-capture-output -n GNRF pytest -q tests/test_dmpnn_coremol.py
```

Expected:

```text
all tests pass
```

## Task 2: Register D-MPNN in Training and Full TCM Scripts

**Files:**
- Modify: `scripts/run_stage1_gate.py`
- Modify: `scripts/compute_tcm_variants.py`
- Test: `tests/test_dmpnn_coremol.py`
- Test: `tests/test_compute_tcm_variants.py`

- [ ] **Step 1: Add CLI choice**

Update both parsers:

```text
--backbone choices include attentivefp, graphformer, dmpnn
```

- [ ] **Step 2: Add D-MPNN construction**

In both `build_model` functions:

```python
if args.backbone == "dmpnn":
    return CoReMolDMPNN(**common_kwargs, readout=args.dmpnn_readout)
```

Add optional D-MPNN-specific args:

```text
--dmpnn_readout choices=["mean", "mean_max"], default="mean"
```

Use existing `hidden_channels`, `num_layers`, `dropout`, `edge_dim`, `out_channels`, and CoReMol args.

- [ ] **Step 3: Preserve warm start and backbone LR scaling**

No special-case needed if `CoReMolDMPNN` exposes `.backbone`; `run_stage1_gate.py` already copies `trained["base"].backbone.state_dict()` into `model.backbone`.

- [ ] **Step 4: Run full test suite**

Run:

```bash
PYTHONPATH=. conda run --no-capture-output -n GNRF pytest -q tests
```

Expected:

```text
all tests pass
```

## Task 3: D-MPNN Baseline Alignment Pilot

**Files:**
- Create: `configs/dmpnn_classification_pilot.json`
- Create: `configs/dmpnn_regression_pilot.json`
- Generated: `results/dmpnn_*`

- [ ] **Step 1: Run smoke task**

Run BBBP for one seed:

```bash
PYTHONPATH=. conda run --no-capture-output -n GNRF python scripts/run_stage1_gate.py \
  --backbone dmpnn \
  --datasets BBBP \
  --seeds 0 \
  --epochs 30 \
  --batch_size 64 \
  --hidden_channels 128 \
  --num_layers 3 \
  --dropout 0.10 \
  --lr 0.001 \
  --weight_decay 1e-5 \
  --patience 8 \
  --split_strategy scaffold \
  --variants base coremol \
  --warm_start_coremol \
  --residual_message delta \
  --residual_score_space distribution \
  --residual_placement post \
  --d_max 2 \
  --support_hops 2 \
  --beta 0.35 \
  --tau 0.7 \
  --residual_gate_init 0.04 \
  --backbone_lr_scale 0.2 \
  --results_name dmpnn_bbbp_smoke_seed0
```

- [ ] **Step 2: Compute Full TCM for smoke**

Run:

```bash
PYTHONPATH=. conda run --no-capture-output -n GNRF python scripts/compute_tcm_variants.py \
  --run_dir results/dmpnn_bbbp_smoke_seed0 \
  --datasets BBBP \
  --seeds 0 \
  --backbone dmpnn \
  --hidden_channels 128 \
  --num_layers 3 \
  --dropout 0.10 \
  --d_max 2 \
  --support_hops 2 \
  --beta 0.35 \
  --tau 0.7 \
  --split_strategy scaffold \
  --residual_placement post \
  --residual_message delta \
  --residual_score_space distribution \
  --max_graphs 96
```

- [ ] **Step 3: Diagnose baseline scale**

Compare local D-MPNN baseline to paper anchors. If baseline is far below anchor, tune only D-MPNN hyperparameters first:

```text
hidden_channels: 128, 256
num_layers: 3, 4, 5
dropout: 0.0, 0.1, 0.2
readout: mean, mean_max
lr: 0.0005, 0.001
class_balance for multi-task classification if needed
```

Do not tune CoReMol until the baseline is reasonable.

## Task 4: CoReMol Tuning With Fixed D-MPNN Baseline

**Files:**
- Generated: `results/dmpnn_classification_*`
- Generated: `results/dmpnn_regression_*`
- Create: `reports/dmpnn_coremol_interim_YYYY_MM_DD.md`

- [ ] **Step 1: Run paired 3-seed classification candidates**

Use selected baseline profile. Initial CoReMol grid:

```text
residual_message: delta, value
residual_score_space: distribution, rir
beta: 0.20, 0.35, 0.65
tau: 0.5, 0.7, 1.0
residual_gate_init: 0.02, 0.04, 0.08
d_max: 2, 3
support_hops: 2, 3
backbone_lr_scale: 0.1, 0.2
```

Change one variable at a time when diagnosing failures.

- [ ] **Step 2: Run paired 3-seed regression candidates**

Start with:

```text
datasets: ESOL FREESOLV LIPO
split: random 80/10/10
normalize_regression: true
hidden_channels: 128
num_layers: 3 or 4
```

Use the same CoReMol grid as classification, but judge RMSE decrease instead of AUC increase.

- [ ] **Step 3: Compute Full TCM for every candidate retained for reporting**

Run `scripts/compute_tcm_variants.py` for each candidate. Reporting candidate must satisfy:

```text
task mean improves in correct direction
Full TCM improves on all selected/reporting seeds if possible
TCM@10 is diagnostic only
```

- [ ] **Step 4: Record failures**

For every failed candidate, record one reason:

```text
baseline too weak
task improves but Full TCM drops
Full TCM improves but task drops
gate too strong
gate too weak
residual insertion too late
readout ignores corrected atom states
```

## Task 5: Final D-MPNN Report and GitHub Update

**Files:**
- Create: `reports/dmpnn_coremol_full_tcm_summary_YYYY_MM_DD.md`
- Create: `reports/dmpnn_coremol_full_tcm_summary_YYYY_MM_DD.csv`
- Modify: `README.md` if D-MPNN becomes stable

- [ ] **Step 1: Summarize final selected results**

Report table columns:

```text
Dataset
Paper D-MPNN anchor
Protocol
Selected seeds
Our D-MPNN baseline mean/std
Our D-MPNN+CoReMol mean/std
Delta task metric
Task wins
Full TCM wins
Mean Delta Full TCM
TCM@10 wins
Run directory
Configuration note
Status
```

- [ ] **Step 2: Add reproducibility commands**

Include exact `run_stage1_gate.py` and `compute_tcm_variants.py` commands for every final run.

- [ ] **Step 3: Verify and commit**

Run:

```bash
PYTHONPATH=. conda run --no-capture-output -n GNRF pytest -q tests
git status --short
```

Then commit:

```bash
git add coremol scripts configs tests reports README.md
git commit -m "feat: add dmpnn coremol backbone experiments"
git push
```

---

## Confirmation Needed Before Execution

I will not start coding or training until confirmed.

Proposed first execution scope:

1. Implement D-MPNN backbone and register it in `run_stage1_gate.py` plus `compute_tcm_variants.py`.
2. Run tests.
3. Run BBBP seed0 smoke.
4. If smoke is valid, run a 3-seed D-MPNN baseline alignment pilot before CoReMol tuning.

Please confirm whether to start with:

- Option A: classification first only: `BBBP, BACE, ClinTox, Tox21, HIV, SIDER, ToxCast`;
- Option B: regression first only: `ESOL, FreeSolv, Lipo`;
- Option C: implementation + BBBP smoke first, then decide full classification/regression order.
