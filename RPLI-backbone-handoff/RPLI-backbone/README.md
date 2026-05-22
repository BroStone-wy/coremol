# RPLI Backbone Handoff Package

This folder is a compact handoff package for another Codex session or another server.

It contains code snapshots and configuration notes needed to understand the current RPLI+CoReMol affinity backbone and adapt the shared carrier to MoleculeNet-style single-molecule tasks.

The source-of-truth files still live in the normal repository paths. Files under this folder are copied snapshots for readability and transfer.

## Main Handoff Document

Read this first:

```text
docs/RPLI_CoReMol_Backbone_and_SingleMolecule_Adaptation.md
```

That document explains the model story line, current affinity construction, final single-stage dual-head readout, and the requested single-molecule adaptation.

## Code Snapshot Map

### `code/coremol/models/rpli_affinity.py`

Current affinity model.

Important concepts:

- `RPLIAffinityConfig`
- `RPLIAffinity`
- `FeatureTransformMLP`
- `RPLIEdgeModel`
- `RPLINodeModel`
- `RPLIContextModel`
- ligand/pocket pooling
- `readout_mode="dual"`
- internal `base_pred` and `coremol_pred`

This is the main reference for building `MolecularRPLI`.

For MoleculeNet adaptation:

- remove the dependency on ligand/pocket `n_nodes`
- replace ligand/pocket pooling with graph pooling
- keep the carrier pattern and dual-head readout idea
- replace Cross-RIR with intra-molecular reference flow

### `code/coremol/modules/cross_rir_adapter.py`

Current CoReMol Cross-RIR adapter.

Important concepts:

- reference graph construction
- `alpha_ref`
- `delta_cross`
- `alpha_rew`
- `interface_flow`
- conservative residual/readout behavior

For MoleculeNet adaptation:

- preserve the reference/reweighted attention idea
- replace ligand-pocket pair construction with atom-atom or motif-aware molecular pairs
- expose diagnostics with names analogous to affinity diagnostics

### `code/coremol/modules/interface_reference_graph.py`

Current ligand-pocket reference graph constructor.

Important concepts:

- `InterfaceReferenceGraph`
- `InterfaceReferenceGraphConstructor`
- ligand-pocket cross-pair selection
- restoration of true reference distances from GEMS `edge_attr[:, 3:7]`

For MoleculeNet adaptation:

- do not reuse ligand/pocket splitting
- create `coremol/modules/molecular_reference_graph.py`
- start with bond reference graph
- then add shortest-path reference pairs
- optionally add RDKit ring/functional-group references later

### `code/coremol/modules/interface_flow_readout.py`

Current interface-flow summarizer.

Important concepts:

- signed flow
- absolute flow
- scalar flow statistics
- graph-level flow representation

For MoleculeNet adaptation:

- reuse the same readout idea for intra-molecular reference flow
- rename only if needed, such as `MolecularReferenceFlowReadout`
- keep output shape stable for graph-level prediction heads

### `code/scripts/train_coremol_affinity.py`

Current affinity training CLI.

Important concepts:

- `--backbone rpli`
- `--variant coremol`
- `--readout_mode dual`
- `--final_blend_alpha`
- `--base_aux_weight`
- `--coremol_aux_weight`
- single-stage training objective
- checkpoint args saving

For MoleculeNet adaptation:

- create `scripts/train_coremol_moleculenet.py`
- support classification and regression
- support label masks for multi-task datasets
- preserve one-command, one-checkpoint training

### `code/scripts/eval_coremol_affinity.py`

Current affinity evaluation CLI.

Important concepts:

- loads checkpoint args
- rebuilds the model
- evaluates one checkpoint
- writes metrics and predictions

For MoleculeNet adaptation:

- create `scripts/eval_coremol_moleculenet.py`
- report ROC-AUC for classification
- report RMSE/MAE/R2 for regression

### `configs/rpli_coremol_single_stage_dual_a080_b100_c020.md`

Final accepted affinity configuration snapshot.

Important result:

```text
CASF2016 RMSE: 1.2983
CASF2016_indep RMSE: 1.3558
```

Use this file to reproduce the final affinity run and to understand the final single-model settings.

## Required Next Files For Single-Molecule Adaptation

Recommended files for the next Codex to create:

```text
coremol/models/rpli_molecule.py
coremol/modules/molecular_reference_graph.py
scripts/train_coremol_moleculenet.py
scripts/eval_coremol_moleculenet.py
tests/test_rpli_molecule.py
tests/test_molecular_reference_graph.py
tests/test_moleculenet_training_cli.py
```

## Single-Molecule Design Constraints

The MoleculeNet implementation must:

- train with one command
- produce one checkpoint
- avoid external ensembles
- support graph-level classification and regression
- support multi-task missing-label masks
- avoid any protein-pocket dependency
- avoid required `lig_emb`
- avoid required `n_nodes=[total, ligand, protein]`
- keep CoReMol as a reference-conditioned adapter

## Recommended First Implementation

Start with the simplest defensible single-molecule reference graph:

```text
bond reference graph
```

Then add:

```text
shortest-path radius <= 2 or 3
```

Only add RDKit motif/ring references after the bond/shortest-path version works.

## Story Line To Preserve

Use this framing:

> RPLI is a shared molecular representation carrier. CoReMol provides a reference-conditioned calibration adapter. In affinity tasks, the reference relation is ligand-pocket interaction; in single-molecule tasks, the reference relation is intra-molecular chemistry.

Do not frame the method as:

```text
GEMS-style architecture
```

or:

```text
prediction ensemble
```

The final affinity model is one model with internal calibrated readout.
