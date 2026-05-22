# RPLI CoReMol Backbone And Single-Molecule Adaptation Handoff

## 1. Purpose

This document is a handoff for adapting the current RPLI+CoReMol affinity backbone to MoleculeNet-style single-molecule tasks.

The current model is a **single-checkpoint, one-command trained model** for protein-ligand affinity prediction. It should not be described as an ensemble. The final architecture is:

```text
RPLI shared carrier
  + CoReMol reference-conditioned calibration adapter
  + single-stage dual-head calibrated readout
```

The next implementation target is to reuse the same story line for single-molecule datasets:

```text
Molecular RPLI shared carrier
  + CoReMol intra-molecular reference adapter
  + graph-level calibrated readout
```

## 2. Non-Negotiable Requirements

- Keep the final model as **one trainable model** and **one checkpoint**.
- Keep training reproducible with **one unified training command**.
- Do not rely on inference-time ensembles or prediction blending between separately trained models.
- Do not describe the model as a copy of another repository architecture.
- Keep CoReMol as an effective plug-in/reference adapter, not as unused decoration.
- For MoleculeNet adaptation, preserve the same conceptual framework:
  - affinity task: ligand-pocket reference graph
  - single-molecule task: intra-molecular reference graph
- Single-molecule code must not require protein pocket nodes, ligand-pocket `n_nodes`, or ligand global embeddings.

## 3. Current Model Story Line

The current affinity model should be presented as:

> RPLI provides a shared ligand-pocket representation carrier. CoReMol constructs a reference-aware interaction flow over the ligand-pocket interface. A calibrated readout then combines a robust global affinity estimate with a CoReMol-conditioned correction inside one model.

The preferred method name for the final readout is:

```text
Reference-Conditioned Calibration Readout
```

or:

```text
CoReMol-Calibrated Affinity Readout
```

Avoid calling it "internal fusion" in paper text. The implementation is dual-head, but the method story is residual calibration.

## 4. RPLI Shared Carrier Construction

The current RPLI backbone is implemented in:

```text
coremol/models/rpli_affinity.py
```

The main components are:

### 4.1 Feature Transform

```text
FeatureTransformMLP: raw node features -> hidden node states
```

It maps large GEMS node features into a compact hidden dimension.

### 4.2 Edge-Aware Message Passing

RPLI uses a `MetaLayer` with:

```text
RPLIEdgeModel: updates edge states from source node, destination node, and edge_attr
RPLINodeModel: GATv2Conv with edge_attr
RPLIContextModel: graph-level carrier update from pooled nodes and context state
```

This carrier is not specific to affinity prediction; it can be reused for single-molecule graph tasks.

### 4.3 Context Carrier

For affinity data, the context carrier is initialized from `lig_emb` when available:

```text
u_0 = context_transform(lig_emb)
```

For single-molecule datasets, `lig_emb` should be optional. The MoleculeNet version should use:

```text
u_0 = learned fallback context
```

or:

```text
u_0 = graph-level pooled atom embedding
```

### 4.4 Ligand-Pocket Pooling

Affinity tasks use `n_nodes=[total, ligand, protein]` to split nodes into ligand and pocket groups:

```text
h_L = mean_pool(ligand nodes)
h_P = mean_pool(pocket nodes)
```

Single-molecule tasks do not have this split. Replace it with:

```text
h_G = graph-level atom pooling
```

Optionally add motif/ring pooling later, but the first version should stay atom-level and graph-level.

## 5. CoReMol Adapter Construction

The affinity adapter is implemented in:

```text
coremol/modules/cross_rir_adapter.py
coremol/modules/interface_reference_graph.py
coremol/modules/interface_flow_readout.py
```

### 5.1 Reference Graph

For affinity prediction, the reference graph connects ligand atoms to pocket nodes:

```text
C_ref^{LP}: ligand-pocket reference relation
```

The GEMS `.pt` files have `pos=None`, but true ligand-pocket distance information is present in:

```text
edge_attr[:, 3:7]
```

The current implementation restores reference distances from those edge attributes.

### 5.2 Cross-RIR Reweighting

CoReMol computes:

```text
alpha_ref = softmax(log(C_ref))
delta_cross = beta * tanh(MLP(pair_features) / tau)
alpha_rew = softmax(log(C_ref) + delta_cross)
```

The interface flow is:

```text
interface_flow = (alpha_rew - alpha_ref) * pair_message
```

This is the key CoReMol signal. It represents how the learned model shifts the reference interface relation.

### 5.3 Interface Flow Readout

The interface flow is summarized into:

```text
phi_ref = [signed_flow, abs_flow, scalar_stats]
```

where scalar stats include:

```text
mean_abs_delta
mean_alpha_entropy
positive_flow_norm
negative_flow_norm
```

## 6. Final Affinity Readout

The final accepted model uses:

```text
readout_mode = dual
```

It has one shared RPLI+CoReMol carrier and two internal heads:

```text
h_base = [u, h_L, h_P]
h_ref  = [u, h_L, h_P, phi_ref]

y_base = f_base(h_base)
y_ref  = f_ref(h_ref)

y_hat = 0.8 * y_base + 0.2 * y_ref
```

Equivalent residual form:

```text
y_hat = y_base + 0.2 * (y_ref - y_base)
```

This should be explained as **controlled reference-conditioned calibration**, not as an external ensemble.

## 7. Training Objective

The final accepted single-stage loss is:

```text
Loss =
  RMSE(y_hat, y)
  + 1.0 * RMSE(y_base, y)
  + 0.2 * MSE(y_ref, y)
```

Interpretation:

- `y_base` preserves robust global affinity estimation.
- `y_ref` learns a CoReMol-conditioned calibration signal.
- `y_hat` is the only prediction used for final evaluation.

## 8. Final Affinity Configuration

Configuration snapshot:

```text
configs/rpli_coremol_single_stage_dual_a080_b100_c020.md
```

Key settings:

```text
backbone: rpli
variant: coremol
hidden_channels: 64
context_channels: 384
dropout: 0.10
conv_dropout: 0.00
readout_mode: dual
final_blend_alpha: 0.8
base_aux_weight: 1.0
coremol_aux_weight: 0.2
cross_position: post
cross_update: readout
interface_gate_init: 1.0
cross_beta: 0.25
cross_tau: 0.75
cross_gate_init: 0.02
cross_gate_max: 0.20
```

Final checkpoint:

```text
results/coremol_net_affinity_gems/rpli_b6aepl_fold0_single_stage_dual_a080_b100_c020_coremol_fold0/best_model.pt
```

External metrics:

```text
CASF2016 RMSE: 1.2983
CASF2016_indep RMSE: 1.3558
CASF2016_indep Pearson: 0.8313
CASF2016_indep Spearman: 0.8276
CASF2016_indep R2: 0.6594
```

This is the preferred current single-model result.

## 9. MoleculeNet Adaptation Plan

The MoleculeNet version should not copy ligand-pocket assumptions. It should instantiate the same method idea on a single molecular graph.

### 9.1 Affinity Version

```text
nodes: ligand atoms + pocket residues/atoms
reference graph: ligand-pocket edges
reference flow: cross-interface flow
readout: affinity regression
```

### 9.2 Single-Molecule Version

```text
nodes: molecule atoms
edges: molecular bonds
reference graph: intra-molecular reference relations
reference flow: atom-atom / motif-aware intra-molecular flow
readout: graph-level classification or regression
```

### 9.3 Proposed Single-Molecule Model

Recommended class:

```text
coremol/models/rpli_molecule.py
```

Suggested structure:

```text
MolecularRPLI
  - atom feature transform
  - edge-aware RPLI message passing
  - graph context carrier
  - molecular reference graph constructor
  - CoReMol intra-reference adapter
  - graph-level dual-head calibrated readout
```

## 10. Single-Molecule Reference Graph Options

Implement in this order.

### Option A: Bond Reference Graph

Use existing molecular bonds and bond features.

This is the safest first implementation because MoleculeNet datasets already provide atom and bond graphs.

```text
C_ref^{mol}(i, j) = reference weight for bonded atom pair
```

### Option B: Shortest-Path Reference Graph

Add atom pairs within a small shortest-path radius, such as 2 or 3 hops.

```text
C_ref^{mol}(i, j) = exp(- shortest_path(i, j) / tau)
```

This gives CoReMol access to near-local chemistry beyond direct bonds.

### Option C: Ring / Functional-Group Reference Graph

If RDKit is available, add reference relations for:

```text
rings
aromatic systems
functional groups
scaffold-level motifs
```

This should be a second-stage improvement, not the first implementation.

### Option D: Learned Reference Adjacency

Only consider this after fixed references work. A learned reference graph is harder to explain and easier to overfit.

## 11. MoleculeNet Task Requirements

The single-molecule implementation must support:

- graph-level classification
- graph-level regression
- multi-task labels
- missing-label masks
- BCEWithLogitsLoss for classification
- MSE/RMSE/MAE for regression
- ROC-AUC reporting for classification datasets
- RMSE/MAE/R2 reporting for regression datasets

Recommended initial datasets:

```text
BBBP: binary classification smoke target
Tox21: multi-task classification with masks
ESOL: regression smoke target
FreeSolv/Lipo: additional regression checks
```

## 12. Files The Next Codex Should Create

Recommended new files:

```text
coremol/models/rpli_molecule.py
coremol/modules/molecular_reference_graph.py
scripts/train_coremol_moleculenet.py
scripts/eval_coremol_moleculenet.py
tests/test_rpli_molecule.py
tests/test_molecular_reference_graph.py
tests/test_moleculenet_training_cli.py
```

Do not break the affinity files while adding these.

## 13. Key Code To Reuse

The folder:

```text
RPLI-backbone/
```

contains a code snapshot and file map for the next Codex. Use it as a readable handoff package. The original source files remain in their normal repo locations.

## 14. Suggested First Milestone For The Next Codex

First milestone:

```text
Train MolecularRPLI+CoReMol on a tiny MoleculeNet subset and pass smoke tests.
```

Minimum acceptance:

- one command trains one model
- no protein-pocket dependency
- no `n_nodes=[total, ligand, protein]` dependency
- no `lig_emb` dependency
- model returns graph-level logits or regression values
- CoReMol diagnostics expose intra-molecular reference flow

Second milestone:

```text
Run BBBP and ESOL small experiments and confirm CoReMol adapter does not degrade the RPLI baseline.
```

## 15. What Not To Emphasize In The Paper

Do not foreground:

- the literal `0.8/0.2` coefficient
- failed external prediction blend attempts
- failed residual-only variants
- implementation-specific loss weights

These are implementation details and can be placed in hyperparameter tables or appendix.

Emphasize:

- shared carrier
- reference-conditioned adapter
- controlled calibration readout
- single-checkpoint training and inference
- transfer from ligand-pocket reference flow to intra-molecular reference flow
