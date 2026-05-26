# GEMS No-Embedding CleanSplit Protocol For Backbone Comparisons

Date: 2026-05-26

This folder records the exact no-embedding protocol used for the current CoReMol affinity experiments. It is intended as the handoff contract for another server that will run CheapNet, GIGN, EHIGN, and their `baseline + CoReMol` variants under the same setting.

## Core Requirement

All compared backbones must use the same GEMS CleanSplit no-embedding protocol:

- same training dataset,
- same fold construction,
- same label scaling,
- same evaluation datasets,
- same metrics,
- no ligand/protein language-model embeddings unless a row is explicitly marked as a separate embedding-based comparison.

The main purpose is to compare backbone behavior and CoReMol adapter improvement, not to compare different feature budgets.

## Dataset Protocol

Use the GEMS precomputed PyTorch datasets:

```text
train:          /home/legion-w5/LabData/Stone/GEMS_datasets/extracted/GEMS_pytorch_datasets/00AEPL_train_cleansplit.pt
casf2016:       /home/legion-w5/LabData/Stone/GEMS_datasets/extracted/GEMS_pytorch_datasets/00AEPL_casf2016.pt
casf2016_indep: /home/legion-w5/LabData/Stone/GEMS_datasets/extracted/GEMS_pytorch_datasets/00AEPL_casf2016_indep.pt
optional:       /home/legion-w5/LabData/Stone/GEMS_datasets/extracted/GEMS_pytorch_datasets/00AEPL_casf2013.pt
optional:       /home/legion-w5/LabData/Stone/GEMS_datasets/extracted/GEMS_pytorch_datasets/00AEPL_casf2013_indep.pt
```

`00AEPL_train_cleansplit.pt` contains 16,491 complexes.

Important: `00AEPL` still stores `lig_emb` with shape `(1, 384)` and metadata `ligand_embeddings=['ChemBERTa_77M']`. For this protocol, models must ignore that field. In the local CoReMol scripts this is enforced with:

```bash
--ignore_lig_emb
```

For CheapNet, GIGN, and EHIGN implementations on another server, apply the equivalent rule:

- do not pass `data.lig_emb` to the model,
- do not initialize a global feature from ChemBERTa,
- do not add protein language-model embeddings,
- do not add external pretrained ligand/protein embeddings,
- keep only the graph/geometry features available in the `00AEPL` graph object.

## Split And Fold Protocol

Use fold0 from the local CleanSplit fold construction:

```text
n_folds: 5
fold: 0
seed: 0
fold construction: StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
stratification labels: rounded scaled affinity labels from the CleanSplit train set
```

Evaluation must be done on the unchanged CASF files:

```text
00AEPL_casf2016.pt
00AEPL_casf2016_indep.pt
```

Do not rebuild or refilter CASF test sets for the main table.

## Label And Metric Protocol

GEMS labels are scaled to `[0, 1]`. Train on the scaled labels. For reporting:

```text
pK_value = scaled_value * 16.0
```

Report at least RMSE in pK, MAE in pK, Pearson, Spearman, Kendall tau, and R2. The main comparison should prioritize RMSE and Pearson on CASF2016 and CASF2016_indep.

## Official GEMS No-Embedding Reference

The protocol anchor is GEMS18e on `00AEPL`, which ignores `lig_emb`.

| Method | Checkpoints | Embedding used | CASF2016 RMSE | CASF2016 Pearson | CASF2016_indep RMSE | CASF2016_indep Pearson |
|---|---:|---|---:|---:|---:|---:|
| GEMS18e official fold0 | 1 | none | 1.4474 | 0.7544 | 1.5500 | 0.7640 |
| GEMS18e official ensemble | 5 | none | 1.3497 | 0.7966 | 1.4440 | 0.8104 |

Use these as reference rows, not as trainable local baselines.

## Local CoReMol Reference Rows

| Method | Checkpoints | Embedding used | CASF2016 RMSE | CASF2016 Pearson | CASF2016 Spearman | CASF2016_indep RMSE | CASF2016_indep Pearson | CASF2016_indep Spearman |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| RPLI baseline | 1 | none, `--ignore_lig_emb` | 1.4677 | 0.7462 | 0.7426 | 1.6109 | 0.7251 | 0.7268 |
| RPLI+CoReMol dual-head | 1 | none, `--ignore_lig_emb` | 1.3621 | 0.8027 | 0.7931 | 1.4359 | 0.8259 | 0.8259 |

Interpretation:

- The matched CoReMol ablation is `RPLI baseline` versus `RPLI+CoReMol dual-head`.
- RPLI+CoReMol improves CASF2016 RMSE by `0.1056`.
- RPLI+CoReMol improves CASF2016_indep RMSE by `0.1750`.
- The single-checkpoint RPLI+CoReMol row is close to the official GEMS18e 5-checkpoint ensemble.

## Main RPLI+CoReMol Configuration

Checkpoint on the original local server:

```text
results/gems_no_embedding_affinity/noemb_rpli_dual_a080_b100_c020_coremol_fold0/best_model.pt
```

Key settings:

```text
dataset_path: /home/legion-w5/LabData/Stone/GEMS_datasets/extracted/GEMS_pytorch_datasets/00AEPL_train_cleansplit.pt
variant: coremol
backbone: rpli
fold: 0
n_folds: 5
seed: 0
epochs: 180
patience: 30
batch_size: 256
lr: 0.001
weight_decay: 0.00001
loss_func: rmse
hidden_channels: 64
context_channels: 384
dropout: 0.10
conv_dropout: 0.00
cross_position: post
cross_update: readout
interface_gate_init: 1.0
readout_mode: dual
final_blend_alpha: 0.8
base_aux_weight: 1.0
coremol_aux_weight: 0.2
selection_metric: rmse
ignore_lig_emb: true
amp: true
ligand_global_dim in checkpoint: 0
best_epoch: 50
best_valid_rmse: 1.3069
```

## Commands Used Locally

RPLI baseline:

```bash
PYTHONPATH=.:/home/legion-w5/LabData/Stone/GEMS conda run -n GNRF python scripts/train_coremol_affinity.py \
  --dataset_path /home/legion-w5/LabData/Stone/GEMS_datasets/extracted/GEMS_pytorch_datasets/00AEPL_train_cleansplit.pt \
  --run_name noemb_rpli_base_rmse \
  --variant base \
  --backbone rpli \
  --fold 0 \
  --n_folds 5 \
  --seed 0 \
  --epochs 160 \
  --patience 25 \
  --batch_size 256 \
  --lr 0.001 \
  --weight_decay 0.00001 \
  --loss_func rmse \
  --hidden_channels 64 \
  --context_channels 384 \
  --dropout 0.10 \
  --conv_dropout 0.00 \
  --ignore_lig_emb \
  --amp \
  --save_dir results/gems_no_embedding_affinity
```

RPLI+CoReMol:

```bash
PYTHONPATH=.:/home/legion-w5/LabData/Stone/GEMS conda run -n GNRF python scripts/train_coremol_affinity.py \
  --dataset_path /home/legion-w5/LabData/Stone/GEMS_datasets/extracted/GEMS_pytorch_datasets/00AEPL_train_cleansplit.pt \
  --run_name noemb_rpli_dual_a080_b100_c020 \
  --variant coremol \
  --backbone rpli \
  --fold 0 \
  --n_folds 5 \
  --seed 0 \
  --epochs 180 \
  --patience 30 \
  --batch_size 256 \
  --lr 0.001 \
  --weight_decay 0.00001 \
  --loss_func rmse \
  --hidden_channels 64 \
  --context_channels 384 \
  --dropout 0.10 \
  --conv_dropout 0.00 \
  --cross_position post \
  --cross_update readout \
  --interface_gate_init 1.0 \
  --readout_mode dual \
  --final_blend_alpha 0.8 \
  --base_aux_weight 1.0 \
  --coremol_aux_weight 0.2 \
  --selection_metric rmse \
  --ignore_lig_emb \
  --amp \
  --save_dir results/gems_no_embedding_affinity
```

Evaluation:

```bash
PYTHONPATH=.:/home/legion-w5/LabData/Stone/GEMS conda run -n GNRF python scripts/eval_coremol_affinity.py \
  --dataset_path /home/legion-w5/LabData/Stone/GEMS_datasets/extracted/GEMS_pytorch_datasets/00AEPL_casf2016.pt \
  --checkpoint results/gems_no_embedding_affinity/noemb_rpli_dual_a080_b100_c020_coremol_fold0/best_model.pt \
  --save_dir results/gems_no_embedding_affinity/noemb_rpli_dual_a080_b100_c020_coremol_fold0/eval_casf2016 \
  --batch_size 256
```

## Instructions For CheapNet, GIGN, And EHIGN

For each backbone, run two paired rows:

1. `Backbone baseline`
2. `Backbone + CoReMol`

The paired rows must share the same train file, fold0 split, seed, label scaling, evaluation files, no-embedding rule, and early-stopping metric where feasible.

Recommended output naming:

```text
results/gems_no_embedding_affinity/<backbone>_noemb_base_fold0/
results/gems_no_embedding_affinity/<backbone>_noemb_coremol_fold0/
```

Recommended main table columns:

```text
method
backbone
variant
embedding_used
train_file
fold
seed
checkpoint_count
casf2016_rmse
casf2016_pearson
casf2016_spearman
casf2016_indep_rmse
casf2016_indep_pearson
casf2016_indep_spearman
notes
```

Target behavior:

- The baseline should be reported honestly even if it is weaker than the paper number.
- `Backbone + CoReMol` should improve over the matched local backbone baseline.
- If a backbone cannot fairly consume GEMS `00AEPL` graph objects without changing features, document that incompatibility rather than mixing protocols.
- If an official paper baseline used different features, report it in a separate paper-value table, not in the local no-embedding table.

## Feature Fairness Checklist

Before reporting a CheapNet/GIGN/EHIGN result as directly comparable, verify:

- [ ] `00AEPL_train_cleansplit.pt` is used.
- [ ] `00AEPL_casf2016.pt` and `00AEPL_casf2016_indep.pt` are used unchanged.
- [ ] `lig_emb` is ignored.
- [ ] no protein embedding is added.
- [ ] no ligand language-model embedding is added.
- [ ] labels are trained in scaled `[0, 1]` units.
- [ ] reported RMSE/MAE are in pK units after multiplying by 16.
- [ ] fold0 split follows `StratifiedKFold(n_splits=5, random_state=0, shuffle=True)`.
- [ ] baseline and `+CoReMol` use the same train/valid split and evaluation code.

## Existing Detailed Report

The fuller local report is:

```text
reports/gems_no_embedding_affinity_baseline_comparison.md
```

The compact YAML companion file in this folder is:

```text
GEMS-noembedding-protocol/noembedding_experiment_config.yaml
```
