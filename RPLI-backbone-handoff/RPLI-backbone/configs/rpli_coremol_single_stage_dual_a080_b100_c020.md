# RPLI CoReMol Single-Stage Dual-Head Calibrator A080 B100 C020

This is the saved configuration snapshot for the current best one-command, one-model final candidate.

## Status

This run is the preferred final single-model result so far.

```text
CASF2016 RMSE: 1.2983
CASF2016_indep RMSE: 1.3558
CASF2016_indep Pearson: 0.8313
CASF2016_indep R2: 0.6594
```

It is a single checkpoint, trained from one command, with no external teacher checkpoint and no inference-time model ensemble.

## Code Entry Points

- Model: `coremol/models/rpli_affinity.py`
- Training CLI: `scripts/train_coremol_affinity.py`
- Evaluation CLI: `scripts/eval_coremol_affinity.py`
- Plan: `docs/superpowers/plans/2026-05-22-rpli-coremol-single-stage-residual-calibrator.md`

## Architecture Settings

```text
backbone: rpli
variant: coremol
hidden_channels: 64
context_channels: 384
dropout: 0.10
conv_dropout: 0.00
readout_mode: dual
final_blend_alpha: 0.8
cross_position: post
cross_update: readout
interface_gate_init: 1.0
cross_beta: 0.25
cross_tau: 0.75
cross_gate_init: 0.02
cross_gate_max: 0.20
```

## Training Objective

The model has one shared RPLI+CoReMol carrier and two internal prediction heads:

- `base_pred`: RMSE-oriented global affinity head
- `coremol_pred`: MSE-oriented CoReMol-conditioned affinity head
- final output: `0.8 * base_pred + 0.2 * coremol_pred`

Training loss:

```text
RMSE(final_pred, y) + 1.0 * RMSE(base_pred, y) + 0.2 * MSE(coremol_pred, y)
```

Checkpoint selection:

```text
selection_metric: rmse
```

## Unified Training Command

```bash
source /home/legion-w5/anaconda3/etc/profile.d/conda.sh && conda activate GNRF && \
PYTHONPATH=/home/legion-w5/LabData/Stone/worktrees/coremol-net-affinity:/home/legion-w5/LabData/Stone/GEMS \
time python scripts/train_coremol_affinity.py \
  --dataset_path /home/legion-w5/LabData/Stone/GEMS_datasets/extracted/GEMS_pytorch_datasets/B6AEPL_train_cleansplit.pt \
  --run_name rpli_b6aepl_fold0_single_stage_dual_a080_b100_c020 \
  --backbone rpli \
  --variant coremol \
  --fold 0 \
  --n_folds 5 \
  --epochs 160 \
  --patience 25 \
  --batch_size 256 \
  --hidden_channels 64 \
  --context_channels 384 \
  --dropout 0.10 \
  --conv_dropout 0.00 \
  --lr 0.001 \
  --weight_decay 0.00001 \
  --loss_func rmse \
  --aux_mse_weight 0.0 \
  --base_aux_weight 1.0 \
  --coremol_aux_weight 0.2 \
  --selection_metric rmse \
  --cross_position post \
  --cross_update readout \
  --readout_mode dual \
  --final_blend_alpha 0.8 \
  --interface_gate_init 1.0 \
  --cross_beta 0.25 \
  --cross_tau 0.75 \
  --cross_gate_init 0.02 \
  --cross_gate_max 0.20 \
  --amp \
  --save_dir results/coremol_net_affinity_gems
```

## Output Paths

```text
checkpoint: results/coremol_net_affinity_gems/rpli_b6aepl_fold0_single_stage_dual_a080_b100_c020_coremol_fold0/best_model.pt
metrics: results/coremol_net_affinity_gems/rpli_b6aepl_fold0_single_stage_dual_a080_b100_c020_coremol_fold0/metrics.csv
CASF2016 eval: results/coremol_net_affinity_gems/eval_single_stage_dual_a080_b100_c020_casf2016/eval_metrics.json
CASF2016_indep eval: results/coremol_net_affinity_gems/eval_single_stage_dual_a080_b100_c020_casf2016_indep/eval_metrics.json
```
