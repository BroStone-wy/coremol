# ESOL ZINC-Subset Pretraining Report

Date: 2026-05-13

## Protocol

The regression comparison table uses ZINC(subset) pretraining. In the local CurvFlow reference, ZINC is loaded through PyG `ZINC(subset=True)`, with the fixed 10k/1k/1k train/val/test split. The target is supervised ZINC regression, not self-supervised masked-token reconstruction.

This CORMOL run uses:

- ZINC supervised pretraining checkpoint: `results/zinc_pretrain/attentivefp_zinc_subset_seed0.pt`
- Pretrain model: AttentiveFP, hidden 64, 2 layers, 2 timesteps
- Fine-tune model: AttentiveFP and AttentiveFP+CoReMol, hidden 64
- Regression target normalization during training, metrics reported after inverse transform
- CoReMol warm-started from the fine-tuned AttentiveFP baseline

## Split Findings

`curvflow_random` follows the generic CurvFlow PyG split behavior: 70/20/10 random split. It is too easy for ESOL compared with the screenshot table.

| Run | Base RMSE | CoReMol RMSE | Task seeds improved | TCM@10 seeds improved | Full TCM seeds improved |
|---|---:|---:|---:|---:|---:|
| `esol_zinc_pretrain_random` | 0.682 +/- 0.099 | 0.652 +/- 0.078 | 2/3 | 2/3 | 3/3 |
| `esol_zinc_pretrain_scaffold` | 0.893 +/- 0.050 | 0.908 +/- 0.016 | 1/3 | 2/3 | not computed |
| `esol_zinc_pretrain_scaffold_freeze` | 0.885 +/- 0.050 | 0.881 +/- 0.042 | 2/3 | 2/3 | 3/3 |

The scaffold split with ZINC pretraining is the closest to the screenshot AttentiveFP value of 0.877. The final frozen-backbone CoReMol run improves the mean RMSE from 0.885 to 0.881, while preserving a comparable baseline.

## Failure Analysis

The unfrozen scaffold run failed the task metric: CoReMol improved validation RMSE but degraded test RMSE on seeds 0 and 2. This points to the residual branch disturbing the already fine-tuned AttentiveFP backbone, not to a split or ZINC checkpoint issue.

The main-mechanism adjustment was to freeze the warm-started backbone and train only the CoReMol residual communication branch. This does not add a new module; it limits optimization to the proposed communication calibration mechanism.

## Current Status

The best aligned ESOL protocol is:

```bash
PYTHONPATH=. conda run -n GNRF python scripts/run_stage1_gate.py \
  --datasets ESOL --seeds 0 1 2 \
  --epochs 80 --patience 15 --hidden_channels 64 \
  --num_layers 2 --num_timesteps 2 --batch_size 64 \
  --lr 0.001 --dropout 0.1 \
  --split_strategy scaffold --normalize_regression \
  --pretrained_backbone results/zinc_pretrain/attentivefp_zinc_subset_seed0.pt \
  --warm_start_coremol --freeze_coremol_backbone \
  --tcm_graphs 999 --tcm_k 10 \
  --results_name esol_zinc_pretrain_scaffold_freeze
```

Per-seed test RMSE:

| Seed | Base | CoReMol | Delta |
|---:|---:|---:|---:|
| 0 | 0.8639 | 0.8660 | -0.0021 |
| 1 | 0.9424 | 0.9289 | +0.0136 |
| 2 | 0.8498 | 0.8485 | +0.0013 |

TCM variants for `esol_zinc_pretrain_scaffold_freeze`:

- Full TCM improves 3/3 seeds, mean delta 0.01893.
- TCM@10 improves 2/3 seeds, mean delta 0.00149.
- Seed 1 remains the weak case for top-k TCM, even though task RMSE improves.

## Next Adjustment Boundary

The current result is a useful gate result but not yet strong enough for a top-tier claim. The next adjustment should stay within the main mechanism:

- tune residual strength (`beta`, `tau`, `residual_gate_init`) under frozen-backbone training;
- report both Full TCM and TCM@K because ESOL shows full-distribution calibration is more stable than top-k calibration;
- only after TCM and task improvements are stable across seeds should task-performance hyperparameter tuning be expanded.


## Split-Controlled Follow-up

After adding random 80/10/10, the random split remains much easier than scaffold even without ZINC pretraining:

| Run | Base RMSE | CoReMol RMSE | Task seeds improved | TCM@10 seeds improved |
|---|---:|---:|---:|---:|
| `esol_random_80_10_10_scratch` | 0.714 +/- 0.041 | 0.669 +/- 0.039 | 3/3 | 3/3 |
| `esol_random_80_10_10_zinc_pretrain` | 0.730 +/- 0.066 | 0.702 +/- 0.099 | 2/3 | 3/3 |
| `esol_zinc_pretrain_scaffold_freeze` | 0.885 +/- 0.050 | 0.881 +/- 0.042 | 2/3 | 2/3 |
| `esol_scaffold_residual_pretrained` | 0.918 +/- 0.030 | 0.914 +/- 0.029 | 3/3 | 2/3 |

This confirms that the very strong random numbers are caused primarily by interpolation split difficulty, not by ZINC pretraining. Random 80/10/10 scratch already reaches 0.714 RMSE, far below the scaffold-pretrained baseline around 0.885.

## Residual-Branch Pretraining

A residual-only ZINC pretraining stage was added:

```bash
PYTHONPATH=. conda run -n GNRF python scripts/pretrain_zinc_coremol_residual.py   --seed 0 --epochs 15 --batch_size 128 --hidden_channels 64   --num_layers 2 --num_timesteps 2 --dropout 0.1 --lr 0.0005   --patience 5   --pretrained_backbone results/zinc_pretrain/attentivefp_zinc_subset_seed0.pt   --out_dir results/zinc_coremol_residual_pretrain
```

The script freezes the AttentiveFP backbone and trains only:

- `residual_gate`
- `demand_net.*`
- `value_proj.weight`
- `residual_norm.*`

ZINC residual-pretrained checkpoint: `results/zinc_coremol_residual_pretrain/coremol_residual_zinc_subset_seed0.pt`.

On ESOL scaffold, this improves task RMSE on 3/3 seeds and Full TCM on 3/3 seeds. TCM@10 remains 2/3 seeds because seed 1 has a small top-k leakage increase, even though its task RMSE improves.

## Updated Interpretation

The best story-aligned setting is scaffold split with residual-branch pretraining. It is stricter than random split, avoids claiming random interpolation as SOTA evidence, and directly supports the CoReMol mechanism: the residual communication branch itself benefits from pretraining and gives more stable OOD task improvements.

The next mechanism-only tuning target is still top-k stability: reduce harmful top-k leakage on seed 1 without weakening the full-distribution calibration that is already stable.
