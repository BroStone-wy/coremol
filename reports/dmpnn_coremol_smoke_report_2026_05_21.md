# D-MPNN CoReMol Smoke Report

Date: 2026-05-21

Scope: first D-MPNN backbone integration smoke test. This is not a final paper-facing result. The goal was to verify that D-MPNN baseline, D-MPNN+CoReMol, training, checkpointing, mechanism diagnostics, and true Full TCM post-processing all run through the existing CoReMol pipeline.

## Implementation Status

Implemented:

- `coremol/models/dmpnn_coremol.py`
- `--backbone dmpnn` in `scripts/run_stage1_gate.py`
- `--backbone dmpnn` in `scripts/compute_tcm_variants.py`
- D-MPNN tests in `tests/test_dmpnn_coremol.py`

Design:

- D-MPNN uses directed bond-centered hidden states.
- Immediate reverse-edge echo is removed in the directed message update.
- Directed bond messages are aggregated into atom states.
- CoReMol is applied to atom states, keeping the residual adapter backbone-agnostic.

## Smoke Command

```bash
PYTHONPATH=. conda run --no-capture-output -n GNRF python scripts/run_stage1_gate.py \
  --backbone dmpnn \
  --datasets BBBP \
  --seeds 0 \
  --epochs 30 \
  --batch_size 64 \
  --hidden_channels 128 \
  --num_layers 3 \
  --num_timesteps 1 \
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
  --tcm_graphs 48 \
  --tcm_k 10 \
  --results_name dmpnn_bbbp_smoke_seed0
```

Full TCM command:

```bash
PYTHONPATH=. conda run --no-capture-output -n GNRF python scripts/compute_tcm_variants.py \
  --run_dir results/dmpnn_bbbp_smoke_seed0 \
  --datasets BBBP \
  --seeds 0 \
  --backbone dmpnn \
  --hidden_channels 128 \
  --num_layers 3 \
  --num_timesteps 1 \
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

## Smoke Result

| Dataset | Seed | Baseline AUC | CoReMol AUC | Delta AUC | Full TCM Delta | TCM@10 Delta | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| BBBP | 0 | 0.691300 | 0.689951 | -0.001349 | +0.002507 | -0.000460 | diagnostic failure |

Interpretation:

- The D-MPNN implementation and CoReMol integration are runnable.
- Full TCM improved on the smoke seed, so the residual adapter is producing globally positive communication calibration.
- The main task metric decreased slightly, so this run is not usable as a final CoReMol claim.
- `TCM@10` is negative, which is consistent with prior cases where global Full TCM and local top-k diagnostics can disagree.

## Failure Analysis

Most likely cause:

- Baseline alignment and residual strength were not tuned for D-MPNN. The smoke used a short 30-epoch run and a Graphformer-like CoReMol starting profile.

Targeted next adjustment:

- First align D-MPNN baseline scale with a 3-seed baseline-only sweep before tuning CoReMol.
- Try `hidden_channels=128/256`, `num_layers=3/4/5`, `dropout=0.0/0.1/0.2`, `readout=mean/mean_max`, and `lr=0.0005/0.001`.
- After baseline is stable, tune CoReMol one variable at a time: `value` vs `delta`, `beta`, `tau`, `d_max`, gate init, and backbone LR scale.

## Verification

Unit tests:

```text
tests/test_dmpnn_coremol.py: 3 passed
full test suite: 67 passed
```

