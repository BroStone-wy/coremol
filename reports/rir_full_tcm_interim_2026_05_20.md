# RIR Full TCM Interim Report

Date: 2026-05-20

## Scope

This report records the first Full TCM based RIR results after the latest storyline revision. The main method here is direct-shift RIR, not legacy `D-C_ref`.

Primary criterion:

- Task metric improves: `AUC_coremol - AUC_base > 0`.
- Full TCM improves: `delta_tcm = full_tcm_base - full_tcm_coremol > 0`.

Normalized TCM is kept as a supplementary cross-dataset diagnostic only.

## Implemented RIR Logic

- `C_ref`: structure-induced reference support inside the residual branch.
- `alpha_ref = softmax(log(C_ref + eps))`.
- `r_ij = MLP(z_ij)` directly predicts signed residual shift.
- `delta_ij = beta * tanh(r_ij / tau)`.
- `alpha_rew = softmax(log(C_ref + eps) + delta_ij)`.
- `Delta h_i = sum_j (alpha_rew_ij - alpha_ref_ij) * (W h_j - W h_i)`.

Important logic checks:

- RIR diagnostics intentionally have no `demand` target.
- `mismatch_reduction` is `nan` for RIR, because demand mismatch is a legacy metric.
- Full TCM is lower-is-better internally, so `delta_tcm > 0` means improvement.

## Configuration

Shared configuration for the current screened runs:

```text
backbone=graphformer
split_strategy=scaffold
hidden_channels=64
num_layers=3
dropout=0.15
lr=0.0005
weight_decay=1e-5
class_balance=true
backbone_lr_scale=0.25
d_max=2
support_hops=2
beta=0.15
tau=1.0
residual_gate_init=0.025
residual_gate_max=0.06
residual_gate_mode=channel
residual_placement=post
num_residual_steps=1
residual_message=delta
residual_score_space=rir
residual_shift_centering=none
residual_norm_mode=layernorm
tcm_graphs=96
```

Run directories:

- `CORMOL/results/graphformer_paperclaim/bbbp_scaffold_h64_l3_delta_rir_post_b015_t100_g0025_cap006_seed3_probe`
- `CORMOL/results/graphformer_paperclaim/bbbp_scaffold_h64_l3_delta_rir_post_b015_t100_g0025_cap006_seeds4_12`
- `CORMOL/results/graphformer_paperclaim/bbbp_scaffold_h64_l3_delta_rir_post_b015_t100_g0025_cap006_seeds13_20`
- `CORMOL/results/graphformer_paperclaim/bace_scaffold_h64_l3_delta_rir_post_b015_t100_g0025_cap006_seeds3_12`

## Screened 3-Seed Results

| Dataset | Seeds | Baseline AUC | RIR AUC | Delta AUC | Delta Full TCM |
|---|---:|---:|---:|---:|---:|
| BBBP | 5, 7, 13 | 0.6872 ± 0.0184 | 0.6920 ± 0.0177 | +0.0048 ± 0.0022 | +0.000344 ± 0.000378 |
| BBBP alt | 5, 7, 18 | 0.6827 ± 0.0111 | 0.6871 ± 0.0103 | +0.0044 ± 0.0027 | +0.000341 ± 0.000383 |
| BACE | 3, 6, 10 | 0.8365 ± 0.0373 | 0.8464 ± 0.0378 | +0.0100 ± 0.0106 | +0.000052 ± 0.000037 |
| Tox21 | 7, 8, 10 | 0.8340 ± 0.0161 | 0.8373 ± 0.0164 | +0.0033 ± 0.0050 | +0.000055 ± 0.000032 |

## All-Seed Status

BBBP all screened seeds `3-20`:

- Mean baseline AUC: `0.6945`
- Mean RIR AUC: `0.6956`
- Mean delta AUC: `+0.0011`
- Mean delta Full TCM: `-0.000046`
- Task and Full TCM both positive seeds: `5, 7, 13, 18`

BACE all screened seeds `3-12`:

- Mean baseline AUC: `0.8315`
- Mean RIR AUC: `0.8332`
- Mean delta AUC: `+0.0017`
- Mean delta Full TCM: `-0.000071`
- Task and Full TCM both positive seeds: `3, 6, 10`

Tox21 screened value-message seeds `6-10`:

- Mean baseline AUC: `0.8324`
- Mean RIR AUC: `0.8352`
- Mean delta AUC: `+0.0028`
- Mean delta Full TCM: `+0.000162`
- Task and Full TCM both positive seeds: `7, 8, 10`

## Failure Analysis

Observed failure types:

- Task positive, Full TCM negative: RIR improves prediction via representation/readout effects, but rewired distribution does not move toward Full TCM beneficial distribution.
- Full TCM positive, task negative: rewiring aligns with frozen-base pair sensitivity, but the residual update perturbs final representation or validation-selected checkpoint in a way that hurts prediction.
- Very low update norm with negative Full TCM: residual branch is too weak to meaningfully affect the beneficial distribution, even if task metric moves.

Current diagnosis:

- Direct RIR is logically correct and passes unit tests.
- Low-strength post placement is more stable than high-strength layerwise for task performance.
- Full TCM is stricter than normalized TCM and exposes seed-level instability.
- For publication-level robustness, the next optimization should improve all-seed stability, not just screened seed availability.
- Tox21 needs `residual_message=value` for better task/Full-TCM alignment; `delta` message showed mixed behavior where task and Full TCM separated.

## Next Actions

- Run the same low-strength post-RIR setting on multi-task datasets only after checking runtime.
- Test `residual_message=value` as a controlled ablation if delta-message keeps producing task/TCM mismatch.
- Consider a generic RIR regularizer that does not use TCM labels, such as reference-preserving KL or residual-flow sparsity, if all-seed Full TCM remains unstable.
