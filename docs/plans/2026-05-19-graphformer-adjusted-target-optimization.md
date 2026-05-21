# Graphformer Adjusted Target Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:systematic-debugging before changing experiment settings, and use superpowers:verification-before-completion before reporting final success.

**Goal:** Optimize Graphformer-CoReMol classification results under the user's adjusted target: reproducible task AUC improvement plus Full TCM improvement, without chasing suspicious Graph Curvature Flow table values for SIDER and ToxCast.

**Architecture:** Keep CoReMol as a backbone-agnostic residual atom-pair communication adapter. Dataset-specific tuning may change Graphformer profile and CoReMol hyperparameters, but the residual calculation remains `value` or `delta` message over generic node states and graph support.

**Tech Stack:** Python, PyTorch, PyG MoleculeNet datasets, `conda` environment `GNRF`, scripts under `CORMOL/scripts`, results under `CORMOL/results`.

---

## Current Target Interpretation

- BBBP: use standard scaffold Graphformer profile scale around CoReMol AUC `0.6865`; improve task mean and Full TCM. Current best candidates still only have Full TCM `2/3`.
- ClinTox: profile scale around baseline AUC `0.9349` is acceptable; current conservative CoReMol run improves mean task AUC and Full TCM `3/3`, but task wins are `2/3`.
- SIDER: do not force paper Graphformer `0.823`; reproducible scale around baseline `0.6595` and CoReMol `0.6667` is acceptable for report replacement, with TCM status still to be verified before final claim.
- ToxCast: do not force paper Graphformer `0.816`; reproducible task-improving scale `0.7664 -> 0.7698` is acceptable as diagnostic/report evidence, but current Full TCM is negative for that profile.

## Task 1: Verify ClinTox Conservative Candidate

**Files:**
- Read: `CORMOL/results/graphformer_profile_sweep/CLINTOX/cat_edge_meanmax/coremol_delta_conservative_g0005_3seeds/raw_metrics.csv`
- Read: `CORMOL/results/graphformer_profile_sweep/CLINTOX/cat_edge_meanmax/coremol_delta_conservative_g0005_3seeds/mechanism_metrics.csv`
- Update report after verification: `CORMOL/results/graphformer_profile_sweep/graphformer_profile_ablation_interim_report.md`

- [x] **Step 1: Run three-seed conservative ClinTox delta experiment**

```bash
PYTHONPATH=CORMOL conda run --no-capture-output -n GNRF python CORMOL/scripts/run_stage1_gate.py \
  --backbone graphformer \
  --datasets CLINTOX \
  --seeds 0 1 2 \
  --epochs 60 \
  --batch_size 64 \
  --hidden_channels 128 \
  --num_layers 6 \
  --num_timesteps 1 \
  --dropout 0.10 \
  --lr 0.0002 \
  --weight_decay 1e-5 \
  --patience 20 \
  --split_strategy random \
  --random_train_fraction 0.8 \
  --random_valid_fraction 0.1 \
  --d_max 3 \
  --support_hops 3 \
  --beta 0.05 \
  --tau 1.2 \
  --residual_gate_init 0.001 \
  --residual_gate_max 0.005 \
  --residual_gate_mode scalar \
  --residual_placement post \
  --num_residual_steps 1 \
  --residual_message delta \
  --residual_norm_mode layernorm \
  --max_grad_norm 5 \
  --tcm_graphs 96 \
  --tcm_k 10 \
  --variants base coremol \
  --fixed_base_dir CORMOL/results/graphformer_profile_sweep/CLINTOX/cat_edge_meanmax/conservative_baseline_checkpoints \
  --warm_start_coremol \
  --backbone_lr_scale 0.05 \
  --graphformer_feature_encoder categorical \
  --graphformer_readout mean_max \
  --graphformer_use_edge_bias \
  --results_name graphformer_profile_sweep/CLINTOX/cat_edge_meanmax/coremol_delta_conservative_g0005_3seeds
```

- [x] **Step 2: Verify current result**

Expected: CoReMol mean task AUC above baseline mean, TCM@10 positive on all three seeds, then verify Full TCM with `compute_tcm_variants.py`.

Observed from `run_stage1_gate.py`: base mean `0.918548`, CoReMol mean `0.919304`, task wins `2/3`; TCM@10 mean `0.00010065`, TCM@10 wins `3/3`.

Observed from `compute_tcm_variants.py`: Full TCM wins `3/3` with deltas `0.001165`, `0.001311`, `0.000752`; TCM@5/10/20 also wins `3/3`.

## Task 2: Run BBBP Conservative Delta Check

**Hypothesis:** The current BBBP conservative `value` run uses a very small gate and improves task mean, but seed0 Full TCM is slightly negative. Switching only the residual message from `value` to `delta` may align the TCM definition with relative communication correction while keeping task AUC near the accepted scale.

**Files:**
- Read fixed-base checkpoints: `CORMOL/results/graphformer_classification_sweep/bbbp_scaffold_h64_l4_edgeaware_value_scalar/checkpoints`
- Create results: `CORMOL/results/graphformer_classification_sweep/bbbp_scaffold_h64_l4_fixedbase_conservative_delta_g0005`

- [ ] **Step 1: Run one-variable BBBP delta experiment**

```bash
PYTHONPATH=CORMOL conda run --no-capture-output -n GNRF python CORMOL/scripts/run_stage1_gate.py \
  --backbone graphformer \
  --datasets BBBP \
  --seeds 0 1 2 \
  --epochs 60 \
  --batch_size 128 \
  --hidden_channels 64 \
  --num_layers 4 \
  --num_timesteps 1 \
  --dropout 0.15 \
  --lr 0.0003 \
  --weight_decay 1e-5 \
  --patience 18 \
  --split_strategy scaffold \
  --d_max 2 \
  --support_hops 2 \
  --beta 0.05 \
  --tau 1.2 \
  --residual_gate_init 0.001 \
  --residual_gate_max 0.005 \
  --residual_gate_mode scalar \
  --residual_placement post \
  --num_residual_steps 1 \
  --residual_message delta \
  --residual_norm_mode layernorm \
  --max_grad_norm 5 \
  --class_balance \
  --pos_weight_cap 20 \
  --tcm_graphs 96 \
  --tcm_k 10 \
  --variants base coremol \
  --fixed_base_dir CORMOL/results/graphformer_classification_sweep/bbbp_scaffold_h64_l4_edgeaware_value_scalar/checkpoints \
  --warm_start_coremol \
  --backbone_lr_scale 0.05 \
  --results_name graphformer_classification_sweep/bbbp_scaffold_h64_l4_fixedbase_conservative_delta_g0005
```

- [x] **Step 2: Compare with BBBP conservative value**

Expected: task mean remains above baseline mean; Full TCM improves to at least `2/3`, preferably `3/3`, without lowering CoReMol mean below `0.6837`.

Observed: `delta` is mathematically equivalent to `value` in the current strict softmax formulation because `sum_j (alpha_cal - alpha_base) = 0` for each source atom, so the `-V(h_i)` term cancels. The `delta` run therefore reproduced the conservative `value` result: base mean `0.682693`, CoReMol mean `0.683752`, task wins `2/3`, TCM@10 wins `2/3`.

Observed from `compute_tcm_variants.py` on the conservative `value` result: Full TCM wins `3/3` with deltas `0.000344`, `0.000348`, `0.000384`; TCM@10 is near-zero and mixed, so it should be reported as a boundary/top-k analysis rather than a failure of the Full TCM story.

## Task 3: Decide Next Single-Variable Adjustment

**Decision rules:**
- If BBBP `delta` improves task mean and TCM wins `3/3`, keep it as the current BBBP candidate.
- If BBBP `delta` improves TCM but hurts task mean, run the same command with `--residual_gate_max 0.003`.
- If BBBP `delta` improves task but not TCM, run the same command with `--beta 0.10` and unchanged gate cap.
- If both fail, return to Phase 1 debugging by comparing per-seed gate value, update norm, and support-hop distribution before any further tuning.

Current status: lowering `beta` on BBBP seed0 reduced the negative TCM@10 magnitude from `-1.46e-5` to `-2.38e-7`, but did not robustly flip the sign. Because Full TCM is already `3/3`, stop lowering `beta`; further tuning should only be done if TCM@10 is elevated to a final hard requirement.

## Task 4: ToxCast TCM/Task Tradeoff Diagnosis

**Hypothesis:** ToxCast task-improving `cat_edge_meanmax` profile has negative Full TCM because the residual gate is too large (`~0.015`) and calibrates task logits while over-correcting communication mismatch. The graph-token profile has positive Full TCM but task drops, suggesting readout path mismatch rather than a CoReMol definition failure.

**Next command only after BBBP is evaluated:**

```bash
PYTHONPATH=CORMOL conda run --no-capture-output -n GNRF python CORMOL/scripts/run_stage1_gate.py \
  --backbone graphformer \
  --datasets TOXCAST \
  --seeds 0 \
  --epochs 60 \
  --batch_size 64 \
  --hidden_channels 128 \
  --num_layers 6 \
  --num_timesteps 1 \
  --dropout 0.10 \
  --lr 0.0002 \
  --weight_decay 1e-5 \
  --patience 20 \
  --split_strategy random \
  --random_train_fraction 0.8 \
  --random_valid_fraction 0.1 \
  --d_max 3 \
  --support_hops 3 \
  --beta 0.05 \
  --tau 1.2 \
  --residual_gate_init 0.001 \
  --residual_gate_max 0.005 \
  --residual_gate_mode scalar \
  --residual_placement post \
  --num_residual_steps 1 \
  --residual_message value \
  --residual_norm_mode layernorm \
  --max_grad_norm 5 \
  --tcm_graphs 96 \
  --tcm_k 10 \
  --variants base coremol \
  --warm_start_coremol \
  --backbone_lr_scale 0.05 \
  --graphformer_feature_encoder categorical \
  --graphformer_readout mean_max \
  --graphformer_use_edge_bias \
  --results_name graphformer_profile_sweep/TOXCAST/cat_edge_meanmax/coremol_value_conservative_g0005_seed0
```

Expected: reduce negative TCM while keeping CoReMol AUC close to or above `0.7698`.
