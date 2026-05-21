# CoReMol User Requirements Memory

This file records the user's standing requirements for the CORMOL project. Read this file before planning, coding, launching experiments, or reporting results.

## Primary Goal

CoReMol must demonstrate both:

- improved main task metrics, and
- improved interpretability/communication metrics, especially Full TCM and, when useful, TCM@10.

Task metrics alone are not enough. TCM improvement alone is not enough for the final claim.

Diagnostic candidates should also be judged by the current gate:

- use true Full TCM, not `TCM@10`/`mechanism_metrics.delta_tcm`;
- prioritize candidates where Full TCM improves on all selected/reporting seeds;
- require the target task metric mean to improve in the correct direction; about `+0.01` AUC remains the preferred strong result, but a smaller positive mean task gain is acceptable when Full TCM is consistently positive;
- if Full TCM fails or the target metric mean is negative, record the run as failure/ablation evidence rather than as an effective diagnostic result.

## Experimental Standard

- Prefer standard, auditable dataset splits that allow fair comparison with reported papers.
- Keep all raw results, configs, checkpoints, logs, and summaries.
- If screening many seeds or profiles, explicitly mark it as screening. Do not hide failed seeds or failed profiles.
- For final claims, use three-seed mean and standard deviation unless the user explicitly asks for a pilot.
- Baselines should be comparable to the paper-reported backbone when claiming paper-level comparison.
- CoReMol should exceed the relevant paper baseline or paper Ours target where possible; if not, report the gap honestly and analyze why.

## Current Adjusted Classification Target

The user has flagged the Graph Curvature Flow-Based Masked Attention table as potentially suspicious for Graphformer/OURS on SIDER and ToxCast because those values are far above most other methods. For the current reproducibility/reporting stage:

- Do not force all results to match suspicious Graphformer/OURS entries.
- SIDER current reproducible scale can be reported around best baseline `0.6595` and CoReMol old best around `0.6667`.
- ToxCast current reproducible scale can be reported around `0.7664 -> 0.7698` for the profile where CoReMol improves task AUC.
- ClinTox current reproducible scale around best baseline `0.9349` is acceptable, but CoReMol must be tuned so task AUC and TCM improve if used as a final CoReMol claim.
- BBBP standard scaffold Graphformer-CoReMol scale around `0.6865` is acceptable, but it also needs CoReMol tuning so task AUC and TCM improve if used as a final CoReMol claim.
- Priority for the next optimization loop: improve TCM interpretability first, while keeping or improving task AUC. A task-only gain with negative TCM remains diagnostic, not final.

## Updated Graphformer Paper-Claim Target

The user now treats about `+0.01` AUC as a preferred strong Graphformer classification target, not an absolute hard gate. Weak positive deltas should still be improved where possible, but a smaller positive mean task gain can be acceptable if true Full TCM improves consistently across the reported seeds.

For Graphformer-backed classification results:

- Always report the locally reproduced Graphformer baseline and the matching Graphformer+CoReMol result.
- The main claim should show a positive relative task improvement, targeting at least about `+0.01` ROC-AUC over the local baseline where feasible.
- This applies to both cases:
  - local baseline is below or only near the paper table, such as BBBP, ClinTox, SIDER, and ToxCast;
  - local baseline is already above the paper table, such as BACE, HIV, and Tox21.
- If seed screening is used, it must be transparently marked as screening, with all screened raw results retained. Selected top-3 seeds may be reported only with a clear selection rule and source directories.
- Final Graphformer summary should not rely on paper-value comparison alone. It must make the CoReMol-vs-local-baseline improvement visibly meaningful.
- Full TCM remains required for the interpretability story. A task gain without positive Full TCM is still a failed/ablation run rather than a usable diagnostic candidate.
- For screening and diagnosis, prefer candidates that satisfy both true Full TCM improvement and about `+0.01` task improvement. If no such stable candidate exists, a consistently positive Full TCM result with a smaller positive mean task gain can be reported transparently as a modest task improvement rather than a strong performance claim.

## Mechanism Boundary

CoReMol is a residual atom-pair communication calibration mechanism. Its core calculation should stay backbone-agnostic:

- input: atom/node states, graph connectivity, batch indices, and optionally generic edge/distance features;
- output: residual correction to atom/node states;
- core idea: calibrate pair communication by residual distribution correction, not by adding unrelated modules.

Do not make the residual reconnection calculation strongly depend on a specific backbone's private internals, such as a particular Graphformer graph token implementation, attention map format, layer name, or task-specific readout trick.

## Generality Requirement

The final story must show CoReMol as a plug-and-play adapter. Target at least 3-4 different backbone families across experiments when feasible, for example:

- AttentiveFP,
- Graphformer/Graphormer-style backbone,
- GIN/GINE or D-MPNN-like molecular GNN,
- later affinity/backbone models such as ligand-pocket or dual-input architectures.

The meaning of "general" can be practical, not overly rigid. It is acceptable to support:

- post-encoder residual insertion,
- layerwise residual insertion,
- encoder-block internal residual insertion,
- different insertion schedules per backbone.

This is acceptable as long as the residual communication computation itself remains essentially the same and is not hard-coded to one backbone's private mechanism.

## Backbone-Specific Tuning Boundary

Backbone configuration and baseline alignment may be task-specific:

- Graphformer submodules, readout, categorical features, local GNN use, and edge/spatial bias can be tuned as backbone hyperparameters.
- Different datasets may need different backbone profiles.
- This tuning is not CoReMol's core claim.

CoReMol hyperparameters may also be tuned:

- `value` versus `delta` residual message,
- `beta`,
- `tau`,
- `d_max`,
- `support_hops`,
- residual gate initialization/cap,
- post versus layerwise placement.

But do not redesign CoReMol into a stack of unrelated modules just to chase a single dataset.

## Current Mechanism Interpretation

Two residual message definitions are both allowed as mechanism variants:

- `value`: message uses `V(h_j)` and directly injects calibrated value communication.
- `delta`: message uses `V(h_j) - V(h_i)` and emphasizes relative pairwise correction.

The final report should explain which variant is used and why. If both are tested, keep both results and select by transparent criteria.

## Failure Analysis Requirement

When a run fails, analyze the cause before changing multiple things:

- baseline too weak or not paper-comparable,
- split/protocol mismatch,
- CoReMol task metric improves but TCM drops,
- TCM improves but task metric drops,
- gate too strong or too weak,
- residual insertion too late or too early,
- backbone readout does not use corrected atom states effectively.

Prefer one-variable changes and record the reasoning. Do not randomly combine modules.

## Affinity-Task Future Constraint

Future affinity prediction tasks may have two molecular/protein inputs, such as ligand and pocket/protein graphs. CoReMol should remain extendable to those settings by applying the same residual communication adapter to:

- ligand graph,
- pocket/protein graph,
- or a generic ligand-pocket interaction graph if defined.

Do not implement a CoReMol variant that only works for single-molecule classification and cannot be explained for dual-input affinity models.

## Reporting Discipline

Every report should answer:

- Which dataset and split?
- Which backbone and profile?
- Which CoReMol variant and hyperparameters?
- Baseline mean/std and CoReMol mean/std?
- Main metric win count across seeds?
- Full TCM and TCM@10 win count where available?
- How does it compare with the paper table?
- What failed and what is the next targeted adjustment?

Do not claim final success unless both task and mechanism criteria are satisfied under the agreed protocol.

## TCM Metric Implementation Note

`CORMOL/scripts/run_stage1_gate.py` writes `mechanism_metrics.delta_tcm` from `estimate_tcm_at_k`, so that column is TCM@k, usually TCM@10, not Full TCM.

For Full TCM and top-k components, use `CORMOL/scripts/compute_tcm_variants.py`. Current Graphformer support was added so existing Graphformer checkpoints can be recomputed without retraining.
