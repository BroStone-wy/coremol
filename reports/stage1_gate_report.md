# CoReMol Stage-1 Gate Report

Scope: BBBP and ESOL, seeds 0/1/2, Base AttentiveFP versus Full CoReMol.

## Per-Seed Gate

| Dataset | Seed | Task Improved | Delta Task | Delta TCM | Gate |
|---|---:|---:|---:|---:|---:|
| BBBP | 0 | True | 0.053088 | 0.000017 | True |
| BBBP | 1 | True | 0.031024 | 0.000233 | True |
| BBBP | 2 | True | 0.115425 | -0.000076 | False |
| ESOL | 0 | True | 0.144606 | -0.000850 | False |
| ESOL | 1 | True | 0.224969 | -0.000561 | False |
| ESOL | 2 | True | 0.285402 | -0.000994 | False |

## Aggregate Metrics

- BBBP: base test_roc_auc=0.626361±0.027125; coremol test_roc_auc=0.692873±0.032660; delta_tcm=0.000058±0.000158
- ESOL: base test_rmse=1.635679±0.089421; coremol test_rmse=1.417353±0.111991; delta_tcm=-0.000802±0.000220

## Analysis

- BBBP: task improved on 3/3 seeds, TCM improved on 2/3 seeds, both improved on 2/3 seeds.
- ESOL: task improved on 3/3 seeds, TCM improved on 0/3 seeds, both improved on 0/3 seeds.

## Gate Decision

FAIL

Failure handling: inspect whether TCM fails because base support normalization, demand collapse, calibration saturation, or residual gate scale is limiting the main mechanism. Any next adjustment should modify those core mechanism parameters only.
