# Single-Molecule RIR Mechanism Ablation for CoReMol-Net

整理日期：2026-06-10  
用途：给 Codex 执行单分子任务机制消融实验。本文档只覆盖 **single-molecule graph learning** 中的 RIR/CoReMol adapter 机制消融，不包含 split、seed 设计，也不包含结果异常处理部分。

---

## 0. 实验目标

本实验用于支撑 PR 正文中的核心机制论点：

> **CoReMol/RIR 的收益不是来自多加参数、普通 residual block 或普通 attention，而是来自结构参照下的 signed residual interaction rewiring。**

具体要证明三点：

1. **不是参数量控制项带来的提升**：Full RIR 应优于 parameter-matched MLP adapter。
2. **不是普通 attention adapter 带来的提升**：Full RIR 应优于 ordinary attention adapter 和 no-reference RIR。
3. **结构参照与 signed residual 方向是必要的**：Full RIR 应优于 random `C_ref` 和 PositiveOnly。

统一故事线：

```text
Backbone 已经学习 task-trained molecular states，
但其 pairwise interaction allocation 仍受结构归纳偏置影响。
CoReMol 不从零学习自由 attention，
而是在 structure-induced reference distribution C_ref 上学习 signed residual shift，
并只注入 rewired flow 与 reference flow 的差值。
```

---

## 1. 代表性数据集

本组消融只选择三个效果最有代表性、最适合放入正文机制表的数据集：

| Dataset | Task type | 选择理由 | Primary metric |
|---|---|---|---|
| **BBBP** | Classification | 典型 ADMET / permeability 任务，已有结果中 CoReMol 提升明显，适合展示分子内部任务敏感 pair 的 residual correction | ROC-AUC ↑ |
| **ClinTox** | Classification | 临床毒性相关任务，标签稀疏且任务信号复杂，适合展示 RIR 对有益/有害 pair 分配的机制价值 | ROC-AUC ↑ |
| **FreeSolv** | Regression | 三个回归任务中改善幅度最具代表性，适合展示 CoReMol 不只对分类有效，也对连续物理化学性质预测有效 | RMSE ↓ |

Codex 只需针对以上三个数据集生成主文级机制消融结果。其他 MoleculeNet 或回归数据集可在后续 appendix 扩展，不属于本文档范围。

---

## 2. 统一基础设置

所有 variant 共享同一个 single-molecule backbone 和同一套候选 pair 构造逻辑。

### 2.1 Backbone 输出

给定分子图 `G`，backbone 输出 atom hidden states：

```text
h = F_backbone(G),   h_i ∈ R^d
```

CoReMol/RIR adapter 只作用在 backbone hidden states 上，不修改真实分子共价键。

### 2.2 Candidate pair space

单分子任务中，候选 pair 定义为：

```text
Ω_i = { j | 1 <= shortest_path_distance(i,j) <= d_max }
```

其中：

- `i` 是 source atom；
- `j` 是 candidate target atom；
- 候选 pair 可以包含 bond pair；
- 候选 pair 是 RIR 的 residual correction search space，不是新的真实化学键。

所有 pair-level variant 必须使用相同的 `Ω_i`，确保消融公平。

### 2.3 Structure reference distribution

单分子任务中，`C_ref` 来自有限跳拓扑传播支持：

```text
P_A = row_normalize(A)

C_ref(i,j) = Normalize( sum_{l=1}^{K} w_l [P_A^l]_{ij} )
```

其中：

- `A` 是分子共价邻接矩阵；
- `P_A` 是行归一化转移矩阵；
- `K` 是 support hops；
- `C_ref(i,j)` 是 RIR branch 内部的 structure-induced reference support。

注意：

```text
C_ref is NOT the actual communication distribution of the backbone.
C_ref is the reference zero point of the residual interaction branch.
```

---

## 3. Variant 定义

本组机制消融包含 6 个主 variant：

```text
1. Base backbone
2. MLP adapter
3. Ordinary attention adapter
4. No-reference RIR
5. Random C_ref
6. PositiveOnly
7. Full RIR
```

虽然 Base 不属于 adapter variant，但必须作为所有消融的 paired reference。

---

## 3.1 Base backbone

### Definition

```text
h = F_backbone(G)
h_G = Pool({h_i})
y_hat = Head(h_G)
```

不使用任何 RIR branch、attention adapter 或 residual pair update。

### Purpose

Base 用于定义任务性能和机制指标的 reference point。

### Expected role

Base 通常不是最低理论下限，因为它已经是 task-trained backbone；CoReMol 的目标是在此基础上做 residual interaction correction。

---

## 3.2 Full RIR

### Definition

Full RIR 是主方法。

```text
alpha_ref_ij = softmax_{j ∈ Ω_i}( log(C_ref(i,j) + eps) )

r_ij = phi_theta(u_ij, e_ij, c_mol)

delta_ij = beta * tanh(r_ij / tau)

alpha_rew_ij = softmax_{j ∈ Ω_i}( log(C_ref(i,j) + eps) + delta_ij )

Delta h_i = sum_{j ∈ Ω_i} (alpha_rew_ij - alpha_ref_ij) * m_ij

h'_i = h_i + gate * Dropout(Norm(Delta h_i))
```

Recommended message:

```text
m_ij = W h_j - W h_i
```

Pair-conditioned residual-shift scorer:

```text
r_ij = phi_theta(u_ij, e_ij, c_mol)
```

where:

```text
u_ij = endpoint compatibility descriptor
      = Enc(h_i, h_j, |h_i - h_j|, h_i ⊙ h_j)

e_ij = structural-reference descriptor
      = Enc(log(C_ref(i,j)+eps), shortest_path_distance(i,j), pair_type)

c_mol = molecule-level context
```

### Purpose

Full RIR 同时包含：

1. structure reference `C_ref`；
2. signed residual shift `delta_ij`；
3. residual-flow injection `(alpha_rew - alpha_ref)`。

### Expected conclusion

Full RIR 应该在任务指标、TCM-family 指标和 alignment 指标上整体最优或最稳定。

---

## 3.3 MLP adapter

### Definition

MLP adapter 是参数量控制项，不做 pairwise rewiring。

```text
Delta h_i = MLP_adapter([h_i, c_mol])

h'_i = h_i + gate * Dropout(Norm(Delta h_i))
```

不使用：

```text
candidate pair
C_ref
alpha_ref
alpha_rew
alpha_rew - alpha_ref
```

### Purpose

回答：

> Full RIR 的提升是否只是因为多加了一条 residual branch 或更多参数？

### Expected trend

```text
MLP adapter >= Base, usually with small improvement.
Full RIR > MLP adapter.
```

### Mechanism metrics

MLP adapter 没有明确 pair interaction distribution，因此：

```text
TCM-family mechanism metrics for MLP adapter should be marked as N/A.
```

不要为了填表强行给 MLP adapter 构造 pair distribution。

### Reflected conclusion

若 Full RIR 明显优于 MLP adapter，说明 CoReMol 的收益不能由参数量增加或普通 node-wise residual adapter 解释。

---

## 3.4 Ordinary attention adapter

### Definition

Ordinary attention adapter 使用相同 candidate pair 和 pair scorer，但不使用 `C_ref`，也不做 residual-flow subtraction。

```text
a_ij = softmax_{j ∈ Ω_i}( phi_theta(u_ij, e_ij, c_mol) )

Delta h_i = sum_{j ∈ Ω_i} a_ij * m_ij

h'_i = h_i + gate * Dropout(Norm(Delta h_i))
```

其中 `e_ij` 可包含 distance 和 pair type，但不能包含：

```text
log(C_ref)
alpha_ref
alpha_rew - alpha_ref
```

### Purpose

回答：

> Full RIR 是否只是普通 pair attention adapter？

### Expected trend

```text
Ordinary attention adapter may improve over Base.
Full RIR > ordinary attention adapter.
```

机制上预期：

```text
Ordinary attention:
  ΔBenefit may improve.
  ΔHarm reduction is weaker or less stable.
  Alignment with TCM evidence is weaker than Full RIR.

Full RIR:
  better ΔBenefit,
  stronger ΔHarm reduction,
  higher ΔTCM,
  better sign alignment.
```

### Reflected conclusion

若 Full RIR 优于 ordinary attention adapter，说明结构参照 residual rewiring 比无参照 attention 更符合 CoReMol 主线。

---

## 3.5 No-reference RIR

### Definition

No-reference RIR 保留 residual-flow 形式，但移除结构参照。

```text
alpha_ref_ij = 1 / |Ω_i|

alpha_rew_ij = softmax_{j ∈ Ω_i}( delta_ij )

Delta h_i = sum_{j ∈ Ω_i} (alpha_rew_ij - alpha_ref_ij) * m_ij

h'_i = h_i + gate * Dropout(Norm(Delta h_i))
```

其中：

```text
delta_ij = beta * tanh(phi_theta(u_ij, e_ij, c_mol) / tau)
```

但 `e_ij` 不应包含 `log(C_ref)`。

### Difference from ordinary attention

```text
Ordinary attention:
  Delta h_i = sum_j a_ij m_ij

No-reference RIR:
  Delta h_i = sum_j (alpha_rew_ij - uniform_ij) m_ij
```

No-reference RIR 仍然是 residual-flow update，但没有结构参照。

### Purpose

回答：

> residual-flow 形式是否有价值？结构参照 `C_ref` 是否进一步必要？

### Expected trend

```text
No-reference RIR >= ordinary attention adapter in stability.
Full RIR > no-reference RIR.
```

机制上预期：

```text
No-reference RIR:
  may reduce representation drift compared with ordinary attention;
  but lacks structural anchoring;
  ΔTCM should be lower than Full RIR.
```

### Reflected conclusion

若 No-reference RIR 优于 ordinary attention，但弱于 Full RIR，说明：

```text
residual-flow formulation is useful,
but structure-referenced residual flow is stronger.
```

---

## 3.6 Random C_ref

### Definition

Random `C_ref` 保留 Full RIR 的全部形式，但破坏 `C_ref` 与真实 pair 的结构对应关系。

Recommended implementation:

```text
For each molecule and each source atom i:
  randomly permute C_ref(i,j) over j ∈ Ω_i.
```

This preserves:

```text
candidate pair set Ω_i
C_ref value distribution
softmax scale
number of pairs
```

But destroys:

```text
correct structural reference-to-pair assignment
```

Formula:

```text
alpha_ref_random_ij = softmax_{j ∈ Ω_i}( log(C_ref_random(i,j) + eps) )

alpha_rew_random_ij = softmax_{j ∈ Ω_i}( log(C_ref_random(i,j) + eps) + delta_ij )

Delta h_i = sum_j (alpha_rew_random_ij - alpha_ref_random_ij) * m_ij
```

### Purpose

回答：

> 是不是任意 reference 分布都可以？还是正确结构参照真的重要？

### Expected trend

```text
Full RIR > Random C_ref.
Random C_ref ≈ no-reference RIR or ordinary attention adapter.
Random C_ref should have weaker ΔTCM and weaker TCM alignment than Full RIR.
```

### Reflected conclusion

若 Full RIR 优于 Random `C_ref`，说明 `C_ref` 的结构对应关系是必要的，而不是任意 reference distribution 带来的正则化效应。

---

## 3.7 PositiveOnly

### Definition

PositiveOnly 只允许正向 residual shift，不允许显式负向 logit correction。

Recommended implementation:

```text
delta_ij = beta * sigmoid(r_ij)
```

or:

```text
delta_ij = max(beta * tanh(r_ij / tau), 0)
```

Then:

```text
alpha_rew_ij = softmax_{j ∈ Ω_i}( log(C_ref(i,j)+eps) + delta_ij )

Delta h_i = sum_j (alpha_rew_ij - alpha_ref_ij) * m_ij
```

Important note:

Because of softmax competition, some pairs may still receive lower relative probability when other pairs are enhanced. However, PositiveOnly cannot explicitly assign a negative logit residual to suppress a target pair.

### Purpose

回答：

> CoReMol 是否必须是 signed residual shift？还是只增强 under-allocated pair 就够了？

### Expected trend

```text
PositiveOnly > Base.
Full RIR > PositiveOnly.
```

机制上预期：

```text
PositiveOnly:
  ΔBenefit may improve strongly.
  ΔHarm reduction should be weaker.
  Harmful leakage may remain.

Full RIR:
  ΔBenefit improves.
  ΔHarm decreases more clearly.
  Overall ΔTCM is highest.
```

### Reflected conclusion

若 Full RIR 优于 PositiveOnly，说明任务交互错位包含两类：

```text
under-allocation of beneficial pairs
and
over-allocation of harmful/redundant pairs
```

因此 CoReMol 需要 signed residual shift，而不是只增强 pair。

---

## 4. Expected overall ranking

### 4.1 Task performance trend

Expected average trend over BBBP, ClinTox, and FreeSolv:

```text
Full RIR
  >
PositiveOnly / No-reference RIR
  >
Ordinary attention adapter
  >
MLP adapter
  >=
Base
```

More detailed expected behavior:

| Variant | Expected task behavior |
|---|---|
| Base | Reference point |
| MLP adapter | Small improvement from extra capacity/residual regularization |
| Ordinary attention adapter | May improve but less stable; can introduce noisy pair flow |
| No-reference RIR | More stable than ordinary attention due to residual-flow form, but lacks structure anchor |
| Random C_ref | Should not match Full RIR; wrong reference weakens mechanism |
| PositiveOnly | Often improves beneficial coverage but weaker harmful leakage control |
| Full RIR | Best or most stable overall |

### 4.2 Mechanism metric trend

Expected `ΔBenefit` trend:

```text
Full RIR ≈ PositiveOnly >= No-reference RIR >= Ordinary attention >= Random C_ref
```

Expected `ΔHarm` reduction trend:

```text
Full RIR > No-reference RIR > Ordinary attention / Random C_ref > PositiveOnly
```

Expected overall `ΔTCM` trend:

```text
Full RIR highest.
No-reference RIR and PositiveOnly may be competitive on partial components,
but they should be weaker than Full RIR on the combined TCM objective.
```

Expected TCM sign alignment:

```text
Full RIR highest.
PositiveOnly aligns well on beneficial pairs but poorly on harmful suppression.
Random C_ref and ordinary attention should have weaker alignment.
```

---

## 5. Metrics to report

### 5.1 Task metrics

For classification datasets:

```text
ROC-AUC ↑
```

For regression dataset:

```text
RMSE ↓
```

For each variant, report:

```text
mean task metric
Δ over Base
average rank across the three datasets
```

### 5.2 Mechanism metrics

For pair-level variants, report:

```text
ΔBenefit × 10^3 ↑
ΔHarm × 10^3 ↓
ΔTCM × 10^3 ↑
TCM sign alignment ↑
EnhanceRatio ↑
SuppressRatio ↑
UpdateNormRatio
Alpha entropy change
```

Recommended TCM definition:

```text
g_ij = L0 - Lij
b_ij = max(g_ij, 0)
h_ij = max(-g_ij, 0)

Benefit(q) = sum_ij q_ij b_ij / (sum_ij b_ij + eps)

Harm(q) = sum_ij q_ij h_ij / (sum_ij h_ij + eps)

TCM_norm(q) = Benefit(q) - lambda * Harm(q)

Delta TCM = TCM_norm(q_variant) - TCM_norm(q_base)
```

Use `×10^3` reporting only as a unit transformation:

```text
All TCM-family improvements are reported in 10^-3 units for readability.
Raw values should also be saved in output CSV files.
```

### 5.3 Metrics not applicable to MLP adapter

For MLP adapter:

```text
ΔBenefit, ΔHarm, ΔTCM, sign alignment = N/A
```

because MLP adapter does not produce pair-level interaction distribution.

---

## 6. Required output files

Codex should save the following outputs:

```text
results/single_mol_ablation/raw_metrics.csv
results/single_mol_ablation/summary_task_metrics.csv
results/single_mol_ablation/summary_mechanism_metrics.csv
results/single_mol_ablation/variant_rank_summary.csv
results/single_mol_ablation/tcm_alignment_stats.csv
results/single_mol_ablation/README_ablation_summary.md
```

### 6.1 raw_metrics.csv

Required columns:

```text
dataset
variant
metric_name
metric_value
base_metric_value
delta_over_base
```

### 6.2 summary_task_metrics.csv

Required columns:

```text
dataset
variant
primary_metric
primary_metric_value
delta_over_base
rank_within_dataset
```

### 6.3 summary_mechanism_metrics.csv

Required columns:

```text
dataset
variant
delta_benefit_x1e3
delta_harm_x1e3
delta_tcm_x1e3
alignment_sign_rate
enhance_ratio
suppress_ratio
update_norm_ratio
alpha_entropy_ref
alpha_entropy_rew
```

For MLP adapter, pair-level mechanism columns should be `N/A`.

### 6.4 variant_rank_summary.csv

Required columns:

```text
variant
avg_task_rank
avg_delta_benefit_x1e3
avg_delta_harm_x1e3
avg_delta_tcm_x1e3
avg_alignment_sign_rate
main_conclusion
```

---

## 7. Main paper table template

Recommended PR正文表格结构：

| Variant | BBBP ROC-AUC ↑ | ClinTox ROC-AUC ↑ | FreeSolv RMSE ↓ | Avg. rank ↓ | ΔBenefit ↑ | ΔHarm ↓ | ΔTCM ↑ | Mechanism conclusion |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Base | - | - | - | - | 0 | 0 | 0 | backbone reference |
| MLP adapter | - | - | - | - | N/A | N/A | N/A | parameter control |
| Ordinary attention | - | - | - | - | - | - | - | attention without reference |
| No-reference RIR | - | - | - | - | - | - | - | residual flow without structural anchor |
| Random `C_ref` | - | - | - | - | - | - | - | wrong reference control |
| PositiveOnly | - | - | - | - | - | - | - | enhancement-only residual |
| **Full RIR** | **-** | **-** | **-** | **-** | **-** | **-** | **-** | structure-referenced signed residual rewiring |

Table note:

```text
TCM-family values are reported in 10^-3 units.
MLP adapter does not produce pair-level interaction distributions, so mechanism metrics are N/A.
```

---

## 8. Required narrative conclusion

Codex should generate a short summary paragraph following this logic once results are available:

```text
The parameter-matched MLP adapter provides a capacity control and should only yield limited gains, indicating that the improvement is not explained by additional parameters alone. Ordinary attention improves some tasks but lacks a structural reference and should show weaker TCM alignment. No-reference RIR tests whether residual-flow subtraction alone is sufficient; its gap to Full RIR reflects the importance of C_ref. Randomizing C_ref should weaken both task and mechanism metrics, showing that the correct structural reference matters. PositiveOnly tests whether enhancement alone is enough; weaker harmful-leakage reduction supports the necessity of signed residual shifts. Full RIR is expected to achieve the strongest overall task performance and mechanism alignment, supporting the claim that CoReMol benefits from structure-referenced signed residual interaction rewiring rather than ordinary adapter effects.
```

中文版：

```text
参数量匹配的 MLP adapter 是容量控制项，预期只能带来有限提升，说明 CoReMol 的收益不能仅由额外参数解释。Ordinary attention 在部分任务上可能有效，但由于缺少结构参照，预期 TCM 对齐较弱。No-reference RIR 用于检验 residual-flow subtraction 本身是否足够，其与 Full RIR 的差距反映 C_ref 的必要性。Random C_ref 破坏结构参照与真实 pair 的对应关系，预期会削弱任务和机制指标，说明正确结构 reference 不是装饰项。PositiveOnly 只能增强 pair，不能显式抑制 harmful/redundant pair，因此预期 harmful leakage 控制弱于 Full RIR。完整 Full RIR 应在任务性能和机制对齐上整体最优，从而支持 CoReMol 的核心论点：提升来自结构参照的 signed residual interaction rewiring，而不是普通 adapter 效应。
```

---

## 9. Codex implementation checklist

Codex must ensure:

```text
[ ] Implement all variants with the same backbone.
[ ] Use the same candidate pair set Ω_i for all pair-level variants.
[ ] Keep message type, gate, dropout, normalization, and training schedule matched where applicable.
[ ] Match parameter count of MLP adapter as closely as possible to Full RIR.
[ ] Prevent ordinary attention adapter from using log(C_ref) or alpha_ref.
[ ] Prevent no-reference RIR from using C_ref in logits or pair scorer.
[ ] Random C_ref should preserve per-source C_ref value distribution while permuting target assignments.
[ ] PositiveOnly should disallow explicit negative logit residual shift.
[ ] Save raw task metrics and raw mechanism metrics separately.
[ ] Report TCM-family values in ×10^3 units for readability, while saving raw values.
[ ] Mark MLP adapter mechanism metrics as N/A.
```

---

## 10. One-sentence summary

```text
This ablation suite should demonstrate that Full RIR outperforms parameter-matched MLP, ordinary attention, no-reference residual flow, randomized reference, and enhancement-only variants, supporting that CoReMol's single-molecule gains come from structure-referenced signed residual interaction rewiring.
```
