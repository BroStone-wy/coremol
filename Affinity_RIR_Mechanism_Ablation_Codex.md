# Affinity RIR Mechanism Ablation for Codex

整理日期：2026-06-10  
用途：给 Codex 执行 **CoReMol-Net-Affinity / Cross-RIR 亲和力机制消融实验**。  
实验范围：**只跑 PDBbind CleanSplit / GEMS-style 配置**。不要跑 CheapNet/GIGN cross-dataset，也不要跑 ATOM3D LBA。

---

## 0. 核心目标

本实验不是为了证明一个专门的 protein--ligand affinity SOTA 模型，而是为了证明：

> **CoReMol 作为分子图学习中的结构参照残差交互重连机制，在 pocket-conditioned molecular graph learning 中也有效。亲和力提升应来自 ligand--pocket interface residual rewiring，而不只是 ligand-only shortcut、pocket pooling 或普通上下文拼接。**

本文件只定义亲和力机制消融，固定使用 CleanSplit 配置。Codex 不要在本实验文档中加入 split / seed 说明。

---

## 1. CleanSplit-only Evaluation Scope

只使用 **PDBbind CleanSplit / GEMS-style protocol**。

优先评估以下 CleanSplit 相关测试集合。如果当前数据包只提供其中一部分，按可用集合执行，但不要引入非 CleanSplit 配置。

```text
Primary:
  PDBbind CleanSplit -> CASF2016

Secondary within CleanSplit protocol:
  PDBbind CleanSplit -> CASF2013, if available
  PDBbind CleanSplit -> CASF2016 independent subset, if available
```

不要跑：

```text
CheapNet/GIGN cross-dataset Test2013/Test2016/Test2019/CSAR
ATOM3D LBA 30% / 60%
LEP
self-defined random affinity split
```

---

## 2. 本组消融要回答的问题

这组消融要支撑 PR 正文中的亲和力故事线：

```text
Ligand-only baseline
  ↓
Pocket context proves the task is pocket-conditioned
  ↓
Intra-RIR improves ligand molecular representation under pocket context
  ↓
Cross-RIR directly rewires ligand--pocket interface flow
  ↓
Full model combines intra-molecular and interface-level residual rewiring
  ↓
Interface flow readout proves Cross-RIR produces explicit predictive evidence
```

对应审稿问题：

| 审稿问题 | 对应实验 |
|---|---|
| 模型是不是只靠 ligand 记忆？ | ligand-only vs pocket context |
| pocket 信息是否真的有用？ | pocket context vs ligand-only |
| 只做 ligand 内部 RIR 是否足够？ | Intra-RIR vs Cross-RIR |
| 亲和力收益是否来自 interface rewiring？ | pocket context vs Cross-RIR vs Full |
| Cross-RIR 的 residual interface flow 是否真的进入预测？ | Full vs w/o interface readout |

最终希望证明：

> **Full CoReMol-Net-Affinity 最强；Cross-RIR 是亲和力任务的关键模块；Intra-RIR 有帮助但不是亲和力任务的全部；interface residual-flow readout 对 affinity prediction 有独立贡献。**

---

## 3. 六个 Variant 的精确定义

本实验只实现以下 6 个 variant：

```text
1. ligand-only
2. pocket context
3. Intra-RIR
4. Cross-RIR
5. Full
6. w/o interface readout
```

所有 variant 应尽量共享相同的 ligand encoder、pocket encoder、hidden dimension、head width、dropout、optimizer setting 和训练预算。唯一变化应来自对应机制模块。

---

## 3.1 Variant 1: ligand-only

### 定义

只使用 ligand graph encoder 和 ligand graph readout，不使用 pocket encoder，不使用 pocket context，不使用 Intra-RIR，不使用 Cross-RIR。

```text
h_L = F_ligand(G_L)

c_L = Pool(h_L)

y_hat = Head(c_L)
```

### 不包含

```text
pocket context
pocket encoder
ligand--pocket interaction
Intra-RIR
Cross-RIR
interface residual flow
```

### 作用

该 variant 是 affinity task 中的 ligand shortcut baseline，用来排除：

> 模型是否只是把 binding affinity 当作 ligand-only molecular property regression 来做。

### 预期趋势

```text
ligand-only 应该是最弱或接近最弱。
```

在 CleanSplit 下，ligand-only 不应接近 Full。如果 ligand-only 接近 Full，说明 pocket-conditioned interaction learning 的证据不足。

### 反映的结论

如果 pocket context / Cross-RIR / Full 明显优于 ligand-only，则可以说明：

> CleanSplit 下的 affinity prediction 不能被 ligand-only shortcut 充分解释，pocket-conditioned interaction information 是必要的。

---

## 3.2 Variant 2: pocket context

### 定义

加入 pocket encoder，但只把 pocket 表示作为全局上下文进入 affinity head。不做 ligand internal residual rewiring，不做 ligand--pocket Cross-RIR。

```text
h_L = F_ligand(G_L)

h_P = F_pocket(G_P)

c_L = Pool(h_L)
c_P = Pool(h_P)

y_hat = Head([c_L, c_P])
```

### 包含

```text
ligand encoder
pocket encoder
ligand pooling
pocket pooling
```

### 不包含

```text
Intra-RIR
Cross-RIR
interface residual flow readout
```

### 作用

回答：

> pocket 信息本身是否能改善 affinity prediction？

### 预期趋势

```text
pocket context > ligand-only
```

但：

```text
Cross-RIR > pocket context
Full > pocket context
```

### 反映的结论

如果 pocket context 优于 ligand-only，可以说明：

> Affinity prediction 不是纯 ligand property prediction；binding-site context 提供了有效环境信息。

如果 Cross-RIR 进一步优于 pocket context，可以说明：

> 简单 pocket pooling 不足以刻画 ligand--pocket interface evidence，需要显式 interface residual rewiring。

---

## 3.3 Variant 3: Intra-RIR

### 定义

在 ligand 内部做 Intra-RIR。该 Intra-RIR 的 residual-shift scorer 可以被 pocket context 调制，但不做 ligand--pocket Cross-RIR。

```text
h_L = F_ligand(G_L)

h_P = F_pocket(G_P)

c_P = Pool(h_P)

For ligand atom pair (i, j):

alpha_ref_ij^L = softmax_j(log(C_ref^L(i,j) + eps))

delta_ij^L = beta * tanh(phi_L(h_i^L, h_j^L, C_ref^L(i,j), d_ij^L, c_P) / tau)

alpha_rew_ij^L = softmax_j(log(C_ref^L(i,j) + eps) + delta_ij^L)

Delta h_i^L,intra = sum_j (alpha_rew_ij^L - alpha_ref_ij^L) * m_ij^L

h_i^{L'} = h_i^L + gate_intra * Norm(Dropout(Delta h_i^L,intra))

c_L' = Pool(h_L')
c_P  = Pool(h_P)

y_hat = Head([c_L', c_P])
```

推荐消息：

```text
m_ij^L = W h_j^L - W h_i^L
```

### 包含

```text
ligand-internal RIR
pocket-conditioned residual-shift scorer
ligand topology reference C_ref^L
```

### 不包含

```text
ligand--pocket Cross-RIR
interface residual flow readout
```

### 作用

回答：

> 只改善 ligand molecular graph representation 是否足以解释 affinity 提升？

### 预期趋势

```text
Intra-RIR > pocket context
Cross-RIR > Intra-RIR
Full > Intra-RIR
```

### 反映的结论

如果 Intra-RIR 优于 pocket context，可以说明：

> 在 pocket-conditioned supervision 下，ligand 内部哪些 atom pair 应该增强或抑制会发生变化；Intra-RIR 能改善 ligand 表示。

如果 Cross-RIR 进一步优于 Intra-RIR，可以说明：

> 亲和力任务的关键不只是 ligand 内部结构表达，而是 ligand--pocket interface evidence。

---

## 3.4 Variant 4: Cross-RIR

### 定义

不做 ligand-internal Intra-RIR，只做 ligand--pocket interface residual rewiring。

候选 cross pair：

```text
(i, r)

i = ligand atom
r = pocket residue or pocket region
```

Cross reference：

```text
C_ref^LP(i,r) = distance/contact-based reference support
```

推荐距离定义：

```text
d_ir = min distance between ligand atom i and atoms in pocket residue r
```

推荐 reference kernel：

```text
C_ref^LP(i,r) = exp(-d_ir^2 / sigma^2) * 1[d_ir <= cutoff]
```

或使用 RBF/contact kernel。

Cross-RIR 公式：

```text
alpha_ref_i,r^LP = softmax_r(log(C_ref^LP(i,r) + eps))

delta_i,r^LP = beta * tanh(phi_LP(h_i^L, h_r^P, C_ref^LP(i,r), d_ir, contact_i,r) / tau)

alpha_rew_i,r^LP = softmax_r(log(C_ref^LP(i,r) + eps) + delta_i,r^LP)

Delta h_i^L,cross = sum_r (alpha_rew_i,r^LP - alpha_ref_i,r^LP) * m_i,r^LP

h_i^{L'} = h_i^L + gate_cross * Norm(Dropout(Delta h_i^L,cross))
```

推荐消息：

```text
m_i,r^LP = W_P h_r^P - W_L h_i^L
```

Interface residual-flow representation：

```text
h_interface = Pool_{(i,r)} [
  (alpha_rew_i,r^LP - alpha_ref_i,r^LP) * m_i,r^LP,
  delta_i,r^LP,
  contact_features_i,r
]
```

Prediction：

```text
c_L' = Pool(h_L')
c_P  = Pool(h_P)

y_hat = Head([c_L', c_P, h_interface])
```

### 包含

```text
ligand--pocket Cross-RIR
cross reference C_ref^LP
interface residual flow readout
```

### 不包含

```text
ligand-internal Intra-RIR
```

### 作用

回答：

> interface-level residual rewiring 本身是否有效？

### 预期趋势

```text
Cross-RIR > pocket context
Cross-RIR > Intra-RIR
Full > Cross-RIR
```

### 反映的结论

如果 Cross-RIR 明显优于 Intra-RIR，可以说明：

> 亲和力任务的主要增益来自 ligand--pocket interface residual rewiring，而不是 ligand-only representation refinement。

---

## 3.5 Variant 5: Full

### 定义

完整 CoReMol-Net-Affinity：

```text
Ligand encoder
+ Pocket encoder
+ Ligand Intra-RIR
+ Ligand--pocket Cross-RIR
+ Interface residual flow readout
+ Affinity head
```

流程：

```text
h_L^0 = F_ligand(G_L)

h_P = F_pocket(G_P)

c_P = Pool(h_P)

h_L^1 = IntraRIR(h_L^0, C_ref^L, c_P)

h_L^2, h_interface = CrossRIR(h_L^1, h_P, C_ref^LP)

h_complex = [
  Pool(h_L^2),
  Pool(h_P),
  h_interface
]

y_hat = Head(h_complex)
```

### 包含

```text
pocket context
ligand-internal residual rewiring
ligand--pocket interface residual rewiring
interface residual flow readout
```

### 作用

作为亲和力任务的主方法，验证 intra-molecular 和 interface-level residual rewiring 的互补性。

### 预期趋势

```text
Full should be the best overall.
```

核心排序：

```text
Full > Cross-RIR
Full > Intra-RIR
Full > pocket context
Full > ligand-only
Full > w/o interface readout
```

### 反映的结论

如果 Full 最强，可以说明：

> ligand internal residual rewiring 和 ligand--pocket interface residual rewiring 是互补的。亲和力任务中，Cross-RIR 负责核心 interface evidence，Intra-RIR 进一步提供 pocket-conditioned ligand representation refinement。

---

## 3.6 Variant 6: w/o interface readout

### 定义

保留 Intra-RIR 和 Cross-RIR 的 hidden-state update，但从最终 affinity head 中移除显式 interface residual-flow representation。

```text
h_L^0 = F_ligand(G_L)

h_P = F_pocket(G_P)

c_P = Pool(h_P)

h_L^1 = IntraRIR(h_L^0, C_ref^L, c_P)

h_L^2, h_interface = CrossRIR(h_L^1, h_P, C_ref^LP)

h_complex = [
  Pool(h_L^2),
  Pool(h_P)
]

y_hat = Head(h_complex)
```

区别：

```text
Full:
  h_complex = [Pool(h_L^2), Pool(h_P), h_interface]

w/o interface readout:
  h_complex = [Pool(h_L^2), Pool(h_P)]
```

### 包含

```text
Intra-RIR
Cross-RIR hidden-state update
```

### 不包含

```text
explicit interface residual flow readout
```

### 作用

回答：

> Cross-RIR 产生的 residual interface flow 是否应该显式进入 affinity prediction？

### 预期趋势

```text
Full > w/o interface readout
w/o interface readout > pocket context
```

w/o interface readout 可能接近 Cross-RIR，但应低于 Full。

### 反映的结论

如果 Full 优于 w/o interface readout，可以说明：

> Cross-RIR 产生的 residual interface flow 不只是中间状态更新；它本身是 affinity prediction 的显式证据，应该被 affinity head 读出。

---

## 4. 总体预期任务性能趋势

任务指标：

```text
RMSE ↓
MAE ↓
Pearson R ↑
Spearman rho ↑
```

总体预期排序：

```text
Full
  >
Cross-RIR ≈ w/o interface readout
  >
Intra-RIR
  >
Pocket context
  >
Ligand-only
```

更重要的是以下 pairwise 趋势：

```text
Pocket context > Ligand-only
Intra-RIR > Pocket context
Cross-RIR > Intra-RIR
Full > Cross-RIR
Full > w/o interface readout
```

这些趋势分别支撑：

| 趋势 | 支撑结论 |
|---|---|
| Pocket context > Ligand-only | pocket context 必要，任务不是 ligand-only shortcut |
| Intra-RIR > Pocket context | pocket-conditioned ligand internal rewiring 有帮助 |
| Cross-RIR > Intra-RIR | 亲和力关键来自 interface residual rewiring |
| Full > Cross-RIR | intra-molecular 和 interface-level rewiring 互补 |
| Full > w/o interface readout | interface residual flow 本身是可预测证据 |

---

## 5. 机制指标要求

### 5.1 Task Metrics

每个 variant 至少报告：

```text
RMSE ↓
MAE ↓
Pearson R ↑
Spearman rho ↑
```

正文主表建议主要展示：

```text
RMSE
Pearson R
Spearman rho
```

MAE 可作为补充列或附表列。

---

### 5.2 Interface-TCM Metrics

对包含 Cross-RIR 的 variant 报 Interface-TCM：

```text
Cross-RIR
Full
w/o interface readout
```

需要保存并报告：

```text
Delta Cross-Benefit × 10^3 ↑
Delta Cross-Harm × 10^3 ↓
Delta Interface-TCM × 10^3 ↑
Interface alignment ↑
Contact beneficial ratio ↑
Contact harmful leakage ↓
Interface alpha entropy
UpdateNormRatio
```

定义：

```text
L0    = frozen base original loss
L_i,r = loss after applying small probe communication on ligand atom i and pocket residue r

g_i,r = L0 - L_i,r

b_i,r = max(g_i,r, 0)
h_i,r = max(-g_i,r, 0)
```

对 interaction distribution `q_i,r`：

```text
Benefit(q) = sum_{i,r} q_i,r * b_i,r / (sum_{i,r} b_i,r + eps)

Harm(q) = sum_{i,r} q_i,r * h_i,r / (sum_{i,r} h_i,r + eps)

TCM_interface(q) = Benefit(q) - lambda * Harm(q)
```

主文报告：

```text
Delta Interface-TCM = TCM_interface(q_RIR) - TCM_interface(q_base)
```

统一单位：

```text
All TCM-family values are reported in 10^-3 units.
```

---

### 5.3 Intra-TCM Metrics

对包含 Intra-RIR 的 variant 可报告 ligand-internal TCM：

```text
Intra-RIR
Full
w/o interface readout
```

指标：

```text
Delta Intra-Benefit × 10^3
Delta Intra-Harm × 10^3
Delta Intra-TCM × 10^3
```

注意：PR 正文中亲和力任务更重要的是 **Interface-TCM**。Intra-TCM 可作为 secondary mechanism metric 或 appendix 指标。

---

## 6. 预期机制趋势

### 6.1 Interface-TCM 总体趋势

```text
Full highest
Cross-RIR second
w/o interface readout positive but lower than Full
Ligand-only / Pocket context / Intra-RIR: N/A or near zero for interface metrics
```

表格化预期：

| Variant | Interface-TCM 预期 |
|---|---|
| ligand-only | N/A |
| pocket context | N/A or near zero |
| Intra-RIR | N/A or near zero |
| Cross-RIR | 明显提升 |
| w/o interface readout | 有提升，但任务性能低于 Full |
| Full | 最高 |

### 6.2 Cross-Benefit / Cross-Harm 趋势

```text
Delta Cross-Benefit:
  Full >= Cross-RIR > w/o interface readout > pocket context

Delta Cross-Harm reduction:
  Full > Cross-RIR >= w/o interface readout > pocket context
```

含义：

```text
Cross-RIR 应提高 beneficial ligand--pocket contact coverage。
Full 应进一步降低 harmful leakage。
w/o interface readout 仍可能有 Cross-RIR alignment，但预测性能低于 Full。
```

### 6.3 任务指标与机制指标的理想关系

最理想结果：

```text
Full:
  RMSE lowest
  Pearson/Spearman highest
  Delta Cross-Benefit highest
  Delta Cross-Harm reduction strongest
  Delta Interface-TCM highest
```

这支撑：

> Full CoReMol-Net-Affinity 的性能提升和机制改善同向。

---

## 7. PR 正文表格模板

建议生成一张主表：

```text
Table: Affinity mechanism ablation under PDBbind CleanSplit.
```

列：

```text
Variant
RMSE ↓
Pearson R ↑
Spearman rho ↑
Delta Cross-Benefit ×10^3 ↑
Delta Cross-Harm ×10^3 ↓
Delta Interface-TCM ×10^3 ↑
Mechanism conclusion
```

模板：

| Variant | RMSE ↓ | Pearson ↑ | Spearman ↑ | ΔCross-Benefit ↑ | ΔCross-Harm ↓ | ΔInterface-TCM ↑ | 机制结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| ligand-only | high | low | low | N/A | N/A | N/A | ligand shortcut baseline |
| pocket context | lower | higher | higher | N/A | N/A | N/A | pocket context helps |
| Intra-RIR | further lower | further higher | higher | N/A | N/A | N/A | ligand rewiring helps but incomplete |
| Cross-RIR | strong lower | strong higher | higher | + | - | + | interface rewiring is key |
| w/o interface readout | moderate/strong | moderate/strong | higher | + | - | + | interface flow exists but is not explicitly read out |
| Full | best | best | best | best | best | best | intra + cross + interface readout |

注意：最终论文表中必须填真实数值，不要直接写 best / high / low。

---

## 8. 每个趋势对应的正文结论

### 8.1 Pocket context > Ligand-only

英文：

```text
The improvement from ligand-only to pocket context indicates that affinity prediction cannot be reduced to ligand-only molecular graph regression; binding-site context provides necessary environmental information.
```

中文：

```text
pocket context 相比 ligand-only 的提升说明，亲和力预测不是简单的 ligand-only 分子性质回归，结合位点环境提供了必要上下文。
```

### 8.2 Intra-RIR > Pocket context

英文：

```text
The gain of Intra-RIR over pocket context suggests that pocket-conditioned intra-molecular residual rewiring improves ligand representations under binding supervision.
```

中文：

```text
Intra-RIR 相比 pocket context 的提升说明，在 pocket 监督下，ligand 内部 atom-pair interaction 的 residual rewiring 有助于改善 ligand 表示。
```

### 8.3 Cross-RIR > Intra-RIR

英文：

```text
The stronger gain of Cross-RIR over Intra-RIR shows that the main affinity improvement comes from ligand--pocket interface residual rewiring rather than ligand-only refinement.
```

中文：

```text
Cross-RIR 优于 Intra-RIR 说明，亲和力任务的主要提升来自 ligand--pocket 界面残差重连，而不只是 ligand 内部表示增强。
```

### 8.4 Full > Cross-RIR

英文：

```text
The full model further improves over Cross-RIR, indicating that intra-molecular and interface-level residual rewiring are complementary.
```

中文：

```text
Full 进一步优于 Cross-RIR，说明分子内部 residual rewiring 和界面 residual rewiring 是互补的。
```

### 8.5 Full > w/o interface readout

英文：

```text
Removing the interface residual-flow readout degrades performance, suggesting that the rewired interface flow is itself predictive evidence and should be explicitly exposed to the affinity head.
```

中文：

```text
去掉 interface residual-flow readout 后性能下降，说明 Cross-RIR 产生的 residual interface flow 本身就是亲和力预测证据，应该显式输入 affinity head。
```

### 8.6 Full has highest Interface-TCM

英文：

```text
Full CoReMol-Net-Affinity achieves the largest Interface-TCM improvement, with increased beneficial contact coverage and reduced harmful leakage, supporting that the performance gain is aligned with task-sensitive interface evidence.
```

中文：

```text
Full CoReMol-Net-Affinity 的 Interface-TCM 改善最大，同时 beneficial contact coverage 提升、harmful leakage 降低，说明性能提升与任务敏感界面证据的对齐改善是一致的。
```

---

## 9. Codex 执行清单

Codex 需要实现并保存以下内容。

### 9.1 Variant names

使用固定命名：

```text
ligand_only
pocket_context
intra_rir
cross_rir
full
without_interface_readout
```

### 9.2 Required outputs

```text
task_metrics.csv
affinity_ablation_summary.csv
interface_tcm_metrics.csv
intra_tcm_metrics.csv
contact_type_stats.csv
rir_affinity_aux_stats.pt or .npz
```

### 9.3 Cross-RIR auxiliary fields

对包含 Cross-RIR 的 variant，必须保存：

```text
complex_id
pdb_id
ligand_atom_index
ligand_atom_type
pocket_residue_index
residue_name
residue_id
chain_id

distance_lig_res
contact_type
C_ref_LP
alpha_ref_LP
alpha_rew_LP
Delta_alpha_LP
delta_LP
interface_flow_norm
TCM_g_LP
TCM_benefit_LP
TCM_harm_LP
```

### 9.4 Intra-RIR auxiliary fields

对包含 Intra-RIR 的 variant，必须保存：

```text
complex_id
ligand_atom_i
ligand_atom_j
shortest_path_distance
C_ref_L
alpha_ref_L
alpha_rew_L
Delta_alpha_L
delta_L
TCM_g_L
TCM_benefit_L
TCM_harm_L
```

### 9.5 Fair comparison constraints

```text
1. 所有 variant 使用相同 ligand encoder 和 pocket encoder 配置。
2. Intra-RIR 和 Cross-RIR 的 hidden dim、gate、dropout、message type 尽量保持一致。
3. Cross-RIR、Full、w/o interface readout 使用相同 cross candidate pair 和相同 C_ref^LP 构造。
4. Full 和 w/o interface readout 的唯一差别应是 affinity head 是否接收 h_interface。
5. 所有 TCM-family 指标统一用 ×10^3 单位报告。
6. 不要在本实验中加入 non-CleanSplit setting。
7. 不要在本实验文档中写 split 和 seeds。
```

---

## 10. 正文结论话术

英文版：

```text
The affinity ablation shows a clear hierarchy of evidence under PDBbind CleanSplit. Adding pocket context improves over the ligand-only model, indicating that the task cannot be explained by ligand memorization alone. Intra-RIR further improves pocket-conditioned ligand representations, but Cross-RIR yields larger gains by explicitly rewiring ligand--pocket interface interactions. The full model achieves the best performance, showing that intra-molecular and interface-level residual rewiring are complementary. Removing the interface residual-flow readout degrades performance, suggesting that the rewired interface flow is itself predictive evidence. Mechanistically, Full CoReMol-Net-Affinity achieves the largest Interface-TCM improvement, with increased beneficial contact coverage and reduced harmful leakage.
```

中文版：

```text
在 PDBbind CleanSplit 下，亲和力消融呈现出清晰的证据层级：加入 pocket context 相比 ligand-only 有提升，说明任务不能只靠 ligand 记忆；Intra-RIR 进一步改善 pocket 条件下的 ligand 表示，但 Cross-RIR 的收益更大，说明亲和力关键来自 ligand--pocket interface residual rewiring。Full 模型最好，说明分子内部重连和界面重连互补。去掉 interface residual-flow readout 后性能下降，说明 Cross-RIR 产生的 residual interface flow 本身就是亲和力预测证据。机制上，Full 模型的 Interface-TCM 最高，同时 beneficial contact coverage 提升、harmful leakage 降低。
```

---

## 11. 一句话给 Codex

```text
Run only the PDBbind CleanSplit affinity mechanism ablation with six variants: ligand_only, pocket_context, intra_rir, cross_rir, full, and without_interface_readout. The expected evidence hierarchy is: Full > Cross-RIR > Intra-RIR > Pocket context > Ligand-only, and Full should also outperform without_interface_readout. Report RMSE/Pearson/Spearman plus Interface-TCM decomposition to show that the gain comes from ligand--pocket residual interface rewiring rather than ligand-only shortcuts or simple pocket pooling.
```
