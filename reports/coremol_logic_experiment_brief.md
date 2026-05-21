# CoReMol 实现逻辑与实验初步设计汇报稿

整理日期：2026-05-18

本文基于 `BORF/CORMOL` 项目中已经落地的代码、计划文档和结果文件整理，目标是让汇报时能够清楚说明：为什么要做 CoReMol、实际代码如何实现、实验如何设计、当前结果说明了什么以及还需要注意什么。

## 1. 一句话概括

CoReMol 的核心思路是：在不改变分子真实共价键图的前提下，给已有分子编码器加一个任务条件化的 residual communication adapter，让模型根据当前分子和任务上下文，对原本的原子间通信分配进行有符号校准。通信不足的 atom pair 被增强，冗余或可能有害的 atom pair 被抑制。

当前项目中已经实现了两类 backbone：

- `AttentiveFP + CoReMol`：主线实验和大部分结果都基于它。
- `Graphformer-lite + CoReMol`：作为 backbone-agnostic 扩展，用于验证 adapter 不是只绑定 AttentiveFP。

## 2. 研究动机

普通 GNN 或分子 Transformer 会把通信资源主要分配给共价邻接、短程路径、attention 偏好的局部结构。这个设计对分子建模很自然，但在性质预测任务里，真正决定标签的证据可能不是“结构上最近”的 pair，而是“对当前任务有用”的 pair。

项目文档把这个问题定义为 **Task-Communication Misallocation**，即“任务通信分配失配”：

- Under-allocation：任务相关 atom pair 通信不足。例如远端疏水结构和极性原子共同影响 BBBP，但局部 message passing 不一定给它们足够通信。
- Over-allocation：某些近邻或高 attention pair 结构上容易被通信，但对当前任务贡献有限，甚至会引入噪声。

因此 CoReMol 不是单纯增加连接，也不是普通 residual connection。它要解决的是：

> 已有 backbone 的通信支持 `C(i,j)` 和当前任务真正需要的通信需求 `D(i,j | c_mol)` 之间是否匹配。

用公式表达就是学习一个 signed residual：

```text
S(i,j) = D(i,j | c_mol) - C(i,j)
```

如果 `S > 0`，说明需求高于当前支持，应该增强通信；如果 `S < 0`，说明当前支持可能过量，应该抑制通信。

## 3. 实际实现的核心逻辑

### 3.1 候选 atom pair

候选 pair 由 `coremol/modules/pair_features.py` 生成：

- 先在原始共价键图上计算最短拓扑距离。
- 默认保留 `1 <= d(i,j) <= d_max` 的有向 pair。
- 默认允许 bond pair 也被校准，因为 signed calibration 既可以增强非键长程通信，也可以抑制已有键上的冗余通信。

早期设计里 `d_max=4`，后续受实验诊断影响，阶段性较优配置倾向 `d_max=2`，因为它更聚焦局部但非完全一跳的有用通信带。

### 3.2 Base communication support

`coremol/modules/base_support.py` 实现有限跳结构支持：

```text
C = normalize(sum_{l=1..K} P_A^l / K)
```

其中 `P_A` 是行归一化的共价邻接转移矩阵，`support_hops` 默认是 3，后续部分实验使用 2。这个 `C(i,j)` 表示原始结构在有限跳传播中天然给 pair `(i,j)` 的通信支持强度，范围归一化到 `[0, 1]`。

这里的关键点是：CoReMol 没有把 residual pair 当作新的化学键，而是把 `C` 当作“当前结构通信基线”。

### 3.3 Task-conditioned demand network

`coremol/modules/coremol_adapter.py` 中的 `CoReMolResidualAdapter` 是核心实现。对每个候选 pair 构造特征：

```text
z_ij = [
  h_i,
  h_j,
  |h_i - h_j|,
  h_i * h_j,
  c_mol,
  distance(i,j) / d_max,
  C(i,j),
  is_bond_pair
]
```

其中 `c_mol` 是当前图内 atom state 的 mean pooling。然后用一个小 MLP 输出需求：

```text
D(i,j | c_mol) = sigmoid(MLP(z_ij))
```

这让需求估计同时依赖局部 pair 表示、分子整体上下文和基础结构支持。

### 3.4 Signed residual calibration

实现中先计算：

```text
S = D - C
delta = beta * tanh(S / tau)
```

然后不是直接使用负边权，而是把 `delta` 加到 base communication logit 上：

```text
base_logits = log(C)
cal_logits = base_logits + delta
alpha_base = softmax(base_logits, grouped_by_source_atom)
alpha_cal  = softmax(cal_logits, grouped_by_source_atom)
```

这样可以保持通信权重仍然是合法的 softmax 分布，同时通过 `alpha_cal - alpha_base` 表达校准方向。

### 3.5 Residual update

节点更新形式是：

```text
Delta h_i = sum_j (alpha_cal_ij - alpha_base_ij) * message(i,j)
h'_i = h_i + gate * Dropout(Norm(Delta h_i))
```

`message(i,j)` 有两种实现：

- `value`：直接使用 `W h_j`。
- `delta`：使用 `W h_j - W h_i`。

后续结果和失败分析显示，`delta` message 更稳，因为它减少了绝对邻居值注入导致的 representation drift。项目的阶段性最终候选更偏向 `residual_message="delta"`。

`gate` 支持 scalar 或 channel 两种模式，并可通过 `residual_gate_max` 做幅度约束。实验中也支持冻结 backbone 或 atom encoder，只训练 residual adapter 和读出头。

### 3.6 Backbone 接入方式

`coremol/models/attentivefp_coremol.py`：

- 先用 AttentiveFP 在原始 bond graph 上编码 atom state。
- CoReMol adapter 可以在 `post`、`layerwise` 或 `both` 位置插入。
- 默认主线是 `post`：先完整编码，再做 residual communication calibration，再 readout。

`coremol/models/graphformer_coremol.py`：

- 实现了一个轻量 Graphformer backbone。
- 包含 node projection、可选 local GINE、最短路 distance bias、多头 attention、mean/max 或 graph token readout。
- 复用同一个 `CoReMolResidualAdapter`，说明 adapter 只依赖 `atoms / edge_index / batch`，不强绑定 AttentiveFP。

## 4. 训练与评估流水线

主入口是 `scripts/run_stage1_gate.py`。它按 dataset、seed、variant 循环执行：

1. 用 PyG `MoleculeNet` 加载数据，任务配置在 `coremol/datasets/moleculenet.py`。
2. 使用 scaffold split 或 CurvFlow-style random split，split 会落盘到 `data/splits`。
3. 在同一 split、同一 seed 下训练 `base` 和 `coremol` 两个 variant。
4. 支持 fixed base checkpoint：先固定 base，再从同一个 base warm-start CoReMol，避免 baseline 和 CoReMol 初始化不公平。
5. 分类任务用 masked BCEWithLogitsLoss，并支持 multi-task 缺失标签；回归任务可做 target normalization。
6. 训练后保存 `raw_metrics.csv`、`summary_metrics.csv`、`mechanism_metrics.csv` 和 checkpoint。

当前 runner 支持的关键控制项：

- backbone：`attentivefp` 或 `graphformer`
- split：`scaffold`、`curvflow_random`、`random`
- residual scope：`d_max`、`support_hops`
- residual strength：`beta`、`tau`、`residual_gate_init`
- residual form：`value` 或 `delta`
- training protocol：warm start、fixed base、freeze backbone、freeze atom encoder、backbone lr scale
- mechanism evaluation：`tcm_graphs`、`tcm_k`

## 5. 机制指标设计

项目没有只看任务指标，而是同时输出机制指标。

### 5.1 EnhanceRatio / SuppressRatio

`coremol/metrics/mechanism.py` 统计：

- `enhance_ratio`：`S > 0` 的 pair 是否真的满足 `alpha_cal > alpha_base`。
- `suppress_ratio`：`S < 0` 的 pair 是否真的满足 `alpha_cal < alpha_base`。
- `mismatch_reduction`：校准后 `D` 与通信分布之间的加权 mismatch 是否下降。
- `calibration_contrast`：正负 residual pair 的校准分布是否有区分。
- `update_atom_norm_ratio`：residual update 相对 atom norm 的大小，帮助诊断 residual 分支是否扰动过大。

### 5.2 TCM：独立任务证据诊断

`coremol_tcm_metric_design.md` 和 `coremol/probes/tcm.py` 里定义了 TCM。核心原则是：任务证据不能来自 CoReMol 自己，否则会变成循环论证。

所以项目使用 frozen base model 的 counterfactual pair sensitivity：

1. 训练并冻结 base model。
2. 对候选 pair `(i,j)` 做很小的 probe communication：`h_i <- h_i + epsilon * h_j`。
3. 观察 probe 前后 loss 变化。
4. loss 下降说明该 pair 是有益通信证据，loss 上升说明可能是有害或冗余通信。

TCM 评价的是通信分布 `q` 是否覆盖有益 pair、避开有害 pair。代码里有两类：

- `TCM@K`：Top-K 有益覆盖和有害泄漏，便于快速诊断。
- `Full TCM`：全分布版本，更稳定，后续报告中被建议作为主机制指标。

需要注意：TCM 默认是 evaluation/probe，不是主训练监督目标。部分实验配置使用 `residual_aux_weight` 做轻量 self-alignment，但不是把 TCM probe 直接当监督标签。

## 6. 初步实验设计

### 6.1 Stage-1 gate

最初的 gate 设计见 `docs/superpowers/specs/2026-05-13-coremol-stage1-gate-design.md`，目标不是刷榜，而是快速验证“任务性能 + 机制指标”是否同时改善。

基本设置：

| 项目 | 设计 |
| --- | --- |
| 数据集 | BBBP 分类、ESOL 回归 |
| Backbone | AttentiveFP |
| 对比 | Base AttentiveFP vs Full CoReMol |
| Split | scaffold 8/1/1 |
| Seeds | 0, 1, 2 |
| 任务指标 | BBBP ROC-AUC 越高越好；ESOL RMSE 越低越好 |
| 机制指标 | TCM@K、Full TCM、EnhanceRatio、SuppressRatio、MismatchReduction |
| 通过条件 | 至少 2/3 seeds 任务指标和 TCM 同方向改善 |

这个 gate 的初始结果是：BBBP 和 ESOL 的任务性能都提升，但 ESOL 的 TCM@10 没有同步提升，因此 gate 被判定为机制未完全通过。这个失败结果推动了后续机制内调参，而不是引入无关模块。

### 6.2 机制内调整路线

后续调整都围绕主机制：

- 从 `d_max=4` 收缩到 `d_max=2`，减少冗余长程 pair。
- 从 `value` message 转向 `delta` message，降低绝对邻居值注入带来的表示漂移。
- 使用 fixed-base/warm-start，让 CoReMol 从相同 base 出发，避免 baseline 不公平。
- 对强 baseline 场景冻结 atom encoder 或 backbone，只训练 adapter 和 readout/head。
- 调整 `beta`、`tau`、gate 初始化，控制 residual 分支强度。

其中 `results/coremol_final_report.md` 总结的阶段性候选是：

```text
d_max = 2
residual_message = delta
beta = 0.65
tau = 0.25
fixed base + warm-start
冻结 atom encoder，训练 CoReMol adapter + readout/head
Full TCM 作为主解释指标，TCM@10 作为稀疏 top-pair 诊断
```

### 6.3 扩展实验设计

扩展实验分成三条线：

1. 分类任务：BBBP、BACE、ClinTox、Tox21、HIV、SIDER、ToxCast，主要看 ROC-AUC。
2. 回归任务：ESOL、FreeSolv、Lipo，主要看 RMSE。
3. Backbone 扩展：Graphformer-lite，用于验证 adapter 是否能迁移到非 AttentiveFP backbone。

实验设计中最重要的公平性控制：

- 同一 dataset/seed/split 下做 paired comparison。
- baseline checkpoint 可以固定复用，CoReMol 从同一个 base warm-start。
- TCM probe 基于 frozen base model，减少解释指标被 CoReMol 自证的风险。
- split 选择要和汇报 claim 绑定：scaffold 更严格，random 可能更容易，需要明确说明。
- seed screening 的结果只能作为阶段性探索，正式汇报要明确 selected seeds 的来源。

## 7. 当前已有结果摘录

### 7.1 初始 Stage-1 gate

来源：`results/stage1_gate_standard_split` 和 `reports/stage1_gate_report.md`。

| Dataset | Base | CoReMol | 任务变化 | TCM 结论 |
| --- | ---: | ---: | ---: | --- |
| BBBP ROC-AUC | 0.6264 ± 0.0271 | 0.6929 ± 0.0327 | +0.0665 | TCM@10 2/3 seeds 改善；Full TCM 3/3 改善 |
| ESOL RMSE | 1.6357 ± 0.0894 | 1.4174 ± 0.1120 | -0.2183 | TCM@10 0/3 seeds 改善；Full TCM 3/3 改善 |

解读：任务性能有效，但 Top-K 机制指标不稳定，因此不能只凭初始结果宣称机制完全成立。

### 7.2 修正后的控制结果

来源：`results/coremol_final_report.md`。

| Dataset | Protocol | Base | CoReMol | 改善 | Full TCM | TCM@10 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| ESOL RMSE | scaffold, strong ZINC-pretrained fixed base | 0.8854 | 0.8702 | -0.0151 | 3/3 seeds 改善 | 3/3 seeds 改善 |
| BBBP ROC-AUC | scaffold fixed base | 0.7131 | 0.7241 | +0.0110 | 3/3 seeds 改善 | 1/3 seeds 改善 |

解读：修正后 ESOL 同时满足任务和机制指标；BBBP 的 Full TCM 稳定，但 TCM@10 仍不稳定，说明 sparse top-pair 诊断更敏感。

### 7.3 多任务扩展结果

来源：`results/curvflow_classification_sweep/classification_7datasets_best_3seed_summary.md`。

| Dataset | Base AUC | CoReMol AUC | CoReMol - Base |
| --- | ---: | ---: | ---: |
| BBBP | 0.6264 ± 0.0271 | 0.6929 ± 0.0327 | +0.0665 |
| BACE | 0.7740 ± 0.0094 | 0.8025 ± 0.0247 | +0.0284 |
| ClinTox | 0.8537 ± 0.0549 | 0.9069 ± 0.0214 | +0.0532 |
| Tox21 | 0.8266 ± 0.0181 | 0.8281 ± 0.0178 | +0.0015 |
| HIV | 0.8100 ± 0.0292 | 0.8256 ± 0.0141 | +0.0156 |
| SIDER | 0.5883 ± 0.0567 | 0.6272 ± 0.0018 | +0.0388 |
| ToxCast | 0.6618 ± 0.0056 | 0.6592 ± 0.0069 | -0.0025 |

ToxCast 是当前弱项：CoReMol 超过论文 AttentiveFP baseline，但略低于项目内 paired baseline。

来源：`results/regression_curvflow_3datasets_best_3seed_summary.md`。

| Dataset | Base RMSE | CoReMol RMSE | CoReMol - Base |
| --- | ---: | ---: | ---: |
| ESOL | 0.8885 ± 0.0101 | 0.8697 ± 0.0078 | -0.0188 |
| FreeSolv | 2.2089 ± 0.1212 | 1.7759 ± 0.1368 | -0.4329 |
| Lipo | 0.7197 ± 0.0458 | 0.6756 ± 0.0610 | -0.0441 |

这些结果支持 CoReMol 在多个 MoleculeNet 任务上有正向趋势，但其中包含筛选 seed 和不同 split/protocol，正式论文或组会报告中必须把 protocol 讲清楚。

## 8. 汇报时建议强调的主线

建议按下面逻辑讲：

1. 现有分子编码器的问题不是“没有更多边”，而是通信分配和任务证据可能失配。
2. CoReMol 保留真实共价图，只在 hidden-state 通信层做 residual calibration。
3. 方法核心是 `D - C`：任务需求减去结构支持，得到有符号通信残差。
4. 实现上通过 logit calibration 改变 softmax 通信分布，避免直接使用负边权。
5. 实验不是只看 AUC/RMSE，还用 frozen base counterfactual probe 做独立机制诊断。
6. 初始 gate 暴露出 Top-K TCM 不稳定，因此后续收缩 pair scope、改用 delta message、引入 fixed-base/warm-start，这些调整都没有偏离主机制。
7. 当前结果表明 Full TCM 和任务性能整体更稳定，但 TCM@10 仍是需要继续优化的风险点。

## 9. 当前风险与下一步

当前最主要的风险：

- TCM@10 不如 Full TCM 稳定，说明最关键 top-pair 的稀疏校准仍可能失败。
- 多任务结果里有 seed screening 和 split 差异，汇报时不能混成一个统一 SOTA claim。
- ToxCast 对项目内 baseline 没有提升，说明方法不是所有任务都稳定增益。
- 早期计划中的 RandomCalib、UnsignedGate、PositiveOnly 等显式 ablation 尚未作为统一 variant 在 runner 中完整落地；当前更多是通过 `d_max`、`message`、`gate`、freeze/warm-start 等机制内 ablation 来验证。

下一步建议：

1. 固定一套正式协议：每个 dataset 明确 split、seed、baseline checkpoint 来源。
2. 完整补上 ablation：Full CoReMol vs RandomCalib vs UnsignedGate vs PositiveOnly vs w/o context vs w/o support。
3. 对 TCM@10 失败样本做 case study，查看 top harmful leakage 来自哪类 pair。
4. 将 Graphformer-lite 结果整理成单独表，验证 backbone-agnostic claim。
5. 如果面向论文，主表只放 protocol 最干净的 paired comparison，筛选实验放 appendix 或 exploration section。

## 10. 关键文件索引

| 内容 | 文件 |
| --- | --- |
| 项目 README | `README.md` |
| Stage-1 算法计划 | `coremol_stage1_algorithm_experiment_plan.md` |
| TCM 指标设计 | `coremol_tcm_metric_design.md` |
| Pair 构造 | `coremol/modules/pair_features.py` |
| Finite-hop support | `coremol/modules/base_support.py` |
| CoReMol adapter | `coremol/modules/coremol_adapter.py` |
| AttentiveFP 接入 | `coremol/models/attentivefp_coremol.py` |
| Graphformer 接入 | `coremol/models/graphformer_coremol.py` |
| 训练主入口 | `scripts/run_stage1_gate.py` |
| TCM probe | `coremol/probes/tcm.py` |
| 机制指标 | `coremol/metrics/mechanism.py` |
| 分类汇总结果 | `results/curvflow_classification_sweep/classification_7datasets_best_3seed_summary.md` |
| 回归汇总结果 | `results/regression_curvflow_3datasets_best_3seed_summary.md` |
| 阶段性控制结果 | `results/coremol_final_report.md` |
