# PRLInet：蛋白–配体亲和力预测论文故事线方案

整理日期：2026-06-29

目标期刊：Expert Systems with Applications，ESWA  
建议论文标题：**Beyond Spatial Contacts: Selective Residue–Ligand Pairwise Interaction Modeling for Protein–Ligand Affinity Prediction**

中文标题建议：**超越空间接触：用于蛋白–配体亲和力预测的选择性残基–配体成对相互作用建模**

---

## 1. 一句话定位

这篇文章应当被包装为一篇**独立的 protein–ligand binding affinity prediction 论文**，而不是 CoReMol 的亲和力应用版。

核心故事线是：

> **空间接触只是候选，不等于亲和力贡献。RPLI-Net 将 residue–ligand spatial contacts 视为高召回候选，并通过选择性残基–配体成对相互作用建模，构建面向亲和力预测的 complex representation。**

因此文章主线不是：

> 我们构建了一个 contact graph，然后用 GNN 做亲和力预测。

而是：

> 现有结构型亲和力模型常将空间接触直接作为相互作用输入，但亲和力预测真正需要的是判断哪些 residue–ligand pair 对 affinity score 有预测贡献。RPLI-Net 因此将 contact candidate construction 与 pairwise interaction contribution modeling 显式解耦：先高召回地产生候选接触，再选择性建模真正有信息量的残基–配体成对相互作用。

---

## 2. 标题与术语选择

### 2.1 推荐标题

**Beyond Spatial Contacts: Selective Residue–Ligand Pairwise Interaction Modeling for Protein–Ligand Affinity Prediction**

这个标题的优点：

1. 不使用 **Evidence**，避免与 evidential learning、evidence theory、Dempster–Shafer evidence 等已有术语冲突。
2. 不使用 **Calibrating / Calibration**，避免把文章拉回 CoReMol 的通信校准叙事。
3. 不使用 **Affinity-aware / Affinity-guided**，避免暗示 interaction module 被 affinity label 显式引导。
4. 不使用 **Task-aware / Task-guided**，避免与 CoReMol 的 task-conditioned communication 主线绑定。
5. 保留核心矛盾：**spatial contact 不等于 predictive contribution**。
6. 突出 RPLI-Net 的真正作用：**selective residue–ligand pairwise interaction modeling**。

### 2.2 文章中建议统一使用的术语

| 想表达的含义 | 推荐用词 | 避免用词 |
| --- | --- | --- |
| 空间接触只是候选 | spatial contact candidates / contact candidate set | evidence |
| 每个 pair 的预测重要性 | pair contribution score | evidence score |
| 选择性建模 | selective pairwise interaction modeling | evidence learning |
| 面向亲和力的复合物表征 | affinity-oriented complex representation | calibrated evidence |
| 可解释结果 | interaction cues / contribution patterns | evidential explanation |
| 核心问题定义 | contact–contribution gap | evidence gap |

建议全文贯穿以下词链：

```text
spatial contacts
  → contact candidates
  → pair contribution scores
  → selective interaction aggregation
  → affinity-oriented complex representation
  → binding affinity prediction
```

---

## 3. 核心问题定义：Contact–Contribution Gap

建议将文章的核心问题定义为：

> **Contact–Contribution Gap**

中文可以称为：

> **接触–贡献差距**

含义：

> Spatial contacts indicate possible physical proximity, but they do not directly indicate predictive contribution to binding affinity.

中文解释：

> 空间接触说明 residue 和 ligand atom / fragment 在几何上靠近，但并不能说明这个 pair 对亲和力预测真的有贡献。

这个概念是全文的理论钩子。它能够把文章从普通 contact graph / complex graph 方法中区分出来。

### 3.1 为什么存在 Contact–Contribution Gap

#### 3.1.1 Geometrically close but weakly informative

有些 residue–ligand pair 在空间上接近，但只是几何邻近，不一定对 binding strength 有明显贡献。

可写作：

> A close residue–ligand distance does not necessarily imply a strong predictive contribution, since many contacts are incidental, redundant, or weakly related to binding strength.

#### 3.1.2 Redundant contacts

一个 residue 可能和 ligand 的多个 atom 都在 cutoff 内，contact graph 会产生大量重复边。如果模型简单聚合这些边，容易对局部区域过度计权。

可写作：

> Dense contact graphs often contain redundant residue–atom pairs, causing the model to repeatedly aggregate similar local patterns.

#### 3.1.3 Context-dependent contacts

同一种空间接触在不同 pocket environment、ligand scaffold 和周围残基上下文中贡献不同。相互作用的预测价值不仅由距离决定，还取决于 local chemistry、ligand topology 和 global complex context。

可写作：

> The contribution of a residue–ligand contact is highly context-dependent, since the same geometric contact may have different implications under different pocket environments and ligand scaffolds.

---

## 4. 整体故事线

建议全文按照以下逻辑推进：

```text
Spatial Contacts
      ↓
Contact Candidates
      ↓
Selective Pairwise Interaction Modeling
      ↓
Affinity-oriented Complex Representation
      ↓
Binding Affinity Prediction
```

每一层的作用：

1. **Spatial Contacts**：通过距离或距离壳层从 complex structure 中提取空间邻近关系。
2. **Contact Candidates**：强调 contact 只是高召回候选，不是最终相互作用贡献。
3. **Selective Pairwise Interaction Modeling**：对每个 residue–ligand pair 进行上下文增强的贡献建模。
4. **Affinity-oriented Complex Representation**：将选择后的 pairwise interaction patterns 聚合成面向亲和力预测的复合物表征。
5. **Binding Affinity Prediction**：通过回归头输出 pKd / pKi / binding affinity。

最重要的写法是：

> The novelty is not the use of spatial contacts, but the explicit separation between contact candidate generation and selective interaction contribution modeling.

中文：

> 创新点不在于使用空间接触本身，而在于显式区分“接触候选生成”和“相互作用贡献建模”。

---

## 5. 模型总体架构

建议模型名保持为：

> **RPLI-Net**

完整定义：

> **RPLI-Net: a Selective Residue–Ligand Pairwise Interaction Modeling Network for protein–ligand affinity prediction**

架构可以分为四个核心模块：

```text
1. Pocket–Ligand Contact Candidate Constructor
2. Protein and Ligand Context Encoder
3. Residue–Ligand Pairwise Interaction Selector
4. Selective Interaction Aggregator and Affinity Readout
```

中文：

```text
1. 蛋白口袋–配体接触候选构建器
2. 蛋白与配体上下文编码器
3. 残基–配体成对相互作用选择器
4. 选择性交互聚合与亲和力预测头
```

架构图建议画成：

```text
Protein pocket structure              Ligand structure
        ↓                                   ↓
Pocket encoder                       Ligand encoder
        ↓                                   ↓
Residue representations              Ligand atom representations
        \                                   /
         \                                 /
          Spatial Contact Candidate Constructor
                    ↓
      Candidate residue–ligand pairs
                    ↓
      RPLI Pairwise Interaction Selector
                    ↓
      Pair contribution scores + pair messages
                    ↓
      Selective Interaction Aggregation
                    ↓
      Affinity-oriented complex representation
                    ↓
      Binding affinity prediction
```

图中建议把 contact candidates 画成大量浅色虚线，把 high-score / selected pairs 画成少量深色线。

图注建议：

> RPLI-Net treats spatial contacts as candidate interactions and selectively aggregates residue–ligand pairwise patterns for affinity prediction.

---

## 6. 模块设计与叙事方式

### 6.1 Spatial Contact Candidate Constructor

这一部分的核心不是宣称“距离 cutoff 就是相互作用”，而是强调它用于产生高召回候选。

核心写法：

> The contact module is designed to generate a high-recall set of residue–ligand interaction candidates, rather than to determine their final contributions.

中文：

> 接触模块的目标是高召回地产生残基–配体候选相互作用，而不是直接决定这些接触对亲和力的贡献。

形式化定义：

```text
E_pl = { (r_i, l_j) | d(r_i, l_j) ≤ δ }
```

其中：

- `r_i` 表示 pocket residue 或 residue atom；
- `l_j` 表示 ligand atom 或 fragment；
- `d(r_i, l_j)` 表示空间距离；
- `δ` 是距离阈值。

如果有多距离壳层，可以写为：

```text
b_ij = RBF(d_ij)
```

或者：

```text
b_ij = DistanceShell(d_ij)
```

推荐用词：

- candidate contact graph
- spatial contact candidate set
- contact candidates

避免写法：

- hard interaction graph
- true interaction graph
- binding evidence graph

### 6.2 Protein and Ligand Context Encoder

这一部分用来支撑一个关键判断：pair 的贡献不能只由距离决定。

核心动机：

> A residue–ligand pair cannot be judged only by distance. Its contribution depends on residue context, ligand topology, and global complex environment.

编码形式：

```text
h_i^P = f_P(r_i, P)
h_j^L = f_L(l_j, L)
```

其中：

- `h_i^P` 是 pocket residue / atom 表示；
- `h_j^L` 是 ligand atom / fragment 表示；
- `f_P` 捕获蛋白口袋局部环境；
- `f_L` 捕获配体拓扑和化学上下文。

建议强调：

1. protein encoder 捕获 pocket local environment；
2. ligand encoder 捕获 ligand topology / chemical context；
3. global complex context 捕获整体 pocket–ligand matching；
4. 后续 pairwise module 使用这些上下文来判断 contact 的预测贡献。

可写作：

> The encoders provide context-aware residue and ligand representations, allowing the following pairwise module to evaluate a contact beyond its geometric distance.

### 6.3 RPLI Pairwise Interaction Selector

这是方法核心。RPLI 不是普通 contact aggregation，而是对每个 contact candidate 构造 pair representation，并预测其 pair contribution score。

候选 pair feature 可以写成：

```text
z_ij = [
  h_i^P,
  h_j^L,
  |h_i^P - h_j^L|,
  h_i^P ⊙ h_j^L,
  φ(d_ij),
  t_i^P,
  t_j^L,
  c_complex
]
```

其中：

- `h_i^P`：residue / pocket representation；
- `h_j^L`：ligand atom / fragment representation；
- `|h_i^P - h_j^L|`：差异信息；
- `h_i^P ⊙ h_j^L`：匹配信息；
- `φ(d_ij)`：距离编码；
- `t_i^P, t_j^L`：残基类型、原子类型、化学属性；
- `c_complex`：全局 complex context。

输出 pair contribution score：

```text
s_ij = g(z_ij)
```

如果使用 softmax attention：

```text
α_ij = softmax_{(i,j) in E_pl}(s_ij)
```

如果使用 gate：

```text
α_ij = sigmoid(g(z_ij))
```

建议称为：

> **pair contribution score**

不要称为：

> evidence score

第一次定义可写：

> RPLI assigns each residue–ligand contact candidate a pair contribution score, indicating its relative importance for constructing the affinity-oriented complex representation.

中文：

> RPLI 为每个残基–配体接触候选分配一个成对贡献分数，用于衡量该 pair 在构建亲和力表征时的相对重要性。

### 6.4 Selective Interaction Aggregator

pair message：

```text
m_ij = ψ(z_ij)
```

选择性聚合：

```text
h_int = Σ_{(i,j) in E_pl} α_ij m_ij
```

复合物表征：

```text
h_complex = Fuse(h_P, h_L, h_int)
```

亲和力预测：

```text
ŷ = MLP(h_complex)
```

核心写法：

> The model does not aggregate all contacts uniformly. Instead, it builds an affinity-oriented interaction representation by prioritizing informative residue–ligand pairs.

中文：

> 模型不是均匀聚合所有 contact，而是通过优先建模更有信息量的 residue–ligand pair，构建面向亲和力预测的 complex representation。

---

## 7. 文章创新点写法

建议 contribution 不要写成“提出一个 contact network”，而要围绕 Contact–Contribution Gap 和 selective pairwise modeling 展开。

### Contribution 1：提出 Contact–Contribution Gap

英文：

> We identify the contact–contribution gap in protein–ligand affinity prediction: spatial contacts provide possible interaction candidates, but they do not directly indicate their predictive contributions to binding affinity.

中文：

> 我们指出蛋白–配体亲和力预测中的接触–贡献差距：空间接触只能提供候选相互作用，并不能直接表明这些 pair 对亲和力预测的贡献。

### Contribution 2：提出 RPLI-Net

英文：

> We propose RPLI-Net, a selective residue–ligand pairwise interaction modeling framework that separates contact candidate construction from interaction contribution modeling.

中文：

> 我们提出 RPLI-Net，一个选择性残基–配体成对相互作用建模框架，将接触候选构建与相互作用贡献建模显式解耦。

### Contribution 3：设计 context-enhanced pairwise selector

英文：

> We design a context-enhanced pairwise interaction selector that evaluates each residue–ligand contact using residue context, ligand topology, geometric distance, and global complex information.

中文：

> 我们设计了上下文增强的成对相互作用选择器，综合 residue context、ligand topology、geometry 和 global complex context 来评估每个候选接触。

### Contribution 4：构建 selective interaction aggregation

英文：

> We construct an affinity-oriented complex representation through selective interaction aggregation, allowing the model to emphasize informative contacts and reduce redundant contact aggregation.

中文：

> 我们通过选择性交互聚合构建面向亲和力预测的复合物表征，使模型能够突出有信息量的接触，并减少冗余接触聚合。

---

## 8. Introduction 写作结构

### 第 1 段：任务重要性

讲 CADD、virtual screening、lead optimization、binding affinity prediction。

落点：

> Accurate binding affinity prediction requires modeling protein–ligand interactions inside the binding pocket.

### 第 2 段：现有方法依赖 contact / complex graph

写现有结构型方法通常基于空间邻近构建 contact graph 或 complex graph。

可写：

> Many structure-based methods construct protein–ligand complex graphs based on spatial proximity, where residue–ligand or atom–atom contacts are connected according to distance cutoffs.

然后指出问题：

> However, these contact graphs mainly describe geometric proximity, not necessarily predictive contribution.

### 第 3 段：提出 Contact–Contribution Gap

这是 Introduction 的关键段。

可写：

> We argue that a fundamental limitation of contact-based affinity models lies in the contact–contribution gap. Spatial contacts are high-recall candidates of possible interactions, but affinity prediction requires distinguishing which contacts are informative, redundant, or weakly related to binding strength.

### 第 4 段：提出 RPLI-Net

可写：

> To address this issue, we propose RPLI-Net, a selective residue–ligand pairwise interaction modeling network. RPLI-Net first constructs spatial contact candidates and then evaluates each residue–ligand pair under local chemical, geometric, and global complex contexts.

继续：

> Instead of treating all contacts uniformly, RPLI-Net learns pair contribution scores and aggregates selected interaction patterns into an affinity-oriented complex representation.

### 第 5 段：贡献总结

放四条 contribution：

1. Contact–Contribution Gap；
2. RPLI-Net framework；
3. context-enhanced pairwise selector；
4. selective interaction aggregation and experimental validation。

---

## 9. Method 章节建议结构

```text
3. Method

3.1 Problem Formulation

3.2 Spatial Contact Candidate Construction
    - pocket residue / ligand atom candidates
    - distance cutoff or distance shells
    - contact graph as high-recall candidate set

3.3 Protein and Ligand Context Encoding
    - pocket encoder
    - ligand encoder
    - global complex context

3.4 Selective Residue–Ligand Pairwise Interaction Modeling
    - pair feature construction
    - pair contribution scoring
    - context-enhanced interaction selector

3.5 Selective Interaction Aggregation
    - weighted pair aggregation
    - residue-level or complex-level aggregation
    - affinity-oriented representation

3.6 Binding Affinity Prediction Objective
```

### 9.1 Problem formulation

输入：

```text
C = (P, L, G)
```

其中：

- `P`：protein pocket；
- `L`：ligand；
- `G`：complex geometry / spatial coordinates。

输出：

```text
ŷ = f(P, L, G)
```

其中 `y` 是 pKd / pKi / binding affinity。

### 9.2 Training objective

基础回归损失：

```text
L_aff = MSE(ŷ, y)
```

如果已经做了 ranking loss，可以写：

```text
L = L_aff + λ L_rank
```

如果没有做 ranking loss，不建议强行写进主方法。

---

## 10. 与普通 contact graph 方法的区别

审稿人可能会问：这不就是 distance cutoff + attention 吗？

建议从以下几个角度区分。

| 方法类型 | 常见做法 | RPLI-Net 的区别 |
| --- | --- | --- |
| Distance-based contact graph | 距离小于 cutoff 就连边 | 距离边只作为 candidate，不代表最终贡献 |
| Uniform contact aggregation | 所有 contact message 都聚合 | 显式学习 pair contribution score |
| 普通 attention | 学习 pair 权重，但通常没有定义 contact 与 contribution 的 gap | 以 Contact–Contribution Gap 为问题定义，设计 context-enhanced pair selector |
| Complex-level graph model | 直接在 protein–ligand complex graph 上 message passing | 先独立编码 protein / ligand，再在 contact candidates 上做 selective pairwise modeling |
| Interaction-type methods | 依赖固定 interaction type 或人工规则 | 不强依赖人工 interaction annotation，而是从 geometry、chemistry 和 context 中学习 pair contribution |

核心 rebuttal 句：

> The novelty is not the use of spatial contacts, but the explicit separation between contact candidate generation and selective interaction contribution modeling.

---

## 11. 实验设计要服务故事线

主实验不要只证明 RMSE / Pearson 提升，还要证明“Beyond Spatial Contacts”这个故事成立。

### 11.1 主结果表

建议指标：

| Metric | Direction | Purpose |
| --- | --- | --- |
| RMSE | lower is better | 回归误差 |
| MAE | lower is better | 绝对误差 |
| Pearson | higher is better | 线性相关性 |
| Spearman | higher is better | 排序相关性 |
| CI | higher is better | ranking consistency |

建议数据设置：

1. PDBbind refined / core split；
2. CASF-2016 scoring power；
3. scaffold split 或 ligand-cluster split；
4. cold-target / cold-protein family split，如果已有实验；
5. PDBbind → CASF transfer，如果已有实验。

### 11.2 消融实验

最核心的消融要围绕 Contact–Contribution Gap。

| Variant | 要证明什么 |
| --- | --- |
| Full RPLI-Net | 完整方法 |
| w/o RPLI selector | 选择性 pairwise modeling 是否有效 |
| Uniform contact aggregation | 不是所有 contact 平均聚合就够 |
| Distance-only contact weighting | 仅靠距离不能代表亲和力贡献 |
| Contact graph GNN | 普通 contact message passing 不如 selective modeling |
| w/o global complex context | pair contribution 需要全局上下文 |
| w/o ligand topology encoder | ligand chemical context 重要 |
| w/o pocket context encoder | residue environment 重要 |
| Random pair weights | 排除只是参数量增加带来的提升 |
| Single-shell vs multi-shell contacts | 验证 contact candidate 范围设计 |

如果实验时间有限，至少需要保留：

1. Full RPLI-Net；
2. w/o pair selector；
3. uniform contact aggregation；
4. distance-only contact weighting。

这四个是标题 **Beyond Spatial Contacts** 的最低支撑。

### 11.3 Case study

建议展示 2–4 个 complex。

每个 case 展示：

1. 原始 contact candidates 分布；
2. RPLI-Net high-score pair contribution 分布；
3. high-score pairs 是否集中在合理 pocket 区域；
4. 对比 baseline 预测错误与 RPLI-Net 修正的样本；
5. 失败样本中 high-score pair 是否受 pose noise 或 dense redundant contacts 影响。

注意用词：

建议写：

> learned pair contribution patterns provide interpretable interaction cues for affinity prediction.

避免写：

> the model discovers true physical binding energy.

---

## 12. Abstract 草稿

```text
Protein–ligand binding affinity prediction relies on effective modeling of interactions within the binding pocket. Existing structure-based methods commonly construct residue–ligand contact graphs according to spatial proximity. However, spatial contacts only indicate possible interaction candidates and do not necessarily reflect their predictive contributions to binding affinity. Many contacts can be redundant, context-dependent, or weakly related to binding strength, leading to a contact–contribution gap in affinity modeling.

To address this issue, we propose RPLI-Net, a selective residue–ligand pairwise interaction modeling framework for protein–ligand affinity prediction. RPLI-Net first constructs a high-recall set of spatial contact candidates and then evaluates each residue–ligand pair using protein context, ligand topology, geometric distance, and global complex information. The resulting pair contribution scores are used to selectively aggregate interaction patterns into an affinity-oriented complex representation.

Extensive experiments on protein–ligand affinity benchmarks demonstrate that RPLI-Net improves prediction accuracy and ranking consistency over competitive baselines. Ablation studies further show that selective pairwise interaction modeling is more effective than uniform contact aggregation or distance-only contact weighting. Case studies indicate that the learned pair contribution patterns provide interpretable interaction cues for affinity prediction.
```

---

## 13. Highlights 草稿

1. A selective residue–ligand pairwise interaction modeling framework is proposed for protein–ligand affinity prediction.
2. The contact–contribution gap is identified to distinguish spatial proximity from predictive contribution.
3. RPLI-Net separates contact candidate construction from interaction contribution modeling.
4. Context-enhanced pair contribution scores support selective interaction aggregation.
5. Experiments and ablations validate the effectiveness of going beyond raw spatial contacts.

---

## 14. 论文结构建议

```text
1. Introduction
   1.1 Importance of protein–ligand affinity prediction
   1.2 Limitations of spatial-contact-based affinity modeling
   1.3 Contact–Contribution Gap
   1.4 Proposed RPLI-Net and contributions

2. Related Work
   2.1 Traditional scoring functions
   2.2 Deep learning for binding affinity prediction
   2.3 Protein–ligand contact and interaction modeling
   2.4 Position of this work

3. Method
   3.1 Problem formulation
   3.2 Spatial contact candidate construction
   3.3 Protein and ligand context encoding
   3.4 Selective residue–ligand pairwise interaction modeling
   3.5 Selective interaction aggregation
   3.6 Affinity prediction objective

4. Experiments
   4.1 Datasets and splits
   4.2 Baselines and metrics
   4.3 Main affinity prediction results
   4.4 CASF scoring/ranking evaluation
   4.5 Generalization under scaffold/cold splits
   4.6 Ablation studies
   4.7 Case studies and interaction cues
   4.8 Efficiency analysis

5. Discussion
   5.1 Why selective pairwise modeling helps
   5.2 Failure cases
   5.3 Limitations

6. Conclusion
```

---

## 15. 与 CoReMol 的关系处理

这篇文章的故事线需要独立，不要写成：

> We extend CoReMol to protein–ligand affinity prediction.

也不要写成：

> The main novelty is residual reconnection for protein–ligand graphs.

推荐处理方式：

1. **标题、摘要、Introduction 主线中不出现 CoReMol。**
2. CoReMol 最多作为 related work 或 method implementation 中的轻量引用。
3. 文章主线始终围绕 protein–ligand affinity prediction、spatial contacts、contact–contribution gap 和 selective pairwise interaction modeling。

可放在 Related Work 最后一段：

```text
Recent molecular representation studies have explored task-conditioned communication adaptation for molecular property prediction. In contrast, our work focuses on protein–ligand affinity prediction and develops a selective residue–ligand pairwise interaction modeling framework, where spatial contacts are treated as candidate interactions rather than final predictive contributions.
```

注意不要使用以下表述：

- residual reconnection；
- graph rewiring；
- task-conditioned communication calibration；
- signed residual calibration；
- D - C；
- evidence learning；
- evidential interaction modeling。

---

## 16. 最终推荐定位

英文定位：

> RPLI-Net does not treat spatial contacts as final interaction evidence. Instead, it regards them as high-recall candidates and performs selective residue–ligand pairwise interaction modeling to construct an affinity-oriented complex representation.

中文定位：

> RPLI-Net 不把空间接触当成最终相互作用依据，而是把它们视为高召回候选，并通过选择性残基–配体成对相互作用建模，构建面向亲和力预测的复合物表征。

最终主线：

> **Beyond Spatial Contacts：亲和力预测的关键不是有没有更多 contact，而是能否从 noisy、redundant、context-dependent contact candidates 中选择性建模真正有预测贡献的 residue–ligand pairwise interaction patterns。**
