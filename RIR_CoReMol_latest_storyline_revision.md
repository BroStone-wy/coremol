# Residual Interaction Rewiring（RIR / CoReMol-RIR）最新修正版要点与故事线

整理日期：2026-05-20  
目标论文题目建议：**Residual Interaction Rewiring for Molecular Graph Learning**

---

## 0. 一句话总结

我们讨论后的最新稳定版本，不再把 CoReMol 强行解释为“真实任务需求 `D` 减去已有通信支持 `C`”。更稳的解释是：

> **用 TCM 诊断结构偏置交互分配是否对齐任务敏感 pair；再提出 Residual Interaction Rewiring，在结构参照交互分布上学习有符号 residual shift，只把重连前后的净变化注入 hidden representation。**

这比原始 `D-C` 版本更稳健，因为它避免了两个核心审稿风险：

1. `D` 是 MLP 学出来的分数，并没有真实监督，不能强说成“真实任务需求”；
2. `C` 是结构参照，不是所有 backbone 的真实通信分布。

---

## 1. 原 md 文件中的主线

原 md 的 CoReMol 主线可以概括为：

> CoReMol 在不改变分子真实共价图的前提下，给已有分子编码器加一个任务条件化 residual communication adapter。模型学习 `D(i,j | c_mol)`，再减去结构支持 `C(i,j)`，得到 signed residual `S = D - C`，然后通过 `alpha_cal - alpha_base` 校准通信分布。

原公式大致为：

```text
C = normalize(sum_l P_A^l / K)

D(i,j | c_mol) = sigmoid(MLP(z_ij))

S = D - C

delta = beta * tanh(S / tau)

alpha_base = softmax(log C)

alpha_cal = softmax(log C + delta)

Delta h_i = sum_j (alpha_cal_ij - alpha_base_ij) * message(i,j)

h'_i = h_i + gate * Delta h_i
```

原版本优点是直观：

```text
任务需要多少 - 结构已经支持多少 = 通信缺口
```

但它存在几个潜在逻辑风险：

- `D` 没有真实需求标签，只是模型学出来的 pair score；
- `C` 是结构支持，不一定是 backbone 的真实通信分布；
- Graph Transformer、GraphGPS、CheapNet 这类模型的实际通信不一定等于 `C`；
- `D-C` 需要假设 `D` 和 `C` 语义同尺度，这个假设偏启发式。

---

## 2. 最新核心定位：Residual Interaction Rewiring

建议论文方法总名为：

```text
Residual Interaction Rewiring, RIR
```

也可以在工程里叫：

```text
CoReMol-RIR
```

最新定义：

> RIR 不修改真实分子键，也不假设 backbone 的真实通信等于结构参照。它在 backbone 表示之外增加一条结构参照的 residual interaction branch。该分支以分子结构诱导的参考交互分布为零点，学习有符号残差偏移，并只把重连前后交互流的净变化注入 hidden representation。

这一定义可以解决之前的争议：

- 不说 backbone 一定按 `C` 通信；
- 不说 `D` 是真实任务需求；
- 不把方法讲成普通 attention；
- 不把真实化学键当成被重连对象。

---

## 3. 最新动机：不是“backbone 不懂任务”，而是“结构先验不总是任务证据”

原始说法容易被误解为：

```text
Backbone 不会学习任务，所以需要 CoReMol 帮它学习任务通信。
```

这个不对。因为不加残差时，backbone 的表示 `h` 当然也会受到 task loss 调制。

最新更稳的动机是：

> 分子 backbone 本身会在任务监督下学习任务自适应表示。但这种自适应受其内置通信机制影响，而这些机制通常带有结构偏置，例如共价拓扑、局部邻域、结构距离或 attention 偏好的交互模式。这些结构先验具有化学意义，但不一定总是与任务敏感 pairwise evidence 完全一致。

更白话地说：

```text
分子模型不能完全脱离结构。
共价键、拓扑距离、空间接触是非常重要的化学先验。
但结构上容易通信的 pair，不一定就是当前任务最需要的 pair。
有些任务相关 pair 可能被结构先验低估；
有些结构上强连接的 pair 可能对当前任务冗余甚至有害。
```

所以真正的领域问题是：

> **如何在保留分子结构先验的同时，让模型能够学习任务相关的交互偏移？**

RIR 的回答是：

> 不从零学习自由 attention，而是在结构参照交互分布上学习有符号残差偏移。

---

## 4. `h` 的最新解释

不要再把原始 `h` 说成“纯结构先验表示”。更准确是：

```text
h = structure-biased, task-trained representation
```

中文解释：

> `h` 是受任务 loss 训练过的表示，但它的形成过程仍然受到 backbone 通信机制和结构归纳偏置影响。

因此：

- RIR 不是让 `h` 第一次变得任务相关；
- RIR 是在一个已经任务训练过的 `h` 上，额外加一条显式、可控、可诊断的 residual interaction update。

---

## 5. `C` 的最新解释：从 `C` 改为 `C_ref`

原 md 中 `C` 被解释为 finite-hop base communication support：

```text
C = normalize(sum_l P_A^l / K)
```

最新建议改名为：

```text
C_ref(i,j): structure-induced reference interaction support
```

中文：

> 结构诱导的参考交互支持。

它不是：

```text
backbone 的真实通信分布
```

也不是：

```text
h 的来源
```

而是：

```text
residual interaction branch 的结构参照零点
```

这样一来，Graph Transformer、CheapNet、GraphGPS-like backbone 都能兼容。因为 RIR 不要求这些模型的实际通信等于 `C_ref`，只要求 `C_ref` 作为 residual branch 的结构参照。

---

## 6. `D` 不再作为主公式核心

原 md 中：

```text
D(i,j | c_mol) = sigmoid(MLP(z_ij))
S = D - C
```

并把 `D` 称为 task-conditioned demand。

最新建议：

> 不再把 `D` 作为最终主公式核心。直接学习 residual shift。

原因：

1. `D` 没有真实监督标签；
2. `D` 只是 MLP 输出分数，不能强说成真实任务需求；
3. `D-C` 需要假设 `D` 和 `C` 同尺度；
4. 论文题目是 Residual Interaction Rewiring，直接学习 residual shift 更贴合。

最新主公式：

```text
r_ij = MLP(z_ij)

delta_ij = beta * tanh(r_ij)
```

其中：

```text
r_ij / delta_ij
```

表示：

> 当前状态下，pair `(i,j)` 相对于结构参照应上调还是下调。

如果后续仍保留旧实现，也建议在论文里把它解释为：

```text
D-C_ref parameterizes a residual shift.
```

而不是：

```text
D is the true task demand.
```

---

## 7. 最新推荐方法公式

推荐主公式如下：

```text
C_ref(i,j): structure-induced reference interaction support

alpha_ref_ij = softmax_j(log(C_ref(i,j) + eps))

z_ij = [
  h_i,
  h_j,
  |h_i - h_j|,
  h_i * h_j,
  c_mol,
  distance(i,j),
  C_ref(i,j),
  is_bond_pair
]

r_ij = MLP(z_ij)

delta_ij = beta * tanh(r_ij)

alpha_rew_ij = softmax_j(log(C_ref(i,j) + eps) + delta_ij)

Delta h_i = sum_j (alpha_rew_ij - alpha_ref_ij) * m_ij

h'_i = h_i + gate * Dropout(Norm(Delta h_i))
```

其中建议：

```text
m_ij = W h_j - W h_i
```

因为 `delta message` 比直接注入 `W h_j` 更稳定，可以减少 representation drift。

---

## 8. `alpha_rew - alpha_ref` 的最新解释

这一项最容易被误解成：

```text
先加了 delta，又减回去了。
```

最新解释必须写清楚：

```text
alpha_ref = structure-reference interaction distribution
alpha_rew = rewired interaction distribution
alpha_rew - alpha_ref = residual interaction flow
```

也就是说：

> `alpha_ref` 和 `alpha_rew` 都属于 residual branch 内部。  
> `alpha_ref` 是结构参照交互流。  
> `alpha_rew` 是加上 residual shift 后的重连交互流。  
> 二者差值是重连前后的净变化。

它不是从 backbone 里扣掉已经发生过的消息。

更严谨的写法是：

```text
Delta h_i = E_{j ~ alpha_rew}[m_ij] - E_{j ~ alpha_ref}[m_ij]
```

中文：

> `Delta h_i` 是重连交互分布下的消息期望，减去结构参照分布下的消息期望。

它体现的是 residual：只注入变化量，而不是再完整加一层 attention/message passing。

---

## 9. 为什么要额外加 residual branch？

这是最新故事中最重要的动机点。

不能说：

```text
backbone 不会学习任务，所以要加 residual branch。
```

应该说：

> Backbone 会学习任务表示，但它的交互适配是隐式的、混在整体表示里的。RIR 把“哪些 pair 相对于结构先验应该增强或抑制”显式建模成一条 residual interaction branch。

额外分支的价值有三点：

### 9.1 显式化

普通 backbone 的 `h` 很难解释哪些 pair 被增强、哪些 pair 被抑制。RIR 通过 `alpha_rew - alpha_ref` 明确给出结构参照下的交互偏移。

### 9.2 可控

如果没有学到可靠偏移：

```text
delta = 0
alpha_rew = alpha_ref
Delta h = 0
```

模块退化为近似恒等，不会默认扰动 backbone。

### 9.3 可诊断

RIR 产生明确的 interaction distribution，可以和 TCM、beneficial coverage、harmful leakage 对应，验证 residual shift 是否更贴近任务敏感 pair。

一句话：

> RIR 不是多加一层网络，而是增加一条结构参照、可控、可诊断的 residual interaction branch。

---

## 10. TCM 在最新故事线中的位置

原 md 中 TCM 是 frozen base counterfactual pair sensitivity：冻结 base model，对候选 pair 做小扰动：

```text
h_i <- h_i + epsilon * h_j
```

观察 loss 变化：

- loss 下降：pair 是有益通信证据；
- loss 上升：pair 可能有害或冗余。

这个设计保留，但故事位置要前移：

> TCM 先作为问题诊断工具出现，而不是最后才作为解释指标。

最新故事线：

```text
结构先验重要
↓
但结构先验不一定等于任务敏感 pair
↓
用 TCM 测现有交互分布是否覆盖任务敏感 pair、是否泄漏到有害 pair
↓
发现 measurable misalignment
↓
提出 RIR 学习结构参照下的 residual shift
```

TCM 高低解释：

```text
TCM 高：
通信分布覆盖更多 beneficial pair，泄漏到 harmful pair 更少。

TCM 低：
通信分布没有覆盖任务敏感 pair，或者把权重放到了有害/冗余 pair 上。
```

---

## 11. TCM 计算建议

设 frozen base 原始 loss 为：

```text
L0
```

对 pair `(i,j)` 加 probe communication 后 loss 为：

```text
Lij
```

定义：

```text
g_ij = L0 - Lij
```

含义：

```text
g_ij > 0:
加这个 pair 通信后 loss 下降，是 beneficial evidence。

g_ij < 0:
加这个 pair 通信后 loss 上升，是 harmful/redundant evidence。
```

然后定义：

```text
b_ij = max(g_ij, 0)

h_ij = max(-g_ij, 0)
```

对通信分布 `q_ij`，建议主文用归一化 Full TCM：

```text
Benefit(q) =
sum_ij q_ij b_ij / (sum_ij b_ij + eps)

Harm(q) =
sum_ij q_ij h_ij / (sum_ij h_ij + eps)

TCM_norm(q) = Benefit(q) - lambda * Harm(q)
```

主文报告：

```text
Delta TCM = TCM_norm(q_RIR) - TCM_norm(q_base)
```

重点：

> 主文的 TCM 改善应报 RIR/CoReMol 相比 baseline 的 paired improvement。

`TCM(alpha_rew) - TCM(alpha_ref)` 只能作为模块内部分析，用来说明 residual shift 是否把结构参照推向任务敏感 pair。

---

## 12. TCM 和任务指标的关系

不能要求：

```text
任务指标提升 ⇔ TCM 必然提升
```

它们相关，但不等价。

原因：

- 任务指标是最终预测性能；
- TCM 是 frozen-base 表示空间里的 pair communication diagnostic；
- 模型可能靠 readout、正则化、整体表示改善而提升任务指标；
- TCM 也可能提升但 residual update 扰动了整体表示，导致任务指标不升。

PR 主文建议：

> 任务指标是第一证据，TCM 是机制证据。最理想是二者同向；如果 TCM@K 不稳定，优先看 Full TCM、beneficial coverage、harmful leakage 和 seed-level consistency。

---

## 13. 与原 md 相比，最新版本更稳的地方

### 13.1 避免 `D` 被质疑

原 md：

```text
D = task demand
S = D - C
```

最新：

```text
r_ij = residual shift
```

不再声称 MLP 输出是真实任务需求。

### 13.2 避免 `C` 被误解为 backbone actual communication

原 md 中 `C` 容易被理解为已有 backbone 通信支持。

最新：

```text
C_ref = residual branch 的 structure reference
```

这样兼容 GNN、Graph Transformer、CheapNet、GraphGPS-like backbone。

### 13.3 避免“先加又减”的误解

最新明确：

```text
alpha_rew - alpha_ref 是 residual branch 内部的两条交互流之差
```

不是从 backbone 中扣消息。

### 13.4 动机更像领域通用问题

原 md 是：

```text
通信分配和任务证据失配。
```

最新更稳：

```text
分子模型天然依赖结构先验，但结构先验不总是任务敏感交互。
TCM 诊断这种错位，RIR 学习结构参照下的残差交互偏移。
```

### 13.5 失败情况更好解释

最新版本明确：

> RIR 只有在任务相关交互相对于结构参照存在可学习偏移、且候选 pair 覆盖了这些交互时更可能提升。

如果 backbone 已经很强、任务主要依赖局部结构、候选 pair 不合适或 residual strength 太大，不提升是合理边界。

---

## 14. 必须补充的消融实验

原 md 已经提到 RandomCalib、UnsignedGate、PositiveOnly、w/o context、w/o support 等 ablation。

最新版本还应补充：

```text
1. Full RIR:
   Delta h = sum (alpha_rew - alpha_ref) m

2. Direct normalized delta:
   Delta h = sum normalize(delta) m

3. Direct alpha_rew:
   Delta h = sum alpha_rew m

4. No reference:
   alpha = softmax(delta)

5. Uniform reference:
   alpha_ref = uniform

6. Random reference:
   C_ref randomly permuted

7. Legacy D-C:
   delta = beta * tanh((D-C_ref)/tau)
```

这些消融要回答：

> 提升到底来自多加一层 attention，还是来自结构参照下的 residual rewiring？

---

## 15. 最新 PR 故事线

### Step 1：结构先验是分子图学习的基础

分子 backbone 通常依赖共价拓扑、局部邻域、结构距离或 attention 偏置组织信息流。

### Step 2：但结构先验不总是任务证据

结构上近的 pair 不一定对当前任务最敏感；结构上弱的 pair 可能携带关键任务信息。

### Step 3：提出 TCM 作为诊断

冻结 base model，对候选 pair 做小通信扰动，用 loss 变化识别 beneficial / harmful pair。TCM 评价模型交互分布是否覆盖 beneficial pair、避免 harmful leakage。

### Step 4：TCM 暴露结构参照与任务敏感 pair 的错位

如果 baseline 的 TCM 低，说明默认交互分配没有充分对齐任务敏感通信证据。

### Step 5：提出 RIR

RIR 不修改真实化学键，不从零学习自由 attention，而是在结构参考交互分布上学习 residual shift：

```text
alpha_ref = softmax(log C_ref)

delta = beta * tanh(MLP(z_ij))

alpha_rew = softmax(log C_ref + delta)

Delta h = sum (alpha_rew - alpha_ref) m_ij
```

### Step 6：实验验证两件事

任务上：

```text
Base backbone vs Base + RIR
```

机制上：

```text
Delta TCM_norm = TCM(q_RIR) - TCM(q_base)
BenefitCoverage ↑
HarmfulLeakage ↓
```

再用 ablation 证明不是普通 attention。

---

## 16. 可放进论文的核心英文段落

> Molecular graph backbones already learn task-adaptive representations under supervision, but this adaptation is mediated by built-in communication mechanisms and structural priors such as covalent topology, local neighborhoods, or distance-based biases. These priors are chemically meaningful, yet they do not always coincide with task-sensitive pairwise evidence. We first introduce TCM, a frozen-base counterfactual diagnostic, to examine whether an interaction distribution covers beneficial communication pairs and avoids harmful leakage. Motivated by the observed mismatch, we propose Residual Interaction Rewiring. RIR preserves the molecular bond graph and learns a signed residual shift over a structure-induced reference interaction distribution. It injects only the difference between the rewired and reference interaction flows into the hidden representation, providing a controlled and interpretable residual branch for task-adaptive molecular communication.

---

## 17. 可放进论文的核心中文表述

> 分子图 backbone 本身会在任务监督下学习任务自适应表示，但这种自适应受内置通信机制和结构先验影响，例如共价拓扑、局部邻域或距离偏置。这些先验具有化学意义，却不总是与任务敏感 pairwise evidence 完全一致。我们首先引入 TCM 作为 frozen-base 反事实诊断，用来检查交互分布是否覆盖有益通信 pair，并避免泄漏到有害 pair。基于这种可测量错位，我们提出 Residual Interaction Rewiring。RIR 保留真实分子键图，在结构诱导的参考交互分布上学习有符号残差偏移，并只将重连交互流与参考交互流之间的差值注入 hidden representation，从而形成一种可控、可解释的任务自适应残差交互分支。

---

## 18. 最终一句话版

> **TCM diagnoses whether structure-biased interaction allocation aligns with task-sensitive communication evidence; RIR then learns signed residual shifts over the structural reference distribution to adapt hidden interaction flow without modifying molecular bonds.**

中文：

> **TCM 诊断结构偏置的交互分配是否对齐任务敏感通信证据；RIR 则在结构参考分布上学习有符号残差偏移，在不修改真实分子键的前提下调整 hidden interaction flow。**

---

## 19. 和原 md 的核心差异表

| 维度 | 原 md 版本 | 最新修正版 |
|---|---|---|
| 核心问题 | 任务需求 `D` 和通信支持 `C` 失配 | 结构参照交互与任务敏感 pair 存在可学习偏移 |
| `C` 含义 | base communication support / 当前结构通信基线 | `C_ref`，residual branch 的结构参照零点 |
| 是否假设 backbone 服从 `C` | 容易被误解为是 | 明确否定 |
| `D` | task-conditioned demand | 不建议做主变量，最多作为 legacy/ablation |
| 主参数 | `S = D - C` | `r_ij = MLP(z_ij)`，直接学 residual shift |
| `delta` | 由 `D-C` 得到 | 由 residual shift 直接得到 |
| `alpha_base` | base communication distribution | `alpha_ref`，结构参照分布 |
| `alpha_cal` | calibrated distribution | `alpha_rew`，重连交互分布 |
| 残差权重 | `alpha_cal - alpha_base` | `alpha_rew - alpha_ref` |
| 更新含义 | 校准已有通信分布 | 注入结构参照 residual branch 的净变化 |
| TCM | 证明通信缺口改善 | 作为独立 pair evidence probe |
| 方法边界 | 更像 adapter 校准 | 更像 structure-referenced residual interaction rewiring |
| 失败解释 | 机制未完全通过 | 可解释为任务/候选/结构参照/强 backbone 条件不满足 |
