# CoReMol：Task-Communication Misallocation 指标设计与实验闭环

本文档整理 CoReMol / Paper B 中关于“已有通信分配是否对任务证据校准”的指标设计。目标是让论文不只是提出一个 signed residual communication 模块，而是先定义一个可诊断的问题，再围绕该问题提出算法和实验验证。

---

## 1. 为什么需要这个指标

Paper B 的核心研究问题不能写成：

> 现有方法没有显式建模 \(D-C\)。

这种说法只是“别人没有用我们的公式”，不是本质研究空白。更合理的问题定义应是：

> 现有分子 encoder 的通信分配可能和当前任务需要的 atom-pair 证据不一致。

这类问题可以称为：

# Task-Communication Misallocation

中文可译为：

> 任务通信分配失配

它关注的是：模型已有的通信资源是否分配到了当前任务真正需要的 atom pairs 上。

---

## 2. 问题的两种表现

### 2.1 Under-allocation：任务相关 pair 通信不足

某些 atom pair 对预测有用，但 backbone 给它们的通信支持不足。

例如，在 BBBP 中，远端疏水骨架和极性杂原子区域可能共同影响血脑屏障渗透性，但局部 GNN 主要沿共价键传播，导致这些非局部 atom-pair 交互没有足够通信支持。

### 2.2 Over-allocation：冗余 pair 被过度通信

某些 atom pair 因为共价邻接、距离近、attention 偏好或结构 mask，被模型分配了较多通信，但它们对当前任务贡献有限，甚至引入噪声。

因此，Paper B 的核心不只是“增强通信”，而是：

> 校准通信分配：该增强的增强，该抑制的抑制。

---

## 3. 指标的基本思想

我们希望定义一个指标，衡量：

\[
\text{model communication distribution}
\quad \text{是否对齐} \quad
\text{task-relevant pair evidence}
\]

也就是比较：

\[
q_{ij}
=
\text{模型已经分配给 atom pair }(i,j)\text{ 的通信强度}
\]

和

\[
p_{ij}
=
\text{独立诊断得到的 atom pair 任务证据}
\]

如果 \(q\) 和任务证据不一致，就说明存在 task-communication misallocation。

---

## 4. 关键原则：任务证据不能来自 CoReMol 自己

任务证据 \(p\) 不能直接用 CoReMol 学到的 \(D_\theta\)，否则会形成循环论证：

> 模型自己定义任务需求，再说自己对齐了任务需求。

因此，任务证据必须来自一个相对独立的诊断过程。本文建议使用：

> frozen base model 的 counterfactual pair sensitivity

即先训练一个 base model，然后冻结它，通过反事实 probe 判断哪些 atom pair 对预测更敏感。

---

## 5. 候选 atom pair 集合

对每个分子图 \(G\)，定义候选 atom pair 集合：

\[
\mathcal P_G
=
\{(i,j): i\neq j,\ d_G(i,j)\le d_{\max}\}
\]

默认设置：

- \(d_{\max}=4\)；
- 小分子也可以使用 all pairs；
- 排除 self-loop；
- 是否包含原始 bond pair 可作为超参数；
- 第一阶段建议包含 bond pairs，因为 signed calibration 同时可以增强不足通信，也可以抑制冗余通信。

---

## 6. 已有通信分配 \(q\)

已有通信分配 \(q\) 表示 base model 当前把多少通信资源分给 pair \((i,j)\)。

### 6.1 如果 backbone 能提供 attention/message weight

例如 AttentiveFP 或 Transformer，可以使用 attention/message weight 作为 base communication：

\[
c_{ij}^{base} = \alpha_{ij}^{base}
\]

然后归一化：

\[
q_{ij}^{base}
=
\frac{c_{ij}^{base}}
{\sum_{(a,b)\in \mathcal P_G} c_{ab}^{base}+\varepsilon}
\]

### 6.2 如果 backbone 不方便提取 attention

可使用 finite-hop support：

\[
c_{ij}^{base}
=
\sum_{\ell=1}^{K}\alpha_\ell [P_A^\ell]_{ij}
\]

其中：

- \(A\)：原始共价键邻接矩阵；
- \(P_A=D(A)^{-1}A\)：row-normalized propagation matrix；
- \(K=3\)；
- \(\alpha_\ell=1/K\)。

归一化：

\[
q_{ij}^{base}
=
\frac{c_{ij}^{base}}
{\sum_{(a,b)\in \mathcal P_G}c_{ab}^{base}+\varepsilon}
\]

### 6.3 校准后的通信分配

CoReMol 校准后得到：

\[
q_{ij}^{cal}
=
\frac{c_{ij}^{cal}}
{\sum_{(a,b)\in \mathcal P_G}c_{ab}^{cal}+\varepsilon}
\]

实验中比较：

\[
\mathrm{TCM}(q^{base})
\quad \text{vs.} \quad
\mathrm{TCM}(q^{cal})
\]

---

## 7. 任务证据 \(p\)：Counterfactual Pair Sensitivity

### 7.1 Frozen base model

先训练一个 base model：

\[
f_0(G)
\]

然后冻结其参数。

### 7.2 对每个候选 pair 注入 probe communication

对每个候选 pair \((i,j)\)，临时加入一个很小的 probe communication：

单向版本：

\[
h_i^{probe}=h_i+\epsilon W h_j
\]

双向版本：

\[
h_i^{probe}=h_i+\epsilon W h_j,\qquad
h_j^{probe}=h_j+\epsilon W h_i
\]

其中：

- \(\epsilon\) 是很小的 probe strength；
- \(W\) 可以是一个固定线性映射或与 backbone message map 对齐；
- probe 只用于诊断，不参与训练。

### 7.3 观察 loss change

定义 pair sensitivity：

\[
r_{ij}
=
\mathcal L(f_0(G),y)
-
\mathcal L(f_{ij}^{probe}(G),y)
\]

解释：

- \(r_{ij}>0\)：加入该 pair 的通信后 loss 下降，说明这对 pair 是潜在有益通信；
- \(r_{ij}<0\)：加入该 pair 的通信后 loss 上升，说明这对 pair 可能是冗余或有害通信；
- \(|r_{ij}|\) 越大，pair 对当前预测越敏感。

---

## 8. 正负任务证据分布

### 8.1 有益任务证据分布 \(p^+\)

\[
e^+_{ij}=[r_{ij}]_+
\]

\[
p^+_{ij}
=
\frac{e^+_{ij}}
{\sum_{(a,b)\in \mathcal P_G} e^+_{ab}+\varepsilon}
\]

\(p^+\) 表示：

> 哪些 pair 如果增加通信，会对当前任务更有帮助。

### 8.2 负向任务证据分布 \(p^-\)

\[
e^-_{ij}=[-r_{ij}]_+
\]

\[
p^-_{ij}
=
\frac{e^-_{ij}}
{\sum_{(a,b)\in \mathcal P_G} e^-_{ab}+\varepsilon}
\]

\(p^-\) 表示：

> 哪些 pair 增强通信后反而可能伤害预测，或至少不应被过度通信。

---

## 9. 主指标：Task-Communication Misallocation, TCM

定义：

\[
\mathrm{TCM}(q)
=
\underbrace{
\mathrm{TV}(q,p^+)
}_{\text{useful evidence mismatch}}
+
\lambda
\underbrace{
\sum_{(i,j)\in\mathcal P_G}q_{ij}p^-_{ij}
}_{\text{harmful evidence leakage}}
\]

其中：

\[
\mathrm{TV}(q,p^+)
=
\frac12
\sum_{(i,j)\in\mathcal P_G}
|q_{ij}-p^+_{ij}|
\]

解释：

- 第一项衡量模型通信分配 \(q\) 是否对齐有益任务证据 \(p^+\)；
- 第二项惩罚模型把通信资源分配到负向 pair 上；
- \(\lambda\) 默认取 1；
- \(\mathrm{TCM}\) 越小，说明通信分配越符合任务证据。

---

## 10. 校准收益：Delta TCM

真正用于报告的机制提升是：

\[
\Delta \mathrm{TCM}
=
\mathrm{TCM}(q^{base})
-
\mathrm{TCM}(q^{cal})
\]

如果：

\[
\Delta \mathrm{TCM}>0
\]

说明 CoReMol 降低了任务通信分配失配。

这可以作为 Paper B 的核心机制指标。

---

## 11. Top-K 简化版本

完整 TCM 可能计算较重。第一阶段可以先实现 Top-K 版本。

### 11.1 正向有益 pair 集合

\[
\mathcal T_K^+
=
\operatorname{TopK}_{(i,j)} r_{ij}
\]

### 11.2 负向 pair 集合

\[
\mathcal T_K^-
=
\operatorname{TopK}_{(i,j)} (-r_{ij})
\]

### 11.3 Useful Coverage@K

\[
\mathrm{UCov@K}(q)
=
\sum_{(i,j)\in\mathcal T_K^+}q_{ij}
\]

越大越好。

它表示：

> 模型通信资源有多少落在任务有益 pair 上。

### 11.4 Harmful Leakage@K

\[
\mathrm{HLeak@K}(q)
=
\sum_{(i,j)\in\mathcal T_K^-}q_{ij}
\]

越小越好。

它表示：

> 模型通信资源有多少分给了负向 pair。

### 11.5 Top-K Misallocation

\[
\mathrm{TCM@K}(q)
=
1-\mathrm{UCov@K}(q)
+
\lambda\mathrm{HLeak@K}(q)
\]

越小越好。

校准收益：

\[
\Delta \mathrm{TCM@K}
=
\mathrm{TCM@K}(q^{base})
-
\mathrm{TCM@K}(q^{cal})
\]

第一阶段推荐先实现 TCM@K，因为它直观、轻量、便于 debug。

---

## 12. TCM 如何支撑全文故事线

### 12.1 研究问题

不是说：

> 现有方法没有用 \(D-C\)。

而是说：

> 现有 molecular encoder 可能存在 task-communication misallocation：有些任务有益 atom pair 通信不足，有些冗余 pair 被过度通信。

TCM 用来诊断这个问题。

### 12.2 方法设计

CoReMol 的设计目标是降低 TCM。

它估计：

\[
S_\theta(i,j|c)=D_\theta(i,j|c)-C(i,j)
\]

再通过：

\[
\delta_{ij}
=
\beta\tanh(S_\theta/\tau)
\]

对 base communication 进行 signed calibration，使：

\[
q^{cal}
\]

更接近 \(p^+\)，并远离 \(p^-\)。

### 12.3 实验验证

实验不仅看 ROC-AUC / RMSE，还要看：

\[
\Delta \mathrm{TCM}>0
\]

并比较：

- Base model；
- Random calibration；
- Unsigned gate；
- Positive-only residual；
- Full CoReMol；
- w/o \(C\)；
- w/o context。

如果 Full CoReMol 的 \(\Delta\mathrm{TCM}\) 最大，并且性能也提升，说明方法确实在校准通信分配。

---

## 13. 为什么 TCM 不是循环论证

TCM 的任务证据 \(p^+,p^-\) 来自 frozen base model 的 counterfactual pair sensitivity，而不是 CoReMol 的 \(D_\theta\)。

因此：

- \(D_\theta\)：模型训练中的 learned demand；
- \(p^+,p^-\)：独立诊断得到的任务证据；
- TCM：post-hoc 机制评估指标；
- TCM 不参与训练目标。

这样可以避免“模型自己定义答案，然后说自己对齐答案”的问题。

---

## 14. 与 CoReMol 原有中间指标的关系

TCM 是更高层的问题诊断指标，其余指标是局部机制指标。

| 指标 | 作用 |
|---|---|
| TCM / TCM@K | 衡量通信分配是否对齐任务证据 |
| \(\Delta\)TCM | 衡量 CoReMol 是否降低通信分配失配 |
| EnhanceRatio | 正残差 pair 是否被增强 |
| SuppressRatio | 负残差 pair 是否被抑制 |
| MismatchReduction | learned demand 和 calibrated support 是否更接近 |
| FunctionalGroupEnrichment | 正残差 pair 是否落在化学合理区域 |
| CounterfactualMasking | top positive residual routes 是否真的影响预测 |

TCM 应该放在机制分析的核心位置。

---

## 15. 可能的审稿质疑与应对

### 15.1 Counterfactual sensitivity 是否可靠？

应对策略：

- 使用多个 probe strength \(\epsilon\)；
- 比较 finite-difference probe 和 gradient-based sensitivity；
- 对 top positive pairs 做 counterfactual masking；
- 检查 top evidence pairs 是否有 functional group enrichment。

### 15.2 \(p^+\) 来自 base model，会不会偏向 base model？

这是合理限制。论文中应表述为：

> TCM diagnoses whether a given backbone allocates communication consistently with its own task-sensitive pair evidence.

也就是说，第一阶段研究的是 **within-backbone communication misallocation**，不是声称 \(p^+\) 是真实化学因果证据。

### 15.3 TCM 是否等价于性能？

不等价。

TCM 是机制诊断指标，不是性能替代指标。论文中应说明：

> TCM is used to diagnose communication allocation behavior, not to replace task metrics.

需要进一步报告：

- \(\Delta\)TCM 与 \(\Delta\)Performance 的相关性；
- Counterfactual masking；
- Functional group enrichment。

---

## 16. 实验表格设计

### 16.1 TCM 机制表

| Model | TCM ↓ | ΔTCM ↑ | UCov@K ↑ | HLeak@K ↓ | Task Metric |
|---|---:|---:|---:|---:|---:|
| Base AttentiveFP | | - | | | |
| RandomCalib | | | | | |
| UnsignedGate | | | | | |
| PositiveOnly | | | | | |
| Full CoReMol | | | | | |

### 16.2 数据集层面汇总

| Dataset | Base TCM ↓ | Full TCM ↓ | ΔTCM ↑ | ΔAUC / ΔRMSE |
|---|---:|---:|---:|---:|
| BBBP | | | | |
| BACE | | | | |
| ESOL | | | | |
| FreeSolv | | | | |

### 16.3 相关性图

画散点图：

\[
\Delta \mathrm{TCM}
\quad \text{vs.} \quad
\Delta \text{Performance}
\]

如果不同数据集和不同消融版本中，\(\Delta\mathrm{TCM}\) 越大，性能提升越明显，则说明该指标和任务效果存在机制关联。

---

## 17. 论文中的定义段落示例

英文版：

> To diagnose whether a molecular encoder allocates communication consistently with task-relevant pair evidence, we introduce Task-Communication Misallocation (TCM). For each molecule, we first estimate a normalized communication distribution over candidate atom pairs from the backbone message weights or finite-hop propagation support. We then estimate task evidence using a frozen-model counterfactual pair probe: a small communication channel is injected into a candidate pair, and the resulting loss change is used to separate useful pair evidence from harmful pair evidence. TCM measures the distance between the model communication distribution and useful evidence, with an additional penalty on communication assigned to harmful pairs. This metric is used only for diagnosis and evaluation, not as a training objective.

中文版：

> 为了诊断分子 encoder 的通信分配是否与任务相关 pair 证据一致，我们提出任务通信分配失配度 TCM。对每个分子，先从 backbone 的 message weight 或 finite-hop support 中估计已有通信分配；再用 frozen model 的反事实 pair probe 估计任务证据，即临时给某个 atom pair 注入小通信通道，并观察 loss 变化，从而区分有益 pair 和负向 pair。TCM 衡量通信分配与有益证据之间的距离，并惩罚分配到负向 pair 的通信质量。该指标仅用于诊断和评估，不作为训练目标。

---

## 18. 第一阶段 Codex 实现建议

建议优先实现 TCM@K，而不是完整 TCM。

### 18.1 实现步骤

1. 训练 base AttentiveFP；
2. 冻结 base model；
3. 对每个 test molecule 构造候选 pair set；
4. 对每个 pair 注入 small probe；
5. 记录 loss change \(r_{ij}\)；
6. 构造 \(\mathcal T_K^+\) 和 \(\mathcal T_K^-\)；
7. 计算 base \(q^{base}\)；
8. 训练 Full CoReMol；
9. 计算 calibrated \(q^{cal}\)；
10. 计算 UCov@K、HLeak@K、TCM@K、ΔTCM@K；
11. 与 RandomCalib、UnsignedGate、PositiveOnly 对比；
12. 输出表格与可视化。

### 18.2 推荐默认值

- \(d_{\max}=4\)；
- \(K_{\text{top}}=10\) 或 top 10%；
- probe strength \(\epsilon=0.01\)；
- 额外测试 \(\epsilon\in\{0.005,0.01,0.02\}\)；
- \(\lambda=1\)；
- 3 seeds。

---

## 19. 第一阶段通过标准

TCM 相关验证通过，需要满足：

1. Base model 的 \(q^{base}\) 对 top useful pairs 的 UCov@K 不高，说明存在 under-allocation；
2. Base model 对 top harmful pairs 的 HLeak@K 不低，说明存在 leakage 或 over-allocation；
3. Full CoReMol 提高 UCov@K；
4. Full CoReMol 降低 HLeak@K；
5. Full CoReMol 的 \(\Delta\mathrm{TCM@K}\) 高于 RandomCalib、UnsignedGate、PositiveOnly；
6. \(\Delta\mathrm{TCM@K}\) 与性能提升方向一致；
7. Counterfactual masking 支持 top useful pairs 的预测贡献。

如果这些不成立，说明 “communication misallocation” 这个故事线需要收缩或重设指标。

---

## 20. 最终论文逻辑

有了 TCM，Paper B 可以形成完整闭环：

1. **问题定义**：现有分子 encoder 可能存在 task-communication misallocation；
2. **指标提出**：提出 TCM / TCM@K 诊断已有通信是否对齐任务证据；
3. **现象验证**：用 base AttentiveFP 证明该失配存在；
4. **算法设计**：提出 CoReMol，通过 signed residual calibration 校准通信分配；
5. **机制验证**：CoReMol 降低 TCM，提高 UCov@K，降低 HLeak@K；
6. **任务验证**：CoReMol 在 MoleculeNet 任务上带来性能提升；
7. **解释验证**：top positive residual routes 与功能团/敏感 pair 有更高一致性；
8. **反事实验证**：mask top positive routes 对预测影响更大。

这条线比“提出模块然后看性能”更扎实。
