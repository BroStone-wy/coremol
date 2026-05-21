# CoReMol Stage-1：算法核心设计与快速验证实验计划

本文档用于交给 Codex 执行第一层实验。目标不是刷榜，而是快速验证 **CoReMol 的核心机制是否成立**：在单分子性质预测中，模型是否能学习任务上下文条件下的 signed residual communication calibration，并在同一 backbone、同一 split、同一训练预算下带来稳定增益。

---

## 0. 当前论文定位

Paper B 暂定主线：

> Context-conditioned residual communication for molecular property prediction.

第一阶段只做单分子版本，不做 PDBbind，不接 CheapNet。原因是 PDBbind/蛋白-配体复合物涉及双分支编码、cross interaction、3D pose、split 泄漏和公平对比等额外变量。当前优先验证 CoReMol 的核心机制。

第一阶段方法名建议：

> **CoReMol: Context-conditioned Residual Communication Calibration for Molecular Property Prediction**

核心区别：

- 不是把 PairAlign 直接应用到分子任务；
- 不是普通 ResNet-style residual connection；
- 不是简单增加 attention gate；
- 而是在 **原始共价键图提供的基础通信结构之外**，学习一组任务上下文条件下的 signed residual communication calibration；
- 正残差增强通信不足的 atom pair；
- 负残差抑制冗余或过度强调的 atom pair。

---

## 1. 第一层实验的核心目标

第一层实验只回答一个问题：

> 在同一个 backbone 下，加入 CoReMol 的 signed residual communication calibration 后，是否能同时带来性能提升和机制指标改善？

必须同时观察两类证据。

### 1.1 性能证据

同一 backbone 对比：

\[
\text{Backbone}
\quad \text{vs.} \quad
\text{Backbone + CoReMol}
\]

第一阶段推荐 backbone：

1. **AttentiveFP**：优先级最高，因为它本身使用 attention/message weight，最适合加入通信校准；
2. GIN：第二阶段补充，用于验证普通 MPNN 场景；
3. Graph Transformer-lite：第二阶段补充，用于验证 attention-bias 场景。

第一阶段先只做 AttentiveFP。

### 1.2 机制证据

核心中间指标：

1. EnhanceRatio：正残差 pair 是否真的被增强；
2. SuppressRatio：负残差 pair 是否真的被抑制；
3. Residual Mismatch Reduction：校准后 demand-support mismatch 是否下降；
4. Calibration Contrast：正负 residual calibration 是否有明确区分；
5. Counterfactual Masking：去掉 top positive residual pairs 是否更影响预测；
6. Functional Group Enrichment@K：top positive residual pairs 是否更集中于化学上有意义的结构区域。

前四个是第一轮必须输出；后两个可在第一轮跑通后补充。

---

## 2. 第一阶段数据集

先不要全量 MoleculeNet。第一轮只跑四个任务：

| 数据集 | 类型 | 指标 | 第一阶段作用 |
|---|---|---|---|
| BBBP | 分类 | ROC-AUC ↑ | 小规模 ADMET 任务，适合可视化和结构解释 |
| BACE | 分类 | ROC-AUC ↑ | 活性相关任务，适合观察结构模式 |
| ESOL | 回归 | RMSE ↓ / MAE ↓ | 溶解度任务，适合官能团/极性解释 |
| FreeSolv | 回归 | RMSE ↓ / MAE ↓ | 水合自由能任务，适合极性/电性 pair 分析 |

数据划分：

- 优先采用 scaffold split；
- train/valid/test = 8/1/1；
- 第一轮 3 seeds；
- 机制成立后扩展到 10 seeds 或官方常用 split 配置。

---

## 3. CoReMol 核心算法设计

### 3.1 输入

单个分子图：

\[
G=(V,E_{\text{bond}},X,E_f)
\]

其中：

- \(V\)：原子集合；
- \(E_{\text{bond}}\)：真实共价键；
- \(X\)：原子特征；
- \(E_f\)：键特征；
- \(y\)：分子性质标签。

注意：CoReMol 不把 residual pair 当作真实化学键。真实共价键图保持不变，residual communication 只作为神经网络计算通信结构。

---

## 4. Backbone 表示

第一阶段 backbone 使用 AttentiveFP。

Backbone 输出：

\[
H = \{h_i\}_{i=1}^{n}
\]

以及分子级上下文：

\[
c_{\text{mol}} = \mathrm{READOUT}(H)
\]

如果 AttentiveFP 代码能直接提取 attention logits/weights，可保存为辅助信息；如果不能，第一阶段不强依赖 backbone 内部 attention，而使用基于共价键图的 finite-hop support 作为 base support。

---

## 5. Candidate atom-pair set

第一阶段候选 pair 集合建议：

\[
\mathcal{C} = \{(i,j): i\neq j,\ d_G(i,j)\le d_{\max}\}
\]

默认：

- \(d_{\max}=4\)；
- 如果分子很小，也可允许 all pairs；
- 排除 self-loop；
- 是否排除原始 bond pair 可作为配置项：
  - `include_bond_pairs=True`：校准所有通信；
  - `include_bond_pairs=False`：只做非键 residual communication；
- 第一轮推荐 `include_bond_pairs=True`，因为 signed calibration 既可增强不足通信，也可抑制冗余通信。

---

## 6. Base communication support \(C(i,j)\)

为了避免 CoReMol 退化为普通 pair scorer，\(C(i,j)\) 必须保留结构含义。

第一阶段推荐使用 finite-hop support：

\[
C(i,j)=\mathrm{Norm}\left(
\sum_{\ell=1}^{K}\alpha_{\ell}[P_A^\ell]_{ij}
\right)
\]

其中：

- \(A\)：原始共价键邻接矩阵；
- \(P_A = D(A)^{-1}A\)：row-normalized propagation matrix；
- \(K=3\)；
- \(\alpha_\ell = 1/K\)；
- `Norm` 将支持值映射到 \([0,1]\)，可用 per-graph min-max 或 sigmoid/log normalization。

推荐实现：

```python
support = sum(alpha[l] * P_power[l][i, j] for l in 1..K)
C_ij = support / (support.max_per_graph + eps)
```

如果使用 attention backbone 并且能够稳定提取 base attention，也可以在后续版本中设置：

\[
C(i,j)=\alpha^{\text{base}}_{ij}
\]

但第一阶段建议先用 finite-hop support，便于实现和解释。

---

## 7. Task-conditioned demand \(D_\theta(i,j \mid c_{\text{mol}})\)

对每个候选 atom pair，构造 pair descriptor：

\[
z_{ij}=[h_i,h_j,c_{\text{mol}},\rho_{ij}]
\]

其中 \(\rho_{ij}\) 包括轻量结构/化学描述：

- topological distance \(d_G(i,j)\)；
- atom type of \(i,j\)；
- whether same ring；
- whether aromatic atoms；
- whether hetero atoms；
- optional: shortest-path bond-type summary；
- optional: formal charge / donor / acceptor indicators if available.

Demand network：

\[
D_\theta(i,j\mid c_{\text{mol}})
=
\sigma(\mathrm{MLP}_D(z_{ij}))
\]

输出范围：

\[
D_\theta(i,j\mid c_{\text{mol}})\in(0,1)
\]

解释：

- \(D_\theta\) 表示当前任务上下文下，该 atom pair 的通信需求；
- 它不是最终 attention；
- 它必须和 base support \(C\) 相减，形成 residual score。

---

## 8. Signed residual communication score

核心定义：

\[
S_\theta(i,j\mid c_{\text{mol}})
=
D_\theta(i,j\mid c_{\text{mol}})
-
C(i,j)
\]

解释：

- \(S_\theta>0\)：任务需求高于当前结构支持，应该增强通信；
- \(S_\theta<0\)：当前结构支持高于任务需求，说明该 pair 在当前任务下可能冗余或过度通信，应该抑制；
- \(S_\theta\approx0\)：当前支持与任务需求接近，保持即可。

---

## 9. Communication calibration

将 signed residual score 映射成有界校准项：

\[
\delta_{ij}
=
\beta \tanh\left(
\frac{S_\theta(i,j\mid c_{\text{mol}})}{\tau}
\right)
\]

默认超参数：

- \(\beta=0.1\) 或 learnable scalar；
- \(\tau=0.5\)；
- 可加 dropout；
- \(\delta_{ij}\in[-\beta,\beta]\)。

\(\delta_{ij}\) 不作为负边权直接使用，而作为通信校准项。

---

## 10. Residual hidden-state update

第一阶段建议使用 “base-support logit calibration + residual difference aggregation”。

### 10.1 Base logit

对候选集合 \(\mathcal{C}_i\)：

\[
\ell^{\text{base}}_{ij}=\log(C(i,j)+\varepsilon)
\]

\[
\alpha^{\text{base}}_{ij}
=
\mathrm{softmax}_{j\in\mathcal{C}_i}
(\ell^{\text{base}}_{ij})
\]

### 10.2 Calibrated logit

\[
\ell^{\text{cal}}_{ij}
=
\ell^{\text{base}}_{ij}
+
\delta_{ij}
\]

\[
\alpha^{\text{cal}}_{ij}
=
\mathrm{softmax}_{j\in\mathcal{C}_i}
(\ell^{\text{cal}}_{ij})
\]

### 10.3 Residual communication update

\[
\Delta h_i
=
\sum_{j\in\mathcal{C}_i}
(\alpha^{\text{cal}}_{ij}-\alpha^{\text{base}}_{ij})
W_V h_j
\]

\[
h'_i
=
h_i
+
\eta\cdot \mathrm{Dropout}(\mathrm{LN}(\Delta h_i))
\]

其中：

- \(\eta\) 可以是 learnable gate，初始化为小值，例如 0.1；
- \(W_V\) 是线性映射；
- 这个设计的好处是：
  - positive residual pair 会提高相对通信权重；
  - negative residual pair 会降低相对通信权重；
  - 最终仍然更新 hidden feature；
  - 避免直接使用负邻接或负边权。

最终 readout：

\[
\hat y = \mathrm{READOUT}(H')
\]

---

## 11. 训练目标

第一阶段主目标只用任务 loss：

分类任务：

\[
\mathcal{L}_{task}=\mathrm{BCEWithLogitsLoss}
\]

回归任务：

\[
\mathcal{L}_{task}=\mathrm{MSELoss}
\]

可选轻量正则：

### 11.1 Calibration magnitude regularization

防止 \(\delta\) 过大：

\[
\mathcal{L}_{mag}
=
\frac{1}{|\mathcal{C}|}\sum_{(i,j)\in\mathcal{C}}\delta_{ij}^{2}
\]

默认：

\[
\lambda_{mag}=10^{-4}
\]

### 11.2 Demand entropy / diversity regularization

第一轮不建议加，避免变量太多。

总损失：

\[
\mathcal{L}
=
\mathcal{L}_{task}
+
\lambda_{mag}\mathcal{L}_{mag}
\]

---

## 12. 第一层对比组

第一层必须至少做以下 5 个版本：

| 版本 | 描述 | 目的 |
|---|---|---|
| Base AttentiveFP | 原始 backbone | 主对照 |
| Base + RandomCalib | 将 \(\delta\) 随机打乱或采样同分布噪声 | 排除“多加扰动就提升” |
| Base + UnsignedGate | 只学习非负 gate，无 signed residual | 区分普通 attention/gate |
| Base + PositiveOnly | 使用 \([S]_+\)，只增强不抑制 | 验证 negative suppression 是否有用 |
| Base + Full CoReMol | 完整 signed residual calibration | 主方法 |

建议第二轮加：

| 版本 | 描述 | 目的 |
|---|---|---|
| Full w/o Context | 去掉 \(c_{\text{mol}}\) | 验证任务上下文必要性 |
| Full w/o \(C\) | 设置 \(S=D_\theta\)，不减 support | 验证 demand-support residual 是否必要 |
| Full with all-pair dense | 不限制候选 pair | 验证候选稀疏约束是否必要 |
| Full with nonbond-only | 只校准非键 pair | 区分 bond and residual communication |

---

## 13. 第一层性能指标

输出表 1：

| Model | BBBP ROC-AUC ↑ | BACE ROC-AUC ↑ | ESOL RMSE ↓ | FreeSolv RMSE ↓ |
|---|---:|---:|---:|---:|
| AttentiveFP | | | | |
| + RandomCalib | | | | |
| + UnsignedGate | | | | |
| + PositiveOnly | | | | |
| + Full CoReMol | | | | |

每个结果需要：

- mean ± std；
- 3 seeds；
- 同一 split 策略；
- 同一训练预算；
- 同一 hidden dim / optimizer / early stopping 策略。

---

## 14. 第一层中间结构指标

### 14.1 EnhanceRatio

正残差 pair 是否真的被增强：

\[
\mathcal{P}^{+}=
\{(i,j):S_{ij}>0\}
\]

\[
\mathrm{EnhanceRatio}
=
\frac{
|\{(i,j)\in\mathcal{P}^{+}:
\alpha^{cal}_{ij}>\alpha^{base}_{ij}\}|
}{
|\mathcal{P}^{+}|+\varepsilon
}
\]

期望：

- Full CoReMol 显著高于 RandomCalib；
- PositiveOnly 也可能高，但没有 suppression 能力；
- UnsignedGate 通常缺少正负区分。

---

### 14.2 SuppressRatio

负残差 pair 是否真的被抑制：

\[
\mathcal{P}^{-}=
\{(i,j):S_{ij}<0\}
\]

\[
\mathrm{SuppressRatio}
=
\frac{
|\{(i,j)\in\mathcal{P}^{-}:
\alpha^{cal}_{ij}<\alpha^{base}_{ij}\}|
}{
|\mathcal{P}^{-}|+\varepsilon
}
\]

期望：

- Full CoReMol 最高；
- PositiveOnly 接近无效；
- 该指标是区分 signed calibration 和 add-only residual 的关键。

---

### 14.3 Residual Mismatch Reduction

校准后 demand-support mismatch 是否下降：

\[
\Delta \mathrm{Mismatch}
=
\frac{
\sum_{(i,j)}w_{ij}|D_{ij}-\alpha^{base}_{ij}|
-
\sum_{(i,j)}w_{ij}|D_{ij}-\alpha^{cal}_{ij}|
}{
\sum_{(i,j)}w_{ij}|D_{ij}-\alpha^{base}_{ij}|+\varepsilon
}
\]

权重 \(w_{ij}\) 建议：

- 第一版使用 \(w_{ij}=|S_{ij}|\)；
- 或只在 top-\(|S|\) pairs 上计算；
- 注意：该指标使用 learned \(D\)，因此主要作为机制 sanity check，不能单独作为强证据。

期望：

- Full CoReMol 高于 w/o \(C\)、RandomCalib、UnsignedGate。

---

### 14.4 Calibration Contrast

正负 calibration 是否分离：

\[
\mathrm{Contrast}
=
\mathbb{E}_{(i,j)\in \mathrm{TopK}(S)}
[\delta_{ij}]
-
\mathbb{E}_{(i,j)\in \mathrm{BottomK}(S)}
[\delta_{ij}]
\]

默认：

- \(K=10\%\times|\mathcal{C}|\)；
- 或每个分子 top-10 pairs。

期望：

- Full CoReMol 具有明显正负 contrast；
- RandomCalib 和 UnsignedGate 不应有清晰 signed contrast。

---

### 14.5 Counterfactual Masking

训练完成后，在 test set 推理阶段做三类 mask：

1. Mask top positive residual pairs；
2. Mask random pairs；
3. Mask top negative residual pairs。

Mask 方式：

- 对 top positive pairs，将 \(\delta_{ij}=0\)；
- 或从候选集合中移除这些 pair 的 residual contribution；
- 不改变 backbone 本身。

分类任务统计：

\[
\Delta \mathrm{AUC}
=
\mathrm{AUC}_{normal}
-
\mathrm{AUC}_{masked}
\]

回归任务统计：

\[
\Delta \mathrm{RMSE}
=
\mathrm{RMSE}_{masked}
-
\mathrm{RMSE}_{normal}
\]

期望：

- Mask top positive residual pairs 的性能退化最大；
- Mask random pairs 退化较小；
- Mask top negative residual pairs 退化小，甚至可能无变化。

这是第一阶段最强机制证据之一。

---

### 14.6 Functional Group Enrichment@K

第一轮可先做 BBBP、BACE、ESOL 的 case-level 或 dataset-level 统计。

取 top positive residual atom pairs：

\[
\mathcal{P}^{+}_{K}
=
\mathrm{TopK}_{(i,j)}(\delta_{ij})
\]

使用 RDKit 标注 atom/pair 是否涉及：

- hetero atom；
- aromatic atom；
- ring atom；
- H-bond donor；
- H-bond acceptor；
- charged atom；
- halogen；
- common functional groups if SMARTS rules are available。

定义：

\[
\mathrm{FG\text{-}Enrichment@K}
=
\frac{
\#\text{top-K residual pairs involving functional-group atoms}
}{
K
}
\]

对比：

- random atom pairs；
- base support / base attention top-K pairs；
- CoReMol positive residual top-K pairs。

期望：

- CoReMol positive residual pairs 更富集于化学上有意义的区域；
- 不要求每个数据集都显著，但至少在 BBBP/BACE/ESOL 有可解释趋势。

---

## 15. 第一层机制指标输出表

输出表 2：

| Model | EnhanceRatio ↑ | SuppressRatio ↑ | MismatchReduction ↑ | Contrast ↑ | MaskingDrop ↑ |
|---|---:|---:|---:|---:|---:|
| RandomCalib | | | | | |
| UnsignedGate | | | | | |
| PositiveOnly | | | | | |
| Full CoReMol | | | | | |

其中：

- MaskingDrop 对分类用 \(\Delta\)AUC；
- MaskingDrop 对回归用 \(\Delta\)RMSE；
- 可以分类/回归分开报。

---

## 16. 第一层可视化

第一轮至少输出以下图：

### 16.1 Calibration heatmap

对每个数据集选 2–3 个代表分子，画：

1. \(D_{ij}\) heatmap；
2. \(C_{ij}\) heatmap；
3. \(S_{ij}=D_{ij}-C_{ij}\) heatmap；
4. \(\delta_{ij}\) heatmap。

### 16.2 Molecule residual route visualization

在 RDKit 2D 分子图上画：

- top positive residual pairs：红/橙线；
- top negative residual pairs：蓝/灰虚线；
- 原始化学键保持黑/灰；
- 不要把 residual pair 画成真实化学键，图例明确写 “residual communication route”。

### 16.3 Masking effect bar chart

展示：

- mask top positive；
- mask random；
- mask top negative；

对 AUC/RMSE 的影响。

---

## 17. 第一层通过标准

如果满足以下条件，进入第二层实验：

1. Full CoReMol 在至少 3/4 个第一阶段数据集上优于 Base 或接近最好；
2. Full CoReMol 优于 RandomCalib 和 UnsignedGate；
3. EnhanceRatio 与 SuppressRatio 明显成立；
4. MismatchReduction 为正，且高于主要消融；
5. Counterfactual Masking 显示 top positive residual pairs 的预测贡献更大；
6. 可视化中 residual routes 不完全随机，至少部分集中于合理化学区域。

如果只性能提升但机制指标不成立，不进入第二层，先修改 \(D_\theta\)、\(C\)、候选 pair 集合或校准强度。

---

## 18. 第一层失败时优先排查项

### 18.1 性能不提升，中间指标也不成立

优先排查：

- \(C(i,j)\) 是否归一化过强或过弱；
- \(\beta\) 是否太小；
- \(\tau\) 是否导致 tanh 饱和；
- candidate pairs 是否过密；
- \(D_\theta\) 是否塌缩到常数；
- \(\eta\) 是否初始化太大导致扰乱 backbone；
- 是否需要先冻结 backbone 训练 calibration，再联合训练。

### 18.2 性能提升，但机制指标不成立

说明可能只是额外参数带来的提升。优先修改：

- 加强 \(C\) 的结构约束；
- 加入 w/o \(C\) 对照；
- 检查 \(D_\theta\) 是否退化为 pair attention；
- 降低 residual branch 容量；
- 使用 counterfactual masking 验证真实贡献。

### 18.3 EnhanceRatio 成立但 SuppressRatio 不成立

说明模型只学会增强，没有真正抑制。尝试：

- 增大 \(\beta\)；
- 调整 \(\tau\)；
- 平衡正负 residual 的 batch statistics；
- 加入 calibration contrast loss，但第一轮不建议默认加。

---

## 19. 推荐代码结构

Codex 执行时建议创建以下文件或模块：

```text
coremol/
  models/
    attentivefp_base.py
    coremol_calibration.py
    coremol_attentivefp.py
  data/
    moleculenet_loader.py
    scaffold_split.py
  utils/
    pair_support.py
    pair_features.py
    metrics_coremol.py
    functional_group.py
    visualization.py
  configs/
    bbbp_coremol.yaml
    bace_coremol.yaml
    esol_coremol.yaml
    freesolv_coremol.yaml
  scripts/
    train_base.py
    train_coremol.py
    eval_mechanism.py
    run_counterfactual_masking.py
    visualize_residual_routes.py
  outputs/
    results/
    mechanism/
    figures/
```

---

## 20. Codex 第一阶段执行清单

### A. 数据与 backbone

- [ ] 跑通 MoleculeNet BBBP、BACE、ESOL、FreeSolv；
- [ ] 使用 scaffold split 8/1/1；
- [ ] 跑通 AttentiveFP baseline；
- [ ] 保存每个 test molecule 的 node embeddings 和 predictions。

### B. CoReMol 模块

- [ ] 实现 finite-hop support \(C(i,j)\)；
- [ ] 实现 pair descriptor \(\rho_{ij}\)；
- [ ] 实现 demand network \(D_\theta\)；
- [ ] 实现 signed residual score \(S=D-C\)；
- [ ] 实现 \(\delta=\beta\tanh(S/\tau)\)；
- [ ] 实现 residual difference aggregation；
- [ ] 接入 AttentiveFP readout。

### C. 消融版本

- [ ] RandomCalib；
- [ ] UnsignedGate；
- [ ] PositiveOnly；
- [ ] Full CoReMol；
- [ ] 可选：w/o Context；
- [ ] 可选：w/o \(C\)。

### D. 中间指标

- [ ] EnhanceRatio；
- [ ] SuppressRatio；
- [ ] MismatchReduction；
- [ ] CalibrationContrast；
- [ ] CounterfactualMasking；
- [ ] FunctionalGroupEnrichment@K。

### E. 输出

- [ ] 表 1：性能对比；
- [ ] 表 2：机制指标；
- [ ] residual heatmaps；
- [ ] RDKit residual route visualizations；
- [ ] masking effect bar charts；
- [ ] 训练日志与配置文件。

---

## 21. 第一层结论模板

如果结果成立，可在论文中形成以下结论：

> Under a controlled AttentiveFP backbone and identical training protocol, CoReMol improves molecular property prediction while producing consistent signed communication calibration. Positive residual scores increase the calibrated communication weights of selected atom pairs, whereas negative residual scores suppress over-supported interactions. The calibrated communication pattern reduces demand-support mismatch, and counterfactual masking shows that top positive residual routes contribute more strongly to prediction than random or negatively calibrated pairs. These results support the core claim that CoReMol learns task-conditioned residual communication rather than merely adding a generic residual connection.

中文结论：

> 在受控 AttentiveFP backbone 和相同训练协议下，CoReMol 不仅提升分子性质预测性能，还表现出一致的 signed communication calibration 行为：正残差增强选中 atom pair 的通信权重，负残差抑制过度支持的交互；校准后 demand-support mismatch 降低；反事实 masking 显示 top positive residual routes 对预测贡献更大。这说明 CoReMol 学到的是任务条件下的残差通信，而不是普通残差连接或普通 attention gate。

---

## 22. 后续第二层扩展方向

第一层成立后，再做：

1. GIN + CoReMol；
2. Graph Transformer-lite + CoReMol；
3. CurvFlow-lite / CurvFlow 对比；
4. Virtual-node routing 对比；
5. full MoleculeNet；
6. PDBbind / CoReBind。

第一阶段不做这些，避免变量太多。
