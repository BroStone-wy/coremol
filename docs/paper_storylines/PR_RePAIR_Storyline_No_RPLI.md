# PR Paper Storyline: RePAIR / RIR for Molecular Graph Learning

> Working title: **RePAIR: Residual Pairwise Interaction Rewiring for Molecular Graph Learning**  
> Alternative title: **Residual Interaction Rewiring for Molecular Graph Learning**

This document records the PR-targeted paper storyline. The key boundary is:

- **This PR paper does not introduce or foreground RPLI.**
- **RPLI is reserved for a separate ESWA-style architecture/system paper.**
- The PR paper focuses on the scientific problem, diagnostic metric, residual interaction operator, and mechanism evidence.

---

## 1. One-sentence thesis

Modern molecular graph models increasingly improve structural representation, but molecular structure is a **reference**, not a task-evidence oracle. The PR paper studies whether structure-induced interaction flow is aligned with task-beneficial pairwise evidence, diagnoses this alignment with **TCM**, and corrects mismatch through **structure-referenced residual pairwise interaction rewiring**.

Chinese summary:

```text
结构告诉模型哪些关系合理；
任务要求模型知道哪些关系有用；
TCM 诊断模型的信息流有没有看对 pair；
RePAIR/RIR 在结构参考分布上学习有符号残差偏移，校正交互流。
```

---

## 2. Core problem: task-conditioned pairwise interaction allocation

### 2.1 What existing models already do well

Existing molecular graph models are increasingly strong at structural representation:

- covalent topology;
- local neighborhoods;
- shortest-path and structural encodings;
- 3D geometry;
- ligand-pocket contact structures;
- local-global graph encoders and graph Transformers.

This paper should **not** claim that structure is useless or wrong. The correct position is:

> Molecular structure is essential, but it provides task-agnostic reference interactions rather than task-specific evidence allocation.

### 2.2 What remains missing

Most molecular tasks provide only graph-level or complex-level labels:

```text
molecule -> property
protein-ligand complex -> affinity
```

They do not directly supervise:

```text
which atom pair should communicate more;
which atom pair should communicate less;
which ligand-pocket contact is beneficial;
which contact is redundant or harmful.
```

Therefore, although a backbone is trained by task loss, its pairwise interaction allocation is still:

- implicit;
- weakly supervised;
- entangled with structural inductive bias;
- difficult to diagnose from final AUC/RMSE alone.

### 2.3 Problem statement

The paper formulates the missing layer as:

> **Task-conditioned pairwise interaction allocation in molecular graph learning.**

Formal question:

```text
Given a structure-induced reference interaction distribution,
how can a model diagnose whether its interaction flow aligns with task-beneficial pairwise evidence,
and correct mismatched pairwise interactions without discarding structural priors?
```

Key sentence:

> **Structure is a reference, not an oracle.**

---

## 3. Diagnostic metric: TCM

### 3.1 Why TCM appears before the model

Without a diagnostic, the claim that structural interaction flow and task evidence may be mismatched is only a story. TCM turns it into a measurable object.

TCM asks:

```text
Does an interaction distribution cover beneficial communication pairs?
Does it avoid harmful or redundant pairs?
Does RIR/RePAIR move interaction flow toward task-beneficial pair evidence?
```

### 3.2 Frozen-base counterfactual pair evidence

Train a base model and freeze it. For each candidate pair `p`, apply a small communication probe and observe loss change:

```text
L0 = original frozen-base loss
Lp = loss after a small communication probe on pair p
gp = L0 - Lp
```

Interpretation:

```text
gp > 0:
  adding communication on pair p lowers loss;
  p is beneficial communication evidence.

gp < 0:
  adding communication on pair p increases loss;
  p is harmful or redundant evidence.
```

### 3.3 TCM functional

For an interaction distribution `q` over candidate pairs:

```text
Benefit(q) = sum_p q_p [gp]_+ / (sum_p [gp]_+ + eps)
Harm(q)    = sum_p q_p [-gp]_+ / (sum_p [-gp]_+ + eps)
TCM(q)     = Benefit(q) - lambda * Harm(q)
```

Interpretation:

```text
High TCM:
  q allocates more flow to beneficial pairs and less to harmful pairs.

Low TCM:
  q misses beneficial pairs or leaks flow to harmful/redundant pairs.
```

### 3.4 What TCM is not

TCM is:

```text
an evaluation-time frozen-base diagnostic.
```

TCM is not:

```text
the main training target;
a replacement for task metrics;
an attention visualization produced by the same model being explained.
```

This distinction is important to avoid circular explanation.

---

## 4. Residual operator: RIR / RePAIR

### 4.1 Core idea

RIR/RePAIR does not learn a new attention distribution from scratch. It learns a **bounded signed residual shift** over a structure-induced reference distribution.

Given candidate set `Omega_i` for source node `i`, define:

```text
q_ref(j | i) = softmax_{j in Omega_i}(log C_ref(i,j) + eps)
```

where `C_ref` is a structure-induced reference support.

Then learn a residual logit field:

```text
rho_ij = beta * tanh(phi_theta(i,j,h,c) / tau)
```

where:

```text
rho_ij > 0: pair (i,j) should be enhanced relative to the structural reference
rho_ij < 0: pair (i,j) should be suppressed relative to the structural reference
```

The rewired distribution is:

```text
q_rew(j | i) = softmax_{j in Omega_i}(log q_ref(j | i) + rho_ij)
```

### 4.2 Interaction operator residual

Define an interaction operator:

```text
T_q(h)_i = sum_{j in Omega_i} q(j | i) m_ij
```

Recommended message form:

```text
m_ij = W h_j - W h_i
```

RIR/RePAIR injects only the residual interaction flow:

```text
R_theta(h)_i = T_{q_rew}(h)_i - T_{q_ref}(h)_i
             = sum_j (q_rew(j|i) - q_ref(j|i)) m_ij
```

Final update:

```text
h'_i = h_i + gamma_i R_theta(h)_i
```

### 4.3 Why this is genuinely residual

The residual nature is not merely the output skip connection. The interaction operator itself is written as a deviation from a reference operator:

```text
rewired interaction operator - reference interaction operator
```

Analogy:

```text
ResNet:
  H(x) = x + F(x)
  learn residual feature mapping F(x) relative to x

RIR/RePAIR:
  h' = h + [T_qrew(h) - T_qref(h)]
  learn residual interaction flow relative to structural reference flow
```

Key sentence:

> **RIR/RePAIR extends the residual principle from feature mappings to interaction operators.**

### 4.4 Why it is not ordinary attention

Ordinary attention:

```text
q_att(j|i) = softmax(score_ij)
h'_i = h_i + sum_j q_att(j|i)m_ij
```

RIR/RePAIR:

```text
q_rew(j|i) = softmax(log q_ref(j|i) + rho_ij)
h'_i = h_i + [T_qrew(h)_i - T_qref(h)_i]
```

Differences:

1. anchored to a structural reference;
2. learns signed enhancement/suppression;
3. injects residual flow rather than a full new attention message;
4. safely degenerates to identity when `rho = 0`.

---

## 5. Candidate pairs and structural reference

### 5.1 Candidate pairs are not new chemical bonds

Candidate pairs define the search space where residual correction is allowed.

They are not:

```text
new chemical bonds;
claims that other pairs are physically irrelevant;
unrestricted dense attention.
```

They are:

```text
a chemically or geometrically constrained interaction space for residual correction.
```

### 5.2 Why not all pairs

Using all pairs risks turning the method into dense attention and adds noise, especially in protein-ligand settings. Candidate selection controls a bias-variance tradeoff:

```text
too narrow: miss task-beneficial pairs;
too wide: introduce noisy or spurious pairs.
```

### 5.3 Single-molecule candidate/reference

For single-molecule property prediction:

```text
candidate pair = atom-atom pair
C_ref = bond topology / shortest-path support / finite-hop propagation support
```

Example:

```text
P_A = row-normalized adjacency
C_ref(i,j) = Normalize(sum_{l=1..K} [P_A^l]_{ij})
```

### 5.4 Affinity candidate/reference

For pocket-conditioned affinity prediction:

```text
candidate pair = ligand atom - pocket residue/contact
C_ref = spatial/contact-based reference support
```

Example:

```text
d_ir = distance between ligand atom i and pocket residue/contact r
C_ref^LP(i,r) = exp(-d_ir^2 / sigma^2) * 1[d_ir <= cutoff]
```

In the PR paper, this affinity part should be presented as a natural instantiation of Cross-RIR, not as a full RPLI architecture.

---

## 6. Two natural instantiations in the PR paper

The PR paper should not split into two independent task papers. It should say that the same residual interaction principle has two molecular graph instantiations.

### 6.1 Intra-molecular RePAIR for single-molecule tasks

```text
nodes: atoms
reference relation: intra-molecular chemistry
candidate pairs: atom-atom
C_ref: covalent topology / finite-hop support
operator: Intra-RePAIR
readout: graph-level classification or regression
```

Main interpretation:

> In single-molecule tasks, RePAIR corrects atom-atom interaction allocation over intra-molecular chemical references.

### 6.2 Cross-RePAIR for pocket-conditioned affinity

```text
nodes: ligand atoms + pocket residues/contacts
reference relation: ligand-pocket interface contact
candidate pairs: ligand atom - pocket residue/contact
C_ref: spatial/contact reference
operator: Cross-RePAIR
readout: affinity regression
```

Main interpretation:

> In affinity prediction, RePAIR corrects ligand-pocket interface interaction allocation over contact references.

Do not introduce RPLI in this PR paper. Use a generic base encoder or minimal ligand/pocket encoder. RPLI is reserved for the ESWA architecture paper.

---

## 7. Expected PR contribution statements

### Contribution 1: Problem formulation

We formulate **task-conditioned pairwise interaction allocation** as a missing layer in molecular graph learning.

### Contribution 2: TCM diagnostic

We introduce **TCM**, a frozen-base counterfactual diagnostic that measures whether an interaction distribution covers beneficial communication pairs and avoids harmful leakage.

### Contribution 3: Residual interaction operator

We propose **RIR/RePAIR**, a structure-referenced residual interaction operator that learns bounded signed shifts over a reference distribution and injects only rewired-minus-reference interaction flow.

### Contribution 4: Unified molecular graph instantiations

We instantiate the same mechanism for intra-molecular atom-pair rewiring and ligand-pocket interface rewiring, showing that the principle applies beyond a single task.

---

## 8. PR experimental storyline

### 8.1 Main performance

Show that RePAIR improves task metrics over the same base encoder.

Single-molecule representative datasets:

```text
BBBP
ClinTox
FreeSolv
```

Optional appendix:

```text
BACE, Tox21, HIV, SIDER, ToxCast, ESOL, Lipo
```

Affinity extension:

```text
PDBbind CleanSplit / CASF-style setting
```

but present it as **pocket-conditioned molecular graph learning**, not affinity SOTA competition.

### 8.2 Single-molecule mechanism ablation

Required variants:

```text
Full RePAIR
MLP adapter
ordinary attention
no-reference
random C_ref
PositiveOnly
```

Expected conclusion:

```text
Full RePAIR > ordinary attention / MLP adapter:
  improvement is not from extra parameters or another attention layer.

Full RePAIR > no-reference / random C_ref:
  structural reference matters.

Full RePAIR > PositiveOnly:
  signed enhancement and suppression are both needed.
```

### 8.3 Affinity mechanism ablation

Required variants:

```text
ligand-only
pocket context
Intra-RIR
Cross-RIR
Full
w/o interface readout
```

Expected conclusion:

```text
pocket context > ligand-only:
  affinity is not only ligand memorization.

Cross-RIR > Intra-RIR:
  interface residual rewiring is central for affinity.

Full > w/o interface readout:
  residual interface flow should be explicitly used as prediction evidence.
```

### 8.4 TCM mechanism table

Report:

```text
Delta Benefit ↑
Delta Harm ↓
Delta TCM ↑
```

Use `10^-3` units for readability.

Interpretation:

```text
Delta Benefit ↑:
  more residual flow reaches beneficial pairs.

Delta Harm ↓:
  less residual flow leaks to harmful/redundant pairs.

Delta TCM ↑:
  interaction allocation is more task-aligned.
```

### 8.5 Visualization

Required PR figures:

1. Method overview: `C_ref -> q_ref -> rho -> q_rew -> T_qrew - T_qref`.
2. TCM alignment scatter:

```text
x-axis: gp = L0 - Lp
y-axis: Delta alpha = q_rew - q_ref
```

3. Single-molecule residual flow visualization:

```text
atom graph + enhanced/suppressed residual pairs + TCM evidence
```

4. Ligand-pocket interface residual flow visualization:

```text
contact map + Delta alpha^LP + Interface-TCM evidence
```

---

## 9. Overfitting defense

Residual correction is task-adaptive, but not unconstrained.

Design constraints:

```text
candidate pair restriction;
structure-induced C_ref anchor;
bounded tanh shift rho in [-beta, beta];
residual flow injection T_qrew - T_qref;
small gate / normalization / dropout;
parameter-matched and attention controls;
held-out task evaluation;
TCM alignment analysis.
```

Preferred response to reviewer concern:

> Task-conditioned correction should not be confused with overfitting. RePAIR constrains correction to a chemically or geometrically meaningful candidate space, anchors it to a structural reference, bounds its magnitude, and injects only residual interaction flow. If no reliable correction is learned, the module safely degenerates to zero update.

---

## 10. What this PR paper should not do

Do not foreground:

```text
RPLI architecture;
dual-head calibrated RPLI readout;
GEMS-style implementation details;
protein-ligand affinity SOTA claims;
inference-time ensembles;
full engineering handoff details.
```

Those belong to the ESWA architecture/system paper.

The PR paper should focus on:

```text
problem formulation;
TCM diagnostic;
residual interaction operator;
single-molecule and interface instantiations;
mechanism evidence.
```

---

## 11. ESWA paper note: motivation and naming need revision

The ESWA paper should not use heavy phrases such as:

```text
reference-preserving latent interaction backbone
```

as the main external-facing name. This phrase is precise but hard to understand.

Potential clearer naming directions:

```text
Calibrated Interaction Network
Interface Calibration Network
Pocket-Ligand Calibration Network
Interaction Flow Calibration Network
```

The ESWA motivation should be rebuilt around a practical problem:

> Protein-ligand affinity models often combine global complex representations and interface interactions, but it is difficult to turn interface interaction evidence into a stable single-model prediction correction. The ESWA paper should present RPLI as a practical interaction-calibration architecture that converts ligand-pocket reference flow into controlled affinity correction within one checkpoint.

This ESWA reframing is intentionally separate from the PR mechanism paper.
