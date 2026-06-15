# PR Paper Storyline Logic: RePAIR / CoReMol for Molecular Graph Learning

Working date: 2026-06-15  
Target venue: **Pattern Recognition (PR)**  
Working title:

```text
RePAIR: Residual Pairwise Interaction Rewiring for Molecular Graph Learning
```

Alternative title if the final method name keeps CoReMol:

```text
CoReMol: Residual Pairwise Interaction Rewiring for Molecular Graph Learning
```

This document defines the **PR-paper story line only**. It intentionally excludes the RPLI architecture. RPLI should be developed as the main architecture in the later ESWA paper.

---

## 0. Hard Boundary Between the PR Paper and the ESWA Paper

### PR paper

The PR paper should be a **mechanism paper**.

It should focus on:

```text
1. A general problem:
   task-conditioned pairwise interaction allocation in molecular graph learning.

2. A diagnostic metric:
   TCM diagnoses whether interaction flow covers beneficial pair evidence and avoids harmful leakage.

3. A residual correction operator:
   RePAIR/RIR learns signed residual shifts over a structure-induced reference interaction distribution.

4. Two natural molecular graph instantiations:
   intra-molecular atom--atom rewiring and ligand--pocket interface rewiring.
```

The PR paper should **not** foreground RPLI, dual-head readout, single-stage affinity system details, or the final affinity engineering architecture.

### ESWA paper

The ESWA paper should become the **architecture / applied system paper**.

It should focus on:

```text
RPLI as the main affinity architecture
+ CoReMol/RePAIR interface calibration
+ calibrated prediction system
+ CleanSplit affinity prediction
+ engineering reproducibility and interpretability
```

The current phrase `reference-preserving` is accurate but too abstract for a broad applied audience. For ESWA, prefer more intuitive wording such as:

```text
interaction-calibrated affinity model
interface-calibrated affinity network
contact-guided interaction calibration
single-model ligand--pocket interaction calibration
```

Do not force ESWA terminology into the PR paper.

---

## 1. One-Sentence Thesis

```text
Molecular structure is a reference, not a task-evidence oracle.
TCM diagnoses whether structure-induced interaction flow aligns with task-beneficial pair evidence,
and RePAIR corrects the mismatch by learning signed residual shifts over the structural reference distribution.
```

Chinese version:

```text
分子结构是参考，不是任务证据 oracle。
TCM 诊断结构诱导的交互流是否对齐任务有益 pair evidence；
RePAIR 则在结构参考分布上学习有符号残差偏移，校正这种错位。
```

---

## 2. Core Problem Definition

The paper should not claim that molecular backbones do not learn tasks. They do.

The more precise problem is:

```text
Modern molecular graph backbones learn task-trained representations,
but their pairwise interaction allocation is implicit, weakly supervised,
and mediated by structural inductive biases.
```

Most molecular labels are graph-level or complex-level:

```text
molecule -> property
protein--ligand complex -> affinity
```

They do not directly label:

```text
which atom pair is beneficial,
which pair is harmful or redundant,
which ligand--pocket contact should be enhanced,
which contact should be suppressed.
```

Therefore, the central problem is:

```text
Task-conditioned pairwise interaction allocation:
How can a model allocate interaction flow toward task-beneficial pairwise evidence
while preserving molecular structure as a reference?
```

This is stronger and more general than saying:

```text
We add an adapter to a molecular backbone.
```

---

## 3. Motivation: Why Structure Is Not Enough

The paper should begin by respecting the current field trend:

```text
Molecular graph learning increasingly benefits from stronger structural priors:
  covalent topology,
  local neighborhoods,
  shortest-path distances,
  3D geometry,
  ligand--pocket contacts,
  structural attention biases.
```

Then make the key transition:

```text
These priors define chemically meaningful reference interactions.
However, a reference interaction is not necessarily a task-specific evidence allocation.
```

Use this distinction consistently:

| Concept | Meaning |
|---|---|
| Structural relation | Which atoms or residues are connected, close, or chemically plausible |
| Task evidence | Which relations actually help the current prediction |
| Reference distribution | A structure-induced default interaction distribution |
| Residual correction | A task-conditioned shift from the reference distribution |

Avoid saying:

```text
Structural priors are not useful.
```

Say instead:

```text
Structural priors are necessary references, but not task-evidence oracles.
```

---

## 4. TCM: Diagnostic Before Model

### 4.1 Why TCM Comes First

If the paper only says structure and task may be mismatched, it is just a story.

TCM turns this into a measurable diagnostic:

```text
Does an interaction distribution cover beneficial communication pairs?
Does it leak into harmful or redundant pairs?
```

This makes the paper more than a performance-driven adapter paper.

### 4.2 Frozen-Base Counterfactual Probe

Train a base model and freeze it.

For each candidate pair `p`, perform a small communication probe and compare loss:

```text
L0 = original frozen-base loss
Lp = loss after probing pair p

gp = L0 - Lp
```

Interpretation:

```text
gp > 0:
  probing pair p decreases the loss;
  p is beneficial communication evidence.

gp < 0:
  probing pair p increases the loss;
  p is harmful or redundant communication evidence.
```

### 4.3 TCM Functional

For an interaction distribution `q`:

```text
Benefit(q) = sum_p q_p [gp]_+ / (sum_p [gp]_+ + eps)

Harm(q) = sum_p q_p [-gp]_+ / (sum_p [-gp]_+ + eps)

TCM(q) = Benefit(q) - lambda * Harm(q)
```

Interpretation:

```text
High TCM:
  q allocates more flow to beneficial pairs and less flow to harmful pairs.

Low TCM:
  q misses beneficial pairs, leaks into harmful pairs, or both.
```

### 4.4 How to Report TCM

Primary mechanism report:

```text
Delta Benefit ↑
Delta Harm ↓
Delta TCM ↑
```

Use `×10^-3` units for readability:

```text
All TCM-family values are reported in 10^-3 units.
```

Do not treat TCM as a replacement for task metrics. The task metric remains primary; TCM is the mechanism evidence.

---

## 5. RePAIR / RIR: Residual Pairwise Interaction Rewiring

### 5.1 Reference Distribution

For candidate set `Omega_i`, define a structure-induced reference support:

```text
C_ref(i,j)
```

Convert it into a reference interaction distribution:

```text
q_ref(j|i) = softmax_{j in Omega_i}(log(C_ref(i,j) + eps))
```

`q_ref` is not the task-optimal distribution. It is the structural reference.

### 5.2 Signed Residual Shift

RePAIR learns a bounded signed residual logit shift:

```text
rho_ij = beta * tanh(phi_theta(i,j,h,c) / tau)
```

Interpretation:

```text
rho_ij > 0:
  pair (i,j) should be enhanced relative to the structural reference.

rho_ij < 0:
  pair (i,j) should be suppressed relative to the structural reference.
```

The bounded `tanh` form controls the magnitude of the task-conditioned correction.

### 5.3 Rewired Distribution

Instead of learning a free attention distribution from scratch, RePAIR performs reference-anchored tilting:

```text
q_rew(j|i) = softmax_{j in Omega_i}(log(q_ref(j|i) + eps) + rho_ij)
```

This says:

```text
Do not replace structure.
Shift the structural reference according to task-conditioned residual evidence.
```

### 5.4 Residual Interaction Operator

Define the interaction operator:

```text
T_q(h)_i = sum_{j in Omega_i} q(j|i) m_ij
```

Recommended message form:

```text
m_ij = W h_j - W h_i
```

RePAIR injects only the residual interaction flow:

```text
RePAIR(h)_i = T_q_rew(h)_i - T_q_ref(h)_i
```

Equivalently:

```text
RePAIR(h)_i = sum_j (q_rew(j|i) - q_ref(j|i)) m_ij
```

Final update:

```text
h'_i = h_i + gamma_i RePAIR(h)_i
```

### 5.5 Main Conceptual Point

The residual nature of RePAIR is not merely a skip connection.

It is residual because:

```text
the interaction operator itself is written as a deviation from a structural reference operator.
```

ResNet analogy:

```text
ResNet:
  H(x) = x + F(x)

RePAIR:
  h' = h + [T_q_rew(h) - T_q_ref(h)]
```

Suggested sentence:

```text
RePAIR extends the residual principle from feature mappings to interaction operators.
```

---

## 6. Candidate Pair Space

Candidate pairs define the search space where residual interaction correction is allowed.

They are not new chemical bonds.

They do not mean excluded pairs are impossible.

They define a constrained interaction space:

```text
RePAIR should correct interaction flow only within chemically or geometrically plausible candidates.
```

### Single-molecule tasks

```text
candidate pair = atom--atom pair
C_ref = bond / shortest-path / finite-hop structural support
```

Example:

```text
1 <= shortest_path_distance(i,j) <= d_max
```

### Affinity extension

```text
candidate pair = ligand atom -- pocket residue/contact pair
C_ref = distance/contact reference support
```

Do not use dense all-pair attention as the main formulation. Dense all-pair interaction should be an ablation.

---

## 7. Two Natural Instantiations in the PR Paper

The PR paper should include two instantiations, but without RPLI.

### 7.1 Intra-molecular RePAIR

Setting:

```text
single molecular graph
nodes = atoms
reference relation = intra-molecular chemistry
candidate pairs = atom--atom pairs
```

Purpose:

```text
Validate residual pairwise interaction rewiring in standard molecular graph learning.
```

Tasks:

```text
BBBP
ClinTox
FreeSolv
```

Additional MoleculeNet and regression datasets can be placed in appendix.

### 7.2 Cross-interface RePAIR

Setting:

```text
pocket-conditioned molecular graph learning
nodes = ligand atoms + pocket residues/nodes
reference relation = ligand--pocket contact
candidate pairs = ligand atom -- pocket residue
```

Purpose:

```text
Validate that the same residual interaction principle applies to ligand--pocket interface relations.
```

Use a generic affinity encoder in the PR paper. Do not describe the RPLI architecture in detail.

---

## 8. What the PR Paper Must Not Do

Do not foreground:

```text
RPLI backbone details
dual-head calibrated readout
single-stage affinity engineering
GEMS-style implementation details
RPLI naming and architecture
ESWA architecture motivation
```

Do not say:

```text
Backbone does not learn task.
```

Say:

```text
Backbone learns task-trained representations, but pairwise interaction allocation remains implicit and weakly supervised.
```

Do not say:

```text
The MLP learns true task demand D.
```

Say:

```text
The scorer parameterizes a signed residual shift relative to the structural reference.
```

Do not say:

```text
RIR is another attention layer.
```

Say:

```text
RePAIR is a reference-anchored residual interaction operator.
```

---

## 9. PR Contributions

### Contribution 1: Problem Formulation and Diagnostic

```text
We formulate task-conditioned pairwise interaction allocation as a missing layer in molecular graph learning and introduce TCM, a frozen-base counterfactual diagnostic for beneficial coverage and harmful leakage.
```

### Contribution 2: Residual Interaction Operator

```text
We propose RePAIR, a structure-referenced residual pairwise interaction operator that learns bounded signed shifts over a reference distribution and injects only the difference between rewired and reference interaction flows.
```

### Contribution 3: Signed Enhancement and Suppression

```text
RePAIR can enhance under-allocated beneficial pairs and suppress over-allocated harmful/redundant pairs, unlike positive-only rewiring or ordinary attention adapters.
```

### Contribution 4: Unified Molecular Graph Instantiations

```text
We instantiate the same principle for intra-molecular atom--atom rewiring and ligand--pocket interface rewiring, showing that task-conditioned interaction allocation is not limited to one molecular task.
```

---

## 10. PR Experimental Structure

### Table 1: Main Task Performance

Single-molecule:

```text
BBBP
ClinTox
FreeSolv
```

Affinity extension:

```text
PDBbind CleanSplit setting only if it supports the Cross-RePAIR mechanism cleanly.
```

Do not turn the PR paper into an affinity SOTA paper.

### Table 2: Single-Molecule Mechanism Ablation

Variants:

```text
Base
MLP adapter
ordinary attention
no-reference RePAIR
random C_ref
PositiveOnly
Full RePAIR
```

Expected conclusion:

```text
Full RePAIR should be best or most stable.
MLP controls parameter count.
Ordinary attention controls extra attention capacity.
No-reference controls residual-flow without structural reference.
Random C_ref tests whether correct structural reference matters.
PositiveOnly tests whether signed suppression is necessary.
```

### Table 3: Affinity Mechanism Ablation

Variants:

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
Cross-RIR should improve over Intra-RIR,
showing that interface residual rewiring is the main affinity mechanism.
Full should be strongest if intra- and cross-rewiring are complementary.
```

### Table 4: TCM Mechanism Table

Report:

```text
Delta Benefit ×10^3 ↑
Delta Harm ×10^3 ↓
Delta TCM ×10^3 ↑
Improved seeds or paired consistency if available
```

### Figure 1: Method Overview

Show:

```text
C_ref -> q_ref
rho -> q_rew
T_q_rew - T_q_ref
h' = h + residual interaction flow
TCM evidence alignment
```

### Figure 2: TCM Alignment Scatter

```text
x-axis: g_p = L0 - Lp
y-axis: Delta q_p = q_rew(p) - q_ref(p)
```

Expected pattern:

```text
g_p > 0 pairs should tend to have positive Delta q.
g_p < 0 pairs should tend to have negative Delta q.
```

### Figure 3: Single-Molecule Visualization

Show:

```text
molecular graph
structural reference edges
enhanced residual pairs
suppressed residual pairs
TCM beneficial/harmful evidence
```

### Figure 4: Interface Visualization

Show:

```text
ligand--pocket contact map
C_ref^LP
Delta q^LP
Interface TCM evidence
```

---

## 11. How to Answer Major Reviewer Concerns

### Concern 1: Is this just ordinary attention?

Answer:

```text
No. Ordinary attention learns q_att from scratch.
RePAIR first defines q_ref from structure, learns a bounded residual shift, and injects only T_q_rew - T_q_ref.
```

Required evidence:

```text
ordinary attention ablation
no-reference ablation
random C_ref ablation
TCM alignment
```

### Concern 2: Is residual correction just task overfitting?

Answer:

```text
Task-conditioned correction is not automatically overfitting.
RePAIR constrains correction by candidate pairs, structural reference anchoring, bounded tanh shifts, gate control, and residual-flow injection.
```

Required evidence:

```text
MLP adapter control
ordinary attention control
random C_ref control
held-out task performance
TCM Benefit/Harm decomposition
correction magnitude analysis if space allows
```

### Concern 3: Does TCM just explain the model using itself?

Answer:

```text
TCM uses a frozen-base counterfactual probe and is not used as the main training objective.
It evaluates whether a distribution aligns with externally probed pair evidence.
```

### Concern 4: Are single-molecule and affinity two unrelated stories?

Answer:

```text
No. They are two interaction spaces under the same abstraction:
  single molecule: atom--atom reference relation;
  affinity: ligand--pocket reference relation.
Both use q_ref, signed residual shift, and residual interaction flow.
```

---

## 12. Intro Skeleton

### Paragraph 1: Structure Is Central

Molecular graph learning relies on structural priors such as covalent topology, local neighborhoods, geometric proximity, and ligand--pocket contacts.

### Paragraph 2: Structure Is Reference, Not Oracle

These priors define chemically meaningful reference interactions, but they do not specify which pairwise relations are beneficial for each downstream task.

### Paragraph 3: Pair-Level Allocation Is Weakly Supervised

Most molecular tasks provide graph-level or complex-level labels, leaving pairwise interaction allocation implicit and weakly supervised.

### Paragraph 4: TCM Diagnostic

We introduce TCM to diagnose whether an interaction distribution covers beneficial communication pairs and avoids harmful leakage.

### Paragraph 5: RePAIR Method

Motivated by TCM, we propose RePAIR, which learns signed residual shifts over a structure-induced reference distribution and injects only the difference between rewired and reference interaction operators.

### Paragraph 6: Scope

We instantiate RePAIR for intra-molecular atom--atom rewiring and ligand--pocket interface rewiring, validating the same mechanism in molecular property prediction and pocket-conditioned molecular graph learning.

---

## 13. Abstract Skeleton

```text
Molecular graph learning increasingly benefits from stronger structural priors, but structure-induced interactions are task-agnostic references rather than task-specific evidence allocations. We study task-conditioned pairwise interaction allocation and introduce TCM, a frozen-base counterfactual diagnostic that measures whether an interaction distribution covers beneficial communication pairs and avoids harmful leakage. Motivated by this diagnosis, we propose RePAIR, a residual pairwise interaction rewiring operator. RePAIR learns bounded signed shifts over a structure-induced reference distribution and injects only the difference between rewired and reference interaction flows. This design preserves structural priors while enabling task-conditioned correction. We instantiate RePAIR for intra-molecular atom-pair rewiring and ligand--pocket interface rewiring, and validate it through task performance, ablations against ordinary adapters and attention, TCM decomposition, and residual-flow visualization.
```

---

## 14. Final PR Positioning

The final PR paper should be positioned as:

```text
A mechanism paper on diagnosing and correcting task-conditioned pairwise interaction allocation in molecular graph learning.
```

It should not be positioned as:

```text
A new affinity architecture paper.
A stronger backbone paper.
A plug-in adapter paper only.
```

Final one-line positioning:

```text
TCM diagnoses where interaction flow is task-mismatched;
RePAIR repairs it through signed residual rewiring over structural references.
```
