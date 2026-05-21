# CoReMol GitHub Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a clean GitHub repository containing the original CoReMol code, configs, tests, experiment reports, and environment instructions so another server can clone it and run new datasets with identical core code.

**Architecture:** Package the `CORMOL` project as the repository root, while excluding local caches, downloaded datasets, checkpoints, and bulky generated artifacts. Reproducibility is handled by committed scripts/configs plus an environment/setup markdown file and data preparation commands.

**Tech Stack:** Python, PyTorch, PyTorch Geometric, RDKit, pandas, scikit-learn, pytest, conda, GitHub CLI or GitHub remote.

---

### Task 1: Confirm Release Scope

**Files:**
- Review: `CORMOL/coremol/`
- Review: `CORMOL/scripts/`
- Review: `CORMOL/configs/`
- Review: `CORMOL/tests/`
- Review: `CORMOL/reports/`
- Review: `CORMOL/results/`

- [ ] **Step 1: Keep core code and runnable assets**

Include these paths in the repository:

```text
CORMOL/README.md
CORMOL/requirements.txt
CORMOL/coremol/
CORMOL/scripts/
CORMOL/configs/
CORMOL/tests/
CORMOL/docs/
CORMOL/reports/
CORMOL/coremol_stage1_algorithm_experiment_plan.md
CORMOL/coremol_tcm_metric_design.md
CORMOL/RIR_CoReMol_latest_storyline_revision.md
```

- [ ] **Step 2: Exclude bulky/generated local artifacts**

Exclude these paths from normal Git tracking:

```text
CORMOL/data/
CORMOL/results/**/checkpoints/
CORMOL/**/*.pt
CORMOL/**/*.pth
CORMOL/**/__pycache__/
CORMOL/.pytest_cache/
CORMOL/third_party/
```

- [ ] **Step 3: Preserve lightweight result summaries**

Keep selected result summaries and reports for reference:

```text
CORMOL/reports/*.md
CORMOL/reports/*.csv
CORMOL/results/final_reports/*.md
CORMOL/results/*summary*.md
CORMOL/results/*record*.md
```

### Task 2: Add Environment Documentation

**Files:**
- Create: `CORMOL/ENVIRONMENT.md`
- Modify: `CORMOL/README.md`

- [ ] **Step 1: Document conda environment**

Create `CORMOL/ENVIRONMENT.md` with:

```markdown
# CoReMol Environment

Recommended environment name: `GNRF`

## Create Environment

```bash
conda create -n GNRF python=3.11 -y
conda activate GNRF
```

## Install PyTorch

Install the PyTorch build matching the target server CUDA version from https://pytorch.org/get-started/locally/.

Example for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

CPU-only fallback:

```bash
pip install torch torchvision torchaudio
```

## Install PyG

Use the wheel URL matching the installed PyTorch and CUDA versions:

```bash
pip install torch-geometric
```

If `torch-scatter` or related extensions are required by the server, install from the PyG wheel index matching the exact PyTorch version.

## Install Project Requirements

```bash
pip install -r requirements.txt
```

## Verify

```bash
python - <<'PY'
import torch
import torch_geometric
import rdkit
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('pyg', torch_geometric.__version__)
print('rdkit ok')
PY
pytest -q tests
```
```

- [ ] **Step 2: Document dataset preparation**

Add commands:

```bash
PYTHONPATH=. python scripts/prepare_datasets.py
```

For MoleculeNet tasks, note that datasets are downloaded/processed locally under `data/`.

- [ ] **Step 3: Document common runs**

Add example commands for:

```bash
PYTHONPATH=. python scripts/run_stage1_gate.py --backbone attentivefp --datasets BBBP --seeds 0 1 2
PYTHONPATH=. python scripts/run_stage1_gate.py --backbone graphformer --datasets ESOL FREESOLV LIPO --seeds 0 1 2 --normalize_regression
PYTHONPATH=. python scripts/compute_tcm_variants.py --run_dir <RUN_DIR> --datasets <DATASET> --seeds 0 1 2 --backbone graphformer
```

### Task 3: Add Git Ignore and Release Manifest

**Files:**
- Create: `CORMOL/.gitignore`
- Create: `CORMOL/RELEASE_MANIFEST.md`

- [ ] **Step 1: Create `.gitignore`**

Use:

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.ipynb_checkpoints/

data/
third_party/

results/**/checkpoints/
*.pt
*.pth
*.ckpt

*.log
wandb/
runs/
```

- [ ] **Step 2: Create `RELEASE_MANIFEST.md`**

Document included and excluded paths:

```markdown
# Release Manifest

Included:
- `coremol/`: CoReMol modules, models, metrics, probes, datasets, training code.
- `scripts/`: dataset preparation, training, sweeps, pretraining, TCM computation.
- `configs/`: experiment configs used during classification/regression runs.
- `tests/`: unit and integration tests.
- `docs/` and `reports/`: design notes, user requirements memory, and selected experiment summaries.

Excluded:
- `data/`: downloaded datasets and processed local artifacts.
- `results/**/checkpoints/`: model checkpoints.
- `third_party/`: downloaded reference repositories.
- Python caches and test caches.

To reproduce data and third-party references, use the scripts documented in `ENVIRONMENT.md`.
```

### Task 4: Initialize Clean Git Repository

**Files:**
- Create/modify: local `.git` metadata inside `CORMOL/`

- [ ] **Step 1: Initialize Git inside `CORMOL`**

Run:

```bash
cd /home/legion-w4/LabData/Stone/BORF/CORMOL
git init
git status --short
```

Expected: Git repository initialized and files shown as untracked.

- [ ] **Step 2: Add clean tracked files**

Run:

```bash
git add README.md ENVIRONMENT.md RELEASE_MANIFEST.md requirements.txt .gitignore
git add coremol scripts configs tests docs reports
git add coremol_stage1_algorithm_experiment_plan.md coremol_tcm_metric_design.md RIR_CoReMol_latest_storyline_revision.md
git status --short
```

Expected: no `data/`, no checkpoint files, no `third_party/`, no cache files staged.

- [ ] **Step 3: Commit**

Run:

```bash
git commit -m "feat: release coremol code and experiment configs"
```

Expected: clean commit containing only source, configs, docs, tests, and selected reports.

### Task 5: Create GitHub Repository and Push

**Files:**
- No source file changes.

- [ ] **Step 1: Confirm target**

Need user confirmation for:

```text
GitHub owner/account:
Repository name:
Visibility: private or public
```

- [ ] **Step 2: Create remote repository**

Preferred if GitHub CLI is authenticated:

```bash
gh repo create <owner>/<repo> --private --source . --remote origin --push
```

For public:

```bash
gh repo create <owner>/<repo> --public --source . --remote origin --push
```

Fallback if user provides an existing empty repo URL:

```bash
git remote add origin <github-url>
git branch -M main
git push -u origin main
```

- [ ] **Step 3: Verify remote clone instructions**

Run:

```bash
git remote -v
git status --short
git ls-files | grep -E '(^data/|checkpoints/|\\.pt$|\\.pth$|^third_party/)' || true
```

Expected: remote is set, working tree clean, and no excluded heavy artifacts are tracked.

### Task 6: Server Download Handoff

**Files:**
- Modify: `CORMOL/README.md`

- [ ] **Step 1: Add clone/setup instructions**

Add:

```markdown
## Fresh Server Setup

```bash
git clone <github-url> CORMOL
cd CORMOL
conda create -n GNRF python=3.11 -y
conda activate GNRF
pip install -r requirements.txt
PYTHONPATH=. python scripts/prepare_datasets.py
pytest -q tests
```
```

- [ ] **Step 2: Add run instructions for new datasets**

Add:

```markdown
## Run New Dataset

```bash
PYTHONPATH=. python scripts/run_stage1_gate.py \
  --backbone graphformer \
  --datasets <DATASET> \
  --seeds 0 1 2 \
  --variants base coremol
```
```

---

## Confirmation Needed

Before execution, confirm:

1. Repository root should be the contents of `CORMOL/`, not the parent `BORF/`.
2. Heavy local data/checkpoints should be excluded from GitHub and regenerated on the target server.
3. Provide target GitHub owner/account, repository name, and visibility.
