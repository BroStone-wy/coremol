# CoReMol Environment

Recommended environment name: `GNRF`.

## Create Environment

```bash
conda create -n GNRF python=3.11 -y
conda activate GNRF
```

## Install PyTorch

Install the PyTorch build that matches the target server CUDA version from the official PyTorch selector.

Example for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

CPU-only fallback:

```bash
pip install torch torchvision torchaudio
```

## Install PyTorch Geometric

```bash
pip install torch-geometric
```

If the target server requires compiled PyG extensions such as `torch-scatter`, install wheels matching the exact PyTorch and CUDA versions from the PyG wheel index.

## Install Project Requirements

```bash
pip install -r requirements.txt
```

## Verify Environment

```bash
python - <<'PY'
import torch
import torch_geometric
import rdkit

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("pyg", torch_geometric.__version__)
print("rdkit ok")
PY
```

Run tests:

```bash
PYTHONPATH=. pytest -q tests
```

## Prepare Datasets

Datasets are not committed to GitHub. They are downloaded and processed locally on each server.

```bash
PYTHONPATH=. python scripts/prepare_datasets.py
```

Generated data is written under `data/`, which is intentionally ignored by git.

## Common Training Commands

Classification example:

```bash
PYTHONPATH=. python scripts/run_stage1_gate.py \
  --backbone attentivefp \
  --datasets BBBP \
  --seeds 0 1 2 \
  --variants base coremol
```

Graphformer regression example:

```bash
PYTHONPATH=. python scripts/run_stage1_gate.py \
  --backbone graphformer \
  --datasets ESOL FREESOLV LIPO \
  --seeds 0 1 2 \
  --split_strategy random \
  --random_train_fraction 0.8 \
  --random_valid_fraction 0.1 \
  --normalize_regression \
  --variants base coremol
```

Full TCM post-hoc example:

```bash
PYTHONPATH=. python scripts/compute_tcm_variants.py \
  --run_dir <RUN_DIR> \
  --datasets <DATASET> \
  --seeds 0 1 2 \
  --backbone graphformer \
  --max_graphs 96
```

Use `reports/graphformer_regression_full_tcm_summary_2026_05_20.md` for a complete saved Graphformer regression configuration and exact reproduction commands.

