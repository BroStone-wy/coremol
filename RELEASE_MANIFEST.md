# Release Manifest

This repository is intended to preserve the CoReMol source code, experiment configs, tests, reports, and setup instructions needed to run the same core implementation on another server.

## Included

- `coremol/`: CoReMol modules, models, metrics, probes, dataset split utilities, and training utilities.
- `scripts/`: dataset preparation, training entry points, sweeps, pretraining utilities, and Full TCM computation.
- `configs/`: experiment configuration files used during classification and regression runs.
- `tests/`: unit and integration tests for core modules, metrics, scripts, and model wrappers.
- `docs/`: project plans and user requirement memory.
- `reports/`: selected experiment summaries and reproducibility reports.
- `README.md`, `ENVIRONMENT.md`, and `requirements.txt`: setup and run instructions.
- `coremol_stage1_algorithm_experiment_plan.md`, `coremol_tcm_metric_design.md`, and `RIR_CoReMol_latest_storyline_revision.md`: design notes and current storyline.

## Excluded

- `data/`: downloaded datasets and processed local artifacts.
- `results/`: generated training runs, checkpoints, and local screening artifacts.
- `third_party/`: downloaded reference repositories.
- `*.pt`, `*.pth`, `*.ckpt`: model checkpoints and binary weights.
- Python caches and test caches.

## Reproducing Local Artifacts

On a fresh server:

```bash
git clone <github-url> coremol
cd coremol
conda create -n GNRF python=3.11 -y
conda activate GNRF
pip install -r requirements.txt
PYTHONPATH=. python scripts/prepare_datasets.py
PYTHONPATH=. pytest -q tests
```

See `ENVIRONMENT.md` for PyTorch/CUDA-specific installation details and example training commands.

