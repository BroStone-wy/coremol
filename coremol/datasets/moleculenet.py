from pathlib import Path

from torch_geometric.datasets import MoleculeNet


TASKS = {
    "BACE": {"type": "classification", "metric": "roc_auc", "target_dim": 1},
    "BBBP": {"type": "classification", "metric": "roc_auc", "target_dim": 1},
    "CLINTOX": {"type": "classification", "metric": "roc_auc", "target_dim": 2, "pyg_name": "ClinTox"},
    "ESOL": {"type": "regression", "metric": "rmse", "target_dim": 1},
    "FREESOLV": {"type": "regression", "metric": "rmse", "target_dim": 1, "pyg_name": "FreeSolv"},
    "HIV": {"type": "classification", "metric": "roc_auc", "target_dim": 1},
    "LIPO": {"type": "regression", "metric": "rmse", "target_dim": 1, "pyg_name": "lipo"},
    "LIPOPHILICITY": {"type": "regression", "metric": "rmse", "target_dim": 1, "pyg_name": "lipo"},
    "SIDER": {"type": "classification", "metric": "roc_auc", "target_dim": 27},
    "TOX21": {"type": "classification", "metric": "roc_auc", "target_dim": 12},
    "TOXCAST": {"type": "classification", "metric": "roc_auc", "target_dim": 617, "pyg_name": "ToxCast"},
}


def load_moleculenet(dataset_name: str, root: str | Path):
    name = dataset_name.upper()
    if name not in TASKS:
        raise ValueError(f"Unsupported dataset {dataset_name}. Expected one of {sorted(TASKS)}.")
    pyg_name = TASKS[name].get("pyg_name", name)
    return MoleculeNet(root=str(Path(root)), name=pyg_name)


def infer_feature_dims(dataset) -> tuple[int, int]:
    sample = dataset[0]
    return int(sample.x.size(-1)), int(sample.edge_attr.size(-1))
