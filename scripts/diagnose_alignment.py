import sys
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coremol.datasets.moleculenet import TASKS, infer_feature_dims, load_moleculenet
from coremol.datasets.scaffold_split import load_or_create_split
from coremol.models.attentivefp_coremol import CoReMolAttentiveFP, CoReMolConfig
from coremol.probes.tcm import _pair_sensitivity_from_base_model


def build(dataset, variant):
    in_channels, edge_dim = infer_feature_dims(dataset)
    return CoReMolAttentiveFP(
        in_channels=in_channels,
        edge_dim=edge_dim,
        hidden_channels=32,
        out_channels=1,
        num_layers=2,
        num_timesteps=2,
        dropout=0.1,
        coremol=CoReMolConfig(enabled=(variant == "coremol"), beta=0.2, tau=0.5),
    )


def corr(a, b):
    if a.numel() < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(torch.corrcoef(torch.stack([a.float(), b.float()]))[0, 1].cpu())


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    ckpt_dir = ROOT / "results" / "stage1_gate" / "checkpoints"
    for dataset_name in ["BBBP", "ESOL"]:
        dataset = load_moleculenet(dataset_name, ROOT / "data" / "moleculenet")
        task_type = TASKS[dataset_name]["type"]
        for seed in [0, 1, 2]:
            split = load_or_create_split(dataset, dataset_name, seed, ROOT / "data" / "splits")
            test_set = [dataset[i] for i in split["test"] if dataset[i].num_nodes > 0]
            base = build(dataset, "base").to(device)
            cal = build(dataset, "coremol").to(device)
            base.load_state_dict(torch.load(ckpt_dir / f"{dataset_name}_{seed}_base.pt", map_location=device))
            cal.load_state_dict(torch.load(ckpt_dir / f"{dataset_name}_{seed}_coremol.pt", map_location=device))
            base.eval()
            cal.eval()
            cors = []
            for data in DataLoader(test_set[:24], batch_size=1):
                data = data.to(device)
                _, diag = cal(data, return_diagnostics=True)
                if not diag:
                    continue
                item = diag[0]
                sensitivity = _pair_sensitivity_from_base_model(base, data, item["pairs"], task_type)
                cors.append(corr(item["residual_score"], sensitivity))
            rows.append({"dataset": dataset_name, "seed": seed, "residual_sensitivity_corr": pd.Series(cors).mean()})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "stage1_gate" / "alignment_diagnostics.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()

