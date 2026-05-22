import argparse
import csv
import json
from argparse import Namespace
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from coremol.datasets.pdbbind_gems import load_affinity_dataset
from coremol.metrics.affinity_metrics import affinity_regression_metrics
from scripts.train_coremol_affinity import _build_model, _pk, _target_scaled


def build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate CoReMol-Net-Affinity checkpoint")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save_dir", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    return parser


@torch.no_grad()
def main(argv: list[str] | None = None) -> Path:
    args = build_eval_parser().parse_args(argv)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_args = Namespace(**checkpoint["args"])

    dataset = load_affinity_dataset(args.dataset_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _build_model(train_args, dataset[0]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()

    ids = []
    true_pk = []
    pred_pk = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        ids.extend([str(item) for item in batch.id])
        true_pk.extend(_pk(_target_scaled(batch)).tolist())
        pred_pk.extend(_pk(pred).tolist())

    metrics = affinity_regression_metrics(true_pk, pred_pk)
    save_dir = args.save_dir or Path(args.checkpoint).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "eval_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (save_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "true_pk", "pred_pk"])
        writer.writeheader()
        for complex_id, y_true, y_pred in zip(ids, true_pk, pred_pk):
            writer.writerow({"id": complex_id, "true_pk": y_true, "pred_pk": y_pred})
    return save_dir


if __name__ == "__main__":
    main()
