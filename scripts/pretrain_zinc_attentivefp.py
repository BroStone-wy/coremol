import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coremol.models.attentivefp_coremol import CoReMolAttentiveFP, CoReMolConfig
from coremol.training.pretraining import load_zinc_subset
from coremol.training.trainer import train_model
from coremol.utils.seed import set_seed


def build_zinc_model(args):
    return CoReMolAttentiveFP(
        in_channels=1,
        edge_dim=1,
        hidden_channels=args.hidden_channels,
        out_channels=1,
        num_layers=args.num_layers,
        num_timesteps=args.num_timesteps,
        dropout=args.dropout,
        coremol=CoReMolConfig(enabled=False),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_timesteps", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--out_dir", type=Path, default=ROOT / "results" / "zinc_pretrain")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    zinc = load_zinc_subset(ROOT / "data" / "zinc")
    model = build_zinc_model(args)
    outcome = train_model(
        model=model,
        train_dataset=zinc["train"],
        valid_dataset=zinc["valid"],
        test_dataset=zinc["test"],
        task_type="regression",
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        log_prefix=f"ZINC/seed{args.seed}/attentivefp",
        normalize_regression=True,
    )

    checkpoint = {
        "model_state_dict": outcome["model"].state_dict(),
        "target_scaler": None,
        "args": vars(args),
        "valid": outcome["valid"],
        "test": outcome["test"],
    }
    ckpt_path = args.out_dir / f"attentivefp_zinc_subset_seed{args.seed}.pt"
    torch.save(checkpoint, ckpt_path)
    pd.DataFrame(
        [
            {
                "dataset": "ZINC_SUBSET",
                "seed": args.seed,
                **{f"valid_{key}": value for key, value in outcome["valid"].items()},
                **{f"test_{key}": value for key, value in outcome["test"].items()},
                "checkpoint": str(ckpt_path),
            }
        ]
    ).to_csv(args.out_dir / f"metrics_seed{args.seed}.csv", index=False)
    print({"checkpoint": str(ckpt_path), "valid": outcome["valid"], "test": outcome["test"]}, flush=True)


if __name__ == "__main__":
    main()
