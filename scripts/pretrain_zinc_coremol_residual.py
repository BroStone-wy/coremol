import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coremol.models.attentivefp_coremol import CoReMolAttentiveFP, CoReMolConfig
from coremol.training.pretraining import freeze_backbone, load_matching_backbone_state, load_zinc_subset
from coremol.training.trainer import train_model
from coremol.utils.seed import set_seed


def build_model(args):
    return CoReMolAttentiveFP(
        in_channels=1,
        edge_dim=1,
        hidden_channels=args.hidden_channels,
        out_channels=1,
        num_layers=args.num_layers,
        num_timesteps=args.num_timesteps,
        dropout=args.dropout,
        coremol=CoReMolConfig(
            enabled=True,
            d_max=args.d_max,
            support_hops=args.support_hops,
            beta=args.beta,
            tau=args.tau,
            residual_gate_init=args.residual_gate_init,
            residual_placement=args.residual_placement,
            num_residual_steps=args.num_residual_steps,
            residual_message=args.residual_message,
            residual_score_space=args.residual_score_space,
            residual_shift_centering=args.residual_shift_centering,
            residual_norm_mode=args.residual_norm_mode,
            dropout=args.dropout,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_timesteps", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--d_max", type=int, default=4)
    parser.add_argument("--support_hops", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--residual_gate_init", type=float, default=0.1)
    parser.add_argument("--residual_placement", choices=["post", "layerwise", "both"], default="post")
    parser.add_argument("--num_residual_steps", type=int, default=1)
    parser.add_argument("--residual_message", choices=["value", "delta"], default="delta")
    parser.add_argument("--residual_score_space", choices=["intensity", "distribution", "rir"], default="intensity")
    parser.add_argument("--residual_shift_centering", choices=["none", "source"], default="none")
    parser.add_argument("--residual_norm_mode", choices=["layernorm", "none"], default="layernorm")
    parser.add_argument("--pretrained_backbone", type=Path, default=None)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--out_dir", type=Path, default=ROOT / "results" / "zinc_coremol_residual_pretrain")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    zinc = load_zinc_subset(ROOT / "data" / "zinc")
    model = build_model(args)
    report = {"loaded": 0, "skipped": []}
    if args.pretrained_backbone is not None:
        pretrained_state = torch.load(args.pretrained_backbone, map_location="cpu")
        report = load_matching_backbone_state(model, pretrained_state)
    if args.freeze_backbone:
        trainable = freeze_backbone(model)
    else:
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    print(
        {
            "loaded_backbone": report["loaded"],
            "skipped": len(report["skipped"]),
            "freeze_backbone": args.freeze_backbone,
            "trainable": trainable,
        },
        flush=True,
    )

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
        log_prefix=f"ZINC/seed{args.seed}/coremol_residual",
        normalize_regression=True,
    )

    ckpt_path = args.out_dir / f"coremol_residual_zinc_subset_seed{args.seed}.pt"
    torch.save(
        {
            "model_state_dict": outcome["model"].state_dict(),
            "args": vars(args),
            "valid": outcome["valid"],
            "test": outcome["test"],
        },
        ckpt_path,
    )
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
