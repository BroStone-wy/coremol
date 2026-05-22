import argparse
import csv
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from coremol.datasets.pdbbind_gems import build_fold_datasets, load_affinity_dataset
from coremol.metrics.affinity_metrics import affinity_regression_metrics
from coremol.models.coremol_net_affinity import CoReMolNetAffinity, CoReMolNetAffinityConfig
from coremol.modules.cross_rir_adapter import CrossRIRConfig
from coremol.utils.seed import set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CoReMol-Net-Affinity on GEMS/PDBbind datasets")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--variant", choices=["base", "coremol"], default="coremol")
    parser.add_argument("--backbone", choices=["gine", "rpli"], default="gine")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--loss_func", choices=["mse", "rmse", "l1", "huber"], default="mse")
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--context_channels", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--conv_dropout", type=float, default=0.0)
    parser.add_argument("--cross_beta", type=float, default=0.5)
    parser.add_argument("--cross_tau", type=float, default=0.5)
    parser.add_argument("--cross_gate_init", type=float, default=0.05)
    parser.add_argument("--cross_gate_max", type=float, default=0.5)
    parser.add_argument("--cross_position", choices=["mid", "post"], default="mid")
    parser.add_argument("--cross_update", choices=["residual", "readout"], default="residual")
    parser.add_argument("--interface_gate_init", type=float, default=1.0)
    parser.add_argument("--readout_mode", choices=["joint", "residual", "dual"], default="joint")
    parser.add_argument("--residual_gate_init", type=float, default=0.05)
    parser.add_argument("--final_blend_alpha", type=float, default=0.6)
    parser.add_argument("--aux_mse_weight", type=float, default=0.0)
    parser.add_argument("--base_aux_weight", type=float, default=0.0)
    parser.add_argument("--coremol_aux_weight", type=float, default=0.0)
    parser.add_argument("--selection_metric", choices=["rmse", "total_loss"], default="rmse")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--limit_train", type=int, default=0)
    parser.add_argument("--limit_valid", type=int, default=0)
    parser.add_argument("--init_checkpoint", type=Path, default=None)
    parser.add_argument("--freeze_backbone_for_adapter", action="store_true")
    parser.add_argument("--save_dir", type=Path, default=Path("results/coremol_net_affinity_gems"))
    parser.add_argument("--num_workers", type=int, default=0)
    return parser


def _target_scaled(batch) -> torch.Tensor:
    return batch.y.view(-1, 1).float()


def _pk(values: torch.Tensor) -> torch.Tensor:
    return values.detach().cpu().view(-1).float() * 16.0


def _build_model(args, sample) -> CoReMolNetAffinity:
    edge_dim = int(sample.edge_attr.size(-1))
    in_channels = int(sample.x.size(-1))
    ligand_global_dim = 0
    lig_emb = getattr(sample, "lig_emb", None)
    if lig_emb is not None:
        ligand_global_dim = int(torch.as_tensor(lig_emb).view(1, -1).size(-1))
    cross_rir = CrossRIRConfig(
        enabled=getattr(args, "variant", "coremol") == "coremol",
        beta=getattr(args, "cross_beta", 0.5),
        tau=getattr(args, "cross_tau", 0.5),
        gate_init=getattr(args, "cross_gate_init", 0.05),
        gate_max=getattr(args, "cross_gate_max", 0.5),
        dropout=getattr(args, "dropout", 0.1),
    )
    if getattr(args, "backbone", "gine") == "rpli":
        from coremol.models.rpli_affinity import RPLIAffinity, RPLIAffinityConfig

        config = RPLIAffinityConfig(
            hidden_channels=args.hidden_channels,
            context_channels=args.context_channels,
            dropout=args.dropout,
            conv_dropout=args.conv_dropout,
            use_cross_rir=args.variant == "coremol",
            cross_position=getattr(args, "cross_position", "mid"),
            cross_update=getattr(args, "cross_update", "residual"),
            interface_gate_init=getattr(args, "interface_gate_init", 1.0),
            readout_mode=getattr(args, "readout_mode", "joint"),
            residual_gate_init=getattr(args, "residual_gate_init", 0.05),
            final_blend_alpha=getattr(args, "final_blend_alpha", 0.6),
            cross_rir=cross_rir,
        )
        return RPLIAffinity(
            in_channels=in_channels,
            edge_dim=edge_dim,
            ligand_global_dim=ligand_global_dim,
            config=config,
        )
    config = CoReMolNetAffinityConfig(
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_cross_rir=args.variant == "coremol",
        cross_rir=cross_rir,
    )
    return CoReMolNetAffinity(
        in_channels=in_channels,
        edge_dim=edge_dim,
        ligand_global_dim=ligand_global_dim,
        config=config,
    )


class RMSELoss(torch.nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)
        self.mse = torch.nn.MSELoss()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(output, target) + self.eps)


def _build_loss(name: str) -> torch.nn.Module:
    if name == "rmse":
        return RMSELoss()
    if name == "l1":
        return torch.nn.L1Loss()
    if name == "huber":
        return torch.nn.HuberLoss(delta=1.0)
    return torch.nn.MSELoss()


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    true_pk = []
    pred_pk = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        true_pk.append(_pk(_target_scaled(batch)))
        pred_pk.append(_pk(pred))
    return affinity_regression_metrics(torch.cat(true_pk).numpy(), torch.cat(pred_pk).numpy())


def _combined_loss(pred: torch.Tensor, target: torch.Tensor, diagnostics: dict, loss_fn, args) -> torch.Tensor:
    loss = loss_fn(pred, target)
    if args.aux_mse_weight > 0:
        loss = loss + args.aux_mse_weight * torch.nn.functional.mse_loss(pred, target)
    if args.base_aux_weight > 0 and "base_pred" in diagnostics:
        loss = loss + args.base_aux_weight * RMSELoss()(diagnostics["base_pred"], target)
    if args.coremol_aux_weight > 0 and "coremol_pred" in diagnostics:
        loss = loss + args.coremol_aux_weight * torch.nn.functional.mse_loss(diagnostics["coremol_pred"], target)
    return loss


@torch.no_grad()
def evaluate_losses(model, loader, device, loss_fn, args) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    for batch in loader:
        batch = batch.to(device)
        output = model(batch, return_diagnostics=True)
        pred, diagnostics = output
        target = _target_scaled(batch).to(device)
        loss = _combined_loss(pred, target, diagnostics, loss_fn, args)
        total_loss += float(loss.item())
        count += 1
    return total_loss / max(count, 1)


def _selection_score(row: dict[str, float], metric: str) -> float:
    if metric == "total_loss":
        return float(row["valid_total_loss"])
    return float(row["valid_rmse"])


def train(args) -> Path:
    set_seed(args.seed)
    dataset = load_affinity_dataset(args.dataset_path)
    train_set, valid_set = build_fold_datasets(dataset, fold=args.fold, n_folds=args.n_folds, seed=args.seed)
    if args.limit_train > 0:
        train_set = train_set[: args.limit_train]
    if args.limit_valid > 0:
        valid_set = valid_set[: args.limit_valid]

    run_dir = args.save_dir / f"{args.run_name}_{args.variant}_fold{args.fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _build_model(args, train_set[0]).to(device)
    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"], strict=False)
    if args.freeze_backbone_for_adapter:
        adapter_prefixes = ("cross_rir", "interface_gate", "head", "base_head", "residual_head", "residual_gate")
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(adapter_prefixes)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable parameters remain after applying freeze settings")
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = _build_loss(args.loss_func)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_score = float("inf")
    best_rmse = float("inf")
    best_epoch = 0
    stale = 0
    rows = []
    if args.init_checkpoint is not None:
        valid = evaluate(model, valid_loader, device)
        row = {
            "epoch": 0,
            "train_loss": 0.0,
            "valid_total_loss": evaluate_losses(model, valid_loader, device, loss_fn, args),
        }
        row.update({f"valid_{key}": value for key, value in valid.items()})
        rows.append(row)
        best_rmse = valid["rmse"]
        best_score = _selection_score(row, args.selection_metric)
        torch.save(
            {
                "model_state": model.state_dict(),
                "args": vars(args),
                "in_channels": int(train_set[0].x.size(-1)),
                "edge_dim": int(train_set[0].edge_attr.size(-1)),
                "ligand_global_dim": int(getattr(model, "ligand_global_dim", 0)),
                "best_epoch": best_epoch,
                "best_valid_rmse": best_rmse,
                "best_valid_score": best_score,
            },
            run_dir / "best_model.pt",
        )
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = batch.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                if args.readout_mode == "dual" or args.base_aux_weight > 0 or args.coremol_aux_weight > 0:
                    pred, diagnostics = model(batch, return_diagnostics=True)
                else:
                    pred = model(batch)
                    diagnostics = {}
                target = _target_scaled(batch).to(device)
                combined_loss = _combined_loss(pred, target, diagnostics, loss_fn, args)
                loss = combined_loss / max(args.grad_accum_steps, 1)
            scaler.scale(loss).backward()
            if step % max(args.grad_accum_steps, 1) == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.item()) * max(args.grad_accum_steps, 1)

        valid = evaluate(model, valid_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(train_loader), 1),
            "valid_total_loss": evaluate_losses(model, valid_loader, device, loss_fn, args),
        }
        row.update({f"valid_{key}": value for key, value in valid.items()})
        rows.append(row)
        score = _selection_score(row, args.selection_metric)
        if score < best_score:
            best_score = score
            best_rmse = valid["rmse"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "in_channels": int(train_set[0].x.size(-1)),
                    "edge_dim": int(train_set[0].edge_attr.size(-1)),
                    "ligand_global_dim": int(getattr(model, "ligand_global_dim", 0)),
                    "best_epoch": best_epoch,
                    "best_valid_rmse": best_rmse,
                    "best_valid_score": best_score,
                },
                run_dir / "best_model.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                break

    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return run_dir


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    return train(args)


if __name__ == "__main__":
    main()
