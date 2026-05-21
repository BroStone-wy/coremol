import argparse
import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coremol.datasets.moleculenet import TASKS, infer_feature_dims, load_moleculenet
from coremol.datasets.random_split import load_or_create_random_split
from coremol.datasets.scaffold_split import load_or_create_split
from coremol.metrics.mechanism import mechanism_summary
from coremol.models.attentivefp_coremol import CoReMolAttentiveFP, CoReMolConfig
from coremol.models.dmpnn_coremol import CoReMolDMPNN
from coremol.models.graphformer_coremol import CoReMolGraphformer
from coremol.probes.tcm import estimate_tcm_at_k
from coremol.training.trainer import RegressionTargetScaler, evaluate_model, fine_tune_alignment, train_model
from coremol.training.pretraining import freeze_attentivefp_atom_encoder, load_matching_backbone_state, load_matching_coremol_state
from coremol.utils.seed import set_seed
from torch_geometric.loader import DataLoader


def subset(dataset, indices):
    return [copy.copy(dataset[int(i)]) for i in indices if dataset[int(i)].num_nodes > 0]


def fixed_base_checkpoint_path(args, dataset_name: str, seed: int):
    if not getattr(args, "fixed_base_dir", ""):
        return None
    return Path(args.fixed_base_dir) / f"{dataset_name}_{seed}_base.pt"


def variant_training_seed(seed: int, variant: str) -> int:
    return seed


def load_fixed_base_state(model, path: str | Path):
    state = path if isinstance(path, dict) else torch.load(path, map_location="cpu")
    current = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state.items():
        if key in current and current[key].shape != value.shape:
            skipped.append(key)
            continue
        compatible[key] = value
    incompatible = model.load_state_dict(compatible, strict=False)
    return {
        "loaded": True,
        "missing": list(incompatible.missing_keys),
        "unexpected": list(incompatible.unexpected_keys),
        "skipped": skipped,
    }


def regression_scaler(train_set, task_type: str, normalize_regression: bool):
    if task_type != "regression" or not normalize_regression:
        return None
    values = torch.cat([data.y.view(data.y.size(0), -1).float() for data in train_set], dim=0)
    return RegressionTargetScaler.from_values(values)


def evaluate_loaded_model(model, valid_set, test_set, train_set, task_type: str, device, batch_size: int, normalize_regression: bool):
    target_scaler = regression_scaler(train_set, task_type, normalize_regression)
    return {
        "valid": evaluate_model(
            model,
            DataLoader(valid_set, batch_size=batch_size),
            task_type,
            device,
            target_scaler=target_scaler,
        ),
        "test": evaluate_model(
            model,
            DataLoader(test_set, batch_size=batch_size),
            task_type,
            device,
            target_scaler=target_scaler,
        ),
        "model": model,
        "target_scaler": target_scaler,
    }


def build_model(dataset, dataset_name: str, task_type: str, variant: str, args):
    in_channels, edge_dim = infer_feature_dims(dataset)
    out_channels = int(TASKS[dataset_name]["target_dim"])
    coremol_config = CoReMolConfig(
        enabled=(variant == "coremol"),
        d_max=args.d_max,
        support_hops=args.support_hops,
        beta=args.beta,
        tau=args.tau,
        residual_gate_init=args.residual_gate_init,
        residual_gate_max=args.residual_gate_max,
        residual_gate_mode=args.residual_gate_mode,
        residual_placement=args.residual_placement,
        num_residual_steps=args.num_residual_steps,
        residual_message=args.residual_message,
        residual_score_space=args.residual_score_space,
        residual_shift_centering=args.residual_shift_centering,
        residual_norm_mode=args.residual_norm_mode,
        dropout=args.dropout,
    )
    common_kwargs = dict(
        in_channels=in_channels,
        edge_dim=edge_dim,
        hidden_channels=args.hidden_channels,
        out_channels=out_channels,
        num_layers=args.num_layers,
        num_timesteps=args.num_timesteps,
        dropout=args.dropout,
        coremol=coremol_config,
    )
    if args.backbone == "graphformer":
        return CoReMolGraphformer(
            **common_kwargs,
            num_heads=args.graphformer_num_heads,
            max_distance=args.graphformer_max_distance,
            use_graph_token=args.graphformer_use_graph_token,
            readout=args.graphformer_readout,
            use_local_gnn=not args.graphformer_no_local_gnn,
            use_distance_bias=not args.graphformer_no_distance_bias,
            use_edge_bias=args.graphformer_use_edge_bias,
            use_degree_encoding=not args.graphformer_no_degree_encoding,
            ffn_ratio=args.graphformer_ffn_ratio,
            norm_style=args.graphformer_norm_style,
            feature_encoder=args.graphformer_feature_encoder,
        )
    if args.backbone == "dmpnn":
        return CoReMolDMPNN(**common_kwargs, readout=args.dmpnn_readout)
    return CoReMolAttentiveFP(**common_kwargs)


@torch.no_grad()
def collect_mechanism(model, dataset, device, max_graphs=64):
    model.eval().to(device)
    diagnostics = []
    for data in dataset[:max_graphs]:
        data = data.to(device)
        if not hasattr(data, "batch") or data.batch is None:
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        _, diag = model(data, return_diagnostics=True)
        diagnostics.extend(diag)
    return mechanism_summary(diagnostics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["BBBP", "ESOL"])
    parser.add_argument("--backbone", choices=["attentivefp", "graphformer", "dmpnn"], default="attentivefp")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_timesteps", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--d_max", type=int, default=4)
    parser.add_argument("--support_hops", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--residual_gate_init", type=float, default=0.1)
    parser.add_argument("--residual_gate_max", type=float, default=0.0)
    parser.add_argument("--residual_gate_mode", choices=["scalar", "channel"], default="scalar")
    parser.add_argument("--residual_placement", choices=["post", "layerwise", "both"], default="post")
    parser.add_argument("--num_residual_steps", type=int, default=1)
    parser.add_argument("--residual_message", choices=["value", "delta"], default="value")
    parser.add_argument("--residual_score_space", choices=["intensity", "distribution", "rir"], default="intensity")
    parser.add_argument("--residual_shift_centering", choices=["none", "source"], default="none")
    parser.add_argument("--residual_norm_mode", choices=["layernorm", "none"], default="layernorm")
    parser.add_argument("--warm_start_coremol", action="store_true")
    parser.add_argument("--pretrained_backbone", type=str, default="")
    parser.add_argument("--pretrained_coremol", type=str, default="")
    parser.add_argument("--normalize_regression", action="store_true")
    parser.add_argument("--split_strategy", choices=["scaffold", "curvflow_random", "random"], default="scaffold")
    parser.add_argument("--random_train_fraction", type=float, default=0.7)
    parser.add_argument("--random_valid_fraction", type=float, default=0.2)
    parser.add_argument("--freeze_coremol_backbone", action="store_true")
    parser.add_argument("--freeze_coremol_atom_encoder", action="store_true")
    parser.add_argument("--align_epochs", type=int, default=0)
    parser.add_argument("--align_weight", type=float, default=0.05)
    parser.add_argument("--residual_aux_weight", type=float, default=0.0)
    parser.add_argument("--ema_decay", type=float, default=0.0)
    parser.add_argument("--class_balance", action="store_true")
    parser.add_argument("--pos_weight_cap", type=float, default=50.0)
    parser.add_argument("--max_grad_norm", type=float, default=0.0)
    parser.add_argument("--backbone_lr_scale", type=float, default=1.0)
    parser.add_argument("--graphformer_use_graph_token", action="store_true")
    parser.add_argument("--graphformer_no_local_gnn", action="store_true")
    parser.add_argument("--graphformer_no_distance_bias", action="store_true")
    parser.add_argument("--graphformer_use_edge_bias", action="store_true")
    parser.add_argument("--graphformer_no_degree_encoding", action="store_true")
    parser.add_argument("--graphformer_readout", choices=["mean", "mean_max", "graph_token"], default="mean_max")
    parser.add_argument("--graphformer_ffn_ratio", type=int, default=4)
    parser.add_argument("--graphformer_norm_style", choices=["pre", "post"], default="pre")
    parser.add_argument("--graphformer_num_heads", type=int, default=4)
    parser.add_argument("--graphformer_max_distance", type=int, default=5)
    parser.add_argument("--graphformer_feature_encoder", choices=["linear", "categorical"], default="linear")
    parser.add_argument("--dmpnn_readout", choices=["mean", "mean_max"], default="mean")
    parser.add_argument("--tcm_graphs", type=int, default=48)
    parser.add_argument("--tcm_k", type=int, default=10)
    parser.add_argument("--results_name", type=str, default="stage1_gate")
    parser.add_argument("--fixed_base_dir", type=str, default="")
    parser.add_argument("--variants", nargs="+", choices=["base", "coremol"], default=["base", "coremol"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = ROOT / "results" / args.results_name
    ckpt_dir = results_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    raw_rows = []
    mechanism_rows = []
    for dataset_name in args.datasets:
        dataset_name = dataset_name.upper()
        dataset = load_moleculenet(dataset_name, ROOT / "data" / "moleculenet")
        task_type = TASKS[dataset_name]["type"]
        for seed in args.seeds:
            set_seed(seed)
            if args.split_strategy in {"curvflow_random", "random"}:
                split = load_or_create_random_split(
                    dataset,
                    dataset_name,
                    seed,
                    ROOT / "data" / "splits",
                    train_fraction=args.random_train_fraction,
                    valid_fraction=args.random_valid_fraction,
                )
            else:
                split = load_or_create_split(dataset, dataset_name, seed, ROOT / "data" / "splits")
            train_set = subset(dataset, split["train"])
            valid_set = subset(dataset, split["valid"])
            test_set = subset(dataset, split["test"])

            pretrained_state = None
            pretrained_coremol_state = None
            if args.pretrained_backbone:
                pretrained_state = torch.load(args.pretrained_backbone, map_location="cpu")
            if args.pretrained_coremol:
                pretrained_coremol_state = torch.load(args.pretrained_coremol, map_location="cpu")
            base_path = fixed_base_checkpoint_path(args, dataset_name, seed)

            trained = {}
            for variant in args.variants:
                set_seed(variant_training_seed(seed, variant))
                model = build_model(dataset, dataset_name, task_type, variant, args)
                if pretrained_coremol_state is not None:
                    if variant == "coremol":
                        report = load_matching_coremol_state(model, pretrained_coremol_state)
                        report_name = "loaded_pretrained_coremol"
                    else:
                        report = load_matching_backbone_state(model, pretrained_coremol_state)
                        report_name = "loaded_pretrained_coremol_backbone"
                    print(
                        f"{dataset_name}/seed{seed}/{variant} {report_name}="
                        f"{report['loaded']} skipped={len(report['skipped'])}",
                        flush=True,
                    )
                elif pretrained_state is not None:
                    report = load_matching_backbone_state(model, pretrained_state)
                    print(
                        f"{dataset_name}/seed{seed}/{variant} loaded_pretrained_backbone="
                        f"{report['loaded']} skipped={len(report['skipped'])}",
                        flush=True,
                    )
                if variant == "base" and base_path is not None and base_path.exists():
                    fixed_report = load_fixed_base_state(model, base_path)
                    model = model.to(device)
                    outcome = evaluate_loaded_model(
                        model=model,
                        valid_set=valid_set,
                        test_set=test_set,
                        train_set=train_set,
                        task_type=task_type,
                        device=device,
                        batch_size=args.batch_size,
                        normalize_regression=args.normalize_regression,
                    )
                    print(
                        f"{dataset_name}/seed{seed}/base loaded_fixed_base={base_path} "
                        f"missing={len(fixed_report['missing'])} unexpected={len(fixed_report['unexpected'])}",
                        flush=True,
                    )
                    trained[variant] = outcome["model"]
                    metric_row = {
                        "dataset": dataset_name,
                        "seed": seed,
                        "variant": variant,
                        "task_type": task_type,
                        **{f"valid_{k}": v for k, v in outcome["valid"].items()},
                        **{f"test_{k}": v for k, v in outcome["test"].items()},
                    }
                    raw_rows.append(metric_row)
                    torch.save(outcome["model"].state_dict(), ckpt_dir / f"{dataset_name}_{seed}_{variant}.pt")
                    pd.DataFrame(raw_rows).to_csv(results_dir / "raw_metrics.csv", index=False)
                    print(metric_row)
                    continue
                if variant == "coremol" and args.warm_start_coremol and "base" in trained:
                    model_state = model.state_dict()
                    for key, value in trained["base"].backbone.state_dict().items():
                        model_state[f"backbone.{key}"] = value
                    model.load_state_dict(model_state)
                if variant == "coremol" and args.freeze_coremol_backbone:
                    for parameter in model.backbone.parameters():
                        parameter.requires_grad = False
                elif variant == "coremol" and args.freeze_coremol_atom_encoder:
                    freeze_attentivefp_atom_encoder(model)
                outcome = train_model(
                    model=model,
                    train_dataset=train_set,
                    valid_dataset=valid_set,
                    test_dataset=test_set,
                    task_type=task_type,
                    device=device,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    patience=args.patience,
                    log_prefix=f"{dataset_name}/seed{seed}/{variant}",
                    normalize_regression=args.normalize_regression,
                    residual_aux_weight=args.residual_aux_weight if variant == "coremol" else 0.0,
                    ema_decay=args.ema_decay,
                    class_balance=args.class_balance,
                    pos_weight_cap=args.pos_weight_cap,
                    max_grad_norm=args.max_grad_norm,
                    backbone_lr_scale=args.backbone_lr_scale if variant == "coremol" else 1.0,
                )
                if variant == "coremol" and args.align_epochs > 0 and "base" in trained:
                    outcome["model"] = fine_tune_alignment(
                        model=outcome["model"],
                        base_model=trained["base"],
                        train_dataset=train_set,
                        task_type=task_type,
                        device=device,
                        epochs=args.align_epochs,
                        lr=args.lr * 0.5,
                        align_weight=args.align_weight,
                    )
                    from torch_geometric.loader import DataLoader
                    from coremol.training.trainer import evaluate_model

                    outcome["valid"] = evaluate_model(
                        outcome["model"], DataLoader(valid_set, batch_size=args.batch_size), task_type, device
                    )
                    outcome["test"] = evaluate_model(
                        outcome["model"], DataLoader(test_set, batch_size=args.batch_size), task_type, device
                    )
                trained[variant] = outcome["model"]
                metric_row = {
                    "dataset": dataset_name,
                    "seed": seed,
                    "variant": variant,
                    "task_type": task_type,
                    **{f"valid_{k}": v for k, v in outcome["valid"].items()},
                    **{f"test_{k}": v for k, v in outcome["test"].items()},
                }
                raw_rows.append(metric_row)
                torch.save(outcome["model"].state_dict(), ckpt_dir / f"{dataset_name}_{seed}_{variant}.pt")
                if variant == "base" and base_path is not None:
                    base_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(outcome["model"].state_dict(), base_path)
                    print(f"{dataset_name}/seed{seed}/base saved_fixed_base={base_path}", flush=True)
                pd.DataFrame(raw_rows).to_csv(results_dir / "raw_metrics.csv", index=False)
                print(metric_row)

            if "base" in trained and "coremol" in trained:
                tcm = estimate_tcm_at_k(
                    base_model=trained["base"],
                    cal_model=trained["coremol"],
                    dataset=test_set,
                    task_type=task_type,
                    device=device,
                    max_graphs=args.tcm_graphs,
                    k=args.tcm_k,
                )
                mech = collect_mechanism(trained["coremol"], test_set, device=device, max_graphs=args.tcm_graphs)
                mechanism_row = {
                    "dataset": dataset_name,
                    "seed": seed,
                    **tcm,
                    **mech,
                }
                mechanism_rows.append(mechanism_row)
                pd.DataFrame(mechanism_rows).to_csv(results_dir / "mechanism_metrics.csv", index=False)
                print(mechanism_row)

    raw = pd.DataFrame(raw_rows)
    raw.to_csv(results_dir / "raw_metrics.csv", index=False)
    numeric_cols = raw.select_dtypes(include="number").columns.tolist()
    raw.groupby(["dataset", "variant"])[numeric_cols].agg(["mean", "std"]).to_csv(results_dir / "summary_metrics.csv")
    pd.DataFrame(mechanism_rows).to_csv(results_dir / "mechanism_metrics.csv", index=False)


if __name__ == "__main__":
    main()
