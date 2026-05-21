import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coremol.datasets.moleculenet import TASKS, infer_feature_dims, load_moleculenet
from coremol.datasets.random_split import load_or_create_random_split
from coremol.datasets.scaffold_split import load_or_create_split
from coremol.metrics.tcm import full_tcm, normalized_tcm_components, tcm_topk_components
from coremol.models.attentivefp_coremol import CoReMolAttentiveFP, CoReMolConfig
from coremol.models.graphformer_coremol import CoReMolGraphformer
from coremol.probes.tcm import _pair_sensitivity_from_base_model


def limit_dataset(dataset, max_graphs: int):
    if max_graphs <= 0:
        return dataset
    return dataset[:max_graphs]


def load_compatible_state(model, state):
    current = model.state_dict()
    compatible = {}
    skipped = []
    unexpected = []
    for key, value in state.items():
        if key not in current:
            unexpected.append(key)
            continue
        if current[key].shape != value.shape:
            skipped.append(key)
            continue
        compatible[key] = value
    incompatible = model.load_state_dict(compatible, strict=False)
    return {
        "loaded": len(compatible),
        "missing": list(incompatible.missing_keys),
        "unexpected": unexpected + list(incompatible.unexpected_keys),
        "skipped": skipped,
    }


def build_model(
    dataset,
    dataset_name: str,
    enabled: bool,
    hidden_channels: int,
    d_max: int,
    support_hops: int,
    backbone: str = "attentivefp",
    in_channels: int | None = None,
    edge_dim: int | None = None,
    out_channels: int | None = None,
    num_layers: int = 2,
    num_timesteps: int = 2,
    dropout: float = 0.1,
    beta: float = 0.2,
    tau: float = 0.5,
    residual_placement: str = "post",
    num_residual_steps: int = 1,
    residual_message: str = "value",
    residual_score_space: str = "intensity",
    residual_shift_centering: str = "none",
    residual_norm_mode: str = "layernorm",
    residual_gate_mode: str = "scalar",
    residual_gate_max: float = 0.0,
    graphformer_num_heads: int = 4,
    graphformer_max_distance: int = 5,
    graphformer_use_graph_token: bool = False,
    graphformer_readout: str = "mean_max",
    graphformer_no_local_gnn: bool = False,
    graphformer_no_distance_bias: bool = False,
    graphformer_use_edge_bias: bool = False,
    graphformer_no_degree_encoding: bool = False,
    graphformer_ffn_ratio: int = 4,
    graphformer_norm_style: str = "pre",
    graphformer_feature_encoder: str = "linear",
):
    if in_channels is None or edge_dim is None:
        inferred_in_channels, inferred_edge_dim = infer_feature_dims(dataset)
        in_channels = inferred_in_channels if in_channels is None else in_channels
        edge_dim = inferred_edge_dim if edge_dim is None else edge_dim
    out_channels = int(TASKS[dataset_name]["target_dim"]) if out_channels is None else out_channels
    coremol = CoReMolConfig(
        enabled=enabled,
        d_max=d_max,
        support_hops=support_hops,
        beta=beta,
        tau=tau,
        residual_placement=residual_placement,
        num_residual_steps=num_residual_steps,
        residual_message=residual_message,
        residual_score_space=residual_score_space,
        residual_shift_centering=residual_shift_centering,
        residual_norm_mode=residual_norm_mode,
        residual_gate_mode=residual_gate_mode,
        residual_gate_max=residual_gate_max,
    )
    common = dict(
        in_channels=in_channels,
        edge_dim=edge_dim,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        num_timesteps=num_timesteps,
        dropout=dropout,
        coremol=coremol,
    )
    if backbone == "graphformer":
        return CoReMolGraphformer(
            **common,
            num_heads=graphformer_num_heads,
            max_distance=graphformer_max_distance,
            use_graph_token=graphformer_use_graph_token,
            readout=graphformer_readout,
            use_local_gnn=not graphformer_no_local_gnn,
            use_distance_bias=not graphformer_no_distance_bias,
            use_edge_bias=graphformer_use_edge_bias,
            use_degree_encoding=not graphformer_no_degree_encoding,
            ffn_ratio=graphformer_ffn_ratio,
            norm_style=graphformer_norm_style,
            feature_encoder=graphformer_feature_encoder,
        )
    return CoReMolAttentiveFP(**common)


@torch.no_grad()
def compute_for_seed(
    dataset_name: str,
    seed: int,
    run_dir: Path,
    device,
    hidden_channels: int,
    backbone: str,
    num_layers: int,
    num_timesteps: int,
    dropout: float,
    d_max: int,
    support_hops: int,
    beta: float,
    tau: float,
    split_strategy: str,
    random_train_fraction: float,
    random_valid_fraction: float,
    residual_placement: str,
    num_residual_steps: int,
    residual_message: str,
    residual_score_space: str,
    residual_shift_centering: str,
    residual_norm_mode: str,
    residual_gate_mode: str,
    residual_gate_max: float,
    max_graphs: int,
    graphformer_num_heads: int,
    graphformer_max_distance: int,
    graphformer_use_graph_token: bool,
    graphformer_readout: str,
    graphformer_no_local_gnn: bool,
    graphformer_no_distance_bias: bool,
    graphformer_use_edge_bias: bool,
    graphformer_no_degree_encoding: bool,
    graphformer_ffn_ratio: int,
    graphformer_norm_style: str,
    graphformer_feature_encoder: str,
):
    dataset = load_moleculenet(dataset_name, ROOT / "data" / "moleculenet")
    task_type = TASKS[dataset_name]["type"]
    if split_strategy in {"curvflow_random", "random"}:
        split = load_or_create_random_split(
            dataset,
            dataset_name,
            seed,
            ROOT / "data" / "splits",
            train_fraction=random_train_fraction,
            valid_fraction=random_valid_fraction,
        )
    else:
        split = load_or_create_split(dataset, dataset_name, seed, ROOT / "data" / "splits")
    test_set = limit_dataset([dataset[i] for i in split["test"] if dataset[i].num_nodes > 0], max_graphs)
    base_model = build_model(
        dataset,
        dataset_name=dataset_name,
        enabled=False,
        hidden_channels=hidden_channels,
        backbone=backbone,
        num_layers=num_layers,
        num_timesteps=num_timesteps,
        dropout=dropout,
        d_max=d_max,
        support_hops=support_hops,
        beta=beta,
        tau=tau,
        graphformer_num_heads=graphformer_num_heads,
        graphformer_max_distance=graphformer_max_distance,
        graphformer_use_graph_token=graphformer_use_graph_token,
        graphformer_readout=graphformer_readout,
        graphformer_no_local_gnn=graphformer_no_local_gnn,
        graphformer_no_distance_bias=graphformer_no_distance_bias,
        graphformer_use_edge_bias=graphformer_use_edge_bias,
        graphformer_no_degree_encoding=graphformer_no_degree_encoding,
        graphformer_ffn_ratio=graphformer_ffn_ratio,
        graphformer_norm_style=graphformer_norm_style,
        graphformer_feature_encoder=graphformer_feature_encoder,
    ).to(device)
    cal_model = build_model(
        dataset,
        dataset_name=dataset_name,
        enabled=True,
        hidden_channels=hidden_channels,
        backbone=backbone,
        num_layers=num_layers,
        num_timesteps=num_timesteps,
        dropout=dropout,
        d_max=d_max,
        support_hops=support_hops,
        beta=beta,
        tau=tau,
        residual_placement=residual_placement,
        num_residual_steps=num_residual_steps,
        residual_message=residual_message,
        residual_score_space=residual_score_space,
        residual_shift_centering=residual_shift_centering,
        residual_norm_mode=residual_norm_mode,
        residual_gate_mode=residual_gate_mode,
        residual_gate_max=residual_gate_max,
        graphformer_num_heads=graphformer_num_heads,
        graphformer_max_distance=graphformer_max_distance,
        graphformer_use_graph_token=graphformer_use_graph_token,
        graphformer_readout=graphformer_readout,
        graphformer_no_local_gnn=graphformer_no_local_gnn,
        graphformer_no_distance_bias=graphformer_no_distance_bias,
        graphformer_use_edge_bias=graphformer_use_edge_bias,
        graphformer_no_degree_encoding=graphformer_no_degree_encoding,
        graphformer_ffn_ratio=graphformer_ffn_ratio,
        graphformer_norm_style=graphformer_norm_style,
        graphformer_feature_encoder=graphformer_feature_encoder,
    ).to(device)
    ckpt = run_dir / "checkpoints"
    base_report = load_compatible_state(base_model, torch.load(ckpt / f"{dataset_name}_{seed}_base.pt", map_location=device))
    cal_report = load_compatible_state(cal_model, torch.load(ckpt / f"{dataset_name}_{seed}_coremol.pt", map_location=device))
    if base_report["missing"] or cal_report["missing"]:
        print(
            {
                "dataset": dataset_name,
                "seed": seed,
                "base_missing": len(base_report["missing"]),
                "cal_missing": len(cal_report["missing"]),
                "base_skipped": len(base_report["skipped"]),
                "cal_skipped": len(cal_report["skipped"]),
            },
            flush=True,
        )
    base_model.eval()
    cal_model.eval()

    per_graph = []
    for data in DataLoader(test_set, batch_size=1, shuffle=False):
        data = data.to(device)
        _, diagnostics = cal_model(data, return_diagnostics=True)
        if not diagnostics:
            continue
        diag = diagnostics[-1]
        sensitivity = _pair_sensitivity_from_base_model(base_model, data, diag["pairs"], task_type)
        row = {
            "dataset": dataset_name,
            "seed": seed,
            "full_tcm_base": float(full_tcm(diag["alpha_base"], sensitivity).cpu()),
            "full_tcm_cal": float(full_tcm(diag["alpha_cal"], sensitivity).cpu()),
        }
        row["delta_full_tcm"] = row["full_tcm_base"] - row["full_tcm_cal"]
        base_norm = normalized_tcm_components(diag["alpha_base"], sensitivity)
        cal_norm = normalized_tcm_components(diag["alpha_cal"], sensitivity)
        row["norm_tcm_base"] = float(base_norm["score"].cpu())
        row["norm_tcm_cal"] = float(cal_norm["score"].cpu())
        row["delta_norm_tcm"] = row["norm_tcm_cal"] - row["norm_tcm_base"]
        row["norm_benefit_base"] = float(base_norm["benefit"].cpu())
        row["norm_benefit_cal"] = float(cal_norm["benefit"].cpu())
        row["delta_norm_benefit"] = row["norm_benefit_cal"] - row["norm_benefit_base"]
        row["norm_harm_base"] = float(base_norm["harm"].cpu())
        row["norm_harm_cal"] = float(cal_norm["harm"].cpu())
        row["delta_norm_harm"] = row["norm_harm_base"] - row["norm_harm_cal"]
        for k in [5, 10, 20]:
            base_comp = tcm_topk_components(diag["alpha_base"], sensitivity, k=k)
            cal_comp = tcm_topk_components(diag["alpha_cal"], sensitivity, k=k)
            for name in ["ucov", "hleak", "tcm"]:
                row[f"{name}{k}_base"] = float(base_comp[name].cpu())
                row[f"{name}{k}_cal"] = float(cal_comp[name].cpu())
                if name == "tcm":
                    row[f"delta_tcm{k}"] = row[f"{name}{k}_base"] - row[f"{name}{k}_cal"]
                elif name == "ucov":
                    row[f"delta_ucov{k}"] = row[f"{name}{k}_cal"] - row[f"{name}{k}_base"]
                else:
                    row[f"delta_hleak{k}"] = row[f"{name}{k}_base"] - row[f"{name}{k}_cal"]
        per_graph.append(row)
    return pd.DataFrame(per_graph).mean(numeric_only=True).to_dict() | {"dataset": dataset_name, "seed": seed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=str(ROOT / "results" / "stage1_gate_standard_split"))
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--backbone", choices=["attentivefp", "graphformer"], default="attentivefp")
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_timesteps", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d_max", type=int, default=4)
    parser.add_argument("--support_hops", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--split_strategy", choices=["scaffold", "curvflow_random", "random"], default="scaffold")
    parser.add_argument("--random_train_fraction", type=float, default=0.7)
    parser.add_argument("--random_valid_fraction", type=float, default=0.2)
    parser.add_argument("--residual_placement", choices=["post", "layerwise", "both"], default="post")
    parser.add_argument("--num_residual_steps", type=int, default=1)
    parser.add_argument("--residual_message", choices=["value", "delta"], default="value")
    parser.add_argument("--residual_score_space", choices=["intensity", "distribution", "rir"], default="intensity")
    parser.add_argument("--residual_shift_centering", choices=["none", "source"], default="none")
    parser.add_argument("--residual_norm_mode", choices=["layernorm", "none"], default="layernorm")
    parser.add_argument("--residual_gate_mode", choices=["scalar", "channel"], default="scalar")
    parser.add_argument("--residual_gate_max", type=float, default=0.0)
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
    parser.add_argument("--max_graphs", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if args.datasets is None or args.seeds is None:
        raw = pd.read_csv(run_dir / "raw_metrics.csv")
        datasets = args.datasets or sorted(raw["dataset"].unique().tolist())
        seeds = args.seeds or sorted(int(seed) for seed in raw["seed"].unique().tolist())
    else:
        datasets = args.datasets
        seeds = args.seeds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for dataset_name in datasets:
        for seed in seeds:
            rows.append(
                compute_for_seed(
                    dataset_name.upper(),
                    seed,
                    run_dir,
                    device,
                    hidden_channels=args.hidden_channels,
                    backbone=args.backbone,
                    num_layers=args.num_layers,
                    num_timesteps=args.num_timesteps,
                    dropout=args.dropout,
                    d_max=args.d_max,
                    support_hops=args.support_hops,
                    beta=args.beta,
                    tau=args.tau,
                    split_strategy=args.split_strategy,
                    random_train_fraction=args.random_train_fraction,
                    random_valid_fraction=args.random_valid_fraction,
                    residual_placement=args.residual_placement,
                    num_residual_steps=args.num_residual_steps,
                    residual_message=args.residual_message,
                    residual_score_space=args.residual_score_space,
                    residual_shift_centering=args.residual_shift_centering,
                    residual_norm_mode=args.residual_norm_mode,
                    residual_gate_mode=args.residual_gate_mode,
                    residual_gate_max=args.residual_gate_max,
                    max_graphs=args.max_graphs,
                    graphformer_num_heads=args.graphformer_num_heads,
                    graphformer_max_distance=args.graphformer_max_distance,
                    graphformer_use_graph_token=args.graphformer_use_graph_token,
                    graphformer_readout=args.graphformer_readout,
                    graphformer_no_local_gnn=args.graphformer_no_local_gnn,
                    graphformer_no_distance_bias=args.graphformer_no_distance_bias,
                    graphformer_use_edge_bias=args.graphformer_use_edge_bias,
                    graphformer_no_degree_encoding=args.graphformer_no_degree_encoding,
                    graphformer_ffn_ratio=args.graphformer_ffn_ratio,
                    graphformer_norm_style=args.graphformer_norm_style,
                    graphformer_feature_encoder=args.graphformer_feature_encoder,
                )
            )
    out = pd.DataFrame(rows)
    out_path = run_dir / "tcm_variants.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
