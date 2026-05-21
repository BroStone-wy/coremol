import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def list_args(name: str, values) -> list[str]:
    return [f"--{name}", *[str(value) for value in values]]


def scalar_args(config: dict) -> list[str]:
    keys = [
        "epochs",
        "backbone",
        "batch_size",
        "hidden_channels",
        "num_layers",
        "num_timesteps",
        "dropout",
        "lr",
        "weight_decay",
        "patience",
        "d_max",
        "support_hops",
        "beta",
        "tau",
        "residual_aux_weight",
        "ema_decay",
        "pos_weight_cap",
        "max_grad_norm",
        "backbone_lr_scale",
        "residual_gate_init",
        "residual_gate_mode",
        "residual_placement",
        "num_residual_steps",
        "residual_message",
        "residual_norm_mode",
        "split_strategy",
        "random_train_fraction",
        "random_valid_fraction",
        "tcm_graphs",
        "tcm_k",
        "graphformer_readout",
        "graphformer_ffn_ratio",
        "graphformer_norm_style",
        "graphformer_num_heads",
        "graphformer_max_distance",
        "graphformer_feature_encoder",
    ]
    args = []
    for key in keys:
        if key in config:
            args.extend([f"--{key}", str(config[key])])
    for flag in [
        "warm_start_coremol",
        "freeze_coremol_backbone",
        "freeze_coremol_atom_encoder",
        "normalize_regression",
        "class_balance",
        "graphformer_use_graph_token",
        "graphformer_no_local_gnn",
        "graphformer_no_distance_bias",
        "graphformer_use_edge_bias",
        "graphformer_no_degree_encoding",
    ]:
        if config.get(flag):
            args.append(f"--{flag}")
    if config.get("variants"):
        args.extend(["--variants", *[str(value) for value in config["variants"]]])
    for path_key in ["pretrained_backbone", "pretrained_coremol", "fixed_base_dir"]:
        if config.get(path_key):
            args.extend([f"--{path_key}", str(config[path_key])])
    return args


def build_command(config: dict, results_name: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_stage1_gate.py"),
        *list_args("datasets", config["datasets"]),
        *list_args("seeds", config["seeds"]),
        *scalar_args(config),
        "--results_name",
        results_name,
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results_name", required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    command = build_command(config, args.results_name)
    results_dir = ROOT / "results" / args.results_name
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "input_config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    (results_dir / "launch_command.txt").write_text(" ".join(command) + "\n")
    print(" ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
