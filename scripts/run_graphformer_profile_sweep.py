import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


PROFILES = {
    "lite_current": {},
    "graph_token_spd": {
        "graphformer_use_graph_token": True,
        "graphformer_readout": "graph_token",
    },
    "graph_token_edge": {
        "graphformer_use_graph_token": True,
        "graphformer_readout": "graph_token",
        "graphformer_use_edge_bias": True,
    },
    "no_local_edge": {
        "graphformer_no_local_gnn": True,
        "graphformer_use_graph_token": True,
        "graphformer_readout": "graph_token",
        "graphformer_use_edge_bias": True,
    },
    "edge_meanmax": {
        "graphformer_use_edge_bias": True,
        "graphformer_readout": "mean_max",
    },
    "cat_edge_meanmax": {
        "graphformer_feature_encoder": "categorical",
        "graphformer_use_edge_bias": True,
        "graphformer_readout": "mean_max",
    },
    "cat_graph_token_edge": {
        "graphformer_feature_encoder": "categorical",
        "graphformer_use_graph_token": True,
        "graphformer_readout": "graph_token",
        "graphformer_use_edge_bias": True,
    },
}


def _list_args(name: str, values) -> list[str]:
    return [f"--{name}", *[str(value) for value in values]]


def _profile_args(profile_name: str) -> list[str]:
    profile = PROFILES[profile_name]
    args: list[str] = []
    for key, value in profile.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        else:
            args.extend([f"--{key}", str(value)])
    return args


def _base_common_args(dataset: str, seeds: list[int], epochs: int, batch_size: int) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_stage1_gate.py"),
        "--backbone",
        "graphformer",
        *_list_args("datasets", [dataset]),
        *_list_args("seeds", seeds),
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--hidden_channels",
        "128",
        "--num_layers",
        "6",
        "--num_timesteps",
        "1",
        "--dropout",
        "0.10",
        "--lr",
        "0.0003",
        "--weight_decay",
        "1e-5",
        "--patience",
        "20",
        "--split_strategy",
        "random",
        "--random_train_fraction",
        "0.8",
        "--random_valid_fraction",
        "0.1",
        "--d_max",
        "3",
        "--support_hops",
        "3",
        "--beta",
        "0.1",
        "--tau",
        "1.0",
        "--residual_gate_init",
        "0.005",
        "--residual_gate_max",
        "0.05",
        "--residual_gate_mode",
        "scalar",
        "--residual_placement",
        "post",
        "--num_residual_steps",
        "1",
        "--residual_norm_mode",
        "layernorm",
        "--max_grad_norm",
        "5",
        "--tcm_graphs",
        "96",
        "--tcm_k",
        "10",
    ]


def _seeds_suffix(seeds: list[int]) -> str:
    return "_".join(str(seed) for seed in seeds)


def build_stage_commands(
    stage: str,
    datasets: list[str],
    profiles: list[str],
    seeds: list[int],
    messages: list[str],
    epochs: int,
    batch_size: int,
) -> list[list[str]]:
    commands: list[list[str]] = []
    for dataset in datasets:
        dataset = dataset.upper()
        for profile in profiles:
            if profile not in PROFILES:
                raise ValueError(f"Unknown profile {profile}. Expected one of {sorted(PROFILES)}.")
            if stage == "baseline":
                for seed in seeds:
                    results_name = f"graphformer_profile_sweep/{dataset}/{profile}/baseline_seed{seed}"
                    baseline_dir = f"CORMOL/results/graphformer_profile_sweep/{dataset}/{profile}/baseline_checkpoints"
                    commands.append(
                        [
                            *_base_common_args(dataset, [seed], epochs, batch_size),
                            "--variants",
                            "base",
                            "--fixed_base_dir",
                            baseline_dir,
                            *_profile_args(profile),
                            "--results_name",
                            results_name,
                        ]
                    )
            elif stage == "coremol":
                baseline_dir = f"CORMOL/results/graphformer_profile_sweep/{dataset}/{profile}/baseline_checkpoints"
                for message in messages:
                    results_name = (
                        f"graphformer_profile_sweep/{dataset}/{profile}/"
                        f"coremol_{message}_seeds{_seeds_suffix(seeds)}"
                    )
                    commands.append(
                        [
                            *_base_common_args(dataset, seeds, epochs, batch_size),
                            "--variants",
                            "base",
                            "coremol",
                            "--fixed_base_dir",
                            baseline_dir,
                            "--warm_start_coremol",
                            "--backbone_lr_scale",
                            "0.2",
                            "--residual_message",
                            message,
                            *_profile_args(profile),
                            "--results_name",
                            results_name,
                        ]
                    )
            else:
                raise ValueError("stage must be baseline or coremol")
    return commands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["baseline", "coremol"], required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--messages", nargs="+", choices=["value", "delta"], default=["delta"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    commands = build_stage_commands(
        stage=args.stage,
        datasets=args.datasets,
        profiles=args.profiles,
        seeds=args.seeds,
        messages=args.messages,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    for command in commands:
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
