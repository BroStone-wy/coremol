import argparse
import json
from pathlib import Path

import pandas as pd


CURVFLOW_ATTENTIVEFP = {
    "BACE": 0.784,
    "CLINTOX": 0.847,
    "TOX21": 0.757,
    "HIV": 0.761,
    "SIDER": 0.606,
    "TOXCAST": 0.637,
}


def summarize_run(run_dir: Path) -> list[dict]:
    raw_path = run_dir / "raw_metrics.csv"
    if not raw_path.exists():
        return []
    raw = pd.read_csv(raw_path)
    mech_path = run_dir / "mechanism_metrics.csv"
    mech = pd.read_csv(mech_path) if mech_path.exists() else pd.DataFrame()
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        config_path = run_dir / "input_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    message = config.get("residual_message", run_dir.name)
    rows = []
    for dataset in sorted(raw["dataset"].unique()):
        subset = raw[raw["dataset"] == dataset]
        base = subset[subset["variant"] == "base"].sort_values("seed")
        core = subset[subset["variant"] == "coremol"].sort_values("seed")
        if base.empty or core.empty:
            continue
        merged = base[["seed", "test_roc_auc"]].merge(
            core[["seed", "test_roc_auc"]],
            on="seed",
            suffixes=("_base", "_coremol"),
        )
        tcm_delta = float("nan")
        tcm_wins = 0
        if not mech.empty and "delta_tcm" in mech.columns:
            m = mech[mech["dataset"] == dataset]
            if not m.empty:
                tcm_delta = float(m["delta_tcm"].mean())
                tcm_wins = int((m["delta_tcm"] > 0).sum())
        base_mean = float(merged["test_roc_auc_base"].mean())
        core_mean = float(merged["test_roc_auc_coremol"].mean())
        rows.append(
            {
                "run_dir": str(run_dir),
                "dataset": dataset,
                "message": message,
                "base_mean_auc": base_mean,
                "coremol_mean_auc": core_mean,
                "auc_improvement": core_mean - base_mean,
                "task_wins": int((merged["test_roc_auc_coremol"] > merged["test_roc_auc_base"]).sum()),
                "tcm_delta_mean": tcm_delta,
                "tcm_wins": tcm_wins,
                "baseline_gap_to_curvflow_attentivefp": base_mean - CURVFLOW_ATTENTIVEFP.get(dataset, float("nan")),
            }
        )
    return rows


def choose_best(summary: pd.DataFrame) -> dict:
    best = {}
    for dataset in sorted(summary["dataset"].unique()):
        candidates = summary[summary["dataset"] == dataset].copy()
        candidates["passes"] = (
            (candidates["auc_improvement"] > 0)
            & (candidates["task_wins"] >= 2)
            & (candidates["tcm_delta_mean"] > 0)
            & (candidates["tcm_wins"] >= 2)
        )
        candidates = candidates.sort_values(["passes", "auc_improvement", "tcm_delta_mean"], ascending=False)
        row = candidates.iloc[0].to_dict()
        best[dataset] = row
        if not bool(row["passes"]):
            best[dataset]["status"] = "needs_rerun"
        else:
            best[dataset]["status"] = "selected"
    return best


def write_report(root: Path, summary: pd.DataFrame, best: dict):
    lines = [
        "# CurvFlow Classification Sweep Report",
        "",
        "| dataset | message | base AUC | CoReMol AUC | improvement | task wins | TCM delta | TCM wins | baseline gap | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for dataset, row in best.items():
        lines.append(
            f"| {dataset} | {row['message']} | {row['base_mean_auc']:.6f} | "
            f"{row['coremol_mean_auc']:.6f} | {row['auc_improvement']:+.6f} | "
            f"{int(row['task_wins'])}/3 | {row['tcm_delta_mean']:+.6f} | "
            f"{int(row['tcm_wins'])}/3 | {row['baseline_gap_to_curvflow_attentivefp']:+.6f} | {row['status']} |"
        )
    lines.extend(["", "## All Runs", "", summary.to_markdown(index=False)])
    (root / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("CORMOL/results/curvflow_classification_sweep"))
    args = parser.parse_args()

    rows = []
    for raw in args.root.glob("*/raw_metrics.csv"):
        rows.extend(summarize_run(raw.parent))
    summary = pd.DataFrame(rows)
    args.root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.root / "summary.csv", index=False)
    if summary.empty:
        (args.root / "best_configs.json").write_text("{}\n")
        (args.root / "report.md").write_text("# CurvFlow Classification Sweep Report\n\nNo completed runs found.\n")
        return
    best = choose_best(summary)
    (args.root / "best_configs.json").write_text(json.dumps(best, indent=2, sort_keys=True))
    write_report(args.root, summary, best)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
