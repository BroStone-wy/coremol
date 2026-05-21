import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def metric_name(dataset):
    return "test_roc_auc" if dataset == "BBBP" else "test_rmse"


def task_improved(dataset, base, cal):
    metric = metric_name(dataset)
    if dataset == "BBBP":
        return cal[metric] > base[metric]
    return cal[metric] < base[metric]


def main():
    results_dir = ROOT / "results" / "stage1_gate"
    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(results_dir / "raw_metrics.csv")
    mech = pd.read_csv(results_dir / "mechanism_metrics.csv")

    lines = [
        "# CoReMol Stage-1 Gate Report",
        "",
        "Scope: BBBP and ESOL, seeds 0/1/2, Base AttentiveFP versus Full CoReMol.",
        "",
        "## Per-Seed Gate",
        "",
        "| Dataset | Seed | Task Improved | Delta Task | Delta TCM | Gate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    gate_rows = []
    for dataset in sorted(raw["dataset"].unique()):
        for seed in sorted(raw[raw["dataset"] == dataset]["seed"].unique()):
            base = raw[(raw.dataset == dataset) & (raw.seed == seed) & (raw.variant == "base")].iloc[0]
            cal = raw[(raw.dataset == dataset) & (raw.seed == seed) & (raw.variant == "coremol")].iloc[0]
            m = mech[(mech.dataset == dataset) & (mech.seed == seed)].iloc[0]
            metric = metric_name(dataset)
            if dataset == "BBBP":
                delta_task = cal[metric] - base[metric]
            else:
                delta_task = base[metric] - cal[metric]
            delta_tcm = m["delta_tcm"]
            ok_task = task_improved(dataset, base, cal)
            ok_tcm = delta_tcm > 0
            gate = ok_task and ok_tcm
            gate_rows.append({"dataset": dataset, "seed": seed, "gate": gate, "task": ok_task, "tcm": ok_tcm})
            lines.append(f"| {dataset} | {seed} | {ok_task} | {delta_task:.6f} | {delta_tcm:.6f} | {gate} |")

    lines.extend(["", "## Aggregate Metrics", ""])
    for dataset in sorted(raw["dataset"].unique()):
        metric = metric_name(dataset)
        base_vals = raw[(raw.dataset == dataset) & (raw.variant == "base")][metric]
        cal_vals = raw[(raw.dataset == dataset) & (raw.variant == "coremol")][metric]
        tcm_vals = mech[mech.dataset == dataset]["delta_tcm"]
        lines.append(
            f"- {dataset}: base {metric}={base_vals.mean():.6f}±{base_vals.std():.6f}; "
            f"coremol {metric}={cal_vals.mean():.6f}±{cal_vals.std():.6f}; "
            f"delta_tcm={tcm_vals.mean():.6f}±{tcm_vals.std():.6f}"
        )

    lines.extend(["", "## Analysis", ""])
    for dataset in sorted(raw["dataset"].unique()):
        rows = [r for r in gate_rows if r["dataset"] == dataset]
        task_ok = sum(r["task"] for r in rows)
        tcm_ok = sum(r["tcm"] for r in rows)
        gate_ok = sum(r["gate"] for r in rows)
        lines.append(f"- {dataset}: task improved on {task_ok}/3 seeds, TCM improved on {tcm_ok}/3 seeds, both improved on {gate_ok}/3 seeds.")

    all_gate = all(sum(r["gate"] for r in gate_rows if r["dataset"] == d) >= 2 for d in sorted(raw["dataset"].unique()))
    lines.extend(["", "## Gate Decision", ""])
    lines.append("PASS" if all_gate else "FAIL")
    if not all_gate:
        lines.append("")
        lines.append("Failure handling: inspect whether TCM fails because base support normalization, demand collapse, calibration saturation, or residual gate scale is limiting the main mechanism. Any next adjustment should modify those core mechanism parameters only.")

    (report_dir / "stage1_gate_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

