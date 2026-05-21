import torch


def mechanism_summary(diagnostics: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    if not diagnostics:
        return {
            "enhance_ratio": float("nan"),
            "suppress_ratio": float("nan"),
            "mismatch_reduction": float("nan"),
            "calibration_contrast": float("nan"),
            "update_atom_norm_ratio": float("nan"),
            "residual_gate": float("nan"),
        }

    rows = []
    update_ratios = []
    gates = []
    for item in diagnostics:
        s = item["residual_score"].detach()
        base_tensor = item.get("alpha_ref", item.get("alpha_base"))
        cal_tensor = item.get("alpha_rew", item.get("alpha_cal"))
        if base_tensor is None or cal_tensor is None:
            continue
        base = base_tensor.detach()
        cal = cal_tensor.detach()
        pos = s > 0
        neg = s < 0
        enhance = ((cal > base) & pos).float().sum() / pos.float().sum().clamp_min(1.0)
        suppress = ((cal < base) & neg).float().sum() / neg.float().sum().clamp_min(1.0)
        demand = item.get("demand_alpha", item.get("demand"))
        if demand is None:
            mismatch = torch.tensor(float("nan"), device=s.device)
        else:
            demand = demand.detach()
            before = (s.abs() * (demand - base).abs()).sum()
            after = (s.abs() * (demand - cal).abs()).sum()
            mismatch = (before - after) / before.clamp_min(1e-8)
        contrast = cal[pos].mean() - cal[neg].mean() if pos.any() and neg.any() else torch.tensor(float("nan"), device=s.device)
        rows.append(torch.stack([enhance, suppress, mismatch, contrast]))
        if "update_atom_norm_ratio" in item:
            update_ratios.append(item["update_atom_norm_ratio"].detach().float().view(()))
        if "residual_gate" in item:
            gates.append(item["residual_gate"].detach().float().abs().mean())

    if rows:
        table = torch.stack(rows)
        means = torch.nanmean(table, dim=0)
    else:
        means = torch.full((4,), float("nan"))
    update_ratio = torch.stack(update_ratios).mean() if update_ratios else torch.tensor(float("nan"))
    gate = torch.stack(gates).mean() if gates else torch.tensor(float("nan"))
    return {
        "enhance_ratio": float(means[0].cpu()),
        "suppress_ratio": float(means[1].cpu()),
        "mismatch_reduction": float(means[2].cpu()),
        "calibration_contrast": float(means[3].cpu()),
        "update_atom_norm_ratio": float(update_ratio.cpu()),
        "residual_gate": float(gate.cpu()),
    }
