import torch
from torch_geometric.loader import DataLoader

from coremol.metrics.tcm import normalized_tcm_score, tcm_at_k


def _loss_value(pred, y, task_type: str):
    y = y.view(pred.size(0), -1).float()
    if task_type == "classification":
        mask = torch.isfinite(y)
        clean_y = torch.where(mask, y, torch.zeros_like(y))
        losses = torch.nn.functional.binary_cross_entropy_with_logits(pred, clean_y, reduction="none")
        losses = torch.where(mask, losses, torch.zeros_like(losses))
        denom = mask.sum(dim=1).clamp_min(1)
        return losses.sum(dim=1) / denom
    return torch.nn.functional.mse_loss(pred, y, reduction="none").mean(dim=1)


@torch.no_grad()
def _pair_sensitivity_from_base_model(base_model, data, pairs, task_type: str, epsilon: float = 0.05):
    atoms = base_model.encode_atoms(data.x.float(), data.edge_index, data.edge_attr.float())
    base_pred, _ = base_model.molecule_readout(atoms, data.batch)
    base_loss = _loss_value(base_pred, data.y, task_type).mean()
    scores = []
    for src, dst in pairs:
        probed_atoms = atoms.clone()
        probed_atoms[int(src)] = probed_atoms[int(src)] + epsilon * atoms[int(dst)]
        pred, _ = base_model.molecule_readout(probed_atoms, data.batch)
        probe_loss = _loss_value(pred, data.y, task_type).mean()
        scores.append(base_loss - probe_loss)
    if not scores:
        return torch.empty(0, device=atoms.device)
    return torch.stack(scores)


@torch.no_grad()
def estimate_tcm_at_k(base_model, cal_model, dataset, task_type: str, device, max_graphs: int = 64, k: int = 10):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    base_model.eval().to(device)
    cal_model.eval().to(device)
    rows = []
    seen = 0
    for data in loader:
        if seen >= max_graphs:
            break
        seen += 1
        data = data.to(device)
        _, cal_diag = cal_model(data, return_diagnostics=True)
        if not cal_diag:
            continue
        diag = cal_diag[-1]
        pair_sensitivity = _pair_sensitivity_from_base_model(
            base_model=base_model,
            data=data,
            pairs=diag["pairs"],
            task_type=task_type,
        )
        if pair_sensitivity.numel() == 0:
            continue
        base_tcm = tcm_at_k(diag["alpha_base"], pair_sensitivity, k=k)
        cal_tcm = tcm_at_k(diag["alpha_cal"], pair_sensitivity, k=k)
        base_norm_tcm = normalized_tcm_score(diag["alpha_base"], pair_sensitivity)
        cal_norm_tcm = normalized_tcm_score(diag["alpha_cal"], pair_sensitivity)
        rows.append(
            (
                float(base_tcm.cpu()),
                float(cal_tcm.cpu()),
                float(base_norm_tcm.cpu()),
                float(cal_norm_tcm.cpu()),
            )
        )

    if not rows:
        return {
            "tcm_base": float("nan"),
            "tcm_cal": float("nan"),
            "delta_tcm": float("nan"),
            "norm_tcm_base": float("nan"),
            "norm_tcm_cal": float("nan"),
            "delta_norm_tcm": float("nan"),
            "num_graphs": 0,
        }
    table = torch.tensor(rows)
    base = float(table[:, 0].mean())
    cal = float(table[:, 1].mean())
    norm_base = float(table[:, 2].mean())
    norm_cal = float(table[:, 3].mean())
    return {
        "tcm_base": base,
        "tcm_cal": cal,
        "delta_tcm": base - cal,
        "norm_tcm_base": norm_base,
        "norm_tcm_cal": norm_cal,
        "delta_norm_tcm": norm_cal - norm_base,
        "num_graphs": len(rows),
    }
