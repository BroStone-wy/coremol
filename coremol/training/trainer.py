from copy import deepcopy

import numpy as np
import torch
from torch_geometric.utils import softmax
from torch_geometric.loader import DataLoader

from coremol.metrics.task_metrics import classification_metrics, regression_metrics
from coremol.probes.tcm import _pair_sensitivity_from_base_model


def _target(data):
    return data.y.view(data.y.size(0), -1).float()


class RegressionTargetScaler:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean.float()
        self.std = std.float().clamp_min(1e-8)

    @classmethod
    def from_values(cls, values: torch.Tensor):
        values = values.float().view(values.size(0), -1)
        return cls(values.mean(dim=0, keepdim=True), values.std(dim=0, keepdim=True))

    def to(self, device):
        return RegressionTargetScaler(self.mean.to(device), self.std.to(device))

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values.float() - self.mean.to(values.device)) / self.std.to(values.device)

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        return values.float() * self.std.to(values.device) + self.mean.to(values.device)


class ExponentialMovingAverage:
    def __init__(self, model, decay: float):
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model):
        for name, parameter in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        backup = {}
        for name, parameter in model.named_parameters():
            if name not in self.shadow:
                continue
            backup[name] = parameter.detach().clone()
            parameter.copy_(self.shadow[name])
        return backup

    @torch.no_grad()
    def restore(self, model, backup):
        for name, parameter in model.named_parameters():
            if name in backup:
                parameter.copy_(backup[name])


def compute_classification_pos_weight(train_dataset, max_weight: float = 50.0) -> torch.Tensor:
    targets = torch.cat([_target(data) for data in train_dataset], dim=0).float()
    mask = torch.isfinite(targets)
    positives = torch.where(mask, targets, torch.zeros_like(targets)).sum(dim=0, keepdim=True)
    observed = mask.sum(dim=0, keepdim=True).float()
    negatives = observed - positives
    return (negatives / positives.clamp_min(1.0)).clamp(1.0, float(max_weight))


def masked_classification_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = torch.isfinite(target)
    if not bool(mask.any()):
        return logits.sum() * 0.0
    clean_target = torch.where(mask, target, torch.zeros_like(target)).clamp(0.0, 1.0)
    losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, clean_target, reduction="none")
    if pos_weight is not None:
        weights = torch.where(target > 0.5, pos_weight.to(logits.device), torch.ones_like(logits))
        losses = losses * weights
    return losses[mask].mean()


def build_optimizer(
    model,
    lr: float,
    weight_decay: float,
    backbone_lr_scale: float = 1.0,
):
    if backbone_lr_scale >= 1.0 or not hasattr(model, "backbone"):
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    backbone_ids = {id(parameter) for parameter in model.backbone.parameters() if parameter.requires_grad}
    backbone_parameters = []
    other_parameters = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in backbone_ids:
            backbone_parameters.append(parameter)
        else:
            other_parameters.append(parameter)

    parameter_groups = []
    if backbone_parameters:
        parameter_groups.append({"params": backbone_parameters, "lr": lr * backbone_lr_scale})
    if other_parameters:
        parameter_groups.append({"params": other_parameters, "lr": lr})
    return torch.optim.Adam(parameter_groups, lr=lr, weight_decay=weight_decay)


def residual_demand_alignment_loss(diagnostics: list[dict[str, torch.Tensor]]) -> torch.Tensor:
    losses = []
    for item in diagnostics:
        pairs = item.get("pairs")
        demand = item.get("demand")
        alpha_cal = item.get("alpha_cal")
        residual_score = item.get("residual_score")
        if pairs is None or demand is None or alpha_cal is None or residual_score is None or pairs.numel() == 0:
            continue
        src = pairs[:, 0]
        target_alpha = softmax(torch.log(demand.detach().clamp_min(1e-6)), src)
        weights = residual_score.detach().abs()
        weighted_error = weights * (alpha_cal - target_alpha).pow(2)
        losses.append(weighted_error.sum() / weights.sum().clamp_min(1e-8))
    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


def train_model(
    model,
    train_dataset,
    valid_dataset,
    test_dataset,
    task_type: str,
    device,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 15,
    log_prefix: str = "",
    normalize_regression: bool = False,
    residual_aux_weight: float = 0.0,
    ema_decay: float = 0.0,
    class_balance: bool = False,
    pos_weight_cap: float = 50.0,
    max_grad_norm: float = 0.0,
    backbone_lr_scale: float = 1.0,
):
    model = model.to(device)
    regression_loss_fn = torch.nn.MSELoss()
    pos_weight = (
        compute_classification_pos_weight(train_dataset, max_weight=pos_weight_cap).to(device)
        if task_type == "classification" and class_balance
        else None
    )
    optimizer = build_optimizer(
        model,
        lr=lr,
        weight_decay=weight_decay,
        backbone_lr_scale=backbone_lr_scale,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    target_scaler = None
    if task_type == "regression" and normalize_regression:
        target_scaler = RegressionTargetScaler.from_values(torch.cat([_target(data) for data in train_dataset], dim=0))

    best_score = None
    best_state = deepcopy(model.state_dict())
    stale = 0
    ema = ExponentialMovingAverage(model, decay=ema_decay) if ema_decay > 0 else None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            if residual_aux_weight > 0:
                pred, diagnostics = model(batch, return_diagnostics=True)
            else:
                pred = model(batch)
                diagnostics = []
            target = _target(batch).to(device)
            if target_scaler is not None:
                target = target_scaler.transform(target)
            if task_type == "classification":
                loss = masked_classification_loss(pred, target, pos_weight=pos_weight)
            else:
                loss = regression_loss_fn(pred, target)
            if residual_aux_weight > 0 and diagnostics:
                aux_loss = residual_demand_alignment_loss(diagnostics).to(device)
                loss = loss + residual_aux_weight * aux_loss
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if ema is not None:
                ema.update(model)

        backup = ema.copy_to(model) if ema is not None else None
        valid = evaluate_model(model, valid_loader, task_type, device, target_scaler=target_scaler)
        score = valid["roc_auc"] if task_type == "classification" else -valid["rmse"]
        if log_prefix and (epoch == 1 or epoch % 5 == 0):
            print(f"{log_prefix} epoch={epoch} valid={valid}", flush=True)
        if best_score is None or score > best_score:
            best_score = score
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if ema is not None:
            ema.restore(model, backup)
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    return {
        "valid": evaluate_model(model, valid_loader, task_type, device, target_scaler=target_scaler),
        "test": evaluate_model(model, test_loader, task_type, device, target_scaler=target_scaler),
        "model": model,
        "target_scaler": target_scaler,
    }


@torch.no_grad()
def evaluate_model(model, loader, task_type: str, device, target_scaler=None) -> dict[str, float]:
    model.eval()
    preds, ys = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        preds.append(out.detach().cpu().numpy())
        ys.append(_target(batch).detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    ys = np.concatenate(ys, axis=0)
    if task_type == "classification":
        return classification_metrics(ys, preds)
    if target_scaler is not None:
        preds = target_scaler.inverse_transform(torch.from_numpy(preds)).numpy()
    return regression_metrics(ys, preds)


def fine_tune_alignment(
    model,
    base_model,
    train_dataset,
    task_type: str,
    device,
    epochs: int = 1,
    max_graphs: int = 256,
    lr: float = 5e-4,
    align_weight: float = 0.05,
):
    if epochs <= 0 or align_weight <= 0:
        return model
    model.train().to(device)
    base_model.eval().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    loader = DataLoader(train_dataset[:max_graphs], batch_size=1, shuffle=True)
    for _ in range(epochs):
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            pred, diagnostics = model(data, return_diagnostics=True)
            if task_type == "classification":
                task_loss = masked_classification_loss(pred, _target(data).to(device))
            else:
                task_loss = loss_fn(pred, _target(data).to(device))
            align_loss = torch.tensor(0.0, device=device)
            if diagnostics:
                diag = diagnostics[0]
                sensitivity = _pair_sensitivity_from_base_model(base_model, data, diag["pairs"], task_type)
                if sensitivity.numel() > 1 and sensitivity.std() > 0:
                    residual = diag["residual_score"]
                    residual = (residual - residual.mean()) / residual.std().clamp_min(1e-6)
                    sensitivity = (sensitivity - sensitivity.mean()) / sensitivity.std().clamp_min(1e-6)
                    align_loss = torch.nn.functional.mse_loss(residual, sensitivity)
            loss = task_loss + align_weight * align_loss
            loss.backward()
            optimizer.step()
    return model
