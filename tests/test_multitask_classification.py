import math

import numpy as np
import torch

from coremol.datasets.moleculenet import TASKS
from coremol.metrics.task_metrics import classification_metrics
from coremol.probes.tcm import _loss_value
from coremol.training.trainer import (
    ExponentialMovingAverage,
    build_optimizer,
    compute_classification_pos_weight,
    masked_classification_loss,
    residual_demand_alignment_loss,
)
from scripts.run_stage1_gate import load_fixed_base_state


def test_remaining_curvflow_classification_tasks_are_registered():
    expected = {
        "BACE": 1,
        "CLINTOX": 2,
        "TOX21": 12,
        "HIV": 1,
        "SIDER": 27,
        "TOXCAST": 617,
    }

    for name, target_dim in expected.items():
        assert TASKS[name]["type"] == "classification"
        assert TASKS[name]["metric"] == "roc_auc"
        assert TASKS[name]["target_dim"] == target_dim


def test_classification_metrics_average_tasks_and_ignore_nan_labels():
    y_true = np.array(
        [
            [0.0, 1.0, np.nan],
            [1.0, 0.0, np.nan],
            [0.0, np.nan, 1.0],
            [1.0, np.nan, 0.0],
        ]
    )
    logits = np.array(
        [
            [-2.0, 2.0, 0.0],
            [2.0, -2.0, 0.0],
            [-1.0, 0.0, 3.0],
            [1.0, 0.0, -3.0],
        ]
    )

    metrics = classification_metrics(y_true, logits)

    assert metrics["roc_auc"] == 1.0
    assert metrics["roc_auc_tasks"] == 3


def test_masked_classification_loss_ignores_nan_labels():
    logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    target = torch.tensor([[1.0, float("nan")], [0.0, float("nan")]])

    loss = masked_classification_loss(logits, target)

    assert math.isclose(float(loss), 0.693147, rel_tol=1e-5)


def test_masked_classification_loss_with_pos_weight_ignores_nan_labels():
    logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    target = torch.tensor([[1.0, float("nan")], [0.0, float("nan")]])

    loss = masked_classification_loss(logits, target, pos_weight=torch.tensor([[2.0, 2.0]]))

    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_compute_classification_pos_weight_ignores_nan_labels():
    dataset = [
        type("Data", (), {"y": torch.tensor([[1.0, float("nan")]])})(),
        type("Data", (), {"y": torch.tensor([[0.0, 1.0]])})(),
        type("Data", (), {"y": torch.tensor([[0.0, 0.0]])})(),
    ]

    pos_weight = compute_classification_pos_weight(dataset)

    assert torch.allclose(pos_weight, torch.tensor([[2.0, 1.0]]))


def test_compute_classification_pos_weight_caps_extreme_weights():
    dataset = [
        type("Data", (), {"y": torch.tensor([[1.0]])})(),
        *[type("Data", (), {"y": torch.tensor([[0.0]])})() for _ in range(100)],
    ]

    pos_weight = compute_classification_pos_weight(dataset, max_weight=10.0)

    assert torch.allclose(pos_weight, torch.tensor([[10.0]]))


def test_pair_sensitivity_loss_ignores_nan_classification_labels():
    logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    target = torch.tensor([[1.0, float("nan")], [0.0, float("nan")]])

    loss = _loss_value(logits, target, "classification")

    assert loss.shape == (2,)
    assert torch.isfinite(loss).all()
    assert math.isclose(float(loss.mean()), 0.693147, rel_tol=1e-5)


def test_residual_demand_alignment_loss_is_finite():
    diagnostics = [
        {
            "pairs": torch.tensor([[0, 1], [0, 2], [1, 0]], dtype=torch.long),
            "demand": torch.tensor([0.8, 0.2, 0.5]),
            "alpha_cal": torch.tensor([0.6, 0.4, 1.0]),
            "residual_score": torch.tensor([0.3, -0.1, 0.2]),
        }
    ]

    loss = residual_demand_alignment_loss(diagnostics)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_exponential_moving_average_updates_and_restores():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = ExponentialMovingAverage(model, decay=0.5)

    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)

    backup = ema.copy_to(model)
    assert torch.allclose(model.weight, torch.tensor([[2.0]]))

    ema.restore(model, backup)
    assert torch.allclose(model.weight, torch.tensor([[3.0]]))


def test_build_optimizer_can_slow_backbone_updates():
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.adapter = torch.nn.Linear(2, 2)
            self.head = torch.nn.Linear(2, 1)

    optimizer = build_optimizer(ToyModel(), lr=1e-3, weight_decay=1e-5, backbone_lr_scale=0.1)

    learning_rates = sorted(group["lr"] for group in optimizer.param_groups)
    assert learning_rates == [1e-4, 1e-3]


def test_load_fixed_base_state_skips_shape_mismatches():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    state = model.state_dict()
    state["0.weight"] = torch.ones(1, 1)

    report = load_fixed_base_state(model, state)

    assert "0.weight" in report["skipped"]
