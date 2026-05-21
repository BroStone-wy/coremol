import copy

import torch
from torch_geometric.data import Data

from coremol.models.attentivefp_coremol import CoReMolAttentiveFP, CoReMolConfig
from coremol.datasets.random_split import curvflow_random_split_indices
from coremol.training.trainer import RegressionTargetScaler
from coremol.training.pretraining import (
    adapt_zinc_item,
    freeze_attentivefp_atom_encoder,
    freeze_backbone,
    load_matching_coremol_state,
    load_matching_backbone_state,
)


def _model(in_channels, edge_dim):
    return CoReMolAttentiveFP(
        in_channels=in_channels,
        edge_dim=edge_dim,
        hidden_channels=8,
        out_channels=1,
        num_layers=2,
        num_timesteps=1,
        dropout=0.0,
        coremol=CoReMolConfig(enabled=False),
    )


def test_zinc_adapter_makes_scalar_edge_attr_2d_float_features():
    data = Data(
        x=torch.tensor([[0], [1]], dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.tensor([1, 2], dtype=torch.long),
        y=torch.tensor([0.5]),
    )

    adapted = adapt_zinc_item(data)

    assert adapted is not data
    assert adapted.x.shape == (2, 1)
    assert adapted.edge_attr.shape == (2, 1)
    assert adapted.x.dtype == torch.float32
    assert adapted.edge_attr.dtype == torch.float32


def test_load_matching_backbone_state_skips_input_layer_shape_mismatches():
    pretrained = _model(in_channels=1, edge_dim=1)
    finetune = _model(in_channels=9, edge_dim=3)
    before = copy.deepcopy(finetune.backbone.state_dict())

    report = load_matching_backbone_state(finetune, pretrained.state_dict())
    after = finetune.backbone.state_dict()

    assert report["loaded"] > 0
    assert "backbone.lin1.weight" in report["skipped"]
    assert torch.equal(after["lin1.weight"], before["lin1.weight"])


def test_load_matching_coremol_state_loads_adapter_and_skips_input_layer_mismatches():
    pretrained = CoReMolAttentiveFP(
        in_channels=1,
        edge_dim=1,
        hidden_channels=8,
        out_channels=1,
        num_layers=2,
        num_timesteps=1,
        dropout=0.0,
        coremol=CoReMolConfig(enabled=True),
    )
    finetune = CoReMolAttentiveFP(
        in_channels=9,
        edge_dim=3,
        hidden_channels=8,
        out_channels=1,
        num_layers=2,
        num_timesteps=1,
        dropout=0.0,
        coremol=CoReMolConfig(enabled=True),
    )
    with torch.no_grad():
        pretrained.adapter.residual_gate.fill_(0.37)
        pretrained.adapter.value_proj.weight.fill_(0.11)
    before = copy.deepcopy(finetune.state_dict())

    report = load_matching_coremol_state(
        finetune,
        {"model_state_dict": pretrained.state_dict()},
    )
    after = finetune.state_dict()

    assert report["loaded"] > 0
    assert "backbone.lin1.weight" in report["skipped"]
    assert torch.equal(after["backbone.lin1.weight"], before["backbone.lin1.weight"])
    assert torch.equal(after["adapter.residual_gate"], pretrained.state_dict()["adapter.residual_gate"])
    assert torch.equal(after["adapter.value_proj.weight"], pretrained.state_dict()["adapter.value_proj.weight"])


def test_regression_target_scaler_round_trips_targets():
    scaler = RegressionTargetScaler.from_values(torch.tensor([[1.0], [3.0], [5.0]]))
    values = torch.tensor([[1.0], [5.0]])

    restored = scaler.inverse_transform(scaler.transform(values))

    assert torch.allclose(restored, values)


def test_curvflow_random_split_defaults_to_70_20_10_layout():
    split = curvflow_random_split_indices(100, seed=0)

    assert len(split["train"]) == 70
    assert len(split["valid"]) == 20
    assert len(split["test"]) == 10
    assert sorted(split["train"] + split["valid"] + split["test"]) == list(range(100))


def test_curvflow_random_split_can_use_80_10_10_layout():
    split = curvflow_random_split_indices(100, seed=0, train_fraction=0.8, valid_fraction=0.1)

    assert len(split["train"]) == 80
    assert len(split["valid"]) == 10
    assert len(split["test"]) == 10
    assert sorted(split["train"] + split["valid"] + split["test"]) == list(range(100))


def test_freeze_backbone_keeps_only_coremol_parameters_trainable():
    model = CoReMolAttentiveFP(
        in_channels=9,
        edge_dim=3,
        hidden_channels=8,
        out_channels=1,
        num_layers=2,
        num_timesteps=1,
        dropout=0.0,
        coremol=CoReMolConfig(enabled=True),
    )

    trainable = freeze_backbone(model)

    assert trainable
    assert all(not name.startswith("backbone.") for name in trainable)
    assert all(
        parameter.requires_grad == (name in trainable)
        for name, parameter in model.named_parameters()
    )


def test_freeze_attentivefp_atom_encoder_keeps_readout_and_adapter_trainable():
    model = CoReMolAttentiveFP(
        in_channels=9,
        edge_dim=3,
        hidden_channels=8,
        out_channels=1,
        num_layers=2,
        num_timesteps=1,
        dropout=0.0,
        coremol=CoReMolConfig(enabled=True),
    )

    trainable = freeze_attentivefp_atom_encoder(model)

    assert "backbone.lin1.weight" not in trainable
    assert "backbone.gate_conv.lin1.weight" not in trainable
    assert any(name.startswith("backbone.mol_conv") for name in trainable)
    assert any(name.startswith("backbone.lin2") for name in trainable)
    assert any(name.startswith("adapter.") for name in trainable)
