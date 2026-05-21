import torch
from torch_geometric.data import Data

from coremol.models.attentivefp_coremol import CoReMolAttentiveFP, CoReMolConfig


def test_layerwise_coremol_runs_after_each_atom_update_stage():
    model = CoReMolAttentiveFP(
        in_channels=1,
        edge_dim=1,
        hidden_channels=8,
        out_channels=1,
        num_layers=2,
        num_timesteps=1,
        dropout=0.0,
        coremol=CoReMolConfig(enabled=True, residual_placement="layerwise", num_residual_steps=1),
    )

    calls = []

    def record_apply(atoms, edge_index, batch):
        calls.append(atoms.detach().clone())
        return atoms, []

    model.apply_coremol = record_apply
    data = Data(
        x=torch.ones(4, 1),
        edge_index=torch.tensor([[0, 1, 2, 1, 2, 3], [1, 0, 1, 2, 3, 2]], dtype=torch.long),
        edge_attr=torch.ones(6, 1),
        batch=torch.zeros(4, dtype=torch.long),
    )

    model(data)

    assert len(calls) == 1 + len(model.backbone.atom_convs)


def test_num_residual_steps_repeats_each_layerwise_calibration():
    model = CoReMolAttentiveFP(
        in_channels=1,
        edge_dim=1,
        hidden_channels=8,
        out_channels=1,
        num_layers=1,
        num_timesteps=1,
        dropout=0.0,
        coremol=CoReMolConfig(enabled=True, residual_placement="layerwise", num_residual_steps=2),
    )

    calls = []

    def record_apply(atoms, edge_index, batch):
        calls.append(atoms.detach().clone())
        return atoms, []

    model.apply_coremol = record_apply
    data = Data(
        x=torch.ones(3, 1),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        edge_attr=torch.ones(4, 1),
        batch=torch.zeros(3, dtype=torch.long),
    )

    model(data)

    assert len(calls) == model.coremol.num_residual_steps * (1 + len(model.backbone.atom_convs))


def test_collect_diagnostics_tags_each_coremol_step():
    model = CoReMolAttentiveFP(
        in_channels=1,
        edge_dim=1,
        hidden_channels=8,
        out_channels=1,
        num_layers=1,
        num_timesteps=1,
        dropout=0.0,
        coremol=CoReMolConfig(enabled=True, residual_placement="post", num_residual_steps=2),
    )

    def record_apply(atoms, edge_index, batch, return_diagnostics=True):
        return atoms, [{"alpha_base": torch.ones(1), "alpha_cal": torch.ones(1)}]

    model.apply_coremol = record_apply
    data = Data(
        x=torch.ones(3, 1),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        edge_attr=torch.ones(4, 1),
        batch=torch.zeros(3, dtype=torch.long),
    )

    _, diagnostics = model(data, return_diagnostics=True)

    assert [item["residual_step"] for item in diagnostics] == [0, 1]
    assert [item["residual_stage"] for item in diagnostics] == ["post", "post"]


def test_collect_diagnostics_handles_empty_step_diagnostics():
    model = CoReMolAttentiveFP(
        in_channels=1,
        edge_dim=1,
        hidden_channels=8,
        out_channels=1,
        num_layers=1,
        num_timesteps=1,
        dropout=0.0,
        coremol=CoReMolConfig(enabled=True, residual_placement="post", num_residual_steps=1),
    )

    def record_apply(atoms, edge_index, batch, return_diagnostics=True):
        return atoms, []

    model.apply_coremol = record_apply
    data = Data(
        x=torch.ones(3, 1),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        edge_attr=torch.ones(4, 1),
        batch=torch.zeros(3, dtype=torch.long),
    )

    _, diagnostics = model(data, return_diagnostics=True)

    assert diagnostics == []


def test_delta_residual_message_does_not_shift_identical_atom_states():
    model = CoReMolAttentiveFP(
        in_channels=1,
        edge_dim=1,
        hidden_channels=4,
        out_channels=1,
        num_layers=1,
        num_timesteps=1,
        dropout=0.0,
        coremol=CoReMolConfig(
            enabled=True,
            residual_message="delta",
            beta=0.2,
            tau=0.5,
            residual_gate_init=1.0,
        ),
    )
    model.eval()
    with torch.no_grad():
        model.adapter.value_proj.weight.copy_(torch.eye(4))
        for parameter in model.adapter.demand_net.parameters():
            parameter.zero_()

    atoms = torch.ones(3, 4)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

    next_atoms, _ = model._apply_single_graph(atoms, edge_index)

    torch.testing.assert_close(next_atoms, atoms)


def test_channel_residual_gate_initializes_as_identity():
    model = CoReMolAttentiveFP(
        in_channels=1,
        edge_dim=1,
        hidden_channels=4,
        out_channels=1,
        num_layers=1,
        num_timesteps=1,
        dropout=0.0,
        coremol=CoReMolConfig(
            enabled=True,
            residual_gate_mode="channel",
            residual_gate_init=0.0,
        ),
    )

    atoms = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

    next_atoms, diagnostics = model._apply_single_graph(atoms, edge_index)

    assert model.adapter.residual_gate.shape == (4,)
    torch.testing.assert_close(next_atoms, atoms)
    assert diagnostics[0]["residual_gate"].shape == (4,)
