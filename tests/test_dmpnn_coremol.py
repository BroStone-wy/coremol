import torch
from torch_geometric.data import Batch, Data

from coremol.models.dmpnn_coremol import CoReMolDMPNN
from coremol.modules.coremol_adapter import CoReMolConfig


def _toy_batch():
    x = torch.tensor(
        [
            [6.0, 0.0, 0.0],
            [7.0, 1.0, 0.0],
            [8.0, 0.0, 1.0],
        ]
    )
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2],
            [1, 0, 2, 1],
        ],
        dtype=torch.long,
    )
    edge_attr = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    return Batch.from_data_list(
        [
            Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([[1.0]])),
            Data(x=x[:2], edge_index=edge_index[:, :2], edge_attr=edge_attr[:2], y=torch.tensor([[0.0]])),
        ]
    )


def test_dmpnn_base_forward_shape():
    model = CoReMolDMPNN(
        in_channels=3,
        edge_dim=2,
        hidden_channels=16,
        out_channels=2,
        num_layers=3,
        num_timesteps=1,
        dropout=0.0,
        coremol=CoReMolConfig(enabled=False),
    )

    out = model(_toy_batch())

    assert out.shape == (2, 2)


def test_dmpnn_coremol_returns_diagnostics():
    model = CoReMolDMPNN(
        in_channels=3,
        edge_dim=2,
        hidden_channels=16,
        out_channels=1,
        num_layers=3,
        num_timesteps=1,
        dropout=0.0,
        coremol=CoReMolConfig(
            enabled=True,
            residual_placement="post",
            residual_message="delta",
            residual_score_space="distribution",
            d_max=2,
        ),
    )

    out, diagnostics = model(_toy_batch(), return_diagnostics=True)

    assert out.shape == (2, 1)
    assert diagnostics
    assert {"pairs", "alpha_base", "alpha_cal", "residual_score"} <= set(diagnostics[-1])


def test_dmpnn_single_graph_coremol_api():
    model = CoReMolDMPNN(
        in_channels=3,
        edge_dim=2,
        hidden_channels=8,
        out_channels=1,
        num_layers=2,
        num_timesteps=1,
        dropout=0.0,
        coremol=CoReMolConfig(enabled=True, residual_message="value", d_max=2),
    )
    atoms = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

    updated, diagnostics = model._apply_single_graph(atoms, edge_index)

    assert updated.shape == atoms.shape
    assert diagnostics
