import torch
from torch import nn

from scripts.compute_tcm_variants import build_model, load_compatible_state


class _TinyDataset:
    pass


def test_build_model_supports_graphformer_backbone():
    model = build_model(
        dataset=_TinyDataset(),
        dataset_name="BBBP",
        enabled=True,
        hidden_channels=16,
        d_max=2,
        support_hops=2,
        backbone="graphformer",
        edge_dim=2,
        in_channels=3,
        out_channels=1,
        num_layers=2,
        num_timesteps=1,
        dropout=0.0,
        graphformer_num_heads=4,
        graphformer_max_distance=3,
        graphformer_use_graph_token=False,
        graphformer_readout="mean_max",
        graphformer_no_local_gnn=False,
        graphformer_no_distance_bias=False,
        graphformer_use_edge_bias=True,
        graphformer_no_degree_encoding=False,
        graphformer_ffn_ratio=2,
        graphformer_norm_style="pre",
        graphformer_feature_encoder="linear",
    )

    x = torch.ones(4, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    edge_attr = torch.ones(edge_index.size(1), 2)
    batch = torch.zeros(4, dtype=torch.long)

    atoms = model.encode_atoms(x, edge_index, edge_attr, batch)
    out, _ = model.molecule_readout(atoms, batch)

    assert out.shape == (1, 1)


def test_build_model_forwards_residual_gate_max():
    model = build_model(
        dataset=_TinyDataset(),
        dataset_name="BBBP",
        enabled=True,
        hidden_channels=16,
        d_max=2,
        support_hops=2,
        residual_gate_max=0.005,
        backbone="graphformer",
        edge_dim=2,
        in_channels=3,
        out_channels=1,
        num_layers=2,
        num_timesteps=1,
        dropout=0.0,
    )

    assert model.coremol.residual_gate_max == 0.005
    assert model.adapter.coremol.residual_gate_max == 0.005


def test_load_compatible_state_skips_missing_and_mismatched_keys():
    model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
    state = {
        "0.weight": torch.ones_like(model.state_dict()["0.weight"]),
        "0.bias": torch.ones(99),
        "extra.weight": torch.ones(1),
    }

    report = load_compatible_state(model, state)

    assert report["loaded"] == 1
    assert "0.bias" in report["skipped"]
    assert "extra.weight" in report["unexpected"]
