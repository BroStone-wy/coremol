import torch
from torch_geometric.data import Data

from coremol.models.graphformer_coremol import CoReMolGraphformer
from coremol.modules.coremol_adapter import CoReMolConfig
from scripts.run_curvflow_classification_sweep import build_command


def _data():
    return Data(
        x=torch.ones(5, 3),
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
            dtype=torch.long,
        ),
        edge_attr=torch.ones(8, 2),
        batch=torch.zeros(5, dtype=torch.long),
    )


def test_graphformer_forward_shape():
    model = CoReMolGraphformer(
        in_channels=3,
        edge_dim=2,
        hidden_channels=16,
        out_channels=2,
        num_layers=2,
        num_timesteps=1,
        dropout=0.0,
        coremol=CoReMolConfig(enabled=False),
    )

    out = model(_data())

    assert out.shape == (1, 2)


def test_graphformer_coremol_returns_diagnostics():
    model = CoReMolGraphformer(
        in_channels=3,
        edge_dim=2,
        hidden_channels=16,
        out_channels=2,
        num_layers=1,
        num_timesteps=1,
        dropout=0.0,
        coremol=CoReMolConfig(enabled=True, residual_message="delta", d_max=2),
    )

    out, diagnostics = model(_data(), return_diagnostics=True)

    assert out.shape == (1, 2)
    assert diagnostics
    assert "alpha_cal" in diagnostics[0]


def test_graphformer_profile_toggles_forward_shape_and_diagnostics():
    model = CoReMolGraphformer(
        in_channels=3,
        edge_dim=2,
        hidden_channels=16,
        out_channels=2,
        num_layers=2,
        num_timesteps=1,
        dropout=0.0,
        coremol=CoReMolConfig(enabled=True, residual_message="delta", d_max=2),
        use_graph_token=True,
        readout="graph_token",
        use_local_gnn=False,
        use_distance_bias=True,
        use_edge_bias=True,
        use_degree_encoding=False,
        ffn_ratio=2,
        norm_style="post",
        num_heads=4,
        max_distance=3,
    )

    out, diagnostics = model(_data(), return_diagnostics=True)

    assert out.shape == (1, 2)
    assert diagnostics
    assert model.backbone.use_graph_token is True
    assert model.backbone.readout == "graph_token"
    assert model.backbone.use_edge_bias is True


def test_graphformer_categorical_feature_encoder_forward_shape():
    model = CoReMolGraphformer(
        in_channels=3,
        edge_dim=2,
        hidden_channels=16,
        out_channels=2,
        num_layers=2,
        num_timesteps=1,
        dropout=0.0,
        coremol=CoReMolConfig(enabled=False),
        feature_encoder="categorical",
        use_edge_bias=True,
    )

    out = model(_data())

    assert out.shape == (1, 2)
    assert model.backbone.feature_encoder == "categorical"


def test_graphformer_layerwise_coremol_runs_after_each_encoder_layer():
    model = CoReMolGraphformer(
        in_channels=3,
        edge_dim=2,
        hidden_channels=16,
        out_channels=2,
        num_layers=3,
        num_timesteps=1,
        dropout=0.0,
        coremol=CoReMolConfig(enabled=True, residual_placement="layerwise", num_residual_steps=1),
    )
    calls = []

    def record_apply(atoms, edge_index, batch, return_diagnostics=True):
        calls.append(atoms.detach().clone())
        return atoms, []

    model.apply_coremol = record_apply

    model(_data())

    assert len(calls) == len(model.backbone.layers)


def test_graphformer_backbone_arg_is_forwarded():
    command = build_command(
        {
            "datasets": ["BBBP"],
            "seeds": [0],
            "backbone": "graphformer",
            "epochs": 1,
        },
        "graphformer_smoke",
    )

    assert "--backbone" in command
    assert command[command.index("--backbone") + 1] == "graphformer"


def test_graphformer_profile_args_are_forwarded():
    command = build_command(
        {
            "datasets": ["SIDER"],
            "seeds": [0],
            "backbone": "graphformer",
            "epochs": 1,
            "graphformer_use_graph_token": True,
            "graphformer_readout": "graph_token",
            "graphformer_use_edge_bias": True,
            "graphformer_no_local_gnn": True,
            "graphformer_no_distance_bias": True,
            "graphformer_no_degree_encoding": True,
            "graphformer_ffn_ratio": 2,
            "graphformer_norm_style": "post",
            "graphformer_num_heads": 8,
            "graphformer_max_distance": 7,
            "graphformer_feature_encoder": "categorical",
        },
        "graphformer_profile_smoke",
    )

    assert "--graphformer_use_graph_token" in command
    assert "--graphformer_readout" in command
    assert command[command.index("--graphformer_readout") + 1] == "graph_token"
    assert "--graphformer_use_edge_bias" in command
    assert "--graphformer_no_local_gnn" in command
    assert "--graphformer_no_distance_bias" in command
    assert "--graphformer_no_degree_encoding" in command
    assert command[command.index("--graphformer_ffn_ratio") + 1] == "2"
    assert command[command.index("--graphformer_norm_style") + 1] == "post"
    assert command[command.index("--graphformer_num_heads") + 1] == "8"
    assert command[command.index("--graphformer_max_distance") + 1] == "7"
    assert command[command.index("--graphformer_feature_encoder") + 1] == "categorical"
