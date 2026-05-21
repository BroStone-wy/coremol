from argparse import Namespace
from pathlib import Path

import torch

from coremol.models.attentivefp_coremol import CoReMolAttentiveFP, CoReMolConfig
from scripts.run_stage1_gate import fixed_base_checkpoint_path
from scripts.run_stage1_gate import load_fixed_base_state
from scripts.run_stage1_gate import variant_training_seed


def test_fixed_base_checkpoint_path_uses_dataset_and_seed():
    args = Namespace(fixed_base_dir="/tmp/coremol-fixed-base")

    path = fixed_base_checkpoint_path(args, "ESOL", 2)

    assert path == Path("/tmp/coremol-fixed-base") / "ESOL_2_base.pt"


def test_fixed_base_checkpoint_path_is_disabled_without_directory():
    args = Namespace(fixed_base_dir="")

    assert fixed_base_checkpoint_path(args, "ESOL", 0) is None


def test_variant_training_seed_depends_only_on_data_seed():
    assert variant_training_seed(2, "base") == 2
    assert variant_training_seed(2, "coremol") == 2


def test_load_fixed_base_state_accepts_legacy_non_adapter_coremol_keys(tmp_path):
    model = CoReMolAttentiveFP(
        in_channels=9,
        edge_dim=3,
        hidden_channels=8,
        out_channels=1,
        num_layers=2,
        num_timesteps=1,
        dropout=0.0,
        coremol=CoReMolConfig(enabled=False),
    )
    state = model.state_dict()
    legacy_state = {}
    for key, value in state.items():
        if key.startswith("adapter."):
            legacy_state[key[len("adapter.") :]] = value
        else:
            legacy_state[key] = value
    path = tmp_path / "legacy.pt"
    torch.save(legacy_state, path)

    report = load_fixed_base_state(model, path)

    assert report["loaded"]
    assert any(key.startswith("adapter.") for key in report["missing"])
    assert any(key.startswith("demand_net") for key in report["unexpected"])
