from scripts.run_graphformer_profile_sweep import build_stage_commands


def test_baseline_stage_builds_profile_probe_command():
    commands = build_stage_commands(
        stage="baseline",
        datasets=["SIDER"],
        profiles=["graph_token_edge"],
        seeds=[0],
        messages=["delta"],
        epochs=3,
        batch_size=32,
    )

    assert len(commands) == 1
    command = commands[0]
    assert "--datasets" in command
    assert command[command.index("--datasets") + 1] == "SIDER"
    assert "--variants" in command
    assert command[command.index("--variants") + 1] == "base"
    assert "--graphformer_use_graph_token" in command
    assert "--graphformer_use_edge_bias" in command
    assert "--graphformer_readout" in command
    assert command[command.index("--graphformer_readout") + 1] == "graph_token"
    assert "--results_name" in command
    assert command[command.index("--results_name") + 1] == "graphformer_profile_sweep/SIDER/graph_token_edge/baseline_seed0"


def test_coremol_stage_builds_fixed_base_value_and_delta_commands():
    commands = build_stage_commands(
        stage="coremol",
        datasets=["TOXCAST"],
        profiles=["no_local_edge"],
        seeds=[0, 1, 2],
        messages=["value", "delta"],
        epochs=5,
        batch_size=16,
    )

    assert len(commands) == 2
    for command, message in zip(commands, ["value", "delta"]):
        assert "--variants" in command
        assert command[command.index("--variants") + 1 : command.index("--variants") + 3] == ["base", "coremol"]
        assert "--fixed_base_dir" in command
        assert command[command.index("--fixed_base_dir") + 1] == (
            "CORMOL/results/graphformer_profile_sweep/TOXCAST/no_local_edge/baseline_checkpoints"
        )
        assert "--residual_message" in command
        assert command[command.index("--residual_message") + 1] == message
        assert "--graphformer_no_local_gnn" in command
        assert "--graphformer_use_edge_bias" in command
        assert "--results_name" in command
        assert command[command.index("--results_name") + 1] == (
            f"graphformer_profile_sweep/TOXCAST/no_local_edge/coremol_{message}_seeds0_1_2"
        )
