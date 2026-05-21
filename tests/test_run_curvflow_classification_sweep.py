from scripts.run_curvflow_classification_sweep import build_command


def test_build_command_forwards_random_split_fractions():
    command = build_command(
        {
            "datasets": ["TOX21"],
            "seeds": [3, 4],
            "split_strategy": "random",
            "random_train_fraction": 0.8,
            "random_valid_fraction": 0.1,
        },
        "curvflow_classification_sweep/tox21_random_80_10_10_smoke",
    )

    assert "--random_train_fraction" in command
    assert command[command.index("--random_train_fraction") + 1] == "0.8"
    assert "--random_valid_fraction" in command
    assert command[command.index("--random_valid_fraction") + 1] == "0.1"
