import torch

from coremol.modules.coremol_adapter import CoReMolConfig, CoReMolResidualAdapter


def test_adapter_keeps_batched_molecules_independent():
    adapter = CoReMolResidualAdapter(hidden_channels=4, coremol=CoReMolConfig(enabled=True))
    atoms = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 3, 4], [1, 0, 4, 3]], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

    _, diagnostics = adapter(atoms, edge_index, batch)

    assert diagnostics
    for item in diagnostics:
        pairs = item["pairs"]
        assert (pairs < 3).all() or (pairs < 2).all()


def test_adapter_diagnostics_report_residual_magnitude():
    adapter = CoReMolResidualAdapter(hidden_channels=4, coremol=CoReMolConfig(enabled=True))
    atoms = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    batch = torch.zeros(3, dtype=torch.long)

    _, diagnostics = adapter(atoms, edge_index, batch)

    assert diagnostics
    for key in ["update_norm_mean", "atom_norm_mean", "update_atom_norm_ratio", "residual_gate"]:
        assert key in diagnostics[0]


def test_adapter_can_preserve_raw_residual_update_scale_without_layernorm():
    adapter = CoReMolResidualAdapter(
        hidden_channels=4,
        coremol=CoReMolConfig(enabled=True, residual_norm_mode="none"),
    )
    updates = torch.randn(3, 4)

    normalized = adapter.normalize_updates(updates)

    torch.testing.assert_close(normalized, updates)


def test_distribution_residual_score_compares_demand_and_support_on_same_scale():
    adapter = CoReMolResidualAdapter(
        hidden_channels=4,
        coremol=CoReMolConfig(
            enabled=True,
            d_max=2,
            support_hops=2,
            residual_score_space="distribution",
            dropout=0.0,
        ),
    )
    for parameter in adapter.demand_net.parameters():
        parameter.data.zero_()

    atoms = torch.randn(4, 4)
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2],
        ],
        dtype=torch.long,
    )

    _, diagnostics = adapter.apply_single_graph(atoms, edge_index)

    residual = diagnostics[0]["residual_score"]
    pairs = diagnostics[0]["pairs"]
    for src in pairs[:, 0].unique():
        src_mask = pairs[:, 0] == src
        assert torch.isclose(residual[src_mask].sum(), torch.tensor(0.0), atol=1e-6)


def test_rir_residual_score_uses_signed_shift_directly():
    adapter = CoReMolResidualAdapter(
        hidden_channels=4,
        coremol=CoReMolConfig(enabled=True, residual_score_space="rir"),
    )
    pair_score = torch.tensor([-2.0, 0.0, 2.0])
    support = torch.tensor([0.2, 0.5, 0.3])
    src = torch.tensor([0, 0, 0], dtype=torch.long)

    residual_score, alpha_ref, demand_alpha = adapter.compute_residual_scores(
        pair_score=pair_score,
        support=support,
        base_logits=torch.log(support),
        src=src,
        num_nodes=1,
    )

    torch.testing.assert_close(residual_score, pair_score)
    torch.testing.assert_close(alpha_ref.sum(), torch.tensor(1.0))
    assert demand_alpha is None


def test_rir_diagnostics_use_reference_and_rewired_names():
    adapter = CoReMolResidualAdapter(
        hidden_channels=4,
        coremol=CoReMolConfig(
            enabled=True,
            d_max=2,
            support_hops=2,
            residual_score_space="rir",
            dropout=0.0,
        ),
    )
    atoms = torch.randn(4, 4)
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2],
        ],
        dtype=torch.long,
    )

    _, diagnostics = adapter.apply_single_graph(atoms, edge_index)
    diag = diagnostics[0]

    assert diag["residual_score_space"] == "rir"
    for key in ["c_ref", "alpha_ref", "alpha_rew", "shift_raw", "rewiring_delta"]:
        assert key in diag
    assert "demand" not in diag
    assert "demand_alpha" not in diag
    torch.testing.assert_close(diag["alpha_ref"], diag["alpha_base"])
    torch.testing.assert_close(diag["alpha_rew"], diag["alpha_cal"])
    torch.testing.assert_close(diag["shift_raw"], diag["residual_score"])


def test_source_centered_rir_shift_is_relative_within_each_source_atom():
    adapter = CoReMolResidualAdapter(
        hidden_channels=4,
        coremol=CoReMolConfig(
            enabled=True,
            residual_score_space="rir",
            residual_shift_centering="source",
        ),
    )
    residual_score = torch.tensor([1.0, 3.0, -2.0, 2.0])
    src = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    centered = adapter.center_residual_scores(residual_score, src=src, num_nodes=2)

    torch.testing.assert_close(centered, torch.tensor([-1.0, 1.0, -2.0, 2.0]))
    for atom in src.unique():
        assert torch.isclose(centered[src == atom].mean(), torch.tensor(0.0), atol=1e-6)
