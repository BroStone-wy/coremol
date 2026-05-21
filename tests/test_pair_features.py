import torch

from coremol.modules.pair_features import candidate_pairs_with_distances
from coremol.datasets.scaffold_split import scaffold_from_smiles


def test_candidate_pairs_exclude_self_and_respect_distance_limit():
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2],
            [1, 0, 2, 1],
        ],
        dtype=torch.long,
    )
    pairs, distances = candidate_pairs_with_distances(
        edge_index=edge_index,
        num_nodes=3,
        d_max=1,
        include_bond_pairs=True,
    )

    assert pairs.tolist() == [[0, 1], [1, 0], [1, 2], [2, 1]]
    assert distances.tolist() == [1.0, 1.0, 1.0, 1.0]


def test_candidate_pairs_include_shortest_two_hop_pairs():
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2],
            [1, 0, 2, 1],
        ],
        dtype=torch.long,
    )
    pairs, distances = candidate_pairs_with_distances(
        edge_index=edge_index,
        num_nodes=3,
        d_max=2,
        include_bond_pairs=True,
    )

    assert pairs.tolist() == [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]
    assert distances.tolist() == [1.0, 2.0, 1.0, 1.0, 2.0, 1.0]


def test_acyclic_molecules_keep_empty_murcko_scaffold_group():
    assert scaffold_from_smiles("CCO") == ""
