import torch

from coremol.modules.base_support import finite_hop_support


def test_finite_hop_support_prefers_direct_neighbors():
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2],
            [1, 0, 2, 1],
        ],
        dtype=torch.long,
    )
    support = finite_hop_support(edge_index=edge_index, num_nodes=3, max_hops=2)

    assert support.shape == (3, 3)
    assert support[0, 0] == 0
    assert support[0, 1] > support[0, 2]
    assert torch.isclose(support.max(), torch.tensor(1.0))

