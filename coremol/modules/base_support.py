import torch


def dense_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=edge_index.device)
    if edge_index.numel() == 0:
        return adj
    adj[edge_index[0], edge_index[1]] = 1.0
    return adj


def row_normalized_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    adj = dense_adjacency(edge_index, num_nodes)
    degree = adj.sum(dim=1, keepdim=True).clamp_min(1.0)
    return adj / degree


def finite_hop_support(
    edge_index: torch.Tensor,
    num_nodes: int,
    max_hops: int = 3,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return per-graph normalized finite-hop communication support in [0, 1]."""
    if num_nodes <= 0:
        return torch.zeros((0, 0), dtype=torch.float32, device=edge_index.device)

    transition = row_normalized_adjacency(edge_index, num_nodes)
    power = transition
    support = torch.zeros_like(transition)
    for _ in range(max_hops):
        support = support + power / float(max_hops)
        power = power @ transition

    support.fill_diagonal_(0.0)
    max_value = support.max()
    if max_value > eps:
        support = support / (max_value + eps)
    return support.clamp(0.0, 1.0)

