import torch


def _shortest_distances(edge_index: torch.Tensor, num_nodes: int, d_max: int) -> torch.Tensor:
    device = edge_index.device
    distances = torch.full((num_nodes, num_nodes), float("inf"), dtype=torch.float32, device=device)
    if num_nodes <= 0:
        return distances

    eye = torch.eye(num_nodes, dtype=torch.bool, device=device)
    distances[eye] = 0.0
    if edge_index.numel() == 0 or d_max < 1:
        return distances

    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    adjacency[edge_index[0], edge_index[1]] = 1.0

    seen = eye.clone()
    frontier = eye.float()
    for hop in range(1, d_max + 1):
        frontier = (frontier @ adjacency > 0).float()
        new_reachable = frontier.bool() & (~seen)
        distances[new_reachable] = float(hop)
        seen = seen | new_reachable
        if bool(seen.all()):
            break
    return distances


def candidate_pairs_with_distances(
    edge_index: torch.Tensor,
    num_nodes: int,
    d_max: int = 4,
    include_bond_pairs: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    distances = _shortest_distances(edge_index, num_nodes, d_max)
    bond_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=edge_index.device)
    if edge_index.numel() > 0:
        bond_mask[edge_index[0], edge_index[1]] = True

    mask = (distances >= 1) & (distances <= float(d_max))
    if not include_bond_pairs:
        mask = mask & (~bond_mask)

    pairs = mask.nonzero(as_tuple=False)
    pair_distances = distances[pairs[:, 0], pairs[:, 1]] if pairs.numel() else torch.empty(0, device=edge_index.device)
    return pairs.long(), pair_distances.float()
