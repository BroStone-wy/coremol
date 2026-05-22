from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class InterfaceReferenceGraph:
    cross_pair_index: torch.Tensor
    c_ref_cross: torch.Tensor
    cross_features: torch.Tensor
    cross_batch: torch.Tensor


class InterfaceReferenceGraphConstructor(nn.Module):
    def __init__(
        self,
        cutoff: float = 6.0,
        topk_per_lig_atom: int = 16,
        sigma: float = 2.0,
    ):
        super().__init__()
        self.cutoff = float(cutoff)
        self.topk_per_lig_atom = int(topk_per_lig_atom)
        self.sigma = float(sigma)

    def forward(self, data) -> InterfaceReferenceGraph:
        if not hasattr(data, "n_nodes"):
            raise ValueError("GEMS-style affinity batches require n_nodes metadata")
        device = data.x.device
        metadata = torch.as_tensor(data.n_nodes, device=device).view(-1, 3).long()
        ptr = getattr(data, "ptr", None)
        if ptr is None:
            ptr = self._ptr_from_metadata(metadata, device=device)

        all_pairs = []
        all_distances = []
        all_topological = []
        all_batch = []
        for graph_id, (total_nodes, n_lig_nodes, n_prot_nodes) in enumerate(metadata.tolist()):
            node_start = int(ptr[graph_id].item())
            lig_idx = torch.arange(node_start, node_start + n_lig_nodes, device=device)
            pocket_idx = torch.arange(node_start + n_lig_nodes, node_start + total_nodes, device=device)
            pairs, distances, topological = self._single_graph_pairs(data, lig_idx, pocket_idx)
            if pairs.numel() == 0:
                continue
            all_pairs.append(pairs)
            all_distances.append(distances)
            all_topological.append(topological)
            all_batch.append(torch.full((pairs.size(0),), graph_id, dtype=torch.long, device=device))

        if not all_pairs:
            empty_pairs = torch.empty(2, 0, dtype=torch.long, device=device)
            empty = torch.empty(0, dtype=torch.float, device=device)
            empty_features = torch.empty(0, 6, dtype=torch.float, device=device)
            return InterfaceReferenceGraph(empty_pairs, empty, empty_features, empty.long())

        pair_matrix = torch.cat(all_pairs, dim=0)
        distances = torch.cat(all_distances, dim=0).float()
        topological = torch.cat(all_topological, dim=0).float()
        cross_batch = torch.cat(all_batch, dim=0)
        c_ref = torch.exp(-(distances.square()) / max(self.sigma * self.sigma, 1e-8)).clamp_min(1e-8)
        cross_features = torch.stack(
            [
                distances / max(self.cutoff, 1e-8),
                c_ref,
                (distances <= 4.0).float(),
                (distances <= 6.0).float(),
                topological,
                torch.ones_like(distances),
            ],
            dim=-1,
        )
        return InterfaceReferenceGraph(pair_matrix.t().contiguous(), c_ref, cross_features, cross_batch)

    def _single_graph_pairs(self, data, lig_idx: torch.Tensor, pocket_idx: torch.Tensor):
        if lig_idx.numel() == 0 or pocket_idx.numel() == 0:
            device = lig_idx.device
            return (
                torch.empty(0, 2, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.float, device=device),
                torch.empty(0, dtype=torch.bool, device=device),
            )

        pos = getattr(data, "pos", None)
        if pos is not None:
            distances = torch.cdist(pos[lig_idx].float(), pos[pocket_idx].float())
            candidate_mask = distances <= self.cutoff
            if self.topk_per_lig_atom > 0:
                k = min(self.topk_per_lig_atom, pocket_idx.numel())
                topk_idx = distances.topk(k=k, largest=False, dim=1).indices
                candidate_mask.scatter_(1, topk_idx, True)
            local_lig, local_pocket = candidate_mask.nonzero(as_tuple=True)
            pair_matrix = torch.stack([lig_idx[local_lig], pocket_idx[local_pocket]], dim=1)
            pair_distances = distances[local_lig, local_pocket]
            topological = self._topological_mask(data.edge_index, pair_matrix)
            return pair_matrix, pair_distances, topological

        pair_matrix = self._cross_edges(data.edge_index, lig_idx, pocket_idx)
        if pair_matrix.numel() == 0:
            lig_grid, pocket_grid = torch.meshgrid(lig_idx, pocket_idx, indexing="ij")
            pair_matrix = torch.stack([lig_grid.reshape(-1), pocket_grid.reshape(-1)], dim=1)
        topological = self._topological_mask(data.edge_index, pair_matrix)
        distances = self._edge_attr_distances(data, pair_matrix)
        return pair_matrix, distances, topological

    def _edge_attr_distances(self, data, pair_matrix: torch.Tensor) -> torch.Tensor:
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is None or edge_attr.size(-1) < 7 or pair_matrix.numel() == 0:
            return torch.full((pair_matrix.size(0),), 2.0, dtype=torch.float, device=pair_matrix.device)
        edge_index = data.edge_index
        node_key_base = int(edge_index.max().item()) + 1
        edge_keys = edge_index[0] * node_key_base + edge_index[1]
        pair_keys = pair_matrix[:, 0] * node_key_base + pair_matrix[:, 1]

        sorted_keys, order = edge_keys.sort()
        insertion = torch.searchsorted(sorted_keys, pair_keys)
        found = insertion < sorted_keys.numel()
        safe_insertion = insertion.clamp_max(max(sorted_keys.numel() - 1, 0))
        found = found & (sorted_keys[safe_insertion] == pair_keys)

        distances = torch.full((pair_matrix.size(0),), 2.0, dtype=torch.float, device=pair_matrix.device)
        if bool(found.any()):
            edge_distances = edge_attr[:, 3:7].float().clamp_min(0.0).min(dim=1).values * 10.0
            distances[found] = edge_distances[order[safe_insertion[found]]]
        return distances

    def _cross_edges(self, edge_index: torch.Tensor, lig_idx: torch.Tensor, pocket_idx: torch.Tensor) -> torch.Tensor:
        lig_mask = torch.isin(edge_index[0], lig_idx) & torch.isin(edge_index[1], pocket_idx)
        reverse_mask = torch.isin(edge_index[1], lig_idx) & torch.isin(edge_index[0], pocket_idx)
        forward = edge_index[:, lig_mask].t()
        reverse = edge_index[[1, 0], :][:, reverse_mask].t()
        pairs = torch.cat([forward, reverse], dim=0)
        if pairs.numel() == 0:
            return pairs
        return torch.unique(pairs, dim=0)

    def _topological_mask(self, edge_index: torch.Tensor, pair_matrix: torch.Tensor) -> torch.Tensor:
        if pair_matrix.numel() == 0:
            return torch.empty(0, dtype=torch.bool, device=edge_index.device)
        cross_edges = self._cross_edges(edge_index, pair_matrix[:, 0].unique(), pair_matrix[:, 1].unique())
        if cross_edges.numel() == 0:
            return torch.zeros(pair_matrix.size(0), dtype=torch.bool, device=pair_matrix.device)
        pair_keys = pair_matrix[:, 0] * (int(edge_index.max().item()) + 1) + pair_matrix[:, 1]
        edge_keys = cross_edges[:, 0] * (int(edge_index.max().item()) + 1) + cross_edges[:, 1]
        return torch.isin(pair_keys, edge_keys)

    def _ptr_from_metadata(self, metadata: torch.Tensor, device) -> torch.Tensor:
        counts = metadata[:, 0]
        ptr = torch.zeros(counts.numel() + 1, dtype=torch.long, device=device)
        ptr[1:] = counts.cumsum(dim=0)
        return ptr
