from dataclasses import dataclass
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, softmax

from coremol.modules.base_support import finite_hop_support
from coremol.modules.pair_features import candidate_pairs_with_distances


@dataclass
class CoReMolConfig:
    enabled: bool = False
    d_max: int = 4
    support_hops: int = 3
    include_bond_pairs: bool = True
    beta: float = 0.1
    tau: float = 0.5
    residual_gate_init: float = 0.1
    residual_gate_max: float = 0.0
    residual_gate_mode: str = "scalar"
    residual_placement: str = "post"
    num_residual_steps: int = 1
    residual_message: str = "value"
    residual_score_space: str = "intensity"
    residual_shift_centering: str = "none"
    residual_norm_mode: str = "layernorm"
    dropout: float = 0.1

    def __post_init__(self):
        if self.residual_placement not in {"post", "layerwise", "both"}:
            raise ValueError("residual_placement must be one of: post, layerwise, both")
        if self.num_residual_steps < 1:
            raise ValueError("num_residual_steps must be >= 1")
        if self.residual_message not in {"value", "delta"}:
            raise ValueError("residual_message must be one of: value, delta")
        if self.residual_score_space not in {"intensity", "distribution", "rir"}:
            raise ValueError("residual_score_space must be one of: intensity, distribution, rir")
        if self.residual_shift_centering not in {"none", "source"}:
            raise ValueError("residual_shift_centering must be one of: none, source")
        if self.residual_norm_mode not in {"layernorm", "none"}:
            raise ValueError("residual_norm_mode must be one of: layernorm, none")
        if self.residual_gate_mode not in {"scalar", "channel"}:
            raise ValueError("residual_gate_mode must be one of: scalar, channel")
        if self.residual_gate_max < 0:
            raise ValueError("residual_gate_max must be >= 0")


class CoReMolResidualAdapter(nn.Module):
    """Backbone-agnostic node-pair residual communication adapter."""

    def __init__(self, hidden_channels: int, coremol: CoReMolConfig):
        super().__init__()
        self.coremol = coremol
        pair_dim = hidden_channels * 5 + 3
        self.demand_net = nn.Sequential(
            nn.Linear(pair_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(coremol.dropout),
            nn.Linear(hidden_channels, 1),
        )
        self.value_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.residual_norm = nn.LayerNorm(hidden_channels)
        if coremol.residual_gate_mode == "channel":
            self.residual_gate = nn.Parameter(torch.full((hidden_channels,), float(coremol.residual_gate_init)))
        else:
            self.residual_gate = nn.Parameter(torch.tensor(float(coremol.residual_gate_init)))
        self._structure_cache = OrderedDict()
        self._structure_cache_limit = 50000

    def effective_residual_gate(self):
        if self.coremol.residual_gate_max <= 0:
            return self.residual_gate
        return self.coremol.residual_gate_max * torch.tanh(self.residual_gate / self.coremol.residual_gate_max)

    def normalize_updates(self, updates):
        if self.coremol.residual_norm_mode == "none":
            return updates
        return self.residual_norm(updates)

    def pair_score_from_features(self, z):
        raw_score = self.demand_net(z).view(-1)
        if self.coremol.residual_score_space == "rir":
            return raw_score, None
        demand = torch.sigmoid(raw_score)
        return demand, demand

    def compute_residual_scores(self, pair_score, support, base_logits, src, num_nodes):
        alpha_base = softmax(base_logits, src, num_nodes=num_nodes)
        if self.coremol.residual_score_space == "rir":
            return pair_score, alpha_base, None
        if self.coremol.residual_score_space == "distribution":
            demand_alpha = softmax(torch.log(pair_score.clamp_min(1e-8)), src, num_nodes=num_nodes)
            return demand_alpha - alpha_base, alpha_base, demand_alpha
        return pair_score - support, alpha_base, None

    def center_residual_scores(self, residual_score, src, num_nodes):
        if self.coremol.residual_shift_centering == "none":
            return residual_score
        source_mean = scatter(residual_score, src, dim=0, dim_size=num_nodes, reduce="mean")
        return residual_score - source_mean[src]

    def forward(self, atoms, edge_index, batch, return_diagnostics: bool = True):
        if not return_diagnostics:
            return self._forward_fast(atoms, edge_index, batch), []

        calibrated = atoms.clone()
        diagnostics = []
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0

        for graph_id in range(num_graphs):
            node_idx = (batch == graph_id).nonzero(as_tuple=False).reshape(-1)
            if node_idx.numel() <= 1:
                continue
            edge_mask = (batch[edge_index[0]] == graph_id) & (batch[edge_index[1]] == graph_id)
            local_edge_index = edge_index[:, edge_mask] - node_idx[0]
            local_atoms = atoms[node_idx]
            updated, diag = self.apply_single_graph(local_atoms, local_edge_index, return_diagnostics=return_diagnostics)
            calibrated[node_idx] = updated
            if return_diagnostics:
                diagnostics.extend(diag)
        return calibrated, diagnostics

    def _forward_fast(self, atoms, edge_index, batch):
        graph_pieces = []
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        for graph_id in range(num_graphs):
            node_idx = (batch == graph_id).nonzero(as_tuple=False).reshape(-1)
            if node_idx.numel() <= 1:
                continue
            edge_mask = (batch[edge_index[0]] == graph_id) & (batch[edge_index[1]] == graph_id)
            local_edge_index = edge_index[:, edge_mask] - node_idx[0]
            pairs, distances, support = self._graph_structure(local_edge_index, int(node_idx.numel()))
            if pairs.numel() == 0:
                continue
            c = support[pairs[:, 0], pairs[:, 1]].clamp_min(1e-8)
            graph_pieces.append((pairs + node_idx[0], distances, c))

        if not graph_pieces:
            return atoms

        pairs = torch.cat([piece[0] for piece in graph_pieces], dim=0)
        distances = torch.cat([piece[1] for piece in graph_pieces], dim=0)
        c = torch.cat([piece[2] for piece in graph_pieces], dim=0)
        src, dst = pairs[:, 0], pairs[:, 1]

        graph_context = scatter(atoms, batch, dim=0, dim_size=num_graphs, reduce="mean")
        mol_context = graph_context[batch[src]]
        dist_feat = (distances.to(atoms.device) / float(max(self.coremol.d_max, 1))).unsqueeze(-1)
        support_feat = c.unsqueeze(-1)
        bond_feat = (distances.to(atoms.device) <= 1.0).float().unsqueeze(-1)
        pair_abs_diff = (atoms[src] - atoms[dst]).abs()
        pair_product = atoms[src] * atoms[dst]
        z = torch.cat(
            [atoms[src], atoms[dst], pair_abs_diff, pair_product, mol_context, dist_feat, support_feat, bond_feat],
            dim=-1,
        )

        pair_score, _ = self.pair_score_from_features(z)
        base_logits = torch.log(c)
        residual_score, all_alpha_base, _ = self.compute_residual_scores(
            pair_score=pair_score,
            support=c,
            base_logits=base_logits,
            src=src,
            num_nodes=atoms.size(0),
        )
        effective_residual_score = self.center_residual_scores(residual_score, src=src, num_nodes=atoms.size(0))
        rewiring_delta = self.coremol.beta * torch.tanh(effective_residual_score / self.coremol.tau)
        cal_logits = base_logits + rewiring_delta

        values = self.value_proj(atoms)
        all_alpha_cal = softmax(cal_logits, src, num_nodes=atoms.size(0))
        diff = all_alpha_cal - all_alpha_base
        message_values = values[dst]
        if self.coremol.residual_message == "delta":
            message_values = values[dst] - values[src]
        residual_updates = scatter(diff.unsqueeze(-1) * message_values, src, dim=0, dim_size=atoms.size(0), reduce="sum")
        normalized_updates = self.normalize_updates(residual_updates)
        dropped_updates = F.dropout(normalized_updates, p=self.coremol.dropout, training=self.training)
        return atoms + self.effective_residual_gate() * dropped_updates

    def apply_single_graph(self, atoms, edge_index, return_diagnostics: bool = True):
        n = atoms.size(0)
        pairs, distances, support = self._graph_structure(edge_index, n)
        if pairs.numel() == 0:
            return atoms, []

        src, dst = pairs[:, 0], pairs[:, 1]
        c = support[src, dst].clamp_min(1e-8)
        mol_context = atoms.mean(dim=0, keepdim=True).expand(pairs.size(0), -1)
        dist_feat = (distances.to(atoms.device) / float(max(self.coremol.d_max, 1))).unsqueeze(-1)
        support_feat = c.unsqueeze(-1)
        bond_feat = (distances.to(atoms.device) <= 1.0).float().unsqueeze(-1)
        pair_abs_diff = (atoms[src] - atoms[dst]).abs()
        pair_product = atoms[src] * atoms[dst]
        z = torch.cat(
            [atoms[src], atoms[dst], pair_abs_diff, pair_product, mol_context, dist_feat, support_feat, bond_feat],
            dim=-1,
        )

        pair_score, demand = self.pair_score_from_features(z)
        base_logits = torch.log(c)
        residual_score, all_alpha_base, demand_alpha = self.compute_residual_scores(
            pair_score=pair_score,
            support=c,
            base_logits=base_logits,
            src=src,
            num_nodes=n,
        )
        effective_residual_score = self.center_residual_scores(residual_score, src=src, num_nodes=n)
        rewiring_delta = self.coremol.beta * torch.tanh(effective_residual_score / self.coremol.tau)
        cal_logits = base_logits + rewiring_delta

        values = self.value_proj(atoms)
        all_alpha_cal = softmax(cal_logits, src, num_nodes=n)
        diff = all_alpha_cal - all_alpha_base
        message_values = values[dst]
        if self.coremol.residual_message == "delta":
            message_values = values[dst] - values[src]
        residual_updates = scatter(diff.unsqueeze(-1) * message_values, src, dim=0, dim_size=n, reduce="sum")
        normalized_updates = self.normalize_updates(residual_updates)
        dropped_updates = F.dropout(normalized_updates, p=self.coremol.dropout, training=self.training)

        effective_gate = self.effective_residual_gate()
        gated_updates = effective_gate * dropped_updates
        next_atoms = atoms + gated_updates
        update_norm = (effective_gate.detach() * normalized_updates.detach()).norm(dim=-1).mean()
        atom_norm = atoms.detach().norm(dim=-1).mean().clamp_min(1e-8)
        diagnostics = []
        if return_diagnostics:
            diag = {
                "pairs": pairs.detach(),
                "support": c,
                "c_ref": c,
                "residual_score": effective_residual_score,
                "residual_score_raw": residual_score,
                "alpha_base": all_alpha_base,
                "alpha_ref": all_alpha_base,
                "alpha_cal": all_alpha_cal,
                "alpha_rew": all_alpha_cal,
                "rewiring_delta": rewiring_delta,
                "residual_score_space": self.coremol.residual_score_space,
                "residual_shift_centering": self.coremol.residual_shift_centering,
                "update_norm_mean": update_norm,
                "atom_norm_mean": atom_norm,
                "update_atom_norm_ratio": update_norm / atom_norm,
                "residual_gate": effective_gate.detach().clone(),
            }
            if self.coremol.residual_score_space == "rir":
                diag["shift_raw"] = pair_score
            else:
                diag["demand"] = demand
                diag["demand_alpha"] = demand_alpha if demand_alpha is not None else demand
            diagnostics = [diag]
        return next_atoms, diagnostics

    def _graph_structure(self, edge_index, num_nodes: int):
        key = self._structure_cache_key(edge_index, num_nodes)
        cached = self._structure_cache.get(key)
        if cached is not None:
            self._structure_cache.move_to_end(key)
            pairs, distances, support = cached
            device = edge_index.device
            return pairs.to(device), distances.to(device), support.to(device)

        pairs, distances = candidate_pairs_with_distances(
            edge_index=edge_index,
            num_nodes=num_nodes,
            d_max=self.coremol.d_max,
            include_bond_pairs=self.coremol.include_bond_pairs,
        )
        support = finite_hop_support(edge_index, num_nodes, max_hops=self.coremol.support_hops)
        self._structure_cache[key] = (pairs.detach().cpu(), distances.detach().cpu(), support.detach().cpu())
        if len(self._structure_cache) > self._structure_cache_limit:
            self._structure_cache.popitem(last=False)
        return pairs, distances, support

    def _structure_cache_key(self, edge_index, num_nodes: int):
        edge_cpu = edge_index.detach().to("cpu", non_blocking=False).contiguous()
        return (
            int(num_nodes),
            int(self.coremol.d_max),
            int(self.coremol.support_hops),
            bool(self.coremol.include_bond_pairs),
            edge_cpu.numpy().tobytes(),
        )
