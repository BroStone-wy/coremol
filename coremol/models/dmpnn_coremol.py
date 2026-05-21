import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import scatter, to_dense_batch

from coremol.modules.coremol_adapter import CoReMolConfig, CoReMolResidualAdapter


class DMPNNBackbone(nn.Module):
    """Directed bond message passing backbone with atom-state readout."""

    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        readout: str = "mean",
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if readout not in {"mean", "mean_max"}:
            raise ValueError("readout must be one of: mean, mean_max")
        self.hidden_channels = hidden_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.readout = readout

        self.atom_proj = nn.Linear(in_channels, hidden_channels)
        self.edge_proj = nn.Linear(edge_dim, hidden_channels)
        self.edge_init = nn.Linear(hidden_channels * 2, hidden_channels)
        self.message_update = nn.Linear(hidden_channels, hidden_channels)
        self.atom_update = nn.Linear(hidden_channels * 2, hidden_channels)
        readout_dim = hidden_channels * 2 if readout == "mean_max" else hidden_channels
        self.graph_norm = nn.LayerNorm(readout_dim)
        self.graph_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(readout_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def encode_atoms(self, x, edge_index, edge_attr, batch=None, layerwise_callback=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        atom_base = self.atom_proj(x.float())
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), self.edge_dim, dtype=x.dtype, device=x.device)
        if edge_index.numel() == 0:
            atoms = F.relu(atom_base)
            if layerwise_callback is not None:
                atoms = layerwise_callback(atoms, 0)
            return atoms

        src, dst = edge_index[0], edge_index[1]
        edge_base = self.edge_proj(edge_attr.float())
        initial_messages = F.relu(self.edge_init(torch.cat([atom_base[src], edge_base], dim=-1)))
        messages = initial_messages
        reverse_index = self._reverse_edge_index(edge_index)

        for layer_idx in range(self.num_layers):
            incoming_to_src = scatter(messages, dst, dim=0, dim_size=x.size(0), reduce="sum")[src]
            has_reverse = reverse_index >= 0
            reverse_messages = torch.zeros_like(messages)
            if bool(has_reverse.any()):
                reverse_messages[has_reverse] = messages[reverse_index[has_reverse]]
            directed_context = incoming_to_src - reverse_messages
            messages = F.relu(initial_messages + self.message_update(directed_context))
            messages = F.dropout(messages, p=self.dropout, training=self.training)
            if layerwise_callback is not None:
                atoms = self.messages_to_atoms(atom_base, messages, edge_index)
                atoms = layerwise_callback(atoms, layer_idx)
                # Project corrected atom states back to directed-edge initial messages.
                initial_messages = F.relu(self.edge_init(torch.cat([atoms[src], edge_base], dim=-1)))
                messages = initial_messages

        return self.messages_to_atoms(atom_base, messages, edge_index)

    def messages_to_atoms(self, atom_base, messages, edge_index):
        incoming = scatter(messages, edge_index[1], dim=0, dim_size=atom_base.size(0), reduce="sum")
        atoms = F.relu(self.atom_update(torch.cat([atom_base, incoming], dim=-1)))
        return F.dropout(atoms, p=self.dropout, training=self.training)

    @staticmethod
    def _reverse_edge_index(edge_index):
        mapping = {
            (int(src), int(dst)): idx
            for idx, (src, dst) in enumerate(zip(edge_index[0].detach().cpu().tolist(), edge_index[1].detach().cpu().tolist()))
        }
        reverse = [
            mapping.get((int(dst), int(src)), -1)
            for src, dst in zip(edge_index[0].detach().cpu().tolist(), edge_index[1].detach().cpu().tolist())
        ]
        return torch.tensor(reverse, dtype=torch.long, device=edge_index.device)

    def forward(self, atoms, batch):
        mean_repr = global_mean_pool(atoms, batch)
        if self.readout == "mean":
            graph_repr = mean_repr
        else:
            dense_atoms, node_mask = to_dense_batch(atoms, batch)
            max_repr = dense_atoms.masked_fill(~node_mask.unsqueeze(-1), -torch.inf).max(dim=1).values
            max_repr = torch.nan_to_num(max_repr, neginf=0.0)
            graph_repr = torch.cat([mean_repr, max_repr], dim=-1)
        return self.graph_head(self.graph_norm(graph_repr))


class CoReMolDMPNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float,
        coremol: CoReMolConfig,
        readout: str = "mean",
    ):
        super().__init__()
        del num_timesteps
        self.backbone = DMPNNBackbone(
            in_channels=in_channels,
            edge_dim=edge_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            readout=readout,
        )
        self.coremol = coremol
        self.adapter = CoReMolResidualAdapter(hidden_channels=hidden_channels, coremol=coremol)

    def _uses_layerwise_residual(self) -> bool:
        return self.coremol.enabled and self.coremol.residual_placement in {"layerwise", "both"}

    def _uses_post_residual(self) -> bool:
        return self.coremol.enabled and self.coremol.residual_placement in {"post", "both"}

    def _apply_coremol_steps(self, atoms, edge_index, batch, stage, collect_diagnostics: bool):
        diagnostics = []
        for step in range(self.coremol.num_residual_steps):
            atoms, step_diagnostics = self.apply_coremol(
                atoms,
                edge_index,
                batch,
                return_diagnostics=collect_diagnostics,
            )
            if collect_diagnostics:
                for item in step_diagnostics:
                    tagged = dict(item)
                    tagged["residual_stage"] = stage
                    tagged["residual_step"] = step
                    diagnostics.append(tagged)
        return atoms, diagnostics

    def encode_atoms(self, x, edge_index, edge_attr=None, batch=None, return_diagnostics: bool = False):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        diagnostics = []
        layerwise_callback = None
        if self._uses_layerwise_residual():
            def layerwise_callback(atoms, layer_idx):
                atoms, step_diagnostics = self._apply_coremol_steps(
                    atoms,
                    edge_index,
                    batch,
                    stage=f"message_layer_{layer_idx}",
                    collect_diagnostics=return_diagnostics,
                )
                diagnostics.extend(step_diagnostics)
                return atoms

        encoded = self.backbone.encode_atoms(
            x=x.float(),
            edge_index=edge_index,
            edge_attr=edge_attr.float() if edge_attr is not None else None,
            batch=batch,
            layerwise_callback=layerwise_callback,
        )
        if return_diagnostics:
            return encoded, diagnostics
        return encoded

    def molecule_readout(self, atoms, batch):
        return self.backbone(atoms, batch), global_mean_pool(atoms, batch)

    def forward(self, data, return_diagnostics: bool = False):
        encoded = self.encode_atoms(
            data.x.float(),
            data.edge_index,
            data.edge_attr.float() if getattr(data, "edge_attr", None) is not None else None,
            batch=data.batch,
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            atoms, diagnostics = encoded
        else:
            atoms, diagnostics = encoded, []
        if self._uses_post_residual():
            atoms, post_diagnostics = self._apply_coremol_steps(
                atoms,
                data.edge_index,
                data.batch,
                stage="post",
                collect_diagnostics=return_diagnostics,
            )
            diagnostics.extend(post_diagnostics)
        out, _ = self.molecule_readout(atoms, data.batch)
        if return_diagnostics:
            return out, diagnostics
        return out

    def apply_coremol(self, atoms, edge_index, batch, return_diagnostics: bool = True):
        return self.adapter(atoms, edge_index, batch, return_diagnostics=return_diagnostics)

    def _apply_single_graph(self, atoms, edge_index):
        return self.adapter.apply_single_graph(atoms, edge_index)
