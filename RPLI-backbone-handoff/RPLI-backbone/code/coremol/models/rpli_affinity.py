from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, MetaLayer, global_add_pool
from torch_geometric.utils import scatter

from coremol.modules.cross_rir_adapter import CrossRIRAdapter, CrossRIRConfig
from coremol.modules.interface_flow_readout import InterfaceFlowReadout


class FeatureTransformMLP(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class RPLIEdgeModel(nn.Module):
    def __init__(self, node_channels: int, edge_channels: int, hidden_channels: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_channels * 2 + edge_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, src, dest, edge_attr, u, batch):
        return self.net(torch.cat([src, dest, edge_attr.float()], dim=-1))


class RPLINodeModel(nn.Module):
    def __init__(self, node_channels: int, edge_channels: int, out_channels: int, heads: int, dropout: float):
        super().__init__()
        if out_channels % heads != 0:
            raise ValueError("out_channels must be divisible by heads")
        self.conv = GATv2Conv(
            node_channels,
            out_channels // heads,
            heads=heads,
            edge_dim=edge_channels,
            dropout=dropout,
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        return F.relu(self.conv(x, edge_index, edge_attr))


class RPLIContextModel(nn.Module):
    def __init__(self, node_channels: int, context_channels: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_channels + context_channels, context_channels),
            nn.ReLU(),
            nn.Linear(context_channels, context_channels),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        pooled = global_add_pool(x, batch=batch)
        return self.net(torch.cat([u, pooled], dim=-1))


@dataclass
class RPLIAffinityConfig:
    hidden_channels: int = 64
    context_channels: int = 384
    dropout: float = 0.1
    conv_dropout: float = 0.0
    heads: int = 4
    use_cross_rir: bool = True
    cross_position: str = "mid"
    cross_update: str = "residual"
    interface_gate_init: float = 1.0
    readout_mode: str = "joint"
    residual_gate_init: float = 0.05
    final_blend_alpha: float = 0.6
    cross_rir: CrossRIRConfig = field(default_factory=CrossRIRConfig)


class RPLIAffinity(nn.Module):
    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        ligand_global_dim: int,
        config: RPLIAffinityConfig | None = None,
    ):
        super().__init__()
        self.config = config or RPLIAffinityConfig()
        hidden = self.config.hidden_channels
        context = self.config.context_channels
        self.ligand_global_dim = int(ligand_global_dim)
        self.node_transform = FeatureTransformMLP(in_channels, 256, hidden, self.config.dropout)
        self.context_transform = (
            nn.Sequential(nn.Linear(self.ligand_global_dim, context), nn.ReLU())
            if self.ligand_global_dim > 0
            else None
        )
        self.context_fallback = nn.Parameter(torch.zeros(1, context))
        self.layer1 = self._build_layer(hidden, edge_dim, hidden, context)
        self.node_norm = nn.BatchNorm1d(hidden)
        self.edge_norm = nn.BatchNorm1d(hidden)
        self.context_norm = nn.LayerNorm(context)
        self.cross_rir = CrossRIRAdapter(hidden_channels=hidden, config=self.config.cross_rir)
        self.interface_readout = InterfaceFlowReadout(hidden_channels=hidden)
        self.interface_gate = nn.Parameter(torch.tensor(float(self.config.interface_gate_init)))
        self.layer2 = self._build_layer(hidden, hidden, hidden, context)
        interface_dim = hidden * 2 + 4
        self.head = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(context + hidden * 2 + interface_dim, hidden),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden, 1),
        )
        self.base_head = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(context + hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden, 1),
        )
        self.residual_head = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(context + hidden * 2 + interface_dim, hidden),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden, 1),
        )
        self.residual_gate = nn.Parameter(torch.tensor(float(self.config.residual_gate_init)))

    def _build_layer(self, node_channels: int, edge_channels: int, out_channels: int, context_channels: int) -> MetaLayer:
        return MetaLayer(
            edge_model=RPLIEdgeModel(node_channels, edge_channels, out_channels, self.config.conv_dropout),
            node_model=RPLINodeModel(
                node_channels,
                out_channels,
                out_channels,
                self.config.heads,
                self.config.conv_dropout,
            ),
            global_model=RPLIContextModel(out_channels, context_channels, self.config.dropout),
        )

    def forward(self, data, return_diagnostics: bool = False):
        x = self.node_transform(data.x)
        u, ligand_global_used = self._initial_context(data, x.device)
        x, edge_attr, u = self.layer1(x, data.edge_index, data.edge_attr.float(), u=u, batch=data.batch)
        x = self.node_norm(x)
        edge_attr = self.edge_norm(edge_attr)
        u = self.context_norm(u)

        diagnostics = {"context_carrier": u, "ligand_global_used": ligand_global_used}
        cross_diag = {}
        if self.config.use_cross_rir and self.config.cross_position == "mid":
            adapted_x, cross_diag = self.cross_rir(x, data, return_diagnostics=True)
            if self.config.cross_update == "residual":
                x = adapted_x
            diagnostics["cross_rir"] = cross_diag

        x, edge_attr, u = self.layer2(x, data.edge_index, edge_attr, u=u, batch=data.batch)
        if self.config.use_cross_rir and self.config.cross_position == "post":
            adapted_x, cross_diag = self.cross_rir(x, data, return_diagnostics=True)
            if self.config.cross_update == "residual":
                x = adapted_x
            diagnostics["cross_rir"] = cross_diag

        ligand_repr, pocket_repr = self._ligand_pocket_pool(x, data)
        if cross_diag:
            interface_repr, interface_stats = self.interface_readout(
                interface_flow=cross_diag["interface_flow"],
                delta_cross=cross_diag["delta_cross"],
                alpha_ref=cross_diag["alpha_ref"],
                alpha_rew=cross_diag["alpha_rew"],
                cross_batch=cross_diag["cross_batch"],
                num_graphs=int(data.num_graphs),
            )
        else:
            interface_repr, interface_stats = self.interface_readout(
                interface_flow=x.new_empty(0, self.config.hidden_channels),
                delta_cross=x.new_empty(0),
                alpha_ref=x.new_empty(0),
                alpha_rew=x.new_empty(0),
                cross_batch=torch.empty(0, dtype=torch.long, device=x.device),
                num_graphs=int(data.num_graphs),
            )
        interface_repr = torch.tanh(self.interface_gate) * interface_repr
        diagnostics["interface_stats"] = interface_stats
        if self.config.readout_mode == "joint":
            pred = self.head(torch.cat([u, ligand_repr, pocket_repr, interface_repr], dim=-1))
        elif self.config.readout_mode == "residual":
            base_pred = self.base_head(torch.cat([u, ligand_repr, pocket_repr], dim=-1))
            residual_pred = self.residual_head(torch.cat([u, ligand_repr, pocket_repr, interface_repr], dim=-1))
            residual_gate = torch.tanh(self.residual_gate)
            pred = base_pred + residual_gate * residual_pred
            diagnostics["base_pred"] = base_pred
            diagnostics["residual_pred"] = residual_pred
            diagnostics["residual_gate"] = residual_gate.detach().clone()
        elif self.config.readout_mode == "dual":
            base_pred = self.base_head(torch.cat([u, ligand_repr, pocket_repr], dim=-1))
            coremol_pred = self.head(torch.cat([u, ligand_repr, pocket_repr, interface_repr], dim=-1))
            alpha = float(self.config.final_blend_alpha)
            pred = alpha * base_pred + (1.0 - alpha) * coremol_pred
            diagnostics["base_pred"] = base_pred
            diagnostics["coremol_pred"] = coremol_pred
            diagnostics["final_blend_alpha"] = alpha
        else:
            raise ValueError(f"Unknown readout_mode: {self.config.readout_mode}")
        return (pred, diagnostics) if return_diagnostics else pred

    def _initial_context(self, data, device):
        if self.context_transform is None or not hasattr(data, "lig_emb") or data.lig_emb is None:
            return self.context_fallback.expand(int(data.num_graphs), -1).to(device), False
        lig_emb = torch.as_tensor(data.lig_emb, device=device).float().view(int(data.num_graphs), -1)
        if lig_emb.size(-1) != self.ligand_global_dim:
            raise ValueError(f"Expected lig_emb dim {self.ligand_global_dim}, got {lig_emb.size(-1)}")
        return self.context_transform(lig_emb), True

    def _ligand_pocket_pool(self, hidden: torch.Tensor, data) -> tuple[torch.Tensor, torch.Tensor]:
        metadata = torch.as_tensor(data.n_nodes, device=hidden.device).view(-1, 3).long()
        ligand_mask = torch.zeros(hidden.size(0), dtype=torch.bool, device=hidden.device)
        pocket_mask = torch.zeros_like(ligand_mask)
        for graph_id, (total_nodes, n_lig_nodes, _) in enumerate(metadata.tolist()):
            start = int(data.ptr[graph_id].item())
            ligand_mask[start : start + n_lig_nodes] = True
            pocket_mask[start + n_lig_nodes : start + total_nodes] = True
        ligand_repr = scatter(
            hidden[ligand_mask],
            data.batch[ligand_mask],
            dim=0,
            dim_size=int(data.num_graphs),
            reduce="mean",
        )
        pocket_repr = scatter(
            hidden[pocket_mask],
            data.batch[pocket_mask],
            dim=0,
            dim_size=int(data.num_graphs),
            reduce="mean",
        )
        return ligand_repr, pocket_repr
