from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.utils import to_dense_batch

from coremol.modules.coremol_adapter import CoReMolConfig, CoReMolResidualAdapter
from coremol.modules.pair_features import candidate_pairs_with_distances


def convert_to_single_emb_ids(features: torch.Tensor, offset: int = 512) -> torch.Tensor:
    if features.dim() == 1:
        features = features.unsqueeze(-1)
    feature_offset = 1 + torch.arange(
        0,
        features.size(-1) * offset,
        offset,
        dtype=torch.long,
        device=features.device,
    )
    return features.long().clamp_min(0) + feature_offset


class GraphformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        dropout: float,
        max_distance: int,
        ffn_ratio: int = 4,
        norm_style: str = "pre",
        use_distance_bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.norm_style = norm_style
        self.use_distance_bias = use_distance_bias
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.distance_bias = nn.Embedding(max_distance + 2, num_heads)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * ffn_ratio, hidden_channels),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        atoms: torch.Tensor,
        distance_ids: torch.Tensor | None,
        key_padding_mask=None,
        edge_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bias_terms = []
        if self.use_distance_bias and distance_ids is not None:
            distance_bias = self.distance_bias(distance_ids.clamp(0, self.max_distance + 1))
            bias_terms.append(distance_bias.permute(0, 3, 1, 2))
        if edge_bias is not None:
            bias_terms.append(edge_bias)
        attn_bias = None
        if bias_terms:
            attn_bias = torch.stack(bias_terms, dim=0).sum(dim=0)
            attn_bias = attn_bias.reshape(-1, atoms.size(1), atoms.size(1)).contiguous()

        h = self.norm1(atoms) if self.norm_style == "pre" else atoms
        padding_bias = None
        if key_padding_mask is not None:
            padding_bias = torch.zeros_like(key_padding_mask, dtype=atoms.dtype)
            padding_bias = padding_bias.masked_fill(key_padding_mask, float("-inf"))
        attended, _ = self.attn(h, h, h, attn_mask=attn_bias, key_padding_mask=padding_bias)
        atoms = atoms + self.dropout(attended)
        if self.norm_style == "post":
            atoms = self.norm1(atoms)
        ffn_input = self.norm2(atoms) if self.norm_style == "pre" else atoms
        atoms = atoms + self.dropout(self.ffn(ffn_input))
        if self.norm_style == "post":
            atoms = self.norm2(atoms)
        if key_padding_mask is not None:
            atoms = atoms.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return atoms


class GraphformerBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        num_heads: int = 4,
        max_distance: int = 5,
        use_graph_token: bool = False,
        readout: str = "mean_max",
        use_local_gnn: bool = True,
        use_distance_bias: bool = True,
        use_edge_bias: bool = False,
        use_degree_encoding: bool = True,
        ffn_ratio: int = 4,
        norm_style: str = "pre",
        feature_encoder: str = "linear",
    ):
        super().__init__()
        if hidden_channels % num_heads != 0:
            raise ValueError("hidden_channels must be divisible by num_heads")
        if readout not in {"mean", "mean_max", "graph_token"}:
            raise ValueError("readout must be one of: mean, mean_max, graph_token")
        if norm_style not in {"pre", "post"}:
            raise ValueError("norm_style must be one of: pre, post")
        if feature_encoder not in {"linear", "categorical"}:
            raise ValueError("feature_encoder must be one of: linear, categorical")
        self.hidden_channels = hidden_channels
        self.max_distance = max_distance
        self.num_heads = num_heads
        self.feature_encoder = feature_encoder
        self.use_graph_token = use_graph_token
        self.readout = readout
        self.use_local_gnn = use_local_gnn
        self.use_distance_bias = use_distance_bias
        self.use_edge_bias = use_edge_bias
        self.use_degree_encoding = use_degree_encoding
        self.node_proj = nn.Linear(in_channels, hidden_channels)
        self.edge_proj = nn.Linear(edge_dim, hidden_channels)
        self.edge_bias_proj = nn.Linear(edge_dim, num_heads, bias=False)
        self.atom_encoder = nn.Embedding((in_channels + 1) * 512, hidden_channels, padding_idx=0)
        self.edge_encoder = nn.Embedding((edge_dim + 1) * 512, hidden_channels, padding_idx=0)
        self.edge_bias_encoder = nn.Embedding((edge_dim + 1) * 512, num_heads, padding_idx=0)
        self.degree_encoder = nn.Embedding(16, hidden_channels)
        self.graph_token = nn.Parameter(torch.zeros(1, 1, hidden_channels))
        self.local_convs = nn.ModuleList(
            [
                GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_channels, hidden_channels),
                        nn.ReLU(),
                        nn.Linear(hidden_channels, hidden_channels),
                    ),
                    edge_dim=hidden_channels,
                )
                for _ in range(max(1, num_layers // 2))
            ]
        )
        self.layers = nn.ModuleList(
            [
                GraphformerEncoderLayer(
                    hidden_channels=hidden_channels,
                    num_heads=num_heads,
                    dropout=dropout,
                    max_distance=max_distance,
                    ffn_ratio=ffn_ratio,
                    norm_style=norm_style,
                    use_distance_bias=use_distance_bias,
                )
                for _ in range(num_layers)
            ]
        )
        self.graph_norm = nn.LayerNorm(hidden_channels)
        readout_dim = hidden_channels * 2 if readout == "mean_max" else hidden_channels
        self.graph_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(readout_dim, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )
        self._last_graph_token: torch.Tensor | None = None
        self._distance_cache: OrderedDict[tuple[int, int, bytes], torch.Tensor] = OrderedDict()
        self._distance_cache_size = 50000

    def encode_batch(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        layerwise_callback=None,
    ) -> torch.Tensor:
        degree = torch.bincount(edge_index[0], minlength=x.size(0)).clamp(max=15)
        atoms = self.node_features(x)
        if self.use_degree_encoding:
            atoms = atoms + self.degree_encoder(degree)
        edge_features = self.edge_features(edge_attr)
        if self.use_local_gnn:
            for conv in self.local_convs:
                atoms = atoms + F.relu(conv(atoms, edge_index, edge_features))
        dense_atoms, node_mask = to_dense_batch(atoms, batch)
        edge_bias = self.batch_edge_bias(edge_index=edge_index, edge_attr=edge_attr, batch=batch, max_nodes=dense_atoms.size(1), device=x.device)
        distance_ids = self.batch_distance_ids(edge_index=edge_index, batch=batch, max_nodes=dense_atoms.size(1), device=x.device)
        if self.use_graph_token:
            graph_tokens = self.graph_token.expand(dense_atoms.size(0), -1, -1)
            dense_atoms = torch.cat([graph_tokens, dense_atoms], dim=1)
            token_mask = torch.ones(node_mask.size(0), 1, dtype=torch.bool, device=node_mask.device)
            node_mask_with_token = torch.cat([token_mask, node_mask], dim=1)
            distance_ids = self._with_graph_token_matrix(distance_ids, fill_value=0)
            edge_bias = self._with_graph_token_bias(edge_bias)
        else:
            node_mask_with_token = node_mask
        key_padding_mask = ~node_mask
        if self.use_graph_token:
            key_padding_mask = ~node_mask_with_token
        for layer_idx, layer in enumerate(self.layers):
            dense_atoms = layer(dense_atoms, distance_ids, key_padding_mask=key_padding_mask, edge_bias=edge_bias)
            if layerwise_callback is not None:
                if self.use_graph_token:
                    graph_token = dense_atoms[:, :1, :]
                    dense_node_atoms = dense_atoms[:, 1:, :]
                else:
                    graph_token = None
                    dense_node_atoms = dense_atoms
                sparse_atoms = dense_node_atoms[node_mask]
                sparse_atoms = layerwise_callback(sparse_atoms, layer_idx)
                dense_node_atoms, _ = to_dense_batch(sparse_atoms, batch, max_num_nodes=node_mask.size(1))
                dense_node_atoms = dense_node_atoms.masked_fill(~node_mask.unsqueeze(-1), 0.0)
                if graph_token is not None:
                    dense_atoms = torch.cat([graph_token, dense_node_atoms], dim=1)
                else:
                    dense_atoms = dense_node_atoms
        if self.use_graph_token:
            self._last_graph_token = dense_atoms[:, 0, :]
            return dense_atoms[:, 1:, :][node_mask]
        self._last_graph_token = None
        return dense_atoms[node_mask]

    def node_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_encoder == "categorical":
            return self.atom_encoder(convert_to_single_emb_ids(x)).sum(dim=-2)
        return self.node_proj(x.float())

    def edge_features(self, edge_attr: torch.Tensor) -> torch.Tensor:
        if self.feature_encoder == "categorical":
            return self.edge_encoder(convert_to_single_emb_ids(edge_attr)).sum(dim=-2)
        return self.edge_proj(edge_attr.float())

    def edge_bias_features(self, edge_attr: torch.Tensor) -> torch.Tensor:
        if self.feature_encoder == "categorical":
            return self.edge_bias_encoder(convert_to_single_emb_ids(edge_attr)).sum(dim=-2)
        return self.edge_bias_proj(edge_attr.float())

    def encode_graph(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        edge_attr = torch.zeros(edge_index.size(1), self.edge_proj.in_features, dtype=x.dtype, device=x.device)
        return self.encode_batch(x, edge_index, edge_attr, batch)

    def batch_distance_ids(self, edge_index: torch.Tensor, batch: torch.Tensor, max_nodes: int, device) -> torch.Tensor:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        distances = torch.full(
            (num_graphs, max_nodes, max_nodes),
            fill_value=self.max_distance + 1,
            dtype=torch.long,
            device=device,
        )
        for graph_id in range(num_graphs):
            node_idx = (batch == graph_id).nonzero(as_tuple=False).reshape(-1)
            local_n = node_idx.numel()
            if local_n == 0:
                continue
            distances[graph_id, torch.arange(local_n, device=device), torch.arange(local_n, device=device)] = 0
            edge_mask = (batch[edge_index[0]] == graph_id) & (batch[edge_index[1]] == graph_id)
            local_edge_index = edge_index[:, edge_mask] - node_idx[0]
            local_distances = self.distance_ids(local_edge_index, int(local_n), device)
            distances[graph_id, :local_n, :local_n] = local_distances
        return distances

    def distance_ids(self, edge_index: torch.Tensor, num_nodes: int, device) -> torch.Tensor:
        cache_key = self._distance_cache_key(edge_index, num_nodes)
        cached = self._distance_cache.get(cache_key)
        if cached is not None:
            self._distance_cache.move_to_end(cache_key)
            return cached.to(device=device, non_blocking=True)

        distances = torch.full(
            (num_nodes, num_nodes),
            fill_value=self.max_distance + 1,
            dtype=torch.long,
            device=device,
        )
        distances.fill_diagonal_(0)
        pairs, pair_distances = candidate_pairs_with_distances(
            edge_index=edge_index,
            num_nodes=num_nodes,
            d_max=self.max_distance,
            include_bond_pairs=True,
        )
        if pairs.numel() > 0:
            pairs = pairs.to(device)
            distances[pairs[:, 0], pairs[:, 1]] = pair_distances.to(device).long().clamp(1, self.max_distance)
        self._distance_cache[cache_key] = distances.detach().cpu()
        if len(self._distance_cache) > self._distance_cache_size:
            self._distance_cache.popitem(last=False)
        return distances

    def batch_edge_bias(self, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor, max_nodes: int, device) -> torch.Tensor | None:
        if not self.use_edge_bias:
            return None
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        edge_bias = torch.zeros(num_graphs, self.num_heads, max_nodes, max_nodes, dtype=torch.float, device=device)
        if edge_index.numel() == 0:
            return edge_bias
        projected = self.edge_bias_features(edge_attr).to(device)
        for graph_id in range(num_graphs):
            node_idx = (batch == graph_id).nonzero(as_tuple=False).reshape(-1)
            if node_idx.numel() == 0:
                continue
            edge_mask = (batch[edge_index[0]] == graph_id) & (batch[edge_index[1]] == graph_id)
            local_edges = edge_index[:, edge_mask] - node_idx[0]
            edge_bias[graph_id, :, local_edges[0], local_edges[1]] = projected[edge_mask].transpose(0, 1)
        return edge_bias

    def _with_graph_token_matrix(self, matrix: torch.Tensor, fill_value: int) -> torch.Tensor:
        expanded = torch.full(
            (matrix.size(0), matrix.size(1) + 1, matrix.size(2) + 1),
            fill_value=fill_value,
            dtype=matrix.dtype,
            device=matrix.device,
        )
        expanded[:, 1:, 1:] = matrix
        return expanded

    def _with_graph_token_bias(self, edge_bias: torch.Tensor | None) -> torch.Tensor | None:
        if edge_bias is None:
            return None
        expanded = edge_bias.new_zeros(edge_bias.size(0), edge_bias.size(1), edge_bias.size(2) + 1, edge_bias.size(3) + 1)
        expanded[:, :, 1:, 1:] = edge_bias
        return expanded

    def _distance_cache_key(self, edge_index: torch.Tensor, num_nodes: int) -> tuple[int, int, bytes]:
        edge_bytes = edge_index.detach().to(device="cpu", dtype=torch.long).contiguous().numpy().tobytes()
        return num_nodes, self.max_distance, edge_bytes

    def forward(self, atoms: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        mean_repr = global_mean_pool(atoms, batch)
        if self.readout == "mean":
            graph_repr = self.graph_norm(mean_repr)
        elif self.readout == "graph_token":
            if self._last_graph_token is not None and self._last_graph_token.size(0) == mean_repr.size(0):
                graph_repr = self.graph_norm(self._last_graph_token.to(mean_repr.device) + mean_repr)
            else:
                graph_repr = self.graph_norm(mean_repr)
        else:
            graph_repr = self.graph_norm(mean_repr)
            dense_atoms, node_mask = to_dense_batch(atoms, batch)
            max_repr = dense_atoms.masked_fill(~node_mask.unsqueeze(-1), -torch.inf).max(dim=1).values
            max_repr = torch.nan_to_num(max_repr, neginf=0.0)
            graph_repr = torch.cat([graph_repr, max_repr], dim=-1)
        return self.graph_head(graph_repr)


class CoReMolGraphformer(nn.Module):
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
        num_heads: int = 4,
        max_distance: int = 5,
        use_graph_token: bool = False,
        readout: str = "mean_max",
        use_local_gnn: bool = True,
        use_distance_bias: bool = True,
        use_edge_bias: bool = False,
        use_degree_encoding: bool = True,
        ffn_ratio: int = 4,
        norm_style: str = "pre",
        feature_encoder: str = "linear",
    ):
        super().__init__()
        del num_timesteps
        self.backbone = GraphformerBackbone(
            in_channels=in_channels,
            edge_dim=edge_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            num_heads=num_heads,
            max_distance=max_distance,
            use_graph_token=use_graph_token,
            readout=readout,
            use_local_gnn=use_local_gnn,
            use_distance_bias=use_distance_bias,
            use_edge_bias=use_edge_bias,
            use_degree_encoding=use_degree_encoding,
            ffn_ratio=ffn_ratio,
            norm_style=norm_style,
            feature_encoder=feature_encoder,
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
            atoms, step_diagnostics = self._call_apply_coremol(
                atoms, edge_index, batch, collect_diagnostics=collect_diagnostics
            )
            if collect_diagnostics:
                for item in step_diagnostics:
                    tagged = dict(item)
                    tagged["residual_stage"] = stage
                    tagged["residual_step"] = step
                    diagnostics.append(tagged)
        return atoms, diagnostics

    def _call_apply_coremol(self, atoms, edge_index, batch, collect_diagnostics: bool):
        try:
            return self.apply_coremol(atoms, edge_index, batch, return_diagnostics=collect_diagnostics)
        except TypeError as exc:
            if "return_diagnostics" not in str(exc):
                raise
            return self.apply_coremol(atoms, edge_index, batch)

    def encode_atoms(self, x, edge_index, edge_attr=None, batch=None, return_diagnostics: bool = False):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        diagnostics = []
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), self.backbone.edge_proj.in_features, dtype=x.dtype, device=x.device)
        layerwise_callback = None
        if self._uses_layerwise_residual():
            def layerwise_callback(atoms, layer_idx):
                atoms, step_diagnostics = self._apply_coremol_steps(
                    atoms,
                    edge_index,
                    batch,
                    stage=f"encoder_layer_{layer_idx}",
                    collect_diagnostics=return_diagnostics,
                )
                diagnostics.extend(step_diagnostics)
                return atoms

        encoded = self.backbone.encode_batch(x, edge_index, edge_attr, batch, layerwise_callback=layerwise_callback)
        if return_diagnostics:
            return encoded, diagnostics
        return encoded

    def molecule_readout(self, atoms, batch):
        out = self.backbone(atoms, batch)
        return out, global_mean_pool(atoms, batch)

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
                atoms, data.edge_index, data.batch, stage="post", collect_diagnostics=return_diagnostics
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
