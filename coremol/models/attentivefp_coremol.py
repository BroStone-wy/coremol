import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.models import AttentiveFP

from coremol.modules.coremol_adapter import CoReMolConfig, CoReMolResidualAdapter


class CoReMolAttentiveFP(nn.Module):
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
    ):
        super().__init__()
        self.backbone = AttentiveFP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
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

    def encode_atoms(self, x, edge_index, edge_attr, batch=None, return_diagnostics: bool = False):
        model = self.backbone
        diagnostics = []
        x = F.leaky_relu(model.lin1(x))
        h = F.elu(model.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=model.dropout, training=self.training)
        x = model.gru(h, x).relu()
        if batch is not None and self._uses_layerwise_residual():
            x, step_diagnostics = self._apply_coremol_steps(
                x, edge_index, batch, stage="input", collect_diagnostics=return_diagnostics
            )
            diagnostics.extend(step_diagnostics)

        for layer_idx, (conv, gru) in enumerate(zip(model.atom_convs, model.atom_grus)):
            h = F.elu(conv(x, edge_index))
            h = F.dropout(h, p=model.dropout, training=self.training)
            x = gru(h, x).relu()
            if batch is not None and self._uses_layerwise_residual():
                x, step_diagnostics = self._apply_coremol_steps(
                    x, edge_index, batch, stage=f"atom_layer_{layer_idx}", collect_diagnostics=return_diagnostics
                )
                diagnostics.extend(step_diagnostics)
        if return_diagnostics:
            return x, diagnostics
        return x

    def molecule_readout(self, atoms, batch):
        model = self.backbone
        row = torch.arange(batch.size(0), device=batch.device)
        mol_edge_index = torch.stack([row, batch], dim=0)
        out = global_add_pool(atoms, batch).relu()
        for _ in range(model.num_timesteps):
            h = F.elu(model.mol_conv((atoms, out), mol_edge_index))
            h = F.dropout(h, p=model.dropout, training=self.training)
            out = model.mol_gru(h, out).relu()
        out = F.dropout(out, p=model.dropout, training=self.training)
        return model.lin2(out), out

    def forward(self, data, return_diagnostics: bool = False):
        encoded = self.encode_atoms(
            data.x.float(),
            data.edge_index,
            data.edge_attr.float(),
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
