from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, softmax

from coremol.modules.interface_reference_graph import InterfaceReferenceGraphConstructor


@dataclass
class CrossRIRConfig:
    enabled: bool = True
    beta: float = 0.5
    tau: float = 0.5
    gate_init: float = 0.05
    gate_max: float = 0.5
    dropout: float = 0.1
    cutoff: float = 6.0
    topk_per_lig_atom: int = 16
    sigma: float = 2.0


class CrossRIRAdapter(nn.Module):
    def __init__(self, hidden_channels: int, config: CrossRIRConfig | None = None):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.config = config or CrossRIRConfig()
        pair_dim = hidden_channels * 4 + 6
        self.reference_graph = InterfaceReferenceGraphConstructor(
            cutoff=self.config.cutoff,
            topk_per_lig_atom=self.config.topk_per_lig_atom,
            sigma=self.config.sigma,
        )
        self.shift_mlp = nn.Sequential(
            nn.Linear(pair_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden_channels, 1),
        )
        self.lig_value = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.pocket_value = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.norm = nn.LayerNorm(hidden_channels)
        self.gate = nn.Parameter(torch.tensor(float(self.config.gate_init)))

    def effective_gate(self) -> torch.Tensor:
        if self.config.gate_max <= 0:
            return self.gate
        return self.config.gate_max * torch.tanh(self.gate / self.config.gate_max)

    def forward(self, hidden: torch.Tensor, data, return_diagnostics: bool = False):
        if not self.config.enabled:
            return (hidden, {}) if return_diagnostics else hidden

        ref = self.reference_graph(data)
        if ref.cross_pair_index.numel() == 0:
            empty_diag = {
                "cross_pair_index": ref.cross_pair_index,
                "c_ref_cross": ref.c_ref_cross,
                "cross_batch": ref.cross_batch,
                "alpha_ref": ref.c_ref_cross,
                "alpha_rew": ref.c_ref_cross,
                "delta_cross": ref.c_ref_cross,
                "interface_flow": hidden.new_empty(0, self.hidden_channels),
            }
            return (hidden, empty_diag) if return_diagnostics else hidden

        lig, pocket = ref.cross_pair_index[0], ref.cross_pair_index[1]
        z = torch.cat(
            [
                hidden[lig],
                hidden[pocket],
                (hidden[lig] - hidden[pocket]).abs(),
                hidden[lig] * hidden[pocket],
                ref.cross_features.to(hidden.device),
            ],
            dim=-1,
        )
        raw_shift = self.shift_mlp(z).view(-1)
        delta_cross = self.config.beta * torch.tanh(raw_shift / max(self.config.tau, 1e-8))
        logits_ref = torch.log(ref.c_ref_cross.to(hidden.device).clamp_min(1e-8))
        alpha_ref = softmax(logits_ref, lig, num_nodes=hidden.size(0))
        alpha_rew = softmax(logits_ref + delta_cross, lig, num_nodes=hidden.size(0))

        pair_message = self.pocket_value(hidden[pocket]) - self.lig_value(hidden[lig])
        interface_flow = (alpha_rew - alpha_ref).unsqueeze(-1) * pair_message
        updates = scatter(interface_flow, lig, dim=0, dim_size=hidden.size(0), reduce="sum")
        updates = F.dropout(self.norm(updates), p=self.config.dropout, training=self.training)
        updated = hidden + self.effective_gate() * updates

        diagnostics = {
            "cross_pair_index": ref.cross_pair_index,
            "c_ref_cross": ref.c_ref_cross,
            "cross_features": ref.cross_features,
            "cross_batch": ref.cross_batch,
            "alpha_ref": alpha_ref,
            "alpha_rew": alpha_rew,
            "delta_cross": delta_cross,
            "interface_flow": interface_flow,
            "pair_message": pair_message,
            "residual_gate": self.effective_gate().detach().clone(),
        }
        return (updated, diagnostics) if return_diagnostics else updated
