import torch
from torch import nn
from torch_geometric.utils import scatter


class InterfaceFlowReadout(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.hidden_channels = int(hidden_channels)

    def forward(
        self,
        interface_flow: torch.Tensor,
        delta_cross: torch.Tensor,
        alpha_ref: torch.Tensor,
        alpha_rew: torch.Tensor,
        cross_batch: torch.Tensor,
        num_graphs: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if interface_flow.numel() == 0:
            repr_ = interface_flow.new_zeros(num_graphs, self.hidden_channels * 2 + 4)
            stats = {
                "mean_abs_delta": interface_flow.new_zeros(num_graphs),
                "mean_alpha_entropy": interface_flow.new_zeros(num_graphs),
                "positive_flow_norm": interface_flow.new_zeros(num_graphs),
                "negative_flow_norm": interface_flow.new_zeros(num_graphs),
            }
            return repr_, stats

        signed_flow = scatter(interface_flow, cross_batch, dim=0, dim_size=num_graphs, reduce="sum")
        abs_flow = scatter(interface_flow.abs(), cross_batch, dim=0, dim_size=num_graphs, reduce="mean")
        mean_abs_delta = scatter(delta_cross.abs(), cross_batch, dim=0, dim_size=num_graphs, reduce="mean")
        entropy_per_pair = -(alpha_rew.clamp_min(1e-8) * alpha_rew.clamp_min(1e-8).log())
        mean_alpha_entropy = scatter(entropy_per_pair, cross_batch, dim=0, dim_size=num_graphs, reduce="mean")
        positive_flow_norm = scatter(interface_flow.clamp_min(0.0).norm(dim=-1), cross_batch, dim=0, dim_size=num_graphs, reduce="sum")
        negative_flow_norm = scatter((-interface_flow).clamp_min(0.0).norm(dim=-1), cross_batch, dim=0, dim_size=num_graphs, reduce="sum")
        scalar_repr = torch.stack(
            [mean_abs_delta, mean_alpha_entropy, positive_flow_norm, negative_flow_norm],
            dim=-1,
        )
        stats = {
            "mean_abs_delta": mean_abs_delta,
            "mean_alpha_entropy": mean_alpha_entropy,
            "positive_flow_norm": positive_flow_norm,
            "negative_flow_norm": negative_flow_norm,
        }
        return torch.cat([signed_flow, abs_flow, scalar_repr], dim=-1), stats
