import torch


def normalize_distribution(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    values = values.float().clamp_min(0.0)
    total = values.sum()
    if total <= eps:
        return torch.full_like(values, 1.0 / max(values.numel(), 1))
    return values / (total + eps)


def tcm_at_k(q: torch.Tensor, pair_sensitivity: torch.Tensor, k: int = 10, lambda_harm: float = 1.0) -> torch.Tensor:
    """Top-K misallocation: lower is better."""
    if q.numel() == 0:
        return torch.tensor(0.0, device=q.device)

    k = int(max(1, min(k, q.numel())))
    q = normalize_distribution(q)
    positive_idx = torch.topk(pair_sensitivity, k=k, largest=True).indices
    negative_idx = torch.topk(-pair_sensitivity, k=k, largest=True).indices
    useful_coverage = q[positive_idx].sum()
    harmful_leakage = q[negative_idx].sum()
    return 1.0 - useful_coverage + lambda_harm * harmful_leakage


def tcm_topk_components(q: torch.Tensor, pair_sensitivity: torch.Tensor, k: int = 10, lambda_harm: float = 1.0):
    if q.numel() == 0:
        zero = torch.tensor(0.0, device=q.device)
        return {"ucov": zero, "hleak": zero, "tcm": zero}
    k = int(max(1, min(k, q.numel())))
    q = normalize_distribution(q)
    positive_idx = torch.topk(pair_sensitivity, k=k, largest=True).indices
    negative_idx = torch.topk(-pair_sensitivity, k=k, largest=True).indices
    ucov = q[positive_idx].sum()
    hleak = q[negative_idx].sum()
    return {"ucov": ucov, "hleak": hleak, "tcm": 1.0 - ucov + lambda_harm * hleak}


def full_tcm(q: torch.Tensor, pair_sensitivity: torch.Tensor, lambda_harm: float = 1.0) -> torch.Tensor:
    if q.numel() == 0:
        return torch.tensor(0.0, device=q.device)
    q = normalize_distribution(q)
    p_pos = normalize_distribution(pair_sensitivity.clamp_min(0.0))
    p_neg = normalize_distribution((-pair_sensitivity).clamp_min(0.0))
    tv = 0.5 * (q - p_pos).abs().sum()
    harmful = (q * p_neg).sum()
    return tv + lambda_harm * harmful


def normalized_tcm_components(q: torch.Tensor, pair_sensitivity: torch.Tensor, lambda_harm: float = 1.0, eps: float = 1e-8):
    """Scale-normalized communication score; higher score is better."""
    if q.numel() == 0:
        zero = torch.tensor(0.0, device=q.device)
        return {"benefit": zero, "harm": zero, "score": zero}

    q = normalize_distribution(q, eps=eps)
    benefit_weight = pair_sensitivity.clamp_min(0.0)
    harm_weight = (-pair_sensitivity).clamp_min(0.0)
    benefit = (q * benefit_weight).sum() / benefit_weight.sum().clamp_min(eps)
    harm = (q * harm_weight).sum() / harm_weight.sum().clamp_min(eps)
    return {"benefit": benefit, "harm": harm, "score": benefit - lambda_harm * harm}


def normalized_tcm_score(q: torch.Tensor, pair_sensitivity: torch.Tensor, lambda_harm: float = 1.0) -> torch.Tensor:
    return normalized_tcm_components(q, pair_sensitivity, lambda_harm=lambda_harm)["score"]
