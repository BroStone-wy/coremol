import torch
import pytest
import math

from coremol.metrics.mechanism import mechanism_summary
from coremol.metrics.tcm import normalized_tcm_score, tcm_at_k


def test_tcm_at_k_rewards_useful_coverage_and_penalizes_harmful_leakage():
    useful_scores = torch.tensor([0.9, 0.2, -0.1, -0.8])
    q_good = torch.tensor([0.7, 0.2, 0.05, 0.05])
    q_bad = torch.tensor([0.05, 0.05, 0.2, 0.7])

    good = tcm_at_k(q_good, useful_scores, k=1)
    bad = tcm_at_k(q_bad, useful_scores, k=1)

    assert good < bad


def test_normalized_tcm_score_uses_normalized_benefit_and_harm():
    q = torch.tensor([0.6, 0.2, 0.2])
    sensitivity = torch.tensor([2.0, 1.0, -3.0])

    score = normalized_tcm_score(q, sensitivity, lambda_harm=1.0)

    expected_benefit = (0.6 * 2.0 + 0.2 * 1.0) / 3.0
    expected_harm = (0.2 * 3.0) / 3.0
    assert score == pytest.approx(expected_benefit - expected_harm)


def test_normalized_tcm_score_is_invariant_to_probe_sensitivity_scale():
    q = torch.tensor([0.6, 0.2, 0.2])
    sensitivity = torch.tensor([2.0, 1.0, -3.0])

    score = normalized_tcm_score(q, sensitivity)
    scaled = normalized_tcm_score(q, sensitivity * 10.0)

    assert score == pytest.approx(float(scaled))


def test_mechanism_summary_includes_residual_update_scale_when_available():
    summary = mechanism_summary(
        [
            {
                "residual_score": torch.tensor([0.2, -0.1]),
                "alpha_base": torch.tensor([0.4, 0.6]),
                "alpha_cal": torch.tensor([0.5, 0.5]),
                "demand": torch.tensor([0.7, 0.2]),
                "update_atom_norm_ratio": torch.tensor(0.3),
                "residual_gate": torch.tensor(0.1),
            }
        ]
    )

    assert summary["update_atom_norm_ratio"] == pytest.approx(0.3)
    assert summary["residual_gate"] == pytest.approx(0.1)


def test_mechanism_summary_uses_distribution_scale_demand_when_available():
    summary = mechanism_summary(
        [
            {
                "residual_score": torch.tensor([0.4, -0.4]),
                "alpha_base": torch.tensor([0.9, 0.1]),
                "alpha_cal": torch.tensor([0.5, 0.5]),
                "demand": torch.tensor([0.99, 0.01]),
                "demand_alpha": torch.tensor([0.5, 0.5]),
            }
        ]
    )

    assert summary["mismatch_reduction"] == pytest.approx(1.0)


def test_mechanism_summary_skips_demand_mismatch_for_rir():
    summary = mechanism_summary(
        [
            {
                "residual_score": torch.tensor([0.4, -0.4]),
                "alpha_ref": torch.tensor([0.2, 0.8]),
                "alpha_rew": torch.tensor([0.6, 0.4]),
                "residual_score_space": "rir",
            }
        ]
    )

    assert summary["enhance_ratio"] == pytest.approx(1.0)
    assert summary["suppress_ratio"] == pytest.approx(1.0)
    assert math.isnan(summary["mismatch_reduction"])
