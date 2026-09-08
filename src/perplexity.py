"""
Definitions 2.4 (Proof Perplexity), 2.5 (theta-invention fraction), and the
Theorem 2.1 lower bound relating them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List


@dataclass
class StepResult:
    name: str
    w_si: int              # internal complexity w(si)
    w_si_B: int             # recovered complexity w(si,B)
    alpha_abl: float
    alpha_mah: float
    alpha_combined: float
    surprisal: float        # Surp(si,B)


def proof_perplexity(steps: List[StepResult]) -> float:
    """Definition 2.4: PP_w(pi,B) = exp( sum_i w_i * Surp(si,B) / sum_i w_i ),
    with w_i = w(si) (the elaborated kernel-term size)."""
    total_weight = sum(s.w_si for s in steps)
    if total_weight == 0:
        return 1.0  # exp(0): a proof with no internal complexity is maximally unsurprising
    weighted_surp = sum(s.w_si * s.surprisal for s in steps)
    return math.exp(weighted_surp / total_weight)


def theta_invention_fraction(steps: List[StepResult], theta: float) -> float:
    """Definition 2.5: beta(theta,B) = sum_{i: alpha(si,B) <= theta} w_i / sum_i w_i."""
    total_weight = sum(s.w_si for s in steps)
    if total_weight == 0:
        return 0.0
    surprising_weight = sum(s.w_si for s in steps if s.alpha_combined <= theta)
    return surprising_weight / total_weight


def theorem_2_1_lower_bound(log_PPw: float, theta: float, kappa_max: float) -> float | None:
    """Theorem 2.1: for kappa(theta) := -log(theta), whenever log PP_w(pi,B) > kappa(theta),

        beta(theta,B) >= (log PP_w(pi,B) - kappa(theta)) / (kappa_max - kappa(theta)).

    Returns None if the hypothesis log PP_w(pi,B) > kappa(theta) does not hold
    (the bound is vacuous / not guaranteed to be informative in that regime)."""
    kappa_theta = -math.log(theta)
    if log_PPw <= kappa_theta:
        return None
    denom = kappa_max - kappa_theta
    if denom <= 0:
        return None
    return (log_PPw - kappa_theta) / denom
