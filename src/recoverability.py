"""
Definitions 2.1 (Ablation Recoverability), 2.3 (Combined Recoverability / Surprisal).
Mahalanobis recoverability (Definition 2.2) lives in mahalanobis.py; this module
combines the two exactly as specified.
"""
from __future__ import annotations

import math


def alpha_min_from_cand_pool(cand_pool_size: int) -> float:
    """alpha_min := 1 / (1 + |Cand_B|)."""
    return 1.0 / (1.0 + cand_pool_size)


def alpha_ablation(w_si: int, w_si_B: int, alpha_min: float) -> float:
    """Definition 2.1: alpha_abl(si,B) = max(1 - w(si,B)/w(si), alpha_min).

    Guards against w_si == 0 (a step with no internal complexity at all is,
    by convention, treated as trivially recoverable up to the floor)."""
    if w_si <= 0:
        return alpha_min
    ratio = w_si_B / w_si
    return max(1.0 - ratio, alpha_min)


def combined_recoverability(alpha_abl: float, alpha_mah: float) -> float:
    """Definition 2.3: alpha(si,B) = min(alpha_abl(si,B), alpha_mah(si))."""
    return min(alpha_abl, alpha_mah)


def surprisal(alpha_combined: float, alpha_min: float) -> float:
    """Surp(si,B) = -log(alpha(si,B)), bounded above by kappa_max = -log(alpha_min).

    alpha_combined is clamped into [alpha_min, 1] first as a numerical safety
    net (it should already lie there by construction of alpha_abl/alpha_mah)."""
    alpha_clamped = min(max(alpha_combined, alpha_min), 1.0)
    return -math.log(alpha_clamped)


def kappa_max(alpha_min: float) -> float:
    return -math.log(alpha_min)
