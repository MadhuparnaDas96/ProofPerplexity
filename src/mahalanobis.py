"""
Implements Definition 2.2 (Mahalanobis Recoverability) and its justification
in Appendix A.4: partition L into clusters by module/topic area, fit a
Ledoit-Wolf shrinkage estimate (mu_c, Sigma_c) per cluster, compute the
Mahalanobis distance D_M(si | c), convert to a chi-squared tail probability
under the Gaussian null, and Bonferroni-correct across clusters.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import chi2
from sklearn.covariance import LedoitWolf

from .lean_parser import Declaration

MISC_CLUSTER = "Mathlib.Misc"


def assign_clusters(index: Dict[str, Declaration], names: List[str],
                     prefix_depth: int, min_cluster_size: int) -> Dict[str, str]:
    """Cluster label per declaration name, by Mathlib module-path prefix
    (Appendix A.4: 'Partition L into clusters c=1,...,C by module or topic
    area'). Clusters below `min_cluster_size` are merged into a single
    MISC_CLUSTER so every (mu_c, Sigma_c) has enough samples for Ledoit-Wolf
    shrinkage to be meaningful."""
    def prefix(module: str) -> str:
        parts = module.split(".")
        return ".".join(parts[:prefix_depth]) if parts else MISC_CLUSTER

    raw_labels: Dict[str, str] = {}
    for name in names:
        decl = index.get(name)
        raw_labels[name] = prefix(decl.module) if decl is not None else MISC_CLUSTER

    counts: Dict[str, int] = defaultdict(int)
    for label in raw_labels.values():
        counts[label] += 1

    final_labels: Dict[str, str] = {}
    for name, label in raw_labels.items():
        final_labels[name] = label if counts[label] >= min_cluster_size else MISC_CLUSTER
    return final_labels


class ClusterModel:
    """Holds a fitted Ledoit-Wolf (mu_c, Sigma_c^{-1}) per cluster."""

    def __init__(self):
        self.mean: Dict[str, np.ndarray] = {}
        self.precision: Dict[str, np.ndarray] = {}  # Sigma_c^{-1}
        self.dim: int = 0

    def fit(self, cluster_to_vectors: Dict[str, List[np.ndarray]]) -> None:
        for label, vectors in cluster_to_vectors.items():
            X = np.vstack(vectors)
            self.dim = X.shape[1]
            if X.shape[0] < 2:
                # Degenerate cluster: fall back to identity covariance so the
                # pipeline still runs, but this cluster will contribute little
                # discriminative power -- increase min_cluster_size if this fires.
                self.mean[label] = X.mean(axis=0) if X.shape[0] else np.zeros(self.dim)
                self.precision[label] = np.eye(self.dim)
                continue
            lw = LedoitWolf().fit(X)
            self.mean[label] = lw.location_
            self.precision[label] = lw.precision_  # already Sigma_c^{-1}

    def mahalanobis(self, phi: np.ndarray, label: str) -> float:
        diff = phi - self.mean[label]
        return float(np.sqrt(diff @ self.precision[label] @ diff))


def fit_cluster_model(phi_by_name: Dict[str, np.ndarray],
                       labels_by_name: Dict[str, str]) -> ClusterModel:
    cluster_to_vectors: Dict[str, List[np.ndarray]] = defaultdict(list)
    for name, phi in phi_by_name.items():
        cluster_to_vectors[labels_by_name[name]].append(phi)
    model = ClusterModel()
    model.fit(cluster_to_vectors)
    return model


def alpha_mah(phi_si: np.ndarray, model: ClusterModel, alpha_min: float) -> Tuple[float, Dict[str, float]]:
    """Definition 2.2: alpha_mah(si) = max(p~(si), alpha_min), where
    p~(si) = min(1, C * min_c p_c(si)) is the Bonferroni-corrected minimum
    p-value across all C clusters, and
    p_c(si) = 1 - F_{chi2_d}(D_M(si|c)^2) under the Gaussian null.

    Returns (alpha_mah, {cluster_label: p_c}) for inspection/debugging.
    """
    labels = list(model.mean.keys())
    C = len(labels)
    d = model.dim
    p_by_cluster: Dict[str, float] = {}
    for label in labels:
        dm = model.mahalanobis(phi_si, label)
        p_c = float(chi2.sf(dm ** 2, df=d))  # 1 - CDF, i.e. the upper tail probability
        p_by_cluster[label] = p_c
    p_min = min(p_by_cluster.values()) if p_by_cluster else 1.0
    p_corrected = min(1.0, C * p_min)
    return max(p_corrected, alpha_min), p_by_cluster
