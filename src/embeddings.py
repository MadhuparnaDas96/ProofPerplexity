"""
Builds the structural feature vector phi(si) used in Definition 2.2, and
phi(d) for every d in L used to fit the per-cluster (mu_c, Sigma_c).

Supports three modes (config: embedding.feature_mode):
  - "structural_only": bag-of-dependencies count vector over the top-K most
    frequently cited declarations in the library ("a premise usage vector
    over cited constants", as described in Section 2.1.1). No API calls.
  - "embedding_only": a text-embedding of the declaration's statement+proof,
    from any OpenAI-compatible embeddings endpoint.
  - "concat": [structural counts | text embedding], then reduced to
    `feature_dim` dimensions via PCA (kept low relative to cluster sizes, as
    required for the chi-squared null in Definition 2.2 to be well-behaved).
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np
import requests
from sklearn.decomposition import PCA

from .lean_parser import Declaration


def _top_k_constants(index: Dict[str, Declaration], k: int,
                      scope: List[str] | None = None) -> List[str]:
    """Top-K most frequently cited constants, counted only over `scope`
    (defaults to the whole index). Restricting to `scope` matters when `index`
    is a full Mathlib checkout: citations are only computed (see
    lean_parser.compute_citations' restrict_to) for a sample of L plus the
    theorem's own steps, so scanning the full index here would silently
    return nothing for the untouched majority -- `scope` should match
    whatever that restrict_to was."""
    names = index.keys() if scope is None else scope
    freq: Counter = Counter()
    for name in names:
        decl = index.get(name)
        if decl is not None:
            freq.update(decl.citations)
    return [name for name, _ in freq.most_common(k)]


def structural_vector(decl: Declaration, vocab: List[str]) -> np.ndarray:
    """Premise-usage count vector over `vocab` (the top-K cited constants
    library-wide), preserving multiplicity as in Cite(d)."""
    counts = Counter(decl.citations)
    return np.array([counts.get(name, 0) for name in vocab], dtype=float)


class EmbeddingClient:
    def __init__(self, api_base: str, api_key: str, model: str):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self.api_key:
            raise RuntimeError(
                "EmbeddingClient: no API key set (check config.embedding.api_key_env "
                "and that the corresponding environment variable is exported)."
            )
        resp = requests.post(
            self.api_base,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={"model": self.model, "input": texts},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        # OpenAI-compatible responses come back in request order.
        return np.array([row["embedding"] for row in data], dtype=float)


def build_feature_vectors(
    index: Dict[str, Declaration],
    names: List[str],
    feature_mode: str,
    feature_dim: int,
    structural_top_k: int,
    embedding_client: EmbeddingClient | None = None,
) -> Dict[str, np.ndarray]:
    """Returns {name: phi(name)} for every name in `names`, all reduced to the
    same `feature_dim`-dimensional space (via a single PCA fit across `names`
    so every vector is comparable)."""
    vocab = _top_k_constants(index, structural_top_k, scope=names) if feature_mode in (
        "structural_only", "concat") else []

    raw_vectors: List[np.ndarray] = []
    for name in names:
        decl = index.get(name)
        pieces = []
        if feature_mode in ("structural_only", "concat"):
            pieces.append(structural_vector(decl, vocab) if decl is not None
                           else np.zeros(len(vocab)))
        if feature_mode in ("embedding_only", "concat"):
            if embedding_client is None:
                raise ValueError(f"feature_mode={feature_mode!r} requires an embedding_client")
            text = decl.raw_text if decl is not None else name
            emb = embedding_client.embed([text])[0]
            pieces.append(emb)
        raw_vectors.append(np.concatenate(pieces) if len(pieces) > 1 else pieces[0])

    raw_matrix = np.vstack(raw_vectors)
    target_dim = min(feature_dim, raw_matrix.shape[0], raw_matrix.shape[1])
    if target_dim < raw_matrix.shape[1] and target_dim >= 1:
        reduced = PCA(n_components=target_dim, random_state=0).fit_transform(raw_matrix)
    else:
        reduced = raw_matrix
    return {name: reduced[i] for i, name in enumerate(names)}
