"""
End-to-end smoke test on a small synthetic dependency graph. No Lean toolchain
or embedding API required: uses feature_mode="structural_only" and
prover.mode="manual". This is the fastest way to check the pipeline runs and
that Theorem 2.1's bound actually holds against a directly-computed beta.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import dependency_tree as dtree
from src import mahalanobis as mah
from src import perplexity as pp
from src import recoverability as rec
from src.embeddings import build_feature_vectors
from src.lean_parser import Declaration


def _make_synthetic_library(n_lib=60, n_clusters=3):
    """Build a fake library of declarations spread across a few 'modules',
    each citing a handful of shared 'hub' declarations plus some noise, so
    the clusters have genuinely different citation statistics."""
    index = {}
    hubs = {c: [f"hub_{c}_{k}" for k in range(3)] for c in range(n_clusters)}
    for c in range(n_clusters):
        for h in hubs[c]:
            index[h] = Declaration(name=h, kind="lemma", module=f"Mod{c}.Hub", raw_text=f"lemma {h} : True := trivial")
    for i in range(n_lib):
        c = i % n_clusters
        name = f"lib_decl_{c}_{i}"
        cites = hubs[c] * ((i % 4) + 1)  # skewed: hub citations repeat with multiplicity
        raw = f"lemma {name} : True := by\n  " + "\n  ".join(f"exact {h}" for h in cites)
        d = Declaration(name=name, kind="lemma", module=f"Mod{c}.Sub", raw_text=raw)
        d.citations = list(cites)
        index[name] = d
    return index, hubs


def _make_synthetic_steps(index, hubs, n_steps=4):
    """Two steps that are 'easy' (cite only hub material heavily, structurally
    typical) and two that are 'hard' (introduce genuinely new sub-structure,
    citing hubs from a DIFFERENT cluster than their own -- i.e. a structural
    outlier) plus internal complexity that a search prover fails to recover."""
    step_names = []
    c0_hub = hubs[0]
    c1_hub = hubs[1]
    # Easy step: typical citation pattern for cluster 0.
    s1 = Declaration(name="s1_easy", kind="lemma", module="Target.Steps",
                      raw_text="lemma s1_easy : True := by\n  exact " + "\n  exact ".join(c0_hub))
    s1.citations = list(c0_hub)
    index["s1_easy"] = s1
    step_names.append("s1_easy")

    # Hard step: cites a hub from a cluster it doesn't belong to (structural
    # outlier) AND introduces a chain of new internal sub-lemmas (one of them
    # cited twice, for multiplicity) -> strictly larger w(si) than the easy step.
    aux1 = Declaration(name="aux_new_1", kind="lemma", module="Target.Steps",
                        raw_text="lemma aux_new_1 : True := trivial")
    aux2 = Declaration(name="aux_new_2", kind="lemma", module="Target.Steps",
                        raw_text="lemma aux_new_2 : True := by\n  exact aux_new_1\n  exact aux_new_1")
    aux2.citations = ["aux_new_1", "aux_new_1"]
    aux3 = Declaration(name="aux_new_3", kind="lemma", module="Target.Steps",
                        raw_text="lemma aux_new_3 : True := by\n  exact aux_new_2")
    aux3.citations = ["aux_new_2"]
    index["aux_new_1"] = aux1
    index["aux_new_2"] = aux2
    index["aux_new_3"] = aux3
    s2 = Declaration(name="s2_hard", kind="lemma", module="Target.Steps",
                      raw_text="lemma s2_hard : True := by\n  exact aux_new_3\n  exact " + c1_hub[0])
    s2.citations = ["aux_new_3", c1_hub[0]]
    index["s2_hard"] = s2
    step_names.append("s2_hard")

    return step_names


def test_end_to_end_theorem_2_1_bound_holds():
    index, hubs = _make_synthetic_library()
    step_names = _make_synthetic_steps(index, hubs)
    L_names = set(index.keys()) - set(step_names) - {"aux_new_1", "aux_new_2", "aux_new_3"}

    pi_order = dtree.topological_order(step_names, index)
    assert set(pi_order) == set(step_names)

    # w(si)
    w_si = {}
    Pi_running = set(L_names)
    for name in pi_order:
        w_si[name] = dtree.internal_complexity(name, Pi_running, index)
        Pi_running |= {name}
    assert w_si["s1_easy"] >= 0
    assert w_si["s2_hard"] > w_si["s1_easy"], "the 'hard' step should have strictly more internal complexity"

    # manual w(si,B): prover recovers the easy step almost fully, fails the hard one
    cand_pool_size = 10
    alpha_min = rec.alpha_min_from_cand_pool(cand_pool_size)
    w_si_B = {
        "s1_easy": max(0, w_si["s1_easy"] - 1),  # nearly fully recovered -> low surprisal
        "s2_hard": w_si["s2_hard"],               # Definition A.5 failure convention
    }

    # Features + clustering (structural only, no API calls)
    l_sample = list(L_names)
    all_names = l_sample + pi_order
    phi_by_name = build_feature_vectors(
        index=index, names=all_names, feature_mode="structural_only",
        feature_dim=6, structural_top_k=12, embedding_client=None,
    )
    labels = mah.assign_clusters(index, l_sample, prefix_depth=1, min_cluster_size=5)
    cluster_model = mah.fit_cluster_model({n: phi_by_name[n] for n in l_sample}, labels)

    steps = []
    for name in pi_order:
        a_mah, _ = mah.alpha_mah(phi_by_name[name], cluster_model, alpha_min)
        a_abl = rec.alpha_ablation(w_si[name], w_si_B[name], alpha_min)
        a_comb = rec.combined_recoverability(a_abl, a_mah)
        surp = rec.surprisal(a_comb, alpha_min)
        steps.append(pp.StepResult(name=name, w_si=w_si[name], w_si_B=w_si_B[name],
                                    alpha_abl=a_abl, alpha_mah=a_mah,
                                    alpha_combined=a_comb, surprisal=surp))

    k_max = rec.kappa_max(alpha_min)
    ppw = pp.proof_perplexity(steps)
    assert ppw >= 1.0 - 1e-9

    theta = 0.3
    beta = pp.theta_invention_fraction(steps, theta)
    assert 0.0 <= beta <= 1.0

    bound = pp.theorem_2_1_lower_bound(math.log(ppw), theta, k_max)
    if bound is not None:
        assert beta >= bound - 1e-9, (
            f"Theorem 2.1 violated: beta={beta} < bound={bound} "
            "(this would indicate a bug in the implementation, not the paper)"
        )

    # The hard step, which the prover failed to recover AND which is a
    # structural outlier relative to its own cluster, should be scored as
    # (close to) maximally surprising.
    hard = next(s for s in steps if s.name == "s2_hard")
    assert hard.alpha_combined <= 0.5 + 1e-9
    easy = next(s for s in steps if s.name == "s1_easy")
    assert easy.surprisal <= hard.surprisal


if __name__ == "__main__":
    test_end_to_end_theorem_2_1_bound_holds()
    print("OK")
