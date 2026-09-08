"""
End-to-end orchestration:

  1. Parse Mathlib (L, a folder) + the theorem project (also a folder, e.g. a
     cloned formal-math proof repo -- a single .lean file also works) into a
     merged declaration index. Citations are computed ONLY for the project's
     own declarations at this stage (see lean_parser.compute_citations).
  2. Restrict to T's dependency closure Dep(T) \\ L (dependency_tree.
     project_dependency_closure) to get s1..sn even when the project folder
     contains other, unrelated theorems -- then topologically order them (pi).
  3. For each step si: compute Pi = L u {s1,...,s_{i-1}}, then w(si).
  4. Run the (pluggable) search prover to get w(si,B) for each step.
  5. Build phi(d) for a sample of L and phi(si) for each step; cluster L by
     module. (Citations for the L sample, if needed by the feature mode, are
     computed only now -- see step 5 in run().)
  6. Fit per-cluster Ledoit-Wolf (mu_c, Sigma_c); compute alpha_mah(si).
  7. Combine into alpha(si,B), Surp(si,B); aggregate into PP_w(pi,B), beta(theta,B),
     and check the Theorem 2.1 bound.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

from . import dependency_tree as dtree
from . import lean_parser as lp
from . import mahalanobis as mah
from . import perplexity as pp
from . import recoverability as rec
from .embeddings import EmbeddingClient, build_feature_vectors
from .search_prover import build_prover


def run(config: dict) -> dict:
    theorem_dir = config["theorem_dir"]
    mathlib_dir = config["mathlib_dir"]
    target_theorem = config["target_theorem_name"]

    # --- 1. Parse (both are folders of .lean sources; a single file also works) ---
    project_decls = lp.scan_lean_sources(theorem_dir)
    if target_theorem not in project_decls:
        raise ValueError(
            f"target_theorem_name {target_theorem!r} was not found among the "
            f"{len(project_decls)} declarations parsed from {theorem_dir}. "
            "Run `python main.py find --dir <theorem_dir>` to list the exact "
            "declaration names the parser found (Lean names can contain "
            "unicode subscripts, e.g. 'Zeta23.thmB\u2080_mult', that are easy to mistype)."
        )

    mathlib_decls = lp.scan_lean_sources(mathlib_dir)
    # Project declarations take precedence on name collisions (e.g. the
    # project vendoring a modified copy of a library lemma).
    index: Dict[str, lp.Declaration] = {**mathlib_decls, **project_decls}
    L_names = set(mathlib_decls.keys()) - set(project_decls.keys())
    known_names = set(index.keys())
    # Built once over the full name set (cheap -- just string splitting) so
    # citations under an `open Namespace` can be resolved to their unique
    # fully-qualified target even without computing Mathlib's own citations.
    suffix_index = lp.build_suffix_index(known_names)

    # Only the project's own declarations need citations to compute Dep(T) and
    # w(si) -- see compute_citations' docstring for why Mathlib's internal
    # citation structure is never needed for this part.
    lp.compute_citations(index, known_names, restrict_to=project_decls.keys(),
                          suffix_index=suffix_index)

    # --- 2. Restrict to T's own dependency closure, then topologically order ---
    # This isolates T's actual proof steps even if theorem_dir contains other,
    # unrelated formalised theorems alongside it.
    step_name_set = dtree.project_dependency_closure(target_theorem, index, L_names)
    if not step_name_set:
        raise ValueError(
            f"{target_theorem!r} has no project-local dependencies under the "
            "citation heuristic -- either it's a one-line proof entirely from "
            "the library (nothing to score), or the regex-based citation "
            "extraction missed its dependencies (see README caveats)."
        )
    pi_order = dtree.topological_order(list(step_name_set), index)

    # --- 3. w(si) for each step ----------------------------------------------
    w_si: Dict[str, int] = {}
    Pi_running = set(L_names)
    for name in pi_order:
        w_si[name] = dtree.internal_complexity(name, Pi_running, index)
        Pi_running = Pi_running | {name}

    # --- 4. alpha_min, Cand_B, w(si,B) ---------------------------------------
    alpha_min = (config.get("alpha_min_override")
                 or rec.alpha_min_from_cand_pool(config["cand_pool_size"]))
    B = config["B"]
    prover = build_prover(config)
    w_si_B: Dict[str, int] = {}
    prover_succeeded: Dict[str, bool] = {}
    for name in pi_order:
        recovered, ok = prover.recover(name, w_si[name], B)
        w_si_B[name] = recovered
        prover_succeeded[name] = ok

    # --- 5. Feature vectors phi(d) for L (sampled) and phi(si) for steps -----
    emb_cfg = config["embedding"]
    embedding_client = None
    if emb_cfg["feature_mode"] in ("embedding_only", "concat"):
        import os
        embedding_client = EmbeddingClient(
            api_base=emb_cfg["api_base"],
            api_key=os.environ.get(emb_cfg["api_key_env"], ""),
            model=emb_cfg["model"],
        )

    # Fit the cluster statistics on L itself (excluding the new steps, since
    # they are not yet part of the library) -- capped for tractability if L is huge.
    l_sample = list(L_names)
    max_library_sample = config.get("max_library_sample_for_fit", 4000)
    if len(l_sample) > max_library_sample:
        import random
        random.Random(0).shuffle(l_sample)
        l_sample = l_sample[:max_library_sample]

    if emb_cfg["feature_mode"] in ("structural_only", "concat"):
        # Structural features need citations for the sampled library
        # declarations too (to build the premise-usage vectors and the
        # library-wide top-K vocabulary) -- computed only for this sample,
        # never for the whole of L, for the same tractability reason as step 1.
        lp.compute_citations(index, known_names, restrict_to=l_sample,
                              suffix_index=suffix_index)

    all_names_for_features = l_sample + pi_order
    phi_by_name = build_feature_vectors(
        index=index,
        names=all_names_for_features,
        feature_mode=emb_cfg["feature_mode"],
        feature_dim=emb_cfg["feature_dim"],
        structural_top_k=emb_cfg["structural_top_k"],
        embedding_client=embedding_client,
    )

    # --- 6. Clustering + Ledoit-Wolf + alpha_mah ------------------------------
    clust_cfg = config["clustering"]
    labels = mah.assign_clusters(
        index, l_sample,
        prefix_depth=clust_cfg["prefix_depth"],
        min_cluster_size=clust_cfg["min_cluster_size"],
    )
    cluster_model = mah.fit_cluster_model(
        {n: phi_by_name[n] for n in l_sample}, labels,
    )

    alpha_mah_by_step: Dict[str, float] = {}
    p_by_cluster_by_step: Dict[str, Dict[str, float]] = {}
    for name in pi_order:
        a_mah, p_by_cluster = mah.alpha_mah(phi_by_name[name], cluster_model, alpha_min)
        alpha_mah_by_step[name] = a_mah
        p_by_cluster_by_step[name] = p_by_cluster

    # --- 7. Combine, aggregate, check Theorem 2.1 -----------------------------
    steps: List[pp.StepResult] = []
    for name in pi_order:
        a_abl = rec.alpha_ablation(w_si[name], w_si_B[name], alpha_min)
        a_mah = alpha_mah_by_step[name]
        a_comb = rec.combined_recoverability(a_abl, a_mah)
        surp = rec.surprisal(a_comb, alpha_min)
        steps.append(pp.StepResult(
            name=name, w_si=w_si[name], w_si_B=w_si_B[name],
            alpha_abl=a_abl, alpha_mah=a_mah, alpha_combined=a_comb, surprisal=surp,
        ))

    k_max = rec.kappa_max(alpha_min)
    ppw = pp.proof_perplexity(steps)
    theta = config["theta"]
    beta = pp.theta_invention_fraction(steps, theta)
    bound = pp.theorem_2_1_lower_bound(math.log(ppw), theta, k_max)

    result = {
        "target_theorem": target_theorem,
        "pi_order": pi_order,
        "alpha_min": alpha_min,
        "kappa_max": k_max,
        "B": B,
        "theta": theta,
        "steps": [
            {
                "name": s.name, "w_si": s.w_si, "w_si_B": s.w_si_B,
                "prover_succeeded": prover_succeeded[s.name],
                "alpha_abl": s.alpha_abl, "alpha_mah": s.alpha_mah,
                "alpha_combined": s.alpha_combined, "surprisal": s.surprisal,
                "cluster_p_values": p_by_cluster_by_step[s.name],
            }
            for s in steps
        ],
        "proof_perplexity": ppw,
        "log_proof_perplexity": math.log(ppw),
        "theta_invention_fraction": beta,
        "theorem_2_1_lower_bound": bound,
        "theorem_2_1_bound_holds": (bound is None) or (beta >= bound - 1e-9),
    }
    return result


def write_results(result: dict, output_dir: str) -> Path:
    out_path = Path(output_dir) / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return out_path
