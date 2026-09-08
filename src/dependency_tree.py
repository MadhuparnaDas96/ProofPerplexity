"""
Implements Definition A.4 (Local Dependency Tree) and the internal complexity
w(si) := |Nodes(T(si | Pi))| - 1, plus the topological ordering pi of Section 1.1 / A.2.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Set

import networkx as nx

from .lean_parser import Declaration


def project_dependency_closure(target: str, index: Dict[str, Declaration],
                                library_names: Set[str]) -> Set[str]:
    """Definition A.3 (Dependency Closure): Dep(T), unfolded transitively
    through the full citation graph (library declarations included, since a
    project declaration can cite a library declaration that itself is
    irrelevant to filter out -- we simply never expand INTO a library
    declaration's own citations, matching the leaf convention of Definition
    A.4 and the fact that L is fixed, pre-existing material).

    Returns Dep(T) restricted to project-local declarations only, i.e.
    exactly {s1,...,sn} in the notation of Section 1.1: 'the entire library L
    ... without our fixed target theorem T and its proof', so the
    intermediate declarations are precisely T's dependencies that are NOT
    already part of L.

    This matters whenever the project folder contains more than one
    formalised theorem (as in a repo proving several independent results):
    without restricting to T's own dependency closure, unrelated
    declarations from other theorems in the same folder would incorrectly be
    counted as T's proof steps.
    """
    visited: Set[str] = set()
    queue: deque[str] = deque([target])
    while queue:
        name = queue.popleft()
        decl = index.get(name)
        if decl is None:
            continue
        if name in library_names and name != target:
            # Leaf convention: a library declaration's own citations are not
            # "new" proof steps of T, so we don't expand further into them.
            continue
        for cited in set(decl.citations):
            if cited == target or cited in visited:
                continue
            visited.add(cited)
            queue.append(cited)
    return {n for n in visited if n not in library_names}


def node_count(d_name: str, Pi: Set[str], index: Dict[str, Declaration],
                memo: Dict[str, int] | None = None) -> int:
    """|Nodes(T(d | Pi))|, i.e. INCLUDING the root, per Definition A.4.

    - If d in Pi: tree is just the root (already-available material is atomic).
    - Else: root's branches are T(c|Pi) for every occurrence c in Cite(d),
      one branch per occurrence (multiplicity, not deduplicated).

    Implemented iteratively (an explicit stack, not Python's call stack) with
    on-path cycle detection. Definition A.4's recursion is only guaranteed to
    terminate because (Gamma, <) is a true DAG (Lemma A.1) -- a guarantee that
    comes from the real Lean kernel's elaboration order. Our citation graph is
    only a heuristic, regex-based approximation of that (see lean_parser.py),
    and on a large real project it CAN contain a spurious cycle the true
    kernel graph would not have -- e.g. two declarations in the same `mutual`
    block appearing to cite each other, or an accidental text-match. If a
    node reappears on its own current expansion path, that back-edge is
    treated as a leaf (contributes 1, not expanded further) rather than
    recursing forever; the iterative structure here also means a very deep
    (but genuinely acyclic) real dependency chain can't hit Python's
    recursion-depth limit either.
    """
    if memo is None:
        memo = {}
    if d_name in Pi:
        return 1
    if d_name in memo:
        return memo[d_name]

    # Each frame: [name, next citation index, running total, citations list]
    stack: List[list] = []
    on_path: Set[str] = set()

    def _push(name: str) -> None:
        decl = index.get(name)
        cites = decl.citations if decl is not None else []
        stack.append([name, 0, 1, cites])
        on_path.add(name)

    _push(d_name)

    while stack:
        frame = stack[-1]
        name, i, _total, cites = frame
        if i >= len(cites):
            stack.pop()
            on_path.discard(name)
            memo[name] = frame[2]
            if stack:
                stack[-1][2] += frame[2]
            continue

        frame[1] += 1
        child = cites[i]

        if child in Pi:
            frame[2] += 1
        elif child in memo:
            frame[2] += memo[child]
        elif child in on_path:
            # Cycle: a heuristic-parsing artifact, not something the real
            # kernel DAG would allow. Break it by treating this occurrence as
            # a leaf rather than looping forever.
            frame[2] += 1
        else:
            _push(child)

    return memo[d_name]


def internal_complexity(step_name: str, Pi: Set[str],
                         index: Dict[str, Declaration]) -> int:
    """w(si) := |Nodes(T(si | Pi))| - 1  (non-root nodes only)."""
    memo: Dict[str, int] = {}
    return node_count(step_name, Pi, index, memo) - 1


def topological_order(step_names: List[str], index: Dict[str, Declaration]) -> List[str]:
    """A linear extension pi = (s1,...,sn) of the direct-dependency relation
    restricted to {s1,...,sn}, satisfying Dep(si) ∩ {s1,...,sn} ⊆ {s1,...,s_{i-1}}
    (Section 1.1 / Appendix A.2, 'Recovering the Topological order pi').

    Guaranteed to exist by Lemma A.1 ((Gamma,<) is a DAG); falls back to file
    order for any pair not comparable via citations."""
    g = nx.DiGraph()
    g.add_nodes_from(step_names)
    step_set = set(step_names)
    for name in step_names:
        decl = index.get(name)
        if decl is None:
            continue
        for cited in decl.citations:
            if cited in step_set and cited != name:
                g.add_edge(cited, name)  # cited must precede name
    try:
        order = list(nx.topological_sort(g))
    except nx.NetworkXUnfeasible:
        # Citation heuristic introduced a spurious cycle (surface-syntax parsing
        # artifact -- see lean_parser.py caveat). Break cycles by falling back to
        # original file order for the offending nodes rather than failing the run.
        order = step_names
    # topological_sort only orders nodes that were added; make sure every step
    # is present even if isolated.
    missing = [n for n in step_names if n not in order]
    return order + missing