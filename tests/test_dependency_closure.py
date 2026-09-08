"""
Verifies dependency_tree.project_dependency_closure correctly isolates a
single target theorem's own proof steps (Dep(T) \\ L) even when the project
folder contains other, independent theorems -- the scenario the
formal-math/zeta23 repo actually presents (theorems A, B, C sharing files).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import dependency_tree as dtree
from src.lean_parser import Declaration


def _decl(name, module, citations):
    d = Declaration(name=name, kind="theorem", module=module, raw_text=f"theorem {name} := sorry")
    d.citations = citations
    return d


def test_dependency_closure_isolates_unrelated_theorems():
    index = {}
    # Shared library material.
    index["lib_lemma_common"] = _decl("lib_lemma_common", "Mathlib.Common", [])

    # Theorem A's own proof steps, sharing the library lemma but otherwise disjoint from B.
    index["a_step_1"] = _decl("a_step_1", "Proj.A", ["lib_lemma_common"])
    index["a_step_2"] = _decl("a_step_2", "Proj.A", ["a_step_1"])
    index["theorem_A"] = _decl("theorem_A", "Proj.A", ["a_step_2", "lib_lemma_common"])

    # Theorem B: an entirely independent result living in the same project folder.
    index["b_step_1"] = _decl("b_step_1", "Proj.B", ["lib_lemma_common"])
    index["theorem_B"] = _decl("theorem_B", "Proj.B", ["b_step_1"])

    library_names = {"lib_lemma_common"}

    steps_A = dtree.project_dependency_closure("theorem_A", index, library_names)
    steps_B = dtree.project_dependency_closure("theorem_B", index, library_names)

    assert steps_A == {"a_step_1", "a_step_2"}, steps_A
    assert steps_B == {"b_step_1"}, steps_B
    # Critically: B's steps must NOT leak into A's closure just because they
    # share a folder and a common library dependency.
    assert "b_step_1" not in steps_A
    assert "a_step_1" not in steps_B
    assert "theorem_A" not in steps_A  # T itself is excluded, per Definition A.3
    assert "lib_lemma_common" not in steps_A  # library material is excluded too


if __name__ == "__main__":
    test_dependency_closure_isolates_unrelated_theorems()
    print("OK")
