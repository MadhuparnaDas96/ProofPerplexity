# Proof Non-Triviality Benchmark

Implementation of the probabilistic non-triviality benchmark from
*"A Probabilistic Approach to Measuring non-triviality in Machine Generated Proofs"*
(Das & Banerjee, NeurIPS 2026 submission).

# Proof Non-Triviality Benchmark

Implementation of the probabilistic non-triviality benchmark from
*"A Probabilistic Approach to Measuring non-triviality in Machine Generated Proofs"*
(Das & Banerjee, NeurIPS 2026 submission).

**The full Mathlib4 library ships bundled in `data/mathlib/Mathlib/`** (232,591
declarations, parsed from the real `leanprover-community/mathlib4` repo — see
`Mathlib library version` below for exactly which snapshot). You only need to
drop in your own proof project folder (e.g. a `zeta23`-style repo) and name a
target theorem `T`; this automatically isolates `T`'s own dependency closure
within that project as its intermediate proof steps `s1, ..., sn` (Definition
A.3, `Dep(T) \ L`) — correctly ignoring any *other*, unrelated theorems that
happen to live in the same project folder — and computes:

- **Ablation recoverability** `α_abl(si, B)` (Definition 2.1)
- **Mahalanobis recoverability** `α_mah(si)` (Definition 2.2)
- **Combined recoverability / surprisal** `α(si,B)`, `Surp(si,B)` (Definition 2.3)
- **Proof perplexity** `PP_w(π, B)` (Definition 2.4)
- **θ-invention fraction** `β(θ, B)` (Definition 2.5), with the Theorem 2.1 lower bound
  checked automatically as a sanity assertion.

## Mathlib library version

The bundled `data/mathlib/Mathlib/` is exactly the `Mathlib/` folder from a
`master`-branch snapshot of `leanprover-community/mathlib4` (8,487 `.lean`
files, ~98MB of source, parsing to 232,591 namespace-qualified declarations).
Its internal directory/module structure is untouched — nothing was
reorganised, renamed, or filtered — so the module-prefix clustering in
`src/mahalanobis.py` operates on the same hierarchy Mathlib itself uses (e.g.
`Mathlib.Analysis.*`, `Mathlib.Order.*`).

**If your proof project pins a different, specific Mathlib commit/tag** (check
its `lean-toolchain`/`lake-manifest.json`), citation matching will be most
accurate if you swap in that exact revision instead — some declarations may
have been renamed or moved between snapshots, and the regex-based matcher
(see below) will silently undercount citations to anything renamed, not error.
To do that: replace the contents of `data/mathlib/Mathlib/` with the `Mathlib/`
folder from that revision; nothing else in the pipeline needs to change.

## What's exact vs. heuristic

The math (Definitions 1.1, 2.1–2.5, Theorem 2.1, Ledoit-Wolf shrinkage, χ² tail
probabilities, Bonferroni correction) is implemented **exactly** as specified in the
paper — see `src/recoverability.py`, `src/mahalanobis.py`, `src/perplexity.py`.

Two pieces are inherently approximations unless you plug in heavier tooling:

1. **Dependency extraction (`src/lean_parser.py`)** — building `Cite(d)`/`Dep(d)`
   (Definitions A.2–A.4) precisely requires Lean's own elaborator, since it must see
   the fully elaborated term, not the surface syntax. This repo ships a regex-based
   parser that:
   - tracks `namespace ... end` nesting line-by-line so declaration names are
     namespace-qualified exactly the way Lean itself refers to them (e.g. a
     `theorem foo` inside `namespace Nat` becomes `Nat.foo`, not `foo` — this
     matters a lot, since Mathlib declares almost everything inside namespaces);
   - resolves an unqualified or partially-qualified citation (as written after an
     `open Namespace`) to its unique fully-qualified target when exactly one
     candidate exists in the whole library, and conservatively skips it otherwise
     rather than guessing;
   - deliberately excludes short tokens (< 4 characters) from that resolution step,
     because on a 200k+-declaration library a generic bound-variable name like `a`,
     `n`, or `h` can coincidentally be the *only* declaration whose name happens to
     end in that exact suffix (e.g. a structure field literally named `a`), which
     would otherwise look "unambiguous" and get wrongly counted as a citation —
     this was caught and fixed by testing against the real bundled Mathlib, not a toy example.

   It still won't see citations introduced purely by elaboration (implicit
   arguments, typeclass resolution, notation unfolding invisible in surface syntax).
   For publication-grade fidelity, swap `parse_file`/`compute_citations` for output
   from `lean4export` or a [LeanDojo](https://github.com/lean-dojo/LeanDojo)-traced
   repo — the rest of the pipeline (tree-building, w(si) computation) consumes a
   plain `{declaration -> ordered citation list}` map and doesn't care where it came
   from.

   **Tractability on the full Mathlib checkout:** citations are only ever computed
   for the theorem project's own declarations (never for the whole of L — see
   `compute_citations`'s docstring for why that's mathematically safe, not just an
   optimisation), so parsing all 232,591 declarations takes about 5-6 seconds, and a
   full run (project parsing + Dep(T) + w(si) + clustering + Mahalanobis fit) takes
   single-digit seconds on the bundled library. The only place a *sample* of L gets
   its own citations computed is if you choose a `structural_only`/`concat` feature
   mode (`embedding.feature_mode`), and even then only for
   `max_library_sample_for_fit` declarations, not all of Mathlib.

2. **Search-budget prover (`src/search_prover.py`)** — computing `w(si, B)` (Definition
   A.5) means actually trying to re-derive `si` from `Pi` within budget `B`. Two modes
   are provided:
   - `manual` (default, and what you asked for): you supply `w(si,B)` per step
     directly in `manual_overrides.json`. This is fully legitimate per the paper —
     Definition 1.1 leaves the choice of search procedure open, and the rest of the
     pipeline only consumes the resulting `w(si,B)` values.
   - `lean_llm_search` (optional): calls an LLM to propose tactic proofs for `si`
     using only `CandB ∪ Pi`, and verifies each candidate by shelling out to
     `lake env lean` (real kernel checking). Requires a local Lean/Lake toolchain and
     an LLM API key. This is slow and is meant as a starting point, not a competitive
     prover — AlphaProof/Aristotle-grade search is out of scope for this repo.

## Setup

```bash
pip install -r requirements.txt

# 1. Mathlib is already bundled in data/mathlib/Mathlib/ -- nothing to do here
#    unless your proof project pins a different revision (see version note above).

# 2. Drop your proof project folder into proofs/ as-is -- no restructuring needed.
#    E.g. for a formal-math-style repo, you only need the specific self-contained
#    project subdirectory (not the whole outer repo):
cp -r /path/to/formal-math/zeta23 proofs/zeta23

# 3. Find the exact, namespace-qualified name of the theorem you want to score
#    (Lean names can carry unicode subscripts, e.g. "Zeta23.thmB₀_mult" -- don't
#    hand-type these, and don't assume the file name tells you the namespace):
python main.py find --dir proofs/zeta23 --query thm

# 4. Edit config.yaml:
#    - set theorem_dir / target_theorem_name (from step 3)
#    - set B (search budget), theta (invention threshold)
#    - set cand_pool_size (or alpha_min_override directly)
#    - set embedding.api_base / api_key_env / model, and export that env var
#    - (optional) point prover.manual_overrides_file at a JSON of {step_name: w(si,B)}

# 5. Run
python main.py run --config config.yaml
```

Results (per-step scores, aggregate proof perplexity, θ-invention fraction, and the
Theorem 2.1 bound check) are written to `outputs/results.json` and printed as a table.

If your proof project formalises more than one theorem in the same files (as a
`zeta23`-style repo proving several related results might), `main.py run` scores
exactly one `target_theorem_name` per invocation — its own `Dep(T)` is isolated
automatically, so unrelated theorems sharing the same folder are correctly excluded.
Rerun with a different `target_theorem_name` to score another one.

## Folder structure

```
config.yaml                    # all run parameters live here
manual_overrides.json          # optional: your own w(si,B) values per step
main.py                        # CLI entry point (`run`, `find`)
src/
  lean_parser.py               # folder-of-.lean -> {declaration: (module, raw_text, cite_sequence)}
  dependency_tree.py           # Definition A.3: Dep(T); Definition A.4: T(d|Pi), w(si)
  search_prover.py             # Definition A.5 + Definition 1.1: w(si,B), pluggable
  embeddings.py                 # phi(si) construction (embedding API + structural counts)
  mahalanobis.py                # module clustering, Ledoit-Wolf, DM, chi2, Bonferroni -> alpha_mah
  recoverability.py             # Definitions 2.1-2.3
  perplexity.py                 # Definitions 2.4-2.5, Theorem 2.1
  pipeline.py                   # orchestration
proofs/                        # your proof project folder(s) go here
data/mathlib/Mathlib/           # the bundled, real Mathlib4 library (see version note above)
outputs/results.json            # written by `main.py run`
tests/
  test_synthetic.py             # end-to-end smoke test on synthetic data (no Lean/API needed)
  test_dependency_closure.py    # verifies Dep(T) isolation when a folder has multiple theorems
```

## Sanity-testing

```bash
python -m pytest tests/ -q
```

`test_synthetic.py` runs the full pipeline on a small synthetic dependency graph and
embedding set, and asserts that `0 <= β(θ,B) <= 1`, `PP_w(π,B) >= 1`, and that the
Theorem 2.1 bound actually holds against the directly-computed `β(θ,B)` — i.e. it
checks the paper's own theorem numerically, not just that the code runs.

`test_dependency_closure.py` checks that when a project folder contains more than one
theorem, scoring one of them doesn't pull in the others' proof steps just because
they share a folder and some library dependencies.

A complete end-to-end demo — a two-step example theorem citing two *real* Mathlib
declarations (`Order.le_succ`, `comp_mul_left`), scored against the actual bundled
232,591-declaration library, using `structural_only` features and manual `w(si,B)` —
runs in well under 10 seconds with no API key needed:

```bash
python main.py run --config config.demo.yaml
```
