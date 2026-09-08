"""
Heuristic extraction of Lean declarations and their citation sequences.

CAVEAT (see README): this approximates Definitions A.1-A.2 (Lean Declaration,
Direct Dependency) by scanning surface syntax, not the elaborated term. It will
under- or over-count citations relative to what the Lean kernel actually sees
(e.g. implicit arguments and typeclass instances resolved during elaboration are
invisible to a text scan). For higher fidelity, replace `parse_file` and
`compute_citations` below with output from `lean4export` or a LeanDojo-traced
repo -- everything downstream only consumes the `Declaration` objects produced
here, keyed by fully-qualified name, so the interface is the same either way.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Iterable, Set, Tuple

# Matches top-level declaration headers: theorem/lemma/def/instance/abbrev NAME ...
# (line-anchored; namespace qualification is handled separately in parse_file)
_DECL_HEADER_RE = re.compile(
    r"^\s*(?:@\[.*?\]\s*)?"
    r"(?:private\s+|protected\s+|noncomputable\s+|scoped\s+|local\s+)*"
    r"(theorem|lemma|def|instance|abbrev)\s+"
    r"([A-Za-z_][A-Za-z0-9_'.\u2080-\u2089\u2090-\u209c]*)"
)
_NAMESPACE_OPEN_RE = re.compile(r"^\s*namespace\s+([A-Za-z_][A-Za-z0-9_'.]*)")
_SECTION_OPEN_RE = re.compile(r"^\s*section\b(?:\s+([A-Za-z_][A-Za-z0-9_'.]*))?")
_END_RE = re.compile(r"^\s*end\b(?:\s+([A-Za-z_][A-Za-z0-9_'.]*))?")

# Identifier token pattern used when scanning a declaration's body for citations.
# Includes Unicode subscript ranges since real Lean/Mathlib names use them
# (e.g. "thmB\u2080_mult").
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_'.\u2080-\u2089\u2090-\u209c]*")

# Lean reserved words / tactics we never want to treat as "citations" of a library
# declaration even though they look like identifiers.
_KEYWORDS = {
    "theorem", "lemma", "def", "instance", "abbrev", "by", "do", "let", "have",
    "show", "from", "fun", "match", "with", "if", "then", "else", "where",
    "namespace", "end", "open", "import", "variable", "variables", "section",
    "noncomputable", "private", "protected", "mutual", "structure", "class",
    "deriving", "calc", "suffices", "exact", "apply", "intro",
    "intros", "rfl", "simp", "rw", "ring", "linarith", "nlinarith", "omega",
    "constructor", "cases", "rcases", "obtain", "induction", "unfold", "sorry",
    "axiom", "example", "universe", "set_option", "scoped", "local", "in",
    "using", "generalizing", "termination_by", "decreasing_by", "conv",
    "first", "try", "repeat", "all_goals", "any_goals", "focus", "case",
    "next", "assumption", "trivial", "contradiction", "tauto", "decide",
    "specialize", "refine", "use", "left", "right",
}


@dataclass
class Declaration:
    name: str                # fully-qualified name (namespace-prefixed), e.g. "Zeta23.thmB\u2080_mult"
    kind: str                # theorem | lemma | def | instance | abbrev
    module: str              # file path relative to library root, dot-separated
    raw_text: str            # full declaration source (header + statement + proof)
    short_name: str | None = None  # the un-namespaced header name; defaults to `name`
    citations: List[str] = field(default_factory=list)  # ordered, WITH multiplicity

    def __post_init__(self):
        if self.short_name is None:
            self.short_name = self.name


def _module_name_for_file(root: Path, file: Path) -> str:
    rel = file.relative_to(root).with_suffix("")
    return ".".join(rel.parts)


def parse_file(path: Path, module_root: Path) -> Dict[str, Declaration]:
    """Parse one .lean file into {fully_qualified_name: Declaration}.

    Tracks a `namespace ... end` stack (line by line) so declaration names are
    namespace-qualified the way real Lean/Mathlib code refers to them (e.g. a
    `theorem foo` inside `namespace Nat ... end Nat` becomes "Nat.foo", not
    "foo") -- this matters a great deal on real Mathlib, which declares almost
    everything inside namespace blocks. `section ... end` blocks are tracked
    too (to correctly pop `end` lines) but do NOT contribute to qualification,
    matching Lean's own semantics."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    module = _module_name_for_file(module_root, path)
    lines = text.split("\n")

    ns_stack: List[Tuple[str, str]] = []  # (tag, name), tag in {"namespace","section"}
    decl_starts: List[Tuple[int, str, str, str]] = []  # (line_idx, kind, qualified, short)

    for i, line in enumerate(lines):
        m_ns = _NAMESPACE_OPEN_RE.match(line)
        if m_ns:
            ns_stack.append(("namespace", m_ns.group(1)))
            continue
        m_sec = _SECTION_OPEN_RE.match(line)
        if m_sec:
            ns_stack.append(("section", m_sec.group(1) or ""))
            continue
        m_end = _END_RE.match(line)
        if m_end:
            if ns_stack:
                ns_stack.pop()
            continue
        m_decl = _DECL_HEADER_RE.match(line)
        if m_decl:
            kind, short_name = m_decl.group(1), m_decl.group(2)
            prefix_parts = [name for tag, name in ns_stack if tag == "namespace" and name]
            qualified = ".".join(prefix_parts + [short_name]) if prefix_parts else short_name
            decl_starts.append((i, kind, qualified, short_name))

    out: Dict[str, Declaration] = {}
    for idx, (line_idx, kind, qualified, short_name) in enumerate(decl_starts):
        end_line = decl_starts[idx + 1][0] if idx + 1 < len(decl_starts) else len(lines)
        chunk = "\n".join(lines[line_idx:end_line])
        out[qualified] = Declaration(name=qualified, short_name=short_name, kind=kind,
                                      module=module, raw_text=chunk)
    return out


def scan_lean_sources(path: str | Path) -> Dict[str, Declaration]:
    """Parse every .lean file under `path` into a name -> Declaration index.

    Accepts EITHER a directory (recursively scanned, e.g. a full Mathlib4
    checkout or an entire multi-file Lake project like a formal-math proof
    repo) OR a single .lean file, so the same function serves both
    `mathlib_dir` and the theorem project folder in the pipeline -- you can
    point this at whatever folder you unzipped/uploaded without restructuring
    it first. Module names are the file's path relative to `path`, dot-separated
    (e.g. a project at proofs/zeta23/ containing Zeta23/FinalMult.lean gets
    module "Zeta23.FinalMult"); declaration NAMES are namespace-qualified per
    parse_file above, independent of the file path."""
    p = Path(path)
    index: Dict[str, Declaration] = {}
    if p.is_dir():
        for lean_file in p.rglob("*.lean"):
            try:
                index.update(parse_file(lean_file, p))
            except (UnicodeDecodeError, OSError):
                continue
    elif p.is_file():
        index.update(parse_file(p, p.parent))
    else:
        raise FileNotFoundError(f"No such file or directory: {path}")
    return index


# Backward-compatible alias: mathlib_dir is scanned the same way as any other
# folder of .lean sources.
def build_declaration_index(mathlib_dir: str) -> Dict[str, Declaration]:
    return scan_lean_sources(mathlib_dir)


def build_suffix_index(names: Iterable[str]) -> Dict[str, List[str]]:
    """Maps every dot-suffix of every qualified name to the list of qualified
    names sharing that suffix, e.g. "Nat.Even.add" contributes suffixes "add",
    "Even.add", "Nat.Even.add". Used by compute_citations to resolve a
    citation written under an `open Namespace` (so the source text only has
    the short form, e.g. "Even.add" or just "add") back to its unique
    fully-qualified declaration -- this approximates Lean's namespace
    resolution without tracking `open` statements explicitly."""
    idx: Dict[str, List[str]] = defaultdict(list)
    for name in names:
        parts = name.split(".")
        for i in range(len(parts)):
            idx[".".join(parts[i:])].append(name)
    return idx


def compute_citations(index: Dict[str, Declaration], known_names: Set[str],
                       restrict_to: Iterable[str] | None = None,
                       suffix_index: Dict[str, List[str]] | None = None,
                       min_token_length_for_suffix_resolution: int = 4) -> None:
    """Fill in `citations` for declarations in `index`: the ordered,
    left-to-right, multiplicity-preserving sequence of *other known
    declaration names* that occur as identifier tokens in its raw text
    (Definition A.4's Cite(d), approximated from surface syntax -- see module
    docstring).

    `known_names` should be the FULL set of fully-qualified names across both
    the library and the theorem project, so citations of either are
    recognised correctly. `suffix_index` (from build_suffix_index over the
    same full name set), if given, additionally resolves an unqualified or
    partially-qualified token to its unique fully-qualified name when exactly
    one candidate exists; ambiguous short names (multiple declarations ending
    in the same suffix) are conservatively skipped rather than guessed.

    `min_token_length_for_suffix_resolution` guards against a specific false
    positive that shows up on real Mathlib: a short, generic bound-variable
    name (e.g. "a", "n", "h") can coincidentally be the *only* declaration in
    a 200k+-declaration library whose name happens to end in that exact
    suffix (e.g. a structure field literally named "a"), which would
    otherwise look "unambiguous" and get wrongly counted as a citation.
    Tokens shorter than this are only matched via the exact `known_names`
    check, never via suffix fallback, since genuine Mathlib declaration names
    are essentially never this short while bound/hypothesis variables often
    are.
    """
    targets = index.values() if restrict_to is None else (
        index[n] for n in restrict_to if n in index
    )
    for decl in targets:
        # Skip the header itself (first line) so a declaration doesn't cite itself
        # via its own name appearing in the signature.
        body = decl.raw_text.split("\n", 1)[1] if "\n" in decl.raw_text else ""
        cites: List[str] = []
        for tok in _IDENT_RE.findall(body):
            if tok == decl.name or tok == decl.short_name or tok in _KEYWORDS:
                continue
            if tok in known_names:
                cites.append(tok)
            elif suffix_index is not None and len(tok) >= min_token_length_for_suffix_resolution:
                candidates = suffix_index.get(tok)
                if candidates and len(candidates) == 1 and candidates[0] != decl.name:
                    cites.append(candidates[0])
        decl.citations = cites
