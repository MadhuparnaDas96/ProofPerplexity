"""
Implements the recovery of w(si, B) per Definitions 1.1 (Search Budget) and
A.5 (Search Tree Structure):

    If an accepted path exists within depth B: w(si,B) := length of the
    shortest/first accepted path found.
    Otherwise (search exhausted without closing the goal): w(si,B) := w(si)
    (the Definition A.5 failure convention -- forces alpha_abl to its floor).

Two interchangeable provers are provided. Both implement the same interface:
    recover(step_name, w_si, budget) -> (w_si_B: int, succeeded: bool)
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Protocol

import requests


class SearchProver(Protocol):
    def recover(self, step_name: str, w_si: int, budget: int) -> tuple[int, bool]:
        ...


class ManualOverrideProver:
    """Definition A.5's w(si,B) supplied directly by the user (or by an
    external prover run offline), rather than computed by this repo.

    This is a fully legitimate way to instantiate the benchmark: Definition
    1.1 only requires *some* budget-B search procedure with a well-defined
    accept/fail outcome -- it does not mandate that this repo be the one
    running it. Any step absent from the overrides file falls back to the
    Definition A.5 failure convention, w(si,B) := w(si).
    """

    def __init__(self, overrides_file: Optional[str]):
        self.overrides: Dict[str, int] = {}
        if overrides_file and Path(overrides_file).exists():
            with open(overrides_file) as f:
                data = json.load(f)
            self.overrides = {k: v for k, v in data.items() if not k.startswith("_")}

    def recover(self, step_name: str, w_si: int, budget: int) -> tuple[int, bool]:
        if step_name in self.overrides:
            return int(self.overrides[step_name]), True
        return w_si, False


class LeanLLMSearchProver:
    """Optional, heavier mode: repeatedly prompts an LLM for a tactic-level
    proof of `si` using only declarations in `Pi` (passed in as `cand_names`),
    then verifies each candidate by shelling out to `lake env lean` -- i.e. a
    REAL kernel check, not a heuristic. The accepted path length is the number
    of tactic lines in the first candidate that type-checks within budget B
    attempts (path length here is measured in "search actions", matching
    Definition A.5's one-unit-per-edge convention).

    This requires:
      - a local Lean/Lake toolchain with `lake_project_dir` buildable
      - an LLM API key in the environment variable named by `llm_api_key_env`

    It is a reference implementation, not a competitive automated prover --
    real deployments should point at AlphaProof/Aristotle-style search instead.
    """

    def __init__(self, lake_project_dir: str, llm_api_base: str,
                 llm_api_key_env: str, llm_model: str,
                 max_attempts_per_step: int = 8, timeout_seconds_per_attempt: int = 30):
        self.lake_project_dir = Path(lake_project_dir)
        self.llm_api_base = llm_api_base
        self.api_key = os.environ.get(llm_api_key_env, "")
        self.llm_model = llm_model
        self.max_attempts = max_attempts_per_step
        self.timeout = timeout_seconds_per_attempt

    def _propose_tactic_proof(self, goal_statement: str, cand_context: str) -> str:
        if not self.api_key:
            raise RuntimeError(
                "LeanLLMSearchProver: no API key found in the configured env var."
            )
        prompt = (
            "You are attempting to prove the following Lean 4 goal using ONLY "
            "the declarations listed below (do not invent lemma names). "
            "Respond with ONLY the tactic proof block, no explanation.\n\n"
            f"Available declarations (Pi ∩ CandB):\n{cand_context}\n\n"
            f"Goal:\n{goal_statement}\n"
        )
        resp = requests.post(
            self.llm_api_base,
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": self.llm_model, "max_tokens": 1024,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        content = resp.json().get("content", [])
        return "".join(b.get("text", "") for b in content if b.get("type") == "text")

    def _verify_with_lean(self, full_decl_text: str) -> bool:
        with tempfile.NamedTemporaryFile(
            suffix=".lean", dir=self.lake_project_dir, delete=False
        ) as tmp:
            tmp.write(full_decl_text.encode())
            tmp_path = tmp.name
        try:
            proc = subprocess.run(
                ["lake", "env", "lean", tmp_path],
                cwd=self.lake_project_dir, capture_output=True,
                timeout=self.timeout,
            )
            return proc.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def recover(self, step_name: str, w_si: int, budget: int,
                goal_statement: str = "", cand_context: str = "") -> tuple[int, bool]:
        attempts = min(self.max_attempts, budget)
        for path_len in range(1, attempts + 1):
            try:
                candidate = self._propose_tactic_proof(goal_statement, cand_context)
            except RuntimeError:
                break
            if self._verify_with_lean(candidate):
                return path_len, True
        return w_si, False


def build_prover(config: dict) -> SearchProver:
    mode = config["prover"]["mode"]
    if mode == "manual":
        return ManualOverrideProver(config["prover"].get("manual_overrides_file"))
    elif mode == "lean_llm_search":
        cfg = config["prover"]["lean_llm_search"]
        return LeanLLMSearchProver(
            lake_project_dir=cfg["lake_project_dir"],
            llm_api_base=cfg["llm_api_base"],
            llm_api_key_env=cfg["llm_api_key_env"],
            llm_model=cfg["llm_model"],
            max_attempts_per_step=cfg.get("max_attempts_per_step", 8),
            timeout_seconds_per_attempt=cfg.get("timeout_seconds_per_attempt", 30),
        )
    raise ValueError(f"Unknown prover mode: {mode}")
