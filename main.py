#!/usr/bin/env python3
"""CLI for the proof non-triviality benchmark."""
from __future__ import annotations

import argparse
import sys

import yaml

from src.lean_parser import scan_lean_sources
from src.pipeline import run, write_results


def _print_report(result: dict) -> None:
    print(f"\nTarget theorem: {result['target_theorem']}")
    print(f"alpha_min = {result['alpha_min']:.4f}   kappa_max = {result['kappa_max']:.4f}"
          f"   B = {result['B']}   theta = {result['theta']}\n")

    header = f"{'step':<30}{'w(si)':>8}{'w(si,B)':>10}{'a_abl':>9}{'a_mah':>9}{'a(si,B)':>10}{'Surp':>8}"
    print(header)
    print("-" * len(header))
    for s in result["steps"]:
        print(f"{s['name']:<30}{s['w_si']:>8}{s['w_si_B']:>10}"
              f"{s['alpha_abl']:>9.4f}{s['alpha_mah']:>9.4f}"
              f"{s['alpha_combined']:>10.4f}{s['surprisal']:>8.4f}")

    print(f"\nProof perplexity PP_w(pi,B)        = {result['proof_perplexity']:.4f}")
    print(f"theta-invention fraction beta(theta) = {result['theta_invention_fraction']:.4f}")
    if result["theorem_2_1_lower_bound"] is not None:
        print(f"Theorem 2.1 lower bound on beta     = {result['theorem_2_1_lower_bound']:.4f}")
        status = "HOLDS" if result["theorem_2_1_bound_holds"] else "VIOLATED (bug!)"
        print(f"Bound check: {status}")
    else:
        print("Theorem 2.1 hypothesis (log PP_w > kappa(theta)) not satisfied; "
              "bound is vacuous for this run.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Proof non-triviality benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the full benchmark pipeline")
    run_p.add_argument("--config", default="config.yaml")

    find_p = sub.add_parser(
        "find", help="List declaration names parsed from a folder (useful for "
                      "getting target_theorem_name exactly right, including "
                      "unicode subscripts)")
    find_p.add_argument("--dir", required=True, help="Folder (or single .lean file) to scan")
    find_p.add_argument("--query", default="", help="Only show names containing this substring")

    args = parser.parse_args()

    if args.command == "find":
        decls = scan_lean_sources(args.dir)
        names = sorted(n for n in decls if args.query in n)
        if not names:
            print(f"No declarations matched {args.query!r} in {args.dir}")
            return 0
        for n in names:
            print(f"{n:<50} [{decls[n].module}]")
        print(f"\n{len(names)} declaration(s) shown out of {len(decls)} parsed total.")
        return 0

    if args.command == "run":
        with open(args.config) as f:
            config = yaml.safe_load(f)
        result = run(config)
        out_path = write_results(result, config["output_dir"])
        _print_report(result)
        print(f"\nFull results written to {out_path}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
