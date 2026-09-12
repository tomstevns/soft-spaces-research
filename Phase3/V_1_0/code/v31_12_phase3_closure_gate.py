#!/usr/bin/env python3
"""Soft Spaces Phase 3 Closure v31.12 — final closure gate.

Reads v31.9, v31.10 and v31.11 JSON summaries and emits the final Phase-3
closure decision. No new physical calculation is performed here.

Decision rule
-------------
PASS            : all three component tests are PASS.
PASS WITH SCOPE : no component is FAIL and at least one is PASS WITH SCOPE.
FAIL            : at least one component test is FAIL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VERSION = "v31.12"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hamiltonian", type=Path, default=Path("v31_9_hamiltonian_robustness.json"))
    parser.add_argument("--null", type=Path, default=Path("v31_10_null_sensitivity.json"))
    parser.add_argument("--perturbation", type=Path, default=Path("v31_11_perturbation_robustness.json"))
    parser.add_argument("--output", type=Path, default=Path("v31_12_phase3_closure_output.txt"))
    parser.add_argument("--json", type=Path, default=Path("v31_12_phase3_closure.json"))
    args = parser.parse_args()

    docs = {
        "hamiltonian_robustness": json.loads(args.hamiltonian.read_text(encoding="utf-8")),
        "null_sensitivity": json.loads(args.null.read_text(encoding="utf-8")),
        "perturbation_robustness": json.loads(args.perturbation.read_text(encoding="utf-8")),
    }
    decisions = {name: doc["decision"] for name, doc in docs.items()}

    if any(value == "FAIL" for value in decisions.values()):
        final = "FAIL"
    elif all(value == "PASS" for value in decisions.values()):
        final = "PASS"
    else:
        final = "PASS WITH SCOPE"

    if final == "PASS":
        interpretation = (
            "Phase 3 is CLOSED. The invariant SoftSpace structure is robust across the "
            "predeclared Hamiltonian, NULL and perturbation variations tested here."
        )
    elif final == "PASS WITH SCOPE":
        interpretation = (
            "Phase 3 is CLOSED WITH EXPLICIT SCOPE. The core structure survives the closure "
            "program, but at least one robustness dimension is restricted and must be stated "
            "explicitly in the final Phase-3 article."
        )
    else:
        interpretation = (
            "Phase 3 is NOT CLOSED by the predeclared gate. At least one robustness dimension "
            "failed and should be understood before operational Phase-4 claims are made."
        )

    lines = [
        "=== Soft Spaces Phase 3 Closure v31.12 — FINAL GATE ===",
        f"Hamiltonian robustness: {decisions['hamiltonian_robustness']}",
        f"NULL sensitivity: {decisions['null_sensitivity']}",
        f"Perturbation robustness: {decisions['perturbation_robustness']}",
        "",
        f"FINAL PHASE 3 CLOSURE DECISION: {final}",
        "",
        interpretation,
        "",
        "Interpretation lock:",
        "  The projector score quantifies perturbative coupling/cancellation associated with",
        "  complete exact eigenspaces. The observed ±E organization is treated as a spectral",
        "  organization of that score. REAL-vs-NULL differences establish structure relative",
        "  to the tested controls; they do not establish universality, experimental realization,",
        "  or practical quantum advantage.",
    ]
    report = "\n".join(lines) + "\n"
    print(report, end="")
    args.output.write_text(report, encoding="utf-8")
    args.json.write_text(json.dumps({
        "version": VERSION,
        "component_decisions": decisions,
        "final_decision": final,
        "interpretation": interpretation,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
