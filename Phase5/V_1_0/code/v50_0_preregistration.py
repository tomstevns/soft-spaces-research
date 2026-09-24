#!/usr/bin/env python3
"""
Soft Spaces Phase 5 — v50_0 preregistration

Purpose
-------
Freeze the scientific design of the first Phase 5 molecular benchmark BEFORE
running outcome-generating simulations.

Primary benchmark
-----------------
Linear H4 in the STO-3G basis.

Scientific question
-------------------
Can Soft Spaces provide predictive or computational value in a molecular
quantum problem by prioritising physically useful determinants/subspaces
better than preregistered baselines?

This script does NOT run molecular calculations.
It writes an immutable-style JSON manifest plus a SHA-256 digest so later
Phase 5 scripts can verify that the preregistration has not drifted.

Usage
-----
    python -X utf8 -u v50_0_preregistration.py

Optional:
    python -X utf8 -u v50_0_preregistration.py --output-dir ..\\results\\preregistration

Expected output
---------------
    phase5_v50_0_preregistration.json
    phase5_v50_0_preregistration.sha256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "v50_0"
PHASE = "Phase 5"
PROJECT = "Molecular Quantum Soft Spaces"
SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Frozen scientific design
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MolecularBenchmark:
    molecule: str
    geometry_family: str
    basis: str
    number_of_atoms: int
    spatial_orbitals_expected: int
    spin_orbitals_expected: int
    electron_count: int
    charge: int
    spin_multiplicity: int
    geometry_parameter: str
    geometry_scan_angstrom: tuple[float, ...]
    primary_geometry_angstrom: float
    rationale: tuple[str, ...]


@dataclass(frozen=True)
class SoftSpacesHypothesis:
    primary_hypothesis: str
    null_hypothesis: str
    prediction_target: str
    real_definition: str
    null_definition: str
    leakage_rule: str


@dataclass(frozen=True)
class Baseline:
    name: str
    description: str
    mandatory: bool = True


@dataclass(frozen=True)
class Metric:
    name: str
    direction: str
    description: str
    primary: bool = False


@dataclass(frozen=True)
class DecisionGate:
    gate_id: str
    name: str
    rule: str
    mandatory: bool = True


@dataclass(frozen=True)
class SeedPolicy:
    master_seed: int
    evaluation_seeds: tuple[int, ...]
    notes: tuple[str, ...]


BENCHMARK = MolecularBenchmark(
    molecule="H4",
    geometry_family="linear equally spaced hydrogen chain",
    basis="STO-3G",
    number_of_atoms=4,
    spatial_orbitals_expected=4,
    spin_orbitals_expected=8,
    electron_count=4,
    charge=0,
    spin_multiplicity=1,
    geometry_parameter="nearest-neighbour H-H distance",
    geometry_scan_angstrom=(
        0.75,
        1.00,
        1.25,
        1.50,
        1.75,
        2.00,
        2.50,
        3.00,
    ),
    primary_geometry_angstrom=1.50,
    rationale=(
        "Small enough for exact/reference calculations in the initial Phase 5 study.",
        "Less trivial than H2 because electron correlation changes materially with geometry.",
        "Eight spin orbitals provide a natural first molecular bridge to the earlier "
        "8-qubit Soft Spaces work.",
        "The geometry scan allows testing whether any Soft Spaces advantage survives "
        "from relatively weak to stronger correlation regimes.",
    ),
)

HYPOTHESIS = SoftSpacesHypothesis(
    primary_hypothesis=(
        "A preregistered Soft Spaces score/ranking will prioritise determinants or "
        "candidate subspaces that carry more molecular wavefunction/energy relevance "
        "than matched NULL controls and mandatory baselines at the same selection budget."
    ),
    null_hypothesis=(
        "At matched selection budget, Soft Spaces provides no reproducible advantage "
        "over matched NULL controls and preregistered baselines."
    ),
    prediction_target=(
        "Rank candidate computational-basis determinants/subspaces before using the "
        "target molecular outcome to choose among them."
    ),
    real_definition=(
        "REAL uses the frozen Soft Spaces construction inherited from the validated "
        "earlier-phase method, applied without outcome-dependent retuning."
    ),
    null_definition=(
        "NULL uses matched controls that preserve the relevant problem size and selection "
        "budget while destroying the specific Soft Spaces structure being tested."
    ),
    leakage_rule=(
        "No ranking hyperparameter, threshold, geometry choice, K value, seed subset, "
        "or metric definition may be changed after inspecting target benchmark outcomes. "
        "Any post-hoc analysis must be labelled exploratory."
    ),
)

BASELINES = (
    Baseline(
        name="Random-K",
        description=(
            "Uniform random selection of K candidate determinants/subspaces from the same "
            "eligible candidate set. Report distribution across preregistered seeds."
        ),
    ),
    Baseline(
        name="Reference-amplitude upper bound",
        description=(
            "Outcome-aware ranking by exact/reference wavefunction importance. This is NOT "
            "a deployable competitor; it is an oracle ceiling used to quantify headroom."
        ),
    ),
    Baseline(
        name="Simple chemistry-informed baseline",
        description=(
            "A preregistered low-cost ranking using only conventional molecular information "
            "available before the target comparison. Exact implementation is frozen in the "
            "first executable benchmark script and may not use target labels."
        ),
    ),
    Baseline(
        name="Matched NULL Soft-Spaces control",
        description=(
            "A control that preserves candidate count, K, geometry, and evaluation protocol "
            "while removing the specific REAL Soft Spaces structure."
        ),
    ),
)

METRICS = (
    Metric(
        name="Captured probability mass",
        direction="higher_is_better",
        description=(
            "Sum of reference-state probability mass contained in the selected K "
            "determinants/subspace."
        ),
        primary=True,
    ),
    Metric(
        name="Energy error after selected-subspace diagonalisation",
        direction="lower_is_better",
        description=(
            "Absolute ground-state energy error relative to the exact/reference value "
            "when diagonalising in the selected subspace."
        ),
        primary=True,
    ),
    Metric(
        name="Precision@K",
        direction="higher_is_better",
        description=(
            "Fraction of selected candidates belonging to a preregistered set of "
            "reference-important determinants."
        ),
    ),
    Metric(
        name="Recall@K",
        direction="higher_is_better",
        description=(
            "Fraction of preregistered reference-important determinants recovered "
            "within the selected K."
        ),
    ),
    Metric(
        name="NDCG@K",
        direction="higher_is_better",
        description=(
            "Rank-sensitive agreement with reference importance ranking."
        ),
    ),
    Metric(
        name="Selection efficiency",
        direction="higher_is_better",
        description=(
            "Reference quality achieved per selected determinant / evaluated candidate."
        ),
    ),
)

K_VALUES = (2, 4, 8, 12, 16)

DECISION_GATES = (
    DecisionGate(
        gate_id="G1",
        name="Validity",
        rule=(
            "Reference molecular calculation, candidate indexing, electron-number sector, "
            "and all bookkeeping checks must pass. Any indexing or sector mismatch is FAIL."
        ),
    ),
    DecisionGate(
        gate_id="G2",
        name="REAL beats matched NULL on primary metrics",
        rule=(
            "Across the preregistered geometry scan, REAL must show positive median advantage "
            "over matched NULL for BOTH primary metrics: captured probability mass and "
            "absolute energy error."
        ),
    ),
    DecisionGate(
        gate_id="G3",
        name="Advantage is not single-geometry",
        rule=(
            "REAL must outperform the matched NULL median on both primary metrics in at "
            "least 6 of the 8 preregistered geometries for at least one preregistered K."
        ),
    ),
    DecisionGate(
        gate_id="G4",
        name="Random baseline separation",
        rule=(
            "For the same qualifying K from G3, REAL must exceed the Random-K median "
            "captured probability mass and improve median absolute energy error."
        ),
    ),
    DecisionGate(
        gate_id="G5",
        name="Practical reduction",
        rule=(
            "A qualifying REAL configuration must reach a preregistered reference-quality "
            "target using no more than 50% of the eligible determinant set. The exact "
            "reference-quality threshold is frozen before outcome inspection in v50_1."
        ),
    ),
    DecisionGate(
        gate_id="G6",
        name="Reproducibility",
        rule=(
            "The primary conclusion must remain unchanged under the full preregistered seed "
            "set. A result depending on selective seed removal is FAIL."
        ),
    ),
)

SEEDS = SeedPolicy(
    master_seed=25043000,
    evaluation_seeds=tuple(range(25043000, 25043012)),
    notes=(
        "Reuse the established 12-seed family for continuity with earlier Soft Spaces work.",
        "Seeds are frozen before Phase 5 outcome generation.",
        "No failed-looking seed may be removed without an independently documented technical reason.",
    ),
)

INDUSTRIAL_VALUE = {
    "definition": (
        "Industrial/practical value means measurable reduction of computational search or "
        "evaluation cost at matched scientific quality, not merely statistically significant "
        "REAL-vs-NULL separation."
    ),
    "acceptable_signals": [
        "Fewer selected determinants for the same energy accuracy.",
        "Higher captured reference mass at the same K.",
        "Fewer expensive downstream evaluations at matched quality.",
        "Stable prioritisation across molecular geometries.",
        "A ranking rule that can be computed more cheaply than the brute-force work it replaces.",
    ],
    "non_claims": [
        "Phase 5 v50_0 does not claim quantum advantage.",
        "Phase 5 v50_0 does not claim industrial deployment readiness.",
        "Phase 5 v50_0 does not claim transfer to chemically large molecules.",
        "A REAL-vs-NULL difference alone is insufficient for an industrial-value claim.",
    ],
}

PHASE5_SEQUENCE = (
    {
        "version": "v50_0",
        "purpose": "Preregistration and manifest freeze",
    },
    {
        "version": "v50_1",
        "purpose": (
            "Reference H4/STO-3G molecular problem construction; exact/reference spectrum "
            "and determinant-sector validation; freeze any remaining purely operational thresholds"
        ),
    },
    {
        "version": "v50_2",
        "purpose": "Implement frozen REAL and matched NULL molecular ranking",
    },
    {
        "version": "v50_3",
        "purpose": "Run mandatory baselines and K-budget sweep",
    },
    {
        "version": "v50_4",
        "purpose": "Geometry robustness and correlation-regime analysis",
    },
    {
        "version": "v50_5",
        "purpose": "Resource/efficiency analysis and practical-value gate",
    },
    {
        "version": "v50_6",
        "purpose": "Independent confirmation / secondary molecule candidate such as LiH",
    },
)


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------

def build_manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "project": PROJECT,
        "phase": PHASE,
        "script_version": SCRIPT_VERSION,
        "status": "PREREGISTERED_BEFORE_OUTCOME_GENERATION",
        "scientific_scope": {
            "core_question": (
                "Can previously established Soft Spaces structure provide predictive or "
                "computational value in a molecular quantum benchmark?"
            ),
            "phase5_non_goal": (
                "Phase 5 is not designed merely to demonstrate that Soft Spaces exist."
            ),
        },
        "benchmark": asdict(BENCHMARK),
        "hypothesis": asdict(HYPOTHESIS),
        "candidate_selection": {
            "selection_object": "computational-basis determinants / selected determinant subspace",
            "k_values": list(K_VALUES),
            "budget_matching_required": True,
            "electron_number_sector_must_be_preserved": True,
        },
        "baselines": [asdict(x) for x in BASELINES],
        "metrics": [asdict(x) for x in METRICS],
        "decision_gates": [asdict(x) for x in DECISION_GATES],
        "seed_policy": asdict(SEEDS),
        "industrial_value": INDUSTRIAL_VALUE,
        "planned_sequence": list(PHASE5_SEQUENCE),
        "analysis_policy": {
            "primary_analysis": "preregistered only",
            "exploratory_analysis_allowed": True,
            "exploratory_labelling_required": True,
            "post_hoc_threshold_changes_prohibited_for_primary_claims": True,
            "report_all_preregistered_geometries": True,
            "report_all_preregistered_k_values": True,
            "report_all_preregistered_seeds": True,
        },
        "software_policy": {
            "python": "3.12 preferred",
            "reproducible_outputs": True,
            "machine_readable_results": True,
            "no_silent_exception_suppression": True,
            "future_scripts_must_verify_preregistration_hash": True,
        },
    }


def canonical_json_bytes(manifest: dict[str, Any]) -> bytes:
    """Return deterministic UTF-8 JSON bytes used for hashing."""
    text = json.dumps(
        manifest,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return text.encode("utf-8")


def sha256_hex(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def write_outputs(output_dir: Path) -> tuple[Path, Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest()
    digest = sha256_hex(manifest)

    human_readable = {
        "_metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "canonical_sha256": digest,
            "hash_scope": (
                "SHA-256 is computed over canonical JSON of the scientific manifest only; "
                "runtime metadata is excluded from the hash."
            ),
        },
        "manifest": manifest,
    }

    json_path = output_dir / "phase5_v50_0_preregistration.json"
    hash_path = output_dir / "phase5_v50_0_preregistration.sha256"

    json_path.write_text(
        json.dumps(human_readable, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    hash_path.write_text(digest + "\n", encoding="ascii")

    return json_path, hash_path, digest


def verify_outputs(json_path: Path, expected_digest: str) -> bool:
    """
    Verify that the scientific manifest embedded in the written JSON has the
    expected canonical SHA-256.
    """
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    manifest = loaded["manifest"]
    actual_digest = sha256_hex(manifest)
    return actual_digest == expected_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze Soft Spaces Phase 5 v50_0 preregistration."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "preregistration",
        help=(
            "Directory for the preregistration manifest. Default assumes this script "
            "lives in Phase5/V_1_0/code."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    json_path, hash_path, digest = write_outputs(args.output_dir)

    verified = verify_outputs(json_path, digest)
    if not verified:
        print("ERROR: preregistration hash verification failed.", file=sys.stderr)
        return 2

    manifest = build_manifest()

    print("=" * 78)
    print("SOFT SPACES — PHASE 5 MOLECULAR QUANTUM PREREGISTRATION")
    print("=" * 78)
    print(f"Version       : {SCRIPT_VERSION}")
    print(f"Benchmark     : {BENCHMARK.molecule} / {BENCHMARK.basis}")
    print(f"Spin orbitals : {BENCHMARK.spin_orbitals_expected}")
    print(f"Electrons     : {BENCHMARK.electron_count}")
    print(f"Geometries    : {len(BENCHMARK.geometry_scan_angstrom)}")
    print(f"K values      : {K_VALUES}")
    print(f"Seeds         : {len(SEEDS.evaluation_seeds)}")
    print(f"Decision gates: {len(DECISION_GATES)}")
    print("-" * 78)
    print(f"Manifest      : {json_path}")
    print(f"SHA-256 file  : {hash_path}")
    print(f"SHA-256       : {digest}")
    print(f"Verified      : {'YES' if verified else 'NO'}")
    print("-" * 78)
    print("NO molecular outcome calculation was executed.")
    print("Scientific design is now machine-readable and hash-verifiable.")
    print("=" * 78)

    # Deliberate additional self-checks.
    assert manifest["status"] == "PREREGISTERED_BEFORE_OUTCOME_GENERATION"
    assert len(SEEDS.evaluation_seeds) == 12
    assert BENCHMARK.spin_orbitals_expected == 8
    assert len(BENCHMARK.geometry_scan_angstrom) == 8
    assert len(DECISION_GATES) == 6

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
