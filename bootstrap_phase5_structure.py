#!/usr/bin/env python3
"""
Bootstrap directory structure for Soft Spaces Phase 5.

Run this script from the project root, for example:

    C:/Users/tomst/PycharmProjects/SoftSpaces_Phase2

Command:

    python -X utf8 -u ./bootstrap_phase5_structure.py

The script is idempotent:
- existing directories are preserved
- existing files are not overwritten
"""

from pathlib import Path
import sys


EXPECTED_ROOT_NAME = "SoftSpaces_Phase2"

PHASE5_DIRS = [
    "Phase5",
    "Phase5/V_1_0",

    "Phase5/V_1_0/code",
    "Phase5/V_1_0/config",

    "Phase5/V_1_0/docs",
    "Phase5/V_1_0/docs/whitepapers",
    "Phase5/V_1_0/docs/preregistration",
    "Phase5/V_1_0/docs/methods",
    "Phase5/V_1_0/docs/closure",

    "Phase5/V_1_0/data",
    "Phase5/V_1_0/data/raw",
    "Phase5/V_1_0/data/external",
    "Phase5/V_1_0/data/processed",

    "Phase5/V_1_0/results",
    "Phase5/V_1_0/results/benchmarks",
    "Phase5/V_1_0/results/ground_truth",
    "Phase5/V_1_0/results/baselines",
    "Phase5/V_1_0/results/nulls",
    "Phase5/V_1_0/results/cost_recall",
    "Phase5/V_1_0/results/audits",

    "Phase5/V_1_0/figures",
    "Phase5/V_1_0/logs",
]


README_TEXT = """# Soft Spaces Phase 5 — Molecular Quantum Soft Spaces

## Purpose

Phase 5 returns Soft Spaces to its native quantum-physical domain.

The main research question is:

> Can Soft Spaces identify quantum-chemically difficult regions of molecular
> configuration space earlier or more efficiently than conventional low-cost
> indicators?

The intended benchmark sequence is:

1. Small molecular system with known quantum-chemical structure.
2. Label-/ground-truth-blind Soft-Spaces hotspot map.
3. Comparison with high-accuracy quantum-chemical reference.
4. Comparison with classical low-cost baselines.
5. Cost-versus-recall / search-space-reduction analysis.
6. Only after successful validation: a pharmaceutically relevant example.

## Scientific discipline

- preregister before scoring
- preserve negative results
- no post-hoc rescue
- compare against strong classical baselines
- distinguish statistical structure from practical advantage
- do not claim industrial value unless benchmarked quantitatively
"""


STRUCTURE_TEXT = """Soft Spaces Phase 5 / V_1_0

code/
    Executable Python programs.

config/
    Frozen configuration files and parameter manifests.

docs/
    whitepapers/       Phase 5 design and scientific rationale.
    preregistration/   Frozen hypotheses, gates and test protocols.
    methods/           Method descriptions and derivations.
    closure/           Formal phase/subphase closure documents.

data/
    raw/               Original local input data.
    external/          Downloaded public benchmark/reference data.
    processed/         Derived matrices and prepared benchmark data.

results/
    benchmarks/        Main benchmark outputs.
    ground_truth/      High-accuracy/reference quantum-chemistry outputs.
    baselines/         Conventional low-cost comparator results.
    nulls/             REAL-vs-NULL and matched-control results.
    cost_recall/       Search-space reduction / cost-versus-recall analyses.
    audits/            Technical and reproducibility audits.

figures/
    Publication and diagnostic figures.

logs/
    Execution logs.
"""


def write_if_missing(path: Path, content: str):
    if path.exists():
        print(f"KEEP   {path}")
        return
    path.write_text(content, encoding="utf-8")
    print(f"CREATE {path}")


def main():
    root = Path.cwd().resolve()

    if root.name.lower() != EXPECTED_ROOT_NAME.lower():
        print()
        print("STOP: Run this script from the SoftSpaces_Phase2 project root.")
        print()
        print("Current directory:")
        print(f"    {root}")
        print()
        print("Expected directory name:")
        print(f"    {EXPECTED_ROOT_NAME}")
        print()
        sys.exit(1)

    print("=" * 72)
    print("Soft Spaces Phase 5 bootstrap")
    print("Project root:")
    print(f"    {root}")
    print("=" * 72)

    for rel in PHASE5_DIRS:
        path = root / rel
        path.mkdir(parents=True, exist_ok=True)
        print(f"DIR    {path}")

    phase5_root = root / "Phase5" / "V_1_0"

    write_if_missing(
        phase5_root / "README.md",
        README_TEXT,
    )

    write_if_missing(
        phase5_root / "STRUCTURE.txt",
        STRUCTURE_TEXT,
    )

    # Keep otherwise-empty directories visible to Git.
    for rel in PHASE5_DIRS:
        path = root / rel
        if path == root / "Phase5":
            continue
        if not any(path.iterdir()):
            (path / ".gitkeep").touch(exist_ok=True)

    print()
    print("=" * 72)
    print("Phase 5 directory structure is ready.")
    print("=" * 72)
    print()
    print("Created root:")
    print(f"    {phase5_root}")
    print()
    print("Next recommended step:")
    print("    docs/whitepapers/Phase5_Molecular_Quantum_Soft_Spaces_Whitepaper.pdf")
    print()


if __name__ == "__main__":
    main()
