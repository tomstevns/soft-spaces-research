#!/usr/bin/env python3
"""
Soft Spaces Phase 5 — v50_1 H4 reference problem

Purpose
-------
Construct and validate the preregistered H4/STO-3G molecular benchmark for
Phase 5, then compute exact reference ground-state data in the fixed
four-electron sector.

This script is outcome-generating, but it does NOT yet run any Soft Spaces
REAL/NULL ranking. Its job is to establish the reference problem against which
later Phase 5 selection methods will be evaluated.

Preregistered benchmark inherited from v50_0
---------------------------------------------
Molecule:       linear H4
Basis:          STO-3G
Charge:         0
Spin:           singlet (PySCF spin=0)
Electrons:      4
Spin orbitals:  expected 8
Geometries Å:   0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00

Method
------
1. Build each molecular geometry with Qiskit Nature / PySCF.
2. Extract the second-quantized electronic Hamiltonian.
3. Map it to qubits with Jordan-Wigner.
4. Form the full 2^8 Hamiltonian matrix.
5. Restrict it explicitly to the fixed N=4 particle-number sector.
6. Exactly diagonalize that sector with NumPy.
7. Add nuclear repulsion to obtain total ground-state energy.
8. Save sector basis, amplitudes, probabilities, and summary diagnostics.

No qubit tapering is used here. The purpose is to keep the full 8-spin-orbital
representation aligned with the Phase 5 preregistration.

Usage
-----
    python -X utf8 -u .\v50_1_h4_reference_problem.py

Optional:
    python -X utf8 -u .\v50_1_h4_reference_problem.py --output-dir ..\results\reference

Dependencies
------------
    numpy
    qiskit
    qiskit-nature
    pyscf

The script checks the v50_0 preregistration manifest when it is available.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_VERSION = "v50_1"
PROJECT = "Molecular Quantum Soft Spaces"
PHASE = "Phase 5"

EXPECTED_MOLECULE = "H4"
EXPECTED_BASIS = "STO-3G"
EXPECTED_NUM_ELECTRONS = 4
EXPECTED_NUM_SPIN_ORBITALS = 8
EXPECTED_NUM_SPATIAL_ORBITALS = 4
EXPECTED_NUM_ALPHA = 2
EXPECTED_NUM_BETA = 2
EXPECTED_CHARGE = 0
EXPECTED_SPIN = 0

GEOMETRIES_ANGSTROM = (
    0.75,
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
    2.50,
    3.00,
)

PRIMARY_GEOMETRY_ANGSTROM = 1.50

DEFAULT_PREREGISTRATION_FILENAME = "phase5_v50_0_preregistration.json"
DEFAULT_PREREGISTRATION_HASH_FILENAME = "phase5_v50_0_preregistration.sha256"


@dataclass(frozen=True)
class GeometryResult:
    spacing_angstrom: float
    atom_specification: str
    num_spatial_orbitals: int
    num_spin_orbitals: int
    num_alpha: int
    num_beta: int
    num_particles_total: int
    full_hilbert_dimension: int
    n4_sector_dimension: int
    electronic_ground_energy_hartree: float
    nuclear_repulsion_energy_hartree: float
    total_ground_energy_hartree: float
    ground_state_norm: float
    particle_number_expectation: float
    max_probability: float
    max_probability_basis_index: int
    max_probability_bitstring_qiskit: str
    significant_determinants_p1e6: int
    significant_determinants_p1e4: int
    participation_ratio: float


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def require_dependencies() -> None:
    missing = []

    if package_version("qiskit") is None:
        missing.append("qiskit")
    if package_version("qiskit-nature") is None:
        missing.append("qiskit-nature")
    if package_version("pyscf") is None:
        missing.append("pyscf")

    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            "Missing required package(s): "
            f"{names}\n\n"
            "Install into the active virtual environment, for example:\n"
            "    python -m pip install qiskit qiskit-nature pyscf\n"
        )


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_hex(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def find_preregistration(default_output_dir: Path) -> tuple[Path | None, Path | None]:
    """
    Look for the v50_0 preregistration relative to Phase5/V_1_0.

    Expected layout:
        code/
        results/preregistration/
        results/reference/
    """
    v1_root = Path(__file__).resolve().parent.parent
    prereg_dir = v1_root / "results" / "preregistration"

    json_path = prereg_dir / DEFAULT_PREREGISTRATION_FILENAME
    hash_path = prereg_dir / DEFAULT_PREREGISTRATION_HASH_FILENAME

    if json_path.exists():
        return json_path, hash_path if hash_path.exists() else None

    # Fallback: a caller may explicitly place v50_1 elsewhere.
    fallback_json = default_output_dir.parent / "preregistration" / DEFAULT_PREREGISTRATION_FILENAME
    fallback_hash = default_output_dir.parent / "preregistration" / DEFAULT_PREREGISTRATION_HASH_FILENAME

    if fallback_json.exists():
        return fallback_json, fallback_hash if fallback_hash.exists() else None

    return None, None


def verify_preregistration(
    prereg_json_path: Path | None,
    prereg_hash_path: Path | None,
) -> dict[str, Any]:
    """
    Verify the scientific manifest hash and key frozen benchmark settings.

    Returns a compact verification report.
    """
    if prereg_json_path is None:
        return {
            "found": False,
            "verified": False,
            "message": "v50_0 preregistration manifest not found.",
        }

    loaded = json.loads(prereg_json_path.read_text(encoding="utf-8"))

    if "manifest" not in loaded:
        raise RuntimeError(
            f"Invalid preregistration file: missing 'manifest': {prereg_json_path}"
        )

    manifest = loaded["manifest"]
    calculated_hash = sha256_hex(manifest)

    embedded_hash = (
        loaded.get("_metadata", {}).get("canonical_sha256")
    )

    stored_hash = None
    if prereg_hash_path is not None:
        stored_hash = prereg_hash_path.read_text(encoding="ascii").strip()

    hash_checks = [
        embedded_hash == calculated_hash if embedded_hash is not None else True,
        stored_hash == calculated_hash if stored_hash is not None else True,
    ]

    if not all(hash_checks):
        raise RuntimeError(
            "v50_0 preregistration hash verification FAILED.\n"
            f"Calculated : {calculated_hash}\n"
            f"Embedded   : {embedded_hash}\n"
            f"Stored     : {stored_hash}"
        )

    benchmark = manifest.get("benchmark", {})

    expected_geometry = [float(x) for x in GEOMETRIES_ANGSTROM]
    actual_geometry = [float(x) for x in benchmark.get("geometry_scan_angstrom", [])]

    frozen_checks = {
        "molecule": benchmark.get("molecule") == EXPECTED_MOLECULE,
        "basis": str(benchmark.get("basis", "")).upper() == EXPECTED_BASIS,
        "electron_count": int(benchmark.get("electron_count", -1)) == EXPECTED_NUM_ELECTRONS,
        "spin_orbitals_expected": int(
            benchmark.get("spin_orbitals_expected", -1)
        ) == EXPECTED_NUM_SPIN_ORBITALS,
        "geometry_scan_angstrom": actual_geometry == expected_geometry,
        "primary_geometry_angstrom": math.isclose(
            float(benchmark.get("primary_geometry_angstrom", float("nan"))),
            PRIMARY_GEOMETRY_ANGSTROM,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
    }

    if not all(frozen_checks.values()):
        failed = [name for name, ok in frozen_checks.items() if not ok]
        raise RuntimeError(
            "v50_0 preregistration content does not match v50_1 frozen expectations. "
            f"Failed checks: {failed}"
        )

    return {
        "found": True,
        "verified": True,
        "path": str(prereg_json_path),
        "hash_path": str(prereg_hash_path) if prereg_hash_path else None,
        "canonical_sha256": calculated_hash,
        "frozen_checks": frozen_checks,
    }


def h4_linear_atom_spec(spacing_angstrom: float) -> str:
    """
    Equally spaced linear H4 chain along z.

    Positions are centered around z=0:
        -1.5d, -0.5d, +0.5d, +1.5d
    """
    positions = (
        -1.5 * spacing_angstrom,
        -0.5 * spacing_angstrom,
        +0.5 * spacing_angstrom,
        +1.5 * spacing_angstrom,
    )
    return "; ".join(f"H 0.0 0.0 {z:.12f}" for z in positions)


def basis_indices_with_particle_number(
    num_qubits: int,
    num_particles: int,
) -> np.ndarray:
    """
    Computational basis indices with exact Hamming weight num_particles.

    Under Jordan-Wigner, occupation number equals qubit Hamming weight.
    """
    return np.array(
        [i for i in range(1 << num_qubits) if i.bit_count() == num_particles],
        dtype=np.int64,
    )


def bitstring_qiskit(index: int, num_qubits: int) -> str:
    """
    Return the conventional printed computational-basis bitstring |q_{n-1}...q_0>.
    """
    return format(index, f"0{num_qubits}b")


def occupation_vector_little_endian(index: int, num_qubits: int) -> list[int]:
    """
    Occupation values in qubit-index order [q0, q1, ..., q(n-1)].
    """
    return [(index >> q) & 1 for q in range(num_qubits)]


def particle_number_expectation(
    sector_indices: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    values = np.array([int(i).bit_count() for i in sector_indices], dtype=float)
    return float(np.dot(values, probabilities))


def participation_ratio(probabilities: np.ndarray) -> float:
    denom = float(np.sum(probabilities**2))
    if denom <= 0.0:
        return 0.0
    return 1.0 / denom


def solve_geometry(
    spacing_angstrom: float,
    output_dir: Path,
) -> tuple[GeometryResult, dict[str, Any]]:
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.units import DistanceUnit

    atom_spec = h4_linear_atom_spec(spacing_angstrom)

    driver = PySCFDriver(
        atom=atom_spec,
        unit=DistanceUnit.ANGSTROM,
        charge=EXPECTED_CHARGE,
        spin=EXPECTED_SPIN,
        basis=EXPECTED_BASIS.lower(),
    )

    problem = driver.run()

    num_spin_orbitals = int(problem.num_spin_orbitals)
    num_spatial_orbitals = int(problem.num_spatial_orbitals)
    num_alpha, num_beta = map(int, problem.num_particles)
    num_particles_total = num_alpha + num_beta

    # Hard validation: these define the preregistered benchmark.
    if num_spatial_orbitals != EXPECTED_NUM_SPATIAL_ORBITALS:
        raise RuntimeError(
            f"H4/STO-3G spatial-orbital mismatch at d={spacing_angstrom}: "
            f"expected {EXPECTED_NUM_SPATIAL_ORBITALS}, got {num_spatial_orbitals}"
        )

    if num_spin_orbitals != EXPECTED_NUM_SPIN_ORBITALS:
        raise RuntimeError(
            f"H4/STO-3G spin-orbital mismatch at d={spacing_angstrom}: "
            f"expected {EXPECTED_NUM_SPIN_ORBITALS}, got {num_spin_orbitals}"
        )

    if (num_alpha, num_beta) != (EXPECTED_NUM_ALPHA, EXPECTED_NUM_BETA):
        raise RuntimeError(
            f"Particle mismatch at d={spacing_angstrom}: "
            f"expected ({EXPECTED_NUM_ALPHA}, {EXPECTED_NUM_BETA}), "
            f"got ({num_alpha}, {num_beta})"
        )

    if num_particles_total != EXPECTED_NUM_ELECTRONS:
        raise RuntimeError(
            f"Electron-count mismatch at d={spacing_angstrom}: "
            f"expected {EXPECTED_NUM_ELECTRONS}, got {num_particles_total}"
        )

    second_q_ops = problem.second_q_ops()
    electronic_hamiltonian = second_q_ops[0]

    mapper = JordanWignerMapper()
    qubit_hamiltonian = mapper.map(electronic_hamiltonian)

    if qubit_hamiltonian.num_qubits != EXPECTED_NUM_SPIN_ORBITALS:
        raise RuntimeError(
            f"Jordan-Wigner qubit count mismatch at d={spacing_angstrom}: "
            f"expected {EXPECTED_NUM_SPIN_ORBITALS}, "
            f"got {qubit_hamiltonian.num_qubits}"
        )

    # 2^8 = 256, so dense construction is intentionally simple and auditable.
    full_matrix_sparse = qubit_hamiltonian.to_matrix(sparse=True)
    full_matrix = np.asarray(full_matrix_sparse.toarray(), dtype=np.complex128)

    expected_full_dim = 1 << EXPECTED_NUM_SPIN_ORBITALS
    if full_matrix.shape != (expected_full_dim, expected_full_dim):
        raise RuntimeError(
            f"Full Hamiltonian dimension mismatch: {full_matrix.shape}"
        )

    sector_indices = basis_indices_with_particle_number(
        EXPECTED_NUM_SPIN_ORBITALS,
        EXPECTED_NUM_ELECTRONS,
    )

    expected_sector_dim = math.comb(
        EXPECTED_NUM_SPIN_ORBITALS,
        EXPECTED_NUM_ELECTRONS,
    )

    if len(sector_indices) != expected_sector_dim:
        raise RuntimeError(
            f"N=4 sector dimension mismatch: expected {expected_sector_dim}, "
            f"got {len(sector_indices)}"
        )

    sector_matrix = full_matrix[np.ix_(sector_indices, sector_indices)]

    hermiticity_error = float(
        np.max(np.abs(sector_matrix - sector_matrix.conjugate().T))
    )
    if hermiticity_error > 1e-10:
        raise RuntimeError(
            f"Sector Hamiltonian is not Hermitian within tolerance: "
            f"max error={hermiticity_error:.3e}"
        )

    eigenvalues, eigenvectors = np.linalg.eigh(sector_matrix)

    electronic_ground_energy = float(np.real(eigenvalues[0]))
    ground_state_sector = np.asarray(eigenvectors[:, 0], dtype=np.complex128)

    norm = float(np.vdot(ground_state_sector, ground_state_sector).real)
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise RuntimeError(
            f"Ground-state vector norm mismatch at d={spacing_angstrom}: {norm}"
        )

    probabilities = np.abs(ground_state_sector) ** 2
    prob_sum = float(probabilities.sum())

    if not math.isclose(prob_sum, 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise RuntimeError(
            f"Probability normalization mismatch at d={spacing_angstrom}: {prob_sum}"
        )

    number_expectation = particle_number_expectation(
        sector_indices,
        probabilities,
    )

    if not math.isclose(
        number_expectation,
        EXPECTED_NUM_ELECTRONS,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError(
            f"Particle-number expectation mismatch: {number_expectation}"
        )

    nuclear_repulsion = float(problem.nuclear_repulsion_energy or 0.0)
    total_ground_energy = electronic_ground_energy + nuclear_repulsion

    max_local_idx = int(np.argmax(probabilities))
    max_basis_index = int(sector_indices[max_local_idx])
    max_probability = float(probabilities[max_local_idx])

    significant_1e6 = int(np.count_nonzero(probabilities >= 1e-6))
    significant_1e4 = int(np.count_nonzero(probabilities >= 1e-4))
    pr = float(participation_ratio(probabilities))

    # Sort determinants by reference probability for later benchmark use.
    ranking = np.argsort(-probabilities)

    determinant_records = []
    for rank, local_idx in enumerate(ranking, start=1):
        basis_index = int(sector_indices[local_idx])
        amp = ground_state_sector[local_idx]
        prob = float(probabilities[local_idx])

        determinant_records.append(
            {
                "rank_by_reference_probability": rank,
                "sector_local_index": int(local_idx),
                "basis_index": basis_index,
                "bitstring_qiskit_qn_to_q0": bitstring_qiskit(
                    basis_index,
                    EXPECTED_NUM_SPIN_ORBITALS,
                ),
                "occupation_q0_to_qn": occupation_vector_little_endian(
                    basis_index,
                    EXPECTED_NUM_SPIN_ORBITALS,
                ),
                "amplitude_real": float(np.real(amp)),
                "amplitude_imag": float(np.imag(amp)),
                "probability": prob,
            }
        )

    result = GeometryResult(
        spacing_angstrom=float(spacing_angstrom),
        atom_specification=atom_spec,
        num_spatial_orbitals=num_spatial_orbitals,
        num_spin_orbitals=num_spin_orbitals,
        num_alpha=num_alpha,
        num_beta=num_beta,
        num_particles_total=num_particles_total,
        full_hilbert_dimension=expected_full_dim,
        n4_sector_dimension=expected_sector_dim,
        electronic_ground_energy_hartree=electronic_ground_energy,
        nuclear_repulsion_energy_hartree=nuclear_repulsion,
        total_ground_energy_hartree=total_ground_energy,
        ground_state_norm=norm,
        particle_number_expectation=number_expectation,
        max_probability=max_probability,
        max_probability_basis_index=max_basis_index,
        max_probability_bitstring_qiskit=bitstring_qiskit(
            max_basis_index,
            EXPECTED_NUM_SPIN_ORBITALS,
        ),
        significant_determinants_p1e6=significant_1e6,
        significant_determinants_p1e4=significant_1e4,
        participation_ratio=pr,
    )

    geometry_payload = {
        "summary": result.__dict__,
        "validation": {
            "hermiticity_max_abs_error": hermiticity_error,
            "probability_sum": prob_sum,
            "sector_dimension_expected": expected_sector_dim,
            "sector_dimension_observed": len(sector_indices),
            "fixed_particle_number": EXPECTED_NUM_ELECTRONS,
            "passed": True,
        },
        "reference_determinants": determinant_records,
        "sector_basis_indices": [int(x) for x in sector_indices],
        "sector_eigenvalues_electronic_hartree": [
            float(np.real(x)) for x in eigenvalues
        ],
    }

    tag = f"{spacing_angstrom:.2f}".replace(".", "p")
    json_path = output_dir / f"h4_sto3g_d_{tag}A_reference.json"
    npz_path = output_dir / f"h4_sto3g_d_{tag}A_reference.npz"

    json_path.write_text(
        json.dumps(geometry_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    np.savez_compressed(
        npz_path,
        spacing_angstrom=np.array(spacing_angstrom, dtype=float),
        sector_indices=sector_indices,
        sector_hamiltonian=sector_matrix,
        eigenvalues_electronic_hartree=eigenvalues,
        ground_state_amplitudes=ground_state_sector,
        ground_state_probabilities=probabilities,
    )

    geometry_payload["files"] = {
        "json": str(json_path),
        "npz": str(npz_path),
    }

    return result, geometry_payload


def write_summary_csv(
    results: list[GeometryResult],
    output_dir: Path,
) -> Path:
    import csv

    path = output_dir / "h4_sto3g_reference_summary.csv"

    fieldnames = list(GeometryResult.__dataclass_fields__.keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)

    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and exactly solve the Phase 5 H4/STO-3G reference benchmark."
    )

    default_output_dir = (
        Path(__file__).resolve().parent.parent / "results" / "reference"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Output directory for reference results.",
    )

    parser.add_argument(
        "--geometry",
        type=float,
        default=None,
        help=(
            "Optional single preregistered H-H spacing in Angstrom. "
            "If omitted, all 8 preregistered geometries are run."
        ),
    )

    return parser.parse_args()


def validate_requested_geometry(value: float | None) -> tuple[float, ...]:
    if value is None:
        return GEOMETRIES_ANGSTROM

    for frozen in GEOMETRIES_ANGSTROM:
        if math.isclose(value, frozen, rel_tol=0.0, abs_tol=1e-12):
            return (frozen,)

    raise ValueError(
        f"Geometry {value} Å is not in the preregistered geometry set: "
        f"{GEOMETRIES_ANGSTROM}"
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("SOFT SPACES — PHASE 5 H4/STO-3G REFERENCE PROBLEM")
    print("=" * 78)
    print(f"Version       : {SCRIPT_VERSION}")
    print(f"Output dir    : {output_dir}")

    require_dependencies()

    prereg_json, prereg_hash = find_preregistration(output_dir)
    prereg_report = verify_preregistration(prereg_json, prereg_hash)

    print(
        f"v50_0 manifest: "
        f"{'VERIFIED' if prereg_report.get('verified') else 'NOT FOUND'}"
    )
    if prereg_report.get("verified"):
        print(f"v50_0 SHA-256 : {prereg_report['canonical_sha256']}")

    geometries = validate_requested_geometry(args.geometry)

    print(f"Geometries    : {geometries}")
    print(f"Electrons     : {EXPECTED_NUM_ELECTRONS}")
    print(f"Spin orbitals : {EXPECTED_NUM_SPIN_ORBITALS}")
    print(f"N=4 dimension : {math.comb(EXPECTED_NUM_SPIN_ORBITALS, EXPECTED_NUM_ELECTRONS)}")
    print("-" * 78)

    all_results: list[GeometryResult] = []
    geometry_payloads: list[dict[str, Any]] = []

    for idx, spacing in enumerate(geometries, start=1):
        print(
            f"[{idx}/{len(geometries)}] H4 linear, d={spacing:.2f} Å ...",
            flush=True,
        )

        result, payload = solve_geometry(spacing, output_dir)
        all_results.append(result)
        geometry_payloads.append(payload)

        print(
            f"    E_elec={result.electronic_ground_energy_hartree:+.12f} Ha | "
            f"E_nuc={result.nuclear_repulsion_energy_hartree:+.12f} Ha | "
            f"E_total={result.total_ground_energy_hartree:+.12f} Ha"
        )
        print(
            f"    max P={result.max_probability:.6f} at "
            f"|{result.max_probability_bitstring_qiskit}> | "
            f"PR={result.participation_ratio:.3f}"
        )

    csv_path = write_summary_csv(all_results, output_dir)

    total_energies = [
        result.total_ground_energy_hartree for result in all_results
    ]

    min_idx = int(np.argmin(total_energies))
    min_result = all_results[min_idx]

    run_summary = {
        "_metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "script_version": SCRIPT_VERSION,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "package_versions": {
                "numpy": package_version("numpy"),
                "qiskit": package_version("qiskit"),
                "qiskit-nature": package_version("qiskit-nature"),
                "pyscf": package_version("pyscf"),
            },
        },
        "project": PROJECT,
        "phase": PHASE,
        "preregistration_verification": prereg_report,
        "benchmark": {
            "molecule": EXPECTED_MOLECULE,
            "geometry_family": "linear equally spaced H4 chain",
            "basis": EXPECTED_BASIS,
            "charge": EXPECTED_CHARGE,
            "spin_pyscf_2S": EXPECTED_SPIN,
            "num_electrons": EXPECTED_NUM_ELECTRONS,
            "num_spatial_orbitals": EXPECTED_NUM_SPATIAL_ORBITALS,
            "num_spin_orbitals": EXPECTED_NUM_SPIN_ORBITALS,
            "mapping": "Jordan-Wigner",
            "qubit_tapering": False,
            "exact_sector_diagonalisation": True,
        },
        "results": [result.__dict__ for result in all_results],
        "minimum_total_energy_among_run_geometries": {
            "spacing_angstrom": min_result.spacing_angstrom,
            "total_ground_energy_hartree": min_result.total_ground_energy_hartree,
        },
        "interpretation_boundary": {
            "soft_spaces_real_null_run": False,
            "claim": (
                "This script establishes molecular reference data only. "
                "No Soft Spaces predictive advantage is tested in v50_1."
            ),
        },
    }

    summary_path = output_dir / "phase5_v50_1_h4_reference_run.json"
    summary_path.write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary_hash = sha256_hex(run_summary)
    summary_hash_path = output_dir / "phase5_v50_1_h4_reference_run.sha256"
    summary_hash_path.write_text(summary_hash + "\n", encoding="ascii")

    print("-" * 78)
    print("REFERENCE BUILD COMPLETE")
    print("-" * 78)
    print(f"Summary CSV   : {csv_path}")
    print(f"Run JSON      : {summary_path}")
    print(f"Run SHA-256   : {summary_hash}")
    print(f"Hash file     : {summary_hash_path}")
    print(
        f"Lowest E_total among executed geometries: "
        f"{min_result.total_ground_energy_hartree:+.12f} Ha "
        f"at d={min_result.spacing_angstrom:.2f} Å"
    )
    print("-" * 78)
    print("PASS: H4/STO-3G reference problem constructed and validated.")
    print("NO Soft Spaces REAL/NULL ranking was executed.")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
