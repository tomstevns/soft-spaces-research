"""Soft Spaces Phase 2A v25.32 — low-depth 4Q hardware pilot.

This is a deliberately small quantum-circuit witness, not a claim that the
full 12Q NumPy calculation has been moved onto quantum hardware.

Protocol
--------
1. Encode the two-dimensional input subspace span{|0000>, |0001>} with a
   shallow four-qubit circuit.
2. REAL uses a GHZ-like encoded subspace. Three seeded NULL controls use the
   same gate skeleton and entangling count, but rotate the encoded basis.
3. Apply the same perturbation to REAL and NULL:
      dephasing  : exp(-i eta Z_0 / 2)
      transverse: exp(-i eta X_0 X_1 X_2 X_3 / 2)
4. Decode and measure. A shot is retained when the decoded outcome is 0000 or
   0001. Averaging the two input basis states estimates Tr(P P') / 2.
5. Normalize each perturbed survival by its eta=0 identity control. The frozen
   pilot gate is Delta = REAL - median(NULL) > 0 for both perturbation families.

Modes
-----
    --mode ideal     Aer without a noise model
    --mode noisy     Aer with a small generic gate/readout noise model (default)
    --mode hardware  IBM Quantum hardware through saved Runtime credentials

Examples
--------
    python soft_spaces_qiskit_hardware_pilot_v25_32.py --mode noisy
    python soft_spaces_qiskit_hardware_pilot_v25_32.py --mode hardware
    python soft_spaces_qiskit_hardware_pilot_v25_32.py --mode hardware --backend BACKEND_NAME

No IBM token is stored in this file. QiskitRuntimeService must already have a
saved account when --mode hardware is selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error


VERSION = "v25.32"
N_QUBITS = 4
ETA = 0.45
SHOTS = 4096
BASE_SEED = 25_032_000
NULL_SEEDS = (25_032_101, 25_032_102, 25_032_103)
FAMILIES = ("dephasing_Z0", "transverse_XXXX")
ACCEPTED = ("0000", "0001")


@dataclass(frozen=True)
class CircuitTag:
    model: str
    null_seed: int | None
    family: str
    eta: float
    basis_state: int


@dataclass(frozen=True)
class ModelFamilyResult:
    model: str
    null_seed: int | None
    family: str
    identity_survival: float
    perturbed_survival: float
    normalized_survival: float
    exact_identity: float
    exact_perturbed: float
    exact_normalized: float


def stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0xFFFFFFFF


def encoding_circuit(model: str, null_seed: int | None = None) -> QuantumCircuit:
    """Return matched shallow REAL/NULL encoders with three entangling gates."""
    circuit = QuantumCircuit(N_QUBITS, name=f"U_{model}")
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(2, 3)

    if model == "REAL":
        # Z rotations change phases inside the GHZ-like two-dimensional space,
        # but do not rotate that space into the surrounding Hilbert space.
        angles = (0.17, -0.23, 0.31, -0.11)
        for qubit, angle in enumerate(angles):
            circuit.rz(angle, qubit)
    elif model == "NULL":
        if null_seed is None:
            raise ValueError("NULL requires a fixed null_seed")
        rng = np.random.default_rng(null_seed)
        angles = rng.uniform(-np.pi, np.pi, size=N_QUBITS)
        for qubit, angle in enumerate(angles):
            circuit.ry(float(angle), qubit)
    else:
        raise ValueError(f"Unknown model: {model}")
    return circuit


def apply_perturbation(circuit: QuantumCircuit, family: str, eta: float) -> None:
    if abs(eta) < 1e-15:
        circuit.id(0)
    elif family == "dephasing_Z0":
        circuit.rz(float(eta), 0)
    elif family == "transverse_XXXX":
        generator = SparsePauliOp.from_list([("XXXX", 1.0)])
        circuit.append(PauliEvolutionGate(generator, time=float(eta) / 2.0), range(N_QUBITS))
    else:
        raise ValueError(f"Unknown perturbation family: {family}")


def experiment_circuit(tag: CircuitTag, measure: bool) -> QuantumCircuit:
    encoder = encoding_circuit(tag.model, tag.null_seed)
    circuit = QuantumCircuit(N_QUBITS, name=f"{tag.model}_{tag.family}_{tag.eta:g}_b{tag.basis_state}")
    if tag.basis_state == 1:
        circuit.x(0)
    circuit.compose(encoder, inplace=True)
    circuit.barrier()
    apply_perturbation(circuit, tag.family, tag.eta)
    circuit.barrier()
    circuit.compose(encoder.inverse(), inplace=True)
    circuit.barrier()
    if measure:
        circuit.measure_all()
    return circuit


def all_tags() -> list[CircuitTag]:
    tags: list[CircuitTag] = []
    models = [("REAL", None)] + [("NULL", seed) for seed in NULL_SEEDS]
    for model, null_seed in models:
        for family in FAMILIES:
            for eta in (0.0, ETA):
                for basis_state in (0, 1):
                    tags.append(CircuitTag(model, null_seed, family, eta, basis_state))
    return tags


def exact_survival(tag: CircuitTag) -> float:
    probabilities = Statevector.from_instruction(experiment_circuit(tag, measure=False)).probabilities()
    return float(probabilities[0] + probabilities[1])


def generic_noise_model() -> NoiseModel:
    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(0.001, 1), ["id", "rz", "sx", "x"])
    model.add_all_qubit_quantum_error(depolarizing_error(0.010, 2), ["cx"])
    model.add_all_qubit_readout_error(ReadoutError([[0.99, 0.01], [0.02, 0.98]]))
    return model


def run_aer(circuits: list[QuantumCircuit], shots: int, mode: str) -> tuple[list[dict[str, int]], str, str | None]:
    if mode == "ideal":
        backend = AerSimulator()
    elif mode == "noisy":
        backend = AerSimulator(noise_model=generic_noise_model())
    else:
        raise ValueError(mode)
    compiled = transpile(circuits, backend, optimization_level=1, seed_transpiler=BASE_SEED)
    result = backend.run(compiled, shots=shots, seed_simulator=BASE_SEED).result()
    counts = [result.get_counts(index) for index in range(len(compiled))]
    return counts, backend.name, None


def run_hardware(
    circuits: list[QuantumCircuit], shots: int, backend_name: str | None
) -> tuple[list[dict[str, int]], str, str]:
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    except ImportError as exc:
        raise RuntimeError("Install qiskit-ibm-runtime before using --mode hardware") from exc

    try:
        service = QiskitRuntimeService()
    except Exception as exc:
        raise RuntimeError(
            "No saved IBM Quantum account was found. Save credentials locally with "
            "QiskitRuntimeService.save_account(...), then rerun --mode hardware."
        ) from exc

    backend = (
        service.backend(backend_name)
        if backend_name
        else service.least_busy(min_num_qubits=N_QUBITS, operational=True, simulator=False)
    )
    pass_manager = generate_preset_pass_manager(
        backend=backend, optimization_level=1, seed_transpiler=BASE_SEED
    )
    isa_circuits = pass_manager.run(circuits)
    job = SamplerV2(mode=backend).run(isa_circuits, shots=shots)
    primitive_result = job.result()
    counts = [pub_result.data.meas.get_counts() for pub_result in primitive_result]
    return counts, backend.name, job.job_id()


def hit_probability(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return float("nan")
    return float(sum(counts.get(bitstring, 0) for bitstring in ACCEPTED) / total)


def collapse_results(
    tags: list[CircuitTag], counts: list[dict[str, int]], exact: list[float]
) -> list[ModelFamilyResult]:
    shot_values: dict[tuple[str, int | None, str, float], list[float]] = {}
    exact_values: dict[tuple[str, int | None, str, float], list[float]] = {}
    for tag, count, exact_value in zip(tags, counts, exact, strict=True):
        key = (tag.model, tag.null_seed, tag.family, tag.eta)
        shot_values.setdefault(key, []).append(hit_probability(count))
        exact_values.setdefault(key, []).append(float(exact_value))

    results: list[ModelFamilyResult] = []
    models = [("REAL", None)] + [("NULL", seed) for seed in NULL_SEEDS]
    for model, null_seed in models:
        for family in FAMILIES:
            key_zero = (model, null_seed, family, 0.0)
            key_eta = (model, null_seed, family, ETA)
            identity = float(np.mean(shot_values[key_zero]))
            perturbed = float(np.mean(shot_values[key_eta]))
            exact_identity = float(np.mean(exact_values[key_zero]))
            exact_perturbed = float(np.mean(exact_values[key_eta]))
            results.append(ModelFamilyResult(
                model=model,
                null_seed=null_seed,
                family=family,
                identity_survival=identity,
                perturbed_survival=perturbed,
                normalized_survival=perturbed / identity if identity > 0 else float("nan"),
                exact_identity=exact_identity,
                exact_perturbed=exact_perturbed,
                exact_normalized=(
                    exact_perturbed / exact_identity if exact_identity > 0 else float("nan")
                ),
            ))
    return results


def family_summary(results: Iterable[ModelFamilyResult]) -> list[dict[str, float | str | bool]]:
    rows = list(results)
    summary: list[dict[str, float | str | bool]] = []
    for family in FAMILIES:
        real = next(row for row in rows if row.family == family and row.model == "REAL")
        null = [
            row.normalized_survival
            for row in rows
            if row.family == family and row.model == "NULL"
        ]
        null_exact = [
            row.exact_normalized for row in rows if row.family == family and row.model == "NULL"
        ]
        null_median = float(median(null))
        exact_null_median = float(median(null_exact))
        delta = real.normalized_survival - null_median
        exact_delta = real.exact_normalized - exact_null_median
        summary.append({
            "family": family,
            "real_normalized": real.normalized_survival,
            "null_median_normalized": null_median,
            "delta": delta,
            "passes_frozen_gate": bool(delta > 0.0),
            "exact_real_normalized": real.exact_normalized,
            "exact_null_median_normalized": exact_null_median,
            "exact_delta": exact_delta,
        })
    return summary


def save_circuit_diagram(output_dir: Path) -> Path:
    tag = CircuitTag("REAL", None, "transverse_XXXX", ETA, 0)
    circuit = experiment_circuit(tag, measure=True)
    path = output_dir / "v25_32_qiskit_hardware_pilot_circuit.png"
    circuit.draw("mpl", filename=str(path), fold=24, idle_wires=False)
    plt.close("all")
    return path


def save_result_chart(summary: list[dict[str, float | str | bool]], output_dir: Path) -> Path:
    labels = ["Dephasing Z₀", "Transverse XXXX"]
    real = [100.0 * float(row["real_normalized"]) for row in summary]
    null = [100.0 * float(row["null_median_normalized"]) for row in summary]
    x = np.arange(len(labels))
    width = 0.34
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.bar(x - width / 2, real, width, label="REAL", color="#1f77b4")
    axis.bar(x + width / 2, null, width, label="Median NULL", color="#a7a7a7")
    axis.set_ylabel("Normalized subspace survival (%)")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 105)
    axis.set_title("Soft Spaces 4Q hardware pilot")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    for index, values in enumerate((real, null)):
        offset = -width / 2 if index == 0 else width / 2
        for position, value in zip(x, values):
            axis.text(position + offset, value + 1.2, f"{value:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    path = output_dir / "v25_32_qiskit_hardware_pilot_results.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def render_report(
    mode: str,
    backend_name: str,
    job_id: str | None,
    shots: int,
    summary: list[dict[str, float | str | bool]],
) -> str:
    lines = [
        "=== Soft Spaces Phase 2A v25.32 — 4Q HARDWARE PILOT ===",
        "Scope: reduced gate-level witness; not the full 12Q classical calculation",
        f"Mode: {mode} | backend: {backend_name} | shots/circuit: {shots}",
        f"Qubits: {N_QUBITS} | eta: {ETA} | NULL seeds: {list(NULL_SEEDS)}",
        "Frozen gate: normalized REAL - median(normalized NULL) > 0 for both families",
    ]
    if job_id is not None:
        lines.append(f"IBM Runtime job id: {job_id}")
    lines.extend([
        "",
        "family               REAL norm    NULL median    DELTA       gate    exact DELTA",
    ])
    for row in summary:
        gate = "PASS" if row["passes_frozen_gate"] else "FAIL"
        lines.append(
            f"{str(row['family']):20s} "
            f"{float(row['real_normalized']):10.6f}  "
            f"{float(row['null_median_normalized']):11.6f}  "
            f"{float(row['delta']):+9.6f}  {gate:>6s}  "
            f"{float(row['exact_delta']):+11.6f}"
        )
    passed = sum(bool(row["passes_frozen_gate"]) for row in summary)
    lines.extend(["", f"Decision: {passed}/{len(summary)} perturbation families pass.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("ideal", "noisy", "hardware"), default="noisy")
    parser.add_argument("--backend", default=None, help="IBM backend name; omit for least busy")
    parser.add_argument("--shots", type=int, default=SHOTS)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.shots < 100:
        raise ValueError("Use at least 100 shots per circuit")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tags = all_tags()
    circuits = [experiment_circuit(tag, measure=True) for tag in tags]
    exact = [exact_survival(tag) for tag in tags]

    if args.mode in ("ideal", "noisy"):
        counts, backend_name, job_id = run_aer(circuits, args.shots, args.mode)
    else:
        counts, backend_name, job_id = run_hardware(circuits, args.shots, args.backend)

    results = collapse_results(tags, counts, exact)
    summary = family_summary(results)
    report = render_report(args.mode, backend_name, job_id, args.shots, summary)
    print(report)

    base = args.output_dir / f"v25_32_qiskit_hardware_pilot_{args.mode}"
    (base.with_suffix(".txt")).write_text(report, encoding="utf-8")
    payload = {
        "version": VERSION,
        "scope": "reduced_4q_gate_level_hardware_witness",
        "mode": args.mode,
        "backend": backend_name,
        "job_id": job_id,
        "shots_per_circuit": args.shots,
        "eta": ETA,
        "base_seed": BASE_SEED,
        "null_seeds": list(NULL_SEEDS),
        "accepted_bitstrings": list(ACCEPTED),
        "environment": {
            "python": platform.python_version(),
            "qiskit": qiskit.__version__,
            "numpy": np.__version__,
        },
        "results": [asdict(row) for row in results],
        "summary": summary,
    }
    (base.with_suffix(".json")).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    circuit_path = save_circuit_diagram(args.output_dir)
    chart_path = save_result_chart(summary, args.output_dir)
    print(f"Circuit diagram: {circuit_path}")
    print(f"Result chart: {chart_path}")


if __name__ == "__main__":
    main()
