#!/usr/bin/env python3
"""
Soft Spaces Phase 4 v40.5 — IBM Heron real-hardware smoke execution.

PURPOSE
-------
Submit the first deliberately small REAL-HARDWARE smoke test after the
v40.4 hardware freeze.

This script:
1. Reads v40_4_ibm_heron_frozen_manifest.json.
2. Refuses to change the frozen backend or initial layout.
3. Reconnects to the saved QiskitRuntimeService account.
4. Verifies that the backend is operational and Heron.
5. Builds one 8-qubit measurement-channel smoke circuit:
       RY(theta) on logical q0
       20 logical CX operations as 10 cancelling CX pairs q0<->q1
       measure logical q0
   The cancelling pairs are a HARDWARE SMOKE CHANNEL ONLY. They preserve the
   noiseless state but exercise two-qubit compilation/noise.
6. Transpiles with optimization_level=0 and the exact frozen initial layout.
7. Prints the transpiled depth, operation counts, and layout information.
8. By default DOES NOT SUBMIT.
9. With --submit, sends exactly one SamplerV2 job to the frozen backend,
   records the job ID, waits for the result, and writes a JSON result file.

IMPORTANT SCIENTIFIC SCOPE
--------------------------
This is NOT the full v40.3b REAL/NULL ranked-search experiment.
It is a controlled hardware smoke test that validates:
* credential/backend access,
* manifest-consistent fixed layout,
* Heron ISA transpilation,
* SamplerV2 submission,
* raw hardware measurement retrieval.

Do not interpret a smoke-test probability as evidence for or against the
Soft-Spaces hypothesis.

Expected environment:
    qiskit~=2.5
    qiskit-ibm-runtime~=0.49

Examples
--------
Dry run only:
    python -X utf8 -u .\v40_5_ibm_heron_execute.py

Submit one real QPU job:
    python -X utf8 -u .\v40_5_ibm_heron_execute.py --submit

Optional low-cost smoke shots:
    python -X utf8 -u .\v40_5_ibm_heron_execute.py --submit --shots 1024

The default shots are read from the frozen v40.4 manifest (normally 4096).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MANIFEST = Path("v40_4_ibm_heron_frozen_manifest.json")
DEFAULT_RESULT = Path("v40_5_ibm_heron_smoke_result.json")
N_LOGICAL = 8
SMOKE_CX_COUNT = 20
DEFAULT_P = 0.50


def die(message: str) -> "NoReturn":
    raise SystemExit(message)


def load_manifest(path: Path) -> dict:
    if not path.exists():
        die(
            f"Manifest not found: {path}\n"
            "Run v40_4_ibm_heron_prepare.py first and keep the generated "
            "v40_4_ibm_heron_frozen_manifest.json in this directory."
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        die(f"Could not read manifest {path}: {exc}")

    required = [
        "protocol_version",
        "backend",
        "processor_type",
        "selected_physical_qubits_sorted",
        "frozen_initial_layout_logical_q0_to_q7",
        "hardware_protocol",
        "freeze_rule",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        die(f"Manifest is missing required fields: {missing}")

    if data["protocol_version"] != "v40.4-heron-prep-1":
        die(
            "Refusing unexpected manifest protocol_version: "
            f"{data['protocol_version']!r}"
        )

    layout = data["frozen_initial_layout_logical_q0_to_q7"]
    if not isinstance(layout, list) or len(layout) != N_LOGICAL:
        die(f"Frozen layout must contain exactly {N_LOGICAL} physical qubits.")

    if len(set(int(q) for q in layout)) != N_LOGICAL:
        die("Frozen layout contains duplicate physical qubits.")

    family = str(data.get("processor_type", {}).get("family", "")).lower()
    if family and family != "heron":
        die(f"Refusing non-Heron frozen manifest: {data['processor_type']}")

    return data


def probability_to_theta(p: float) -> float:
    if not (0.0 < p < 1.0):
        die("--p must satisfy 0 < p < 1.")
    # For RY(theta)|0>, P(1) = sin(theta/2)^2.
    return 2.0 * math.asin(math.sqrt(p))


def build_smoke_circuit(p: float, cx_count: int):
    from qiskit import QuantumCircuit

    if cx_count < 0 or cx_count % 2:
        die("--cx-count must be a non-negative even integer.")

    theta = probability_to_theta(p)

    qc = QuantumCircuit(N_LOGICAL, 1, name="v40_5_heron_smoke")
    qc.ry(theta, 0)

    # Deliberate logical-identity stress block.
    # CX^2 = I, therefore each pair ideally cancels without changing the
    # encoded q0 probability.  optimization_level=0 is used so this remains
    # a hardware/transpilation stress channel rather than being optimized away.
    for _ in range(cx_count // 2):
        qc.cx(0, 1)
        qc.cx(0, 1)

    qc.measure(0, 0)
    return qc


def safe_job_id(job) -> str | None:
    try:
        value = job.job_id()
        return str(value) if value is not None else None
    except Exception:
        return None


def extract_counts_and_p1(result):
    """
    SamplerV2 result extraction for a single circuit with classical register c.

    Qiskit PrimitiveResult / SamplerPubResult normally exposes:
        result[0].data.c.get_counts()
    We retain a small fallback path for minor container naming differences.
    """
    pub = result[0]
    data = pub.data

    counts = None

    if hasattr(data, "c"):
        cdata = data.c
        if hasattr(cdata, "get_counts"):
            counts = cdata.get_counts()

    if counts is None:
        # Search result data fields for the first BitArray-like object.
        for name in dir(data):
            if name.startswith("_"):
                continue
            try:
                obj = getattr(data, name)
            except Exception:
                continue
            if hasattr(obj, "get_counts"):
                try:
                    counts = obj.get_counts()
                    break
                except Exception:
                    pass

    if counts is None:
        raise RuntimeError(
            "SamplerV2 returned a result, but counts could not be extracted."
        )

    counts = {str(k): int(v) for k, v in counts.items()}
    total = sum(counts.values())
    if total <= 0:
        raise RuntimeError("SamplerV2 returned zero total shots.")

    # One measured classical bit: outcome strings should be "0" / "1".
    ones = 0
    for bitstring, n in counts.items():
        cleaned = bitstring.replace(" ", "")
        if cleaned and cleaned[-1] == "1":
            ones += n

    return counts, ones / total, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Frozen v40.4 manifest (default: {DEFAULT_MANIFEST}).",
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=DEFAULT_RESULT,
        help=f"Result JSON written after submission (default: {DEFAULT_RESULT}).",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit one SamplerV2 QPU job. Without this flag: dry run only.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=None,
        help="Shots for this smoke run. Default: frozen manifest shots.",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=DEFAULT_P,
        help=f"Encoded ideal P(1) on logical q0 (default: {DEFAULT_P}).",
    )
    parser.add_argument(
        "--cx-count",
        type=int,
        default=SMOKE_CX_COUNT,
        help=f"Even number of deliberate logical CX gates (default: {SMOKE_CX_COUNT}).",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    backend_name = str(manifest["backend"])
    frozen_layout = [
        int(q) for q in manifest["frozen_initial_layout_logical_q0_to_q7"]
    ]
    frozen_protocol = manifest["hardware_protocol"]

    frozen_shots = int(frozen_protocol.get("shots", 4096))
    shots = frozen_shots if args.shots is None else int(args.shots)
    if shots <= 0:
        die("--shots must be positive.")

    # We allow a smaller explicit smoke-shot override, but never silently change it.
    if shots != frozen_shots:
        print(
            f"NOTICE: smoke-test shots override: {shots} "
            f"(frozen full-protocol value is {frozen_shots})."
        )
        print("This override is for smoke validation only, not the frozen primary run.")

    try:
        import qiskit
        import qiskit_ibm_runtime
        from qiskit.transpiler import generate_preset_pass_manager
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    except Exception as exc:
        die(
            "Missing Qiskit/IBM Runtime packages in the active environment.\n"
            'Expected approximately: qiskit~=2.5 and qiskit-ibm-runtime~=0.49\n'
            f"Original error: {exc}"
        )

    print("=== Soft Spaces Phase 4 v40.5 IBM HERON HARDWARE SMOKE TEST ===")
    print(f"Qiskit: {qiskit.__version__}")
    print(f"qiskit-ibm-runtime: {qiskit_ibm_runtime.__version__}")
    print(f"Manifest: {args.manifest}")
    print(f"Frozen backend: {backend_name}")
    print(f"Frozen logical q0..q7 -> physical: {frozen_layout}")
    print(f"Smoke P(1) ideal target: {args.p:.6f}")
    print(f"Logical CX count: {args.cx_count}")
    print(f"Shots: {shots}")

    service = QiskitRuntimeService()
    backend = service.backend(backend_name)

    status = backend.status()
    pt = getattr(backend, "processor_type", None)
    pt = pt if isinstance(pt, dict) else {}

    print(f"Backend operational now: {status.operational}")
    print(f"Pending jobs now: {status.pending_jobs}")
    print(f"Processor now: {pt}")

    if not status.operational:
        die("Frozen backend is currently not operational; refusing submission.")

    family = str(pt.get("family", "")).lower()
    if family and family != "heron":
        die(f"Backend is no longer reported as Heron: {pt}")

    if backend.name != backend_name:
        die(
            f"Backend mismatch: manifest={backend_name}, runtime={backend.name}. "
            "Refusing to continue."
        )

    circuit = build_smoke_circuit(args.p, args.cx_count)

    print("\nAbstract circuit:")
    print(f"  qubits: {circuit.num_qubits}")
    print(f"  depth: {circuit.depth()}")
    print(f"  ops: {dict(circuit.count_ops())}")

    # Level 0 matches the frozen v40.4/v40.3b protocol intent:
    # map/translate/routing only, with no aggressive optimization.
    pass_manager = generate_preset_pass_manager(
        optimization_level=0,
        backend=backend,
        initial_layout=frozen_layout,
        seed_transpiler=40_500_001,
    )
    isa_circuit = pass_manager.run(circuit)

    print("\nTranspiled ISA circuit:")
    print(f"  qubits: {isa_circuit.num_qubits}")
    print(f"  depth: {isa_circuit.depth()}")
    print(f"  ops: {dict(isa_circuit.count_ops())}")

    # Preserve useful mapping details as printable strings without depending
    # on private TranspileLayout internals.
    layout_repr = repr(getattr(isa_circuit, "layout", None))
    print(f"  layout: {layout_repr}")

    metadata = {
        "version": "v40.5-heron-smoke-1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_file": str(args.manifest),
        "manifest_protocol_version": manifest["protocol_version"],
        "backend": backend_name,
        "processor_type_at_run": pt,
        "backend_operational_at_run": bool(status.operational),
        "pending_jobs_at_run": int(status.pending_jobs),
        "frozen_initial_layout_logical_q0_to_q7": frozen_layout,
        "shots": shots,
        "frozen_full_protocol_shots": frozen_shots,
        "ideal_p1": float(args.p),
        "logical_cx_count": int(args.cx_count),
        "abstract_depth": int(circuit.depth()),
        "abstract_ops": {str(k): int(v) for k, v in circuit.count_ops().items()},
        "isa_depth": int(isa_circuit.depth()),
        "isa_ops": {str(k): int(v) for k, v in isa_circuit.count_ops().items()},
        "transpiler_optimization_level": 0,
        "transpiler_seed": 40_500_001,
        "layout_repr": layout_repr,
        "scientific_scope": (
            "Real-QPU smoke validation only; NOT the full v40.3b REAL/NULL "
            "ranked-search experiment and NOT a hypothesis test."
        ),
    }

    # Always write a preflight record, even in dry-run mode.
    preflight_path = args.result.with_name(args.result.stem + "_preflight.json")
    preflight_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nPreflight record written: {preflight_path}")

    if not args.submit:
        print("\nDRY RUN COMPLETE.")
        print("NO QPU JOB WAS SUBMITTED.")
        print("\nTo submit exactly one smoke job:")
        print(
            f"  python -X utf8 -u .\\{Path(__file__).name} "
            f"--submit --shots {shots}"
        )
        return

    # Explicitly gate the only expensive/consequential action behind --submit.
    print("\nSUBMIT ENABLED: submitting exactly one SamplerV2 job...")
    sampler = SamplerV2(mode=backend)
    job = sampler.run([isa_circuit], shots=shots)
    job_id = safe_job_id(job)

    print(f"Job ID: {job_id}")
    print("Waiting for hardware result...")

    result = job.result()
    counts, measured_p1, total_shots = extract_counts_and_p1(result)

    metadata.update(
        {
            "submitted": True,
            "job_id": job_id,
            "result_received_utc": datetime.now(timezone.utc).isoformat(),
            "counts": counts,
            "total_shots_returned": total_shots,
            "measured_p1": measured_p1,
            "delta_p1_measured_minus_ideal": measured_p1 - float(args.p),
        }
    )

    args.result.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n=== HARDWARE RESULT ===")
    print(f"Counts: {counts}")
    print(f"Measured P(1): {measured_p1:.8f}")
    print(f"Ideal encoded P(1): {args.p:.8f}")
    print(f"Measured - ideal: {measured_p1 - float(args.p):+.8f}")
    print(f"Result written: {args.result}")
    print("\nIMPORTANT: this is a smoke-test result, not a Soft-Spaces decision.")


if __name__ == "__main__":
    main()
