#!/usr/bin/env python3
"""
Soft Spaces Phase 4 v40.6 — first scientific REAL-vs-NULL IBM Heron hardware test.

Scientific role
---------------
This is the first real-QPU version of the frozen v40.3b ranked-search endpoint:

    -E source prominence
        -> monotone probability encoding
        -> explicit 8Q measurement-channel circuit
        -> real IBM Heron hardware
        -> measured source ranking
        -> +E ranked search
        -> REAL versus matched NULL

The classical frozen Phase-3 kernel still computes source prominence.
This is therefore a REAL-HARDWARE measurement-channel test, NOT a complete
Hamiltonian/projector-native quantum implementation.

Frozen primary protocol
-----------------------
* backend/layout/shots/seeds/models are read from
  v40_4_ibm_heron_frozen_manifest.json
* expected backend: IBM Heron
* seeds: 25043000..25043011
* models: REAL and NULL
* primary two-qubit checkpoint: 20 logical CX operations
* shots: 4096
* K: 8
* transpiler optimization level: 0
* no mitigation, twirling, DD or ZNE

Hardware-specific adaptation
----------------------------
v40.3b cycles the paired identity CX blocks over logical spectator qubits.
On real hardware, not every logical spectator is necessarily directly coupled
to q0. To avoid SWAP routing and to keep the experiment confined to the frozen
8Q subgraph, v40.6 cycles ONLY over logical spectators whose frozen physical
qubits share a calibrated native-2Q edge with frozen physical q0.

Every noise-loading block is still:
    CX(0,q); CX(0,q)
and is therefore exactly identity in the noiseless logical circuit.

The script verifies after transpilation that:
* all 2Q operations stay inside the frozen 8Q physical set;
* no routing leaves the frozen subgraph;
* the transpiled circuit contains exactly the frozen requested number of
  native 2Q gates (normally 20 CZ on Heron).

Safety
------
Default is PREFLIGHT ONLY. No QPU job is submitted unless --submit is given.

Before submission the program prints:
* candidate circuit counts by seed/model
* total circuits
* shots/circuit
* total circuit executions = circuits * shots
* fixed backend/layout
* hardware edge schedule
* transpiled depth / 2Q-gate ranges

It refuses submission above --max-executions (default 2,500,000).

Dependencies
------------
Place this file beside:
    v31_7_Soft_Spaces_independent_replication.py
    v40_0_application_screening.py
    v40_4_ibm_heron_frozen_manifest.json

Environment:
    qiskit ~= 2.5
    qiskit-ibm-runtime ~= 0.49

Usage
-----
Preflight only:
    python -X utf8 -u .\v40_6a_ibm_heron_real_null.py

Submit the frozen scientific test:
    python -X utf8 -u .\v40_6a_ibm_heron_real_null.py --submit
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


VERSION = "v40.6a-heron-real-null-runtime-preflight-1"

BASE_MODULE = "v31_7_Soft_Spaces_independent_replication"
APP_MODULE = "v40_0_application_screening"

DEFAULT_MANIFEST = Path("v40_4_ibm_heron_frozen_manifest.json")
DEFAULT_OUTPUT = Path("v40_6a_ibm_heron_real_null_result.json")
DEFAULT_PREFLIGHT = Path("v40_6a_ibm_heron_real_null_preflight.json")
DEFAULT_MAX_ESTIMATED_QPU_SECONDS = 540.0  # preserve ~60 s of a 10 min Open-plan budget

DEFAULT_K = 8
DEFAULT_MAX_EXECUTIONS = 2_500_000
BOOTSTRAPS = 200_000
BOOTSTRAP_SEED = 40_600_001
TRANSPILE_SEED = 40_600_101


def die(message: str) -> "NoReturn":
    raise SystemExit(message)


def load_softspaces():
    try:
        base = importlib.import_module(BASE_MODULE)
    except ModuleNotFoundError as exc:
        die(
            f"Could not import {BASE_MODULE}.py\n"
            f"Place this script beside {BASE_MODULE}.py."
        )

    try:
        app0 = importlib.import_module(APP_MODULE)
    except ModuleNotFoundError as exc:
        die(
            f"Could not import {APP_MODULE}.py\n"
            f"Place this script beside {APP_MODULE}.py."
        )

    return base, app0


def load_manifest(path: Path) -> dict:
    if not path.exists():
        die(f"Frozen manifest not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    required = [
        "protocol_version",
        "backend",
        "processor_type",
        "selected_physical_qubits_sorted",
        "frozen_initial_layout_logical_q0_to_q7",
        "internal_2q_edges",
        "hardware_protocol",
        "freeze_rule",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        die(f"Manifest missing required fields: {missing}")

    if data["protocol_version"] != "v40.4-heron-prep-1":
        die(
            "Unexpected manifest protocol_version: "
            f"{data['protocol_version']!r}"
        )

    protocol = data["hardware_protocol"]
    if int(protocol.get("logical_qubits", -1)) != 8:
        die("Frozen manifest does not specify 8 logical qubits.")

    if int(protocol.get("primary_cx_checkpoint", -1)) != 20:
        die("Frozen manifest primary_cx_checkpoint is not 20.")

    if int(protocol.get("shots", -1)) != 4096:
        die("Frozen manifest shots are not 4096.")

    if protocol.get("seed_ensemble") != "25043000..25043011":
        die(
            "Frozen seed ensemble mismatch: "
            f"{protocol.get('seed_ensemble')!r}"
        )

    if list(protocol.get("models", [])) != ["REAL", "NULL"]:
        die(f"Frozen model list mismatch: {protocol.get('models')!r}")

    if int(protocol.get("transpiler_optimization_level", -1)) != 0:
        die("Frozen optimization level is not 0.")

    return data


def extract_prominence_and_pairs(app0, base, seed: int, model: str):
    result = app0.build_prominence(base, seed, model)

    if len(result) == 2:
        prominence, pairs = result
    elif len(result) == 3:
        _, prominence, pairs = result
    else:
        raise ValueError(
            "Unexpected return shape from "
            "v40_0_application_screening.build_prominence(): "
            f"{len(result)} values"
        )

    return prominence, pairs


def monotone_probability_map(source: np.ndarray) -> np.ndarray:
    lo = float(np.min(source))
    hi = float(np.max(source))

    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Non-finite source prominence encountered.")

    if hi <= lo:
        return np.full_like(source, 0.5, dtype=float)

    z = (source - lo) / (hi - lo)
    return 0.05 + 0.90 * z


def theta_from_probability(p: float) -> float:
    p = float(np.clip(p, 0.0, 1.0))
    return 2.0 * math.asin(math.sqrt(p))


def logical_hardware_neighbors(manifest: dict) -> list[int]:
    """
    Return logical spectator indices directly connected to logical q0
    inside the frozen physical 8Q subgraph.
    """
    layout = [int(q) for q in manifest["frozen_initial_layout_logical_q0_to_q7"]]
    physical_to_logical = {p: i for i, p in enumerate(layout)}
    p0 = layout[0]

    neighbors = set()
    for edge in manifest["internal_2q_edges"]:
        a = int(edge["q0"])
        b = int(edge["q1"])
        if a == p0 and b in physical_to_logical:
            q = physical_to_logical[b]
            if q != 0:
                neighbors.add(q)
        elif b == p0 and a in physical_to_logical:
            q = physical_to_logical[a]
            if q != 0:
                neighbors.add(q)

    result = sorted(neighbors)
    if not result:
        die(
            "Frozen manifest contains no native 2Q edge from logical q0 "
            "to another frozen logical qubit."
        )

    return result


def build_candidate_circuit(
    QuantumCircuit,
    probability: float,
    total_cx: int,
    spectator_schedule: list[int],
):
    if total_cx < 0 or total_cx % 2:
        raise ValueError("total_cx must be a non-negative even integer.")
    if not spectator_schedule:
        raise ValueError("spectator_schedule must not be empty.")

    qc = QuantumCircuit(8, 1)
    qc.ry(theta_from_probability(probability), 0)

    pair_count = total_cx // 2
    for pair_idx in range(pair_count):
        q = spectator_schedule[pair_idx % len(spectator_schedule)]
        qc.cx(0, q)
        qc.cx(0, q)

    qc.measure(0, 0)
    return qc


def rank_metrics(
    measured_source: np.ndarray,
    target: np.ndarray,
    positions: np.ndarray,
    k: int,
):
    records = [
        {"pos": int(pos), "source": float(src), "target": float(tgt)}
        for pos, src, tgt in zip(positions, measured_source, target)
    ]

    predicted = sorted(records, key=lambda row: (-row["source"], row["pos"]))
    truth = sorted(records, key=lambda row: (-row["target"], row["pos"]))

    true_top = {row["pos"] for row in truth[:k]}
    predicted_top = {row["pos"] for row in predicted[:k]}

    precision = len(true_top & predicted_top) / float(k)

    predicted_positions = {
        row["pos"]: rank + 1
        for rank, row in enumerate(predicted)
    }
    evaluations = max(predicted_positions[pos] for pos in true_top)

    relevance = {
        row["pos"]: float(k - rank)
        for rank, row in enumerate(truth[:k])
    }

    def dcg(rows):
        value = 0.0
        for rank, row in enumerate(rows[:k], start=1):
            rel = relevance.get(row["pos"], 0.0)
            value += (2.0**rel - 1.0) / math.log2(rank + 1.0)
        return value

    ideal = dcg(truth)
    ndcg = dcg(predicted) / ideal if ideal > 0.0 else float("nan")

    return float(precision), int(evaluations), float(ndcg)


def bootstrap_paired_ci(values: np.ndarray, bootstraps: int, seed: int):
    rng = np.random.default_rng(seed)
    n = values.size
    samples = rng.choice(values, size=(bootstraps, n), replace=True)
    stats = np.mean(samples, axis=1)
    low, high = np.quantile(stats, [0.025, 0.975])
    return float(np.mean(values)), float(low), float(high)


def extract_counts(pub_result) -> dict[str, int]:
    data = pub_result.data

    if hasattr(data, "c") and hasattr(data.c, "get_counts"):
        counts = data.c.get_counts()
        return {str(k): int(v) for k, v in counts.items()}

    if hasattr(data, "meas") and hasattr(data.meas, "get_counts"):
        counts = data.meas.get_counts()
        return {str(k): int(v) for k, v in counts.items()}

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
                return {str(k): int(v) for k, v in counts.items()}
            except Exception:
                pass

    raise RuntimeError("Could not extract counts from SamplerV2 result.")


def p1_from_counts(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        raise RuntimeError("Zero total shots returned.")

    ones = 0
    for bitstring, count in counts.items():
        clean = bitstring.replace(" ", "")
        if clean and clean[-1] == "1":
            ones += int(count)

    return float(ones) / float(total)


def physical_twoq_indices(circuit) -> list[tuple[int, int, str]]:
    out = []
    for inst in circuit.data:
        if len(inst.qubits) == 2:
            a = circuit.find_bit(inst.qubits[0]).index
            b = circuit.find_bit(inst.qubits[1]).index
            out.append((int(a), int(b), str(inst.operation.name)))
    return out


def estimate_qpu_usage_locally(
    backend,
    isa_circuits,
    shots: int,
    frozen_physical_qubits: list[int],
):
    """
    Local IBM-style QPU usage estimate.

    IBM's documented baseline is:

        per-sub-job overhead
        + (rep_delay + circuit_length + init_duration) * executions

    The detailed estimate below schedules the already-transpiled ISA circuits
    with ALAP scheduling and uses circuit.estimate_duration(backend.target).

    No QPU job is submitted.
    """
    from qiskit.transpiler import generate_preset_pass_manager

    rep_delay = getattr(backend, "default_rep_delay", None)
    if rep_delay is None:
        # IBM documents 250 us as the common default; keep the fallback explicit.
        rep_delay = 250e-6
        rep_delay_source = "fallback_250us"
    else:
        rep_delay = float(rep_delay)
        rep_delay_source = "backend.default_rep_delay"

    reset_durations = []
    try:
        reset_map = backend.target["reset"]
        for q in frozen_physical_qubits:
            try:
                props = reset_map[(int(q),)]
                dur = getattr(props, "duration", None)
                if dur is not None and math.isfinite(float(dur)):
                    reset_durations.append(float(dur))
            except Exception:
                pass
    except Exception:
        pass

    # Resets on the participating qubits are effectively parallel for this
    # baseline; use the longest participating reset duration once per shot.
    init_duration = max(reset_durations) if reset_durations else 0.0

    scheduling_pm = generate_preset_pass_manager(
        target=backend.target,
        optimization_level=0,
        scheduling_method="alap",
    )

    scheduled = scheduling_pm.run(isa_circuits)

    durations = []
    for circuit in scheduled:
        try:
            duration = float(circuit.estimate_duration(backend.target))
        except Exception as exc:
            raise RuntimeError(
                "Could not estimate scheduled circuit duration with "
                "circuit.estimate_duration(backend.target)."
            ) from exc
        durations.append(duration)

    if not durations:
        raise RuntimeError("No circuit durations available for runtime estimate.")

    per_circuit_shot_times = [
        d + init_duration + rep_delay
        for d in durations
    ]

    # One primitive job can be internally split into multiple sub-jobs.
    # The exact split is service-side and cannot be known locally.
    one_subjob_overhead = 2.0
    detailed_seconds_one_subjob = (
        one_subjob_overhead
        + shots * float(sum(per_circuit_shot_times))
    )

    total_executions = len(isa_circuits) * shots
    quick_seconds = 2.0 + 0.00035 * total_executions

    return {
        "method": "IBM documented local scheduled-circuit baseline",
        "rep_delay_seconds": rep_delay,
        "rep_delay_source": rep_delay_source,
        "init_duration_seconds": init_duration,
        "circuit_duration_seconds_min": min(durations),
        "circuit_duration_seconds_mean": float(np.mean(durations)),
        "circuit_duration_seconds_max": max(durations),
        "circuit_shot_time_seconds_min": min(per_circuit_shot_times),
        "circuit_shot_time_seconds_mean": float(np.mean(per_circuit_shot_times)),
        "circuit_shot_time_seconds_max": max(per_circuit_shot_times),
        "total_executions": total_executions,
        "estimated_qpu_seconds_detailed_one_subjob":
            detailed_seconds_one_subjob,
        "estimated_qpu_minutes_detailed_one_subjob":
            detailed_seconds_one_subjob / 60.0,
        "estimated_qpu_seconds_quick_formula": quick_seconds,
        "estimated_qpu_minutes_quick_formula": quick_seconds / 60.0,
        "overhead_note": (
            "Detailed estimate assumes one ~2 s sub-job overhead. IBM Runtime "
            "may split a large primitive job into multiple sub-jobs, adding "
            "approximately 2 s per additional sub-job."
        ),
    }


def prepare_all(base, app0, manifest, backend, pass_manager, QuantumCircuit, k: int):
    seeds = [int(s) for s in base.HAMILTONIAN_SEEDS]

    if seeds != list(range(25043000, 25043012)):
        die(
            "Frozen base.HAMILTONIAN_SEEDS mismatch.\n"
            f"Observed: {seeds}"
        )

    total_cx = int(manifest["hardware_protocol"]["primary_cx_checkpoint"])
    shots = int(manifest["hardware_protocol"]["shots"])
    frozen_layout = [
        int(q) for q in manifest["frozen_initial_layout_logical_q0_to_q7"]
    ]
    frozen_set = set(int(q) for q in manifest["selected_physical_qubits_sorted"])
    spectators = logical_hardware_neighbors(manifest)

    rows = []
    all_isa = []
    circuit_meta = []

    depth_values = []
    twoq_values = []

    for seed_index, seed in enumerate(seeds, start=1):
        print(f"[preflight] seed {seed_index}/12: {seed}", flush=True)
        row = {"seed": seed}

        for model in ("REAL", "NULL"):
            prominence, pairs = extract_prominence_and_pairs(app0, base, seed, model)

            if len(pairs) < k:
                die(f"Seed {seed} {model}: only {len(pairs)} pairs for K={k}.")

            source = np.asarray(
                [float(prominence[neg]) for neg, _ in pairs], dtype=float
            )
            target = np.asarray(
                [float(prominence[pos]) for _, pos in pairs], dtype=float
            )
            positions = np.asarray(
                [int(pos) for _, pos in pairs], dtype=int
            )
            probabilities = monotone_probability_map(source)

            start = len(all_isa)

            for candidate_index, p in enumerate(probabilities):
                qc = build_candidate_circuit(
                    QuantumCircuit=QuantumCircuit,
                    probability=float(p),
                    total_cx=total_cx,
                    spectator_schedule=spectators,
                )

                isa = pass_manager.run(qc)

                twoq = physical_twoq_indices(isa)
                used_physical = set()
                for a, b, _name in twoq:
                    used_physical.add(a)
                    used_physical.add(b)

                outside = sorted(used_physical - frozen_set)
                if outside:
                    die(
                        f"Routing escaped frozen 8Q set for seed={seed} "
                        f"model={model} candidate={candidate_index}: {outside}"
                    )

                if len(twoq) != total_cx:
                    die(
                        f"Unexpected transpiled 2Q gate count for seed={seed} "
                        f"model={model} candidate={candidate_index}: "
                        f"requested {total_cx}, observed {len(twoq)}."
                    )

                all_isa.append(isa)
                depth_values.append(int(isa.depth()))
                twoq_values.append(len(twoq))

                circuit_meta.append({
                    "seed": seed,
                    "model": model,
                    "candidate_index": candidate_index,
                    "position": int(positions[candidate_index]),
                    "target": float(target[candidate_index]),
                    "ideal_encoded_p1": float(p),
                })

            stop = len(all_isa)

            row[model.lower()] = {
                "candidate_count": len(probabilities),
                "slice_start": start,
                "slice_stop": stop,
                "target": target,
                "positions": positions,
            }

        rows.append(row)

    summary = {
        "seeds": seeds,
        "models": ["REAL", "NULL"],
        "k": k,
        "shots": shots,
        "primary_cx": total_cx,
        "spectator_schedule_logical": spectators,
        "spectator_schedule_physical": [frozen_layout[q] for q in spectators],
        "total_circuits": len(all_isa),
        "total_executions": len(all_isa) * shots,
        "depth_min": min(depth_values) if depth_values else None,
        "depth_max": max(depth_values) if depth_values else None,
        "native_2q_min": min(twoq_values) if twoq_values else None,
        "native_2q_max": max(twoq_values) if twoq_values else None,
    }

    return rows, all_isa, circuit_meta, summary


def json_safe_preflight(rows, summary, manifest, backend_name):
    counts = []
    for row in rows:
        counts.append({
            "seed": row["seed"],
            "real_candidates": row["real"]["candidate_count"],
            "null_candidates": row["null"]["candidate_count"],
        })

    return {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_scope": (
            "Real IBM Heron measurement-channel ranked-search test; "
            "not full Hamiltonian/projector-native quantum execution."
        ),
        "backend": backend_name,
        "processor_type": manifest["processor_type"],
        "frozen_layout_logical_q0_to_q7":
            manifest["frozen_initial_layout_logical_q0_to_q7"],
        "frozen_physical_qubits":
            manifest["selected_physical_qubits_sorted"],
        "protocol": manifest["hardware_protocol"],
        "hardware_adaptation": {
            "rule": (
                "Paired logical CX identity blocks cycle only over frozen "
                "logical spectators directly connected to logical q0, avoiding "
                "SWAP routing outside the frozen 8Q subgraph."
            ),
            "spectator_schedule_logical":
                summary["spectator_schedule_logical"],
            "spectator_schedule_physical":
                summary["spectator_schedule_physical"],
        },
        "candidate_counts": counts,
        "total_circuits": summary["total_circuits"],
        "shots_per_circuit": summary["shots"],
        "total_executions": summary["total_executions"],
        "transpiled_depth_min": summary["depth_min"],
        "transpiled_depth_max": summary["depth_max"],
        "native_2q_gates_min": summary["native_2q_min"],
        "native_2q_gates_max": summary["native_2q_max"],
        "submitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--max-executions",
        type=int,
        default=DEFAULT_MAX_EXECUTIONS,
        help=(
            "Hard submission guard on total circuits*shots "
            f"(default {DEFAULT_MAX_EXECUTIONS:,})."
        ),
    )
    parser.add_argument(
        "--max-estimated-qpu-seconds",
        type=float,
        default=DEFAULT_MAX_ESTIMATED_QPU_SECONDS,
        help=(
            "Hard submission guard on locally estimated QPU usage in seconds "
            f"(default {DEFAULT_MAX_ESTIMATED_QPU_SECONDS:.0f})."
        ),
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit the preflighted frozen test to the real IBM QPU.",
    )
    args = parser.parse_args()

    if args.k != 8:
        die("Frozen primary endpoint requires K=8.")
    if args.max_executions < 1:
        die("--max-executions must be positive.")
    if args.max_estimated_qpu_seconds <= 0:
        die("--max-estimated-qpu-seconds must be positive.")

    manifest = load_manifest(args.manifest)
    base, app0 = load_softspaces()

    try:
        import qiskit
        import qiskit_ibm_runtime
        from qiskit import QuantumCircuit
        from qiskit.transpiler import generate_preset_pass_manager
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    except Exception as exc:
        die(
            "Missing Qiskit/IBM Runtime packages.\n"
            'Expected approximately qiskit~=2.5 and qiskit-ibm-runtime~=0.49\n'
            f"Original error: {exc}"
        )

    backend_name = str(manifest["backend"])
    frozen_layout = [
        int(q) for q in manifest["frozen_initial_layout_logical_q0_to_q7"]
    ]

    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    status = backend.status()
    pt = getattr(backend, "processor_type", None)
    pt = pt if isinstance(pt, dict) else {}

    print("=== Soft Spaces Phase 4 v40.6 IBM HERON REAL-vs-NULL ===")
    print(f"Python: {platform.python_version()}")
    print(f"Qiskit: {qiskit.__version__}")
    print(f"qiskit-ibm-runtime: {qiskit_ibm_runtime.__version__}")
    print(f"Backend: {backend.name}")
    print(f"Processor: {pt}")
    print(f"Operational: {status.operational}")
    print(f"Pending jobs: {status.pending_jobs}")
    print(f"Frozen layout q0..q7 -> physical: {frozen_layout}")

    if backend.name != backend_name:
        die("Runtime backend does not match frozen manifest.")
    if not status.operational:
        die("Frozen backend is currently not operational.")
    family = str(pt.get("family", "")).lower()
    if family and family != "heron":
        die(f"Backend is not reported as Heron: {pt}")

    pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=0,
        initial_layout=frozen_layout,
        seed_transpiler=TRANSPILE_SEED,
    )

    rows, isa_circuits, circuit_meta, summary = prepare_all(
        base=base,
        app0=app0,
        manifest=manifest,
        backend=backend,
        pass_manager=pass_manager,
        QuantumCircuit=QuantumCircuit,
        k=args.k,
    )

    runtime_estimate = estimate_qpu_usage_locally(
        backend=backend,
        isa_circuits=isa_circuits,
        shots=summary["shots"],
        frozen_physical_qubits=[
            int(q) for q in manifest["selected_physical_qubits_sorted"]
        ],
    )

    preflight = json_safe_preflight(
        rows=rows,
        summary=summary,
        manifest=manifest,
        backend_name=backend_name,
    )
    preflight["runtime_estimate"] = runtime_estimate
    preflight["max_estimated_qpu_seconds_guard"] = (
        args.max_estimated_qpu_seconds
    )
    args.preflight.write_text(json.dumps(preflight, indent=2), encoding="utf-8")

    print("\n=== PREFLIGHT ===")
    print(f"Seeds: 12")
    print(f"Models: REAL + NULL")
    print(f"Primary checkpoint: CX={summary['primary_cx']}")
    print(
        "Hardware spectator schedule logical: "
        f"{summary['spectator_schedule_logical']}"
    )
    print(
        "Hardware spectator schedule physical: "
        f"{summary['spectator_schedule_physical']}"
    )
    print(f"Total candidate circuits: {summary['total_circuits']}")
    print(f"Shots per circuit: {summary['shots']}")
    print(f"Total executions: {summary['total_executions']:,}")
    print(
        "Transpiled depth range: "
        f"{summary['depth_min']} .. {summary['depth_max']}"
    )
    print(
        "Native 2Q-gate range: "
        f"{summary['native_2q_min']} .. {summary['native_2q_max']}"
    )
    print(f"Submission guard: {args.max_executions:,} executions")
    print("\n=== LOCAL QPU-RUNTIME ESTIMATE ===")
    print(
        "Backend rep_delay: "
        f"{runtime_estimate['rep_delay_seconds'] * 1e6:.1f} us "
        f"({runtime_estimate['rep_delay_source']})"
    )
    print(
        "Scheduled circuit duration min/mean/max: "
        f"{runtime_estimate['circuit_duration_seconds_min'] * 1e6:.1f} / "
        f"{runtime_estimate['circuit_duration_seconds_mean'] * 1e6:.1f} / "
        f"{runtime_estimate['circuit_duration_seconds_max'] * 1e6:.1f} us"
    )
    print(
        "Detailed IBM-style estimate (1 sub-job): "
        f"{runtime_estimate['estimated_qpu_seconds_detailed_one_subjob']:.1f} s "
        f"= {runtime_estimate['estimated_qpu_minutes_detailed_one_subjob']:.2f} min"
    )
    print(
        "IBM quick-formula estimate: "
        f"{runtime_estimate['estimated_qpu_seconds_quick_formula']:.1f} s "
        f"= {runtime_estimate['estimated_qpu_minutes_quick_formula']:.2f} min"
    )
    print(
        "Runtime submission guard: "
        f"{args.max_estimated_qpu_seconds:.1f} s"
    )
    print(
        "NOTE: service-side splitting can add ~2 s per additional sub-job."
    )
    print(f"Preflight written: {args.preflight}")

    if summary["total_executions"] > args.max_executions:
        die(
            "\nREFUSING SUBMISSION: total execution count exceeds guard.\n"
            f"Observed {summary['total_executions']:,}; "
            f"guard {args.max_executions:,}.\n"
            "Do not increase the guard until the QPU budget has been reviewed."
        )

    detailed_estimate = float(
        runtime_estimate["estimated_qpu_seconds_detailed_one_subjob"]
    )
    quick_estimate = float(
        runtime_estimate["estimated_qpu_seconds_quick_formula"]
    )
    conservative_estimate = max(detailed_estimate, quick_estimate)

    if conservative_estimate > args.max_estimated_qpu_seconds:
        print(
            "\nRUNTIME GUARD: SUBMISSION WOULD BE REFUSED."
        )
        print(
            f"Conservative local estimate: {conservative_estimate:.1f} s "
            f"({conservative_estimate/60.0:.2f} min)"
        )
        print(
            f"Configured limit: {args.max_estimated_qpu_seconds:.1f} s "
            f"({args.max_estimated_qpu_seconds/60.0:.2f} min)"
        )
        if args.submit:
            die(
                "REFUSING SUBMISSION: estimated QPU usage exceeds runtime guard."
            )

    if not args.submit:
        print("\nPREFLIGHT COMPLETE.")
        print("NO QPU JOB WAS SUBMITTED.")
        print("\nIf the preflight numbers are accepted, submit with:")
        print(
            f"  python -X utf8 -u .\\{Path(__file__).name} --submit"
        )
        return

    print("\nSUBMIT ENABLED.")
    print(
        f"Submitting {len(isa_circuits)} circuits x {summary['shots']} shots "
        f"= {summary['total_executions']:,} executions..."
    )

    sampler = SamplerV2(mode=backend)
    job = sampler.run(isa_circuits, shots=summary["shots"])
    job_id = str(job.job_id())

    print(f"Job ID: {job_id}")
    print("Waiting for QPU result...")

    primitive_result = job.result()

    if len(primitive_result) != len(isa_circuits):
        die(
            f"Unexpected result length: {len(primitive_result)} "
            f"for {len(isa_circuits)} circuits."
        )

    measured_p1 = []
    raw_counts = []
    for pub in primitive_result:
        counts = extract_counts(pub)
        raw_counts.append(counts)
        measured_p1.append(p1_from_counts(counts))

    seed_rows = []

    for row in rows:
        out = {"seed": row["seed"]}

        for model in ("REAL", "NULL"):
            key = model.lower()
            info = row[key]
            start = info["slice_start"]
            stop = info["slice_stop"]

            measured = np.asarray(measured_p1[start:stop], dtype=float)
            target = np.asarray(info["target"], dtype=float)
            positions = np.asarray(info["positions"], dtype=int)

            p, e, n = rank_metrics(
                measured_source=measured,
                target=target,
                positions=positions,
                k=args.k,
            )

            out[f"{key}_p"] = p
            out[f"{key}_eval"] = e
            out[f"{key}_ndcg"] = n
            out[f"{key}_pairs"] = int(info["candidate_count"])

        out["delta_p"] = out["real_p"] - out["null_p"]
        out["delta_ndcg"] = out["real_ndcg"] - out["null_ndcg"]
        out["eval_saved"] = out["null_eval"] - out["real_eval"]
        seed_rows.append(out)

    d_p = np.asarray([r["delta_p"] for r in seed_rows], dtype=float)
    d_n = np.asarray([r["delta_ndcg"] for r in seed_rows], dtype=float)
    d_e = np.asarray([r["eval_saved"] for r in seed_rows], dtype=float)

    p_mean, p_low, p_high = bootstrap_paired_ci(
        d_p, BOOTSTRAPS, BOOTSTRAP_SEED
    )
    n_mean, n_low, n_high = bootstrap_paired_ci(
        d_n, BOOTSTRAPS, BOOTSTRAP_SEED + 1
    )
    e_mean, e_low, e_high = bootstrap_paired_ci(
        d_e, BOOTSTRAPS, BOOTSTRAP_SEED + 2
    )

    p_sign = base.exact_sign_test_positive(d_p)
    p_wilc = base.wilcoxon_signed_rank_positive(d_p)

    c1 = p_mean > 0.0
    c2 = p_low > 0.0
    c3 = float(p_sign[2]) < 0.05
    c4 = float(p_wilc[1]) < 0.05
    passed = all((c1, c2, c3, c4))

    final = {
        **preflight,
        "submitted": True,
        "job_id": job_id,
        "result_received_utc": datetime.now(timezone.utc).isoformat(),
        "seed_rows": seed_rows,
        "primary_summary": {
            "mean_real_precision_at_8":
                float(np.mean([r["real_p"] for r in seed_rows])),
            "mean_null_precision_at_8":
                float(np.mean([r["null_p"] for r in seed_rows])),
            "mean_delta_precision_at_8": p_mean,
            "bootstrap_95_ci_delta_precision_at_8": [p_low, p_high],
            "one_sided_sign_test_p": float(p_sign[2]),
            "one_sided_wilcoxon_p": float(p_wilc[1]),
            "mean_eval_saved": e_mean,
            "bootstrap_95_ci_eval_saved": [e_low, e_high],
            "mean_delta_ndcg_at_8": n_mean,
            "bootstrap_95_ci_delta_ndcg_at_8": [n_low, n_high],
            "C1_mean_delta_precision_gt_0": c1,
            "C2_bootstrap_lower_gt_0": c2,
            "C3_sign_p_lt_0_05": c3,
            "C4_wilcoxon_p_lt_0_05": c4,
            "PASS": passed,
        },
        "per_circuit": [
            {
                **meta,
                "measured_p1": float(p1),
                "counts": counts,
            }
            for meta, p1, counts in zip(
                circuit_meta, measured_p1, raw_counts
            )
        ],
    }

    args.output.write_text(json.dumps(final, indent=2), encoding="utf-8")

    print("\n=== PRIMARY HARDWARE RESULT ===")
    print(
        f"Mean Precision@8 REAL: "
        f"{final['primary_summary']['mean_real_precision_at_8']:.6f}"
    )
    print(
        f"Mean Precision@8 NULL: "
        f"{final['primary_summary']['mean_null_precision_at_8']:.6f}"
    )
    print(f"Mean delta Precision@8: {p_mean:+.6f}")
    print(f"95% bootstrap CI: [{p_low:+.6f}, {p_high:+.6f}]")
    print(f"One-sided sign-test p: {float(p_sign[2]):.8f}")
    print(f"One-sided Wilcoxon p: {float(p_wilc[1]):.8f}")
    print(f"C1: {'PASS' if c1 else 'FAIL'}")
    print(f"C2: {'PASS' if c2 else 'FAIL'}")
    print(f"C3: {'PASS' if c3 else 'FAIL'}")
    print(f"C4: {'PASS' if c4 else 'FAIL'}")
    print(f"FINAL v40.6 HARDWARE DECISION: {'PASS' if passed else 'FAIL'}")
    print(f"Result written: {args.output}")


if __name__ == "__main__":
    main()
