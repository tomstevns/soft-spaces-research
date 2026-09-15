#!/usr/bin/env python3
"""
Soft Spaces Phase 4 v40.4 — IBM Heron hardware preparation / freeze.

PURPOSE
-------
Prepare (but do NOT submit) the first real-hardware Soft Spaces Phase-4 test.

Primary backend:
    ibm_aachen  (IBM Heron r3, eu-de)

What this program does:
1. Connects to IBM Quantum using the user's configured QiskitRuntimeService.
2. Verifies that the requested backend is Heron and operational.
3. Reads the CURRENT calibration snapshot.
4. Builds the native 2Q connectivity graph.
5. Selects a low-error connected 8-qubit subgraph.
6. Orders those 8 physical qubits for use as a frozen initial layout.
7. Saves a JSON manifest containing backend, processor revision, calibration
   timestamp, physical qubits, readout errors, 2Q edges/errors, and protocol.
8. DOES NOT submit any QPU job.

Scientific principle:
The hardware comparison should keep the logical REAL/NULL experiment as close
as possible to v40.3b while allowing only unavoidable hardware-specific
translation/routing.

Recommended environment:
    qiskit~=2.5
    qiskit-ibm-runtime~=0.47

Run:
    python -X utf8 -u .\v40_4_ibm_heron_prepare.py

Optional:
    python -X utf8 -u .\v40_4_ibm_heron_prepare.py --backend ibm_aachen

Output:
    v40_4_ibm_heron_frozen_manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path


N_LOGICAL = 8
DEFAULT_BACKEND = "ibm_aachen"
DEFAULT_OUTPUT = Path("v40_4_ibm_heron_frozen_manifest.json")
BEAM_WIDTH = 600

# Scoring weights. 2Q error dominates because the Phase-4 stress channel
# deliberately uses many entangling gates.
W_2Q = 1.0
W_READOUT = 0.20


def finite(x):
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False


def get_processor_type(backend):
    pt = getattr(backend, "processor_type", None)
    return pt if isinstance(pt, dict) else {}


def get_measure_error(backend, q):
    try:
        prop = backend.target["measure"][(q,)]
        err = getattr(prop, "error", None)
        if finite(err):
            return float(err)
    except Exception:
        pass
    try:
        props = backend.properties()
        err = props.readout_error(q)
        if finite(err):
            return float(err)
    except Exception:
        pass
    return None


def native_2q_candidates(backend):
    preferred = ("cz", "ecr", "cx")
    names = set(getattr(backend.target, "operation_names", []))
    return [g for g in preferred if g in names]


def collect_graph(backend):
    gate_names = native_2q_candidates(backend)
    if not gate_names:
        raise RuntimeError("No supported native 2Q gate (cz/ecr/cx) found.")

    adjacency = defaultdict(set)
    edge_error = {}
    edge_gate = {}

    # Prefer the first available gate family, typically CZ on Heron.
    gate = gate_names[0]
    inst = backend.target[gate]

    for qargs, props in inst.items():
        if qargs is None or len(qargs) != 2:
            continue
        a, b = int(qargs[0]), int(qargs[1])
        err = getattr(props, "error", None)
        if not finite(err):
            continue
        key = tuple(sorted((a, b)))
        e = float(err)
        if key not in edge_error or e < edge_error[key]:
            edge_error[key] = e
            edge_gate[key] = gate
        adjacency[a].add(b)
        adjacency[b].add(a)

    if not edge_error:
        raise RuntimeError(f"No calibrated {gate} edges found.")

    return gate, adjacency, edge_error, edge_gate


def connected(nodes, adjacency):
    nodes = set(nodes)
    if not nodes:
        return False
    start = next(iter(nodes))
    seen = {start}
    dq = deque([start])
    while dq:
        u = dq.popleft()
        for v in adjacency[u]:
            if v in nodes and v not in seen:
                seen.add(v)
                dq.append(v)
    return len(seen) == len(nodes)


def subgraph_edges(nodes, edge_error):
    s = set(nodes)
    out = []
    for (a, b), err in edge_error.items():
        if a in s and b in s:
            out.append((a, b, err))
    return out


def score_set(nodes, adjacency, edge_error, readout):
    if not connected(nodes, adjacency):
        return float("inf")

    edges = subgraph_edges(nodes, edge_error)
    if len(edges) < len(nodes) - 1:
        return float("inf")

    # Use the best n-1 internal calibrated edges as an optimistic connected
    # error proxy, plus all-qubit readout quality.
    e2 = sorted(e for _, _, e in edges)
    mean_2q = sum(e2[: max(1, len(nodes)-1)]) / max(1, len(nodes)-1)

    ros = [readout[q] for q in nodes if finite(readout.get(q))]
    mean_ro = sum(ros) / len(ros) if ros else 1.0

    # Small penalty for weak internal connectivity.
    edge_penalty = 0.0002 * max(0, (len(nodes)-1) - len(edges))
    return W_2Q * mean_2q + W_READOUT * mean_ro + edge_penalty


def beam_select_8(adjacency, edge_error, readout, n=8):
    eligible = sorted(q for q, nbs in adjacency.items() if nbs)
    beam = [frozenset([q]) for q in eligible]

    for size in range(2, n + 1):
        cand = {}
        for s in beam:
            frontier = set()
            for q in s:
                frontier.update(adjacency[q])
            frontier.difference_update(s)

            for q in frontier:
                ns = frozenset(set(s) | {q})
                if len(ns) != size:
                    continue
                sc = score_set(ns, adjacency, edge_error, readout)
                old = cand.get(ns)
                if old is None or sc < old:
                    cand[ns] = sc

        if not cand:
            raise RuntimeError(f"Could not construct a connected {size}-qubit set.")
        beam = [s for s, _ in sorted(cand.items(), key=lambda kv: kv[1])[:BEAM_WIDTH]]

    best = min(beam, key=lambda s: score_set(s, adjacency, edge_error, readout))
    return sorted(best), score_set(best, adjacency, edge_error, readout)


def shortest_distances(root, nodes, adjacency):
    allowed = set(nodes)
    dist = {root: 0}
    dq = deque([root])
    while dq:
        u = dq.popleft()
        for v in adjacency[u]:
            if v in allowed and v not in dist:
                dist[v] = dist[u] + 1
                dq.append(v)
    return dist


def order_layout(nodes, adjacency, edge_error, readout):
    # Logical q0 is the central control/readout qubit in the v40.3b circuit.
    # Choose the physical qubit with high degree, low local 2Q error and low
    # readout error as physical location of logical q0.
    node_set = set(nodes)

    def root_score(q):
        internal = [n for n in adjacency[q] if n in node_set]
        errs = []
        for n in internal:
            key = tuple(sorted((q, n)))
            if key in edge_error:
                errs.append(edge_error[key])
        mean_e = sum(errs)/len(errs) if errs else 1.0
        ro = readout.get(q)
        ro = ro if finite(ro) else 1.0
        # lower is better; degree rewarded
        return mean_e + 0.2*ro - 0.0005*len(internal)

    root = min(nodes, key=root_score)
    dist = shortest_distances(root, nodes, adjacency)

    rest = [q for q in nodes if q != root]
    rest.sort(key=lambda q: (
        dist.get(q, 99),
        readout.get(q) if finite(readout.get(q)) else 1.0,
        q
    ))
    return [root] + rest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        import qiskit
        import qiskit_ibm_runtime
        from qiskit_ibm_runtime import QiskitRuntimeService
    except Exception as exc:
        raise SystemExit(
            "Missing IBM Runtime packages. Install in .venv_qiskit:\n"
            '  python -m pip install "qiskit~=2.5" "qiskit-ibm-runtime~=0.47"\n'
            f"Original error: {exc}"
        )

    service = QiskitRuntimeService()
    backend = service.backend(args.backend)

    status = backend.status()
    pt = get_processor_type(backend)

    print(f"Backend: {backend.name}")
    print(f"Processor: {pt}")
    print(f"Operational: {status.operational}")
    print(f"Pending jobs: {status.pending_jobs}")

    family = str(pt.get("family", "")).lower()
    if family and family != "heron":
        raise SystemExit(f"Refusing non-Heron backend: processor_type={pt}")

    gate, adjacency, edge_error, edge_gate = collect_graph(backend)

    readout = {q: get_measure_error(backend, q) for q in range(backend.num_qubits)}

    nodes, set_score = beam_select_8(adjacency, edge_error, readout, N_LOGICAL)
    layout = order_layout(nodes, adjacency, edge_error, readout)

    edges = sorted(
        [(a, b, e) for a, b, e in subgraph_edges(nodes, edge_error)],
        key=lambda x: (x[0], x[1]),
    )

    # Calibration timestamp if exposed through legacy properties.
    calibration_timestamp = None
    try:
        props = backend.properties(refresh=True)
        ts = getattr(props, "last_update_date", None)
        if ts is not None:
            calibration_timestamp = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
    except Exception:
        pass

    manifest = {
        "protocol_version": "v40.4-heron-prep-1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend.name,
        "processor_type": pt,
        "backend_num_qubits": backend.num_qubits,
        "backend_operational_at_freeze": bool(status.operational),
        "pending_jobs_at_freeze": int(status.pending_jobs),
        "calibration_timestamp": calibration_timestamp,
        "native_2q_gate_used_for_selection": gate,
        "selected_physical_qubits_sorted": nodes,
        "frozen_initial_layout_logical_q0_to_q7": layout,
        "selection_score": set_score,
        "readout_error_by_physical_qubit": {
            str(q): readout[q] for q in layout
        },
        "internal_2q_edges": [
            {"q0": a, "q1": b, "error": e, "gate": edge_gate[tuple(sorted((a,b)))]}
            for a, b, e in edges
        ],
        "hardware_protocol": {
            "logical_qubits": 8,
            "primary_cx_checkpoint": 20,
            "shots": 4096,
            "seed_ensemble": "25043000..25043011",
            "models": ["REAL", "NULL"],
            "application": "-E prominence -> measured source ranking -> +E ranked search",
            "primary_metric": "Precision@8",
            "secondary_metrics": ["target evaluations", "NDCG@8"],
            "transpiler_optimization_level": 0,
            "fixed_initial_layout": True,
            "error_mitigation": "none for primary raw comparison",
            "twirling": "disabled unless explicitly added in a separate secondary analysis",
            "dynamical_decoupling": "disabled for primary comparison",
            "zne": "disabled for primary comparison",
            "submission": "NOT performed by this preparation script",
        },
        "freeze_rule": (
            "Once this manifest is accepted for the hardware run, do not change "
            "the backend, physical qubits, initial layout, shots, seed ensemble, "
            "or REAL/NULL analysis rules based on observed outcomes."
        ),
    }

    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nSelected connected 8Q subgraph:")
    print("  physical qubits:", nodes)
    print("  logical q0..q7 layout:", layout)
    print("  selection score:", f"{set_score:.8g}")
    print("  native 2Q gate:", gate)
    print("  internal calibrated 2Q edges:", len(edges))
    print("\nFrozen manifest written to:")
    print(" ", args.output)
    print("\nNO QPU JOB WAS SUBMITTED.")


if __name__ == "__main__":
    main()
