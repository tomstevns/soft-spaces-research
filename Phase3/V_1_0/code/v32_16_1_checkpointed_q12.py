#!/usr/bin/env python3
"""Soft Spaces Phase 3.2 v32.16.1 — checkpointed 12Q cross-qubit robustness.

Purpose
-------
Run the same degeneracy-closed diagnostics used in v32.14/v32.16 at 12 qubits
without requiring the whole pipeline to finish in one runtime window.

Stages
------
  eig          build H, diagonalize, save eigenvalues/eigenvectors + metadata
  null         generate and save the frozen Haar NULL basis
  score        compute REAL/NULL model scores for both perturbation families
  descriptors  compute prominence and basis-invariant A_group/C_group
  all          run all stages sequentially

The mathematics is unchanged. The optimization is computational:
intermediate results are checkpointed, and degeneracy-closed descriptors use
one U^† V P projection per target eigenspace instead of repeated projections.
"""

from __future__ import annotations
import argparse, csv, importlib.util, json, sys, time
from pathlib import Path
import numpy as np

SEED_DEFAULT = 25_042_000
QUBITS = 12
DIM = 1 << QUBITS
N_TERMS = 11
FAMILIES = ("dephasing", "transverse")


def load_base():
    path = Path(__file__).resolve().parent / "v32_14_degeneracy_closed.py"
    if not path.exists():
        raise FileNotFoundError("v32_14_degeneracy_closed.py must be beside this file.")
    spec = importlib.util.spec_from_file_location("v3214_q12_base", path)
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    m.N_QUBITS = QUBITS
    m.DIM = DIM
    m.N_TERMS = N_TERMS
    return m


def prefix(seed: int) -> Path:
    return Path(__file__).resolve().parent / f"v32_16_1_q12_seed{seed}"


def stage_eig(m, seed: int):
    p = prefix(seed)
    t = time.time()
    h = m.dense_pauli_sum(QUBITS, m.random_hamiltonian_terms(seed))
    print(f"Hamiltonian built: {h.shape}", flush=True)
    t0 = time.time()
    evals, evecs = np.linalg.eigh(h)
    print(f"Diagonalization: {time.time()-t0:.1f}s", flush=True)
    np.save(str(p) + "_evals.npy", evals)
    np.save(str(p) + "_evecs.npy", evecs)
    groups = m.degenerate_groups(evals)
    meta = {
        "seed": seed, "n_qubits": QUBITS, "dimension": DIM,
        "n_terms": N_TERMS, "n_groups": len(groups),
        "group_sizes": [int(len(g)) for g in groups],
    }
    Path(str(p) + "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"eig stage complete: {time.time()-t:.1f}s; groups={len(groups)}")


def stage_null(m, seed: int):
    p = prefix(seed)
    t = time.time()
    null_seed = m.stable_hash_int(
        f"{m.FROZEN_MODEL_VERSION}|NULL|{seed}|terms{m.N_TERMS}"
    )
    u = m.haar_unitary(DIM, null_seed)
    np.save(str(p) + "_null.npy", u)
    print(f"null stage complete: {time.time()-t:.1f}s")


def stage_score(m, seed: int):
    p = prefix(seed)
    evals = np.load(str(p) + "_evals.npy")
    groups = m.degenerate_groups(evals)
    bases = {
        "real": np.load(str(p) + "_evecs.npy", mmap_mode="r"),
        "null": np.load(str(p) + "_null.npy", mmap_mode="r"),
    }
    for family in FAMILIES:
        pert = m.family_terms(
            m.stable_hash_int(f"{m.FROZEN_MODEL_VERSION}|{family}|PERT|{seed}"),
            family,
        )
        for basis_name, vecs in bases.items():
            t = time.time()
            scores = m.model_scores(evals, vecs, pert, groups)
            Path(str(p) + f"_{basis_name}_{family}.json").write_text(
                json.dumps({str(k): float(v) for k, v in scores.items()}, indent=2),
                encoding="utf-8",
            )
            print(f"{basis_name.upper()} {family}: {time.time()-t:.1f}s")


def fast_group_descriptor(m, evals, evecs, groups, target: int, pert):
    p_group = groups[target]
    p_vec = evecs[:, p_group]
    energy = float(np.mean(evals[p_group]))
    acted = m.apply_pauli_sum(p_vec, pert)
    b_to_p = evecs.conj().T @ acted

    terms = []
    for gi, g in enumerate(groups):
        if gi == target:
            continue
        eg = float(np.mean(evals[g]))
        delta = energy - eg
        weight = delta / (delta * delta + m.ENERGY_REG * m.ENERGY_REG)
        block = b_to_p[g, :]
        tg = weight * (block.conj().T @ block)
        terms.append(0.5 * (tg + tg.conj().T))
    return m.interference_descriptors_group_closed(terms)


def _load_scores(p: Path, basis: str, family: str):
    d = json.loads(Path(str(p) + f"_{basis}_{family}.json").read_text(encoding="utf-8"))
    return {int(k): float(v) for k, v in d.items()}


def rankdata(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def stage_descriptors(m, seed: int):
    p = prefix(seed)
    evals = np.load(str(p) + "_evals.npy")
    evecs = np.load(str(p) + "_evecs.npy", mmap_mode="r")
    groups = m.degenerate_groups(evals)
    energies = np.asarray([m.eigenspace_energy(evals, g) for g in groups], float)
    pairs = m.opposite_energy_pairs(energies)

    real_by = {fam: _load_scores(p, "real", fam) for fam in FAMILIES}
    null_by = {fam: _load_scores(p, "null", fam) for fam in FAMILIES}
    robust = m.robust_difference(real_by, null_by, len(groups))
    prom = m.prominence_values(robust, len(groups))

    rows = []
    for family in FAMILIES:
        pert = m.family_terms(
            m.stable_hash_int(f"{m.FROZEN_MODEL_VERSION}|{family}|PERT|{seed}"),
            family,
        )
        t = time.time()
        for a, b in pairs:
            gm, gp = (a, b) if energies[a] <= energies[b] else (b, a)
            pair_prom = max(prom.get(gm, float("-inf")), prom.get(gp, float("-inf")))
            if not np.isfinite(pair_prom):
                continue
            d = fast_group_descriptor(m, evals, evecs, groups, gm, pert)
            rows.append({
                "seed": seed, "family": family, "g_minus": gm, "g_plus": gp,
                "E_abs": float(abs(energies[gp])), "prominence": float(pair_prom),
                "A_group": float(d["A_group"]), "C_group": float(d["C_group"]),
                "cross_sum": float(d["cross_sum"]), "n_qubits": QUBITS,
                "dimension": DIM, "n_groups": len(groups), "n_pairs": len(pairs),
            })
        print(f"{family} descriptors: {time.time()-t:.1f}s")

    rows_path = Path(str(p) + "_rows.csv")
    with rows_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    stats = []
    for fam in FAMILIES:
        rr = [r for r in rows if r["family"] == fam]
        y = [r["prominence"] for r in rr]
        for metric in ("A_group", "C_group", "cross_sum"):
            rho = spearman([r[metric] for r in rr], y)
            stats.append({
                "seed": seed, "family": fam, "metric": metric,
                "spearman": rho, "n_pairs": len(rr), "n_qubits": QUBITS,
            })
            print(f"{fam:10s} {metric:10s} rho={rho:+.4f}")

    stats_path = Path(str(p) + "_stats.csv")
    with stats_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        w.writeheader(); w.writerows(stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--stage", choices=("eig", "null", "score", "descriptors", "all"), default="all")
    args = ap.parse_args()
    m = load_base()
    if args.stage in ("eig", "all"): stage_eig(m, args.seed)
    if args.stage in ("null", "all"): stage_null(m, args.seed)
    if args.stage in ("score", "all"): stage_score(m, args.seed)
    if args.stage in ("descriptors", "all"): stage_descriptors(m, args.seed)


if __name__ == "__main__":
    main()
