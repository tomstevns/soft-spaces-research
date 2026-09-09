#!/usr/bin/env python3
"""
v24.18 explanatory ridge diagnostics from existing v24.14 repeat txt files.

Purpose
-------
Move from hotspot hunting to explanation. This script post-processes the
existing v24.14 repeat outputs and compares a small fixed set of positions:

- strong anchor
- strong neighbors
- weaker / mixed neighbors

It produces a compact report centered on simple explanatory proxies derived
from the hotspot evidence already present in the txt files.

Important note
--------------
This version does NOT reconstruct full Lindblad operators or projectors.
Instead it computes practical diagnostic proxies from the available evidence:
- recurrence strength
- favorability
- dynamic gain
- fid_uncond contribution
- logical_purity contribution
- structure_drift contribution
- subspace_ret contribution

Interpretation
--------------
A position looks more "Lindblad-compatible" when it tends to show:
- higher repeat_count
- higher fav_rate
- higher dynamic_gain
- more negative structure_drift
- positive logical_purity
- positive subspace_ret / fid_uncond
"""

import argparse
import re
from pathlib import Path
from statistics import median

HEADER_RE = re.compile(r'base_seed=(\d+)')
DELTA_RE = re.compile(
    r'REAL_Q4 - NULL_Q4 deltas \(reduced q50\): '
    r'fid_uncond=([+\-]?\d*\.?\d+) \| '
    r'fid_cond=([+\-]?\d*\.?\d+) \| '
    r'leak=([+\-]?\d*\.?\d+) \| '
    r'subspace_ret=([+\-]?\d*\.?\d+) \| '
    r'logical_coh=([+\-]?\d*\.?\d+) \| '
    r'logical_purity=([+\-]?\d*\.?\d+) \| '
    r'structure_drift=([+\-]?\d*\.?\d+)'
)
HOTSPOT_RE = re.compile(
    r'hotspot\(seed,i,j\)=\((\d+),(\d+),(\d+)\) \| pos=\((\d+), (\d+)\) \| '
    r'contrast=([+\-]?\d*\.?\d+) \| hotspot_score=([+\-]?\d*\.?\d+) \| '
    r'dynamic_gain=([+\-]?\d*\.?\d+) \| favorable=(True|False)'
)
HOTSPOT_DELTA_RE = re.compile(
    r'deltas q50: fid_uncond=([+\-]?\d*\.?\d+) \| logical_purity=([+\-]?\d*\.?\d+) '
    r'\| structure_drift=([+\-]?\d*\.?\d+) \| subspace_ret=([+\-]?\d*\.?\d+)'
)
BYPOS_RE = re.compile(
    r'pos=\((\d+), (\d+)\) \| count=(\d+) \| batch_count=(\d+) \| batches=\[[^\]]*\] \| '
    r'favorable=(\d+)/(\d+) \| anti=(\d+)/(\d+) \| fav_rate=([+\-]?\d*\.?\d+) \| '
    r'anti_rate=([+\-]?\d*\.?\d+) \| stab=([+\-]?\d*\.?\d+) \| label=([A-Za-z_]+) \| '
    r'dyn_gain_med=([+\-]?\d*\.?\d+) \| logical_purity_med=([+\-]?\d*\.?\d+) \| '
    r'structure_drift_med=([+\-]?\d*\.?\d+)'
)


def parse_args():
    ap = argparse.ArgumentParser(description="v24.18 explanatory ridge diagnostics")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--positions", nargs="+", required=True,
                    help='Positions like "157,158" "155,156" "153,154" "156,157" "158,159"')
    ap.add_argument("--output", required=True)
    return ap.parse_args()


def parse_pos(s: str):
    m = re.match(r"\s*(\d+)\s*,\s*(\d+)\s*$", s)
    if not m:
        raise ValueError(f"Bad position format: {s!r}")
    return int(m.group(1)), int(m.group(2))


def fmt(x, digits=4):
    if x is None:
        return "NA"
    return f"{x:+.{digits}f}"


def parse_file(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    base_seed = None
    m = HEADER_RE.search(text)
    if m:
        base_seed = int(m.group(1))

    deltas = []
    hotspots = []
    recurrence = []

    pending = None
    in_bypostable = False

    for line in lines:
        dm = DELTA_RE.search(line)
        if dm:
            deltas.append({
                "fid_uncond": float(dm.group(1)),
                "fid_cond": float(dm.group(2)),
                "leak": float(dm.group(3)),
                "subspace_ret": float(dm.group(4)),
                "logical_coh": float(dm.group(5)),
                "logical_purity": float(dm.group(6)),
                "structure_drift": float(dm.group(7)),
            })

        hm = HOTSPOT_RE.search(line)
        if hm:
            pending = {
                "seed": int(hm.group(1)),
                "i": int(hm.group(4)),
                "j": int(hm.group(5)),
                "contrast": float(hm.group(6)),
                "hotspot_score": float(hm.group(7)),
                "dynamic_gain": float(hm.group(8)),
                "favorable": hm.group(9) == "True",
            }
            continue

        if pending is not None:
            hdm = HOTSPOT_DELTA_RE.search(line)
            if hdm:
                pending.update({
                    "fid_uncond": float(hdm.group(1)),
                    "logical_purity": float(hdm.group(2)),
                    "structure_drift": float(hdm.group(3)),
                    "subspace_ret": float(hdm.group(4)),
                })
                hotspots.append(pending)
                pending = None
                continue

        if line.strip() == "by_pos:":
            in_bypostable = True
            continue
        if in_bypostable and line.startswith("by_pos_realcoarse:"):
            in_bypostable = False
            continue
        if in_bypostable:
            bm = BYPOS_RE.search(line.strip())
            if bm:
                recurrence.append({
                    "pos": (int(bm.group(1)), int(bm.group(2))),
                    "count": int(bm.group(3)),
                    "batch_count": int(bm.group(4)),
                    "fav_rate": float(bm.group(9)),
                    "anti_rate": float(bm.group(10)),
                    "stab": float(bm.group(11)),
                    "label": bm.group(12),
                    "dyn_gain_med": float(bm.group(13)),
                    "logical_purity_med": float(bm.group(14)),
                    "structure_drift_med": float(bm.group(15)),
                })

    return {
        "path": str(path),
        "base_seed": base_seed,
        "deltas": deltas,
        "hotspots": hotspots,
        "recurrence": recurrence,
    }


def aggregate_position(parsed, pos):
    # hotspot-level evidence
    hits = []
    for ridx, item in enumerate(parsed, start=1):
        for h in item["hotspots"]:
            if (h["i"], h["j"]) == pos:
                row = dict(h)
                row["repeat_id"] = ridx
                row["base_seed"] = item["base_seed"]
                hits.append(row)

    # recurrence rows per repeat
    recur_rows = []
    for ridx, item in enumerate(parsed, start=1):
        for r in item["recurrence"]:
            if r["pos"] == pos:
                row = dict(r)
                row["repeat_id"] = ridx
                row["base_seed"] = item["base_seed"]
                recur_rows.append(row)

    if hits:
        repeat_count = len({h["repeat_id"] for h in hits})
        fav_rate = sum(1 for h in hits if h["favorable"]) / len(hits)
        dyn_gain_med = median(h["dynamic_gain"] for h in hits)
        fid_uncond_med = median(h["fid_uncond"] for h in hits)
        logical_purity_med = median(h["logical_purity"] for h in hits)
        structure_drift_med = median(h["structure_drift"] for h in hits)
        subspace_ret_med = median(h["subspace_ret"] for h in hits)
        contrast_med = median(h["contrast"] for h in hits)
        score_med = median(h["hotspot_score"] for h in hits)
    else:
        repeat_count = 0
        fav_rate = None
        dyn_gain_med = None
        fid_uncond_med = None
        logical_purity_med = None
        structure_drift_med = None
        subspace_ret_med = None
        contrast_med = None
        score_med = None

    if recur_rows:
        batch_count_best = max(r["batch_count"] for r in recur_rows)
        recur_fav_rate_med = median(r["fav_rate"] for r in recur_rows)
        recur_anti_rate_med = median(r["anti_rate"] for r in recur_rows)
        recur_stab_med = median(r["stab"] for r in recur_rows)
        recur_dyn_med = median(r["dyn_gain_med"] for r in recur_rows)
        recur_lp_med = median(r["logical_purity_med"] for r in recur_rows)
        recur_sd_med = median(r["structure_drift_med"] for r in recur_rows)
        labels = sorted(set(r["label"] for r in recur_rows))
    else:
        batch_count_best = 0
        recur_fav_rate_med = None
        recur_anti_rate_med = None
        recur_stab_med = None
        recur_dyn_med = None
        recur_lp_med = None
        recur_sd_med = None
        labels = []

    # simple explanatory proxy score
    # higher is better: repeat_count, fav_rate, dyn_gain, fid_uncond, logical_purity, subspace_ret, negative structure_drift
    proxy = 0.0
    proxy += 2.0 * repeat_count
    if fav_rate is not None:
        proxy += 3.0 * fav_rate
    if dyn_gain_med is not None:
        proxy += 200.0 * dyn_gain_med
    if fid_uncond_med is not None:
        proxy += 100.0 * fid_uncond_med
    if logical_purity_med is not None:
        proxy += 300.0 * logical_purity_med
    if subspace_ret_med is not None:
        proxy += 100.0 * subspace_ret_med
    if structure_drift_med is not None:
        proxy += -120.0 * structure_drift_med  # more negative drift is better

    return {
        "pos": pos,
        "hits_n": len(hits),
        "repeat_count": repeat_count,
        "fav_rate": fav_rate,
        "dyn_gain_med": dyn_gain_med,
        "fid_uncond_med": fid_uncond_med,
        "logical_purity_med": logical_purity_med,
        "structure_drift_med": structure_drift_med,
        "subspace_ret_med": subspace_ret_med,
        "contrast_med": contrast_med,
        "score_med": score_med,
        "batch_count_best": batch_count_best,
        "recur_fav_rate_med": recur_fav_rate_med,
        "recur_anti_rate_med": recur_anti_rate_med,
        "recur_stab_med": recur_stab_med,
        "recur_dyn_med": recur_dyn_med,
        "recur_lp_med": recur_lp_med,
        "recur_sd_med": recur_sd_med,
        "labels": labels,
        "explanatory_proxy": proxy,
    }


def build_report(parsed, positions):
    lines = []
    lines.append("=== v24.18 explanatory ridge diagnostics ===")
    lines.append(f"inputs={len(parsed)} | positions={positions}")
    lines.append("")

    lines.append("=== Input files ===")
    for idx, item in enumerate(parsed, start=1):
        lines.append(
            f"repeat {idx}: file={Path(item['path']).name} | base_seed={item['base_seed']} | "
            f"hotspots={len(item['hotspots'])} | recurrence_rows={len(item['recurrence'])}"
        )
    lines.append("")

    all_deltas = [d for item in parsed for d in item["deltas"]]
    if all_deltas:
        global_summary = {k: median(d[k] for d in all_deltas) for k in ["fid_uncond","fid_cond","leak","subspace_ret","logical_coh","logical_purity","structure_drift"]}
        lines.append("=== Global reduced-q50 medians across inputs ===")
        lines.append(
            " | ".join([
                f"fid_uncond={fmt(global_summary['fid_uncond'])}",
                f"fid_cond={fmt(global_summary['fid_cond'])}",
                f"leak={fmt(global_summary['leak'])}",
                f"subspace_ret={fmt(global_summary['subspace_ret'])}",
                f"logical_coh={fmt(global_summary['logical_coh'])}",
                f"logical_purity={fmt(global_summary['logical_purity'])}",
                f"structure_drift={fmt(global_summary['structure_drift'])}",
            ])
        )
        lines.append("")

    rows = [aggregate_position(parsed, p) for p in positions]
    rows.sort(key=lambda x: (-x["explanatory_proxy"], -x["repeat_count"], -(x["dyn_gain_med"] or -999)))

    lines.append("=== Position comparison table ===")
    for r in rows:
        lines.append(
            f"pos={r['pos']} | proxy={r['explanatory_proxy']:.3f} | repeat_count={r['repeat_count']} | hits_n={r['hits_n']} | "
            f"fav_rate={fmt(r['fav_rate'])} | dyn_gain_med={fmt(r['dyn_gain_med'])} | fid_uncond_med={fmt(r['fid_uncond_med'])} | "
            f"logical_purity_med={fmt(r['logical_purity_med'])} | structure_drift_med={fmt(r['structure_drift_med'])} | "
            f"subspace_ret_med={fmt(r['subspace_ret_med'])} | contrast_med={fmt(r['contrast_med'])} | hotspot_score_med={fmt(r['score_med'])}"
        )
    lines.append("")

    lines.append("=== Recurrence diagnostics by chosen position ===")
    for r in rows:
        lines.append(
            f"pos={r['pos']} | batch_count_best={r['batch_count_best']} | recur_fav_rate_med={fmt(r['recur_fav_rate_med'])} | "
            f"recur_anti_rate_med={fmt(r['recur_anti_rate_med'])} | recur_stab_med={fmt(r['recur_stab_med'])} | "
            f"recur_dyn_med={fmt(r['recur_dyn_med'])} | recur_lp_med={fmt(r['recur_lp_med'])} | recur_sd_med={fmt(r['recur_sd_med'])} | "
            f"labels={r['labels']}"
        )
    lines.append("")

    if rows:
        best = rows[0]
        lines.append("=== v24.18 reading ===")
        lines.append(
            f"Best explanatory candidate among the chosen positions is {best['pos']} "
            f"with proxy={best['explanatory_proxy']:.3f}."
        )
        lines.append(
            "Read the proxy as a practical summary of: repeat recurrence, favorability, "
            "dynamic gain, positive fid/subspace retention, positive logical purity, and "
            "negative structure drift."
        )
        lines.append(
            "The next explanatory step should compare the strongest position against the "
            "nearest weaker neighbors and ask which term differs most: fid_uncond, "
            "logical_purity, structure_drift, or subspace_ret."
        )
        lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    positions = [parse_pos(s) for s in args.positions]
    parsed = [parse_file(p) for p in args.inputs]
    report = build_report(parsed, positions)
    Path(args.output).write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
