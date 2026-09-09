#!/usr/bin/env python3
import argparse
import ast
import os
import re
import statistics
from collections import defaultdict

LINE_RE = re.compile(
    r"pos=\((?P<i>\d+),\s*(?P<j>\d+)\)"
    r"\s*\|\s*count=(?P<count>\d+)"
    r"\s*\|\s*batch_count=(?P<batch_count>\d+)"
    r"\s*\|\s*batches=(?P<batches>\[[^\]]*\])"
    r"\s*\|\s*favorable=(?P<favorable>\d+)\/\d+"
    r"\s*\|\s*anti=(?P<anti>\d+)\/\d+"
    r"\s*\|\s*fav_rate=(?P<fav_rate>[+\-]?\d*\.?\d+)"
    r"\s*\|\s*anti_rate=(?P<anti_rate>[+\-]?\d*\.?\d+)"
    r"\s*\|\s*stab=(?P<stab>[+\-]?\d*\.?\d+)"
    r"\s*\|\s*label=(?P<label>[A-Za-z_]+)"
    r"\s*\|\s*dyn_gain_med=(?P<dyn_gain>[+\-]?\d*\.?\d+)"
    r"\s*\|\s*logical_purity_med=(?P<logical_purity>[+\-]?\d*\.?\d+)"
    r"\s*\|\s*structure_drift_med=(?P<structure_drift>[+\-]?\d*\.?\d+)"
)

DELTA_RE = re.compile(
    r"REAL_Q4 - NULL_Q4 deltas \(reduced q50\):"
    r".*?logical_purity=(?P<logical_purity>[+\-]?\d*\.?\d+)"
    r"\s*\|\s*structure_drift=(?P<structure_drift>[+\-]?\d*\.?\d+)"
)

BASE_SEED_RE = re.compile(r"base_seed=(\d+)")
QUBITS_RE = re.compile(r"Qubits:\s*(\d+)")


def median(xs):
    return statistics.median(xs) if xs else 0.0


def fmt(x: float) -> str:
    return f"{x:+.4f}"


def classify(repeats, favored, anti):
    if repeats >= 2 and favored > 0 and anti == 0:
        return "robust_favorable"
    if repeats >= 2:
        return "mixed_recurrent"
    return "single_repeat"


def parse_file(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    text = "".join(lines)

    base_seed = int(BASE_SEED_RE.search(text).group(1)) if BASE_SEED_RE.search(text) else None
    n_qubits = int(QUBITS_RE.search(text).group(1)) if QUBITS_RE.search(text) else None

    global_lp = []
    global_sd = []
    for m in DELTA_RE.finditer(text):
        global_lp.append(float(m.group("logical_purity")))
        global_sd.append(float(m.group("structure_drift")))

    entries = []
    in_by_pos = False
    for line in lines:
        s = line.strip()
        if s == "by_pos:":
            in_by_pos = True
            continue
        if in_by_pos and (s.startswith("by_pos_realcoarse:") or s.startswith("=== ") or not s):
            in_by_pos = False
        if not in_by_pos:
            continue
        m = LINE_RE.search(s)
        if not m:
            continue
        batches = ast.literal_eval(m.group("batches"))
        entries.append(
            {
                "pos": (int(m.group("i")), int(m.group("j"))),
                "count": int(m.group("count")),
                "batch_count": int(m.group("batch_count")),
                "batches": batches,
                "favorable": int(m.group("favorable")),
                "anti": int(m.group("anti")),
                "fav_rate": float(m.group("fav_rate")),
                "anti_rate": float(m.group("anti_rate")),
                "stab": float(m.group("stab")),
                "label": m.group("label"),
                "dyn_gain_med": float(m.group("dyn_gain")),
                "logical_purity_med": float(m.group("logical_purity")),
                "structure_drift_med": float(m.group("structure_drift")),
            }
        )
    return {
        "path": path,
        "file": os.path.basename(path),
        "base_seed": base_seed,
        "n_qubits": n_qubits,
        "entries": entries,
        "global_lp": median(global_lp),
        "global_sd": median(global_sd),
    }


def merge(parsed, region_width):
    exact = defaultdict(list)
    region = defaultdict(list)
    for ridx, item in enumerate(parsed, start=1):
        for e in item["entries"]:
            e2 = dict(e)
            e2["repeat_id"] = ridx
            exact[e["pos"]].append(e2)
            region_key = (e["pos"][0] // region_width, e["pos"][1] // region_width)
            region[region_key].append(e2)
    return exact, region


def summarize_bucket(bucket, total_repeats):
    repeat_ids = sorted({x["repeat_id"] for x in bucket})
    repeats = len(repeat_ids)
    count = len(bucket)
    favored = sum(1 for x in bucket if x["favorable"] > x["anti"])
    anti = sum(1 for x in bucket if x["anti"] > x["favorable"])
    fav_rate = favored / count if count else 0.0
    anti_rate = anti / count if count else 0.0
    stab = fav_rate - anti_rate
    return {
        "repeats": repeats,
        "repeat_rate": repeats / total_repeats if total_repeats else 0.0,
        "count": count,
        "favored": favored,
        "anti": anti,
        "fav_rate": fav_rate,
        "anti_rate": anti_rate,
        "stab": stab,
        "label": classify(repeats, favored, anti),
        "dyn_gain_med": median([x["dyn_gain_med"] for x in bucket]),
        "logical_purity_med": median([x["logical_purity_med"] for x in bucket]),
        "structure_drift_med": median([x["structure_drift_med"] for x in bucket]),
        "repeat_ids": repeat_ids,
    }


def sort_key(item):
    key, s = item
    return (
        s["repeats"],
        s["favored"] - s["anti"],
        s["stab"],
        s["dyn_gain_med"],
        s["logical_purity_med"],
        -s["structure_drift_med"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--region_width", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    parsed = [parse_file(p) for p in args.inputs]
    total_repeats = len(parsed)
    exact, region = merge(parsed, args.region_width)

    exact_summ = {k: summarize_bucket(v, total_repeats) for k, v in exact.items()}
    region_summ = {k: summarize_bucket(v, total_repeats) for k, v in region.items()}

    global_lp = median([p["global_lp"] for p in parsed])
    global_sd = median([p["global_sd"] for p in parsed])

    out = []
    out.append("=== v24.12 REAL-only hotspot merger ===")
    out.append(f"inputs={len(parsed)} | region_width={args.region_width}")
    for idx, p in enumerate(parsed, start=1):
        out.append(
            f"repeat {idx}: file={p['file']} | base_seed={p['base_seed']} | hotspots={len(p['entries'])} | "
            f"logical_purity_med={fmt(p['global_lp'])} | structure_drift_med={fmt(p['global_sd'])}"
        )
    out.append("")
    out.append("=== Global dynamic summary across inputs ===")
    out.append(f"logical_purity_med={fmt(global_lp)} | structure_drift_med={fmt(global_sd)}")
    out.append("")
    out.append("=== Cross-repeat hotspot recurrence by exact position (v24.12 REAL-only merger) ===")
    for pos, s in sorted(exact_summ.items(), key=sort_key, reverse=True)[: args.top_k]:
        out.append(
            f"pos={pos} | repeats={s['repeats']}/{total_repeats} | repeat_rate={s['repeat_rate']:.2f} | count={s['count']} | "
            f"favored={s['favored']}/{s['count']} | anti={s['anti']}/{s['count']} | "
            f"fav_rate={s['fav_rate']:.2f} | anti_rate={s['anti_rate']:.2f} | stab={fmt(s['stab'])} | label={s['label']} | "
            f"dyn_gain_med={fmt(s['dyn_gain_med'])} | logical_purity_med={fmt(s['logical_purity_med'])} | "
            f"structure_drift_med={fmt(s['structure_drift_med'])} | repeat_ids={s['repeat_ids']}"
        )
    out.append("")
    out.append("=== Cross-repeat hotspot recurrence by region (v24.12 REAL-only merger) ===")
    for reg, s in sorted(region_summ.items(), key=sort_key, reverse=True)[: args.top_k]:
        out.append(
            f"region={reg} | repeats={s['repeats']}/{total_repeats} | repeat_rate={s['repeat_rate']:.2f} | count={s['count']} | "
            f"favored={s['favored']}/{s['count']} | anti={s['anti']}/{s['count']} | "
            f"fav_rate={s['fav_rate']:.2f} | anti_rate={s['anti_rate']:.2f} | stab={fmt(s['stab'])} | label={s['label']} | "
            f"dyn_gain_med={fmt(s['dyn_gain_med'])} | logical_purity_med={fmt(s['logical_purity_med'])} | "
            f"structure_drift_med={fmt(s['structure_drift_med'])} | repeat_ids={s['repeat_ids']}"
        )
    out.append("")
    out.append("=== v24.12 REAL-only reading ===")
    out.append("1) Use exact-position recurrence to test whether the same anchor survives across independent repeats.")
    out.append("2) Use region recurrence to test whether a broader family remains stable even when the exact anchor drifts.")
    out.append("3) If exact anchors fragment but nearby regions remain recurrent, move the next simulator version toward region-broad REAL-only priors.")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
