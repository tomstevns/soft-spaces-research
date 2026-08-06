#!/usr/bin/env python3
import argparse
import importlib.util
import os
import re
import sys

POS_LINE_RE = re.compile(
    r"pos=\((\d+),\s*(\d+)\)\s*\|\s*repeats=(\d+)/(\d+)\s*\|.*?"
    r"fav_rate=([+\-]?\d*\.?\d+)\s*\|\s*anti_rate=([+\-]?\d*\.?\d+)\s*\|"
    r"\s*stab=([+\-]?\d*\.?\d+)\s*\|\s*label=([A-Za-z_]+)"
)

REGION_LINE_RE = re.compile(
    r"region=\((\d+),\s*(\d+)\)\s*\|\s*repeats=(\d+)/(\d+)\s*\|.*?"
    r"fav_rate=([+\-]?\d*\.?\d+)\s*\|\s*anti_rate=([+\-]?\d*\.?\d+)\s*\|"
    r"\s*stab=([+\-]?\d*\.?\d+)\s*\|\s*label=([A-Za-z_]+)"
)

def parse_tuple_text(s):
    m = re.match(r"\s*(\d+)\s*,\s*(\d+)\s*$", s)
    if not m:
        raise ValueError(f"Could not parse tuple from {s!r}")
    return int(m.group(1)), int(m.group(2))

def parse_args():
    ap = argparse.ArgumentParser(description="v24.15 local ridge-refinement REAL-only wrapper")
    ap.add_argument("--merger_txt", required=True)
    ap.add_argument("--prior_apply_to", choices=["real", "both", "null"], default="real")
    ap.add_argument("--region_width", type=int, default=8)
    ap.add_argument("--weight_power", type=float, default=1.0)
    ap.add_argument("--ridge_region_bonus_scale", type=float, default=0.260)
    ap.add_argument("--ridge_anchor_bonus_scale", type=float, default=0.110)
    ap.add_argument("--ridge_support_bonus_scale", type=float, default=0.070)
    ap.add_argument("--outside_region_penalty_scale", type=float, default=0.015)
    ap.add_argument("--top_support_points", type=int, default=4)
    ap.add_argument("--min_repeat_rate_support", type=float, default=0.66)
    ap.add_argument("--min_stab_support", type=float, default=0.30)
    ap.add_argument("--dominant_anchor", type=str, default=None)
    ap.add_argument("--dominant_region", type=str, default=None)
    ap.add_argument("--wrapper_help", action="store_true")
    args, unknown = ap.parse_known_args()
    args.base_passthrough = unknown
    return args

def load_base_module():
    here = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(here, "v1_0_Soft_Space_4_8_qubits_v24_9_2.py")
    if not os.path.exists(base_path):
        raise FileNotFoundError(
            f"Base engine not found at {base_path}. "
            "Place this file next to v1_0_Soft_Space_4_8_qubits_v24_9_2.py"
        )
    spec = importlib.util.spec_from_file_location("softspace_v24_9_2_base", base_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

def parse_merger_txt(path):
    positions = []
    regions = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            pm = POS_LINE_RE.search(s)
            if pm:
                positions.append({
                    "pos": (int(pm.group(1)), int(pm.group(2))),
                    "repeats": int(pm.group(3)),
                    "total_repeats": int(pm.group(4)),
                    "fav_rate": float(pm.group(5)),
                    "anti_rate": float(pm.group(6)),
                    "stab": float(pm.group(7)),
                    "label": pm.group(8),
                })
                continue
            rm = REGION_LINE_RE.search(s)
            if rm:
                regions.append({
                    "region": (int(rm.group(1)), int(rm.group(2))),
                    "repeats": int(rm.group(3)),
                    "total_repeats": int(rm.group(4)),
                    "fav_rate": float(rm.group(5)),
                    "anti_rate": float(rm.group(6)),
                    "stab": float(rm.group(7)),
                    "label": rm.group(8),
                })
    return {"positions": positions, "regions": regions}

def choose_primary_region(parsed, override):
    if override is not None:
        return override
    if not parsed["regions"]:
        return (19, 19)
    return max(
        parsed["regions"],
        key=lambda x: (x["repeats"], x["fav_rate"] - x["anti_rate"], x["stab"], x["fav_rate"])
    )["region"]

def choose_primary_anchor(parsed, primary_region, region_width, override):
    if override is not None:
        return override
    in_region = [
        p for p in parsed["positions"]
        if (p["pos"][0] // region_width, p["pos"][1] // region_width) == primary_region
    ]
    if not in_region:
        return (157, 158)
    return max(
        in_region,
        key=lambda x: (x["repeats"], x["fav_rate"] - x["anti_rate"], x["stab"], x["fav_rate"])
    )["pos"]

def choose_support_points(parsed, primary_region, region_width, primary_anchor, top_support_points, min_repeat_rate_support, min_stab_support):
    support = []
    for p in parsed["positions"]:
        pos = p["pos"]
        region = (pos[0] // region_width, pos[1] // region_width)
        repeat_rate = p["repeats"] / max(1, p["total_repeats"])
        if region != primary_region or pos == primary_anchor:
            continue
        if repeat_rate < min_repeat_rate_support:
            continue
        if p["stab"] < min_stab_support:
            continue
        support.append(p)
    support.sort(
        key=lambda x: (
            -x["repeats"],
            -(x["fav_rate"] - x["anti_rate"]),
            -x["stab"],
            abs(x["pos"][0] - primary_anchor[0]) + abs(x["pos"][1] - primary_anchor[1]),
        )
    )
    pts = [x["pos"] for x in support[:top_support_points]]
    if not pts:
        defaults = [(153, 154), (155, 156)]
        pts = [p for p in defaults if p != primary_anchor]
    return pts

def parse_engine_output_counts(text):
    candidates_seen = 0
    model_counts = {}
    for m in re.finditer(r"Model:\s*([A-Za-z0-9_]+)\s*\|\s*candidates=(\d+)", text):
        model = m.group(1)
        count = int(m.group(2))
        model_counts[model] = model_counts.get(model, 0) + count
        candidates_seen += count
    return candidates_seen, model_counts

def build_summary(args, primary_region, primary_anchor, support_points, adjusted, model_counts, candidates_seen):
    lines = []
    lines.append("=== v24.15 prior application summary ===")
    lines.append("v24.15 local ridge-refinement REAL-only summary")
    lines.append(f"apply_to={args.prior_apply_to} | weight_power={args.weight_power:.3f}")
    lines.append(
        f"primary_region={primary_region}/anchor={primary_anchor}/"
        f"region_bonus={args.ridge_region_bonus_scale:.3f}/"
        f"anchor_bonus={args.ridge_anchor_bonus_scale:.3f}/"
        f"support_bonus={args.ridge_support_bonus_scale:.3f}/"
        f"outside_penalty={args.outside_region_penalty_scale:.3f}"
    )
    lines.append(f"support_points={support_points} | count={len(support_points)}")
    lines.append(f"region_priors=1 | candidates_seen={candidates_seen} | adjusted={adjusted}")
    lines.append(f"model_counts={model_counts}")
    lines.append("Local ridge-refinement priors were applied before stable selection.")
    return "\n".join(lines) + "\n"

def main():
    args = parse_args()
    if args.wrapper_help:
        print("v24.15 wrapper loaded. Use --merger_txt plus base engine args.")
        return

    primary_region_override = parse_tuple_text(args.dominant_region) if args.dominant_region else None
    primary_anchor_override = parse_tuple_text(args.dominant_anchor) if args.dominant_anchor else None

    parsed = parse_merger_txt(args.merger_txt)
    primary_region = choose_primary_region(parsed, primary_region_override)
    primary_anchor = choose_primary_anchor(parsed, primary_region, args.region_width, primary_anchor_override)
    support_points = choose_support_points(
        parsed, primary_region, args.region_width, primary_anchor,
        args.top_support_points, args.min_repeat_rate_support, args.min_stab_support
    )

    base = load_base_module()
    original_run = base.run_paired_batches

    def patched_run_paired_batches(*run_args, **run_kwargs):
        real, null, scoreboards, perturb_summaries, open_system_summaries, interference_summaries = original_run(*run_args, **run_kwargs)

        def should_apply(model_name):
            if args.prior_apply_to == "both":
                return model_name in ("REAL", "NULL_HAAR_BASIS")
            if args.prior_apply_to == "real":
                return model_name == "REAL"
            if args.prior_apply_to == "null":
                return model_name == "NULL_HAAR_BASIS"
            return False

        adjusted = 0

        def adjust_bucket(cands, model_name):
            nonlocal adjusted
            if not should_apply(model_name):
                return
            for c in cands:
                pos = (int(c.i), int(c.j))
                region = (pos[0] // args.region_width, pos[1] // args.region_width)
                bonus = 0.0
                if region == primary_region:
                    bonus += args.ridge_region_bonus_scale
                else:
                    bonus -= args.outside_region_penalty_scale
                if pos == primary_anchor:
                    bonus += args.ridge_anchor_bonus_scale
                elif pos in support_points:
                    bonus += args.ridge_support_bonus_scale
                if args.weight_power != 1.0 and bonus != 0.0:
                    sign = 1.0 if bonus >= 0 else -1.0
                    bonus = sign * (abs(bonus) ** args.weight_power)
                c.score = float(c.score + bonus)
                adjusted += 1

        adjust_bucket(real, "REAL")
        adjust_bucket(null, "NULL_HAAR_BASIS")

        base._v24_15_adjusted = adjusted
        return real, null, scoreboards, perturb_summaries, open_system_summaries, interference_summaries

    base.run_paired_batches = patched_run_paired_batches

    sys.argv = [sys.argv[0]] + args.base_passthrough
    base.main()

    out_path = None
    for i, tok in enumerate(args.base_passthrough):
        if tok == "--output" and i + 1 < len(args.base_passthrough):
            out_path = args.base_passthrough[i + 1]
            break

    if out_path and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        candidates_seen, model_counts = parse_engine_output_counts(text)
        adjusted = int(getattr(base, "_v24_15_adjusted", 0))
        summary = build_summary(args, primary_region, primary_anchor, support_points, adjusted, model_counts, candidates_seen)
        with open(out_path, "a", encoding="utf-8") as f:
            if not text.endswith("\n"):
                f.write("\n")
            f.write("\n")
            f.write(summary)

if __name__ == "__main__":
    main()
