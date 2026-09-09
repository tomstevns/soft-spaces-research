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
    ap = argparse.ArgumentParser(description="v24.16 diagnostic conservative ridge-focus REAL-only wrapper")
    ap.add_argument("--merger_txt", required=True)
    ap.add_argument("--prior_apply_to", choices=["real", "both", "null"], default="real")
    ap.add_argument("--region_width", type=int, default=8)
    ap.add_argument("--weight_power", type=float, default=1.0)
    ap.add_argument("--primary_region_bonus_scale", type=float, default=0.280)
    ap.add_argument("--primary_anchor_bonus_scale", type=float, default=0.080)
    ap.add_argument("--secondary_region_bonus_scale", type=float, default=0.120)
    ap.add_argument("--secondary_anchor_bonus_scale", type=float, default=0.030)
    ap.add_argument("--ridge_support_bonus_scale", type=float, default=0.020)
    ap.add_argument("--top_support_points", type=int, default=3)
    ap.add_argument("--min_repeat_rate_support", type=float, default=0.66)
    ap.add_argument("--min_stab_support", type=float, default=0.30)
    ap.add_argument("--primary_region", type=str, default=None)
    ap.add_argument("--primary_anchor", type=str, default=None)
    ap.add_argument("--secondary_region", type=str, default=None)
    ap.add_argument("--secondary_anchor", type=str, default=None)
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
            "Place this wrapper in the same folder as v1_0_Soft_Space_4_8_qubits_v24_9_2.py"
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

def region_of(pos, region_width):
    return (pos[0] // region_width, pos[1] // region_width)

def select_best_region(regions, override=None, skip=None):
    if override is not None:
        return override
    pool = [r for r in regions if r["region"] != skip]
    if not pool:
        return (19, 19) if skip != (19, 19) else (12, 12)
    return max(pool, key=lambda x: (x["repeats"], x["fav_rate"] - x["anti_rate"], x["stab"], x["fav_rate"]))["region"]

def select_best_anchor(positions, target_region, region_width, override=None):
    if override is not None:
        return override
    pool = [p for p in positions if region_of(p["pos"], region_width) == target_region]
    if not pool:
        return (157, 158) if target_region == (19, 19) else (102, 103)
    return max(pool, key=lambda x: (x["repeats"], x["fav_rate"] - x["anti_rate"], x["stab"], x["fav_rate"]))["pos"]

def choose_support_points(positions, primary_region, primary_anchor, region_width, top_support_points, min_repeat_rate_support, min_stab_support):
    support = []
    for p in positions:
        pos = p["pos"]
        if pos == primary_anchor:
            continue
        if region_of(pos, region_width) != primary_region:
            continue
        repeat_rate = p["repeats"] / max(1, p["total_repeats"])
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
    chosen = [x["pos"] for x in support[:top_support_points]]
    if not chosen:
        defaults = [(153, 154), (155, 156)]
        chosen = [p for p in defaults if p != primary_anchor]
    return chosen

def parse_engine_output_counts(text):
    candidates_seen = 0
    model_counts = {}
    for m in re.finditer(r"Model:\s*([A-Za-z0-9_]+)\s*\|\s*candidates=(\d+)", text):
        model = m.group(1)
        count = int(m.group(2))
        model_counts[model] = model_counts.get(model, 0) + count
        candidates_seen += count
    return candidates_seen, model_counts

def build_summary(args, primary_region, primary_anchor, secondary_region, secondary_anchor, support_points, adjusted, candidates_seen, model_counts):
    lines = []
    lines.append("=== v24.16 prior application summary ===")
    lines.append("v24.16 diagnostic conservative ridge-focus summary")
    lines.append(f"apply_to={args.prior_apply_to} | weight_power={args.weight_power:.3f}")
    lines.append(f"primary={primary_region}/anchor={primary_anchor}/rb={args.primary_region_bonus_scale:.3f}/ab={args.primary_anchor_bonus_scale:.3f}")
    lines.append(f"secondary={secondary_region}/anchor={secondary_anchor}/rb={args.secondary_region_bonus_scale:.3f}/ab={args.secondary_anchor_bonus_scale:.3f}")
    lines.append(f"support_points={support_points} | support_bonus={args.ridge_support_bonus_scale:.3f} | count={len(support_points)}")
    lines.append(f"region_priors=2 | candidates_seen={candidates_seen} | adjusted={adjusted}")
    lines.append(f"model_counts={model_counts}")
    lines.append("v24.16 keeps the v24.14 asymmetry and adds only a mild local ridge emphasis.")
    return "\n".join(lines) + "\n"

def main():
    args = parse_args()
    if args.wrapper_help:
        print("v24.16 wrapper loaded. Use --merger_txt plus base engine args.")
        return

    parsed = parse_merger_txt(args.merger_txt)
    positions = parsed["positions"]
    regions = parsed["regions"]

    primary_region_override = parse_tuple_text(args.primary_region) if args.primary_region else None
    primary_anchor_override = parse_tuple_text(args.primary_anchor) if args.primary_anchor else None
    secondary_region_override = parse_tuple_text(args.secondary_region) if args.secondary_region else None
    secondary_anchor_override = parse_tuple_text(args.secondary_anchor) if args.secondary_anchor else None

    primary_region = select_best_region(regions, primary_region_override)
    primary_anchor = select_best_anchor(positions, primary_region, args.region_width, primary_anchor_override)
    secondary_region = select_best_region(regions, secondary_region_override, skip=primary_region)
    secondary_anchor = select_best_anchor(positions, secondary_region, args.region_width, secondary_anchor_override)
    support_points = choose_support_points(
        positions, primary_region, primary_anchor, args.region_width,
        args.top_support_points, args.min_repeat_rate_support, args.min_stab_support
    )

    base = load_base_module()
    original_generate = base.generate_candidates_for_seed
    base._v24_16_adjusted = 0

    def patched_generate_candidates_for_seed(model, seed, cache, n_terms, eps_neighbor, ent_step, leak_step, times, keep_mass, iso_eps):
        cands = original_generate(model, seed, cache, n_terms, eps_neighbor, ent_step, leak_step, times, keep_mass, iso_eps)

        def should_apply(model_name):
            if args.prior_apply_to == "both":
                return model_name in ("REAL", "NULL_HAAR_BASIS")
            if args.prior_apply_to == "real":
                return model_name == "REAL"
            if args.prior_apply_to == "null":
                return model_name == "NULL_HAAR_BASIS"
            return False

        if not should_apply(model):
            return cands

        out = []
        adjusted = 0
        for c in cands:
            pos = (int(c.i), int(c.j))
            reg = region_of(pos, args.region_width)
            bonus = 0.0

            if reg == primary_region:
                bonus += args.primary_region_bonus_scale
            elif reg == secondary_region:
                bonus += args.secondary_region_bonus_scale

            if pos == primary_anchor:
                bonus += args.primary_anchor_bonus_scale
            elif pos == secondary_anchor:
                bonus += args.secondary_anchor_bonus_scale

            if pos in support_points:
                bonus += args.ridge_support_bonus_scale

            if args.weight_power != 1.0 and bonus != 0.0:
                sign = 1.0 if bonus >= 0 else -1.0
                bonus = sign * (abs(bonus) ** args.weight_power)

            if bonus != 0.0:
                c = type(c)(**{**c.__dict__, "score": float(c.score + bonus)})
                adjusted += 1

            out.append(c)

        base._v24_16_adjusted += adjusted
        return out

    base.generate_candidates_for_seed = patched_generate_candidates_for_seed

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
        summary = build_summary(
            args, primary_region, primary_anchor, secondary_region, secondary_anchor,
            support_points, int(base._v24_16_adjusted), candidates_seen, model_counts
        )
        with open(out_path, "a", encoding="utf-8") as f:
            if not text.endswith("\n"):
                f.write("\n")
            f.write("\n")
            f.write(summary)

if __name__ == "__main__":
    main()
