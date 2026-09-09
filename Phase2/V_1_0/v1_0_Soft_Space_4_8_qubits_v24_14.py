import argparse
import importlib.util
import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_base_module(base_path: Path):
    spec = importlib.util.spec_from_file_location("soft_space_v24_9_2_base", str(base_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base module from {base_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_wrapper_args(argv: List[str]) -> Tuple[argparse.Namespace, List[str]]:
    if "--wrapper_help" in argv:
        return argparse.Namespace(
            merger_txt=None,
            prior_apply_to="real",
            top_regions=2,
            primary_region_bonus_scale=0.280,
            primary_anchor_bonus_scale=0.080,
            secondary_region_bonus_scale=0.120,
            secondary_anchor_bonus_scale=0.030,
            weight_power=1.0,
            require_recurrent=True,
            require_primary_robust=True,
            wrapper_help=True,
        ), []
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--merger_txt", required=True)
    parser.add_argument("--prior_apply_to", choices=["both", "real", "null"], default="real")
    parser.add_argument("--top_regions", type=int, default=2)
    parser.add_argument("--primary_region_bonus_scale", type=float, default=0.280)
    parser.add_argument("--primary_anchor_bonus_scale", type=float, default=0.080)
    parser.add_argument("--secondary_region_bonus_scale", type=float, default=0.120)
    parser.add_argument("--secondary_anchor_bonus_scale", type=float, default=0.030)
    parser.add_argument("--weight_power", type=float, default=1.0)
    parser.add_argument("--allow_single_repeat", action="store_true")
    parser.add_argument("--allow_nonrobust_primary", action="store_true")
    parser.add_argument("--wrapper_help", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    args.require_recurrent = not args.allow_single_repeat
    args.require_primary_robust = not args.allow_nonrobust_primary
    return args, remaining


def _parse_list_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.strip().strip('[]').split(',') if x.strip()]


def parse_merger_report(path: Path) -> Dict[str, Any]:
    txt = path.read_text(encoding="utf-8", errors="replace")
    region_width_match = re.search(r"region_width=(\d+)", txt)
    region_width = int(region_width_match.group(1)) if region_width_match else 8

    global_match = re.search(
        r"=== Global dynamic summary across inputs ===\nlogical_purity_med=([+\-0-9.]+) \| structure_drift_med=([+\-0-9.]+)",
        txt,
    )
    global_dynamic = {
        "logical_purity_med": float(global_match.group(1)) if global_match else 0.0,
        "structure_drift_med": float(global_match.group(2)) if global_match else 0.0,
    }

    def extract_section(title_regex: str) -> List[str]:
        pattern = rf"=== {title_regex} ===\n(.*?)(?:\n=== |\Z)"
        m = re.search(pattern, txt, flags=re.S)
        if not m:
            return []
        return [ln.strip() for ln in m.group(1).splitlines() if ln.strip()]

    pos_lines = extract_section(r"Cross-repeat hotspot recurrence by exact position \(v24\.13 merger\)")
    region_lines = extract_section(r"Cross-repeat hotspot recurrence by region \(v24\.13 merger\)")

    def parse_common_lines(lines: List[str], kind: str) -> List[Dict[str, Any]]:
        out = []
        for ln in lines:
            m = re.match(
                rf"{kind}=\((\d+),\s*(\d+)\) \| repeats=(\d+)/(\d+) \| repeat_rate=([0-9.]+) \| count=(\d+) \| favored=(\d+)/(\d+) \| anti=(\d+)/(\d+) \| fav_rate=([0-9.]+) \| anti_rate=([0-9.]+) \| stab=([+\-0-9.]+) \| label=([A-Za-z_]+) \| dyn_gain_med=([+\-0-9.]+) \| logical_purity_med=([+\-0-9.]+) \| structure_drift_med=([+\-0-9.]+) \| repeat_ids=\[(.*?)\]$",
                ln,
            )
            if not m:
                continue
            key = "pos" if kind == "pos" else "region"
            out.append({
                key: (int(m.group(1)), int(m.group(2))),
                "repeats": int(m.group(3)),
                "repeat_total": int(m.group(4)),
                "repeat_rate": float(m.group(5)),
                "count": int(m.group(6)),
                "favored": int(m.group(7)),
                "anti": int(m.group(9)),
                "fav_rate": float(m.group(11)),
                "anti_rate": float(m.group(12)),
                "stab": float(m.group(13)),
                "label": m.group(14),
                "dyn_gain_med": float(m.group(15)),
                "logical_purity_med": float(m.group(16)),
                "structure_drift_med": float(m.group(17)),
                "repeat_ids": _parse_list_ints(m.group(18)),
            })
        return out

    return {
        "region_width": region_width,
        "global_dynamic": global_dynamic,
        "position_entries": parse_common_lines(pos_lines, "pos"),
        "region_entries": parse_common_lines(region_lines, "region"),
    }


def region_to_window(region: Tuple[int, int], region_width: int) -> Tuple[int, int, int, int]:
    i0 = region[0] * region_width
    i1 = i0 + region_width - 1
    j0 = region[1] * region_width
    j1 = j0 + region_width - 1
    return i0, i1, j0, j1


def build_v24_14_prior_pack(
    report: Dict[str, Any],
    top_regions: int,
    primary_region_bonus_scale: float,
    primary_anchor_bonus_scale: float,
    secondary_region_bonus_scale: float,
    secondary_anchor_bonus_scale: float,
    require_recurrent: bool,
    require_primary_robust: bool,
) -> Dict[str, Any]:
    region_entries = report["region_entries"]
    pos_entries = report["position_entries"]
    region_width = report["region_width"]

    def region_score(x: Dict[str, Any]) -> Tuple:
        return (
            x["repeats"],
            x["fav_rate"],
            -x["anti_rate"],
            x["stab"],
            x["dyn_gain_med"],
            x["logical_purity_med"],
            -x["structure_drift_med"],
        )

    candidates = [x for x in region_entries if (x["repeats"] >= 2 if require_recurrent else True)]
    if not candidates:
        candidates = list(region_entries)
    candidates = sorted(candidates, key=region_score, reverse=True)
    selected = candidates[: max(1, top_regions)]
    if not selected:
        raise RuntimeError("No merger regions could be selected for v24.14")

    pos_by_region: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for p in pos_entries:
        rg = (p["pos"][0] // region_width, p["pos"][1] // region_width)
        pos_by_region.setdefault(rg, []).append(p)

    region_priors = []
    for idx, r in enumerate(selected):
        rg = r["region"]
        anchors = sorted(
            pos_by_region.get(rg, []),
            key=lambda p: (
                p["repeats"], p["fav_rate"], -p["anti_rate"], p["stab"], p["dyn_gain_med"], p["logical_purity_med"], -p["structure_drift_med"]
            ),
            reverse=True,
        )
        best_anchor = anchors[0]["pos"] if anchors else None
        role = "primary" if idx == 0 else "secondary"
        if role == "primary":
            region_bonus_scale = primary_region_bonus_scale
            anchor_bonus_scale = primary_anchor_bonus_scale
        else:
            region_bonus_scale = secondary_region_bonus_scale
            anchor_bonus_scale = secondary_anchor_bonus_scale
        if role == "primary" and require_primary_robust and r["label"] != "robust_favorable":
            raise RuntimeError("Primary region from merger is not robust_favorable; rerun with --allow_nonrobust_primary if intentional.")
        region_priors.append({
            "region": list(rg),
            "window": list(region_to_window(rg, region_width)),
            "role": role,
            "label": r["label"],
            "repeat_rate": r["repeat_rate"],
            "fav_rate": r["fav_rate"],
            "anti_rate": r["anti_rate"],
            "stab": r["stab"],
            "dyn_gain_med": r["dyn_gain_med"],
            "logical_purity_med": r["logical_purity_med"],
            "structure_drift_med": r["structure_drift_med"],
            "region_bonus_scale": float(region_bonus_scale),
            "anchor_bonus_scale": float(anchor_bonus_scale),
            "best_anchor": list(best_anchor) if best_anchor else None,
            "best_anchor_repeats": anchors[0]["repeats"] if anchors else None,
            "best_anchor_fav_rate": anchors[0]["fav_rate"] if anchors else None,
            "best_anchor_label": anchors[0]["label"] if anchors else None,
        })

    return {
        "version": "v24.14",
        "source": "v24.13 dual-region REAL-only merger",
        "region_width": region_width,
        "global_dynamic_summary": report["global_dynamic"],
        "selection_policy": {
            "mode": "asymmetric_dual_region_real_only_prior",
            "base_score": "existing_score",
            "bonus_formula": "base_score - region_bonus(role) - anchor_bonus(role)",
            "primary_region_bonus_scale": float(primary_region_bonus_scale),
            "primary_anchor_bonus_scale": float(primary_anchor_bonus_scale),
            "secondary_region_bonus_scale": float(secondary_region_bonus_scale),
            "secondary_anchor_bonus_scale": float(secondary_anchor_bonus_scale),
            "exploration_fraction": 0.20,
            "hard_constraints": False,
            "notes": [
                "Keep REAL-only targeting.",
                "Use the merger winner as the primary region and the runner-up as a softer secondary region.",
                "Default expectation from v24.13 merger: region (19,19) primary, region (12,12) secondary.",
            ],
        },
        "region_priors": region_priors,
    }


class PriorApplier:
    def __init__(self, prior_pack: Dict[str, Any], apply_to: str, weight_power: float):
        self.prior_pack = prior_pack
        self.apply_to = apply_to
        self.weight_power = float(weight_power)
        self.region_priors = prior_pack.get("region_priors", [])
        self.total_adjusted = 0
        self.total_candidates = 0
        self.model_counts = {"REAL": 0, "NULL_HAAR_BASIS": 0}

    def model_enabled(self, model: str) -> bool:
        if self.apply_to == "both":
            return True
        if self.apply_to == "real":
            return model == "REAL"
        if self.apply_to == "null":
            return model == "NULL_HAAR_BASIS"
        return True

    def compute_bonus(self, i: int, j: int) -> Tuple[float, float, Optional[Tuple[int, int]]]:
        best_region_bonus = 0.0
        best_anchor_bonus = 0.0
        best_region = None
        for rp in self.region_priors:
            quality = max(0.0, float(rp.get("fav_rate", 0.0)) - float(rp.get("anti_rate", 0.0)))
            weight = max(1e-9, quality) ** self.weight_power
            win = rp.get("window")
            if win and len(win) == 4:
                i0, i1, j0, j1 = map(int, win)
                if i0 <= i <= i1 and j0 <= j <= j1:
                    rb = float(rp.get("region_bonus_scale", 0.0)) * weight
                    if rb > best_region_bonus:
                        best_region_bonus = rb
                        best_region = tuple(rp.get("region", [None, None]))
            anchor = rp.get("best_anchor")
            if anchor and len(anchor) == 2 and int(anchor[0]) == int(i) and int(anchor[1]) == int(j):
                ab = float(rp.get("anchor_bonus_scale", 0.0)) * weight
                if ab > best_anchor_bonus:
                    best_anchor_bonus = ab
        return best_region_bonus, best_anchor_bonus, best_region

    def apply(self, model: str, cands: List[Any]) -> List[Any]:
        self.total_candidates += len(cands)
        self.model_counts[model] = self.model_counts.get(model, 0) + len(cands)
        if not self.model_enabled(model):
            return cands
        out = []
        for c in cands:
            region_bonus, anchor_bonus, _ = self.compute_bonus(int(c.i), int(c.j))
            bonus = region_bonus + anchor_bonus
            if bonus > 0.0:
                self.total_adjusted += 1
                try:
                    c = replace(c, score=max(0.0, float(c.score) - bonus))
                except Exception:
                    c.score = max(0.0, float(c.score) - bonus)
            out.append(c)
        return out

    def summary_text(self) -> str:
        pieces = []
        for rp in self.region_priors:
            pieces.append(
                f"{rp['role']}={tuple(rp['region'])}/anchor={tuple(rp['best_anchor']) if rp.get('best_anchor') else None}/"
                f"rb={rp['region_bonus_scale']:.3f}/ab={rp['anchor_bonus_scale']:.3f}/"
                f"fav={rp['fav_rate']:.2f}/anti={rp['anti_rate']:.2f}/stab={rp['stab']:+.4f}"
            )
        return (
            "v24.14 asymmetric dual-region REAL-only summary\n"
            f"apply_to={self.apply_to} | weight_power={self.weight_power:.3f}\n"
            + " | ".join(pieces)
            + "\n"
            + f"region_priors={len(self.region_priors)} | candidates_seen={self.total_candidates} | adjusted={self.total_adjusted}\n"
            + f"model_counts={self.model_counts}"
        )


def main():
    wrapper_args, remaining = parse_wrapper_args(sys.argv[1:])
    if wrapper_args.wrapper_help:
        print("Wrapper args: --merger_txt PATH [--prior_apply_to both|real|null] [--top_regions 2] [--primary_region_bonus_scale 0.280] [--primary_anchor_bonus_scale 0.080] [--secondary_region_bonus_scale 0.120] [--secondary_anchor_bonus_scale 0.030] [--weight_power 1.0]")
        return

    here = Path(__file__).resolve().parent
    base_path = here / "v1_0_Soft_Space_4_8_qubits_v24_9_2.py"
    if not base_path.exists():
        raise FileNotFoundError(f"Base file not found next to wrapper: {base_path}")
    merger_path = Path(wrapper_args.merger_txt)
    if not merger_path.exists():
        raise FileNotFoundError(f"Merger TXT not found: {merger_path}")

    report = parse_merger_report(merger_path)
    prior_pack = build_v24_14_prior_pack(
        report=report,
        top_regions=wrapper_args.top_regions,
        primary_region_bonus_scale=wrapper_args.primary_region_bonus_scale,
        primary_anchor_bonus_scale=wrapper_args.primary_anchor_bonus_scale,
        secondary_region_bonus_scale=wrapper_args.secondary_region_bonus_scale,
        secondary_anchor_bonus_scale=wrapper_args.secondary_anchor_bonus_scale,
        require_recurrent=wrapper_args.require_recurrent,
        require_primary_robust=wrapper_args.require_primary_robust,
    )

    base = load_base_module(base_path)
    applier = PriorApplier(prior_pack=prior_pack, apply_to=wrapper_args.prior_apply_to, weight_power=wrapper_args.weight_power)

    orig_generate = base.generate_candidates_for_seed
    def wrapped_generate_candidates_for_seed(*args, **kwargs):
        model = kwargs.get("model")
        if model is None and len(args) >= 1:
            model = args[0]
        cands = orig_generate(*args, **kwargs)
        return applier.apply(str(model), cands)
    base.generate_candidates_for_seed = wrapped_generate_candidates_for_seed

    print("=== v24.14: asymmetric dual-region REAL-only wrapper ===")
    print(f"Merger TXT: {merger_path}")
    print(json.dumps(prior_pack, indent=2))
    print(applier.summary_text())
    print("Notes: derived from the v24.13 merger; primary emphasis is placed on the stronger recurrent region while the runner-up remains as a softer secondary target.")

    sys.argv = [str(base_path)] + remaining
    try:
        base.main()
    finally:
        out_path = None
        for idx, tok in enumerate(remaining):
            if tok == "--output" and idx + 1 < len(remaining):
                out_path = remaining[idx + 1]
                break
        if out_path:
            try:
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write("\n\n=== v24.14 prior application summary ===\n")
                    f.write(applier.summary_text() + "\n")
                    f.write("Asymmetric dual-region REAL-only priors were applied before stable selection.\n")
            except Exception:
                pass


if __name__ == "__main__":
    main()
