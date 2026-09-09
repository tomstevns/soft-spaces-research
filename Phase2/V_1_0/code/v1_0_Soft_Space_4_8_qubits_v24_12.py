import argparse
import importlib.util
import json
import os
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
        return argparse.Namespace(prior_json=None, prior_apply_to="both", prior_region_bonus_scale=None, prior_anchor_bonus_scale=None, prior_weight_power=1.0, prior_debug_top=10, wrapper_help=True), []
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--prior_json", required=True)
    parser.add_argument("--prior_apply_to", choices=["both", "real", "null"], default="both")
    parser.add_argument("--prior_region_bonus_scale", type=float, default=None)
    parser.add_argument("--prior_anchor_bonus_scale", type=float, default=None)
    parser.add_argument("--prior_weight_power", type=float, default=1.0)
    parser.add_argument("--prior_debug_top", type=int, default=10)
    parser.add_argument("--wrapper_help", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


class PriorApplier:
    def __init__(self, prior_pack: Dict[str, Any], apply_to: str, region_bonus_scale: Optional[float], anchor_bonus_scale: Optional[float], weight_power: float):
        self.prior_pack = prior_pack
        self.apply_to = apply_to
        self.region_bonus_scale = float(region_bonus_scale if region_bonus_scale is not None else prior_pack["selection_policy"]["region_bonus_scale"])
        self.anchor_bonus_scale = float(anchor_bonus_scale if anchor_bonus_scale is not None else prior_pack["selection_policy"]["anchor_bonus_scale"])
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
            w = float(rp.get("target_weight", 0.0)) ** self.weight_power
            win = rp.get("window")
            if win and len(win) == 4:
                i0, i1, j0, j1 = map(int, win)
                if i0 <= i <= i1 and j0 <= j <= j1:
                    rb = self.region_bonus_scale * w
                    if rb > best_region_bonus:
                        best_region_bonus = rb
                        best_region = tuple(rp.get("region", [None, None]))
            anchor = rp.get("best_anchor")
            if anchor and len(anchor) == 2 and int(anchor[0]) == int(i) and int(anchor[1]) == int(j):
                ab = self.anchor_bonus_scale * w
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
        return (
            "v24.12 soft-region-prior summary\n"
            f"apply_to={self.apply_to} | region_bonus_scale={self.region_bonus_scale:.3f} | "
            f"anchor_bonus_scale={self.anchor_bonus_scale:.3f} | weight_power={self.weight_power:.3f}\n"
            f"region_priors={len(self.region_priors)} | candidates_seen={self.total_candidates} | adjusted={self.total_adjusted}\n"
            f"model_counts={self.model_counts}"
        )


def main():
    wrapper_args, remaining = parse_wrapper_args(sys.argv[1:])
    if wrapper_args.wrapper_help:
        print("Wrapper args: --prior_json PATH [--prior_apply_to both|real|null] [--prior_region_bonus_scale X] [--prior_anchor_bonus_scale X] [--prior_weight_power X]")
        return

    here = Path(__file__).resolve().parent
    base_path = here / "v1_0_Soft_Space_4_8_qubits_v24_9_2.py"
    if not base_path.exists():
        raise FileNotFoundError(f"Base file not found next to wrapper: {base_path}")
    prior_path = Path(wrapper_args.prior_json)
    if not prior_path.exists():
        raise FileNotFoundError(f"Prior JSON not found: {prior_path}")

    prior_pack = json.loads(prior_path.read_text(encoding="utf-8"))
    base = load_base_module(base_path)
    applier = PriorApplier(
        prior_pack=prior_pack,
        apply_to=wrapper_args.prior_apply_to,
        region_bonus_scale=wrapper_args.prior_region_bonus_scale,
        anchor_bonus_scale=wrapper_args.prior_anchor_bonus_scale,
        weight_power=wrapper_args.prior_weight_power,
    )

    orig_generate = base.generate_candidates_for_seed
    def wrapped_generate_candidates_for_seed(*args, **kwargs):
        model = kwargs.get("model")
        if model is None and len(args) >= 1:
            model = args[0]
        cands = orig_generate(*args, **kwargs)
        return applier.apply(str(model), cands)

    base.generate_candidates_for_seed = wrapped_generate_candidates_for_seed

    orig_run = base.run_paired_batches
    def wrapped_run_paired_batches(*args, **kwargs):
        return orig_run(*args, **kwargs)
    base.run_paired_batches = wrapped_run_paired_batches

    print("=== v24.12: soft region-prior simulator-side wrapper ===")
    print(f"Prior JSON: {prior_path}")
    print(applier.summary_text())
    print("Notes: applies soft score bonus inside target windows and best anchors before stable selection.")

    # Forward the remaining args to the base script.
    sys.argv = [str(base_path)] + remaining
    try:
        base.main()
    finally:
        # Attempt to append a compact summary into the output file if one was requested.
        out_path = None
        for idx, tok in enumerate(remaining):
            if tok == "--output" and idx + 1 < len(remaining):
                out_path = remaining[idx + 1]
                break
        if out_path:
            try:
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write("\n\n=== v24.12 prior application summary ===\n")
                    f.write(applier.summary_text() + "\n")
                    f.write("Soft region priors were applied before stable selection.\n")
            except Exception:
                pass


if __name__ == "__main__":
    main()
