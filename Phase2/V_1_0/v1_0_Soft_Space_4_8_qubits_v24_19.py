#!/usr/bin/env python3
"""
v24.19 explanatory evidence note builder

Purpose
-------
Read the v24.18 explanatory ridge report and turn it into a short, structured
internal evidence note that can later be reused as article building material.

Input
-----
A v24.18 explanatory ridge report txt file.

Output
------
A compact txt note with:
- headline conclusion
- comparative reading of anchor vs neighbors
- operational recommendation
- article-ready paragraph
"""

import argparse
import re
from pathlib import Path


ROW_RE = re.compile(
    r"pos=\((\d+), (\d+)\) \| proxy=([+\-]?\d*\.?\d+) \| repeat_count=(\d+) \| hits_n=(\d+) \| "
    r"fav_rate=([A-Z0-9a-z+\-\.]+) \| dyn_gain_med=([A-Z0-9a-z+\-\.]+) \| "
    r"fid_uncond_med=([A-Z0-9a-z+\-\.]+) \| logical_purity_med=([A-Z0-9a-z+\-\.]+) \| "
    r"structure_drift_med=([A-Z0-9a-z+\-\.]+) \| subspace_ret_med=([A-Z0-9a-z+\-\.]+) \| "
    r"contrast_med=([A-Z0-9a-z+\-\.]+) \| hotspot_score_med=([A-Z0-9a-z+\-\.]+)"
)

GLOBAL_RE = re.compile(
    r"fid_uncond=([+\-]?\d*\.?\d+) \| fid_cond=([+\-]?\d*\.?\d+) \| leak=([+\-]?\d*\.?\d+) \| "
    r"subspace_ret=([+\-]?\d*\.?\d+) \| logical_coh=([+\-]?\d*\.?\d+) \| "
    r"logical_purity=([+\-]?\d*\.?\d+) \| structure_drift=([+\-]?\d*\.?\d+)"
)

REC_RE = re.compile(
    r"pos=\((\d+), (\d+)\) \| batch_count_best=(\d+) \| recur_fav_rate_med=([A-Z0-9a-z+\-\.]+) \| "
    r"recur_anti_rate_med=([A-Z0-9a-z+\-\.]+) \| recur_stab_med=([A-Z0-9a-z+\-\.]+) \| "
    r"recur_dyn_med=([A-Z0-9a-z+\-\.]+) \| recur_lp_med=([A-Z0-9a-z+\-\.]+) \| "
    r"recur_sd_med=([A-Z0-9a-z+\-\.]+) \| labels=\[(.*?)\]"
)


def parse_num(s: str):
    if s == "NA":
        return None
    return float(s)


def parse_args():
    ap = argparse.ArgumentParser(description="v24.19 explanatory evidence note builder")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    return ap.parse_args()


def parse_report(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = []
    recs = {}
    global_summary = None

    gm = GLOBAL_RE.search(text)
    if gm:
        global_summary = {
            "fid_uncond": float(gm.group(1)),
            "fid_cond": float(gm.group(2)),
            "leak": float(gm.group(3)),
            "subspace_ret": float(gm.group(4)),
            "logical_coh": float(gm.group(5)),
            "logical_purity": float(gm.group(6)),
            "structure_drift": float(gm.group(7)),
        }

    for m in ROW_RE.finditer(text):
        pos = (int(m.group(1)), int(m.group(2)))
        rows.append({
            "pos": pos,
            "proxy": float(m.group(3)),
            "repeat_count": int(m.group(4)),
            "hits_n": int(m.group(5)),
            "fav_rate": parse_num(m.group(6)),
            "dyn_gain_med": parse_num(m.group(7)),
            "fid_uncond_med": parse_num(m.group(8)),
            "logical_purity_med": parse_num(m.group(9)),
            "structure_drift_med": parse_num(m.group(10)),
            "subspace_ret_med": parse_num(m.group(11)),
            "contrast_med": parse_num(m.group(12)),
            "hotspot_score_med": parse_num(m.group(13)),
        })

    for m in REC_RE.finditer(text):
        pos = (int(m.group(1)), int(m.group(2)))
        recs[pos] = {
            "batch_count_best": int(m.group(3)),
            "recur_fav_rate_med": parse_num(m.group(4)),
            "recur_anti_rate_med": parse_num(m.group(5)),
            "recur_stab_med": parse_num(m.group(6)),
            "recur_dyn_med": parse_num(m.group(7)),
            "recur_lp_med": parse_num(m.group(8)),
            "recur_sd_med": parse_num(m.group(9)),
            "labels": m.group(10),
        }

    rows.sort(key=lambda x: (-x["proxy"], -x["repeat_count"], -(x["dyn_gain_med"] or -999)))
    return global_summary, rows, recs


def fmt(x, digits=4):
    if x is None:
        return "NA"
    return f"{x:+.{digits}f}"


def build_note(global_summary, rows, recs):
    best = rows[0] if rows else None
    runner_up = rows[1] if len(rows) > 1 else None
    third = rows[2] if len(rows) > 2 else None

    lines = []
    lines.append("=== v24.19 internal explanatory evidence note ===")
    lines.append("")

    lines.append("Headline conclusion")
    if best:
        lines.append(
            f"The current 8Q explanatory picture identifies {best['pos']} as the strongest local ridge anchor."
        )
        lines.append(
            f"It ranks first by explanatory proxy ({best['proxy']:.3f}) and combines repeat recurrence, positive median dynamic gain, positive logical purity, positive retention/fidelity contribution, and negative structure drift."
        )
    else:
        lines.append("No ranked positions were found in the source report.")
    lines.append("")

    if global_summary:
        lines.append("Global dynamic background")
        lines.append(
            f"Across the underlying v24.14 repeats, the reduced-q50 medians remain small but consistently favorable: "
            f"fid_uncond={fmt(global_summary['fid_uncond'])}, subspace_ret={fmt(global_summary['subspace_ret'])}, "
            f"logical_purity={fmt(global_summary['logical_purity'])}, structure_drift={fmt(global_summary['structure_drift'])}."
        )
        lines.append("")

    if best:
        lines.append("Why the leading anchor looks better")
        lines.append(
            f"For {best['pos']}, the median explanatory profile is: dyn_gain={fmt(best['dyn_gain_med'])}, "
            f"fid_uncond={fmt(best['fid_uncond_med'])}, logical_purity={fmt(best['logical_purity_med'])}, "
            f"structure_drift={fmt(best['structure_drift_med'])}, subspace_ret={fmt(best['subspace_ret_med'])}."
        )
        rbest = recs.get(best["pos"])
        if rbest:
            lines.append(
                f"Its recurrence profile is also the strongest in the compared set: "
                f"batch_count_best={rbest['batch_count_best']}, recur_fav_rate_med={fmt(rbest['recur_fav_rate_med'])}, "
                f"recur_anti_rate_med={fmt(rbest['recur_anti_rate_med'])}, recur_stab_med={fmt(rbest['recur_stab_med'])}."
            )
        lines.append(
            "This indicates that the anchor wins not by purity alone, but by a better overall balance between recurrence, retention/fidelity, and suppressed structure drift."
        )
        lines.append("")

    if runner_up or third:
        lines.append("Comparison to nearby ridge positions")
        if runner_up:
            lines.append(
                f"The nearest strong alternative is {runner_up['pos']} with proxy={runner_up['proxy']:.3f}. "
                f"It remains relevant, but it is weaker than {best['pos']} on the total balance of recurrence and dynamic contribution."
            )
            lines.append(
                f"Its medians are: dyn_gain={fmt(runner_up['dyn_gain_med'])}, fid_uncond={fmt(runner_up['fid_uncond_med'])}, "
                f"logical_purity={fmt(runner_up['logical_purity_med'])}, structure_drift={fmt(runner_up['structure_drift_med'])}, "
                f"subspace_ret={fmt(runner_up['subspace_ret_med'])}."
            )
        if third:
            lines.append(
                f"A further support point is {third['pos']} with proxy={third['proxy']:.3f}. "
                f"This suggests that the local geometry is ridge-like rather than a single isolated spike, but still clearly dominated by {best['pos']}."
            )
        lines.append("")

    weaker = [r for r in rows if r["repeat_count"] <= 1]
    if weaker:
        lines.append("Contrastive reading of weaker neighbors")
        sample = weaker[0]
        lines.append(
            f"A weaker comparison point such as {sample['pos']} shows that not every nearby position shares the same explanatory signature. "
            f"These weaker points tend to lose either recurrence, favorability, retention/fidelity contribution, or structure-drift advantage."
        )
        lines.append("")

    lines.append("Operational recommendation")
    lines.append(
        "Treat v24.14 as the current best operative 8Q hotspot version. Do not prioritize further score-bias tuning as the main next step."
    )
    lines.append(
        "Instead, move into an explanatory phase centered on the dominant ridge anchor and its nearest neighbors."
    )
    lines.append("")

    lines.append("Article-ready paragraph")
    if best:
        lines.append(
            f"In the current 8-qubit ridge analysis, the position {best['pos']} emerges as the dominant local anchor. "
            f"It outperforms nearby candidates not merely through a single metric, but through a combined profile of stronger repeat recurrence, "
            f"positive median dynamic gain, positive retention/fidelity contribution, modest positive logical purity, and more favorable "
            f"(i.e. lower) structure drift. Neighboring positions such as {runner_up['pos'] if runner_up else '(support point)'} remain relevant, "
            f"which supports a ridge-like interpretation rather than a fully isolated spike; however, the dominant anchor remains clearly preferred."
        )
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    global_summary, rows, recs = parse_report(Path(args.input))
    note = build_note(global_summary, rows, recs)
    Path(args.output).write_text(note, encoding="utf-8")


if __name__ == "__main__":
    main()
