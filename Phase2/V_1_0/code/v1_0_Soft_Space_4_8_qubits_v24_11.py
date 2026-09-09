#!/usr/bin/env python3
import argparse, json, math, os, re
from typing import List, Dict, Any, Tuple

IMPORTED_RE = re.compile(r"target_region=\((\d+),\s*(\d+)\) \| pos_window=\[(\d+)-(\d+)\]x\[(\d+)-(\d+)\] \| priority=([+\-]?[0-9]*\.?[0-9]+) \| source_label=([^\n]+)")
REGION_DIAG_RE = re.compile(r"region=\((\d+),\s*(\d+)\) \| repeats=(\d+)/(\d+) \| count=(\d+) \| favored=(\d+)/(\d+) \| anti=(\d+)/(\d+) \| dyn_gain_med=([+\-]?[0-9]*\.?[0-9]+) \| logical_purity_med=([+\-]?[0-9]*\.?[0-9]+) \| structure_drift_med=([+\-]?[0-9]*\.?[0-9]+) \| repeat_ids=\[([^\]]*)\]")
ANCHOR_RE = re.compile(r"anchor_pos=\((\d+),\s*(\d+)\) \| region=\((\d+),\s*(\d+)\) \| anchor_priority=([+\-]?[0-9]*\.?[0-9]+) \| repeats=(\d+)/(\d+) \| favored=(\d+)/(\d+) \| dyn_gain_med=([+\-]?[0-9]*\.?[0-9]+) \| logical_purity_med=([+\-]?[0-9]*\.?[0-9]+) \| structure_drift_med=([+\-]?[0-9]*\.?[0-9]+) \| repeat_ids=\[([^\]]*)\]")
GLOBAL_RE = re.compile(r"logical_purity_med=([+\-]?[0-9]*\.?[0-9]+) \| structure_drift_med=([+\-]?[0-9]*\.?[0-9]+)")
DELTA_RE = re.compile(r"targeted_minus_untargeted: dyn_gain=([+\-]?[0-9]*\.?[0-9]+) \| logical_purity=([+\-]?[0-9]*\.?[0-9]+) \| structure_drift=([+\-]?[0-9]*\.?[0-9]+) \| fav_rate=([+\-]?[0-9]*\.?[0-9]+)")


def parse_ids(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def parse_report(path: str) -> Dict[str, Any]:
    txt = open(path, 'r', encoding='utf-8', errors='replace').read()
    out: Dict[str, Any] = {'regions': [], 'anchors': [], 'global': {}, 'targeted_delta': {}}

    imported = {}
    for m in IMPORTED_RE.finditer(txt):
        imported[(int(m.group(1)), int(m.group(2)))] = {
            'window': (int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6))),
            'priority': float(m.group(7)),
            'source_label': m.group(8).strip(),
        }

    for m in REGION_DIAG_RE.finditer(txt):
        reg = (int(m.group(1)), int(m.group(2)))
        meta = imported.get(reg, {})
        repeats = int(m.group(3))
        repeat_total = int(m.group(4))
        favored = int(m.group(6))
        favored_total = int(m.group(7))
        anti = int(m.group(8))
        anti_total = int(m.group(9))
        fav_rate = favored / favored_total if favored_total else 0.0
        anti_rate = anti / anti_total if anti_total else 0.0
        stab = repeat_total and ((fav_rate - anti_rate) * (repeats / repeat_total)) or 0.0
        label = meta.get('source_label', 'unknown')
        out['regions'].append({
            'region': reg,
            'window': meta.get('window'),
            'priority': float(meta.get('priority', 0.0)),
            'repeats': repeats,
            'repeat_total': repeat_total,
            'repeat_rate': repeats / repeat_total if repeat_total else 0.0,
            'favored': favored,
            'favored_total': favored_total,
            'anti': anti,
            'anti_total': anti_total,
            'fav_rate': fav_rate,
            'anti_rate': anti_rate,
            'stab': stab,
            'label': label,
            'dyn_gain_med': float(m.group(10)),
            'logical_purity_med': float(m.group(11)),
            'structure_drift_med': float(m.group(12)),
            'contrast_med': 0.0,
            'repeat_ids': parse_ids(m.group(13)),
        })

    for m in ANCHOR_RE.finditer(txt):
        out['anchors'].append({
            'anchor_pos': (int(m.group(1)), int(m.group(2))),
            'region': (int(m.group(3)), int(m.group(4))),
            'anchor_priority': float(m.group(5)),
            'repeats': int(m.group(6)),
            'repeat_total': int(m.group(7)),
            'favored': int(m.group(8)),
            'favored_total': int(m.group(9)),
            'dyn_gain_med': float(m.group(10)),
            'logical_purity_med': float(m.group(11)),
            'structure_drift_med': float(m.group(12)),
            'repeat_ids': parse_ids(m.group(13)),
        })

    globals_found = GLOBAL_RE.findall(txt)
    if globals_found:
        lp, sd = globals_found[-1]
        out['global'] = {'logical_purity_med': float(lp), 'structure_drift_med': float(sd)}

    deltas = DELTA_RE.findall(txt)
    if deltas:
        dg, lp, sd, fr = deltas[-1]
        out['targeted_delta'] = {'dyn_gain': float(dg), 'logical_purity': float(lp), 'structure_drift': float(sd), 'fav_rate': float(fr)}
    return out


def region_window(region: Tuple[int, int], width: int) -> Tuple[int, int, int, int]:
    a, b = region
    return (a * width, a * width + width - 1, b * width, b * width + width - 1)


def normalize_weights(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    total = sum(max(0.0, x['target_weight_raw']) for x in items)
    for x in items:
        x['target_weight'] = (x['target_weight_raw'] / total) if total > 0 else 0.0
    return items


def build_prior_pack(report: Dict[str, Any], region_width: int, top_k: int, favor_singletons: bool=False) -> Dict[str, Any]:
    regions = sorted(report['regions'], key=lambda r: r['priority'], reverse=True)
    if not favor_singletons:
        regions = [r for r in regions if r['label'] == 'robust_favorable'] + [r for r in regions if r['label'] != 'robust_favorable']
    selected = regions[:top_k]
    anchors_by_region: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for a in report['anchors']:
        anchors_by_region.setdefault(tuple(a['region']), []).append(a)

    priors = []
    for r in selected:
        reg = tuple(r['region'])
        anchors = sorted(anchors_by_region.get(reg, []), key=lambda a: a['anchor_priority'], reverse=True)
        best_anchor = anchors[0] if anchors else None
        raw_weight = max(0.0, r['priority'])
        priors.append({
            'region': reg,
            'window': region_window(reg, region_width),
            'label': r['label'],
            'priority': r['priority'],
            'repeat_rate': r['repeat_rate'],
            'fav_rate': r['fav_rate'],
            'stab': r['stab'],
            'dyn_gain_med': r['dyn_gain_med'],
            'logical_purity_med': r['logical_purity_med'],
            'structure_drift_med': r['structure_drift_med'],
            'contrast_med': r['contrast_med'],
            'target_weight_raw': raw_weight,
            'best_anchor': best_anchor['anchor_pos'] if best_anchor else None,
            'best_anchor_priority': best_anchor['anchor_priority'] if best_anchor else None,
            'best_anchor_repeats': best_anchor['repeats'] if best_anchor else None,
        })
    priors = normalize_weights(priors)

    global_dyn = report.get('global', {})
    targeted_delta = report.get('targeted_delta', {})
    selection_policy = {
        'mode': 'soft_region_prior',
        'base_score': 'existing_score',
        'bonus_formula': 'base_score + region_bonus + anchor_bonus',
        'region_bonus_scale': round(max(0.10, min(0.75, 0.20 + 30 * max(0.0, targeted_delta.get('logical_purity', 0.0)) + 10 * max(0.0, -targeted_delta.get('structure_drift', 0.0)))), 3),
        'anchor_bonus_scale': round(max(0.05, min(0.50, 0.10 + 20 * max(0.0, targeted_delta.get('dyn_gain', 0.0)))), 3),
        'exploration_fraction': 0.25,
        'hard_constraints': False,
        'notes': [
            'Use target regions as priors, not hard filters.',
            'Keep an exploration tail outside target regions to avoid collapse.',
            'Prefer anchors inside top repeat-stable regions when ties occur.'
        ]
    }

    return {
        'version': 'v24.11',
        'region_width': region_width,
        'global_dynamic_summary': global_dyn,
        'targeted_vs_untargeted_delta': targeted_delta,
        'selection_policy': selection_policy,
        'region_priors': priors,
    }


def write_text_report(path: str, pack: Dict[str, Any]) -> None:
    lines = []
    lines.append('=== v24.11 simulator targeting prior pack ===')
    lines.append(f"region_width={pack['region_width']} | priors={len(pack['region_priors'])}")
    g = pack.get('global_dynamic_summary', {})
    if g:
        lines.append(f"global_dynamic: logical_purity_med={g.get('logical_purity_med', 0.0):+0.4f} | structure_drift_med={g.get('structure_drift_med', 0.0):+0.4f}")
    d = pack.get('targeted_vs_untargeted_delta', {})
    if d:
        lines.append(f"targeted_delta: dyn_gain={d.get('dyn_gain', 0.0):+0.4f} | logical_purity={d.get('logical_purity', 0.0):+0.4f} | structure_drift={d.get('structure_drift', 0.0):+0.4f} | fav_rate={d.get('fav_rate', 0.0):+0.2f}")
    pol = pack['selection_policy']
    lines.append('')
    lines.append('=== Suggested soft-prior selection policy ===')
    lines.append(f"mode={pol['mode']} | region_bonus_scale={pol['region_bonus_scale']:.3f} | anchor_bonus_scale={pol['anchor_bonus_scale']:.3f} | exploration_fraction={pol['exploration_fraction']:.2f} | hard_constraints={pol['hard_constraints']}")
    for n in pol['notes']:
        lines.append(f"- {n}")
    lines.append('')
    lines.append('=== Region priors for simulator-side targeting ===')
    for rp in pack['region_priors']:
        w = rp['window']
        ba = rp['best_anchor']
        ba_txt = f"{ba}" if ba is not None else 'None'
        lines.append(
            f"target_region={rp['region']} | pos_window=[{w[0]}-{w[1]}]x[{w[2]}-{w[3]}] | target_weight={rp['target_weight']:.3f} | priority={rp['priority']:.3f} | label={rp['label']} | repeat_rate={rp['repeat_rate']:.2f} | fav_rate={rp['fav_rate']:.2f} | stab={rp['stab']:+.3f} | dyn_gain_med={rp['dyn_gain_med']:+.4f} | logical_purity_med={rp['logical_purity_med']:+.4f} | structure_drift_med={rp['structure_drift_med']:+.4f} | best_anchor={ba_txt}"
        )
    lines.append('')
    lines.append('=== Operational reading (v24.11) ===')
    lines.append('1) Feed the region prior list into simulator-side candidate ranking as a soft bonus, not a hard filter.')
    lines.append('2) Use the best_anchor inside each target region as a tie-break or seed preference.')
    lines.append('3) Compare targeted selection vs baseline on logical_purity, structure_drift, and favorable rate.')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser(description='v24.11 simulator targeting prior pack builder')
    ap.add_argument('--priority_report', required=True)
    ap.add_argument('--region_width', type=int, default=8)
    ap.add_argument('--top_k_regions', type=int, default=6)
    ap.add_argument('--favor_singletons', action='store_true')
    ap.add_argument('--output', required=True, help='Text report output path')
    ap.add_argument('--json_output', default='', help='Optional JSON pack output path')
    args = ap.parse_args()

    report = parse_report(args.priority_report)
    pack = build_prior_pack(report, args.region_width, args.top_k_regions, favor_singletons=args.favor_singletons)
    write_text_report(args.output, pack)
    if args.json_output:
        with open(args.json_output, 'w', encoding='utf-8') as f:
            json.dump(pack, f, indent=2)
    print(f"Wrote {args.output}")
    if args.json_output:
        print(f"Wrote {args.json_output}")

if __name__ == '__main__':
    main()
