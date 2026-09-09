#!/usr/bin/env python3
import argparse, os, re, statistics
from collections import defaultdict

BATCH_RE = re.compile(r'^Batch\s+(\d+)/(\d+)\s+\(seed_offset=(\d+)\)')
HEADER_BASE_RE = re.compile(r'base_seed=(\d+)')
HOTSPOT_RE = re.compile(
    r"hotspot\(seed,i,j\)=\((\d+),(\d+),(\d+)\) \| pos=\((\d+),\s*(\d+)\) \| contrast=([\d.+-eE]+) \| hotspot_score=([\d.+-eE]+) \| dynamic_gain=([+-]?[\d.]+) \| favorable=(True|False)"
)
DELTA_RE = re.compile(
    r"deltas q50: fid_uncond=([+-]?[\d.]+) \| logical_purity=([+-]?[\d.]+) \| structure_drift=([+-]?[\d.]+) \| subspace_ret=([+-]?[\d.]+)"
)
GLOBAL_DELTA_RE = re.compile(
    r"REAL_Q4 - NULL_Q4 deltas \(reduced q50\): .*?logical_purity=([+-]?[\d.]+) \| structure_drift=([+-]?[\d.]+)"
)


def median(vals):
    return statistics.median(vals) if vals else float('nan')


def parse_file(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()
    base_seed = None
    current_batch = None
    pending = None
    hotspots = []
    global_deltas = []
    for line in lines:
        if base_seed is None:
            m = HEADER_BASE_RE.search(line)
            if m:
                base_seed = int(m.group(1))
        m = BATCH_RE.match(line.strip())
        if m:
            current_batch = int(m.group(1))
            continue
        m = GLOBAL_DELTA_RE.search(line)
        if m:
            global_deltas.append((float(m.group(1)), float(m.group(2))))
        m = HOTSPOT_RE.search(line.strip())
        if m and current_batch is not None:
            pending = {
                'seed': int(m.group(1)), 'i': int(m.group(2)), 'j': int(m.group(3)),
                'pos': (int(m.group(4)), int(m.group(5))),
                'contrast': float(m.group(6)), 'hotspot_score': float(m.group(7)),
                'dynamic_gain': float(m.group(8)), 'favorable': m.group(9) == 'True',
                'batch': current_batch,
            }
            hotspots.append(pending)
            continue
        m = DELTA_RE.search(line.strip())
        if m and pending is not None:
            pending.update({
                'fid_uncond': float(m.group(1)),
                'logical_purity': float(m.group(2)),
                'structure_drift': float(m.group(3)),
                'subspace_ret': float(m.group(4)),
            })
            pending = None
    return {'path': path, 'base_seed': base_seed, 'hotspots': hotspots, 'global_deltas': global_deltas}


def classify(appearances, total_repeats, fav_count, anti_count, total_count):
    fav_rate = fav_count / total_count if total_count else 0.0
    anti_rate = anti_count / total_count if total_count else 0.0
    repeat_rate = appearances / total_repeats if total_repeats else 0.0
    stability = repeat_rate * (fav_rate - anti_rate)
    if appearances >= 2 and fav_rate == 1.0 and anti_rate == 0.0:
        label = 'robust_favorable'
    elif appearances >= 2 and anti_rate == 1.0 and fav_rate == 0.0:
        label = 'robust_anti'
    elif appearances >= 2:
        label = 'mixed_recurrent'
    else:
        label = 'single_repeat'
    return repeat_rate, fav_rate, anti_rate, stability, label


def region_key(pos, width):
    return (pos[0] // width, pos[1] // width)


def build_rows(parsed, width, exact=False):
    total_repeats = len(parsed)
    grouped = defaultdict(list)
    for rep_idx, rep in enumerate(parsed, start=1):
        for h in rep['hotspots']:
            item = dict(h)
            item['repeat'] = rep_idx
            key = h['pos'] if exact else region_key(h['pos'], width)
            grouped[key].append(item)
    rows = []
    for key, items in grouped.items():
        repeats = sorted({x['repeat'] for x in items})
        appearances = len(repeats)
        fav_count = sum(1 for x in items if x['favorable'])
        anti_count = sum(1 for x in items if not x['favorable'])
        repeat_rate, fav_rate, anti_rate, stability, label = classify(appearances, total_repeats, fav_count, anti_count, len(items))
        dyn_med = median([x['dynamic_gain'] for x in items])
        lp_med = median([x.get('logical_purity', 0.0) for x in items])
        sd_med = median([x.get('structure_drift', 0.0) for x in items])
        contrast_med = median([x.get('contrast', 0.0) for x in items])
        hotspot_score_med = median([x.get('hotspot_score', 0.0) for x in items])
        # priority score: reward recurrence + favorable sign + dynamic magnitude + purity, penalize drift less negative? Actually smaller drift is better, so use -structure_drift when negative.
        drift_bonus = max(0.0, -sd_med)
        priority = (1.8 * repeat_rate) + (1.2 * max(0.0, fav_rate - anti_rate)) + (18.0 * max(0.0, dyn_med)) + (250.0 * max(0.0, lp_med)) + (120.0 * drift_bonus) + (0.15 * max(0.0, contrast_med - 1.0))
        rows.append({
            'key': key,
            'repeats': repeats,
            'repeat_count': appearances,
            'repeat_rate': repeat_rate,
            'count': len(items),
            'fav_count': fav_count,
            'anti_count': anti_count,
            'fav_rate': fav_rate,
            'anti_rate': anti_rate,
            'stability': stability,
            'label': label,
            'dyn_gain_med': dyn_med,
            'logical_purity_med': lp_med,
            'structure_drift_med': sd_med,
            'contrast_med': contrast_med,
            'hotspot_score_med': hotspot_score_med,
            'priority_score': priority,
        })
    rows.sort(key=lambda r: (r['label'] != 'robust_favorable', -r['priority_score'], -r['repeat_count'], -r['stability']))
    return rows


def top_exact_per_region(exact_rows, width):
    best = {}
    for row in exact_rows:
        rk = region_key(row['key'], width)
        if rk not in best or row['priority_score'] > best[rk]['priority_score']:
            best[rk] = row
    return best


def main():
    ap = argparse.ArgumentParser(description='v24.9.4 prioritize repeat-stable hotspot regions and exact seeds from separate v24.9.2 outputs')
    ap.add_argument('--inputs', nargs='+', required=True)
    ap.add_argument('--region_width', type=int, default=8)
    ap.add_argument('--top_n', type=int, default=12)
    ap.add_argument('--output', default='v24_9_4_priority_report.txt')
    args = ap.parse_args()

    parsed = [parse_file(p) for p in args.inputs]
    exact_rows = build_rows(parsed, args.region_width, exact=True)
    region_rows = build_rows(parsed, args.region_width, exact=False)
    exact_best_by_region = top_exact_per_region(exact_rows, args.region_width)

    all_lp = [x[0] for rep in parsed for x in rep['global_deltas']]
    all_sd = [x[1] for rep in parsed for x in rep['global_deltas']]

    out = []
    w = out.append
    w('=== v24.9.4 repeat-stable hotspot prioritizer ===')
    w(f'inputs={len(parsed)} | region_width={args.region_width}')
    for idx, rep in enumerate(parsed, start=1):
        glp = [x[0] for x in rep['global_deltas']]
        gsd = [x[1] for x in rep['global_deltas']]
        w(f"repeat {idx}: file={os.path.basename(rep['path'])} | base_seed={rep['base_seed']} | hotspots={len(rep['hotspots'])} | logical_purity_med={median(glp):+.4f} | structure_drift_med={median(gsd):+.4f}")
    w('')
    w('=== Global dynamic summary across inputs ===')
    w(f'logical_purity_med={median(all_lp):+.4f} | structure_drift_med={median(all_sd):+.4f}')
    w('')
    w('=== Priority regions for next dynamic targeting (v24.9.4) ===')
    for row in region_rows[:args.top_n]:
        w(f"region={row['key']} | priority={row['priority_score']:.3f} | repeats={row['repeat_count']}/{len(parsed)} | repeat_rate={row['repeat_rate']:.2f} | favored={row['fav_count']}/{row['count']} | anti={row['anti_count']}/{row['count']} | fav_rate={row['fav_rate']:.2f} | anti_rate={row['anti_rate']:.2f} | stab={row['stability']:+.3f} | label={row['label']} | dyn_gain_med={row['dyn_gain_med']:+.4f} | logical_purity_med={row['logical_purity_med']:+.4f} | structure_drift_med={row['structure_drift_med']:+.4f} | contrast_med={row['contrast_med']:+.4f} | repeat_ids={row['repeats']}")
    w('')
    w('=== Best exact anchor per priority region (v24.9.4) ===')
    shown = 0
    for reg in region_rows:
        rk = reg['key']
        if rk in exact_best_by_region:
            ex = exact_best_by_region[rk]
            w(f"region={rk} -> anchor_pos={ex['key']} | anchor_priority={ex['priority_score']:.3f} | repeats={ex['repeat_count']}/{len(parsed)} | label={ex['label']} | dyn_gain_med={ex['dyn_gain_med']:+.4f} | logical_purity_med={ex['logical_purity_med']:+.4f} | structure_drift_med={ex['structure_drift_med']:+.4f} | repeat_ids={ex['repeats']}")
            shown += 1
            if shown >= args.top_n:
                break
    w('')
    w('=== Suggested targeting panel (v24.9.4) ===')
    for reg in region_rows[:min(6, len(region_rows))]:
        rk = reg['key']
        a0, b0 = rk[0]*args.region_width, rk[1]*args.region_width
        a1, b1 = a0 + args.region_width - 1, b0 + args.region_width - 1
        w(f"target_region={rk} | pos_window=[{a0}-{a1}]x[{b0}-{b1}] | priority={reg['priority_score']:.3f} | label={reg['label']} | rationale=repeat-stable favorable family")
    w('')
    w('=== v24.9.4 reading ===')
    w('1) v24.9.4 treats repeat-stable regions as the primary dynamic target, not single exact hotspots alone.')
    w('2) priority combines repeat recurrence, favorable sign consistency, dynamic gain, logical purity, and low structure drift.')
    w('3) the best exact anchor inside each strong region can be used as an operational seed for the next simulator-side targeting step.')

    txt = '\n'.join(out) + '\n'
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(txt)
    print(txt, end='')

if __name__ == '__main__':
    main()
