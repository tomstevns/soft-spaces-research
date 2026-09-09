#!/usr/bin/env python3
import argparse, os, re, statistics
from collections import defaultdict

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
PRIORITY_REGION_RE = re.compile(r"region=\((\d+),\s*(\d+)\) \| priority=([\d.]+) .*?label=([A-Za-z_]+)")


def med(vals):
    return statistics.median(vals) if vals else float('nan')


def parse_v2492(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()
    base_seed = None
    pending = None
    hotspots = []
    global_deltas = []
    for line in lines:
        if base_seed is None:
            m = HEADER_BASE_RE.search(line)
            if m:
                base_seed = int(m.group(1))
        m = GLOBAL_DELTA_RE.search(line)
        if m:
            global_deltas.append((float(m.group(1)), float(m.group(2))))
        m = HOTSPOT_RE.search(line.strip())
        if m:
            pending = {
                'seed': int(m.group(1)), 'i': int(m.group(2)), 'j': int(m.group(3)),
                'pos': (int(m.group(4)), int(m.group(5))),
                'contrast': float(m.group(6)), 'hotspot_score': float(m.group(7)),
                'dynamic_gain': float(m.group(8)), 'favorable': m.group(9) == 'True',
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


def parse_priority_regions(path, top_k, allow_single=False):
    rows = []
    in_block = False
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for raw in f:
            line = raw.strip()
            if line == '=== Priority regions for next dynamic targeting (v24.9.4) ===':
                in_block = True
                continue
            if in_block and line.startswith('==='):
                break
            if in_block:
                m = PRIORITY_REGION_RE.search(line)
                if m:
                    region = (int(m.group(1)), int(m.group(2)))
                    priority = float(m.group(3))
                    label = m.group(4)
                    if allow_single or label != 'single_repeat':
                        rows.append((region, priority, label))
    rows.sort(key=lambda x: (-x[1], x[0]))
    return rows[:top_k]


def region_key(pos, width):
    return (pos[0] // width, pos[1] // width)


def summarize(items):
    if not items:
        return None
    fav = sum(1 for x in items if x['favorable'])
    anti = len(items) - fav
    return {
        'count': len(items),
        'fav_count': fav,
        'anti_count': anti,
        'fav_rate': fav / len(items),
        'anti_rate': anti / len(items),
        'dyn_gain_med': med([x['dynamic_gain'] for x in items]),
        'logical_purity_med': med([x.get('logical_purity', 0.0) for x in items]),
        'structure_drift_med': med([x.get('structure_drift', 0.0) for x in items]),
        'contrast_med': med([x.get('contrast', 0.0) for x in items]),
    }


def main():
    ap = argparse.ArgumentParser(description='v24.10 targeted region evaluator from v24.9.4 priority report + separate v24.9.2 repeats')
    ap.add_argument('--inputs', nargs='+', required=True, help='v24.9.2 repeat txt files')
    ap.add_argument('--priority_report', required=True, help='v24.9.4 priority report txt')
    ap.add_argument('--region_width', type=int, default=8)
    ap.add_argument('--top_k_regions', type=int, default=6)
    ap.add_argument('--include_single_repeat_regions', action='store_true')
    ap.add_argument('--output', default='v24_10_targeted_region_evaluator.txt')
    args = ap.parse_args()

    repeats = [parse_v2492(p) for p in args.inputs]
    target_rows = parse_priority_regions(args.priority_report, args.top_k_regions, args.include_single_repeat_regions)
    target_regions = {r for r, _, _ in target_rows}

    all_hotspots = []
    targeted = []
    untargeted = []
    region_buckets = defaultdict(list)
    exact_buckets = defaultdict(list)
    for ridx, rep in enumerate(repeats, start=1):
        for h in rep['hotspots']:
            item = dict(h)
            item['repeat'] = ridx
            item['region'] = region_key(item['pos'], args.region_width)
            all_hotspots.append(item)
            exact_buckets[item['pos']].append(item)
            region_buckets[item['region']].append(item)
            if item['region'] in target_regions:
                targeted.append(item)
            else:
                untargeted.append(item)

    out = []
    w = out.append
    w('=== v24.10 targeted region evaluator ===')
    w(f'inputs={len(repeats)} | priority_report={os.path.basename(args.priority_report)} | region_width={args.region_width} | top_k_regions={args.top_k_regions}')
    for idx, rep in enumerate(repeats, start=1):
        glp = [x[0] for x in rep['global_deltas']]
        gsd = [x[1] for x in rep['global_deltas']]
        w(f"repeat {idx}: file={os.path.basename(rep['path'])} | base_seed={rep['base_seed']} | hotspots={len(rep['hotspots'])} | logical_purity_med={med(glp):+.4f} | structure_drift_med={med(gsd):+.4f}")
    w('')
    w('=== Target regions imported from v24.9.4 ===')
    for region, priority, label in target_rows:
        a0, b0 = region[0]*args.region_width, region[1]*args.region_width
        a1, b1 = a0 + args.region_width - 1, b0 + args.region_width - 1
        w(f'target_region={region} | pos_window=[{a0}-{a1}]x[{b0}-{b1}] | priority={priority:.3f} | source_label={label}')
    w('')

    all_s = summarize(all_hotspots)
    tar_s = summarize(targeted)
    unt_s = summarize(untargeted)
    w('=== Targeted vs untargeted hotspot quality ===')
    w(f"all_hotspots: count={all_s['count']} | fav_rate={all_s['fav_rate']:.2f} | dyn_gain_med={all_s['dyn_gain_med']:+.4f} | logical_purity_med={all_s['logical_purity_med']:+.4f} | structure_drift_med={all_s['structure_drift_med']:+.4f} | contrast_med={all_s['contrast_med']:+.4f}")
    if tar_s:
        w(f"targeted_hotspots: count={tar_s['count']} | fav_rate={tar_s['fav_rate']:.2f} | dyn_gain_med={tar_s['dyn_gain_med']:+.4f} | logical_purity_med={tar_s['logical_purity_med']:+.4f} | structure_drift_med={tar_s['structure_drift_med']:+.4f} | contrast_med={tar_s['contrast_med']:+.4f}")
    if unt_s:
        w(f"untargeted_hotspots: count={unt_s['count']} | fav_rate={unt_s['fav_rate']:.2f} | dyn_gain_med={unt_s['dyn_gain_med']:+.4f} | logical_purity_med={unt_s['logical_purity_med']:+.4f} | structure_drift_med={unt_s['structure_drift_med']:+.4f} | contrast_med={unt_s['contrast_med']:+.4f}")
    if tar_s and unt_s:
        w(f"targeted_minus_untargeted: dyn_gain={tar_s['dyn_gain_med']-unt_s['dyn_gain_med']:+.4f} | logical_purity={tar_s['logical_purity_med']-unt_s['logical_purity_med']:+.4f} | structure_drift={tar_s['structure_drift_med']-unt_s['structure_drift_med']:+.4f} | fav_rate={tar_s['fav_rate']-unt_s['fav_rate']:+.2f}")
    w('')

    w('=== Region-level targeted panel diagnostics (v24.10) ===')
    for region, priority, label in target_rows:
        items = region_buckets.get(region, [])
        s = summarize(items)
        reps = sorted({x['repeat'] for x in items})
        if not s:
            w(f'region={region} | repeats=0/{len(repeats)} | count=0 | note=no hotspots landed in this region across supplied repeats')
            continue
        w(f"region={region} | repeats={len(reps)}/{len(repeats)} | count={s['count']} | favored={s['fav_count']}/{s['count']} | anti={s['anti_count']}/{s['count']} | dyn_gain_med={s['dyn_gain_med']:+.4f} | logical_purity_med={s['logical_purity_med']:+.4f} | structure_drift_med={s['structure_drift_med']:+.4f} | repeat_ids={reps}")
    w('')

    anchor_rows = []
    for pos, items in exact_buckets.items():
        reg = region_key(pos, args.region_width)
        if reg not in target_regions:
            continue
        reps = sorted({x['repeat'] for x in items})
        s = summarize(items)
        priority = 1.5*(len(reps)/len(repeats)) + 15*max(0.0,s['dyn_gain_med']) + 250*max(0.0,s['logical_purity_med']) + 120*max(0.0,-s['structure_drift_med'])
        anchor_rows.append((priority, pos, reg, reps, s))
    anchor_rows.sort(reverse=True)
    w('=== Exact anchors inside targeted regions (v24.10) ===')
    for priority, pos, reg, reps, s in anchor_rows[:max(args.top_k_regions, 8)]:
        w(f"anchor_pos={pos} | region={reg} | anchor_priority={priority:.3f} | repeats={len(reps)}/{len(repeats)} | favored={s['fav_count']}/{s['count']} | dyn_gain_med={s['dyn_gain_med']:+.4f} | logical_purity_med={s['logical_purity_med']:+.4f} | structure_drift_med={s['structure_drift_med']:+.4f} | repeat_ids={reps}")
    w('')
    w('=== Operational reading for next simulator-side step (v24.10) ===')
    w('1) Use the imported target regions as priors for candidate preference rather than hard constraints at first.')
    w('2) Measure whether targeted-region selection increases logical_purity and makes structure_drift more negative relative to untargeted hotspots.')
    w('3) Use the strongest exact anchors inside the best regions as seed positions for the first simulator-side targeting implementation.')

    txt = '\n'.join(out) + '\n'
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(txt)
    print(txt, end='')

if __name__ == '__main__':
    main()
