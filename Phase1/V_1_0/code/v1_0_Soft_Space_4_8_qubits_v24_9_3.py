#!/usr/bin/env python3
import argparse, ast, math, os, re, statistics
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


def classify_row(fav_count, anti_count, items, appearances, total_repeats):
    fav_rate = fav_count / len(items) if len(items) else 0.0
    anti_rate = anti_count / len(items) if len(items) else 0.0
    repeat_rate = appearances / total_repeats if total_repeats else 0.0
    stability = repeat_rate * (fav_rate - anti_rate)
    if appearances >= 2 and fav_rate >= 1.0 and anti_rate == 0:
        label = 'robust_favorable'
    elif appearances >= 2 and anti_rate >= 1.0 and fav_rate == 0:
        label = 'robust_anti'
    elif appearances >= 2:
        label = 'mixed_recurrent'
    else:
        label = 'single_repeat'
    return stability, fav_rate, anti_rate, label


def median_or_nan(vals):
    return statistics.median(vals) if vals else float('nan')


def parse_file(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()
    base_seed = None
    batches = []
    global_deltas = []
    current_batch = None
    pending_hotspot = None
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
        hs = HOTSPOT_RE.search(line.strip())
        if hs and current_batch is not None:
            pending_hotspot = {
                'seed': int(hs.group(1)), 'i': int(hs.group(2)), 'j': int(hs.group(3)),
                'pos': (int(hs.group(4)), int(hs.group(5))),
                'contrast': float(hs.group(6)), 'hotspot_score': float(hs.group(7)),
                'dynamic_gain': float(hs.group(8)), 'favorable': hs.group(9) == 'True',
                'batch': current_batch,
            }
            batches.append(pending_hotspot)
            continue
        dm = DELTA_RE.search(line.strip())
        if dm and pending_hotspot is not None:
            pending_hotspot.update({
                'fid_uncond': float(dm.group(1)),
                'logical_purity': float(dm.group(2)),
                'structure_drift': float(dm.group(3)),
                'subspace_ret': float(dm.group(4)),
            })
            pending_hotspot = None
    return {
        'path': path,
        'base_seed': base_seed,
        'hotspots': batches,
        'global_deltas': global_deltas,
    }


def region_key(pos, width):
    a, b = pos
    return (a // width, b // width)


def aggregate(parsed, width):
    total_repeats = len(parsed)
    by_pos = defaultdict(list)
    by_region = defaultdict(list)
    for rep_idx, rep in enumerate(parsed, start=1):
        seen_pos = set()
        seen_region = set()
        for h in rep['hotspots']:
            record = dict(h)
            record['repeat'] = rep_idx
            record['base_seed'] = rep['base_seed']
            by_pos[h['pos']].append(record)
            by_region[region_key(h['pos'], width)].append(record)
    def build_rows(grouped, key_formatter):
        rows = []
        for key, items in grouped.items():
            repeats = sorted(set(x['repeat'] for x in items))
            appearances = len(repeats)
            fav_count = sum(1 for x in items if x.get('favorable'))
            anti_count = sum(1 for x in items if not x.get('favorable'))
            stab, fav_rate, anti_rate, label = classify_row(fav_count, anti_count, items, appearances, total_repeats)
            rows.append({
                'class_key': key_formatter(key),
                'raw_key': key,
                'count': len(items),
                'repeat_count': appearances,
                'repeats': repeats,
                'favorable_count': fav_count,
                'anti_count': anti_count,
                'favorable_rate': fav_rate,
                'anti_rate': anti_rate,
                'repeat_rate': appearances / total_repeats if total_repeats else 0.0,
                'stability_score': stab,
                'label': label,
                'dynamic_gain_med': median_or_nan([x.get('dynamic_gain', float('nan')) for x in items]),
                'logical_purity_med': median_or_nan([x.get('logical_purity', float('nan')) for x in items]),
                'structure_drift_med': median_or_nan([x.get('structure_drift', float('nan')) for x in items]),
                'contrast_med': median_or_nan([x.get('contrast', float('nan')) for x in items]),
            })
        rows.sort(key=lambda r: (r['label'] != 'robust_favorable', -r['repeat_count'], -r['stability_score'], -abs(r['dynamic_gain_med'])))
        return rows
    return {
        'by_pos': build_rows(by_pos, lambda k: f'pos={k}'),
        'by_region': build_rows(by_region, lambda k: f'region={k}'),
        'total_repeats': total_repeats,
    }


def main():
    ap = argparse.ArgumentParser(description='v24.9.3 merge/analyze hotspot recurrence across separate v24.9.2 repeat outputs')
    ap.add_argument('--inputs', nargs='+', required=True, help='One or more v24.9.2 output txt files from separate repeats')
    ap.add_argument('--region_width', type=int, default=8, help='Region bucket width for cross-repeat hotspot family aggregation')
    ap.add_argument('--top_n', type=int, default=12)
    ap.add_argument('--output', default='v24_9_3_merge_report.txt')
    args = ap.parse_args()

    parsed = [parse_file(p) for p in args.inputs]
    agg = aggregate(parsed, args.region_width)

    lines = []
    w = lines.append
    w('=== v24.9.3 hotspot family merger ===')
    w(f'inputs={len(parsed)} | region_width={args.region_width}')
    for idx, rep in enumerate(parsed, start=1):
        glp = [x[0] for x in rep['global_deltas']]
        gsd = [x[1] for x in rep['global_deltas']]
        w(f"repeat {idx}: file={os.path.basename(rep['path'])} | base_seed={rep['base_seed']} | hotspots={len(rep['hotspots'])} | logical_purity_med={median_or_nan(glp):+.4f} | structure_drift_med={median_or_nan(gsd):+.4f}")
    all_glp = [x[0] for rep in parsed for x in rep['global_deltas']]
    all_gsd = [x[1] for rep in parsed for x in rep['global_deltas']]
    w('')
    w('=== Global dynamic summary across inputs ===')
    w(f"logical_purity_med={median_or_nan(all_glp):+.4f} | structure_drift_med={median_or_nan(all_gsd):+.4f}")
    w('')
    w('=== Cross-repeat hotspot recurrence by exact position (v24.9.3) ===')
    for row in agg['by_pos'][:args.top_n]:
        w(
            f"{row['class_key']} | repeats={row['repeat_count']}/{agg['total_repeats']} | repeat_rate={row['repeat_rate']:.2f} | count={row['count']} | favored={row['favorable_count']}/{row['count']} | anti={row['anti_count']}/{row['count']} | fav_rate={row['favorable_rate']:.2f} | anti_rate={row['anti_rate']:.2f} | stab={row['stability_score']:+.3f} | label={row['label']} | dyn_gain_med={row['dynamic_gain_med']:+.4f} | logical_purity_med={row['logical_purity_med']:+.4f} | structure_drift_med={row['structure_drift_med']:+.4f} | contrast_med={row['contrast_med']:+.4f} | repeat_ids={row['repeats']}"
        )
    w('')
    w(f'=== Cross-repeat hotspot families by region width={args.region_width} (v24.9.3) ===')
    for row in agg['by_region'][:args.top_n]:
        w(
            f"{row['class_key']} | repeats={row['repeat_count']}/{agg['total_repeats']} | repeat_rate={row['repeat_rate']:.2f} | count={row['count']} | favored={row['favorable_count']}/{row['count']} | anti={row['anti_count']}/{row['count']} | fav_rate={row['favorable_rate']:.2f} | anti_rate={row['anti_rate']:.2f} | stab={row['stability_score']:+.3f} | label={row['label']} | dyn_gain_med={row['dynamic_gain_med']:+.4f} | logical_purity_med={row['logical_purity_med']:+.4f} | structure_drift_med={row['structure_drift_med']:+.4f} | contrast_med={row['contrast_med']:+.4f} | repeat_ids={row['repeats']}"
        )
    w('')
    w('=== v24.9.3 reading ===')
    w('1) exact position recurrence asks whether the same hotspot position survives across independent repeats.')
    w('2) region recurrence asks whether nearby hotspots form a repeat-stable family even when the exact position drifts.')
    w('3) robust_favorable means repeat_count>=2 and no anti-sign among those appearances.')
    txt = '\n'.join(lines) + '\n'
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(txt)
    print(txt, end='')

if __name__ == '__main__':
    main()
