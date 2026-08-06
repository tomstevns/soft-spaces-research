#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from statistics import median

HEADER_RE = re.compile(r'base_seed=(\d+)')
DELTA_RE = re.compile(
    r'REAL_Q4 - NULL_Q4 deltas \(reduced q50\): '
    r'fid_uncond=([+\-]?\d*\.?\d+) \| '
    r'fid_cond=([+\-]?\d*\.?\d+) \| '
    r'leak=([+\-]?\d*\.?\d+) \| '
    r'subspace_ret=([+\-]?\d*\.?\d+) \| '
    r'logical_coh=([+\-]?\d*\.?\d+) \| '
    r'logical_purity=([+\-]?\d*\.?\d+) \| '
    r'structure_drift=([+\-]?\d*\.?\d+)'
)
HOTSPOT_RE = re.compile(
    r'hotspot\(seed,i,j\)=\((\d+),(\d+),(\d+)\) \| pos=\((\d+), (\d+)\) \| '
    r'contrast=([+\-]?\d*\.?\d+) \| hotspot_score=([+\-]?\d*\.?\d+) \| '
    r'dynamic_gain=([+\-]?\d*\.?\d+) \| favorable=(True|False)'
)
HOTSPOT_DELTA_RE = re.compile(
    r'deltas q50: fid_uncond=([+\-]?\d*\.?\d+) \| logical_purity=([+\-]?\d*\.?\d+) '
    r'\| structure_drift=([+\-]?\d*\.?\d+) \| subspace_ret=([+\-]?\d*\.?\d+)'
)
BYPOS_RE = re.compile(
    r'pos=\((\d+), (\d+)\) \| count=(\d+) \| batch_count=(\d+) \| batches=\[[^\]]*\] \| '
    r'favorable=(\d+)/(\d+) \| anti=(\d+)/(\d+) \| fav_rate=([+\-]?\d*\.?\d+) \| '
    r'anti_rate=([+\-]?\d*\.?\d+) \| stab=([+\-]?\d*\.?\d+) \| label=([A-Za-z_]+) \| '
    r'dyn_gain_med=([+\-]?\d*\.?\d+) \| logical_purity_med=([+\-]?\d*\.?\d+) \| '
    r'structure_drift_med=([+\-]?\d*\.?\d+)'
)

def parse_args():
    ap = argparse.ArgumentParser(description='v24.17 diagnostic ridge report from existing v24.14 repeats')
    ap.add_argument('--inputs', nargs='+', required=True)
    ap.add_argument('--anchor_i', type=int, default=157)
    ap.add_argument('--anchor_j', type=int, default=158)
    ap.add_argument('--radius', type=int, default=8)
    ap.add_argument('--top_k', type=int, default=20)
    ap.add_argument('--output', required=True)
    return ap.parse_args()

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def parse_file(path):
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    lines = text.splitlines()
    base_seed = None
    m = HEADER_RE.search(text)
    if m:
        base_seed = int(m.group(1))
    deltas = []
    hotspots = []
    recurrence = []
    pending = None
    in_bypostable = False

    for line in lines:
        dm = DELTA_RE.search(line)
        if dm:
            deltas.append({
                'fid_uncond': float(dm.group(1)),
                'fid_cond': float(dm.group(2)),
                'leak': float(dm.group(3)),
                'subspace_ret': float(dm.group(4)),
                'logical_coh': float(dm.group(5)),
                'logical_purity': float(dm.group(6)),
                'structure_drift': float(dm.group(7)),
            })

        hm = HOTSPOT_RE.search(line)
        if hm:
            pending = {
                'seed': int(hm.group(1)),
                'i': int(hm.group(4)),
                'j': int(hm.group(5)),
                'contrast': float(hm.group(6)),
                'hotspot_score': float(hm.group(7)),
                'dynamic_gain': float(hm.group(8)),
                'favorable': hm.group(9) == 'True',
            }
            continue

        if pending is not None:
            hdm = HOTSPOT_DELTA_RE.search(line)
            if hdm:
                pending.update({
                    'fid_uncond': float(hdm.group(1)),
                    'logical_purity': float(hdm.group(2)),
                    'structure_drift': float(hdm.group(3)),
                    'subspace_ret': float(hdm.group(4)),
                })
                hotspots.append(pending)
                pending = None
                continue

        if line.strip() == 'by_pos:':
            in_bypostable = True
            continue
        if in_bypostable and line.startswith('by_pos_realcoarse:'):
            in_bypostable = False
            continue
        if in_bypostable:
            bm = BYPOS_RE.search(line.strip())
            if bm:
                recurrence.append({
                    'pos': (int(bm.group(1)), int(bm.group(2))),
                    'count': int(bm.group(3)),
                    'batch_count': int(bm.group(4)),
                    'fav_rate': float(bm.group(9)),
                    'anti_rate': float(bm.group(10)),
                    'stab': float(bm.group(11)),
                    'label': bm.group(12),
                    'dyn_gain_med': float(bm.group(13)),
                    'logical_purity_med': float(bm.group(14)),
                    'structure_drift_med': float(bm.group(15)),
                })
    return {'path': str(path), 'base_seed': base_seed, 'deltas': deltas, 'hotspots': hotspots, 'recurrence': recurrence}

def fmt(x, digits=4):
    return 'NA' if x is None else f'{x:+.{digits}f}'

def summarize(parsed, anchor, radius, top_k):
    local_hotspots = []
    local_recur = []
    for ridx, item in enumerate(parsed, start=1):
        for h in item['hotspots']:
            pos = (h['i'], h['j'])
            dist = manhattan(pos, anchor)
            if dist <= radius:
                hh = dict(h)
                hh['repeat_id'] = ridx
                hh['base_seed'] = item['base_seed']
                hh['dist'] = dist
                local_hotspots.append(hh)
        for r in item['recurrence']:
            dist = manhattan(r['pos'], anchor)
            if dist <= radius:
                rr = dict(r)
                rr['repeat_id'] = ridx
                rr['base_seed'] = item['base_seed']
                rr['dist'] = dist
                local_recur.append(rr)

    by_pos = {}
    for h in local_hotspots:
        pos = (h['i'], h['j'])
        by_pos.setdefault(pos, []).append(h)

    ranked = []
    for pos, vals in by_pos.items():
        ranked.append({
            'pos': pos,
            'n': len(vals),
            'repeat_count': len({v['repeat_id'] for v in vals}),
            'fav_rate': sum(1 for v in vals if v['favorable']) / len(vals),
            'dyn_gain_med': median(v['dynamic_gain'] for v in vals),
            'logical_purity_med': median(v['logical_purity'] for v in vals),
            'structure_drift_med': median(v['structure_drift'] for v in vals),
            'contrast_med': median(v['contrast'] for v in vals),
            'score_med': median(v['hotspot_score'] for v in vals),
            'dist': manhattan(pos, anchor),
        })
    ranked.sort(key=lambda x: (-x['repeat_count'], -x['fav_rate'], -x['dyn_gain_med'], x['dist'], x['pos']))

    all_deltas = [d for item in parsed for d in item['deltas']]
    global_summary = None
    if all_deltas:
        global_summary = {k: median(d[k] for d in all_deltas) for k in ['fid_uncond','fid_cond','leak','subspace_ret','logical_coh','logical_purity','structure_drift']}
    return local_hotspots, local_recur, ranked[:top_k], global_summary

def build_report(parsed, anchor, radius, top_k):
    local_hotspots, local_recur, ranked, global_summary = summarize(parsed, anchor, radius, top_k)
    lines = []
    lines.append('=== v24.17 diagnostic ridge report ===')
    lines.append(f'inputs={len(parsed)} | anchor={anchor} | radius={radius} | top_k={top_k}')
    lines.append('')
    lines.append('=== Input files ===')
    for idx, item in enumerate(parsed, start=1):
        lines.append(f"repeat {idx}: file={Path(item['path']).name} | base_seed={item['base_seed']} | hotspots={len(item['hotspots'])} | recurrence_rows={len(item['recurrence'])}")
    lines.append('')
    if global_summary:
        lines.append('=== Global reduced-q50 medians across inputs ===')
        lines.append(
            ' | '.join([
                f"fid_uncond={fmt(global_summary['fid_uncond'])}",
                f"fid_cond={fmt(global_summary['fid_cond'])}",
                f"leak={fmt(global_summary['leak'])}",
                f"subspace_ret={fmt(global_summary['subspace_ret'])}",
                f"logical_coh={fmt(global_summary['logical_coh'])}",
                f"logical_purity={fmt(global_summary['logical_purity'])}",
                f"structure_drift={fmt(global_summary['structure_drift'])}",
            ])
        )
        lines.append('')
    lines.append('=== Local hotspot evidence within radius ===')
    if not local_hotspots:
        lines.append('No hotspot evidence found within requested radius.')
    else:
        for h in sorted(local_hotspots, key=lambda x: (x['repeat_id'], x['dist'], x['i'], x['j'], -x['dynamic_gain'])):
            lines.append(
                f"repeat={h['repeat_id']} | seed={h['seed']} | pos=({h['i']}, {h['j']}) | dist={h['dist']} | fav={h['favorable']} | "
                f"dyn_gain={fmt(h['dynamic_gain'])} | contrast={fmt(h['contrast'])} | hotspot_score={fmt(h['hotspot_score'])} | "
                f"fid_uncond={fmt(h['fid_uncond'])} | logical_purity={fmt(h['logical_purity'])} | structure_drift={fmt(h['structure_drift'])} | "
                f"subspace_ret={fmt(h['subspace_ret'])}"
            )
    lines.append('')
    lines.append('=== Local recurrence rows within radius ===')
    if not local_recur:
        lines.append('No recurrence summary rows found within requested radius.')
    else:
        for r in sorted(local_recur, key=lambda x: (x['repeat_id'], x['dist'], x['pos'])):
            lines.append(
                f"repeat={r['repeat_id']} | pos={r['pos']} | dist={r['dist']} | batch_count={r['batch_count']} | "
                f"fav_rate={fmt(r['fav_rate'])} | anti_rate={fmt(r['anti_rate'])} | stab={fmt(r['stab'])} | "
                f"dyn_gain_med={fmt(r['dyn_gain_med'])} | logical_purity_med={fmt(r['logical_purity_med'])} | "
                f"structure_drift_med={fmt(r['structure_drift_med'])} | label={r['label']}"
            )
    lines.append('')
    lines.append('=== Ranked positions near anchor (from hotspot evidence) ===')
    if not ranked:
        lines.append('No positions ranked within requested radius.')
    else:
        for row in ranked:
            lines.append(
                f"pos={row['pos']} | dist={row['dist']} | repeat_count={row['repeat_count']} | n={row['n']} | "
                f"fav_rate={fmt(row['fav_rate'])} | dyn_gain_med={fmt(row['dyn_gain_med'])} | "
                f"logical_purity_med={fmt(row['logical_purity_med'])} | structure_drift_med={fmt(row['structure_drift_med'])} | "
                f"contrast_med={fmt(row['contrast_med'])} | hotspot_score_med={fmt(row['score_med'])}"
            )
    lines.append('')
    lines.append('=== v24.17 reading ===')
    lines.append('1) Use this report to test whether the v24.14 ridge around the dominant anchor is genuinely local and coherent.')
    lines.append('2) If the neighborhood around the chosen anchor is sparse or unstable, treat that as evidence against further score-bias tuning.')
    lines.append('3) If nearby positions repeatedly show better structure_drift and logical_purity, promote them into the next explanatory analysis stage.')
    lines.append('')
    return '\n'.join(lines)

def main():
    args = parse_args()
    parsed = [parse_file(p) for p in args.inputs]
    anchor = (args.anchor_i, args.anchor_j)
    report = build_report(parsed, anchor, args.radius, args.top_k)
    Path(args.output).write_text(report, encoding='utf-8')

if __name__ == '__main__':
    main()
