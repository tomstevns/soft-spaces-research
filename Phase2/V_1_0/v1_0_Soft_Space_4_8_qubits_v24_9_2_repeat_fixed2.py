#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, importlib.util, sys
from pathlib import Path

BASE_NAME = 'v1_0_Soft_Space_4_8_qubits_v24_9_2.py'

def load_base_module():
    here = Path(__file__).resolve().parent
    base_path = here / BASE_NAME
    if not base_path.exists():
        raise FileNotFoundError(f'Base file not found next to repeat wrapper: {base_path}')
    spec = importlib.util.spec_from_file_location('softspace_v24_9_2_base', str(base_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

def main() -> None:
    base = load_base_module()
    ap = argparse.ArgumentParser(description='v24.9.2 repeat wrapper (aggregates hotspot recurrence across repeats).')
    ap.add_argument('--n_qubits', type=int, default=4)
    ap.add_argument('--n_terms', type=int, default=5)
    ap.add_argument('--seeds_per_batch', type=int, default=5000)
    ap.add_argument('--batches', type=int, default=3)
    ap.add_argument('--repeat_runs', type=int, default=3)
    ap.add_argument('--repeat_seed_stride', type=int, default=10000000)
    ap.add_argument('--base_seed', type=int, default=0)
    ap.add_argument('--batch_stride', type=int, default=1000000)
    ap.add_argument('--eps_neighbor', type=float, default=0.05)
    ap.add_argument('--keep_mass', type=float, default=0.90)
    ap.add_argument('--ent_step', type=float, default=0.1)
    ap.add_argument('--leak_step', type=float, default=0.05)
    ap.add_argument('--times', type=float, nargs='+', default=[0.5, 1.0, 1.5])
    ap.add_argument('--stable_frac', type=float, default=0.01)
    ap.add_argument('--stable_leak_max', type=float, default=None)
    ap.add_argument('--stable_leak_quantile', type=float, default=None)
    ap.add_argument('--perturb_eta', type=float, nargs='+', default=[])
    ap.add_argument('--perturb_reps', type=int, default=3)
    ap.add_argument('--perturb_seeds', type=int, default=0)
    ap.add_argument('--perturb_terms', type=int, default=None)
    ap.add_argument('--perturb_pairs_per_seed', type=int, default=5)
    ap.add_argument('--iso_eps', type=float, default=1e-8)
    ap.add_argument('--no_iso_quartiles', action='store_true')
    ap.add_argument('--iso_quartiles_all_eta', action='store_true')
    ap.add_argument('--open_system', action='store_true')
    ap.add_argument('--os_include_baselines', action='store_true')
    ap.add_argument('--os_pairs_per_batch', type=int, default=400)
    ap.add_argument('--os_noise_model', type=str, default='dephasing', choices=['dephasing', 'amp_damp', 'both', 'depolar'])
    ap.add_argument('--os_gamma_phi', type=float, default=0.01)
    ap.add_argument('--os_gamma_1', type=float, default=0.01)
    ap.add_argument('--os_t_max', type=float, default=5.0)
    ap.add_argument('--os_t_steps', type=int, default=25)
    ap.add_argument('--os_dt_internal', type=float, default=None)
    ap.add_argument('--os_states_mode', type=str, default='zx', choices=['z', 'zx'])
    ap.add_argument('--logical_basis', type=str, default='eigen', choices=['eigen', 'noise_diag'])
    ap.add_argument('--os_noise_qubits', type=str, default='all', choices=['all', 'subset'])
    ap.add_argument('--os_noise_subset', type=int, nargs='+', default=None)
    ap.add_argument('--os_use_stable_pool', action='store_true')
    ap.add_argument('--os_report_quantiles', type=float, nargs=3, default=[0.1, 0.5, 0.9])
    ap.add_argument('--os_time_mode', type=str, default='mean', choices=['snapshot', 'mean', 'auc', 'late', 'peak', 'peak_post_init'])
    ap.add_argument('--os_time_late_power', type=float, default=2.0)
    ap.add_argument('--contrast_focus', action='store_true')
    ap.add_argument('--kernel_zoom', action='store_true')
    ap.add_argument('--kernel_top_centers', type=int, default=8)
    ap.add_argument('--kernel_radius', type=int, default=2)
    ap.add_argument('--kernel_min_neighbors', type=int, default=3)
    ap.add_argument('--interference_test', action='store_true')
    ap.add_argument('--if_pairs_per_batch', type=int, default=200)
    ap.add_argument('--if_use_stable_pool', action='store_true')
    ap.add_argument('--topK', type=int, default=25)
    ap.add_argument('--min_overall', type=int, default=3)
    ap.add_argument('--min_stable', type=int, default=3)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--q_bins', type=int, default=10)
    ap.add_argument('--q_bins_coarse', type=int, default=6)
    ap.add_argument('--p_tail_max', type=float, default=None)
    ap.add_argument('--bootstrap', type=int, default=200)
    ap.add_argument('--output', type=str, default='v24_9_2_repeat_output.txt')
    args = ap.parse_args()

    cache = base.PauliCache.build(args.n_qubits)
    d = cache.d
    global_open_system = []

    with open(args.output, 'w', encoding='utf-8') as f:
        def out(s: str = '') -> None:
            print(s)
            f.write(s + '\n')

        out('=== v24.9.2 repeat wrapper ===')
        out(f'Qubits: {args.n_qubits} (d={d}) | repeats={args.repeat_runs} | batches/repeat={args.batches}')
        out(f'repeat_seed_stride={args.repeat_seed_stride} | batch_stride={args.batch_stride}')
        out('')

        for r in range(int(args.repeat_runs)):
            rep_base_seed = int(args.base_seed + r * args.repeat_seed_stride)
            out(f'--- Repeat {r+1}/{args.repeat_runs} | base_seed={rep_base_seed} ---')
            real, null, scoreboards, perturb_summaries, open_system_summaries, interference_summaries = base.run_paired_batches(
                cache=cache,
                n_terms=args.n_terms,
                seeds_per_batch=args.seeds_per_batch,
                n_batches=args.batches,
                base_seed=rep_base_seed,
                batch_stride=args.batch_stride,
                eps_neighbor=args.eps_neighbor,
                ent_step=args.ent_step,
                leak_step=args.leak_step,
                times=list(args.times),
                keep_mass=args.keep_mass,
                iso_eps=args.iso_eps,
                stable_frac=args.stable_frac,
                stable_leak_max=args.stable_leak_max,
                stable_leak_quantile=args.stable_leak_quantile,
                topK=args.topK,
                min_overall=args.min_overall,
                min_stable=args.min_stable,
                alpha=args.alpha,
                q_bins=args.q_bins,
                q_bins_coarse=args.q_bins_coarse,
                p_tail_max=args.p_tail_max,
                bootstrap=args.bootstrap,
                perturb_eta=list(args.perturb_eta),
                perturb_reps=args.perturb_reps,
                perturb_seeds=args.perturb_seeds,
                perturb_terms=args.perturb_terms,
                perturb_pairs_per_seed=args.perturb_pairs_per_seed,
                open_system=bool(args.open_system),
                os_include_baselines=bool(args.os_include_baselines),
                os_pairs_per_batch=int(args.os_pairs_per_batch),
                os_noise_model=str(args.os_noise_model),
                os_gamma_phi=float(args.os_gamma_phi),
                os_gamma_1=float(args.os_gamma_1),
                os_t_max=float(args.os_t_max),
                os_t_steps=int(args.os_t_steps),
                os_dt_internal=args.os_dt_internal,
                os_states_mode=str(args.os_states_mode),
                os_logical_basis=str(args.logical_basis),
                os_noise_qubits=str(args.os_noise_qubits),
                os_noise_subset=args.os_noise_subset,
                os_use_stable_pool=bool(args.os_use_stable_pool),
                os_report_quantiles=tuple(float(x) for x in args.os_report_quantiles),
                os_time_mode=str(args.os_time_mode),
                os_time_late_power=float(args.os_time_late_power),
                contrast_focus=bool(args.contrast_focus),
                kernel_zoom=bool(args.kernel_zoom),
                kernel_top_centers=int(args.kernel_top_centers),
                kernel_radius=int(args.kernel_radius),
                kernel_min_neighbors=int(args.kernel_min_neighbors),
                interference_test=bool(args.interference_test),
                if_pairs_per_batch=int(args.if_pairs_per_batch),
                if_use_stable_pool=bool(args.if_use_stable_pool),
            )
            global_open_system.extend(open_system_summaries)
            for b, sb in enumerate(scoreboards, start=1):
                out(f"Repeat {r+1} Batch {b}: delta_entropy={sb['delta_median_entropy_bits']:+.3f} | delta_leak={sb['delta_median_leak']:+.3f} | delta_dom={sb['delta_median_dom']:+.3f} | d={sb['entropy_cohens_d']:+.3f}")
            gh = base.aggregate_hotspots_across_batches(open_system_summaries) if bool(args.open_system) else {'by_pos': [], 'by_pos_realcoarse': []}
            panels = base.hotspot_stability_panels(gh, top_n=5) if bool(args.open_system) else {}
            rf = panels.get('robust_favorable', []) if isinstance(panels, dict) else []
            mr = panels.get('mixed_recurrent', []) if isinstance(panels, dict) else []
            if rf:
                out('  Repeat robust_favorable:')
                for row in rf[:5]:
                    out(f"    {row.get('class_key','?')} | batch_count={int(row.get('batch_count',0))} | fav_rate={float(row.get('favorable_rate', float('nan'))):.2f} | stab={float(row.get('stability_score', float('nan'))):+.3f} | dyn_gain_med={float(row.get('dynamic_gain_med', float('nan'))):+.4f}")
            elif mr:
                out('  Repeat mixed_recurrent:')
                for row in mr[:5]:
                    out(f"    {row.get('class_key','?')} | batch_count={int(row.get('batch_count',0))} | fav_rate={float(row.get('favorable_rate', float('nan'))):.2f} | anti_rate={float(row.get('anti_rate', float('nan'))):.2f} | stab={float(row.get('stability_score', float('nan'))):+.3f}")
            else:
                out('  Repeat hotspot recurrence: none beyond single-batch')
            out('')

        out('=== Hotspot recurrence across all repeats (v24.9.2 repeat) ===')
        global_hot = base.aggregate_hotspots_across_batches(global_open_system) if bool(args.open_system) else {'by_pos': [], 'by_pos_realcoarse': []}
        for label, rows in [('by_pos', global_hot.get('by_pos', [])), ('by_pos_realcoarse', global_hot.get('by_pos_realcoarse', []))]:
            if not rows:
                continue
            out(f'{label}:')
            for row in rows[:10]:
                out(f"  {row.get('class_key','?')} | count={int(row.get('count',0))} | batch_count={int(row.get('batch_count',0))} | batches={row.get('batches',[])} | favorable={int(row.get('favorable_count',0))}/{int(row.get('count',0))} | anti={int(row.get('anti_count',0))}/{int(row.get('count',0))} | fav_rate={float(row.get('favorable_rate', float('nan'))):.2f} | anti_rate={float(row.get('anti_rate', float('nan'))):.2f} | stab={float(row.get('stability_score', float('nan'))):+.3f} | label={row.get('label','?')} | dyn_gain_med={float(row.get('dynamic_gain_med', float('nan'))):+.4f} | logical_purity_med={float(row.get('logical_purity_med', float('nan'))):+.4f} | structure_drift_med={float(row.get('structure_drift_med', float('nan'))):+.4f}")
        panels = base.hotspot_stability_panels(global_hot, top_n=8) if bool(args.open_system) else {}
        if isinstance(panels, dict):
            out('=== Hotspot stability panels across all repeats ===')
            for panel_name in ['robust_favorable', 'robust_anti', 'mixed_recurrent']:
                rows = panels.get(panel_name, [])
                if not rows:
                    continue
                out(f'{panel_name}:')
                for row in rows:
                    out(f"  {row.get('class_key','?')} | batch_count={int(row.get('batch_count',0))} | fav_rate={float(row.get('favorable_rate', float('nan'))):.2f} | anti_rate={float(row.get('anti_rate', float('nan'))):.2f} | stab={float(row.get('stability_score', float('nan'))):+.3f} | dyn_gain_med={float(row.get('dynamic_gain_med', float('nan'))):+.4f} | logical_purity_med={float(row.get('logical_purity_med', float('nan'))):+.4f} | structure_drift_med={float(row.get('structure_drift_med', float('nan'))):+.4f}")
        out('')
        out('=== End of v24.9.2 repeat ===')

if __name__ == '__main__':
    main()
