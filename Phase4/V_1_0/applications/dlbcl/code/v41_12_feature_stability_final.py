
#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

import v41_7_corrected_preprocessing_audit as v7
import v41_3_softspaces_mapping_probe as v3

VERSION = "v41.12"
PANEL_SIZE = 16
DEFAULT_RESAMPLES = 500
DEFAULT_SAMPLE_FRACTION = 0.80
DEFAULT_SEED = 4112001

def project_dir_from_script():
    return Path(__file__).resolve().parent.parent

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_locked_inputs(project_dir):
    X, y, genes, folds, splits_path = v7.load_inputs(project_dir)
    if X.shape != (350, 22154):
        raise RuntimeError(f"Unexpected locked input shape: {X.shape}")
    counts = Counter(y.tolist())
    if counts != Counter({0:183, 1:167}):
        raise RuntimeError(f"Unexpected class counts: {counts}")
    return X, y, genes, folds, splits_path

def softspaces_panel_from_raw(X_raw, y, genes, panel_size):
    Xtr, _, pool_genes, chosen_global, _ = v7.preprocess_corrected(
        X_raw, X_raw, genes
    )
    H = v3.class_contrast_operator(Xtr, y)
    _, coupling, diagnostics = v3.softspaces_decompose_and_score(H)
    rank = v3.stable_rank(coupling, pool_genes)
    top = rank[:panel_size]
    return (
        [str(g) for g in pool_genes[top]],
        [float(coupling[i]) for i in top],
        np.asarray(pool_genes, dtype=object),
        np.asarray(chosen_global, dtype=int),
        diagnostics,
    )

def jaccard(a,b):
    a,b=set(a),set(b)
    return len(a&b)/len(a|b) if (a or b) else 1.0

def pairwise_jaccards(panels):
    vals=[]
    for i in range(len(panels)):
        for j in range(i+1,len(panels)):
            vals.append(jaccard(panels[i],panels[j]))
    return np.asarray(vals,float)

def bootstrap_mean_ci(values, seed, reps=20000):
    values=np.asarray(values,float)
    if len(values)==0: return np.nan,np.nan,np.nan
    if len(values)==1:
        v=float(values[0]); return v,v,v
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(values),size=(reps,len(values)))
    means=values[idx].mean(axis=1)
    return float(values.mean()), float(np.quantile(means,.025)), float(np.quantile(means,.975))

def stratified_bootstrap_indices(y, fraction, rng):
    out=[]
    for cls in sorted(np.unique(y)):
        idx=np.where(y==cls)[0]
        n=max(1,int(round(len(idx)*fraction)))
        out.extend(rng.choice(idx,size=n,replace=True).tolist())
    out=np.asarray(out,int); rng.shuffle(out); return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n-resamples",type=int,default=DEFAULT_RESAMPLES)
    ap.add_argument("--sample-fraction",type=float,default=DEFAULT_SAMPLE_FRACTION)
    ap.add_argument("--panel-size",type=int,default=PANEL_SIZE)
    ap.add_argument("--seed",type=int,default=DEFAULT_SEED)
    ap.add_argument("--output-dir",type=Path,default=None)
    args=ap.parse_args()

    if not .5 <= args.sample_fraction <= 1.0:
        raise SystemExit("--sample-fraction must be between 0.5 and 1.0")

    project_dir=project_dir_from_script()
    outdir=args.output_dir or (project_dir/"results"/"stability")
    outdir.mkdir(parents=True,exist_ok=True)

    print(f"=== {VERSION} FEATURE STABILITY ===")
    print("Project:",project_dir)
    X,y,genes,folds,splits_path=load_locked_inputs(project_dir)
    print("Locked matrix:",X.shape)
    print("Class counts: GCB=",int(np.sum(y==0)),"ABC=",int(np.sum(y==1)))

    frozen_panel,frozen_scores,frozen_pool,_,_=softspaces_panel_from_raw(
        X,y,genes,args.panel_size
    )
    print("\nFull-development 16-gene panel:")
    for i,(g,s) in enumerate(zip(frozen_panel,frozen_scores),1):
        print(f"{i:2d}. {g:<20} score={s:.8f}")

    rng=np.random.default_rng(args.seed)
    selection_counts=Counter()
    variance_pool_counts=Counter()
    real_panels=[]; null_panels=[]; overlap_frozen=[]; panel_rows=[]

    for r in range(args.n_resamples):
        idx=stratified_bootstrap_indices(y,args.sample_fraction,rng)
        Xr=X[idx]; yr=y[idx]
        panel,scores,pool_genes,_,diag=softspaces_panel_from_raw(
            Xr,yr,genes,args.panel_size
        )
        real_panels.append(panel)
        for g in panel: selection_counts[g]+=1
        for g in pool_genes: variance_pool_counts[str(g)]+=1

        null_idx=rng.choice(len(pool_genes),size=args.panel_size,replace=False)
        null_panel=[str(pool_genes[i]) for i in null_idx]
        null_panels.append(null_panel)

        ov=len(set(panel)&set(frozen_panel))/args.panel_size
        overlap_frozen.append(ov)
        panel_rows.append({
            "resample":r,
            "n_samples":len(idx),
            "n_gcb":int(np.sum(yr==0)),
            "n_abc":int(np.sum(yr==1)),
            "overlap_fraction_with_frozen16":ov,
            "panel":"|".join(panel),
            "null_panel":"|".join(null_panel),
            "coupling_mean":diag.get("coupling_mean"),
            "coupling_median":diag.get("coupling_median"),
            "coupling_max":diag.get("coupling_max"),
            "topP_abs_fraction":diag.get("topP_abs_fraction"),
        })
        if (r+1)%max(1,args.n_resamples//20)==0:
            print(f"[v41.12] {r+1}/{args.n_resamples}",flush=True)

    gene_rows=[]
    frozen_set=set(frozen_panel)
    for g in genes.astype(str):
        gene_rows.append({
            "gene":g,
            "selection_count":selection_counts[g],
            "selection_frequency":selection_counts[g]/args.n_resamples,
            "variance_pool_count":variance_pool_counts[g],
            "variance_pool_frequency":variance_pool_counts[g]/args.n_resamples,
            "in_frozen_16":g in frozen_set,
        })
    gene_df=pd.DataFrame(gene_rows).sort_values(
        ["selection_frequency","variance_pool_frequency","gene"],
        ascending=[False,False,True],kind="mergesort"
    )
    panel_df=pd.DataFrame(panel_rows)

    real_mean,real_lo,real_hi=bootstrap_mean_ci(pairwise_jaccards(real_panels),args.seed+100)
    null_mean,null_lo,null_hi=bootstrap_mean_ci(pairwise_jaccards(null_panels),args.seed+101)
    ov_mean,ov_lo,ov_hi=bootstrap_mean_ci(np.asarray(overlap_frozen),args.seed+102)

    stable80=gene_df.loc[gene_df.selection_frequency>=.80,"gene"].tolist()
    stable60=gene_df.loc[gene_df.selection_frequency>=.60,"gene"].tolist()
    stable50=gene_df.loc[gene_df.selection_frequency>=.50,"gene"].tolist()

    gene_csv=outdir/"v41_12_gene_selection_frequencies.csv"
    panel_csv=outdir/"v41_12_resample_panels.csv"
    txt=outdir/"v41_12_feature_stability_summary.txt"
    manifest=outdir/"v41_12_feature_stability_manifest.json"

    gene_df.to_csv(gene_csv,index=False)
    panel_df.to_csv(panel_csv,index=False)

    manifest.write_text(json.dumps({
        "version":VERSION,
        "development_dataset":"GSE10846",
        "locked_input_shape":list(X.shape),
        "class_counts":{"GCB":int(np.sum(y==0)),"ABC":int(np.sum(y==1))},
        "n_resamples":args.n_resamples,
        "sample_fraction":args.sample_fraction,
        "panel_size":args.panel_size,
        "random_seed":args.seed,
        "variance_pool":int(getattr(v7,"VARIANCE_POOL",len(frozen_pool))),
        "softspaces_P_DIM":int(getattr(v3,"P_DIM",-1)),
        "frozen_full_development_panel":frozen_panel,
        "locked_cv_splits":str(splits_path),
        "locked_cv_splits_sha256":sha256_file(splits_path),
        "preprocessing_source":"v41_7_corrected_preprocessing_audit.py",
        "ranking_source":"v41_3_softspaces_mapping_probe.py",
        "null_definition":"random 16 from same resample-specific variance pool",
        "external_labels_used":False,
        "GSE87371_used":False
    },indent=2),encoding="utf-8")

    frozen_stats=gene_df[gene_df.in_frozen_16].sort_values("selection_frequency",ascending=False)

    lines=[
        f"=== {VERSION} FEATURE STABILITY ===","",
        f"Samples: {X.shape[0]}",
        f"Genes: {X.shape[1]}",
        f"Resamples: {args.n_resamples}",
        f"Sample fraction: {args.sample_fraction}",
        f"Panel size: {args.panel_size}",
        f"VARIANCE_POOL: {getattr(v7,'VARIANCE_POOL','unknown')}",
        f"P_DIM: {getattr(v3,'P_DIM','unknown')}","",
        "Full-development 16-gene panel:"
    ]
    lines += [f"{i:2d}. {g:<20} score={s:.8f}" for i,(g,s) in enumerate(zip(frozen_panel,frozen_scores),1)]
    lines += [
        "",
        f"REAL mean pairwise Jaccard: {real_mean:.6f}",
        f"REAL 95% CI: [{real_lo:.6f}, {real_hi:.6f}]",
        f"NULL mean pairwise Jaccard: {null_mean:.6f}",
        f"NULL 95% CI: [{null_lo:.6f}, {null_hi:.6f}]",
        "",
        f"Mean overlap with frozen16: {ov_mean:.6f}",
        f"95% CI: [{ov_lo:.6f}, {ov_hi:.6f}]",
        "",
        f"Stable core >=80% ({len(stable80)}): {', '.join(stable80) if stable80 else '(none)'}",
        f"Stable core >=60% ({len(stable60)}): {', '.join(stable60) if stable60 else '(none)'}",
        f"Stable core >=50% ({len(stable50)}): {', '.join(stable50) if stable50 else '(none)'}",
        "",
        "Frozen 16 selection frequencies:"
    ]
    for _,row in frozen_stats.iterrows():
        lines.append(f"{row.gene:<20} selection={row.selection_frequency:.3f} pool={row.variance_pool_frequency:.3f}")

    txt.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n"+"\n".join(lines))
    print("\nWrote:")
    for p in (gene_csv,panel_csv,txt,manifest): print(" ",p)

if __name__=="__main__":
    main()
