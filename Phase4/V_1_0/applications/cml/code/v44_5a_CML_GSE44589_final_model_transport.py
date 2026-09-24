#!/usr/bin/env python3
from pathlib import Path
import csv, gzip, hashlib, json, re
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FROZEN_R = 16
TOP_K = 256

def project_dir():
    return Path(__file__).resolve().parent.parent

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def parse_tab_line(line):
    return [x.strip().strip('"') for x in line.rstrip("\r\n").split("\t")]

def parse_gse130404(path):
    sample_meta = defaultdict(list)
    inside = False
    header = None
    probes, rows = [], []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not inside and line.startswith("!Sample_"):
                parts = parse_tab_line(line)
                sample_meta[parts[0]].append(parts[1:])
                continue
            if line.startswith("!series_matrix_table_begin"):
                inside = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if not inside:
                continue
            if header is None:
                header = parse_tab_line(line)
                continue
            if line.strip():
                parts = line.rstrip("\r\n").split("\t")
                probes.append(parts[0].strip().strip('"'))
                rows.append([float(x.strip().strip('"')) for x in parts[1:]])
    sample_ids = header[1:]
    X_probe = pd.DataFrame(np.asarray(rows, dtype=float).T, index=sample_ids, columns=probes)
    n = len(sample_ids)
    meta_rows = [{"geo_accession": gsm} for gsm in sample_ids]
    for key, occurrences in sample_meta.items():
        for oi, vals in enumerate(occurrences, 1):
            if len(vals) != n:
                continue
            field = key if len(occurrences) == 1 else f"{key}__{oi}"
            for i, v in enumerate(vals):
                meta_rows[i][field] = v
    parsed = []
    for row in meta_rows:
        chars = {}
        for k, v in row.items():
            if k.startswith("!Sample_characteristics_ch1") and ":" in str(v):
                name, value = str(v).split(":", 1)
                chars[name.strip().lower()] = value.strip()
        parsed.append({
            "geo_accession": row["geo_accession"],
            "disease_stage": chars.get("disease stage", ""),
            "bcr_abl1_3m": chars.get("bcr-abl1 at 3 month", "")
        })
    return X_probe, pd.DataFrame(parsed).set_index("geo_accession")

def parse_platform(path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    b = next(i for i,x in enumerate(lines) if x.startswith("!platform_table_begin"))
    e = next(i for i,x in enumerate(lines) if x.startswith("!platform_table_end"))
    table = lines[b+1:e]
    header = table[0].split("\t")
    low = {x.strip().lower(): x for x in header}
    id_col = low.get("id")
    symbol_col = next((low[c] for c in ["symbol","gene symbol","gene_symbol","genesymbol"] if c in low), None)
    entrez_col = next((orig for k,orig in low.items() if "entrez" in k.replace(" ","").replace("_","")), None)
    if id_col is None or symbol_col is None:
        raise RuntimeError(f"Cannot identify platform columns in {path.name}")
    symbol_to_probes = defaultdict(set)
    symbol_to_entrez = defaultdict(set)
    entrez_to_probes = defaultdict(set)
    reader = csv.DictReader(table[1:], fieldnames=header, delimiter="\t")
    for row in reader:
        probe = str(row.get(id_col,"")).strip()
        raw_sym = str(row.get(symbol_col,"")).strip()
        for sep in ("///",";",","):
            raw_sym = raw_sym.replace(sep,"|")
        syms = {x.strip() for x in raw_sym.split("|") if x.strip() and x.strip().upper() not in {"---","NA","N/A","NULL"}}
        eids = set()
        if entrez_col:
            eids = set(re.findall(r"(?<!\d)(\d+)(?!\d)", str(row.get(entrez_col,""))))
            eids.discard("0")
        for sym in syms:
            symbol_to_probes[sym].add(probe)
            symbol_to_entrez[sym].update(eids)
        for eid in eids:
            entrez_to_probes[eid].add(probe)
    return {
        "symbol_to_probes": dict(symbol_to_probes),
        "symbol_to_entrez": dict(symbol_to_entrez),
        "entrez_to_probes": dict(entrez_to_probes)
    }

def aggregate_probe_to_gene(X_probe, mapping):
    out = {}
    for gene, probes in mapping.items():
        usable = [p for p in probes if p in X_probe.columns]
        if usable:
            out[gene] = X_probe.loc[:, usable].mean(axis=1)
    return pd.DataFrame(out, index=X_probe.index)

def make_y(meta):
    y=[]
    for gsm,row in meta.iterrows():
        if str(row["disease_stage"]).strip().lower() != "diagnostic chronic phase":
            raise RuntimeError(f"{gsm}: unexpected disease stage")
        resp = str(row["bcr_abl1_3m"]).strip()
        if resp == ">10%": y.append(1)
        elif resp == "<10%": y.append(0)
        else: raise RuntimeError(f"{gsm}: unresolved development response {resp!r}")
    return np.asarray(y, dtype=int)

def main():
    project = project_dir()
    docs = project/"docs"
    ds = project/"results"/"direct_subspace"

    protocol = docs/"v44_5_CML_EXTERNAL_GENERALIZATION_PREREGISTRATION.txt"
    pmanifest = docs/"v44_5_CML_EXTERNAL_GENERALIZATION_PREREGISTRATION_manifest.json"
    universe = ds/"v44_1b_eligible_genes.txt"

    for p in [protocol,pmanifest,universe]:
        if not p.exists():
            raise FileNotFoundError(p)

    pm = json.loads(pmanifest.read_text(encoding="utf-8"))
    expected = pm["protocol_sha256"]
    actual = sha256_file(protocol)

    print("=== v44.5a FINAL DEVELOPMENT MODEL + GSE44589 TECHNICAL TRANSPORT ===")
    print("Expected protocol SHA:", expected)
    print("Actual protocol SHA:  ", actual)
    if expected != actual:
        raise SystemExit("FAIL: v44.5 protocol SHA mismatch.")
    print("PASS: v44.5 protocol verified.")
    print("External response outcomes used: NO\n")

    eligible = [x.strip() for x in universe.read_text(encoding="utf-8").splitlines() if x.strip()]

    series = project/"data"/"external"/"GSE130404"/"raw"/"GSE130404_series_matrix.txt.gz"
    gpl10558 = project/"data"/"external"/"GSE130404"/"platform"/"GPL10558_full_geo_table.txt"
    gpl570 = project/"data"/"external"/"GSE44589"/"platform"/"GPL570_full_geo_table.txt"
    for p in [series,gpl10558,gpl570]:
        if not p.exists():
            raise FileNotFoundError(p)

    X_probe, meta = parse_gse130404(series)
    devp = parse_platform(gpl10558)
    extp = parse_platform(gpl570)
    X_gene = aggregate_probe_to_gene(X_probe, devp["symbol_to_probes"])
    y = make_y(meta)

    available = [g for g in eligible if g in X_gene.columns]
    if len(available) < TOP_K:
        raise RuntimeError(f"Only {len(available)} eligible genes available.")

    Xall = X_gene.loc[:, available].to_numpy(dtype=float)
    gene_arr = np.asarray(available, dtype=str)
    var = np.var(Xall, axis=0, ddof=1)
    order = np.lexsort((gene_arr, -var))
    top_idx = order[:TOP_K]
    top_genes = gene_arr[top_idx].tolist()
    Xraw = Xall[:, top_idx]

    scaler = StandardScaler()
    X = scaler.fit_transform(Xraw)
    X0, X1 = X[y==0], X[y==1]
    H = np.einsum("ni,nj->ij",X1,X1)/len(X1) - np.einsum("ni,nj->ij",X0,X0)/len(X0)
    H = 0.5*(H+H.T)
    evals,U = np.linalg.eigh(H)
    eo = np.argsort(np.abs(evals))[::-1]
    U16 = U[:, eo[:FROZEN_R]]

    model = LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced", max_iter=5000, random_state=445001)
    model.fit(X@U16, y)

    rows=[]
    ok_count=0
    for i,gene in enumerate(top_genes):
        exact = gene in extp["symbol_to_probes"]
        dev_eids = set(devp["symbol_to_entrez"].get(gene,set()))
        matched = {eid for eid in dev_eids if extp["entrez_to_probes"].get(eid)}
        if exact:
            status, ok = "EXACT_SYMBOL", True
        elif len(matched)==1:
            status, ok = "ENTREZ_ID_MATCH", True
        elif len(matched)>1:
            status, ok = "AMBIGUOUS_ENTREZ", False
        else:
            status, ok = "UNRESOLVED", False
        ok_count += int(ok)
        rows.append({
            "coordinate_index": i,
            "development_gene": gene,
            "development_entrez_ids": "|".join(sorted(dev_eids)),
            "matched_entrez_ids": "|".join(sorted(matched)),
            "gpl570_status": status,
            "transportable": int(ok),
            "projector_diagonal_mass": float(np.sum(U16[i,:]**2))
        })

    df = pd.DataFrame(rows)
    retained = float(df.loc[df["transportable"]==1,"projector_diagonal_mass"].sum())
    status = "TECHNICALLY EVALUABLE" if ok_count==TOP_K else "TECHNICALLY NON-EVALUABLE"

    model_path = ds/"v44_5a_frozen_development_model.npz"
    audit_path = ds/"v44_5a_GSE44589_transport_audit.csv"
    summary_path = ds/"v44_5a_GSE44589_transport_summary.txt"
    manifest_path = ds/"v44_5a_manifest.json"

    np.savez_compressed(
        model_path,
        genes=np.asarray(top_genes,dtype="U"),
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        U16=U16,
        eigenvalues=evals[eo[:FROZEN_R]],
        logistic_coef=model.coef_,
        logistic_intercept=model.intercept_,
        classes=model.classes_
    )
    df.to_csv(audit_path,index=False)

    summary = [
        "=== Soft Spaces / CML v44.5a FINAL DEVELOPMENT MODEL + GSE44589 TECHNICAL TRANSPORT ===",
        "",
        "External response outcomes used: NO",
        f"Eligible universe: {len(available)}",
        f"Final Top-256: {TOP_K}",
        f"Frozen rank: {FROZEN_R}",
        f"Transportable coordinates: {ok_count} / {TOP_K}",
        f"Retained projector mass: {retained:.6f} / {FROZEN_R:.6f}",
        f"Retained projector fraction: {retained/FROZEN_R:.6f}",
        "",
        f"v44.5a STATUS: {status}",
        "",
        "No external response score has been calculated."
    ]
    summary_path.write_text("\n".join(summary)+"\n",encoding="utf-8")

    manifest_path.write_text(json.dumps({
        "version":"v44.5a",
        "status":status,
        "top_k":TOP_K,
        "frozen_r":FROZEN_R,
        "transportable_coordinates":ok_count,
        "retained_projector_mass":retained,
        "retained_projector_fraction":retained/FROZEN_R,
        "external_outcomes_used":False,
        "sha256":{
            "v44_5_protocol":actual,
            "eligible_universe":sha256_file(universe),
            "frozen_model":sha256_file(model_path),
            "transport_audit":sha256_file(audit_path),
            "execution_script":sha256_file(Path(__file__).resolve())
        }
    },indent=2),encoding="utf-8")

    print(summary_path.read_text(encoding="utf-8"))

if __name__=="__main__":
    main()
