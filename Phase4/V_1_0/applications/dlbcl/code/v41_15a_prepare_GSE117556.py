#!/usr/bin/env python3
from __future__ import annotations
import csv, gzip, hashlib, json, re, shutil, urllib.request
from collections import defaultdict
from pathlib import Path

VERSION="v41.15a"
EXPECTED_PREREG_SHA256="979f7f4880fa3936edcdcd1055641bf5922abcc00caa3917e0fed8c8e464b56b"
FROZEN_PANEL=[
"DNER","SPIC","TCL1B","NPIPA5///NPIPB6///NPIPB8///NPIPB3","EML6","MMP20",
"KCCAT333","IGLJ3///IGLV1-44///CKAP2///IGLV@///IGLC1","RNF183","MYOCD",
"UMODL1","CTAG2","IGK///IGKC","LOC283454///HRK","RGS13","TCL1A"]
SERIES_URL="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE117nnn/GSE117556/matrix/GSE117556_series_matrix.txt.gz"
ANNOT_URL="https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL14nnn/GPL14951/annot/GPL14951.annot.gz"

def project_dir(): return Path(__file__).resolve().parent.parent
def sha256_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()

def find_prereg(project):
    for p in [project/"docs"/"preregistration"/"v41_15_PREREGISTRATION.txt",
              project/"docs"/"preregistration"/"v41_15"/"v41_15_PREREGISTRATION.txt"]:
        if p.exists(): return p
    raise FileNotFoundError("v41_15_PREREGISTRATION.txt not found")

def download(url,p):
    if p.exists() and p.stat().st_size>0:
        print("[download] reuse:",p); return
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(str(p)+".part")
    print("[download]",url)
    req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0 SoftSpaces-v41.15a"})
    with urllib.request.urlopen(req,timeout=120) as r, open(tmp,"wb") as f: shutil.copyfileobj(r,f)
    tmp.replace(p)

def split_symbols(s):
    parts=re.split(r"\s*///\s*|\s*//\s*|[;,|]",str(s or "").strip())
    return [x.strip() for x in parts if x.strip() and x.strip() not in {"---","NA","N/A"}]

def parse_annotation(p):
    with gzip.open(p,"rt",encoding="utf-8",errors="replace") as f:
        header=None
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            header=line.rstrip("\r\n"); break
        if header is None: raise RuntimeError("No annotation header")
        delim="\t" if "\t" in header else ","
        fields=next(csv.reader([header],delimiter=delim))
        lut={x.strip().lower():x for x in fields}
        idcol=next((lut[x] for x in ["id","probe_id","probe id","probeid","ilmnid","array_address_id"] if x in lut),None)
        symcol=next((lut[x] for x in ["gene symbol","gene_symbol","symbol","gene symbols","gene_assignment"] if x in lut),None)
        if not idcol or not symcol: raise RuntimeError(f"Cannot identify columns: {fields}")
        d=defaultdict(set)
        for row in csv.DictReader(f,fieldnames=fields,delimiter=delim):
            probe=str(row.get(idcol,"")).strip()
            if not probe: continue
            for s in split_symbols(row.get(symcol,"")): d[probe].add(s)
    return dict(d),{"probe_id_column":idcol,"gene_symbol_column":symcol}

def parse_series_ids(p):
    ids=set(); inside=False
    with gzip.open(p,"rt",encoding="utf-8",errors="replace") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"): inside=True; continue
            if line.startswith("!series_matrix_table_end"): break
            if not inside or not line.strip() or line.startswith('"ID_REF"') or line.startswith("ID_REF"): continue
            ids.add(line.split("\t",1)[0].strip().strip('"'))
    if not ids: raise RuntimeError("No measured probe IDs found")
    return ids

def main():
    project=project_dir()
    prereg=find_prereg(project)
    actual=sha256_file(prereg)
    print("Expected:",EXPECTED_PREREG_SHA256)
    print("Actual:  ",actual)
    if actual!=EXPECTED_PREREG_SHA256: raise SystemExit("FAIL preregistration SHA mismatch")
    print("PASS preregistration integrity")

    root=project/"data"/"external"/"GSE117556"
    series=root/"raw"/"GSE117556_series_matrix.txt.gz"
    annot=root/"platform"/"GPL14951.annot.gz"
    results=project/"results"/"stability"; results.mkdir(parents=True,exist_ok=True)
    download(SERIES_URL,series); download(ANNOT_URL,annot)

    p2s,cols=parse_annotation(annot)
    measured=parse_series_ids(series)
    s2p=defaultdict(set)
    for probe,syms in p2s.items():
        if probe not in measured: continue
        for s in syms: s2p[s.upper()].add(probe)

    rows=[]
    for rank,coord in enumerate(FROZEN_PANEL,1):
        comps=[x.strip() for x in coord.split("///") if x.strip()]
        matches=[(c,sorted(s2p.get(c.upper(),set()))) for c in comps if s2p.get(c.upper())]
        if not matches:
            status="MISSING"; usable=False; resolved=[]; probes=[]
        else:
            resolved=[c for c,_ in matches]; probes=sorted({p for _,ps in matches for p in ps})
            if len(comps)==1: status="EXACT"; usable=True
            elif len(matches)==1: status="RESOLVED-COMPOUND"; usable=True
            else: status="AMBIGUOUS"; usable=False
        rows.append({"frozen_rank":rank,"frozen_coordinate":coord,"status":status,
                     "resolved_symbols":"///".join(resolved),"n_resolved_symbols":len(resolved),
                     "probe_ids":"|".join(probes),"n_probes":len(probes),
                     "confirmatory_usable":usable})

    audit=results/"v41_15_mapping_audit.csv"
    with open(audit,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    counts=defaultdict(int)
    for r in rows:
        counts[r["status"]]+=1
        print(f'{r["frozen_rank"]:2d}. {r["frozen_coordinate"]:<45} {r["status"]:<18} probes={r["n_probes"]}')
    gate="PASS" if all(r["confirmatory_usable"] for r in rows) else "TECHNICALLY_NON_EVALUABLE"
    print("\nELIGIBILITY GATE:",gate)

    manifest={"version":VERSION,"gse":"GSE117556","gpl":"GPL14951",
              "preregistration_sha256":actual,
              "series_matrix":str(series),"series_matrix_sha256":sha256_file(series),
              "platform_annotation":str(annot),"platform_annotation_sha256":sha256_file(annot),
              "annotation_columns":cols,"status_counts":dict(counts),
              "eligibility_gate":gate,"expression_values_analyzed":False,
              "external_labels_used":False,"mapping_audit":str(audit)}
    mp=results/"v41_15a_prepare_manifest.json"
    mp.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("\nWrote:",audit); print("Wrote:",mp)

if __name__=="__main__":
    main()
