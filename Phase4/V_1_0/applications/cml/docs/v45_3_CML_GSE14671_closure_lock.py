#!/usr/bin/env python3
from pathlib import Path
import hashlib
EXPECTED_SHA256="2fe8940d7cdce28f08f51fc31bdd23336142f4a80a6191bb5fc2b159f4619a89"
p=Path(__file__).resolve().parent/"v45_3_CML_GSE14671_CLOSURE_AUDIT.txt"
a=hashlib.sha256(p.read_bytes()).hexdigest()
print("Expected:",EXPECTED_SHA256)
print("Actual:  ",a)
raise SystemExit("FAIL") if a!=EXPECTED_SHA256 else print("PASS: v45.3 closure unchanged.")
