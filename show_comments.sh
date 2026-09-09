#!/usr/bin/env bash
cd "$(dirname "$0")"
python3 - <<'PY'
import json, os
for f in ["export/review_verdicts/tables.json","export/review_verdicts/comments.json","export/review_verdicts/assembly.json"]:
    if not os.path.exists(f): continue
    v=json.load(open(f))
    notes=[(k,e.get("note") if isinstance(e,dict) else e,(e.get("v") if isinstance(e,dict) else "")) for k,e in v.items() if (e.get("note") if isinstance(e,dict) else e)]
    if notes:
        print(f"\n=== {f} ({len(notes)} comments) ===")
        for k,n,verd in notes: print(f"  [{verd}] {k[-28:]}: {n}")
PY
