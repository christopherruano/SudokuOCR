#!/usr/bin/env bash
# Dump pending re-OCR flag notes (what the reviewer wants the agent to look at). The agent checks this.
cd "$(dirname "$0")"
[ -f export/reocr_notes.json ] || { echo "no re-OCR notes yet"; exit 0; }
python3 - <<'PY'
import json
notes = json.load(open('export/reocr_notes.json'))
pend = [(ih, n) for ih, n in notes.items() if not n.get('done')]
print(f"{len(pend)} pending noted re-OCR flag(s) for the agent to act on:")
for ih, n in pend:
    print(f"  [{n.get('ts')}] {ih}")
    print(f"      note : {n.get('note')}")
    print(f"      page : {n.get('title','')}")
PY
