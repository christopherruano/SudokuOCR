#!/usr/bin/env python3
"""Dry-run the generic converter on sample frozen non-VII pages: show parsed structure
and frozen-column count before (corpus) vs after (generic Mistral parse). No writes."""
import sys, os, json, glob
sys.argv_keep = sys.argv
sys.argv = ["x"]
import run_mistral_ocr as RM
import detect_misfires as MF
import parse_mistral_generic as G

ROOT = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")


def corpus_path(ih):
    g = glob.glob(os.path.join(CORPUS, "*", f"{ih}_oneshot.json"))
    return g[0] if g else None


def frozen_count(d):
    n = 0
    for s in d.get("data", []):
        rows = s.get("rows", [])
        if not rows:
            continue
        rh = next((k for k in rows[0] if not isinstance(rows[0].get(k), dict)), None)
        n += len(MF.section_misfires(rows, rh))
    return n


def main():
    ihs = [x.strip() for x in open(sys.argv_keep[1])] if len(sys.argv_keep) > 1 else \
          [x.strip() for x in open("/private/tmp/claude-501/-Users-chrisruano-Research-SudokuOCR/5f3ddb5c-80ba-4984-9580-17da5d5af496/scratchpad/samples.txt") if x.strip()]
    for ih in ihs:
        cp = corpus_path(ih)
        md = os.path.join(RM.OUT, f"{ih}.mistral.md")
        if not cp or not os.path.exists(md):
            print(f"{ih[:12]}: missing corpus or md"); continue
        old = json.load(open(cp))
        title = (old.get("metadata", {}) or {}).get("title", "")
        rowkey = None
        for s in old.get("data", []):
            if s.get("rows"):
                rowkey = next((k for k in s["rows"][0] if not isinstance(s["rows"][0].get(k), dict)), None)
                break
        secs = G.parse_sections(md, rowkey=rowkey or "row")
        new = G.to_oneshot(secs, title, old.get("source_image", ""))
        of, nf = frozen_count(old), frozen_count(new)
        orows = sum(len(s.get("rows", [])) for s in old.get("data", []))
        nrows = sum(len(s.get("rows", [])) for s in new.get("data", []))
        ngroups = len(new["metadata"]["column_groups"])
        print(f"\n{'='*90}\n{ih[:12]}  {title[:60]}")
        print(f"  FROZEN  {of:3d} -> {nf:3d}   rows {orows}->{nrows}  sections {len(old.get('data',[]))}->{len(secs)}  groups(new)={ngroups}")
        print(f"  new groups: {new['metadata']['column_groups'][:12]}")
        for s in new["data"][:4]:
            r0 = s["rows"][0]
            print(f"    [{s['name'][:28]}] {len(s['rows'])}r  sample: {json.dumps(r0)[:200]}")


if __name__ == "__main__":
    main()
