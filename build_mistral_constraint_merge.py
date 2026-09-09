#!/usr/bin/env python3
"""Mistral fallback for the DIFFICULT broken tables, gated by CONSTRAINT-FAILURE REDUCTION (not the
agreement-with-Gemini gate emit_mistral uses — that's wrong for already-broken Gemini reads). For every
member page that currently fails a constraint AND has cached Mistral markdown, parse the Mistral read,
compute its failures, and OVERLAY the corpus read ONLY IF Mistral has strictly fewer failures and >= as
many rows. Backs up to *_oneshot.premistral.json; never touches human_corrected. Free (uses the cache).
  python3 build_mistral_constraint_merge.py [--apply]
"""
import sys, os, json, glob, shutil
sys.argv_apply = "--apply" in sys.argv
APPLY = "--apply" in sys.argv
sys.argv = ["x"]
import census_review_app as A
import emit_mistral_oneshot as E
import parse_mistral_md as P

V1 = "export_final/sendout_final/sendout_v1"
MD = os.path.join(V1, "mistral_md")


def fails_rows(d):
    f = r = 0
    for s in (d or {}).get("data", []):
        rows = s.get("rows", [])
        _, fl = A.constraint_report(rows)
        f += len(fl); r += len(rows)
    return f, r


def main():
    # member imghashes of pages that currently fail a constraint
    targets = {}
    for run in A.assembly_runs():
        for m in run.get("members", []):
            d, ih, _ = A.load_table(m.get("table_id", ""))
            if not d or not ih:
                continue
            cf, cr = fails_rows(d)
            if cf > 0:
                targets[ih] = (m.get("table_id", ""), cf, cr,
                               (d.get("metadata", {}) or {}).get("human_corrected"),
                               (d.get("metadata", {}) or {}).get("title", ""),
                               d.get("source_image", ih + ".png"))
    improved = no_md = poor = worse = human = 0
    gain = 0
    for ih, (tid, cf, cr, hc, title, src) in targets.items():
        md = os.path.join(MD, ih + ".mistral.md")
        if not os.path.exists(md):
            no_md += 1; continue
        if hc:
            human += 1; continue
        try:
            secs = P.mistral_parse(md)
            mone = E.to_oneshot(secs, title, src)
        except Exception:
            poor += 1; continue
        mf, mr = fails_rows(mone)
        if mr < 4:
            poor += 1; continue
        if mf < cf and mr >= cr:
            improved += 1; gain += (cf - mf)
            if APPLY:
                gpath = E.find_oneshot(ih)
                if gpath:
                    shutil.copy(gpath, gpath.replace("_oneshot.json", "_oneshot.premistral.json"))
                    json.dump(mone, open(gpath, "w"), indent=1)
        else:
            worse += 1
    print(f"Mistral failure-gated merge ({'APPLIED' if APPLY else 'dry-run'}):")
    print(f"  failing pages with cached Mistral checked: {len(targets) - no_md}")
    print(f"  WOULD OVERLAY (Mistral fewer failures): {improved}  (failure reduction: {gain})")
    print(f"  rejected — Mistral not better: {worse} | poor/thin parse: {poor} | human_corrected: {human} | no cached md: {no_md}")


if __name__ == "__main__":
    main()
