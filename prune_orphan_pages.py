#!/usr/bin/env python3
"""Prune orphaned static-site pages after an assembly drop.

build_static_site names each page  <safe_filename>_<md5(table_id)[:7]>.html , so the
trailing _<7hex> segment uniquely identifies a live table_id. Any tables/*.html whose
trailing hash is NOT a current DB table_id is an orphan (e.g. an old assembled multi-page
table that no longer exists with NO_ASSEMBLY=1). This removes those .html + matching
data/<page>.csv/.xlsx. Idempotent. Dry-run by default; pass --apply to delete.
  python3 prune_orphan_pages.py [--apply]
"""
import os, re, glob, sqlite3, hashlib, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
DB = os.path.join(V1, "census_explorer.sqlite")
OUT = os.path.join(ROOT, "census_site")
APPLY = "--apply" in sys.argv
HEX7 = re.compile(r"_([0-9a-f]{7})$")


def main():
    con = sqlite3.connect(DB)
    valid = {hashlib.md5(r[0].encode()).hexdigest()[:7]
             for r in con.execute("SELECT table_id FROM tables")}
    print(f"live tables in DB: {len(valid):,}")

    htmls = glob.glob(os.path.join(OUT, "tables", "*.html"))
    orphans = []
    for h in htmls:
        stem = os.path.basename(h)[:-5]            # strip .html
        m = HEX7.search(stem)
        if not m or m.group(1) not in valid:
            orphans.append((h, stem))
    print(f"rendered pages: {len(htmls):,} | orphans (not in DB): {len(orphans):,}")

    if not APPLY:
        for h, _ in orphans[:8]:
            print("  would delete:", os.path.basename(h))
        print("dry-run — pass --apply to delete orphans + their data/ files")
        return

    removed = 0
    for h, stem in orphans:
        for p in (h,
                  os.path.join(OUT, "data", stem + ".csv"),
                  os.path.join(OUT, "data", stem + ".xlsx")):
            if os.path.exists(p):
                os.remove(p); removed += 1
    print(f"deleted {removed} files for {len(orphans):,} orphaned pages")


if __name__ == "__main__":
    main()
