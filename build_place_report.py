#!/usr/bin/env python3
"""Place-standardization audit: run every raw place label in the corpus through
canonical_places.canonicalize and emit export/place_canonical_report.csv so mapping
gaps are visible. Read-only over corpus inputs; writes only the report CSV.

Label sources:
  run_title_label  - what build_best_dataset.province_of() extracts per run
                     (export/tableVII_runs.json x export/document_titles.csv)
  run_region_field - the raw `region` bucket string on each run
  geo_raw_name     - raw_geo_name rows from the explorer geo table, with the geo
                     dict's own canonical/entity_type carried for cross-checking

  python3 build_place_report.py
"""
import csv, json, os, re, sqlite3
from collections import Counter

from canonical_places import CANON, canonicalize

ROOT = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(ROOT, "export_final/sendout_final/sendout_v1/census_explorer.sqlite")
OUT = os.path.join(ROOT, "export/place_canonical_report.csv")


def province_of(title, region):
    # mirror of build_best_dataset.province_of (kept in sync by hand; do not import - that
    # module imports census_review_app and mutates sys.argv on import)
    m = re.search(r"vol(?:ume)?\.?\s*[IVXLC\d]+[-A-B]*\s*[,.]\s*([^.:;]+?)(?:\s*[.:;]|\s+pt\b|\s+part\b|$)",
                  title or "", re.I)
    if m:
        return re.sub(r"^the\s+", "", m.group(1).strip())
    return (region or "").strip() or "(unknown)"


def main():
    rows = []

    runs = json.load(open(os.path.join(ROOT, "export/tableVII_runs.json")))
    titles = {r["saoa"]: r for r in csv.DictReader(open(os.path.join(ROOT, "export/document_titles.csv")))}
    title_labels, region_labels = Counter(), Counter()
    for run in runs:
        t = titles.get(str(run["saoa"]), {})
        title_labels[province_of(t.get("official_title", ""), run.get("region", ""))] += 1
        region_labels[(run.get("region") or "").strip()] += 1
    for lbl, n in title_labels.items():
        rows.append(("run_title_label", lbl, n, "", ""))
    for lbl, n in region_labels.items():
        rows.append(("run_region_field", lbl, n, "", ""))

    con = sqlite3.connect(DB)
    for raw, n, gcanon, gtype in con.execute(
            "SELECT raw_geo_name, n_tables, canonical_name, entity_type FROM geo"):
        rows.append(("geo_raw_name", raw or "", int(n or 0), gcanon or "", gtype or ""))
    con.close()

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["raw_label", "source", "n_occurrences", "canonical_name", "kind",
                    "flag", "geo_canonical", "geo_entity_type"])
        n_gap = 0
        for source, lbl, n, gcanon, gtype in sorted(rows, key=lambda r: (r[0], -r[2], r[1])):
            canon, kind = canonicalize(lbl)
            # fall-through = kind 'province' but the name isn't a known CANON canonical
            flag = ""
            if kind == "province" and canon not in CANON:
                flag = "UNMAPPED_FALLTHROUGH"
                n_gap += 1
            w.writerow([lbl, source, n, canon or "", kind, flag, gcanon, gtype])
    print(f"wrote {OUT}: {len(rows)} labels, {n_gap} unmapped fall-throughs")


if __name__ == "__main__":
    main()
