#!/usr/bin/env python3
"""Merge the 12,258 keeper tables into the EXPLORER FORMAT (tidy facts + catalog),
NOT as a raw cell_facts dump. Reads keeper cell_facts_rich.csv only as the source;
reuses build_query_dataset's classify/geo/denorm so keepers look identical to
existing tables in the explorer (catalog tree + matrix-beside-scan views).

Outputs (sendout_v1/):
  keeper_facts.csv     tidy numeric cells (same columns as unified_facts.csv)
  keeper_catalog.csv   one row per keeper table (same columns as catalog_tables.csv)
build_explorer_db.py then loads existing + keeper into one SQLite.
"""
import csv, os, re
from collections import defaultdict, Counter
import build_query_dataset as B

csv.field_size_limit(10 ** 7)
ROOT, V1 = B.ROOT, B.V1
KCAT = os.path.join(ROOT, "export/keeper_inference/source_catalog.csv")
KCELLS = os.path.join(ROOT, "export/keeper_inference/cell_facts_rich.csv")
DOC = re.compile(r"([0-9a-f]{16})_t\d+")

STATUS_MAP = {
    "accepted_strict": "keeper_accepted", "accepted_contextual_unvalidated": "keeper_accepted",
    "accepted_proportional_tolerance": "keeper_accepted",
    "review_family_context": "keeper_review", "review_unvalidated_counts": "keeper_review",
    "review_proportional_failures": "keeper_review",
    "rejected_constraint_failures": "keeper_rejected",
}
CAT_COLS = ["table_id", "status", "year", "decade", "canonical_name", "entity_type",
            "parent_geo", "broad_region", "geo_level", "canonical_topic", "subject",
            "measure_class", "additive", "table_family", "measure_category", "dims_present",
            "age_coverage", "n_facts", "review_note", "source_image", "member_images", "title"]


def main():
    geo = B.load_geo()
    emeta = B.load_meta()
    # backfill year from the document (dochash -> modal year of its tables)
    doc_year = defaultdict(Counter)
    for r in csv.DictReader(open(os.path.join(V1, "provenance.csv"))):
        if r["dochash"] and r["year"].strip():
            doc_year[r["dochash"]][r["year"]] += 1
    dy = {d: c.most_common(1)[0][0] for d, c in doc_year.items()}

    kmeta, kstatus = {}, {}
    for r in csv.DictReader(open(KCAT)):
        tid = r["table_id"]; m = DOC.search(tid); dh = m.group(1) if m else ""
        yr = r.get("year", "").strip() or dy.get(dh, "")
        kmeta[tid] = {"year": yr, "title": r.get("title", ""), "geo_name": r.get("geo_name", ""),
                      "geo_level": r.get("geo_level", ""), "table_family": r.get("table_family", ""),
                      "canonical_topic": r.get("table_family", ""), "measure_category": "",
                      "source_image": r.get("source_image", "")}
        kstatus[tid] = STATUS_MAP.get(r.get("effective_quality_status", ""), "keeper_review")

    # geography over the union so keepers inherit their volume's province by doc-propagation
    geores = B.resolve_geographies({**emeta, **kmeta}, geo)

    fout = open(os.path.join(V1, "keeper_facts.csv"), "w", newline="")
    fw = csv.DictWriter(fout, fieldnames=B.OUT_COLS, extrasaction="ignore"); fw.writeheader()
    cout = open(os.path.join(V1, "keeper_catalog.csv"), "w", newline="")
    cw = csv.DictWriter(cout, fieldnames=CAT_COLS, extrasaction="ignore"); cw.writeheader()

    st_counter = Counter()
    stats = {"fid": 1, "ntab": 0, "nfact": 0}

    def flush(tid, cells):
        st = kstatus.get(tid, "keeper_review"); st_counter[st] += 1
        m = kmeta[tid]; gx = geores.get(tid, {}); ttl = m["title"]
        dc = Counter(c.get("measure_unit", "") for c in cells).most_common(1)
        dom = dc[0][0] if dc else ""
        mx = 0.0
        for c in cells:
            try: mx = max(mx, abs(float(c["value"])))
            except (ValueError, TypeError): pass
        subj, meas, addv = B.classify_subject_measure(m, dom, ttl, mx)
        yr = m["year"]; dec = (yr[:3] + "0s") if yr[:3].isdigit() else ""
        sexes = set(c.get("sex", "") for c in cells); dims = []
        if {"Male", "Female"} & sexes: dims.append("sex")
        if any(c.get("civil_condition") for c in cells): dims.append("civil_condition")
        if any(c.get("religion") for c in cells): dims.append("religion")
        if any(c.get("column_dimension") == "age" or c.get("row_dimension") == "age" for c in cells): dims.append("age")
        ages = [int(c["resolved_age_lower"]) for c in cells if str(c.get("resolved_age_lower", "")).lstrip("-").isdigit()]
        denorm = {"subject": subj, "measure_class": meas, "additive": addv, "year": yr, "decade": dec,
                  "canonical_name": gx.get("canonical_name", ""), "entity_type": gx.get("entity_type", ""),
                  "parent_geo": gx.get("parent_geo", ""), "broad_region": gx.get("broad_region", ""),
                  "geo_level": gx.get("geo_level", ""), "canonical_topic": m["canonical_topic"],
                  "table_family": m["table_family"], "measure_category": "", "source_image": m["source_image"],
                  "title": ttl, "status": st}
        cw.writerow({"table_id": tid, "status": st, "subject": subj, "measure_class": meas, "additive": addv,
                     "review_note": "", "year": yr, "decade": dec,
                     "canonical_name": denorm["canonical_name"], "entity_type": denorm["entity_type"],
                     "parent_geo": denorm["parent_geo"], "broad_region": denorm["broad_region"],
                     "geo_level": denorm["geo_level"], "canonical_topic": denorm["canonical_topic"],
                     "table_family": denorm["table_family"], "measure_category": "",
                     "dims_present": ";".join(dims), "age_coverage": (f"{min(ages)}-{max(ages)}" if ages else ""),
                     "n_facts": (len(cells) if st != "keeper_rejected" else 0),
                     "source_image": m["source_image"], "member_images": m["source_image"], "title": ttl})
        stats["ntab"] += 1
        if st == "keeper_rejected":
            return
        cleaned, _ = B.clean_synth_persons(cells)
        for f in cleaned:
            row = {k: f.get(k, "") for k in B.KEEP}
            row["fact_id"] = f"K{stats['fid']:09d}"; stats["fid"] += 1
            row.update(denorm); row["age_aggregate"] = B._agg_flag(f)
            if row.get("sex") not in B.VALID_SEX: row["sex"] = ""
            if (row.get("civil_condition") or "").replace(".", "").isdigit(): row["civil_condition"] = ""
            fw.writerow(row); stats["nfact"] += 1

    cur, buf = None, []
    for r in csv.DictReader(open(KCELLS)):
        t = r["table_id"]
        if t != cur:
            if cur is not None and cur in kmeta and buf:
                flush(cur, buf)
            cur, buf = t, []
        buf.append(r)
    if cur in kmeta and buf:
        flush(cur, buf)

    fout.close(); cout.close()
    print(f"keeper_catalog.csv: {stats['ntab']:,} tables   by status: {dict(st_counter)}")
    print(f"keeper_facts.csv:   {stats['nfact']:,} tidy cells (accepted+review only)")


if __name__ == "__main__":
    main()
