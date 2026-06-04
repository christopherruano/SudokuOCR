#!/usr/bin/env python3
"""Extract tables from the corpus in an organized way (for collaborators).

Filter by region / year / subject / measure / geography / status and dump each
matching table as a clean matrix CSV (and/or Excel) into an output folder, named
by year_place_title. Reuses the explorer's matrix reconstruction.

Examples:
  python3 extract_tables.py --region "South India" --year-from 1901 --year-to 1911
  python3 extract_tables.py --subject population --measure count --fmt both --out ./mytables
  python3 extract_tables.py --geography Mysore --subject civil_condition
"""
import argparse, os, sqlite3, sys
import census_explorer as ce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./extracted_tables")
    ap.add_argument("--region"); ap.add_argument("--geography"); ap.add_argument("--subject")
    ap.add_argument("--measure", help="measure_class: count/distribution/rate/percentage/ratio/variation")
    ap.add_argument("--entity-type"); ap.add_argument("--year"); ap.add_argument("--year-from", type=int)
    ap.add_argument("--year-to", type=int)
    ap.add_argument("--status", default="confirmed_sent,recovered", help="comma list")
    ap.add_argument("--fmt", choices=["csv", "xlsx", "both"], default="csv")
    a = ap.parse_args()

    where, params = [], []
    sts = [s.strip() for s in a.status.split(",") if s.strip()]
    where.append(f"status IN ({','.join('?'*len(sts))})"); params += sts
    for col, val in (("broad_region", a.region), ("canonical_name", a.geography),
                     ("subject", a.subject), ("measure_class", a.measure),
                     ("entity_type", a.entity_type), ("year", a.year)):
        if val:
            where.append(f"{col}=?"); params.append(val)
    if a.year_from:
        where.append("CAST(year AS INTEGER)>=?"); params.append(a.year_from)
    if a.year_to:
        where.append("CAST(year AS INTEGER)<=?"); params.append(a.year_to)

    con = sqlite3.connect(f"file:{ce.DB}?mode=ro", uri=True); con.row_factory = sqlite3.Row
    tbls = con.execute(f"SELECT * FROM tables WHERE {' AND '.join(where)} ORDER BY year, canonical_name", params).fetchall()
    if not tbls:
        print("no tables match those filters."); sys.exit(0)
    os.makedirs(a.out, exist_ok=True)
    seen = set()
    for t in tbls:
        facts = con.execute("SELECT * FROM facts WHERE table_id=? ORDER BY rowid", (t["table_id"],)).fetchall()
        ms = ce.build_matrices(facts)
        nm = ce.safe_filename(t); base = nm; k = 2
        while nm in seen:
            nm = f"{base}_{k}"; k += 1
        seen.add(nm)
        if a.fmt in ("csv", "both"):
            open(os.path.join(a.out, nm + ".csv"), "w").write(ce.build_table_csv(t, ms))
        if a.fmt in ("xlsx", "both"):
            with open(os.path.join(a.out, nm + ".xlsx"), "wb") as fh:
                fh.write(ce.build_table_xlsx(t, ms))
    con.close()
    print(f"extracted {len(tbls)} tables ({a.fmt}) -> {a.out}/")


if __name__ == "__main__":
    main()
