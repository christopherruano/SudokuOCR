#!/usr/bin/env python3
"""Load the unified dataset into a single indexed SQLite DB for the explorer app.

Tables
  facts   - 785k tidy numeric cells, denormalized (year/geo/topic/measure/dims/status)
  tables  - the catalog / MAP (one row per table, full corpus)
  geo     - geography reference (raw -> canonical, entity_type, region, parent)
  tables_fts - FTS5 over title/topic/region for free-text table discovery
"""
import csv, os, sqlite3

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
DB = os.path.join(V1, "census_explorer.sqlite")

NUM = {"value", "resolved_age_lower", "resolved_age_upper", "year", "n_facts"}


# keeper tables that failed QA stay OFF the explorer (only good ones shown)
BAD_STATUS = {"keeper_badread", "keeper_suspect", "keeper_rejected",
              "keeper_reocr", "keeper_not_table"}


def bad_table_ids(paths):
    """table_ids whose keeper-catalog status failed QA -> kept off the explorer."""
    bad = set()
    for p in paths:
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            if r.get("status", "") in BAD_STATUS:
                bad.add(r["table_id"])
    return bad


def load(con, table, paths, drop_status=False, drop_ids=None):
    """Stream one or more same-schema CSVs into a single table (memory-safe).
    drop_status: skip rows whose status is a failed-QA keeper status.
    drop_ids: skip rows whose table_id is in this set (failed-QA, by id)."""
    if isinstance(paths, str):
        paths = [paths]
    paths = [p for p in paths if os.path.exists(p)]
    cols = list(csv.DictReader(open(paths[0])).fieldnames)
    decl = ", ".join(f'"{c}" {"REAL" if c=="value" else ("INTEGER" if c in NUM else "TEXT")}' for c in cols)
    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.execute(f"CREATE TABLE {table} ({decl})")

    def cast(c, v):
        if c not in NUM or v in ("", "None", None):
            return v if c not in NUM else None
        try:
            return float(v) if c == "value" else int(float(v))
        except ValueError:
            return None

    sql = f"INSERT INTO {table} VALUES ({','.join('?'*len(cols))})"
    n, skip, batch = 0, 0, []
    for p in paths:
        for r in csv.DictReader(open(p)):
            if drop_status and r.get("status", "") in BAD_STATUS:
                skip += 1; continue
            if drop_ids and r.get("table_id", "") in drop_ids:
                skip += 1; continue
            batch.append([cast(c, r.get(c, "")) for c in cols]); n += 1
            if len(batch) >= 50000:
                con.executemany(sql, batch); batch = []
    if batch:
        con.executemany(sql, batch)
    if skip:
        print(f"  ({table}: held back {skip:,} failed-QA keeper rows)")
    return n, cols


def _append_facts(con, path):
    """Append correction rows into the existing facts table (same OUT_COLS schema)."""
    cols = [d[1] for d in con.execute("PRAGMA table_info(facts)").fetchall()]

    def cast(c, v):
        if c not in NUM or v in ("", "None", None):
            return v if c not in NUM else None
        try:
            return float(v) if c == "value" else int(float(v))
        except ValueError:
            return None
    sql = f"INSERT INTO facts VALUES ({','.join('?'*len(cols))})"
    n, batch = 0, []
    for r in csv.DictReader(open(path)):
        batch.append([cast(c, r.get(c, "")) for c in cols]); n += 1
        if len(batch) >= 50000:
            con.executemany(sql, batch); batch = []
    if batch:
        con.executemany(sql, batch)
    return n


def main():
    if os.path.exists(DB):
        os.remove(DB)
    con = sqlite3.connect(DB)
    bad = bad_table_ids([os.path.join(V1, "keeper_catalog.csv")])
    # NO_ASSEMBLY=1 -> show every page as its own read: don't subsume members, don't load assembled_*.
    no_assembly = os.environ.get("NO_ASSEMBLY") == "1"
    # member pages folded into an assembled multi-page table are hidden (show the assembled one)
    amp = os.path.join(V1, "assembled_members.json")
    import json as _json
    subsumed = (set(t for members in _json.load(open(amp)).values() for t in members)
                if (os.path.exists(amp) and not no_assembly) else set())
    asm_facts = [] if no_assembly else [os.path.join(V1, "assembled_facts.csv")]
    asm_catalog = [] if no_assembly else [os.path.join(V1, "assembled_catalog.csv")]
    # Opus-corrected tables: drop their ORIGINAL facts, load the correction instead
    cpath = os.path.join(V1, "corrected_tables.json")
    corrected = set(_json.load(open(cpath))["corrected"]) if os.path.exists(cpath) else set()
    drop = bad | subsumed
    nf, _ = load(con, "facts", [os.path.join(V1, "unified_facts.csv"), os.path.join(V1, "keeper_facts.csv"),
                                *asm_facts], drop_status=True, drop_ids=drop | corrected)
    if corrected:
        cf = os.path.join(V1, "corrected_facts.csv")
        added = _append_facts(con, cf) if os.path.exists(cf) else 0
        nf += added
        print(f"  (overlaid {added:,} corrected cells for {len(corrected)} Opus-re-read tables; originals kept on disk)")
    nt, _ = load(con, "tables", [os.path.join(V1, "catalog_tables.csv"), os.path.join(V1, "keeper_catalog.csv"),
                                 *asm_catalog], drop_status=True, drop_ids=drop)
    if no_assembly:
        print("  (NO_ASSEMBLY=1: every page shown as its own read; assembled multi-page tables skipped)")
    else:
        print(f"  (subsumed {len(subsumed):,} member pages into assembled multi-page tables)")
    ng, _ = load(con, "geo", os.path.join(V1, "geo_reference.csv"))
    # column-check fixes (atlas/apply_column_fixes.py): added/relinked scans and review notes
    ov = os.path.join(V1, "column_fix_overlay.json")
    if os.path.exists(ov):
        ov = _json.load(open(ov))
        for tid, im in ov.get("images", {}).items():
            con.execute("UPDATE tables SET member_images=? WHERE table_id=?", (im, tid))
        for tid, note in ov.get("notes", {}).items():
            old = (con.execute("SELECT review_note FROM tables WHERE table_id=?", (tid,)).fetchone() or [""])[0] or ""
            con.execute("UPDATE tables SET review_note=? WHERE table_id=?", ((old + " | " if old else "") + note, tid))
        for tid, n in (_json.load(open(cpath)).get("counts", {}) if os.path.exists(cpath) else {}).items():
            con.execute("UPDATE tables SET n_facts=? WHERE table_id=?", (n, tid))
        print(f"  (column-fix overlay: {len(ov.get('images', {}))} scan sets, {len(ov.get('notes', {}))} notes)")

    for c in ("canonical_topic", "subject", "measure_class", "additive", "age_aggregate",
              "broad_region", "entity_type", "canonical_name", "year", "measure_unit", "status",
              "table_id", "sex", "table_family"):
        con.execute(f'CREATE INDEX idx_f_{c} ON facts("{c}")')
    for c in ("canonical_topic", "subject", "measure_class", "broad_region", "entity_type",
              "year", "status", "canonical_name"):
        con.execute(f'CREATE INDEX idx_t_{c} ON tables("{c}")')

    # FTS for free-text table discovery
    con.execute("CREATE VIRTUAL TABLE tables_fts USING fts5(table_id, title, canonical_topic, canonical_name, broad_region, content='tables', content_rowid='rowid')")
    con.execute("INSERT INTO tables_fts(rowid, table_id, title, canonical_topic, canonical_name, broad_region) SELECT rowid, table_id, title, canonical_topic, canonical_name, broad_region FROM tables")
    con.commit()
    print(f"census_explorer.sqlite  facts={nf:,}  tables={nt:,}  geo={ng:,}")
    print("  size:", round(os.path.getsize(DB) / 1e6, 1), "MB")
    # sample query
    cur = con.execute("SELECT canonical_topic, broad_region, COUNT(*) n FROM facts GROUP BY 1,2 ORDER BY n DESC LIMIT 5")
    print("  top topic x region:")
    for row in cur.fetchall():
        print("    ", row)
    con.close()


if __name__ == "__main__":
    main()
