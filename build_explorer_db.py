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


def load(con, table, path):
    rows = list(csv.DictReader(open(path)))
    cols = list(rows[0].keys())
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
    con.executemany(
        f"INSERT INTO {table} VALUES ({','.join('?'*len(cols))})",
        [[cast(c, r[c]) for c in cols] for r in rows],
    )
    return len(rows), cols


def main():
    if os.path.exists(DB):
        os.remove(DB)
    con = sqlite3.connect(DB)
    nf, _ = load(con, "facts", os.path.join(V1, "unified_facts.csv"))
    nt, _ = load(con, "tables", os.path.join(V1, "catalog_tables.csv"))
    ng, _ = load(con, "geo", os.path.join(V1, "geo_reference.csv"))

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
