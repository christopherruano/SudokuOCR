#!/usr/bin/env python3
"""Restore cell facts for published tables whose facts were lost before the Aug-6 build.

A table qualifies only if (a) it is in the site's status set, (b) the current DB has no
facts for it, (c) its page image is NOT also a current keeper/assembled table (those are
superseded duplicates and are dropped from the site instead), and (d) the June-13 snapshot
(versions/pre_reingest_20260613) has its facts. Restored ids are logged to
atlas/out/restored_from_june.csv. Writes into the LOCAL derived DB copy only.
"""
import sqlite3, re, json, csv, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CUR = os.path.join(ROOT, "export_final/sendout_final/sendout_v1/census_explorer.sqlite")
OLD = "/Volumes/Expansion/SudokuOCR/export_final/sendout_final/sendout_v1/versions/pre_reingest_20260613/census_explorer.sqlite"

def hashes(*s):
    return {h for x in s for h in re.findall(r"[0-9a-f]{40}", x or "")}

def main():
    cur = sqlite3.connect(CUR)
    old = sqlite3.connect(f"file:{OLD}?mode=ro&immutable=1", uri=True)
    keep = set()
    for tid, si, mi in cur.execute("select table_id,source_image,member_images from tables "
                                   "where status in ('keeper_accepted','keeper_review','assembled')"):
        keep |= hashes(tid, si, mi)
    cand = [t for (t,) in cur.execute("select t.table_id from tables t where t.status in ('confirmed_sent','recovered') "
                                      "and not exists (select 1 from facts f where f.table_id=t.table_id)")]
    have = {t for (t,) in old.execute("select distinct table_id from facts")}
    todo = [t for t in cand if t in have and not (hashes(t) & keep)]
    cols = [r[1] for r in cur.execute("pragma table_info(facts)")]
    n = 0
    with open(os.path.join(ROOT, "atlas/out/restored_from_june.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["table_id", "n_facts", "source"])
        for t in todo:
            rows = old.execute(f"select {','.join(chr(34)+c+chr(34) for c in cols)} from facts where table_id=?", (t,)).fetchall()
            cur.executemany(f"insert into facts values ({','.join('?'*len(cols))})", rows)
            w.writerow([t, len(rows), "versions/pre_reingest_20260613/census_explorer.sqlite"])
            n += len(rows)
    cur.commit()
    sup = [t for t in cand if hashes(t) & keep]
    json.dump(sup, open(os.path.join(ROOT, "atlas/out/superseded_ids.json"), "w"))
    print(f"no-facts tables: {len(cand)} | superseded by a newer read: {len(sup)} | restored: {len(todo)} tables, {n} facts "
          f"| still without facts: {len(cand) - len(sup) - len(todo)}")

if __name__ == "__main__":
    main()
