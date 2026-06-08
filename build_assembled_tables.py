#!/usr/bin/env python3
"""Assemble verified multi-page groups into single logical tables for the explorer.

Conservative per spot-verify: the AI chop over-groups on NON-contiguous pages that
merely share age-bands (different territories/sections). So we split each chop
group into CONTIGUOUS printed-page runs and assemble only those (the reliable
left/right-half / continuation / district-series stitches).

Per assembled table:
  - members = the contiguous run's pages, de-duped by physical page (source image)
  - canonical age-band rows from the best-labeled member, propagated to label-dropped pages
  - member facts re-keyed under one assembled table_id, columns unioned, coords de-duped
Outputs: assembled_facts.csv, assembled_catalog.csv, assembled_members.json (subsumed).
"""
import os, re, json, csv
from collections import defaultdict, Counter

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
GAP = 3
AGE_RE = re.compile(r"^\s*\d+\s*[-–]\s*\d+|^\s*\d+\s*(and over|\+|plus|-)|^\s*under|total|all ages", re.I)
# output columns identical to unified_facts / keeper_facts so the DB unions cleanly
import build_query_dataset as B
OUT_COLS = B.OUT_COLS
CAT_COLS = ["table_id", "status", "year", "decade", "canonical_name", "entity_type", "parent_geo",
            "broad_region", "geo_level", "canonical_topic", "subject", "measure_class", "additive",
            "table_family", "measure_category", "dims_present", "age_coverage", "n_facts",
            "review_note", "source_image", "member_images", "title"]


def label_quality(labels):
    rl = [x for x in labels if x]
    return sum(1 for x in rl if AGE_RE.search(x)) / len(rl) if rl else 0.0


def main():
    reads = {r["table_id"]: r for r in json.load(open(os.path.join(V1, "page_reads_fixed.json")))}
    groups = json.load(open(os.path.join(V1, "chop_groups.json")))
    prov = {r["table_id"]: r for r in csv.DictReader(open(os.path.join(V1, "provenance.csv")))}

    # collect member table_ids we need facts for
    need = set()
    for g in groups:
        need.update(g["member_table_ids"])

    # facts (ordered) + denorm template per member table_id
    facts = defaultdict(list)
    denorm = {}
    for src in ("unified_facts.csv", "keeper_facts.csv"):
        p = os.path.join(V1, src)
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            t = r["table_id"]
            if t in need:
                facts[t].append(r)
                if t not in denorm:
                    denorm[t] = {k: r.get(k, "") for k in B.DENORM}

    def distinct_rows(t):
        out = []
        for f in facts.get(t, []):
            rl = (f.get("row_label_std", "") or "").strip()
            if not out or out[-1] != rl:
                out.append(rl)
        return [x for x in out if x != ""]

    asm_facts, asm_cat, subsumed = [], [], {}
    fid = 1
    gi = 0
    for g in groups:
        # member pages, de-duped by physical page (source image basename)
        mem = []
        seen_pg = set()
        for t in g["member_table_ids"]:
            pg = reads.get(t, {}).get("page_number", "")
            simg = os.path.basename((prov.get(t, {}).get("source_image", "") or t))
            key = (pg, simg)
            if key in seen_pg:
                continue
            seen_pg.add(key)
            mem.append((int(pg) if str(pg).isdigit() else 0, t))
        mem.sort()
        # split into contiguous printed-page runs
        runs, cur = [], [mem[0]] if mem else []
        for it in mem[1:]:
            if it[0] - cur[-1][0] <= GAP:
                cur.append(it)
            else:
                runs.append(cur); cur = [it]
        if cur:
            runs.append(cur)
        for run in runs:
            members = [t for _, t in run if facts.get(t)]
            if len(members) < 2:
                continue
            gi += 1
            aid = f"asm_{g['dochash']}_{gi:04d}"
            # canonical rows = best-labeled member at modal row count
            counts = Counter(len(distinct_rows(t)) for t in members)
            modal = counts.most_common(1)[0][0]
            best = max((t for t in members if len(distinct_rows(t)) == modal),
                       key=lambda t: label_quality(distinct_rows(t)), default=members[0])
            canon = distinct_rows(best)
            best_q = label_quality(canon)
            dn = denorm.get(best, {})
            # re-key facts, propagate weak row labels by position, dedupe coords
            coords = set()
            n_emit = 0
            for t in members:
                weak = label_quality(distinct_rows(t)) < 0.4 and best_q >= 0.5 and len(distinct_rows(t)) == modal
                # build position->canonical map for this member if propagating
                row_seq, last = [], None
                for f in facts[t]:
                    rl = (f.get("row_label_std", "") or "").strip()
                    if rl != last:
                        row_seq.append(rl); last = rl
                pos = -1; lastrl = None
                for f in facts[t]:
                    rl = (f.get("row_label_std", "") or "").strip()
                    if rl != lastrl:
                        pos += 1; lastrl = rl
                    eff_row = canon[pos] if (weak and pos < len(canon)) else rl
                    key = (f.get("section", ""), eff_row, f.get("column_group_std", ""),
                           f.get("sub_column_std", ""), f.get("sex", ""), f.get("religion", ""))
                    if key in coords:
                        continue
                    coords.add(key)
                    row = {k: f.get(k, "") for k in B.KEEP}
                    row["table_id"] = aid
                    row["row_label_std"] = eff_row
                    row["fact_id"] = f"A{fid:09d}"; fid += 1
                    row.update(dn)
                    row["status"] = "assembled"
                    row["age_aggregate"] = B._agg_flag(f)
                    asm_facts.append(row); n_emit += 1
            member_imgs = ";".join(prov.get(t, {}).get("source_image", "") for t in members)
            p0 = prov.get(best, {})
            asm_cat.append({"table_id": aid, "status": "assembled",
                            "year": dn.get("year", ""), "decade": dn.get("decade", ""),
                            "canonical_name": dn.get("canonical_name", ""), "entity_type": dn.get("entity_type", ""),
                            "parent_geo": dn.get("parent_geo", ""), "broad_region": dn.get("broad_region", ""),
                            "geo_level": dn.get("geo_level", ""), "canonical_topic": dn.get("canonical_topic", ""),
                            "subject": dn.get("subject", ""), "measure_class": dn.get("measure_class", ""),
                            "additive": dn.get("additive", ""), "table_family": dn.get("table_family", ""),
                            "measure_category": dn.get("measure_category", ""), "dims_present": "",
                            "age_coverage": "", "n_facts": n_emit,
                            "review_note": f"assembled {len(members)} pages: {g['subject'][:80]}",
                            "source_image": p0.get("source_image", ""), "member_images": member_imgs,
                            "title": g["subject"][:120]})
            subsumed[aid] = members

    with open(os.path.join(V1, "assembled_facts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS, extrasaction="ignore")
        w.writeheader(); [w.writerow(r) for r in asm_facts]
    with open(os.path.join(V1, "assembled_catalog.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAT_COLS, extrasaction="ignore")
        w.writeheader(); [w.writerow(r) for r in asm_cat]
    json.dump(subsumed, open(os.path.join(V1, "assembled_members.json"), "w"))

    print(f"assembled multi-page tables: {len(asm_cat)}")
    print(f"  member pages subsumed: {sum(len(m) for m in subsumed.values())}")
    print(f"  assembled facts: {len(asm_facts):,}")
    print(f"  multi-page tables >=3 pages: {sum(1 for m in subsumed.values() if len(m)>=3)}")


if __name__ == "__main__":
    main()
