#!/usr/bin/env python3
"""Row-level multi-page stitching for AGE tables (Neil).

Stitch basis: same original volume (dochash) + consecutive printed page numbers +
same table number/subject. Within a logical run, members share ONE row scheme
(the same age bands in the same order). Where a member's row stub was mis-read /
dropped, borrow the canonical labels from the labeled siblings (by position).

Output: stitch_assembly.json
  groups[]: {gid, dochash, table_number, year, region, family,
             members:[{table_id, page, n_rows, label_quality}],
             canonical_rows:[...], propagated:[table_ids needing labels],
             needs_ai: bool}
plus stitch_verify_queue.json for the AI subject-match/row-alignment check.
"""
import os, re, json, csv
from collections import defaultdict, Counter

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
SEG_SPLIT = 12
AGEFAM = {"age_civil_condition", "age_sex_population", "age_by_religion", "proportional_age",
          "caste_class_age", "single_year_age", "age_sex_civil_condition", "infirmity_age"}
AGE_RE = re.compile(r"^\s*\d+\s*[-–]\s*\d+|^\s*\d+\s*(and over|\+|plus|-)|^\s*under|total|all ages", re.I)
CONTINUATION_PAT = re.compile(r"(?:\bcont(?:d|inued)?\.?\b|\bconcl(?:d|uded)?\.?\b|\bconcluded\b)", re.I)
_STOP = {"the", "and", "for", "each", "sex", "table", "subsidiary", "part", "cont", "contd",
         "continued", "concld", "concluded", "age", "and", "civil", "condition"}


def title_tokens(s):
    s = CONTINUATION_PAT.sub("", (s or "").lower())
    return {w for w in re.findall(r"[a-z0-9]+", s) if len(w) >= 3 and w not in _STOP}


def title_sim(a, b):
    ta, tb = title_tokens(a), title_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def label_quality(rows):
    rls = [r for r in rows if r]
    if not rls:
        return 0.0
    return sum(1 for r in rls if AGE_RE.search(r)) / len(rls)


def main():
    reads = {r["table_id"]: r for r in json.load(open(os.path.join(V1, "page_reads_fixed.json")))}
    # title per table_id (for subject-coherent grouping)
    title = {}
    for r in csv.DictReader(open(os.path.join(V1, "provenance.csv"))):
        title[r["table_id"]] = r.get("title", "") or ""
    # ordered rows + family per table (preserve extraction order)
    rows = defaultdict(list)
    fam = {}
    for src in ("keeper_facts.csv", "unified_facts.csv"):
        p = os.path.join(V1, src)
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            t = r["table_id"]
            fam.setdefault(t, r.get("table_family", ""))
            rl = (r.get("row_label_std", "") or "").strip()
            seen = rows[t]
            if not seen or seen[-1] != rl:        # collapse consecutive dup-label cells into one row
                seen.append(rl)

    # group by (dochash, table_number), order by page, segment into contiguous runs
    by = defaultdict(list)
    for t, rd in reads.items():
        if fam.get(t) not in AGEFAM:
            continue
        pg = str(rd.get("page_number", ""))
        if not pg.isdigit():
            continue
        by[(rd["dochash"], (rd.get("table_number", "") or "?").strip().upper())].append((int(pg), t, rd))

    groups, verify = [], []
    gid = 0
    for (dh, tn), items in by.items():
        items.sort()
        # contiguous-page segments
        segs, cur = [], [items[0]]
        for it in items[1:]:
            if it[0] - cur[-1][0] <= SEG_SPLIT:
                cur.append(it)
            else:
                segs.append(cur); cur = [it]
        segs.append(cur)
        for seg in segs:
            if len(seg) < 2:
                continue
            gid += 1
            members = []
            for pg, t, rd in seg:
                rr = [x for x in rows.get(t, []) if x != ""]   # non-empty distinct rows
                members.append({"table_id": t, "page": pg, "n_rows": len(rr),
                                "label_quality": round(label_quality(rows.get(t, [])), 2),
                                "year": rd.get("year", ""), "region": rd.get("region", "")})
            # canonical = best-labeled member at the modal row count
            modal_n = Counter(m["n_rows"] for m in members).most_common(1)[0][0]
            best = max((m for m in members if m["n_rows"] == modal_n),
                       key=lambda m: m["label_quality"], default=members[0])
            canonical = [x for x in rows.get(best["table_id"], []) if x != ""]
            # members that need label propagation: same row count, weak labels
            propagated = [m["table_id"] for m in members
                          if m["table_id"] != best["table_id"] and m["n_rows"] == modal_n
                          and m["label_quality"] < 0.4 and best["label_quality"] >= 0.5]
            needs_ai = bool(propagated) or len({m["n_rows"] for m in members}) > 1
            g = {"gid": gid, "dochash": dh, "table_number": tn, "year": best["year"],
                 "region": best["region"], "family": fam.get(best["table_id"], ""),
                 "n_members": len(members), "page_range": f"{seg[0][0]}-{seg[-1][0]}",
                 "members": members, "canonical_rows": canonical[:40],
                 "canonical_from": best["table_id"], "best_quality": best["label_quality"],
                 "propagated": propagated, "needs_ai": needs_ai}
            groups.append(g)
            if needs_ai:
                verify.append(g)

    json.dump(groups, open(os.path.join(V1, "stitch_assembly.json"), "w"))
    json.dump(verify, open(os.path.join(V1, "stitch_verify_queue.json"), "w"))
    multi = [g for g in groups if g["n_members"] >= 2]
    print(f"logical multi-page AGE tables (>=2 contiguous pages): {len(multi)}")
    print(f"  total member pages: {sum(g['n_members'] for g in multi)}")
    print(f"  groups with label-propagation needed: {sum(1 for g in groups if g['propagated'])}")
    print(f"  member pages getting borrowed labels: {sum(len(g['propagated']) for g in groups)}")
    print(f"  groups flagged for AI review (propagation or row-count mismatch): {len(verify)}")


if __name__ == "__main__":
    main()
