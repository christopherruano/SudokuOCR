#!/usr/bin/env python3
"""Neil #1: identify multi-page tables + per-PDF page order (deterministic).

Each table_id encodes its source document (16-hex) and its sequence in that
document (_tNNNN_) = reading/page order. Group by document, order by sequence,
and link continuation tables ('contd.'/'continued'/'concluded') to the head
table they continue. Writes sendout_v1/multipage_groups.json:
  { groups: { head_table_id: [ordered member table_ids...] },
    page_order: { table_id: {doc, seq, group, is_cont} } }
"""
import csv, re, os, json
from collections import defaultdict

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
SEQ = re.compile(r"([0-9a-f]{16})_t(\d+)_")
CONT = re.compile(r"contd|continued|concluded|concld|cont['.]|-\s*concld", re.I)


def norm_title(s):
    s = CONT.sub("", s or "").lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(s.split())[:48]


def main():
    cat = {r["table_id"]: r for r in csv.DictReader(open(os.path.join(V1, "catalog_tables.csv")))
           if r["status"] in ("confirmed_sent", "recovered")}
    docs = defaultdict(list)
    for t, r in cat.items():
        m = SEQ.search(t)
        if m:
            docs[m.group(1)].append((int(m.group(2)), t, r))

    groups, page_order = {}, {}
    for doc, items in docs.items():
        items.sort(key=lambda x: x[0])
        head = None
        head_norm = ""
        for seq, t, r in items:
            title = r.get("title") or ""
            is_cont = bool(CONT.search(title))
            nt = norm_title(title)
            # a continuation attaches to the open head if titles are compatible
            if head and is_cont and (nt == head_norm or nt in head_norm or head_norm in nt or not nt):
                groups[head].append(t)
                page_order[t] = {"doc": doc, "seq": seq, "group": head, "is_cont": True}
            else:
                head = t; head_norm = nt
                groups[head] = [t]
                page_order[t] = {"doc": doc, "seq": seq, "group": head, "is_cont": False}

    multi = {h: m for h, m in groups.items() if len(m) > 1}
    json.dump({"groups": groups, "page_order": page_order},
              open(os.path.join(V1, "multipage_groups.json"), "w"))
    n_member = sum(len(m) for m in multi.values())
    print(f"source documents: {len(docs)}")
    print(f"logical tables (groups): {len(groups)}")
    print(f"  multi-page groups: {len(multi)}  covering {n_member} member tables")
    print(f"  single-page tables: {len(groups) - len(multi)}")
    # show a few multi-page examples
    for h, m in list(sorted(multi.items(), key=lambda x: -len(x[1])))[:5]:
        print(f"   {len(m)} pages: {(cat[h].get('title') or '')[:50]} [{cat[h].get('year')}, {cat[h].get('canonical_name')}]")
    print("wrote multipage_groups.json")


if __name__ == "__main__":
    main()
