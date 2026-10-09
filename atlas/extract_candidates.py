#!/usr/bin/env python3
"""Collect every string in the corpus that might name a place, with context.

Sources (all facts/tables in census_explorer.sqlite):
  row   row labels of tables laid out place x (age|civil condition|sex)
  colg  column groups of age-row tables whose column axis is not a census dimension
  sect  section headers of age-row tables
  title 'X DISTRICT'/'X STATE'... phrases in table titles
  tag   the table's own canonical_name
Output atlas/out/place_strings.csv — one row per normalized string.
"""
import sqlite3, re, unicodedata, csv, collections, os
HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "..", "export_final/sendout_final/sendout_v1/census_explorer.sqlite")

JUNK = re.compile(r"(?i)^(row|col|column|section|part|total|sub-?total|grand|all\b|above|under|up to|over|\d|ditto|"
    r"males?\b|females?\b|persons|number|age|years?\b|hindu|muham|mahomed|musalman|muslim|christian|jain|parsi|sikh|budd|"
    r"animist|tribal|jew|zoroas|others?\b|europe|eurasian|anglo|landowner|agricultur|non-|married|unmarried|widow|single|"
    r"divorced|urban|rural|literate|illiterate|insane|blind|deaf|leper|natural population|actual population|"
    r"born|immigrant|emigrant|caste|religion|occupation|language|$)|ditto|,\s*$")
TITLE_PL = re.compile(r"([A-Z][A-Za-zÀ-ɏ'.\- ]{2,40}?)\s+(DISTRICT|District|STATE|State|DIVISION|Division|"
                      r"AGENCY|Agency|TALUK|Taluk|TALUKA|Taluka|TAHSIL|Tahsil|ZILA|PRANT|Prant|PARGANA|Pargana)\b")

def norm(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"\b(district|districts|dist|zila|zilla|the)\b", " ", s)
    return re.sub(r"[^a-z]", "", s)

def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    meta = {r[0]: r[1:] for r in con.execute(
        "select table_id, broad_region, canonical_name, parent_geo, year, title from tables")}
    agg = collections.defaultdict(lambda: {"raw": collections.Counter(), "src": collections.Counter(),
                                           "tables": set(), "ctx": collections.Counter()})
    def add(s, src, tid):
        s = (s or "").strip()
        if not s or JUNK.search(s) or len(norm(s)) < 3:
            return
        a = agg[norm(s)]
        a["raw"][s] += 1; a["src"][src] += 1; a["tables"].add(tid)
        m = meta.get(tid)
        if m:
            a["ctx"][f"{m[0] or '?'} | {m[1] or '?'} | {m[3] or '?'} | {(m[4] or '')[:70]}"] += 1
    q = {
      "row": "select distinct table_id,row_label_std from facts where row_dimension='other' and column_dimension in ('age','civil_condition','sex')",
      "colg": "select distinct table_id,column_group_std from facts where row_dimension='age' and column_dimension='other'",
      "sect": "select distinct table_id,section from facts where row_dimension='age' and section!=''",
    }
    for src, sql in q.items():
        for tid, s in con.execute(sql):
            add(s, src, tid)
    for tid, m in meta.items():
        add(m[1], "tag", tid)
        for g in TITLE_PL.finditer(m[4] or ""):
            add(g.group(1), "title", tid)
    out = os.path.join(HERE, "out", "place_strings.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "label", "variants", "n_tables", "sources", "context"])
        for k, a in sorted(agg.items(), key=lambda kv: -len(kv[1]["tables"])):
            w.writerow([k, a["raw"].most_common(1)[0][0],
                        " | ".join(r for r, _ in a["raw"].most_common(4)[1:]),
                        len(a["tables"]), ";".join(f"{s}:{n}" for s, n in a["src"].most_common()),
                        " || ".join(c for c, _ in a["ctx"].most_common(3))])
    print(f"{len(agg)} candidate place strings -> {out}")

if __name__ == "__main__":
    main()
