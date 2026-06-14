#!/usr/bin/env python3
"""Build a PRECISE cluster re-OCR queue for the genuine_reocr Table VII tables only
(single scans that truly dropped a column / digit errors) — NOT the half-page or
transposed cases (those are fixed locally). Each image is paired with its expected
column schema (from sibling templates) so the cluster oneshot prompt is forced to
capture the full grid. Writes:
  tableVII_genuine_reocr.txt   – cluster image paths (one per line, run FIRST)
  tableVII_genuine_schema.json – imghash -> expected columns/age rows (schema hint)
"""
import os, re, csv, json, sqlite3
import census_explorer as ce

csv.field_size_limit(10 ** 7)
V1 = os.path.dirname(ce.DB)
HEX = re.compile(r"([0-9a-f]{40})")
CLUSTER = "/n/home11/cruano/census_toc/digitized_tables"


def main():
    cats = [r for r in csv.DictReader(open(os.path.join(V1, "imperfect_categorized.csv")))
            if r["category"] == "genuine_reocr"]
    templ = json.load(open(os.path.join(V1, "group_schema_templates.json"))) if \
        os.path.exists(os.path.join(V1, "group_schema_templates.json")) else {}
    con = sqlite3.connect(f"file:{ce.DB}?mode=ro", uri=True); con.row_factory = sqlite3.Row

    images, schema = [], {}
    seen = set()
    for r in cats:
        row = con.execute("SELECT source_image,member_images FROM tables WHERE table_id=?", (r["table_id"],)).fetchone()
        raw = ((row["member_images"] if row else "") or "") + ";" + ((row["source_image"] if row else "") or "")
        for path in re.split(r"[;,|]\s*", raw):
            path = path.strip()
            if not path.endswith(".png") or "digitized_tables" not in path:
                continue
            ih = HEX.search(path)
            if not ih or ih.group(1) in seen:
                continue
            seen.add(ih.group(1))
            # normalize to the cluster path
            tail = path.split("digitized_tables/", 1)[1]
            cpath = f"{CLUSTER}/{tail}"
            images.append(cpath)
            tm = templ.get(ih.group(1))
            if tm:
                schema[ih.group(1)] = {"columns": [c["name"] for c in tm["column_groups"]],
                                       "age_rows": tm.get("row_labels", [])[:30]}
            else:
                schema[ih.group(1)] = {"columns": ["Population", "Unmarried", "Married", "Widowed"], "age_rows": []}

    with open(os.path.join(V1, "tableVII_genuine_reocr.txt"), "w") as f:
        f.write("\n".join(images) + "\n")
    json.dump(schema, open(os.path.join(V1, "tableVII_genuine_schema.json"), "w"))
    print(f"genuine re-OCR queue: {len(images)} images ({len(cats)} tables)")
    print(f"  with sibling schema template: {sum(1 for v in schema.values() if v['age_rows'])}")
    print("wrote tableVII_genuine_reocr.txt + tableVII_genuine_schema.json")


if __name__ == "__main__":
    main()
