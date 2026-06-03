#!/usr/bin/env python3
"""
Triage analyzer for multipage table groups.

Reads the assembled groups + local OCR JSON and, for every group, classifies
each member page as HEAD / CONTINUED / CONCLUDED from its OCR'd title, then
emits a per-group decision CSV in the schema requested for the review queue:

  group_id,decision,correct_first_source_json,missing_preceding_source_jsons,
  extra_member_source_jsons,correct_member_order_source_jsons,notes

Plus triage helper columns: confidence, needs_visual, n_pages, year, province,
table_family, member_classes.

IMPORTANT: exact intra-group page ORDER and many first/last calls cannot be
determined from titles alone (titles are often wrapped-header fragments and the
stored order is hash-sorted). Those groups are marked needs_visual=1 so the
vision pass can resolve them. This script never fabricates an order it cannot
justify; it only encodes the title-marker partial order.
"""
import json, re, os, csv, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
JSONL = os.path.join(ROOT, "export_final/assembled_all_read_v2/assembled_tables.jsonl")
JSON_ROOT = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")
OUT = os.path.join(ROOT, "export_final/assembled_all_read_v2/group_order_triage.csv")

csv.field_size_limit(10**7)


def local_json(p):
    i = p.find("/digitized_tables/")
    return JSON_ROOT + p[i + len("/digitized_tables"):] if i >= 0 else None


def title_of(p):
    lp = local_json(p)
    if not lp or not os.path.exists(lp):
        return ""
    try:
        return (json.load(open(lp)).get("metadata", {}).get("title") or "").strip()
    except Exception:
        return ""


def table_index(p):
    m = re.search(r"_t(\d+)_", p)
    return int(m.group(1)) if m else -1


# strip a leading "FINAL CENSUS TABLE No. <num> [- continued/concluded]." prefix,
# then read the SUB-TABLE marker from what remains.
LEAD = re.compile(
    r"^\s*(final\s+census\s+)?(supplemental\s+)?(form\s+[ivxl0-9a-z.]+\.?\s*)?"
    r"table\s+no\.?\s*[ivxl0-9]+\s*[-–—]?\s*"
    r"(continued|concluded|contd\.?|concld\.?)?\s*\.?\s*",
    re.I,
)
CONCL = re.compile(r"conclud|concld", re.I)
CONT = re.compile(r"continu|contd", re.I)


def classify(title):
    if not title:
        return "UNK"
    t = title.replace("—", "-").replace("–", "-")
    rem = LEAD.sub("", t, count=1)
    if CONCL.search(rem):
        return "CONCL"
    if CONT.search(rem):
        return "CONT"
    return "HEAD"


def main():
    groups = []
    # index of (title-stem -> list of (class)) to detect a concluded page existing elsewhere
    for line in open(JSONL):
        o = json.loads(line)
        if o.get("n_pages", 0) < 2:
            continue
        mems = []
        for j in o.get("page_source_jsons", []):
            t = title_of(j)
            mems.append({"json": j, "title": t, "cls": classify(t), "idx": table_index(j)})
        groups.append({"o": o, "mems": mems})

    rows = []
    for g in groups:
        o = g["o"]; mems = g["mems"]
        classes = [m["cls"] for m in mems]
        cc = {c: classes.count(c) for c in ("HEAD", "CONT", "CONCL", "UNK")}
        idxs = sorted(set(m["idx"] for m in mems))
        idx_contig = (len(idxs) <= 1) or all(
            b - a <= 1 for a, b in zip(idxs, idxs[1:])
        )

        decision = "needs_visual"
        confidence = "low"
        needs_visual = 1
        first_json = ""
        missing_preceding = ""
        extra = ""
        notes = []

        heads = [m for m in mems if m["cls"] == "HEAD"]
        concls = [m for m in mems if m["cls"] == "CONCL"]

        # ---- confident clean: exactly one head, at least one concluded, single dir or contiguous
        if cc["UNK"] == 0 and cc["HEAD"] == 1 and cc["CONCL"] >= 1 and idx_contig:
            decision = "likely_ok"
            confidence = "medium"
            first_json = heads[0]["json"]
            if cc["CONCL"] > 1:
                notes.append(f"{cc['CONCL']} concluded pages (multiple sub-tables?)")
                confidence = "low"; needs_visual = 1
            else:
                needs_visual = 1  # order among middle pages still unverified
                notes.append("first/last from markers; intra-order needs visual")

        # ---- no head at all but has continued/concluded -> preceding page(s) missing
        elif cc["HEAD"] == 0 and (cc["CONT"] or cc["CONCL"]):
            decision = "missing_preceding"
            confidence = "medium"
            missing_preceding = "UNKNOWN_NOT_IN_DATA"
            notes.append("no HEAD page among members; group starts mid-table")

        # ---- multiple heads -> probably should be split / has extras
        elif cc["HEAD"] >= 2:
            decision = "check_split"
            notes.append(f"{cc['HEAD']} HEAD pages; possible distinct tables merged")

        # ---- everything else
        else:
            decision = "needs_visual"
            notes.append("ambiguous title markers")

        if cc["UNK"]:
            notes.append(f"{cc['UNK']} pages with empty/unreadable title")
        if not idx_contig:
            notes.append(f"non-contiguous table_index span {idxs}")

        # title-marker partial order: HEAD(s) -> CONT(s) -> UNK -> CONCL(s), stable within class
        order_rank = {"HEAD": 0, "CONT": 1, "UNK": 2, "CONCL": 3}
        ordered = sorted(mems, key=lambda m: (order_rank[m["cls"]], m["idx"]))
        order_jsons = "|".join(m["json"] for m in ordered)

        rows.append({
            "group_id": o["group_id"],
            "decision": decision,
            "correct_first_source_json": first_json,
            "missing_preceding_source_jsons": missing_preceding,
            "extra_member_source_jsons": extra,
            "correct_member_order_source_jsons": order_jsons,
            "notes": "; ".join(notes),
            "confidence": confidence,
            "needs_visual": needs_visual,
            "n_pages": o["n_pages"],
            "year": o.get("year"),
            "province": o.get("province"),
            "table_family": o.get("table_family"),
            "member_classes": "".join(
                {"HEAD": "H", "CONT": "c", "CONCL": "C", "UNK": "?"}[m["cls"]] for m in ordered
            ),
        })

    cols = ["group_id", "decision", "correct_first_source_json",
            "missing_preceding_source_jsons", "extra_member_source_jsons",
            "correct_member_order_source_jsons", "notes", "confidence",
            "needs_visual", "n_pages", "year", "province", "table_family",
            "member_classes"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # summary to stderr
    from collections import Counter
    print(f"multi-page groups analyzed: {len(rows)}", file=sys.stderr)
    print("decision counts:", dict(Counter(r["decision"] for r in rows)), file=sys.stderr)
    print("needs_visual:", sum(r["needs_visual"] for r in rows), file=sys.stderr)
    print("class-pattern top:", Counter(r["member_classes"] for r in rows).most_common(12),
          file=sys.stderr)
    print("wrote", OUT, file=sys.stderr)


if __name__ == "__main__":
    main()
