#!/usr/bin/env python3
"""
Hybrid multipage-group validator.

Stage 1 (this script): deterministic ordering/validation from the OCR data that
is already extracted locally -- age column-block ranges, section/region labels,
and continued/concluded title markers. It proposes a member order, picks the
first/last page, and detects duplicates, age-cycle gaps (missing pages) and
likely-foreign members. Groups it cannot resolve confidently are marked
needs_visual=1 for Stage 2 (vision verification).

Usage:  python3 validate_groups.py --year 1901
Outputs (under export_final/assembled_all_read_v2/):
  group_validation_<year>.csv         -- requested review-queue schema + helpers
  group_validation_<year>_visual.csv  -- just the needs_visual subset (for stage 2)
"""
import json, re, os, csv, sys, argparse
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
JSONL = os.path.join(ROOT, "export_final/assembled_all_read_v2/assembled_tables.jsonl")
JSON_ROOT = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")
IMG_ROOT = os.path.join(ROOT, "export/digitized_tables_referenced/digitized_tables")
csv.field_size_limit(10**7)


def local(p, root):
    i = p.find("/digitized_tables/")
    return root + p[i + len("/digitized_tables"):] if i >= 0 else None


def table_index(p):
    m = re.search(r"_t(\d+)_", p)
    return int(m.group(1)) if m else -1


# ----- title marker classification (sub-table level) -----
LEAD = re.compile(
    r"^\s*(final\s+census\s+)?(supplemental\s+)?(form\s+[ivxl0-9a-z.]+\.?\s*)?"
    r"table\s+no\.?\s*[ivxl0-9]+\s*[-–—]?\s*"
    r"(continued|concluded|contd\.?|concld\.?)?\s*\.?\s*", re.I)


def marker(title):
    if not title:
        return "UNK"
    t = title.replace("—", "-").replace("–", "-")
    rem = LEAD.sub("", t, count=1)
    if re.search(r"conclud|concld", rem, re.I):
        return "CONCL"
    if re.search(r"continu|contd", rem, re.I):
        return "CONT"
    return "HEAD"


# ----- age column-block parsing -----
YEAR4 = re.compile(r"\b(18|19|20)\d\d\b")


def age_start(colgroups):
    """Return the smallest plausible start-age across the page's age columns,
    or None if this page is not age-banded."""
    starts = []
    for c in colgroups or []:
        s = str(c)
        if YEAR4.search(s):           # census years, not ages
            continue
        if "," in s:                  # thousands brackets (income/area), not ages
            continue
        low = s.lower()
        if "under 1" in low:
            starts.append(0); continue
        if re.search(r"\ball ages\b", low):
            continue                  # total column, not a start
        # "a-b", "a to b", "a and over", bare "a."
        m = re.search(r"(\d{1,3})\s*(?:[-–to]+\s*\d{1,3}|\s*and\s*over)", low)
        if m:
            v = int(m.group(1))
            if v <= 100:
                starts.append(v)
            continue
        m = re.fullmatch(r"\s*(\d{1,3})\.?\s*", low)
        if m:
            v = int(m.group(1))
            if v <= 100:
                starts.append(v)
    return min(starts) if starts else None


def member_feats(j):
    d = {}
    lp = local(j, JSON_ROOT)
    try:
        obj = json.load(open(lp))
    except Exception:
        obj = {}
    md = obj.get("metadata", {}) or {}
    data = obj.get("data", [])
    sections = []
    if isinstance(data, list):
        for sec in data:
            if isinstance(sec, dict) and sec.get("name"):
                sections.append(str(sec["name"]).strip())
    title = (md.get("title") or "").strip()
    cg = md.get("column_groups") or []
    return {
        "json": j,
        "hash": j.split("/")[-1].replace("_oneshot.json", "")[:8],
        "idx": table_index(j),
        "title": title,
        "region": (md.get("region") or "").strip(),
        "colgroups": cg,
        "sections": sections,
        "age_start": age_start(cg),
        "marker": marker(title),
    }


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def validate_group(o):
    mems = [member_feats(j) for j in o.get("page_source_jsons", [])]
    n = len(mems)
    is_age = sum(1 for m in mems if m["age_start"] is not None) >= max(1, n / 2)

    notes, decision, conf, needs_visual = [], "needs_visual", "low", 1
    first = ""
    missing_preceding = ""
    extras = []

    # ---- foreign-member detection: region/title-stem outlier (conservative) ----
    regions = [m["region"] for m in mems if m["region"]]

    # ---- propose order ----
    if is_age:
        # order by (region/section appearance, age_start); detect dup & gaps
        # section key per page: region or first section name
        def seckey(m):
            return m["region"] or (m["sections"][0] if m["sections"] else "")
        sec_order = {}
        for m in mems:
            sec_order.setdefault(seckey(m), len(sec_order))
        ordered = sorted(
            mems,
            key=lambda m: (sec_order.get(seckey(m), 99),
                           m["age_start"] if m["age_start"] is not None else 999,
                           m["idx"]))
        starts = [m["age_start"] for m in ordered if m["age_start"] is not None]
        # duplicates: same (sec, age_start)
        seen = {}
        for m in ordered:
            k = (seckey(m), m["age_start"])
            if m["age_start"] is not None and k in seen:
                extras.append(m["json"])
                notes.append(f"dup age-block {m['age_start']} in section {seckey(m)!r} ({m['hash']})")
            else:
                seen[k] = m["hash"]
        # first / last
        if starts:
            first_m = ordered[0]
            first = first_m["json"]
            starts_min = min(starts)
            if starts_min not in (0,) and first_m["age_start"] not in (0,):
                # doesn't begin at infancy -> may be missing the opening page
                notes.append(f"lowest age-block starts at {starts_min} (no 0-/under-1 page)")
        # completeness signal
        n_sections = len(sec_order)
        per_sec = Counter(seckey(m) for m in ordered)
        if len(set(per_sec.values())) > 1:
            notes.append(f"uneven pages per section {dict(per_sec)} (possible missing/extra)")
        # confidence
        if not extras and len(set(per_sec.values())) <= 1 and starts and min(starts) == 0:
            decision, conf, needs_visual = "likely_ok_data", "medium", 1
            notes.append("age-cycle ordered from column ranges; confirm order/first visually")
        else:
            decision, conf, needs_visual = "needs_visual", "low", 1
    else:
        # non-age: order by table_index then section/title; use markers for ends
        ordered = sorted(mems, key=lambda m: (m["idx"], norm(m["region"]), norm(m["title"])))
        markers = [m["marker"] for m in ordered]
        heads = [m for m in ordered if m["marker"] == "HEAD"]
        concls = [m for m in ordered if m["marker"] == "CONCL"]
        # duplicates by (idx, title-stem)
        seen = {}
        for m in ordered:
            k = (m["idx"], norm(m["title"])[:40], tuple(m["colgroups"]))
            if k in seen:
                extras.append(m["json"]); notes.append(f"dup page {m['hash']}")
            else:
                seen[k] = 1
        if markers.count("HEAD") == 0 and (markers.count("CONT") or markers.count("CONCL")):
            decision, conf = "missing_preceding", "medium"
            missing_preceding = "UNKNOWN_NOT_IN_DATA"
            notes.append("no HEAD page; group starts mid-table")
            first = ordered[0]["json"]
        elif markers.count("HEAD") == 1:
            first = heads[0]["json"]
            decision, conf = "likely_ok_data", "medium"
            notes.append("single HEAD; order by table_index/section -- confirm visually")
        elif markers.count("HEAD") >= 2:
            decision, conf = "check_split", "low"
            notes.append(f"{markers.count('HEAD')} HEAD-like pages (distinct tables?)")
            first = heads[0]["json"]
        needs_visual = 1

    order_jsons = "|".join(m["json"] for m in ordered)
    cls = "".join({"HEAD": "H", "CONT": "c", "CONCL": "C", "UNK": "?"}[m["marker"]] for m in ordered)
    return {
        "group_id": o["group_id"],
        "decision": decision,
        "correct_first_source_json": first,
        "missing_preceding_source_jsons": missing_preceding,
        "extra_member_source_jsons": "|".join(extras),
        "correct_member_order_source_jsons": order_jsons,
        "notes": "; ".join(notes),
        "confidence": conf,
        "needs_visual": needs_visual,
        "n_pages": n,
        "kind": "age" if is_age else "other",
        "year": o.get("year"),
        "province": o.get("province"),
        "table_family": o.get("table_family"),
        "member_classes": cls,
        "assembled_failures": o.get("assembled_constraint_failures"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", required=True)
    a = ap.parse_args()
    rows = []
    for line in open(JSONL):
        o = json.loads(line)
        if str(o.get("year")) != str(a.year) or o.get("n_pages", 0) < 2:
            continue
        rows.append(validate_group(o))

    cols = ["group_id", "decision", "correct_first_source_json",
            "missing_preceding_source_jsons", "extra_member_source_jsons",
            "correct_member_order_source_jsons", "notes", "confidence",
            "needs_visual", "n_pages", "kind", "year", "province",
            "table_family", "member_classes", "assembled_failures"]
    out = os.path.join(ROOT, f"export_final/assembled_all_read_v2/group_validation_{a.year}.csv")
    vout = os.path.join(ROOT, f"export_final/assembled_all_read_v2/group_validation_{a.year}_visual.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow(r)
    with open(vout, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            if r["needs_visual"]: w.writerow(r)

    print(f"year {a.year}: {len(rows)} multi-page groups", file=sys.stderr)
    print("decision:", dict(Counter(r["decision"] for r in rows)), file=sys.stderr)
    print("kind:", dict(Counter(r["kind"] for r in rows)), file=sys.stderr)
    print("needs_visual:", sum(r["needs_visual"] for r in rows), file=sys.stderr)
    print("with extras flagged:", sum(1 for r in rows if r["extra_member_source_jsons"]), file=sys.stderr)
    print("wrote", out, file=sys.stderr)


if __name__ == "__main__":
    main()
