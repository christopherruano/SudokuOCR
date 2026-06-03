#!/usr/bin/env python3
"""
Stage 2: merge vision page-ordering metadata with the groups and emit the final
per-group review-queue CSV.

Inputs:
  - assembled_tables.jsonl (groups, with source_file = the PDF)
  - vbatch/result_*.json   (vision: hash -> {page_number,is_head,end_marker,section,header_text})

Logic per group (members ordered by printed page_number):
  - first page  = HEAD page if present, else lowest-numbered page
  - missing_preceding = first member is not a HEAD and its marker is "continued"
                        (the table's masthead page is not in this group)
  - last page   = CONCLUDED page if present, else highest-numbered page
  - duplicates  = two members sharing the same page_number
  - missing interior = gaps in the page-number run within the group's span
  - cross-ref: within the same source PDF, locate which OTHER group holds each
    missing/adjacent page number (helps decide merges); flag merge candidates.

Usage: python3 finalize_validation.py --year 1901
"""
import json, os, csv, glob, argparse, re
from collections import defaultdict, Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
JSONL = os.path.join(ROOT, "export_final/assembled_all_read_v2/assembled_tables.jsonl")
VBATCH = os.path.join(ROOT, "export_final/assembled_all_read_v2/vbatch")


def load_vision():
    V = {}
    v2 = os.path.join(ROOT, "export_final/assembled_all_read_v2/vbatch2")
    recheck = os.path.join(v2, "result_recheck.json")
    files = sorted(glob.glob(os.path.join(VBATCH, "result_*.json")))
    files += sorted(f for f in glob.glob(os.path.join(v2, "result_*.json")) if f != recheck)
    files += [recheck]  # recheck overrides win (loaded last)
    for f in files:
        if not os.path.exists(f):
            continue
        try:
            V.update(json.load(open(f)))
        except Exception as e:
            print("WARN bad result file", f, e)
    return V


def identity_of(v):
    """Prefer the vision agent's explicit table_identity; else parse header_text."""
    ti = (v.get("table_identity") or "").strip()
    if ti:
        m = TABLE_ID.search(ti) or TABLE_ID.search("TABLE " + ti)
        if m:
            return ("SUBTABLE " if re.search(r"sub", ti, re.I) else "TABLE ") + m.group(2).upper()
    return table_identity(v.get("header_text") or "")


def hash_of(j):
    return j.split("/")[-1].replace("_oneshot.json", "")


TABLE_ID = re.compile(r"(SUBSIDIARY\s+)?TABLE\s*(?:NO\.?\s*)?([IVXLC]+|\d+)\b", re.I)


def table_identity(hdr):
    """Identity of the table a head page belongs to. Prefer the printed
    'TABLE <n>' / 'SUBSIDIARY TABLE <n>' token; fall back to a normalized header
    prefix (marked HDR: so callers can treat it as lower-confidence)."""
    if not hdr:
        return "HDR:"
    m = TABLE_ID.search(hdr)
    if m:
        return ("SUBTABLE " if m.group(1) else "TABLE ") + m.group(2).upper()
    return "HDR:" + re.sub(r"[^A-Z]", "", hdr.upper())[:16]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--year", required=True)
    a = ap.parse_args()
    V = load_vision()

    # ALL groups this year that carry pages (multi + single) feed the page index;
    # only multi-page groups (n_pages>=2) get an output row.
    all_groups, out_groups = [], []
    for line in open(JSONL):
        o = json.loads(line)
        if str(o.get("year")) != str(a.year):
            continue
        if o.get("page_source_jsons"):
            all_groups.append(o)
        if o.get("n_pages", 0) >= 2:
            out_groups.append(o)

    # (source_file, page_no) -> list of (group_id, hash); hash -> source_json;
    # group_id -> dominant printed table identity; (source_file,page_no) -> is_head
    page_index = defaultdict(list)
    hash2json = {}
    group_tid = {}
    head_at = {}
    for o in all_groups:
        head_ids = []
        for j in o["page_source_jsons"]:
            h = hash_of(j); hash2json[h] = j
            v = V.get(h, {})
            pn = v.get("page_number")
            if pn is not None:
                page_index[(o["source_file"], pn)].append((o["group_id"], h))
                if v.get("is_head"):
                    head_at[(o["source_file"], pn)] = (o["group_id"], h, identity_of(v))
            if v.get("is_head"):
                tid = identity_of(v)
                if not tid.startswith("HDR:"):
                    head_ids.append(tid)
        group_tid[o["group_id"]] = Counter(head_ids).most_common(1)[0][0] if head_ids else None

    groups = out_groups
    rows = []
    for o in groups:
        src = o["source_file"]
        mems = []
        for j in o["page_source_jsons"]:
            v = V.get(hash_of(j), {})
            mems.append({
                "json": j, "hash": hash_of(j)[:8],
                "pn": v.get("page_number"),
                "head": bool(v.get("is_head")),
                "end": v.get("end_marker") or "none",
                "sec": v.get("section") or "",
                "hdr": v.get("header_text") or "",
                "tid": identity_of(v),
            })
        have_pn = [m for m in mems if m["pn"] is not None]
        no_pn = [m for m in mems if m["pn"] is None]
        ordered = sorted(have_pn, key=lambda m: m["pn"]) + no_pn

        # suspect page numbers: an in-group page >60 away from every other member
        # is almost always an OCR misread (e.g. "98" read as "981"). Drop from gap math.
        suspect_pn = []
        if len(have_pn) >= 3:
            allpn = sorted(m["pn"] for m in have_pn)
            for m in have_pn:
                if min(abs(m["pn"] - x) for x in allpn if x != m["pn"]) > 60:
                    suspect_pn.append(m)
        good_pn = [m for m in have_pn if m not in suspect_pn]

        notes, extras = [], []
        decision, conf = "ok", "high"
        first_json = ""
        missing_preceding = ""
        head_pages = ""
        merge_with = ""

        heads = [m for m in ordered if m["head"]]
        concl = [m for m in ordered if m["end"] == "concluded"]
        recovered_head = ""
        proposed_split = ""

        # duplicates (same page number)
        pn_count = Counter(m["pn"] for m in have_pn)
        dups = [pn for pn, c in pn_count.items() if c > 1]
        if dups:
            seen = set()
            for m in ordered:
                if m["pn"] in dups:
                    if m["pn"] in seen:
                        extras.append(m["json"])
                    seen.add(m["pn"])

        if suspect_pn:
            notes.append("suspect/misread page#: " + ", ".join(f"p{m['pn']}({m['hash']})" for m in suspect_pn))

        this_tid = group_tid.get(o["group_id"])
        # interior gaps within the span + where they live (suspect page#s excluded)
        missing_interior, merge_match, adj_other = [], [], []
        if len(good_pn) >= 2:
            pns = sorted(set(m["pn"] for m in good_pn))
            gaps = sorted(set(range(pns[0], pns[-1] + 1)) - set(pns))
            missing_interior = gaps
            located = []
            for gp in gaps:
                others = sorted(set(g for g, _ in page_index.get((src, gp), []) if g != o["group_id"]))
                if not others:
                    located.append(f"p{gp}->NOT_DIGITIZED"); continue
                tgt = others[0]
                tt = group_tid.get(tgt)
                same = not (this_tid and tt and this_tid != tt)  # compatible unless both known & different
                if same:
                    merge_match.append(tgt); located.append(f"p{gp}->MERGE:{tgt.split('::')[-1][:20]}")
                else:
                    adj_other.append(tgt); located.append(f"p{gp}->other-table({tt}):{tgt.split('::')[-1][:16]}")
            if gaps:
                notes.append("gaps: " + ", ".join(located))
        # pages this group shares with ANOTHER group (e.g. two scan-halves of one
        # printed page, or a page wrongly split across groups) -> merge signal
        shared_match = []
        for m in have_pn:
            for g, _h in page_index.get((src, m["pn"]), []):
                if g == o["group_id"]:
                    continue
                tt = group_tid.get(g)
                if not (this_tid and tt and this_tid != tt):  # compatible unless both known & different
                    shared_match.append(g)
        gap_in_other = merge_match + shared_match
        if shared_match:
            notes.append("shares page number(s) with: " +
                          ", ".join(sorted(set(g.split('::')[-1][:20] for g in shared_match))))

        # first page candidate (heads win; else lowest page; else first member)
        if heads:
            first_json = heads[0]["json"]
        elif have_pn:
            first_json = ordered[0]["json"]
        elif ordered:
            first_json = ordered[0]["json"]

        # ---- decision by severity ----
        # distinct table identities among head pages (real split vs reprinted masthead)
        head_ids = [(m, m["tid"]) for m in heads]
        distinct_ids = sorted(set(i for _, i in head_ids))
        distinct_table_ids = sorted(set(i for i in distinct_ids if not i.startswith("HDR:")))

        if not have_pn:
            all_ids = set(m["tid"] for m in mems if not m["tid"].startswith("HDR:"))
            if len(all_ids) == 1:
                decision, conf = "ok_unnumbered", "medium"
                notes.insert(0, f"no Arabic page numbers (roman folio/rotated); single table {next(iter(all_ids))} -> grouping ok, order unverified")
            elif len(all_ids) >= 2:
                decision, conf = "split_candidate", "high"
                head_pages = ""
                notes.insert(0, "no page numbers; multiple distinct table identities present -> SPLIT: " + ", ".join(sorted(all_ids)))
            else:
                decision, conf = "needs_manual", "low"
                notes.insert(0, "no legible page numbers and no table identity")
        elif len(distinct_table_ids) >= 2:
            decision, conf = "split_candidate", "high"
            head_pages = ",".join(str(m["pn"]) for m in heads)
            by_id = {}
            for m, i in head_ids:
                by_id.setdefault(i, []).append(m["pn"])
            notes.insert(0, "distinct tables merged -> SPLIT: " +
                         " || ".join(f"{i} @p{sorted(p, key=lambda x: (x is None, x))}" for i, p in by_id.items()))
            # build concrete sub-groups: each head starts a sub-table; assign each
            # member to the most recent preceding head (by page order).
            sub = {}
            cur = None
            for m in ordered:
                if m["head"]:
                    cur = m["tid"]
                key = cur or "UNASSIGNED_PRE_FIRST_HEAD"
                sub.setdefault(key, []).append(m["json"])
            proposed_split = " ;; ".join(f"{k}:: " + "|".join(v) for k, v in sub.items())
        elif len(distinct_ids) >= 2:
            # mix involves an unparseable header -> uncertain, needs eyes
            decision, conf = "check_split", "medium"
            head_pages = ",".join(str(m["pn"]) for m in heads)
            notes.insert(0, "possible distinct tables (header identity uncertain): " +
                         " || ".join(f"p{m['pn']}:{m['hdr'][:34]}" for m in heads))
        elif len(heads) >= 2:
            # several head pages but ALL the same table -> one table reprinting its
            # masthead; not a split. Fall through to gap logic below.
            notes.append(f"{len(heads)} pages reprint masthead of one table ({distinct_ids[0]})")
            if gap_in_other:
                decision, conf = "merge_candidate", "high"
                merge_with = "|".join(sorted(set(gap_in_other)))
                notes.insert(0, "continuation pages sit in other group(s) -> merge")
            elif missing_interior:
                decision, conf = "incomplete_not_digitized", "medium"
                notes.insert(0, f"{len(missing_interior)} interior page(s) not digitized")
            else:
                decision, conf = "ok", "high"
        elif not heads and ordered[0]["end"] == "continued":
            decision, conf = "missing_preceding", "high"
            # try to recover the head page: scan backwards for a head page in this PDF
            fp = ordered[0]["pn"]
            rec = None
            for k in range(1, 12):
                hit = head_at.get((src, fp - k))
                if hit:
                    rec = (fp - k, hit); break
                if (src, fp - k) in page_index:
                    continue  # a non-head digitized page sits between; keep scanning
            if rec:
                pn_h, (g_h, h_h, id_h) = rec
                recovered_head = hash2json.get(h_h, "")
                missing_preceding = recovered_head
                conf = "high"
                notes.insert(0, f"no HEAD in group; recovered head at p.{pn_h} ({id_h}) in group {g_h.split('::')[-1][:24]}")
            else:
                missing_preceding = "NOT_DIGITIZED_OR_NO_PAGE_NUMBER"
                notes.insert(0, f"no HEAD page; earliest is p.{fp} marked continued; head not found in corpus")
        elif dups:
            decision, conf = "has_duplicate", "high"
            notes.insert(0, f"duplicate page numbers {sorted(dups)}")
        elif gap_in_other:
            decision, conf = "merge_candidate", "high"
            merge_with = "|".join(sorted(set(gap_in_other)))
            notes.insert(0, "continuation pages sit in other group(s) -> merge")
        elif missing_interior:
            decision, conf = "incomplete_not_digitized", "medium"
            notes.insert(0, f"{len(missing_interior)} interior page(s) not digitized")
        else:
            decision, conf = "ok", "high"

        if concl and have_pn and concl[-1]["pn"] != max(m["pn"] for m in have_pn):
            notes.append("CONCLUDED page is not the highest-numbered (check)")
        if heads and have_pn and heads[0]["pn"] != min(m["pn"] for m in have_pn) and len(heads) == 1:
            notes.append("HEAD page is not the lowest-numbered (check)")

        order_jsons = "|".join(m["json"] for m in ordered)
        order_pretty = " | ".join(
            f"p{m['pn']}" + ("(head)" if m["head"] else "") + ("(concl)" if m["end"] == "concluded" else "")
            for m in ordered)
        rows.append({
            "group_id": o["group_id"],
            "decision": decision,
            "correct_first_source_json": first_json,
            "missing_preceding_source_jsons": missing_preceding,
            "extra_member_source_jsons": "|".join(extras),
            "correct_member_order_source_jsons": order_jsons,
            "notes": "; ".join(notes),
            "confidence": conf,
            "n_pages": o["n_pages"],
            "page_sequence": order_pretty,
            "missing_interior_page_numbers": ",".join(map(str, missing_interior)),
            "head_pages": head_pages,
            "merge_with_group_ids": merge_with,
            "recovered_head_source_json": recovered_head,
            "proposed_split_subgroups": proposed_split,
            "source_pdf": os.path.basename(src),
            "year": o.get("year"),
            "province": o.get("province"),
            "table_family": o.get("table_family"),
        })

    cols = ["group_id", "decision", "correct_first_source_json",
            "missing_preceding_source_jsons", "extra_member_source_jsons",
            "correct_member_order_source_jsons", "notes", "confidence",
            "n_pages", "page_sequence", "missing_interior_page_numbers",
            "head_pages", "merge_with_group_ids", "recovered_head_source_json",
            "proposed_split_subgroups",
            "source_pdf", "year", "province", "table_family"]
    out = os.path.join(ROOT, f"export_final/assembled_all_read_v2/group_review_{a.year}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow(r)

    import sys
    print(f"year {a.year}: {len(rows)} groups", file=sys.stderr)
    print("decision:", dict(Counter(r["decision"] for r in rows)), file=sys.stderr)
    print("confidence:", dict(Counter(r["confidence"] for r in rows)), file=sys.stderr)
    print("groups w/ extras:", sum(1 for r in rows if r["extra_member_source_jsons"]), file=sys.stderr)
    print("groups w/ interior gaps:", sum(1 for r in rows if r["missing_interior_page_numbers"]), file=sys.stderr)

    # ---- self-verification ----
    mp = [r for r in rows if r["decision"] == "missing_preceding"]
    rec = [r for r in mp if r["recovered_head_source_json"]]
    print(f"VERIFY missing_preceding: {len(mp)} total, {len(rec)} heads recovered in corpus", file=sys.stderr)
    # within each PDF, each digitized page number should map to ONE table identity.
    # Flag page numbers that two different proposed tables both claim (real conflicts).
    claim = defaultdict(set)
    for (src, pn), holders in page_index.items():
        for g, h in holders:
            claim[(os.path.basename(src), pn)].add(group_tid.get(g) or g.split("::")[-1][:18])
    conflicts = {k: v for k, v in claim.items() if len(v) > 1}
    print(f"VERIFY page-number conflicts (same p# claimed by >1 table id): {len(conflicts)}", file=sys.stderr)
    for k, v in list(conflicts.items())[:8]:
        print("   ", k, "->", sorted(v), file=sys.stderr)
    print("wrote", out, file=sys.stderr)


if __name__ == "__main__":
    main()
