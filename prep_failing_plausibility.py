#!/usr/bin/env python3
"""Generate inputs for the existing llm_plausibility_review_app.py, scoped to the
FAILING Table VII, so they can be reviewed/marked in the tool you already use:
  - failing_plausibility_results.csv  (the review rows: image + excel + the exact problem)
  - failing_packets.jsonl             (audit detail per table)
  - per-table Excel workbooks under sendout_final/review/failing_tableVII/
Each row's flags/reason/review_questions state precisely what's wrong (which cells
fail U+M+W=Population, which civil column is missing), so marking is fast.
Launch:  python3 llm_plausibility_review_app.py \
           --results-csv export_final/final_validated/llm_plausibility_review/failing/failing_plausibility_results.csv \
           --packets-jsonl .../failing/failing_packets.jsonl \
           --decisions-csv .../failing/failing_decisions.csv --port 5002"""
import os, re, csv, json, sqlite3, hashlib
import census_explorer as ce

csv.field_size_limit(10 ** 7)
V1 = os.path.dirname(ce.DB)
SENDOUT = "export_final/sendout_final"
OUTDIR = "export_final/final_validated/llm_plausibility_review/failing"
XLSX_REL = "review/failing_tableVII"            # under SENDOUT
XLSX_ABS = os.path.join(SENDOUT, XLSX_REL)
CIV = {"U": re.compile(r"unmarr|single|never", re.I), "M": re.compile(r"marri", re.I),
       "W": re.compile(r"widow", re.I), "P": re.compile(r"popul|persons|total", re.I)}
AGE = re.compile(r"\d+\s*[-–]\s*\d+|\d+\s*and over|under|all ages|total", re.I)

RES_COLS = ["table_id", "status", "llm_decision", "confidence", "flags", "reason",
            "suggested_next_action", "review_questions", "model", "provider", "key_slot",
            "priority_score", "priority_reasons", "source_image_local", "excel_path",
            "province", "year", "table_family", "measure_category", "format_class"]


def civ(s):
    s = s or ""
    for k in ("U", "W", "M", "P"):
        if CIV[k].search(s):
            return k
    return ""


def num(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def failing_detail(facts):
    coord = {}
    cols = set()
    for f in facts:
        kc = civ(f["column_group_std"]) or civ(f["sub_column_std"]) or civ(f["row_label_std"])
        if kc:
            cols.add(kc)
        am = AGE.search(f["row_label_std"] or "") or AGE.search(f["column_group_std"] or "")
        age = re.sub(r"\s+", "", am.group(0)).lower() if am else ""
        sex = (f["sex"] or f["sub_column_std"] or "").lower()[:1]
        v = num(f["value"])
        if kc and age and v is not None:
            coord.setdefault((age, sex), {})[kc] = v
    bad = []
    for (age, sex), d in coord.items():
        if all(k in d for k in ("U", "M", "W", "P")):
            s = d["U"] + d["M"] + d["W"]
            if abs(d["P"] - s) > max(2, 0.02 * max(1, d["P"])):
                bad.append(f"{age}/{sex}: U+M+W={int(s)} vs Pop={int(d['P'])}")
    missing = {"U": "Unmarried", "M": "Married", "W": "Widowed"}
    miss = [missing[k] for k in ("U", "M", "W") if k not in cols]
    return bad, miss


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(XLSX_ABS, exist_ok=True)
    led = [r for r in csv.DictReader(open(os.path.join(V1, "tableVII_verdicts.csv")))
           if r["status"] in ("unrecoverable", "retrying")]

    def order(r):
        rr = r["reason"]
        return 0 if r["status"] == "retrying" else (3 if "sibling" in rr else 4 if "incomplete" in rr else 1)
    led.sort(key=order)

    con = sqlite3.connect(f"file:{ce.DB}?mode=ro", uri=True); con.row_factory = sqlite3.Row
    results, packets = [], []
    for r in led:
        tid = r["table_id"]
        t = con.execute("SELECT * FROM tables WHERE table_id=?", (tid,)).fetchone()
        if not t:
            continue
        facts = con.execute("SELECT * FROM facts WHERE table_id=? ORDER BY rowid", (tid,)).fetchall()
        td = dict(t)
        # local scan (repo-relative)
        raw = (t["member_images"] or t["source_image"] or "")
        img_local = ""
        for p in re.split(r"[;,|]", raw):
            loc = ce.resolve_img(p.strip()) if p.strip() else None
            if loc:
                img_local = os.path.relpath(loc); break
        # per-table excel
        h = hashlib.md5(tid.encode()).hexdigest()[:12]
        mats = ce.build_matrices(facts)
        xlsx_name = f"{ce.safe_filename(td)[:50]}_{h}.xlsx"
        with open(os.path.join(XLSX_ABS, xlsx_name), "wb") as fh:
            fh.write(ce.build_table_xlsx(td, mats))
        excel_path = f"{XLSX_REL}/{xlsx_name}"
        # the exact problem
        bad, miss = failing_detail(facts)
        flags = ";".join(([f"missing:{','.join(miss)}"] if miss else []) +
                         ([f"umw_sum_fail:{len(bad)}"] if bad else []) + ([r["last_issues"]] if r["last_issues"] else []))
        reason = (f"verdict={r['status']} ({r['reason'][:70]}). " +
                  (f"Missing civil column(s): {', '.join(miss)}. " if miss else "") +
                  (f"{len(bad)} cells fail U+M+W=Population, e.g. " + "; ".join(bad[:4]) if bad else ""))
        qs = []
        if miss: qs.append(f"Is the {', '.join(miss)} column actually present on the scan? If so, read it.")
        if bad: qs.append("For the flagged cells, which digit is misread so U+M+W=Population?")
        qs.append("Mark CORRECT if the read matches the scan, else note the fix.")
        results.append({"table_id": tid, "status": "needs_review", "llm_decision": "",
                        "confidence": "", "flags": flags, "reason": reason,
                        "suggested_next_action": "human_fix", "review_questions": " | ".join(qs),
                        "model": "", "provider": "", "key_slot": "",
                        "priority_score": (100 if r["status"] == "retrying" else 60),
                        "priority_reasons": r["reason"][:50],
                        "source_image_local": img_local, "excel_path": excel_path,
                        "province": td.get("canonical_name", ""), "year": td.get("year", ""),
                        "table_family": td.get("table_family", ""), "measure_category": td.get("measure_category", ""),
                        "format_class": ""})
        packets.append({"table_id": tid,
                        "metadata": {"province": td.get("canonical_name", ""), "year": str(td.get("year", "")),
                                     "title": td.get("title", ""), "table_family": td.get("table_family", "")},
                        "deterministic_audit": {"plausibility_status": "fail", "issue_flags": flags,
                                                "blocking_reasons": reason},
                        "review_priority": {"score": (100 if r["status"] == "retrying" else 60),
                                            "reasons": [r["reason"][:50]]},
                        "recommended_checks": qs,
                        "summary": {"fact_rows": len(facts), "umw_fail_cells": len(bad),
                                    "missing_columns": miss}})
    with open(os.path.join(OUTDIR, "failing_plausibility_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RES_COLS, extrasaction="ignore")
        w.writeheader(); [w.writerow(x) for x in results]
    with open(os.path.join(OUTDIR, "failing_packets.jsonl"), "w") as f:
        for p in packets:
            f.write(json.dumps(p) + "\n")
    print(f"failing-review inputs: {len(results)} tables")
    print(f"  -> {OUTDIR}/failing_plausibility_results.csv + failing_packets.jsonl")
    print(f"  -> {len(results)} per-table Excel in {XLSX_ABS}/")


if __name__ == "__main__":
    main()
