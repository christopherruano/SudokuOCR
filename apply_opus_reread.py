#!/usr/bin/env python3
"""Apply Opus vision re-reads: validate each (U+M+W=Population, completeness),
convert accepted ones to tidy cell facts, and write a CORRECTIONS overlay
(corrected_facts.csv + corrected_tables.json). Originals are untouched — the
explorer prefers the correction for a table_id and drops its original facts.
Reads opus_reread_result.json (saved from the workflow)."""
import os, re, csv, json
import build_query_dataset as B

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
COLMAP = {"pop": "Population", "unm": "Unmarried", "mar": "Married", "wid": "Widowed"}


def parse_age(a):
    a = (a or "").strip()
    m = re.match(r"(\d+)\s*[-–]\s*(\d+)", a)
    if m:
        return a, int(m.group(1)), int(m.group(2)), 0
    m = re.match(r"(\d+)\s*(?:and over|\+|and upwards)", a, re.I)
    if m:
        return a, int(m.group(1)), "", 1
    return a, "", "", 0


def main():
    reads = json.load(open(os.path.join(V1, "opus_reread_result.json")))
    cands = {c["idx"]: c for c in json.load(open(os.path.join(V1, "opus_reread_candidates.json")))}
    # denorm + KIND per table (constraint depends on the table's nature, not assumed UMW)
    denorm, kind = {}, {}
    for r in csv.DictReader(open(os.path.join(V1, "keeper_catalog.csv"))):
        denorm[r["table_id"]] = {k: r.get(k, "") for k in B.DENORM if k in r}
        kind[r["table_id"]] = (r.get("measure_class", "count"), r.get("table_family", ""))

    def expected_base(reads_cells):
        """For a civil-condition table, infer the per-cell base: a COUNT sums U+M+W=Population;
        a DISTRIBUTION sums U+M+W to ~1000/10000. Detect from the data itself."""
        sums = [u + m + w for (p, u, m, w) in reads_cells if None not in (u, m, w)]
        if not sums:
            return None
        import statistics
        med = statistics.median(sums)
        for base in (1000, 10000, 100):
            if abs(med - base) <= 0.05 * base:
                return base       # distribution to this base
        return "population"        # counts -> compare to Population column

    out_facts, accepted, rejected = [], {}, []
    fid = 1
    for rd in reads:
        c = cands.get(rd.get("idx"))
        if not c or not rd.get("is_tableVII") or not rd.get("sections"):
            rejected.append((rd.get("idx"), "not_tableVII_or_empty")); continue
        tid = c["table_id"]
        mc, fam = kind.get(tid, ("count", ""))
        is_civil = fam in ("age_civil_condition", "age_sex_civil_condition")
        # gather cells + count rows
        cells, nrows, hasW = [], 0, 0
        for sec in rd["sections"]:
            for row in sec.get("rows", []):
                nrows += 1
                for sx in ("m", "f"):
                    P, U, M, W = (row.get(f"pop_{sx}"), row.get(f"unm_{sx}"), row.get(f"mar_{sx}"), row.get(f"wid_{sx}"))
                    if W is not None:
                        hasW += 1
                    cells.append((P, U, M, W))
        if nrows < 4:
            rejected.append((rd.get("idx"), f"too_few_rows {nrows}")); continue
        # KIND-AWARE constraint:
        if is_civil:
            if hasW == 0:
                rejected.append((rd.get("idx"), "civil table missing Widowed")); continue
            base = expected_base(cells)
            checked = bad = 0
            for (P, U, M, W) in cells:
                if None in (U, M, W):
                    continue
                s = U + M + W
                if base == "population":
                    if P is None:
                        continue
                    checked += 1; bad += abs(P - s) > max(2, 0.02 * max(1, P))
                elif isinstance(base, (int, float)):
                    checked += 1; bad += abs(s - base) > 0.03 * base       # distribution sums to the base
            if checked and bad > 0.1 * checked:
                rejected.append((rd.get("idx"), f"sum_fail({base}) {bad}/{checked}")); continue
            measure_unit = "" if base == "population" else f"per_{int(base)}" if isinstance(base, (int, float)) else ""
        else:
            # age x sex (no civil condition): require Population M/F, no UMW constraint
            mf = sum(1 for (P, U, M, W) in cells if P is not None)
            if mf < 4:
                rejected.append((rd.get("idx"), "age_sex: too few M/F values")); continue
            measure_unit = "" if mc == "count" else (mc or "")
        # convert to cell facts
        dn = denorm.get(tid, {})
        n0 = len(out_facts)
        for sec in rd["sections"]:
            relig = sec.get("name", "") if len(rd["sections"]) > 1 else ""
            for row in sec.get("rows", []):
                age, lo, hi, oe = parse_age(row.get("age"))
                for pfx, cg in COLMAP.items():
                    for sx, sexname in (("m", "Male"), ("f", "Female")):
                        v = row.get(f"{pfx}_{sx}")
                        if v is None:
                            continue
                        fr = {"fact_id": f"C{fid:09d}", "table_id": tid, "section": relig,
                              "row_label_std": age, "row_dimension": "age",
                              "column_group_std": cg, "column_dimension": "civil_condition",
                              "sub_column_std": sexname.lower() + "s", "sex": sexname,
                              "religion": relig, "resolved_age_label_std": age,
                              "resolved_age_lower": lo, "resolved_age_upper": hi,
                              "resolved_age_open_ended": "True" if oe else "", "measure_unit": measure_unit, "value": v}
                        fr.update(dn); fr["status"] = "corrected"; fr["age_aggregate"] = "1" if "total" in age.lower() else "0"
                        out_facts.append(fr); fid += 1
        accepted[tid] = len(out_facts) - n0

    with open(os.path.join(V1, "corrected_facts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=B.OUT_COLS, extrasaction="ignore")
        w.writeheader(); [w.writerow(r) for r in out_facts]
    json.dump({"corrected": list(accepted), "counts": accepted}, open(os.path.join(V1, "corrected_tables.json"), "w"))
    print(f"re-reads: {len(reads)}  accepted: {len(accepted)}  rejected: {len(rejected)}")
    print(f"corrected facts written: {len(out_facts):,}")
    if rejected[:5]:
        print("sample rejects:", rejected[:5])


if __name__ == "__main__":
    main()
