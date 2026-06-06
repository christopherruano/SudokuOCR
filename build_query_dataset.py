#!/usr/bin/env python3
"""Build the unified, query-ready census dataset (the "map" + the facts).

Sources
  confirmed-sendable facts : sendout_v1/facts_sendout.csv          (status=confirmed)
  recovered facts          : local cell_facts_rich.csv (tail+core)  (status=recovered)
                             for tables triaged cleared_false_failure / cleared_in_scope_now
  table metadata           : all_tables_full_metadata.csv  (year, geo_name, canonical_topic, ...)
  geography canon          : sendout_v1/geo_reference.csv

Cleaning applied to recovered facts (parity with the confirmed cleaning):
  drop a persons/All cell when, at the same (section,row,column_group), BOTH a
  males and a females cell exist AND persons == males + females  (provably
  synthesized by the old prompt; never strip a genuinely-printed Persons column).

Outputs (sendout_v1/)
  unified_facts.csv   one tidy row per numeric cell, denormalized with
                      year / geography (canonical) / topic / measure / dims / provenance / status
  catalog_tables.csv  one row per table = THE MAP: topic, geography, year, measure,
                      dims-present, n_facts, status, source_image, title  (covers ALL tables)
"""
import csv, os, glob
from collections import defaultdict, Counter

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
SF = os.path.join(ROOT, "export_final/sendout_final")
V1 = os.path.join(SF, "sendout_v1")
META = os.path.join(ROOT, "export_final/team_ready/catalogs/all_tables_full_metadata.csv")
RICH = [
    os.path.join(ROOT, "export/final_project_exports/tail_exploratory_v2/cell_facts_rich.csv"),
    os.path.join(ROOT, "export/final_project_exports/core_curated_v2_final/cell_facts_rich.csv"),
]

DENORM = ["year", "decade", "canonical_name", "entity_type", "parent_geo", "broad_region",
          "geo_level", "canonical_topic", "subject", "measure_class", "additive",
          "table_family", "measure_category", "source_image", "title", "status"]

import re as _re
_VAR = _re.compile(r"variation|increase|decrease|growth|\bchange\b|relative growth", _re.I)
_PCT = _re.compile(r"percentage|per cent|per-cent|\bp\.c\.", _re.I)
_DIST = _re.compile(r"distribution of (1|10)[, ]?000|per (1|10)[, ]?000 (of |persons|by)|out of (1|10)[, ]?000|per mille", _re.I)
_RATE = _re.compile(r"\brate\b|per 1[, ]?000|per 10[0,]00|per mille|per thousand", _re.I)
_RATIO = _re.compile(r"\bratio\b|proportion of (the )?sexes|per 1[, ]?000 males|females per", _re.I)
_MORT = _re.compile(r"\bdeath|mortalit|deaths registered|born alive|birth[- ]?rate|still[- ]?birth|infant", _re.I)
_LIT = _re.compile(r"literac|illiterac|\beducation|instruction|reading|writing|literate", _re.I)


def classify_subject_measure(meta, dom_unit, title, max_val=None):
    """subject = what is counted; measure_class = how expressed; additive = safe to sum.
    Title-first, then dominant unit, with a magnitude gate (a 'rate'/'percentage'/'distribution'
    whose values are huge is really a count that was mis-tagged upstream)."""
    tf = (meta.get("table_family") or "").lower()
    t = title or ""
    # subject — pull mortality/literacy OUT of population (they polluted population queries)
    if _MORT.search(t):
        subject = "mortality"
    elif _LIT.search(t):
        subject = "literacy"
    elif "religion" in tf:
        subject = "religion"
    elif "civil_condition" in tf or "marital" in tf:
        subject = "civil_condition"
    else:
        subject = "population"
    # measure_class — most specific first
    if _VAR.search(t):
        measure = "variation"
    elif dom_unit == "percentage" or _PCT.search(t):
        measure = "percentage"
    elif _DIST.search(t):
        measure = "distribution"
    elif dom_unit in ("rate", "per_10000", "per_1000") or _RATE.search(t):
        measure = "rate"
    elif _RATIO.search(t):
        measure = "ratio"
    else:
        measure = "count"
    # magnitude gate: a proportional measure whose values exceed its natural ceiling is really a
    # mis-tagged count (per-class: % <=~100, ratio/distribution per-1k/10k, rate up to per-100k).
    GATE = {"percentage": 1000, "ratio": 50000, "distribution": 50000, "rate": 200000}
    if measure in GATE and max_val is not None and max_val > GATE[measure]:
        measure = "count"
    return subject, measure, ("1" if measure == "count" else "0")
KEEP = ["fact_id", "table_id", "section", "row_label_std", "row_dimension", "row_group_value",
        "column_group_std", "column_dimension", "column_group_value", "sub_column_std",
        "sex", "civil_condition", "religion", "resolved_age_label_std", "resolved_age_lower",
        "resolved_age_upper", "resolved_age_open_ended", "measure_unit", "value"]
OUT_COLS = KEEP + DENORM + ["age_aggregate"]
VALID_SEX = {"Male", "Female", "All", ""}


def _agg_flag(f):
    lab = (f.get("resolved_age_label_std") or "").lower()
    tot = str(f.get("resolved_age_total") or "")
    return "1" if ("total" in lab or "all age" in lab or tot in ("True", "1")) else "0"


def load_geo():
    g = {}
    for r in csv.DictReader(open(os.path.join(V1, "geo_reference.csv"))):
        g[r["raw_geo_name"]] = r
    return g


def load_meta():
    m = {}
    for r in csv.DictReader(open(META)):
        m[r["table_id"]] = r
    return m


_DOC = _re.compile(r"([0-9a-f]{16})_t\d+_")
_GENERIC_PLACE = {
    "", "/", "∅", "province", "provincial", "provincial total", "provincial summary",
    "provincial total of", "town", "country", "unknown", "state", "district", "city",
    "the state", "whole province", "state summary", "state total", "total", "native states",
    "province (districts)", "british territory", "british districts", "country.", "provinces",
    "hilly", "provincial and urban", "nat. states", "subsidiary", "the totals", "general/sample",
}
_PROVINCE_TYPES = {"british_province", "princely_state", "modern_state"}
_VAGUE_TYPES = {"unresolved", "unspecified", "", "not_geography", "native_states_agg"}


def _dochash(tid):
    m = _DOC.search(tid or "")
    return m.group(1) if m else ""


def resolve_geographies(meta, geo):
    """Every table is from one provincial census volume, so tables sharing a source
    document share a province. Resolve generic/unresolved geography by propagating the
    document's 'home' province (modal province-level sibling) down to its vague tables."""
    def base(tid):
        gn = meta.get(tid, {}).get("geo_name", "")
        g = geo.get(gn, {})
        return {"canonical_name": g.get("canonical_name") or gn, "entity_type": g.get("entity_type", ""),
                "parent_geo": g.get("parent", ""), "broad_region": g.get("broad_region", ""),
                "geo_level": meta.get(tid, {}).get("geo_level", ""), "geo_source": "direct"}

    docs = defaultdict(list)
    for tid in meta:
        docs[_dochash(tid)].append(tid)
    home = {}
    for d, tids in docs.items():
        if not d:
            continue
        prov, rep = Counter(), {}
        for tid in tids:
            b = base(tid)
            if b["entity_type"] in _PROVINCE_TYPES and (b["canonical_name"] or "").lower() not in _GENERIC_PLACE:
                prov[b["canonical_name"]] += 1; rep[b["canonical_name"]] = b
        if prov:
            home[d] = rep[prov.most_common(1)[0][0]]
        else:  # fallback 1: modal parent province of district siblings
            par = Counter(base(t)["parent_geo"] for t in tids if base(t)["parent_geo"]
                          and base(t)["parent_geo"] not in ("British India", "Native States"))
            if par:
                hp = par.most_common(1)[0][0]
                reg = next((base(t)["broad_region"] for t in tids if base(t)["broad_region"]), "")
                home[d] = {"canonical_name": hp, "entity_type": "british_province", "parent_geo": "British India",
                           "broad_region": reg, "geo_level": "province", "geo_source": "direct"}
            else:  # fallback 2: an all-India volume (only India-level siblings, e.g. 1881 imperial tables)
                ai = next((base(t) for t in tids if base(t)["entity_type"] == "all_india"), None)
                if ai:
                    home[d] = ai

    out = {}
    for tid in meta:
        b = base(tid)
        h = home.get(_dochash(tid))
        generic = (b["canonical_name"] or "").lower() in _GENERIC_PLACE
        vague = b["entity_type"] in _VAGUE_TYPES
        if h and (generic or vague):
            out[tid] = {**h, "geo_source": "doc_propagated"}          # replace fully with volume's province
        elif h and not b["broad_region"]:
            out[tid] = {**b, "broad_region": h["broad_region"],
                        "parent_geo": b["parent_geo"] or h["canonical_name"], "geo_source": "region_propagated"}
        else:
            out[tid] = b
    return out


def status_map():
    """table_id -> pipeline status for the catalog."""
    st = {}
    for r in csv.DictReader(open(os.path.join(V1, "tables_status.csv"))):
        st[r["table_id"]] = "confirmed_sent" if r["status"].startswith("SEND") else "hold_reocr"
    for fn, lab in (("cleared_false_failure.csv", "recovered"),
                    ("cleared_in_scope_now.csv", "recovered"),
                    ("genuine_rerun.csv", "needs_reocr"),
                    ("keep_review.csv", "needs_human_review")):
        p = os.path.join(SF, "review/rerun_retriage", fn)
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                st.setdefault(r["table_id"], lab)
    return st


def clean_synth_persons(facts):
    """Drop provably-synthesized persons cells (persons == males+females at coord)."""
    coord = defaultdict(dict)  # (sec,row,grp) -> {sub: (idx, value)}
    for i, f in enumerate(facts):
        sub = f.get("sub_column_std") or {"All": "persons", "Male": "males", "Female": "females"}.get(f.get("sex", ""), "")
        if sub in ("persons", "males", "females"):
            try:
                v = float(f["value"]) if f["value"] not in ("", None) else None
            except ValueError:
                v = None
            coord[(f.get("section", ""), f.get("row_label_std", ""), f.get("column_group_std", ""))][sub] = (i, v)
    drop = set()
    for cols in coord.values():
        if {"persons", "males", "females"} <= set(cols):
            pi, pv = cols["persons"]; _, mv = cols["males"]; _, fv = cols["females"]
            if pv is not None and mv is not None and fv is not None and abs(pv - (mv + fv)) < 1e-6:
                drop.add(pi)
    return [f for i, f in enumerate(facts) if i not in drop], len(drop)


def main():
    geo = load_geo()
    meta = load_meta()
    geores = resolve_geographies(meta, geo)   # doc-propagated geography (every table tied to its volume's province)
    status = status_map()
    ovp = os.path.join(V1, "manual_overrides.json")
    ov = __import__("json").load(open(ovp)) if os.path.exists(ovp) else {"title": {}, "note": {}, "status": {}}

    recoverable = set()
    for fn in ("cleared_false_failure.csv", "cleared_in_scope_now.csv"):
        for r in csv.DictReader(open(os.path.join(SF, "review/rerun_retriage", fn))):
            recoverable.add(r["table_id"])
    # Phase-1 deterministic recovery: tables that passed re-validation under type-aware constraints
    p1 = os.path.join(V1, "recovered_phase1.json")
    phase1 = set(__import__("json").load(open(p1))["recovered"]) if os.path.exists(p1) else set()
    recoverable |= phase1
    rp = os.path.join(V1, "relabel_overrides.json")
    relabel_ov = __import__("json").load(open(rp)) if os.path.exists(rp) else {}

    # 1) confirmed sendable facts (already cleaned)
    by_tbl = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(V1, "facts_sendout.csv"))):
        by_tbl[r["table_id"]].append(r)
    confirmed_tables = set(by_tbl)

    # 2) recovered facts from rich files
    rec_raw = defaultdict(list)
    for path in RICH:
        if not os.path.exists(path):
            continue
        for r in csv.DictReader(open(path)):
            t = r.get("table_id", "")
            if t in recoverable and t not in confirmed_tables:
                rec_raw[t].append(r)
    dropped = 0
    for t, facts in rec_raw.items():
        cleaned, d = clean_synth_persons(facts)
        by_tbl[t] = cleaned
        dropped += d
    recovered_tables = set(rec_raw)

    # dominant measure_unit + max abs value per table (drive measure_class + magnitude gate)
    dom, maxv = {}, {}
    for t, facts in by_tbl.items():
        c = Counter(f.get("measure_unit", "") for f in facts)
        dom[t] = c.most_common(1)[0][0] if c else ""
        mx = 0.0
        for f in facts:
            try:
                mx = max(mx, abs(float(f["value"])))
            except (ValueError, TypeError):
                pass
        maxv[t] = mx

    # 3) emit unified facts
    def denorm(t):
        m = meta.get(t, {})
        gx = geores.get(t, {})
        yr = m.get("year", "")
        dec = (yr[:3] + "0s") if yr[:3].isdigit() else ""
        stt = ov["status"].get(t) or ("recovered" if t in phase1 else None) \
            or status.get(t, "confirmed_sent" if t in confirmed_tables else "recovered")
        ttl = ov["title"].get(t) or m.get("title", "")
        subject, measure_class, additive = classify_subject_measure(m, dom.get(t, ""), ttl, maxv.get(t))
        return {
            "subject": subject, "measure_class": measure_class, "additive": additive,
            "year": yr, "decade": dec,
            "canonical_name": gx.get("canonical_name", ""),
            "entity_type": gx.get("entity_type", ""), "parent_geo": gx.get("parent_geo", ""),
            "broad_region": gx.get("broad_region", ""), "geo_level": gx.get("geo_level", ""),
            "canonical_topic": m.get("canonical_topic", ""), "table_family": m.get("table_family", ""),
            "measure_category": m.get("measure_category", ""),
            "source_image": m.get("source_image", ""),
            "title": ov["title"].get(t) or m.get("title", ""), "status": stt,
        }

    fid = 1
    n_conf = n_rec = 0
    with open(os.path.join(V1, "unified_facts.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_COLS, extrasaction="ignore")
        w.writeheader()
        for t in sorted(by_tbl):
            d = denorm(t)
            rmap = relabel_ov.get(t)
            for f in by_tbl[t]:
                row = {k: f.get(k, "") for k in KEEP}
                row["fact_id"] = f"F{fid:09d}"; fid += 1
                row.update(d)
                row["age_aggregate"] = _agg_flag(f)
                # vision relabel (M/F -> Unmarried/Married/Widowed etc.), complete tables only
                if rmap:
                    cur = row.get("sub_column_std") or {"All": "persons", "Male": "males", "Female": "females"}.get(row.get("sex", ""), "")
                    if cur in rmap:
                        row["sub_column_std"] = rmap[cur]; row["sex"] = ""
                # sanitize leaked fields (field-shift artifacts flagged in QA)
                if row.get("sex") not in VALID_SEX:
                    row["sex"] = ""
                if (row.get("civil_condition") or "").replace(".", "").isdigit():
                    row["civil_condition"] = ""
                w.writerow(row)
            if t in recovered_tables:
                n_rec += len(by_tbl[t])
            else:
                n_conf += len(by_tbl[t])

    # 4) catalog = THE MAP (every table in the corpus)
    fcount = {t: len(v) for t, v in by_tbl.items()}
    dims_present = {}
    for t, facts in by_tbl.items():
        sexes = set(f.get("sex", "") for f in facts)
        dims = []
        if {"Male", "Female"} & sexes:
            dims.append("sex")
        if any(f.get("civil_condition") for f in facts):
            dims.append("civil_condition")
        if any(f.get("religion") for f in facts):
            dims.append("religion")
        if any(f.get("column_dimension") == "age" or f.get("row_dimension") == "age" for f in facts):
            dims.append("age")
        ages = [int(f["resolved_age_lower"]) for f in facts if str(f.get("resolved_age_lower", "")).lstrip("-").isdigit()]
        dims_present[t] = (";".join(dims), (f"{min(ages)}-{max(ages)}" if ages else ""))

    cat_cols = ["table_id", "status", "year", "decade", "canonical_name", "entity_type",
                "parent_geo", "broad_region", "geo_level", "canonical_topic", "subject",
                "measure_class", "additive", "table_family", "measure_category", "dims_present",
                "age_coverage", "n_facts", "review_note", "source_image", "member_images", "title"]
    all_tids = set(meta) | set(by_tbl)
    with open(os.path.join(V1, "catalog_tables.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cat_cols, extrasaction="ignore")
        w.writeheader()
        for t in sorted(all_tids):
            m = meta.get(t, {})
            gx = geores.get(t, {})
            yr = m.get("year", "")
            dp, ac = dims_present.get(t, ("", ""))
            ttl = ov["title"].get(t) or m.get("title", "")
            subj, meas, addv = classify_subject_measure(m, dom.get(t, ""), ttl, maxv.get(t))
            w.writerow({
                "table_id": t, "status": ov["status"].get(t) or ("recovered" if t in phase1 else None)
                or status.get(t, "excluded_or_unscoped"),
                "subject": subj, "measure_class": meas, "additive": addv,
                "review_note": ov["note"].get(t, ""),
                "year": yr, "decade": (yr[:3] + "0s") if yr[:3].isdigit() else "",
                "canonical_name": gx.get("canonical_name", ""),
                "entity_type": gx.get("entity_type", ""), "parent_geo": gx.get("parent_geo", ""),
                "broad_region": gx.get("broad_region", ""), "geo_level": gx.get("geo_level", ""),
                "canonical_topic": m.get("canonical_topic", ""), "table_family": m.get("table_family", ""),
                "measure_category": m.get("measure_category", ""), "dims_present": dp,
                "age_coverage": ac, "n_facts": fcount.get(t, 0),
                "source_image": m.get("source_image", ""),
                "member_images": m.get("assembled_member_source_images", "") or m.get("source_image", ""),
                "title": ov["title"].get(t) or m.get("title", ""),
            })

    print(f"unified_facts.csv: {fid-1:,} facts  ({n_conf:,} confirmed + {n_rec:,} recovered)")
    print(f"  recovered tables: {len(recovered_tables)}   synthesized-persons cells dropped: {dropped:,}")
    print(f"  total sent+recovered tables: {len(by_tbl):,}")
    print(f"catalog_tables.csv: {len(all_tids):,} tables (full corpus map)")
    print("  by status:", dict(Counter(status.get(t, 'excluded_or_unscoped') for t in all_tids)))


if __name__ == "__main__":
    main()
