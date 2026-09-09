#!/usr/bin/env python3
"""Detect multi-page Table VII RUNS with ROBUST boundaries. Reads the master page document.
Improvements over v1:
  - PRECISE VII-content classifier (topic-based, not the loose 'vii' substring that matches 'VIII'
    and not bare 'age' which catches 'Education by Age'). VII = age+sex/civil, or civil-condition,
    or ages-by-religion — and NOT education/language/birthplace/occupation/infirmity/caste/literacy.
  - GAP-BRIDGING: a single interior page that's OCR-garbage (footer/blank/untitled) between two
    VII pages does NOT split the run (that was a main cause of wrong start/end).
  - BOUNDARY CONFIDENCE: each run gets start_conf/end_conf + flags so review focuses on uncertain ends.
  python3 build_tableVII_runs.py  ->  export/tableVII_runs.json
"""
import csv, re, os, json, collections, glob

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
MASTER = os.path.join(V1, "page_number_status.csv")
DIG = "export/ocr_results_referenced/digitized_tables"

AGE = re.compile(r"\bages?\b", re.I)
SEXCIV = re.compile(r"\bsex\b|civil\s*condition|marital|conjugal", re.I)
CIVIL = re.compile(r"civil\s*condition", re.I)
BYREL = re.compile(r"ages?\s+by\s+relig", re.I)
# clear OTHER-table topics -> a VII run must break here
OTHER = re.compile(r"education|literacy|language|birth[\s-]*place|occupation|means of livelihood|"
                   r"infirmit|caste|tribe|race|nationality|religion of|housing|town|village|"
                   r"variation|migration|density|christian sect|civil\s*condition.*literacy", re.I)
CONTD = re.compile(r"contd|continued|\(cont|concluded", re.I)
PART_START = re.compile(r"part\s*[i1a]\b|part\s*one|^\s*table\s*vii\.?\s*$|^\s*table\s*vii\.?\s*[-.]", re.I)


def vii_content(title):
    """precise: is this page Table VII content (age/sex/civil-condition)?"""
    t = title or ""
    if not t.strip():
        return None  # ambiguous (untitled) — candidate for gap-bridge, not a hard yes/no
    # an OTHER topic that ISN'T civil-condition -> not VII
    if OTHER.search(t) and not CIVIL.search(t):
        return False
    if BYREL.search(t):
        return True
    if CIVIL.search(t):
        return True
    if AGE.search(t) and SEXCIV.search(t):
        return True
    return False


def vol_meta():
    m = {}
    p = os.path.join(V1, "documents.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            m[r.get("dochash", "")] = (r.get("year", ""), r.get("region", ""), r.get("source_pdf", ""))
    return m


def boundary_gap_titles():
    """page titles from the boundary-gap OCR (newly-filled start/interior pages), keyed by saoa->page."""
    import json as _j
    out = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(ROOT, "export/boundary_gap_ocr/**/*_oneshot.json"), recursive=True):
        m = re.search(r"(\d+)_p(\d+)", os.path.basename(f))
        if not m:
            continue
        try:
            d = _j.load(open(f))
        except Exception:
            continue
        out[m.group(1)][int(m.group(2))] = ((d.get("metadata", {}) or {}).get("title", "") or "")
    return out


def main():
    meta = vol_meta()
    sa_to_dh = {re.search(r"saoa\.crl\.(\d+)", v[2]).group(1): dh
                for dh, v in meta.items() if re.search(r"saoa\.crl\.(\d+)", v[2] or "")}
    # per volume: pdf_page -> (best title, vii_flag, table_dir, imghash)
    byvol = collections.defaultdict(dict)
    for r in csv.DictReader(open(MASTER)):
        if not str(r.get("pdf_page", "")).isdigit():
            continue
        dm = re.match(r"([0-9a-f]{16})_t", r["table_dir"])
        if not dm:
            continue
        pg = int(r["pdf_page"]); dh = dm.group(1)
        title = r.get("title", "")
        # keep the most informative title per page (longest non-garbage)
        prev = byvol[dh].get(pg)
        if prev is None or len(title) > len(prev[0]):
            byvol[dh][pg] = (title, vii_content(title), r["table_dir"], r["imghash"])
    # MERGE boundary-gap OCR titles (the newly-filled start pages) so the detector sees them
    import glob
    for sa, pages in boundary_gap_titles().items():
        dh = sa_to_dh.get(sa)
        if not dh:
            continue
        for pg, title in pages.items():
            if pg not in byvol[dh] and vii_content(title) is not False:
                byvol[dh][pg] = (title, vii_content(title), f"bgap_{sa}", f"bgap_{sa}_p{pg}")

    runs = []
    for dh, pages in byvol.items():
        order = sorted(pages)
        n = len(order)
        i = 0
        while i < n:
            pg = order[i]
            if pages[pg][1] is not True:
                i += 1; continue
            # start a run at i; extend across consecutive pdf pages that are VII or bridgeable
            run_pages = [pg]
            bridged = []   # interior page numbers that are missing/garbage but inside the run
            j = i
            while j + 1 < n:
                npg = order[j + 1]
                gap = npg - order[j]
                cont = pages[npg][1]
                if gap == 1 and cont is True:
                    run_pages.append(npg); j += 1
                elif gap == 1 and cont is None and (j + 2 < n and order[j + 2] == npg + 1 and pages[order[j + 2]][1] is True):
                    # bridge ONE untitled/garbage interior page flanked by VII
                    run_pages.append(npg); j += 1
                elif gap == 2 and cont is True:
                    # bridge ONE missing (not-OCR'd) interior page between two VII pages
                    bridged.append(order[j] + 1)
                    run_pages.append(npg); j += 1
                else:
                    break
            # CONT-MARKER backward extension: a run must START at a no-'cont' page.
            # If the first page says 'continued', walk back across the (now page-continuous) VII
            # pages to the true no-cont beginning. Stop at a gap or a different table.
            while CONTD.search(pages[run_pages[0]][0]):
                cand = run_pages[0] - 1
                if cand in pages and pages[cand][1] is True:
                    run_pages.insert(0, cand)
                    if not CONTD.search(pages[cand][0]):
                        break  # reached the fresh (no-cont) start
                elif cand in pages and pages[cand][1] is None and (cand - 1) in pages and pages[cand - 1][1] is True:
                    run_pages.insert(0, cand); bridged.append(cand)  # cross one blank/garbage page
                else:
                    break  # gap (missing page) or different table -> can't extend further here
            if len(run_pages) >= 2:
                yr, reg, pdf = meta.get(dh, ("", "", ""))
                sa = re.search(r"saoa\.crl\.(\d+)", pdf)
                first_t = pages[run_pages[0]][0]; last_t = pages[run_pages[-1]][0]
                # boundary confidence
                before = order[i - 1] if i > 0 else None
                after = order[j + 1] if j + 1 < n else None
                before_vii = before is not None and pages[before][1] is True and before == run_pages[0] - 1
                after_vii = after is not None and pages[after][1] is True and after == run_pages[-1] + 1
                start_clean = (not before_vii) and (not CONTD.search(first_t))
                end_clean = not after_vii
                # cont-marker start/end classification
                start_type = "clean_start" if not CONTD.search(first_t) else "continued_origin_missing"
                end_type = "concluded" if re.search(r"concl", last_t, re.I) else "open_end"
                flags = []
                pageset = set(order)
                # gap pages immediately before a 'continued' start (the true start, not yet OCR'd)
                missing_before = []
                if CONTD.search(first_t):
                    p = run_pages[0] - 1
                    while p > 0 and p not in pageset and run_pages[0] - p <= 6:
                        missing_before.append(p); p -= 1
                    flags.append("first_page_says_continued→start_may_be_earlier")
                if before_vii:
                    flags.append("VII_page_immediately_before→start_too_late")
                if after_vii:
                    flags.append("VII_page_immediately_after→end_too_early")
                if bridged:
                    flags.append(f"bridged_missing_interior_pages:{bridged}")
                runs.append({
                    "dochash": dh, "saoa": sa.group(1) if sa else dh, "year": yr, "region": reg,
                    "tnum": "VII", "page_range": [run_pages[0], run_pages[-1]], "n_pages": len(run_pages),
                    "start_clean": start_clean, "end_clean": end_clean, "flags": flags,
                    "start_type": start_type, "end_type": end_type,
                    "bridged_pages": bridged, "missing_before": sorted(missing_before),
                    "members": [{"page": p, "table_id": f"{DIG}/{pages[p][2]}/{pages[p][3]}_oneshot",
                                 "title": pages[p][0][:50]} for p in run_pages],
                })
            i = j + 1

    # uncertain-boundary runs first (so review targets them), then by length
    runs.sort(key=lambda r: (r["start_clean"] and r["end_clean"], -r["n_pages"]))
    json.dump(runs, open(os.path.join(ROOT, "export/tableVII_runs.json"), "w"), indent=1)
    # boundary-gap re-OCR worklist: saoa<TAB>page for the missing start/interior pages
    gaps = []
    for r in runs:
        for p in r.get("missing_before", []) + r.get("bridged_pages", []):
            gaps.append(f"{r['saoa']}\t{p}")
    open(os.path.join(ROOT, "export/tableVII_boundary_gaps.txt"), "w").write("\n".join(sorted(set(gaps))) + "\n")
    dist = collections.Counter(r["n_pages"] for r in runs)
    uncertain = sum(1 for r in runs if not (r["start_clean"] and r["end_clean"]))
    clean_start = sum(1 for r in runs if r["start_type"] == "clean_start")
    missing_origin = sum(1 for r in runs if r["start_type"] == "continued_origin_missing")
    concluded = sum(1 for r in runs if r["end_type"] == "concluded")
    print(f"  boundary-gap pages to re-OCR: {len(set(gaps))} -> export/tableVII_boundary_gaps.txt")
    print(f"  CONT-MARKER PASS: clean no-cont start={clean_start} | start still missing (gap)={missing_origin} | concluded-end={concluded}")
    print(f"Table VII runs: {len(runs)} (across {len(byvol)} volumes) | pages: {sum(r['n_pages'] for r in runs)}")
    print(f"  with UNCERTAIN boundary (review first): {uncertain}")
    print(f"  run-length distribution: {dict(sorted(dist.items()))}")
    print(f"  longest run: {max((r['n_pages'] for r in runs), default=0)} pages")


if __name__ == "__main__":
    main()
