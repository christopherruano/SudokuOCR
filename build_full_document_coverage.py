#!/usr/bin/env python3
"""
build_full_document_coverage.py

ONE row per source document across the WHOLE drive (all 427 volumes), unifying:
  - documents.csv                  (which volumes we've processed; region/year)
  - export/document_tableVII_index.csv   (our OCR-based Table VII fan-out: sub-areas, gaps)
  - export/drive_tableVII_scan.csv       (text-layer candidate Table VII block for ALL volumes)
  - cc_volumes.csv                 (title / year for unprocessed volumes)

For each document the Table VII location = our OCR fan-out if we've processed it, else the
drive text-layer candidate block. Produces a single stage per document so the human can go
volume-by-volume and confirm/find the Table VII.

Output: export/document_coverage.csv   (keyed by saoa; dochash when processed)

stage (one unified status, worst-first for the worklist):
  ASSEMBLED     — processed, Table VII found & fully OCR'd (0 gaps)
  PARTIAL       — processed, Table VII found but has OCR gaps
  LOCATED       — Table VII block located (drive scan) but NOT yet OCR'd  <-- do it ourselves
  NEEDS_LOOK    — only scattered text hits (WEAK) — human should eyeball
  NOT_LOCATED   — text layer present, no age-table header found (NONE)
  NEEDS_OCR     — no text layer (NO_TEXT) — must render+OCR to even locate
"""
import csv, os, re, collections

V = "export_final/sendout_final/sendout_v1"
OUT = "export/document_coverage.csv"


def saoa_of(s):
    m = re.search(r'(\d{6,})', s or '')
    return m.group(1) if m else ''


def main():
    docs = list(csv.DictReader(open(os.path.join(V, "documents.csv"))))
    doc_by_saoa = {}
    dochash_by_saoa = {}
    for d in docs:
        sid = saoa_of(os.path.basename(d['source_pdf']))
        if sid:
            doc_by_saoa[sid] = d
            dochash_by_saoa[sid] = d['dochash']

    # our OCR-based fan-out index, keyed by dochash -> saoa
    idx_by_saoa = {}
    idxp = "export/document_tableVII_index.csv"
    if os.path.exists(idxp):
        for r in csv.DictReader(open(idxp)):
            sp = next((d['source_pdf'] for d in docs if d['dochash'] == r['dochash']), '')
            sid = saoa_of(os.path.basename(sp))
            if sid:
                idx_by_saoa[sid] = r

    # cc_volumes meta (title/year/region) for labeling, keyed by saoa
    cc = {}
    for r in csv.DictReader(open(os.path.join(V, "cc_volumes.csv"))):
        cc[r['saoa_id']] = r

    # drive scan is the universe of volumes physically present
    scan = list(csv.DictReader(open("export/drive_tableVII_scan.csv")))

    def clean_region(sid, d):
        if d and d.get('region'):
            return d['region']
        c = cc.get(sid, {})
        # prefer a short province_or_volume, else title
        pv = (c.get('province_or_volume') or '').strip()
        if pv and len(pv) < 40 and not pv[0].islower():
            return pv
        return (c.get('title') or pv or '')[:50]

    out = []
    for s in scan:
        sid = s['saoa']
        d = doc_by_saoa.get(sid)
        idx = idx_by_saoa.get(sid)
        processed = int(bool(d))
        region = clean_region(sid, d)
        year = (d or {}).get('year') or cc.get(sid, {}).get('year', '') or s.get('year', '')
        # our OCR-based VII, if any
        our_pages = int(idx['n_vii_pages']) if idx and idx.get('n_vii_pages', '').isdigit() else 0
        our_gap = int(idx['n_gap']) if idx and idx.get('n_gap', '').isdigit() else 0
        our_span = idx.get('page_span', '') if idx else ''
        n_sub = idx.get('n_subtables', '') if idx else ''
        our_kind = idx.get('kind', '') if idx else ''
        scan_status = s['status']
        cand = s.get('best_run', '')
        # unified stage
        if idx and our_kind == 'genuine_vii' and our_pages > 0:
            stage = 'ASSEMBLED' if our_gap == 0 else 'PARTIAL'
            vii_pages = our_span
            src = 'ocr'
        elif scan_status in ('FOUND', 'SHORT'):
            stage = 'LOCATED'; vii_pages = cand; src = 'textlayer'
        elif scan_status == 'WEAK':
            stage = 'NEEDS_LOOK'; vii_pages = s.get('vii_runs', ''); src = 'textlayer'
        elif scan_status == 'NONE':
            stage = 'NOT_LOCATED'; vii_pages = ''; src = ''
        else:
            stage = 'NEEDS_OCR'; vii_pages = ''; src = ''
        out.append(dict(
            saoa=sid, dochash=dochash_by_saoa.get(sid, ''), year=year, region=region,
            title=(cc.get(sid, {}).get('title', '') or '')[:70],
            pages=s.get('pages', ''), processed=processed,
            stage=stage, vii_source=src, vii_pages=vii_pages,
            scan_status=scan_status, scan_candidate=cand,
            our_vii_pages=our_pages, our_gaps=our_gap, n_subareas=n_sub,
            path=s.get('path', '')))

    order = {'PARTIAL': 0, 'LOCATED': 1, 'NEEDS_LOOK': 2, 'NOT_LOCATED': 3,
             'NEEDS_OCR': 4, 'ASSEMBLED': 5}
    out.sort(key=lambda r: (order.get(r['stage'], 9), -int(r['our_gaps'] or 0), r['year']))
    os.makedirs('export', exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)

    st = collections.Counter(r['stage'] for r in out)
    print(f"=== {len(out)} source documents (whole drive) ===")
    for k in ('ASSEMBLED', 'PARTIAL', 'LOCATED', 'NEEDS_LOOK', 'NOT_LOCATED', 'NEEDS_OCR'):
        print(f"  {k:12}: {st.get(k,0)}")
    have = st['ASSEMBLED'] + st['PARTIAL'] + st['LOCATED']
    print(f"\nTable VII located for {have}/{len(out)} documents ({100*have//len(out)}%)")
    print(f"  ready/needs-finishing (processed): {st['ASSEMBLED']+st['PARTIAL']}")
    print(f"  located but NOT yet OCR'd (do ourselves): {st['LOCATED']}")
    print(f"  need a human look / render: {st['NEEDS_LOOK']+st['NOT_LOCATED']+st['NEEDS_OCR']}")
    print(f"\nwrote {OUT}")


if __name__ == '__main__':
    main()
