#!/usr/bin/env python3
"""
build_drive_tableVII_scan.py

TOUCH EVERY PDF ON THE DRIVE and locate its Imperial Table VII (Age, Sex & Civil
Condition) — or a VII-LIKE age table — purely from the embedded TEXT LAYER. No
OCR, no page rendering, no memory pressure. The archive/CRL scans carry a (noisy)
text layer; the age-table running-headers ("TABLE VII … Age, Sex and Civil
Condition", "Age Returns", "FINAL CENSUS TABLE No. VII", etc.) survive well enough
to find the table's page block across census years 1871–1961.

Output: export/drive_tableVII_scan.csv  (one row per distinct drive volume)
  saoa, year, region, title, path, pages, has_text, n_hits, vii_runs, best_run,
  best_run_len, signal, status, already_processed, already_has_vii

status:
  FOUND       — a contiguous run (>=2 pp) with a strong age+civil/Table-VII signal
  SHORT       — only 1 strong page (short table, or only the summary present)
  WEAK        — scattered single hits only (likely TOC/cross-refs; needs a look)
  NONE        — text layer present but no age-table header found
  NO_TEXT     — no usable text layer (would need OCR to locate)
"""
import csv, os, re, glob, sys, collections

DRIVE = "/Volumes/Expansion/census_pdfs"
V = "export_final/sendout_final/sendout_v1"
OUT = "export/drive_tableVII_scan.csv"

# strong = unmistakably the age/sex/civil-condition table; medium = age-table name alone
STRONG = re.compile(r'(age,?\s+sex,?\s+and\s+civil\s+condition|'
                    r'civil\s+condition\b.{0,25}\bage|\bage\b.{0,25}civil\s+condition|'
                    r'conjugal\s+condition\b.{0,25}age|age\b.{0,25}conjugal)', re.I)
MEDIUM = re.compile(r'(imperial\s+table\s+vii\b|final census table\W+no\.?\s*vii|'
                    r'\btable\s+(no\.?\s*)?vii\b|\bage\s+returns?\b|'
                    r'single[\s-]*year\s+age|distribution.{0,40}by\s+age)', re.I)
EXC = re.compile(r'occupation|literac|maternity|infirm|\bblind\b|insane|deaf|leper|vaccinat|'
                 r'immigrant|emigrant|birth-?place|birthplace|\beducation\b', re.I)


def distinct_drive_pdfs():
    out = {}
    for p in glob.glob(os.path.join(DRIVE, "**", "saoa.crl.*.pdf"), recursive=True):
        m = re.search(r'saoa\.crl\.(\d+)\.pdf', p)
        if not m:
            continue
        sid = m.group(1)
        ym = re.search(r'/(\d{4})/', p)
        # prefer a path that lives under a YEAR folder (canonical copy)
        if sid not in out or (ym and '/census_pdfs/census_pdfs/' in out[sid]):
            out[sid] = p
    return out


def runs_of(pages, maxgap=3):
    if not pages:
        return []
    pages = sorted(pages); runs = [[pages[0], pages[0]]]
    for p in pages[1:]:
        if p - runs[-1][1] <= maxgap:
            runs[-1][1] = p
        else:
            runs.append([p, p])
    return runs


def scan(path):
    import fitz
    doc = fitz.open(path)
    n = doc.page_count
    strong, medium, total_txt = set(), set(), 0
    for i in range(n):
        t = doc.load_page(i).get_text()
        total_txt += len(t)
        # search header-ish window first (first 600 chars carry running headers in these scans)
        for m in STRONG.finditer(t):
            if not EXC.search(t[max(0, m.start() - 30):m.start() + 70]):
                strong.add(i + 1); break
        if i + 1 not in strong:
            for m in MEDIUM.finditer(t):
                if not EXC.search(t[max(0, m.start() - 30):m.start() + 70]):
                    medium.add(i + 1); break
    doc.close()
    return n, total_txt, strong, medium


def _worker(arg):
    """Process-pool worker: scan ONE volume, return raw detection (classification done in main)."""
    sid, path = arg
    try:
        n, txt, strong, medium = scan(path)
        return (sid, path, n, txt, sorted(strong), sorted(medium), None)
    except Exception as e:
        return (sid, path, None, None, [], [], str(e)[:40])


def main():
    drive = distinct_drive_pdfs()
    # context: which saoa are already processed / already have VII flagged
    docs = list(csv.DictReader(open(os.path.join(V, "documents.csv"))))
    proc = set()
    for d in docs:
        m = re.search(r'(\d{6,})', os.path.basename(d['source_pdf']))
        if m:
            proc.add(m.group(1))
    # region/title/year per saoa
    meta = {}
    for r in csv.DictReader(open(os.path.join(V, "cc_volumes.csv"))):
        meta[r['saoa_id']] = (r.get('year', ''), r.get('province_or_volume', '') or r.get('title', ''))
    # VII-flagged saoa from the page master
    flagged = set()
    pm = os.path.join(V, "page_number_status.csv")
    if os.path.exists(pm):
        seen = set()
        for r in csv.DictReader(open(pm)):
            if r.get('is_tableVII') == '1':
                m = re.search(r'saoa\.crl\.(\d+)', r.get('source_pdf', '') or '')
                # source_pdf in page master may be the cluster path; map via documents instead
        # simpler: flagged-by-doc -> saoa via documents.csv dochash join is overkill; rely on 'proc' + has-VII from index
    # use document_tableVII_index for already-has-vii by saoa
    has_vii = set()
    idxp = "export/document_tableVII_index.csv"
    if os.path.exists(idxp):
        dmap = {d['dochash']: d for d in docs}
        for r in csv.DictReader(open(idxp)):
            sp = dmap.get(r['dochash'], {}).get('source_pdf', '')
            m = re.search(r'(\d{6,})', os.path.basename(sp))
            if m and r.get('kind') in ('genuine_vii',) and int(r.get('n_vii_pages', 0)) > 0:
                has_vii.add(m.group(1))

    rows = []
    items = sorted(drive.items())
    total = len(items)
    workers = int(os.environ.get("WORKERS", "4"))
    sys.stderr.write(f"scanning {total} drive volumes with {workers} workers…\n"); sys.stderr.flush()
    from concurrent.futures import ProcessPoolExecutor
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = ex.map(_worker, items, chunksize=2)
        for (sid, path, n, txt, strong, medium, err) in results:
            done += 1
            if done % 25 == 0:
                sys.stderr.write(f"  {done}/{total}\n"); sys.stderr.flush()
            strong = set(strong); medium = set(medium)
            if err is not None:
                yr, ttl = meta.get(sid, ('', ''))
                rows.append(dict(saoa=sid, year=yr, region=ttl[:40], title=ttl[:60], path=path,
                                 pages='', has_text='', n_hits='', vii_runs='', best_run='',
                                 best_run_len='', signal='', status='ERROR:' + err,
                                 already_processed=int(sid in proc), already_has_vii=int(sid in has_vii)))
                continue
            allpages = strong | medium
            sruns = runs_of(sorted(strong))
            aruns = runs_of(sorted(allpages))
            # best block = longest contiguous all-hit run containing >=1 strong page (else longest strong run)
            best = None
            cand = [r for r in aruns if any(strong and (r[0] <= s <= r[1]) for s in strong)] or sruns
            if cand:
                best = max(cand, key=lambda r: r[1] - r[0])
            blen = (best[1] - best[0] + 1) if best else 0
            if best and blen >= 2 and strong:
                status, signal = 'FOUND', 'strong'
            elif strong:
                status, signal = 'SHORT', 'strong'
            elif allpages:
                status, signal = 'WEAK', 'medium-only'
            elif txt > 2000:
                status, signal = 'NONE', 'no-header'
            else:
                status, signal = 'NO_TEXT', 'no-textlayer'
            yr, ttl = meta.get(sid, ('', ''))
            rows.append(dict(
                saoa=sid, year=yr, region=ttl[:40], title=ttl[:70], path=path, pages=n,
                has_text=int(txt > 2000), n_hits=len(allpages),
                vii_runs=';'.join(f"{a}-{b}" for a, b in aruns[:12]),
                best_run=(f"{best[0]}-{best[1]}" if best else ''), best_run_len=blen,
                signal=signal, status=status,
                already_processed=int(sid in proc), already_has_vii=int(sid in has_vii)))

    os.makedirs('export', exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---- summary ----
    st = collections.Counter(r['status'] for r in rows)
    found = [r for r in rows if r['status'] in ('FOUND', 'SHORT')]
    unproc_found = [r for r in found if not r['already_processed']]
    print(f"\n=== {len(rows)} distinct drive volumes scanned ===")
    for s in ('FOUND', 'SHORT', 'WEAK', 'NONE', 'NO_TEXT'):
        print(f"  {s:8}: {st.get(s,0)}")
    print(f"\nHave a locatable Table VII / VII-like block (FOUND+SHORT): {len(found)} of {len(rows)} "
          f"({100*len(found)//len(rows)}%)")
    print(f"  …of which NOT yet processed (NEW coverage)            : {len(unproc_found)}")
    print(f"  already processed                                     : {sum(1 for r in found if r['already_processed'])}")
    print(f"\nwrote {OUT}")
    # by year coverage
    byyr = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        byyr[r['year']][0] += 1
        if r['status'] in ('FOUND', 'SHORT'):
            byyr[r['year']][1] += 1
    print("\nVII-locatable by census year (found/total):")
    for y in sorted(byyr):
        t, f = byyr[y]
        print(f"  {y or '?':6}: {f}/{t}")


if __name__ == '__main__':
    main()
