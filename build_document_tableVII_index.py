#!/usr/bin/env python3
"""
build_document_tableVII_index.py

Per-document AND per-sub-table inventory of Imperial Table VII (Age, Sex & Civil
Condition), built to TEST (not assume) "one Table VII per document" and to surface
the FINEST geographic/categorical breakdown each document actually contains.

A single document's Table VII fans out along several axes, varying by census year:
  - PART       : I=General/provincial summary, II=By Districts(&States), III=Districts-detail / Cities
  - TERRITORY  : named Taluk / Tahsil / District / City (the fine GEOGRAPHIC breakdown)
  - RELIGION   : All Religions / Hindu / Musalman / Christian / Jain / Parsi / Buddhist / Animistic / Sikh
  - CIVIL      : Unmarried / Married / Widowed split (1931-style)

Outputs:
  export/document_tableVII_index.csv   -- one row per DOCUMENT (summary + collapse ratio)
  export/tableVII_subtables.csv        -- one row per LEAF SUB-TABLE (the fine breakdown)

Read-only on all inputs. Does NOT touch tableVII_runs.json or the page master.
"""
import csv, json, collections, re, os, sys

PAGE_MASTER = "export_final/sendout_final/sendout_v1/page_number_status.csv"
PROV_FULL   = "export/provenance_full.csv"
RUNS        = "export/tableVII_runs.json"
OUT_DOC     = "export/document_tableVII_index.csv"
OUT_LEAF    = "export/tableVII_subtables.csv"

# --- title classification -------------------------------------------------
GEN   = re.compile(r'(age,?\s*sex,?\s*and\s*civil\s*condition|table\s*vii\b|civil\s*condition.*\bage\b|distribution of the population by (civil condition|age))', re.I)
CASTE = re.compile(r'\b(caste|supplementary table e|tribe|by selected castes)\b', re.I)
PART  = re.compile(r'part\s*[\.\-]?\s*([ivx]+|\d+|[a-d])\b', re.I)
UNIT  = re.compile(r'\b([A-Z][A-Za-z\.À-ſ]+(?:\s+[A-Z][A-Za-z\.À-ſ]+)?)\s+(Taluk|Tahsil|Tehsil|Zillah|Pargana|Subdivision|Sub-division)\b')
CITY  = re.compile(r'\bcit(?:y|ies)\b[\-\.\s]*([A-Z][A-Za-z]+)?', re.I)
# British-province "By Districts (and States)" tables: districts are ROWS inside, not separate titled tables.
BYDIST = re.compile(r'by\s+districts?(\s+(and|or)\s+states?)?', re.I)
RELIG = re.compile(r'\b(all religions?|hindu|brahmanic hindu|musalman|muslim|mohammedan|christian|jains?|jain|parsi|parsee|buddhist|animistic|animist|sikh|tribal|other religions?)\b', re.I)
CIVIL = re.compile(r'\b(unmarried|married|widowed)\b', re.I)
CONT  = re.compile(r'\b(continued|contd|concluded|concld|concl|ctd|cld)\b', re.I)

def year_of(src):
    m = re.search(r'/(\d{4})/', src or '')
    return m.group(1) if m else ''

def classify_title(t):
    """Return dict describing the leaf identity encoded in one page title."""
    t = (t or '').strip()
    d = {'genuine': bool(GEN.search(t)), 'caste': bool(CASTE.search(t)),
         'part': '', 'territory': '', 'religion': '', 'civil': '', 'cont': bool(CONT.search(t))}
    mp = PART.search(t)
    if mp: d['part'] = mp.group(1).upper().replace('PART', '').strip('.- ')
    mu = UNIT.search(t)
    if mu: d['territory'] = f"{mu.group(1).strip()} {mu.group(2)}"
    else:
        mc = CITY.search(t)
        if mc and mc.group(1): d['territory'] = f"City-{mc.group(1)}"
    mr = RELIG.search(t)
    if mr: d['religion'] = mr.group(1).title()
    mci = CIVIL.search(t)
    if mci: d['civil'] = mci.group(1).title()
    d['bydist'] = bool(BYDIST.search(t))
    return d

def unit_type(c):
    """Two-tier model: how fine a unit IS this page's sub-table?
      territory_leaf : native-state Taluk/Tahsil etc. printed as its OWN titled table -> split out.
      city_leaf      : Part III city table (Bombay, Poona...) -> own table.
      district_rows  : British 'By Districts (and States)' -> ONE unit, districts are ROWS inside (verify rows, don't split).
      religion_leaf / civil_leaf / part_leaf : categorical splits printed as separate tables.
      general        : provincial summary / general table."""
    if c['territory'].startswith('City-'): return 'city_leaf'
    if c['territory']: return 'territory_leaf'
    if c['bydist']:    return 'district_rows'
    if c['religion']:  return 'religion_leaf'
    if c['civil']:     return 'civil_leaf'
    if c['part']:      return 'part_leaf'
    return 'general'

def leaf_key(c):
    """Identity that distinguishes one geographic/categorical sub-table from another.
    Continuation/concluded pages share the key of their leaf (part+territory+religion+civil).
    For 'By Districts' tables territory is empty by design -> one leaf per (part)."""
    return (c['part'], c['territory'], c['religion'], c['civil'], c['bydist'])

def main():
    # authoritative per-SOURCE-DOCUMENT identity (region + year), keyed by dochash
    DOCS = "export_final/sendout_final/sendout_v1/documents.csv"
    doc_meta = {}
    if os.path.exists(DOCS):
        for r in csv.DictReader(open(DOCS)):
            doc_meta[r['dochash']] = (r.get('region', ''), r.get('year', ''), r.get('source_pdf', ''))

    rows = list(csv.DictReader(open(PAGE_MASTER)))
    seen = set(); ded = [r for r in rows if not (r['imghash'] in seen or seen.add(r['imghash']))]
    vii = [r for r in ded if r.get('is_tableVII') == '1']

    # OCR'd pages per doc (from provenance_full)
    prov = list(csv.DictReader(open(PROV_FULL))) if os.path.exists(PROV_FULL) else []
    ocr_pages = collections.defaultdict(set)   # dochash -> set(pdf_page int)
    for r in prov:
        if r['pdf_page'].lstrip('-').isdigit():
            ocr_pages[r['dochash']].add(int(r['pdf_page']))

    # manual runs per doc
    runs = json.load(open(RUNS)) if os.path.exists(RUNS) else []
    runs_per_doc = collections.Counter(r['dochash'] for r in runs)

    # group VII pages by doc
    bydoc = collections.defaultdict(list)
    for r in vii:
        bydoc[r['dochash']].append(r)

    leaf_rows = []
    doc_rows = []
    for dh, rs in bydoc.items():
        src = rs[0]['source_pdf']
        region, yr_meta, _ = doc_meta.get(dh, ('', '', ''))
        yr = yr_meta or year_of(src)
        # dedup to pages, sort
        pages = {}
        for r in rs:
            if r['pdf_page'].isdigit():
                pages.setdefault(int(r['pdf_page']), r)
        ordered = sorted(pages.items())
        gen_n = caste_n = 0
        # A leaf = a maximal CONTIGUOUS run of pages sharing one identity (so two separate
        # "Part I" printings, or non-adjacent unlabeled pages, become SEPARATE leaves — faithful
        # to how the territorial/categorical sub-tables physically sit in the volume).
        leaves = []          # list of {key,label,pages[],axes:set,utype}
        cur = None
        axes = set()
        prev = None          # last identity seen, for forward-fill onto bare continuation pages
        prev_pg = None
        for pg, r in ordered:
            c = classify_title(r['title'])
            gen_n += c['genuine']; caste_n += c['caste']
            # forward-fill: a bare/continuation page (no new identity declared) inside a contiguous
            # run inherits the previous page's identity (printed running-headers drop the taluk name
            # on verso pages). Reset if there's a page gap > 2 (likely a new section).
            bare = not (c['part'] or c['territory'] or c['religion'] or c['civil'] or c['bydist'])
            if bare and prev is not None and prev_pg is not None and (pg - prev_pg) <= 2:
                c = dict(c); c.update({k: prev[k] for k in ('part', 'territory', 'religion', 'civil', 'bydist')})
            if c['part'] or c['territory'] or c['religion'] or c['civil'] or c['bydist']:
                prev = c
            prev_pg = pg
            if c['part']: axes.add('PART')
            if c['territory']: axes.add('TERRITORY')
            if c['religion']: axes.add('RELIGION')
            if c['civil']: axes.add('CIVIL')
            k = leaf_key(c)
            # start a new leaf when identity changes OR pages are no longer contiguous (gap > 2)
            if cur is None or cur['key'] != k or (pg - cur['pages'][-1]) > 2:
                lbl = ' / '.join(x for x in [
                    ('Part ' + c['part']) if c['part'] else '',
                    c['territory'], c['religion'], c['civil'],
                    'By Districts' if (c['bydist'] and not c['territory']) else ''] if x) or '(general / unlabeled)'
                cur = {'key': k, 'label': lbl, 'pages': [], 'axes': set(), 'utype': unit_type(c)}
                leaves.append(cur)
            cur['pages'].append(pg)
            for a in ('part', 'territory', 'religion', 'civil'):
                if c[a]: cur['axes'].add(a)
        ocp = ocr_pages.get(dh, set())
        # leaf-level rows
        for L in leaves:
            pgs = sorted(L['pages'])
            ocrd = [p for p in pgs if p in ocp]
            gaps = [p for p in pgs if p not in ocp]
            leaf_rows.append({
                'dochash': dh, 'year': yr, 'source_pdf': os.path.basename(src),
                'subtable_label': L['label'], 'unit_type': L['utype'],
                'axes': '+'.join(sorted(L['axes'])) or 'none',
                'n_pages': len(pgs), 'n_ocrd': len(ocrd), 'n_gap': len(gaps),
                'page_first': pgs[0], 'page_last': pgs[-1],
                'pages': ' '.join(map(str, pgs)),
                'gap_pages': ' '.join(map(str, gaps[:40])),
            })
        n_pages = len(ordered)
        n_ocrd = sum(1 for pg, _ in ordered if pg in ocp)
        doc_rows.append({
            'dochash': dh, 'region': region, 'year': yr, 'source_pdf': os.path.basename(src),
            'n_vii_pages': n_pages, 'n_ocrd': n_ocrd, 'n_gap': n_pages - n_ocrd,
            'pct_ocrd': round(100 * n_ocrd / n_pages) if n_pages else 0,
            'n_subtables': len(leaves),
            'axes': '+'.join(sorted(axes)) or 'none',
            'n_manual_runs': runs_per_doc.get(dh, 0),
            'collapse_ratio': round(len(leaves) / runs_per_doc[dh], 1) if runs_per_doc.get(dh) else '',
            'gen_pages': gen_n, 'caste_pages': caste_n,
            'kind': 'caste_lookalike' if caste_n > gen_n else ('genuine_vii' if gen_n else 'unlabeled'),
            'page_span': f"{ordered[0][0]}-{ordered[-1][0]}" if ordered else '',
        })

    os.makedirs('export', exist_ok=True)
    doc_rows.sort(key=lambda d: (-d['n_gap'], -d['n_subtables']))
    leaf_rows.sort(key=lambda d: (d['dochash'], d['page_first']))
    with open(OUT_DOC, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(doc_rows[0].keys())); w.writeheader(); w.writerows(doc_rows)
    with open(OUT_LEAF, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(leaf_rows[0].keys())); w.writeheader(); w.writerows(leaf_rows)

    # --- console summary ---
    nd = len(doc_rows)
    genuine = [d for d in doc_rows if d['kind'] == 'genuine_vii']
    caste = [d for d in doc_rows if d['kind'] == 'caste_lookalike']
    multi = [d for d in doc_rows if d['n_subtables'] > 1]
    complete = [d for d in genuine if d['n_gap'] == 0]
    has_terr = [d for d in doc_rows if 'TERRITORY' in d['axes']]
    collapsing = [d for d in doc_rows if isinstance(d['collapse_ratio'], float) and d['collapse_ratio'] > 1.5]
    print(f"=== {nd} documents with Table VII pages ===")
    print(f"  genuine general-pop VII : {len(genuine)}")
    print(f"  caste/other lookalike   : {len(caste)}")
    print(f"  >1 distinct sub-table   : {len(multi)}  (assumption 'one VII per doc' FALSE for these)")
    print(f"  has named TERRITORY axis: {len(has_terr)}")
    print(f"  fully OCR'd (0 gaps)    : {len(complete)} of {len(genuine)} genuine")
    print(f"  assembly collapses fine breakdown (subtables/runs > 1.5): {len(collapsing)} docs")
    print(f"  total leaf sub-tables   : {len(leaf_rows)}")
    ut = collections.Counter(r['unit_type'] for r in leaf_rows)
    print("  leaf unit_type tally    : " + ', '.join(f"{k}={v}" for k, v in ut.most_common()))
    print(f"\nwrote {OUT_DOC} ({len(doc_rows)} rows) and {OUT_LEAF} ({len(leaf_rows)} rows)")
    print("\nTop 12 docs by OCR gap (most incomplete first):")
    print("  %-18s %-5s %4s %4s %4s %5s %4s  %-22s %s" % ('dochash','year','pgs','ocr','gap','subtb','runs','axes','kind'))
    for d in doc_rows[:12]:
        print("  %-18s %-5s %4d %4d %4d %5d %4s  %-22s %s" % (
            d['dochash'], d['year'], d['n_vii_pages'], d['n_ocrd'], d['n_gap'],
            d['n_subtables'], d['n_manual_runs'], d['axes'], d['kind']))

if __name__ == '__main__':
    main()
