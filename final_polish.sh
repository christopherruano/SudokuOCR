#!/bin/bash
cd /Users/chrisruano/Research/SudokuOCR
while pgrep -f post_chain_finish.sh >/dev/null 2>&1; do sleep 120; done
echo "[$(date +%H:%M)] finisher done — final polish"
echo "--- [a] fresh stitch discovery on the post-reingest corpus ---"
python3 - <<'PY' 2>&1 | grep -vE "indexing|images indexed"
import csv, os, glob, json, re
CORPUS="export/ocr_results_referenced/digitized_tables"; V1="export_final/sendout_final/sendout_v1"
loc={}
for r in csv.DictReader(open(f"{V1}/page_number_status.csv")):
    if r["pdf_page"].lstrip("-").isdigit(): loc.setdefault(r["imghash"],(r["dochash"],int(r["pdf_page"])))
def groups_of(d):
    return {g.lower() for s in d.get("data",[]) for row in s.get("rows",[]) for g,v in row.items() if isinstance(v,dict)}
def is_vii(d):
    t=str((d.get("metadata",{}) or {}).get("title","")).lower()
    return ("civil condition" in t or ("age" in t and "sex" in t)) and not re.search(r"race|caste|tribe|afflicted|infirm",t)
lefts=[]; rights=[]
for f in glob.glob(os.path.join(CORPUS,"*","*_oneshot.json")):
    try: d=json.load(open(f))
    except: continue
    if not is_vii(d): continue
    if sum(len(s.get("rows",[])) for s in d.get("data",[]))<4: continue
    if (d.get("metadata",{}) or {}).get("stitched_halfpage"): continue
    g=groups_of(d); m=re.search(r"([0-9a-f]{40})",os.path.basename(f))
    ih=m.group(1) if m else None
    if not ih or ih not in loc: continue
    has_unm=any("unmarr" in x for x in g); has_wid=any("widow" in x for x in g); has_tot=any(("total" in x or "popul" in x) for x in g)
    rec={"ih":ih,"dochash":loc[ih][0],"page":loc[ih][1]}
    if (has_tot or has_unm) and not has_wid: lefts.append(rec)
    elif has_wid and not has_tot and not has_unm: rights.append(rec)
json.dump(lefts,open("export/stitch_lefts.json","w")); json.dump(rights,open("export/stitch_rights.json","w"))
print(f"halves: {len(lefts)} LEFT / {len(rights)} RIGHT")
PY
python3 stitch_pilot.py --apply 2>&1 | grep -E "CLEAN|left halves" | head -2
echo "--- [b] regenerate all researcher bundles on fresh data ---"
python3 build_best_dataset.py 2>&1 | tail -2
python3 extract_pyramids.py 2>&1 | tail -1 && python3 validate_pyramids.py 2>&1 | head -2
python3 build_comparable.py 2>&1 | tail -1
python3 build_single_age.py 2>&1 | grep -vE "indexing|images" | tail -1
python3 compute_severity.py 2>&1 | head -1
echo "--- [c] DATA_GUIDE page ---"
python3 - <<'PY'
import csv, io, zipfile, datetime, html
stamp=datetime.datetime.now().strftime("%Y-%m-%d")
z=zipfile.ZipFile("census_site/data/tableVII_best.zip")
best=list(csv.DictReader(io.TextIOWrapper(z.open("catalog.csv"))))
from collections import Counter
q=Counter(r["quality"] for r in best)
page=f"""<!doctype html><meta charset=utf-8><title>Data guide</title>
<style>body{{font:15px/1.6 Georgia,serif;max-width:860px;margin:24px auto;color:#1a2233;padding:0 16px}}
h1,h2{{font-family:system-ui;color:#1f3864}} code{{background:#eef;padding:0 4px}} .dl{{background:#0a5d1e;color:#fff;padding:3px 10px;border-radius:6px;text-decoration:none}}</style>
<h1>HEB 91r — Age structure of colonial India: data guide</h1>
<p>Digitized Imperial <b>Table VII (Age, Sex &amp; Civil Condition)</b> from the Census of India, 1871–1951,
with full provenance from page scan to cell. Updated {stamp}.</p>
<h2>The four data products</h2>
<ol>
<li><a class=dl href='data/tableVII_best.zip'>tableVII_best.zip</a> — <b>the prime dataset</b>: the best table
for each of {len(best)} place-years (human-verified tables volumes first). One CSV per place-year in the
printed table's shape (age rows × Population/Unmarried/Married/Widowed × persons/males/females), plus a tidy
long file and a catalog with quality labels &amp; links to each table's rendered page beside its scan.</li>
<li><a href='data/tableVII_comparable.zip'>tableVII_comparable.zip</a> — <b>comparison layer</b>: harmonized
decade bands so places compare across years and years across places; per-sex shares included.</li>
<li><a href='singleage.html'>★ year-by-year age returns</a> — <b>analysis layer</b>: every table with
single-year age detail, unbinned (age-heaping, smoothing, single-age demography).</li>
<li><a href='data/coverage_matrix.png'>coverage matrix</a> (<a href='data/coverage_matrix.csv'>csv</a>) —
what exists for every province × census year and from which source class.</li>
</ol>
<h2>Quality labels</h2>
<p><b>verified</b> ({q.get('verified',0)}): passes every internal identity (M+F=Persons; Unmarried+Married+Widowed=Population; column sums).
<b>minor-issues</b> ({q.get('minor-issues',0)}): a few cells fail — usually one misread digit or a typo in the printed original (kept as printed).
<b>needs-review</b> ({q.get('needs-review',0)}): larger discrepancies, use with care.
Passing checks means internal consistency — a strong but not absolute signal.</p>
<h2>Caveats a reader should know</h2>
<ul><li>The 1941 census published very few tables (wartime) — gaps there are historical.</li>
<li>National-volume breakdowns give province rows only (coarser than provincial volumes).</li>
<li>Administrative boundaries change between censuses; the catalog names the exact source volume per table.</li>
<li>Age heaping (digit preference at 0/5) is a property of the enumeration, visible in the single-year returns.</li></ul>
<p><a href='index.html'>← back to the explorer</a></p>"""
open("census_site/data_guide.html","w").write(page)
print("data_guide.html written")
PY
echo "--- [d] final mini-deploy ---"
for try in 1 2 3 4 5 6; do
  OUT=$(npx wrangler pages deploy census_site --project-name=heb91r-5a20dde1 --branch=main --commit-dirty=true 2>&1 | tail -3)
  echo "$OUT"
  echo "$OUT" | grep -q "Deployment complete" && { echo "FINAL POLISH DEPLOYED"; break; }
  sleep 90
done
echo "[$(date +%H:%M)] === FINAL POLISH COMPLETE ==="
