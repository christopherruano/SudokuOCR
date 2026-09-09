#!/usr/bin/env python3
"""Highlights page on census_site: surfaces the most important Age/Sex/Civil-Condition
tables (existing core stock + newly-incorporated), most-important-first then arranged
by provenance, each linking into the explorer. Reads the site's own search index (IDX).
Writes census_site/highlights.html + a homepage link. No re-render, low memory.
"""
import re, json, html, os, csv, glob, datetime, sqlite3, hashlib
STAMP = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

SITE = "census_site"
DB = "export_final/sendout_final/sendout_v1/census_explorer.sqlite"
# the complete mapped universe of relevant pages still being transcribed (OCR in progress)
LEDGER = "export_final/pdf_recovery_incoming_2026-06-16_no_gaps_ledger_v2_refined/source_pages_still_need_render.csv"
OCR_DIRS = ["export/flashdrive_ocr", "export/money_renders_ocr", "export/money_gapfill_ocr",
            "export_final/gemini_no_gap_ocr_2026-06-16_v2_best"]
VII = re.compile(r"civil\s*condition|age,?\s*sex|marital|conjugal|age\s*and\s*sex|sex\s*and\s*age", re.I)
IMPERIAL = re.compile(r"(imperial\s*)?table\s*vii\b.*(age|civil)|age,?\s*sex\s*and\s*civil\s*condition|\bc[-\s.]*iii\b", re.I)
ST = {"keeper_accepted": ("verified", "ok"), "confirmed_sent": ("verified", "ok"),
      "assembled": ("assembled", "ok"), "recovered": ("recovered", "ok"),
      "keeper_review": ("review", "warn")}


def ocr_status(n_browsable, n_imperial):
    """Coverage status: the whole relevant universe is mapped; some pages are still being read.
    on-the-way = gap-ledger pages identified but not yet browsable (OCR in progress / pending)."""
    ledger = set()
    if os.path.exists(LEDGER):
        for r in csv.DictReader(open(LEDGER)):
            ledger.add((r.get("saoa_id"), r.get("pdf_page")))
    done = set()
    for d in OCR_DIRS:
        for f in glob.glob(f"{d}/**/*_oneshot.json", recursive=True):
            m = re.search(r"(\d+)_p(\d+)", os.path.basename(f))
            if m:
                done.add((m.group(1), m.group(2)))
    onway = len(ledger)
    return dict(mapped=n_browsable + onway, browsable=n_browsable, imperial=n_imperial,
                onway=onway, queued=len(ledger - done), pending=len(ledger & done))


def page_stems():
    """map md5(table_id)[:7] -> rendered page filename stem (from the site's own tables/)."""
    m = {}
    for f in os.listdir(os.path.join(SITE, "tables")):
        if f.endswith(".html"):
            mm = re.search(r"_([0-9a-f]{7})$", f[:-5])
            if mm:
                m[mm.group(1)] = f[:-5]
    return m


def featured(stems, n=10):
    """Curate standout COMPLETE Imperial Table VII tables: verified status, dense (many cells),
    a rendered page + a real scan thumbnail, one per region/year for variety. Drives the showcase."""
    if not os.path.exists(DB):
        return []
    con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
    rows = con.execute("SELECT table_id,title,canonical_name,broad_region,year,status,n_facts,source_image "
                       "FROM tables").fetchall()
    con.close()
    cand = [r for r in rows if r["title"] and IMPERIAL.search(r["title"] or "")
            and r["status"] in ("keeper_accepted", "confirmed_sent") and (r["n_facts"] or 0) >= 200]
    cand.sort(key=lambda r: -(r["n_facts"] or 0))
    picks, seen, per_region = [], set(), {}
    for r in cand:
        stem = stems.get(hashlib.md5(r["table_id"].encode()).hexdigest()[:7])
        b = os.path.basename(r["source_image"] or "").replace(".png",".jpg")
        if not stem or not b or not os.path.exists(os.path.join(SITE, "img", b)):
            continue
        key = (r["broad_region"], r["year"])
        if key in seen or per_region.get(r["broad_region"], 0) >= 2:   # spread across provinces
            continue
        seen.add(key)
        per_region[r["broad_region"]] = per_region.get(r["broad_region"], 0) + 1
        picks.append({"stem": stem, "img": b, "region": r["broad_region"] or "—",
                      "prov": r["canonical_name"] or "", "year": r["year"] or "",
                      "title": r["title"], "n": r["n_facts"] or 0})
        if len(picks) >= n:
            break
    return picks


def main():
    h = open(os.path.join(SITE, "search.html"), encoding="utf-8").read()
    idx = json.loads(re.search(r'const IDX=(\[\{.*?\}\]);', h, re.S).group(1))
    rel = [r for r in idx if VII.search(r.get("t", "") or "") or "civil_condition" in (r.get("k", "") or "")]

    def tier(r):
        return 0 if IMPERIAL.search(r.get("t", "") or "") else 1

    def yr(r):
        try:
            return int(r.get("y") or 9999)
        except Exception:
            return 9999
    rel.sort(key=lambda r: (tier(r), (r.get("g", "") or "~~~").strip().lower(), yr(r), r.get("t", "")))

    css = """
    body{font:14px/1.5 system-ui,sans-serif;margin:0;background:#fbfdff;color:#1a2233}
    header{background:#1f3864;color:#fff;padding:12px 20px} header a{color:#cfe0ff}
    h1{margin:0;font-size:18px} .sub{opacity:.9;font-size:12px;margin-top:3px}
    .bar{margin:10px 20px} .bar input{width:100%;max-width:520px;padding:8px 11px;border:1px solid #cce;border-radius:6px}
    .sec{margin:14px 20px 4px;font-weight:700;color:#1f3864;font-size:15px;border-bottom:2px solid #e3e9f5;padding-bottom:3px}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:8px;margin:6px 20px}
    .c{display:block;background:#fff;border:1px solid #e2e6ee;border-left:4px solid #1f3864;border-radius:7px;padding:9px 11px;text-decoration:none;color:inherit}
    .c:hover{box-shadow:0 2px 8px rgba(0,0,0,.08)}
    .c .t{font-weight:600;font-size:13px} .c .m{color:#667;font-size:12px;margin-top:2px}
    .b{padding:1px 7px;border-radius:10px;font-size:11px;font-weight:700}
    .b.ok{background:#c6efce;color:#0a5d1e}.b.warn{background:#ffeb9c;color:#7a5b00}
    .show{display:flex;gap:12px;overflow-x:auto;padding:6px 20px 12px;scroll-snap-type:x proximity}
    .card{flex:0 0 220px;scroll-snap-align:start;background:#fff;border:1px solid #e2e6ee;border-radius:9px;
      overflow:hidden;text-decoration:none;color:inherit;box-shadow:0 1px 3px rgba(0,0,0,.05)}
    .card:hover{box-shadow:0 3px 12px rgba(0,0,0,.13)}
    .card img{width:100%;height:150px;object-fit:cover;object-position:top;background:#f3f5f9;display:block}
    .card .cap{padding:7px 9px} .card .cap b{font-size:12px;color:#1f3864}
    .card .cap .ti{color:#566;font-size:11px;margin-top:2px;line-height:1.35}
    .card .cap .n{color:#0a5d1e;font-size:11px;font-weight:700;margin-top:3px}
    """
    imp = sum(1 for r in rel if tier(r) == 0)
    st = ocr_status(len(rel), imp)
    picks = featured(page_stems(), n=10)
    show = ""
    if picks:
        cards = []
        for p in picks:
            cap = f"{html.escape(str(p['region']))} · {html.escape(str(p['year']))}"
            ti = html.escape((p["prov"] or "")[:46])
            cards.append(
                f'<a class=card href="tables/{html.escape(p["stem"])}.html">'
                f'<img loading=lazy src="img/{html.escape(p["img"])}">'
                f'<div class=cap><b>{cap}</b><div class=ti>{ti}</div>'
                f'<div class=n>{p["n"]:,} cells transcribed</div></div></a>')
        show = ('<div class=sec>Showcase — complete Imperial Table VII tables</div>'
                f'<div class=show>{"".join(cards)}</div>')
    status_html = (
        "<div style='margin:10px 20px;padding:11px 14px;background:#eef3fb;border:1px solid #d4def0;"
        "border-radius:8px;font-size:13px;color:#1a2233'>"
        f"<b>Coverage:</b> every relevant Age, Sex &amp; Civil-Condition table across all provinces and "
        f"census years is mapped &mdash; <b>{st['mapped']:,}</b> catalogued. "
        f"<span style='color:#0a5d1e;font-weight:700'>&#10003; {st['browsable']:,} transcribed &amp; browsable now</span> "
        f"({st['imperial']:,} Imperial Table VII). "
        f"<span style='color:#7a5b00;font-weight:700'>&#9203; {st['onway']:,} identified pages are being read by OCR</span> "
        "&mdash; they appear here automatically as each finishes.</div>")
    out = [f"<!doctype html><meta charset=utf-8><title>Highlights — Age, Sex & Civil Condition</title><style>{css}</style>",
           "<header><h1>Highlighted tables — Age, Sex &amp; Civil Condition</h1>"
           f"<div class=sub>{len(rel)} tables browsable · the Imperial Table VII / Age-Sex-Civil-Condition series first, "
           "then arranged by province &amp; year · <b>Last updated " + STAMP + "</b> · <a href='index.html'>&larr; full explorer</a></div></header>",
           status_html,
           show,
           "<div class=bar><input id=q placeholder='filter by province, year, title…'></div>"]
    cur_tier = None
    for r in rel:
        tt = tier(r)
        if tt != cur_tier:
            out.append('</div>' if cur_tier is not None else '')
            out.append(f'<div class=sec>{"Imperial Table VII — Age, Sex &amp; Civil Condition" if tt==0 else "Related civil-condition / age-sex tables"}</div><div class=grid>')
            cur_tier = tt
        lab, cls = ST.get(r.get("s", ""), ("digitized", "ok"))
        s = f"{r.get('g','')} {r.get('y','')} {r.get('t','')}".lower()
        out.append(f'<a class=c href="tables/{html.escape(r["p"])}.html" data-s="{html.escape(s)}">'
                   f'<div class=t>{html.escape((r.get("t","") or "(untitled)")[:80])}</div>'
                   f'<div class=m>{html.escape(r.get("g","") or "—")} · {html.escape(str(r.get("y","") or ""))} '
                   f'<span class="b {cls}">{lab}</span></div></a>')
    out.append('</div>')
    out.append("<script>const q=document.getElementById('q');q.addEventListener('input',()=>{const v=q.value.toLowerCase();"
               "document.querySelectorAll('.c').forEach(e=>e.style.display=(!v||e.dataset.s.includes(v))?'':'none');});</script>")
    open(os.path.join(SITE, "highlights.html"), "w", encoding="utf-8").write("\n".join(out))

    # homepage banner (marker-wrapped so it UPDATES with live coverage numbers each rebuild)
    ip = os.path.join(SITE, "index.html"); ih = open(ip, encoding="utf-8").read()
    banner = ("<!--HLBANNER-->"
              '<div style="margin:10px 16px;padding:10px 14px;background:#1f3864;color:#fff;border-radius:8px">'
              f'<b>Age, Sex &amp; Civil Condition &mdash; {st["mapped"]:,} tables mapped.</b> '
              f'&#10003; {st["browsable"]:,} browsable now ({st["imperial"]:,} Imperial Table VII); '
              f'&#9203; {st["onway"]:,} more being transcribed by OCR (appear as they finish). '
              '<a href="highlights.html" style="color:#ffd966;font-weight:700;text-decoration:none">Open highlights &rarr;</a>'
              "</div><!--/HLBANNER-->")
    # drop any prior banner (new marker block + legacy marker-less div) before re-injecting
    ih = re.sub(r"<!--HLBANNER-->.*?<!--/HLBANNER-->", "", ih, flags=re.S)
    ih = re.sub(r'<div style="margin:10px 16px;padding:10px 14px;background:#1f3864;color:#fff;border-radius:8px">.*?</div>',
                "", ih, flags=re.S)
    anchor = "click a region to drill into the columns</span></header>"
    if anchor in ih:
        ih = ih.replace(anchor, anchor + banner, 1)
    else:
        ih = ih.replace("</header>", "</header>" + banner, 1)
    open(ip, "w", encoding="utf-8").write(ih)
    print(f"wrote {SITE}/highlights.html: {len(rel)} browsable relevant "
          f"({imp} Imperial Table VII, {len(rel)-imp} related)")
    print(f"coverage: mapped={st['mapped']:,} browsable={st['browsable']:,} "
          f"on-the-way={st['onway']:,} (queued {st['queued']:,}, pending-incorp {st['pending']:,})")


if __name__ == "__main__":
    main()
