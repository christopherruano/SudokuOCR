#!/usr/bin/env python3
"""Create a local browser for reviewing the static-site update manifest.

The HTML is standalone, reads no server state, and stores decisions in
localStorage until exported. It is only a review surface; it does not alter the
DB, current static site, or v4 artifacts beyond writing this HTML file.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import re
import sqlite3
from urllib.parse import quote


ROOT = os.path.dirname(os.path.abspath(__file__))
SENDOUT = os.path.join(ROOT, "export_final", "sendout_final", "sendout_v1")
DB = os.path.join(SENDOUT, "census_explorer.sqlite")
SITE_DIR = os.path.join(ROOT, "census_site")
V4_DIR = os.path.join(
    ROOT,
    "export_final",
    "pdf_recovery_incoming_2026-06-16_all_ocr_component_assembly_v4_adjacency",
)
REVIEW_CSV = os.path.join(V4_DIR, "static_site_update_review_queue.csv")
BACKFILL_CSV = os.path.join(V4_DIR, "static_site_update_review_image_backfill.csv")
OUT_HTML = os.path.join(V4_DIR, "static_site_update_review_browser.html")
OUT_SUMMARY = os.path.join(V4_DIR, "static_site_update_review_browser_summary.json")
OUT_MISSING_IMAGES = os.path.join(V4_DIR, "static_site_update_review_rows_missing_images.csv")


def read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def split_paths(raw: str) -> list[str]:
    return [p.strip() for p in (raw or "").split(";") if p.strip()]


def safe_filename(row: dict) -> str:
    base = f"{row.get('year')}_{row.get('canonical_name')}_{(row.get('title') or '')[:40]}"
    return (re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_")[:90]) or "table"


def rel_url(path: str) -> str:
    if not path:
        return ""
    abs_path = path if os.path.isabs(path) else os.path.join(ROOT, path)
    rel = os.path.relpath(abs_path, V4_DIR)
    # Keep path separators intact; quote special characters in each segment.
    return "/".join(quote(part) for part in rel.split(os.sep))


def load_current_table_context(table_ids: set[str]) -> dict[str, dict]:
    if not table_ids or not os.path.exists(DB):
        return {}
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    out = {}
    for tid in table_ids:
        row = con.execute("SELECT * FROM tables WHERE table_id=?", (tid,)).fetchone()
        if not row:
            continue
        d = dict(row)
        h = hashlib.md5(tid.encode()).hexdigest()[:7]
        page = f"{safe_filename(d)[:70]}_{h}.html"
        site_path = os.path.join(SITE_DIR, "tables", page)
        raw = d.get("member_images") or d.get("source_image") or ""
        imgs = split_paths(raw)
        out[tid] = {
            "current_site_url": rel_url(site_path) if os.path.exists(site_path) else "",
            "current_images": imgs,
            "current_title_full": d.get("title", ""),
            "current_status_db": d.get("status", ""),
        }
    con.close()
    return out


def load_image_backfills() -> dict[str, list[str]]:
    if not os.path.exists(BACKFILL_CSV):
        return {}
    out = {}
    for row in read_csv(BACKFILL_CSV):
        paths = [p for p in split_paths(row.get("rendered_image_paths", "")) if os.path.exists(p)]
        if paths:
            out[row.get("proposed_table_id", "")] = paths
    return out


def enrich_rows(rows: list[dict[str, str]]) -> tuple[list[dict], dict]:
    current_ids = {r.get("current_site_table_id", "") for r in rows if r.get("current_site_table_id")}
    current = load_current_table_context(current_ids)
    backfills = load_image_backfills()
    enriched = []
    counts = {
        "review_rows": len(rows),
        "rows_with_v4_images": 0,
        "rows_using_rendered_backfill_images": 0,
        "rows_using_current_images": 0,
        "rows_without_images": 0,
        "rows_with_current_site_link": 0,
    }

    for i, r in enumerate(rows, 1):
        d = dict(r)
        ctx = current.get(d.get("current_site_table_id", ""), {})
        v4_imgs = split_paths(d.get("member_images", ""))
        backfill_imgs = backfills.get(d.get("proposed_table_id", ""), [])
        current_imgs = ctx.get("current_images", [])
        display_imgs = v4_imgs or backfill_imgs or current_imgs
        if v4_imgs:
            counts["rows_with_v4_images"] += 1
            image_source = "v4 adjacency group"
        elif backfill_imgs:
            counts["rows_using_rendered_backfill_images"] += 1
            image_source = "rendered exact PDF page backfill"
        elif current_imgs:
            counts["rows_using_current_images"] += 1
            image_source = "current site table"
        else:
            counts["rows_without_images"] += 1
            image_source = "none"
        if ctx.get("current_site_url"):
            counts["rows_with_current_site_link"] += 1

        d["review_index"] = i
        d["image_source"] = image_source
        d["display_images"] = [{"path": p, "url": rel_url(p), "basename": os.path.basename(p)} for p in display_imgs]
        d["current_site_url"] = ctx.get("current_site_url", "")
        d["current_title_full"] = ctx.get("current_title_full", d.get("current_title", ""))
        d["current_status_db"] = ctx.get("current_status_db", d.get("current_status", ""))
        d["ocr_artifact_urls"] = [
            {"path": p, "url": rel_url(p), "basename": os.path.basename(p)}
            for p in split_paths(d.get("member_ocr_artifacts", ""))
        ]
        enriched.append(d)
    return enriched, counts


def render_html(rows: list[dict], counts: dict) -> str:
    payload = json.dumps(rows, ensure_ascii=False)
    counts_json = json.dumps(counts, ensure_ascii=False)
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Static Site Update Review</title>
<style>
 :root {{
   --ink:#172017; --muted:#647064; --line:#d9ded6; --panel:#fbfaf4; --paper:#fffef9;
   --accent:#326b45; --accent2:#9a5b28; --warn:#a3362d; --blue:#245d7a;
 }}
 *{{box-sizing:border-box}}
 body{{margin:0;font:14px/1.45 Georgia,'Iowan Old Style',serif;color:var(--ink);background:#eee8d8}}
 header{{padding:14px 18px;background:#263522;color:#fff;display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}}
 header h1{{font-size:19px;margin:0;font-weight:700;letter-spacing:.2px}}
 header span{{color:#d8e3d2;font-size:12px}}
 .toolbar{{position:sticky;top:0;z-index:5;background:#f7f1df;border-bottom:1px solid var(--line);padding:10px 14px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
 input,select,textarea{{font:13px/1.3 ui-sans-serif,system-ui,sans-serif;border:1px solid #c8cfc3;border-radius:5px;background:#fff;padding:6px}}
 button{{border:0;border-radius:5px;padding:7px 10px;background:#d8dfd2;cursor:pointer;font:700 12px ui-sans-serif,system-ui,sans-serif}}
 button.primary{{background:var(--accent);color:#fff}} button.warn{{background:var(--warn);color:#fff}}
 .layout{{display:grid;grid-template-columns:410px 1fr;min-height:calc(100vh - 98px)}}
 .list{{border-right:1px solid var(--line);background:#f6f0df;overflow:auto;max-height:calc(100vh - 98px)}}
 .card{{padding:10px 12px;border-bottom:1px solid var(--line);cursor:pointer;background:#fbf7ea}}
 .card:hover,.card.sel{{background:#e8f0e2}} .card.done{{opacity:.55}}
 .rank{{display:inline-block;background:#263522;color:#fff;border-radius:99px;padding:1px 7px;font:700 11px ui-sans-serif,system-ui,sans-serif;margin-right:6px}}
 .action{{font:700 11px ui-sans-serif,system-ui,sans-serif;color:var(--blue);text-transform:uppercase}}
 .title{{margin-top:5px;font-weight:700}} .meta,.small{{color:var(--muted);font-size:12px}} .small{{font-family:ui-sans-serif,system-ui,sans-serif}}
 .detail{{padding:16px 18px;overflow:auto;max-height:calc(100vh - 98px);background:var(--paper)}}
 .detail h2{{margin:0 0 4px;font-size:22px}} .pill{{display:inline-block;border:1px solid var(--line);border-radius:99px;padding:2px 8px;margin:2px;background:#fff;font:12px ui-sans-serif,system-ui,sans-serif}}
 .decision{{margin:12px 0;padding:10px;border:1px solid var(--line);background:var(--panel);border-radius:8px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}}
 .decision button.sel{{outline:3px solid #e2bd71}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:12px;margin-top:12px}}
 figure{{margin:0;background:#1f211d;padding:8px;border-radius:8px}} figcaption{{color:#d9dfd1;font:12px ui-sans-serif,system-ui,sans-serif;margin:0 0 5px}}
 img{{max-width:100%;display:block;background:#fff;margin:auto;border:1px solid #555}}
 table{{border-collapse:collapse;margin-top:10px;width:100%;font:12px ui-sans-serif,system-ui,sans-serif}}
 th,td{{border:1px solid var(--line);padding:5px 7px;text-align:left;vertical-align:top}} th{{background:#edf2e7;width:180px}}
 a{{color:#1f5e7a}} .empty{{padding:30px;color:var(--muted)}}
 @media(max-width:900px){{.layout{{grid-template-columns:1fr}}.list{{max-height:260px;border-right:0;border-bottom:1px solid var(--line)}}.detail{{max-height:none}}}}
</style></head>
<body>
<header><h1>Static Site Update Review</h1><span>v4 adjacency manifest · local-only decisions · export before closing if needed</span></header>
<div class="toolbar">
 <b id="counts"></b>
 <select id="actionFilter" onchange="renderList()"><option value="">all actions</option></select>
 <select id="priorityFilter" onchange="renderList()"><option value="">all priorities</option></select>
 <select id="decisionFilter" onchange="renderList()"><option value="">all decisions</option><option value="pending">pending</option><option value="accept">accept</option><option value="reject">reject</option><option value="needs_rebuild">needs rebuild</option><option value="defer">defer</option></select>
 <input id="q" oninput="renderList()" placeholder="search title/place/pdf/action" size="34">
 <button onclick="prevRow()">↑ Prev</button><button onclick="nextRow()">↓ Next</button>
 <button class="primary" onclick="exportJSON()">Export JSON</button><button onclick="exportCSV()">Export CSV</button>
</div>
<div class="layout"><div class="list" id="list"></div><div class="detail" id="detail"></div></div>
<script>
const ROWS = {payload};
const BUILD_COUNTS = {counts_json};
const KEY = 'static_site_update_review_decisions_v1';
let decisions = JSON.parse(localStorage.getItem(KEY) || '{{}}');
let filtered = [], selected = 0;
const $ = id => document.getElementById(id);
function esc(s){{return String(s||'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
function uniq(a){{return [...new Set(a.filter(Boolean))];}}
function initFilters(){{
  for (const [id, field] of [['actionFilter','manifest_action'],['priorityFilter','review_priority_label']]) {{
    const sel=$(id); uniq(ROWS.map(r=>r[field])).sort().forEach(v=>{{const o=document.createElement('option');o.value=v;o.textContent=v;sel.appendChild(o);}});
  }}
}}
function rowText(r){{return [r.current_title,r.adjacency_primary_title,r.canonical_name,r.year,r.broad_region,r.manifest_action,r.adjacency_page_range,r.source_pdf_ids,r.member_pdf_pages].join(' ').toLowerCase();}}
function renderList(){{
  const af=$('actionFilter').value, pf=$('priorityFilter').value, df=$('decisionFilter').value, q=$('q').value.toLowerCase().trim();
  filtered = ROWS.filter(r => (!af || r.manifest_action===af) && (!pf || r.review_priority_label===pf) && (!q || rowText(r).includes(q)) && (!df || (df==='pending' ? !decisions[r.proposed_table_id]?.decision : decisions[r.proposed_table_id]?.decision===df)));
  if (selected >= filtered.length) selected = Math.max(0, filtered.length-1);
  $('counts').textContent = `${{filtered.length}} shown · ${{ROWS.length}} review rows · ${{Object.keys(decisions).length}} decisions`;
  $('list').innerHTML = filtered.map((r,i)=>{{
    const d=decisions[r.proposed_table_id]||{{}};
    return `<div class="card ${{i===selected?'sel':''}} ${{d.decision?'done':''}}" onclick="selectRow(${{i}})">
      <div><span class=rank>${{esc(r.review_priority_rank)}}</span><span class=action>${{esc(r.manifest_action)}}</span></div>
      <div class=title>${{esc(r.year)}} · ${{esc(r.canonical_name||'(unknown)')}}</div>
      <div class=small>${{esc((r.current_title||r.adjacency_primary_title||'').slice(0,110))}}</div>
      <div class=meta>${{esc(r.adjacency_page_range)}} · ${{esc(r.adjacency_priority_label)}} ${{d.decision?'· '+esc(d.decision):''}}</div>
    </div>`;
  }}).join('') || '<div class=empty>No rows match filters.</div>';
  renderDetail();
}}
function selectRow(i){{selected=i;renderList();}}
function prevRow(){{if(selected>0){{selected--;renderList();}}}}
function nextRow(){{if(selected<filtered.length-1){{selected++;renderList();}}}}
function setDecision(decision){{
  const r=filtered[selected]; if(!r) return;
  const notes=$('notes')?.value||'';
  decisions[r.proposed_table_id]={{decision,notes,ts:new Date().toISOString(),proposed_table_id:r.proposed_table_id,current_site_table_id:r.current_site_table_id,adjacency_group_id:r.adjacency_group_id,manifest_action:r.manifest_action}};
  localStorage.setItem(KEY, JSON.stringify(decisions));
  renderList();
}}
function saveNotes(){{
  const r=filtered[selected]; if(!r) return;
  const d=decisions[r.proposed_table_id]||{{}};
  d.notes=$('notes').value; d.ts=new Date().toISOString();
  decisions[r.proposed_table_id]=d; localStorage.setItem(KEY, JSON.stringify(decisions)); renderList();
}}
function renderDetail(){{
  const r=filtered[selected]; if(!r) {{$('detail').innerHTML='<div class=empty>Select a row.</div>'; return;}}
  const d=decisions[r.proposed_table_id]||{{}};
  const imgs=(r.display_images||[]).map((im,i)=>`<figure><figcaption>${{i+1}}/${{r.display_images.length}} · ${{esc(im.basename)}}<br>${{esc(im.path)}}</figcaption><a href="${{esc(im.url)}}" target=_blank><img loading=lazy src="${{esc(im.url)}}"></a></figure>`).join('');
  const arts=(r.ocr_artifact_urls||[]).map(a=>`<a target=_blank href="${{esc(a.url)}}">${{esc(a.basename)}}</a>`).join('<br>') || '';
  $('detail').innerHTML = `<h2>${{esc(r.year)}} · ${{esc(r.canonical_name||'(unknown)')}}</h2>
    <div><span class=pill>${{esc(r.review_priority_rank)}} ${{esc(r.review_priority_label)}}</span><span class=pill>${{esc(r.manifest_action)}}</span><span class=pill>pages ${{esc(r.adjacency_page_range)}}</span><span class=pill>${{esc(r.image_source)}} images</span></div>
    <p><b>Current:</b> ${{esc(r.current_title||'(none)')}}<br><b>Adjacency:</b> ${{esc(r.adjacency_primary_title||'(none)')}}</p>
    <div class=decision>
      <b>Decision</b>
      ${{['accept','reject','needs_rebuild','defer'].map(x=>`<button class="${{d.decision===x?'sel':''}}" onclick="setDecision('${{x}}')">${{x}}</button>`).join('')}}
      <textarea id=notes placeholder="notes" rows=2 cols=44 onblur="saveNotes()">${{esc(d.notes||'')}}</textarea>
    </div>
    <p>${{r.current_site_url ? `<a target=_blank href="${{esc(r.current_site_url)}}">Open current static-site table</a>` : '<span class=meta>No current static-site table link</span>'}}</p>
    <table>
      <tr><th>Reason</th><td>${{esc(r.reason)}}</td></tr>
      <tr><th>Proposed ID</th><td>${{esc(r.proposed_table_id)}}</td></tr>
      <tr><th>Current ID</th><td>${{esc(r.current_site_table_id)}}</td></tr>
      <tr><th>Adjacency group</th><td>${{esc(r.adjacency_group_id)}}</td></tr>
      <tr><th>Page kinds</th><td>${{esc(r.adjacency_page_kinds)}}</td></tr>
      <tr><th>Review flags</th><td>${{esc(r.adjacency_review_flags || r.member_edge_review_flags || '')}}</td></tr>
      <tr><th>Source PDFs</th><td>${{esc(r.source_pdf_ids || r.source_pdfs || '')}}</td></tr>
      <tr><th>PDF pages</th><td>${{esc(r.member_pdf_pages)}}</td></tr>
      <tr><th>OCR artifacts</th><td>${{arts}}</td></tr>
      <tr><th>Rows / failed constraints</th><td>${{esc(r.member_rows_extracted_total)}} / ${{esc(r.member_constraint_failed_total)}}</td></tr>
    </table>
    <div class=grid>${{imgs || '<div class=empty>No image path available for this row.</div>'}}</div>`;
}}
function exportJSON(){{
  const blob=new Blob([JSON.stringify(Object.values(decisions),null,2)],{{type:'application/json'}});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='static_site_update_review_decisions.json'; a.click();
}}
function exportCSV(){{
  const cols=['proposed_table_id','current_site_table_id','adjacency_group_id','manifest_action','decision','notes','ts'];
  const lines=[cols.join(',')].concat(Object.values(decisions).map(d=>cols.map(c=>`"${{String(d[c]||'').replaceAll('"','""')}}"`).join(',')));
  const blob=new Blob([lines.join('\\n')],{{type:'text/csv'}});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='static_site_update_review_decisions.csv'; a.click();
}}
document.addEventListener('keydown', e=>{{if(e.key==='ArrowDown')nextRow(); if(e.key==='ArrowUp')prevRow();}});
initFilters(); renderList();
</script></body></html>"""


def main() -> None:
    rows = read_csv(REVIEW_CSV)
    enriched, counts = enrich_rows(rows)
    counts["output"] = OUT_HTML
    counts["summary_output"] = OUT_SUMMARY
    counts["missing_images_output"] = OUT_MISSING_IMAGES
    with open(OUT_HTML, "w") as f:
        f.write(render_html(enriched, counts))
    with open(OUT_SUMMARY, "w") as f:
        json.dump(counts, f, indent=2)
    missing = [
        {
            "review_index": r.get("review_index", ""),
            "review_priority_rank": r.get("review_priority_rank", ""),
            "review_priority_label": r.get("review_priority_label", ""),
            "manifest_action": r.get("manifest_action", ""),
            "proposed_table_id": r.get("proposed_table_id", ""),
            "adjacency_group_id": r.get("adjacency_group_id", ""),
            "adjacency_page_range": r.get("adjacency_page_range", ""),
            "member_pdf_pages": r.get("member_pdf_pages", ""),
            "adjacency_primary_title": r.get("adjacency_primary_title", ""),
            "source_pdf_ids": r.get("source_pdf_ids", ""),
            "reason": r.get("reason", ""),
        }
        for r in enriched
        if not r.get("display_images")
    ]
    write_csv(
        OUT_MISSING_IMAGES,
        missing,
        [
            "review_index",
            "review_priority_rank",
            "review_priority_label",
            "manifest_action",
            "proposed_table_id",
            "adjacency_group_id",
            "adjacency_page_range",
            "member_pdf_pages",
            "adjacency_primary_title",
            "source_pdf_ids",
            "reason",
        ],
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
