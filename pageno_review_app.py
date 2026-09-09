#!/usr/bin/env python3
"""Human page-number review website.

Gemini read a printed page number off every scan EXCEPT ~282 (overwhelmingly
Roman-numeral pages on supplementary/subsidiary tables, where the Roman page
number is easy to confuse with the Roman table number in the title). This little
site shows you those scans one at a time so you can just TYPE the page number you
see (Roman like 'xiii' or Arabic like '13' — both convert to an int), or mark it
as having none.

For context it also shows the OTHER pages of the same assembled table and the
numbers already read for them, so you can see the sequence you're slotting into.

Decisions are saved immediately to:
  export_final/sendout_final/sendout_v1/pageno_parts/human_pageno_decisions.json

Run:
  python3 pageno_review_app.py [--queue null|all|fallback] [--port 5005]
Then open http://127.0.0.1:5005
"""
import argparse, io, json, os, re, threading
from datetime import datetime, timezone

from flask import Flask, request, jsonify, send_file, redirect
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
PARTS = os.path.join(V1, "pageno_parts")
JSONL = os.path.join(PARTS, "gemini_reads.jsonl")
ROMAN_JSONL = os.path.join(PARTS, "roman_reads.jsonl")
DECISIONS = os.path.join(PARTS, "human_pageno_decisions.json")

app = Flask(__name__)
LOCK = threading.Lock()

ROMAN = {"m": 1000, "d": 500, "c": 100, "l": 50, "x": 10, "v": 5, "i": 1}


def roman_to_int(s):
    s = s.strip().lower()
    if not s or any(ch not in ROMAN for ch in s):
        return None
    total, prev = 0, 0
    for ch in reversed(s):
        val = ROMAN[ch]
        total += -val if val < prev else val
        prev = max(prev, val)
    return total if 0 < total < 4000 else None


def parse_pageno(raw):
    """Return (int_value, numeral_system) or (None, 'none')."""
    if raw is None:
        return None, "none"
    t = str(raw).strip().strip(".[]() ").strip()
    if t == "":
        return None, "none"
    if re.fullmatch(r"\d{1,4}", t):
        return int(t), "arabic"
    rv = roman_to_int(t)
    if rv is not None:
        return rv, "roman"
    return None, "invalid"


def load_reads():
    by_hash = {}
    for line in open(JSONL):
        line = line.strip()
        if line:
            try:
                r = json.loads(line)
                by_hash[r["imghash"]] = r
            except Exception:
                pass
    return by_hash


def load_decisions():
    if os.path.exists(DECISIONS):
        return json.load(open(DECISIONS))
    return {}


def save_decisions(d):
    tmp = DECISIONS + ".tmp"
    json.dump(d, open(tmp, "w"), indent=1)
    os.replace(tmp, DECISIONS)


READS = {}
SUGGEST = {}              # imghash -> roman-aware AI suggestion
DUAL = {}                 # imghash -> dual-number read {volume_page, section_page, ...}
PROBLEMS = {}             # imghash -> {reasons, table_pages, page}
MEMBERS = {}
TITLES = {}               # asm_id -> title
PR_PAGENO = {}            # table_id -> int page number (gemini)
HASH_TO_ASM = {}          # imghash -> [asm_ids]
ASM_MEMBER_HASHES = {}    # asm_id -> [imghash,...] in member order, deduped
QUEUE = []                # ordered list of imghashes to review (scan mode)
TABLE_QUEUE = []          # ordered list of asm_ids to review (table mode)
HASH_RE = re.compile(r"([0-9a-f]{40})_oneshot")


def imghash(tid):
    m = HASH_RE.search(tid or "")
    return m.group(1) if m else None


def load_suggestions():
    out = {}
    if os.path.exists(ROMAN_JSONL):
        for line in open(ROMAN_JSONL):
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    out[r["imghash"]] = r
                except Exception:
                    pass
    return out


def load_titles():
    t = {}
    import csv
    for fn in ("tableVII_verdicts.csv", "assembled_catalog.csv"):
        p = os.path.join(V1, fn)
        if not os.path.exists(p):
            continue
        for row in csv.DictReader(open(p)):
            tid = row.get("table_id") or row.get("asm_table_id") or ""
            ttl = row.get("title") or row.get("review_note") or ""
            if tid and tid not in t and ttl:
                t[tid] = ttl
    return t


def load_dual():
    out = {}
    p = os.path.join(PARTS, "dualnum_reads.jsonl")
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    out[r["imghash"]] = r
                except Exception:
                    pass
    return out


def build_state(queue_mode, tables_scope="attention"):
    global READS, SUGGEST, DUAL, PROBLEMS, MEMBERS, TITLES, HASH_TO_ASM, ASM_MEMBER_HASHES, QUEUE, TABLE_QUEUE, PROBLEM_TABLES
    READS = load_reads()
    SUGGEST = load_suggestions()
    DUAL = load_dual()
    TITLES = load_titles()
    pf = os.path.join(PARTS, "problems.json")
    problem_asm_order = []
    if os.path.exists(pf):
        for r in json.load(open(pf)):
            PROBLEMS[r["imghash"]] = r
            for a in r["asm_ids"]:
                if a not in problem_asm_order:
                    problem_asm_order.append(a)
    MEMBERS = json.load(open(os.path.join(V1, "assembled_members.json")))
    for h, r in READS.items():
        for a in r.get("asm_ids", []):
            HASH_TO_ASM.setdefault(h, []).append(a)
    for a, tids in MEMBERS.items():
        hs = []
        for t in tids:
            h = imghash(t)
            if h and h in READS and h not in hs:   # resolvable physical pages, deduped
                hs.append(h)
        ASM_MEMBER_HASHES[a] = hs

    # table-mode queue: problem tables first, then the rest of the multi-page tables
    problem_set = set(problem_asm_order)
    PROBLEM_TABLES = problem_set
    multi = [a for a, hs in ASM_MEMBER_HASHES.items() if len(hs) > 1]
    rest_multi = sorted(a for a in multi if a not in problem_set)
    if tables_scope == "attention":
        TABLE_QUEUE = problem_asm_order + rest_multi      # attention first, all multi after
    elif tables_scope == "problems":
        TABLE_QUEUE = list(problem_asm_order)
    else:  # 'all'
        TABLE_QUEUE = problem_asm_order + rest_multi + sorted(
            a for a, hs in ASM_MEMBER_HASHES.items() if len(hs) == 1 and a not in problem_set)

    if queue_mode == "problems":
        # preserve problems.json order (dup -> null -> lowconf -> outlier, grouped by table)
        pf = os.path.join(PARTS, "problems.json")
        QUEUE = [r["imghash"] for r in json.load(open(pf)) if r["imghash"] in READS]
        return

    def want(r):
        if queue_mode == "all":
            return True
        if queue_mode == "fallback":
            return bool(r.get("used_fallback"))
        return r.get("page_number") is None  # 'null' (default)

    items = [h for h, r in READS.items() if want(r)]
    # group by first assembled table so same-table pages are adjacent
    items.sort(key=lambda h: (sorted(READS[h].get("asm_ids", ["~"]))[0], READS[h]["scan_path"]))
    QUEUE = items


def crop_bytes(scan, mode):
    im = Image.open(scan).convert("RGB")
    w, h = im.size
    if mode == "top":
        im = im.crop((0, 0, w, max(1, int(h * 0.34))))
        if im.width > 1500:
            im = im.resize((1500, int(im.height * 1500 / im.width)))
    elif mode == "full_small":
        if im.width > 1100:
            im = im.resize((1100, int(im.height * 1100 / im.width)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return buf


@app.route("/img/<h>")
def img(h):
    r = READS.get(h)
    if not r:
        return "no", 404
    scan = os.path.join(ROOT, r["scan_path"])
    if not os.path.exists(scan):
        return "missing", 404
    mode = request.args.get("mode", "top")
    if mode == "raw":
        return send_file(scan, mimetype="image/png")
    return send_file(crop_bytes(scan, mode), mimetype="image/jpeg")


def context_sequence(h):
    """For the assembled table(s) this scan is in, list members + their current page no."""
    dec = load_decisions()
    out = []
    seen = set()
    for a in READS[h].get("asm_ids", []):
        seq = []
        for mh in ASM_MEMBER_HASHES.get(a, []):
            if mh in seen and mh != h:
                pass
            pn = None
            src = ""
            if mh in dec and dec[mh].get("page_number") is not None:
                pn = dec[mh]["page_number"]; src = "you"
            elif READS.get(mh, {}).get("page_number") is not None:
                pn = READS[mh]["page_number"]; src = "ai"
            seq.append({"hash": mh, "page": pn, "src": src, "is_current": mh == h})
        out.append({"asm_id": a, "seq": seq})
        seen.update(ASM_MEMBER_HASHES.get(a, []))
    return out


@app.route("/state")
def state():
    dec = load_decisions()
    pending = [h for h in QUEUE if h not in dec]
    return jsonify({
        "total": len(QUEUE),
        "decided": sum(1 for h in QUEUE if h in dec),
        "pending": len(pending),
    })


@app.route("/item")
def item():
    dec = load_decisions()
    idx = request.args.get("i")
    if idx is not None and idx != "":
        i = max(0, min(len(QUEUE) - 1, int(idx)))
    else:
        # first undecided
        i = 0
        for j, h in enumerate(QUEUE):
            if h not in dec:
                i = j
                break
    if not QUEUE:
        return jsonify({"empty": True})
    h = QUEUE[i]
    r = READS[h]
    ctx = context_sequence(h)
    # neighbour page numbers already known for this scan's table(s)
    neighbours = set()
    for t in ctx:
        for s in t["seq"]:
            if not s["is_current"] and s["page"] is not None:
                neighbours.add(s["page"])
    sug = SUGGEST.get(h)
    suggestion = None
    if sug and sug.get("page_number_int") is not None:
        pv = sug["page_number_int"]
        suggestion = {
            "page_number": pv,
            "raw": sug.get("page_number_raw"),
            "numeral": sug.get("numeral_system"),
            "confidence": sug.get("confidence"),
            "title": sug.get("title_text"),
            "table_number": sug.get("table_number_in_title"),
            "conflict": pv in neighbours,           # same number already used in this table
            "fits": (not neighbours) or (min(neighbours) - 30 <= pv <= max(neighbours) + 30 and pv not in neighbours),
        }
    return jsonify({
        "i": i, "total": len(QUEUE), "hash": h,
        "scan_path": r["scan_path"],
        "filename": os.path.basename(r["scan_path"]),
        "ai_read": r.get("page_number"),
        "ai_conf": r.get("confidence"),
        "continuation": r.get("continuation_marker"),
        "used_fallback": r.get("used_fallback"),
        "n_tables": len(r.get("asm_ids", [])),
        "decision": dec.get(h),
        "suggestion": suggestion,
        "problem": PROBLEMS.get(h),
        "context": ctx,
        "decided_count": sum(1 for x in QUEUE if x in dec),
    })


@app.route("/save", methods=["POST"])
def save():
    data = request.get_json(force=True)
    h = data["hash"]
    raw = data.get("raw", "")
    none = data.get("none", False)
    if none:
        val, system = None, "none"
    else:
        val, system = parse_pageno(raw)
        if system == "invalid":
            return jsonify({"ok": False, "error": f"can't parse '{raw}'"}), 400
    with LOCK:
        dec = load_decisions()
        dec[h] = {
            "page_number": val, "raw": raw, "numeral": system,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        save_decisions(dec)
    return jsonify({"ok": True, "page_number": val, "numeral": system})


def current_for(h, dec):
    """Best current page number: human > first-pass top-Arabic > dual Arabic-volume > Roman section."""
    if h in dec:
        v = dec[h].get("page_number")
        return (int(v) if v is not None else None, "human")
    r = READS.get(h, {})
    if r.get("page_number") is not None:
        return (int(r["page_number"]), "firstpass")
    d = DUAL.get(h)
    if d and d.get("volume_page") is not None:
        return (int(d["volume_page"]), "volume")
    if d and d.get("section_page") is not None:
        return (int(d["section_page"]), "section")
    s = SUGGEST.get(h)
    if s and s.get("page_number_int") is not None:
        return (int(s["page_number_int"]), "roman")
    return (None, "")


def section_for(h):
    d = DUAL.get(h)
    if d and d.get("section_page") is not None and d.get("volume_page") is not None:
        return d["section_page"]
    return None


def table_members(asm_id, dec):
    out = []
    for h in ASM_MEMBER_HASHES.get(asm_id, []):
        page, source = current_for(h, dec)
        s = SUGGEST.get(h) or {}
        d = DUAL.get(h) or {}
        r = READS.get(h, {})
        out.append({
            "hash": h,
            "filename": os.path.basename(r.get("scan_path", h)),
            "page": page, "source": source,
            "section_page": section_for(h),
            "top_raw": d.get("top_raw"), "bottom_raw": d.get("bottom_raw"),
            "ai_raw": s.get("page_number_raw"), "ai_numeral": s.get("numeral_system"),
            "ai_title": s.get("title_text"), "ai_table_no": s.get("table_number_in_title"),
            "continuation": r.get("continuation_marker"),
            "decided": h in dec,
        })
    out.sort(key=lambda m: (0, m["page"]) if m["page"] is not None else (1, 0))
    return out


PROBLEM_TABLES = set()


@app.route("/tables")
def tables():
    dec = load_decisions()
    out = []
    for a in TABLE_QUEUE:
        hs = ASM_MEMBER_HASHES.get(a, [])
        pages = [p for p in (current_for(h, dec)[0] for h in hs) if p is not None]
        out.append({
            "asm_id": a, "title": TITLES.get(a, ""), "n": len(hs), "n_numbered": len(pages),
            "problem": a in PROBLEM_TABLES, "dup": len(pages) != len(set(pages)),
            "all_done": bool(hs) and all(h in dec for h in hs),
        })
    return jsonify({"tables": out, "total": len(out)})


@app.route("/tabledetail")
def tabledetail():
    dec = load_decisions()
    t = request.args.get("t", "0")
    if t.isdigit():
        idx = max(0, min(len(TABLE_QUEUE) - 1, int(t)))
        asm = TABLE_QUEUE[idx] if TABLE_QUEUE else None
    else:
        asm = t
        idx = TABLE_QUEUE.index(asm) if asm in TABLE_QUEUE else -1
    if not asm or asm not in ASM_MEMBER_HASHES:
        return jsonify({"empty": True})
    mems = table_members(asm, dec)
    return jsonify({
        "idx": idx, "total": len(TABLE_QUEUE), "asm_id": asm,
        "title": TITLES.get(asm, ""),
        "dochash": asm.split("_")[1] if asm.startswith("asm_") and len(asm.split("_")) > 1 else "",
        "is_problem": asm in PROBLEM_TABLES,
        "members": mems, "sequence": [m["page"] for m in mems],
        "tables_done": sum(1 for a in TABLE_QUEUE if ASM_MEMBER_HASHES.get(a) and all(h in dec for h in ASM_MEMBER_HASHES[a])),
    })


INDEX = """<!doctype html><html><head><meta charset="utf-8"><title>Page-number review</title>
<style>
 *{box-sizing:border-box} body{font:14px -apple-system,Segoe UI,sans-serif;margin:0;background:#111;color:#eee}
 header{position:sticky;top:0;background:#1b1b1f;border-bottom:1px solid #333;padding:10px 16px;display:flex;gap:18px;align-items:center;z-index:5}
 .bar{flex:1;height:8px;background:#333;border-radius:5px;overflow:hidden} .bar i{display:block;height:100%;background:#3ad07a}
 .wrap{display:flex;gap:16px;padding:16px;align-items:flex-start}
 .left{flex:1.6;min-width:0} .right{flex:1;min-width:280px}
 .imgbox{background:#000;border:1px solid #333;border-radius:8px;overflow:auto;max-height:74vh}
 .imgbox img{width:100%;display:block}
 .controls{margin-top:12px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
 input#pn{font-size:30px;width:160px;padding:6px 10px;text-align:center;border-radius:8px;border:2px solid #3ad07a;background:#000;color:#fff}
 button{font-size:15px;padding:9px 14px;border-radius:8px;border:1px solid #444;background:#2a2a30;color:#eee;cursor:pointer}
 button.primary{background:#3ad07a;color:#022;border-color:#3ad07a;font-weight:700}
 button.none{background:#5a2a2a;border-color:#844}
 .hint{color:#9ad;min-height:20px;font-size:15px}
 .card{background:#1b1b1f;border:1px solid #333;border-radius:8px;padding:12px;margin-bottom:12px}
 .seq{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
 .pg{padding:4px 8px;border-radius:6px;background:#26262c;border:1px solid #3a3a42;font-variant-numeric:tabular-nums}
 .pg.cur{background:#3ad07a;color:#022;font-weight:700} .pg.you{border-color:#3ad07a} .pg.none{color:#888}
 .meta{color:#aaa;font-size:12px} a{color:#7bf} .ai{color:#fb7}
 .toggle{font-size:12px;padding:4px 8px}
 kbd{background:#000;border:1px solid #444;border-radius:4px;padding:1px 5px;font-size:11px}
 #suggest{margin:8px 0;font-size:13px}
 .sg{background:#222a33;border:1px solid #3a4a5a;border-radius:8px;padding:8px 10px}
 .sg b{font-size:18px;color:#9fe3ff} .sg .badge{padding:2px 7px;border-radius:10px;font-size:11px;margin-left:6px}
 .badge.fit{background:#1e4d2b;color:#9f9} .badge.conflict{background:#5a2a2a;color:#fbb} .badge.tbl{background:#3a3320;color:#fd9}
 .sg .use{margin-left:8px;font-size:12px;padding:3px 9px;background:#9fe3ff;color:#023;border:none;font-weight:700}
 #probbanner:not(:empty){margin:6px 0;padding:8px 10px;border-radius:8px;background:#3a2030;border:1px solid #7a3a55}
 .rsn{padding:2px 8px;border-radius:10px;font-size:12px;margin-right:6px;font-weight:700}
 .rsn.dup{background:#5a2a2a;color:#fbb} .rsn.null{background:#5a4a20;color:#fe9} .rsn.outlier{background:#3a2a5a;color:#cbf} .rsn.lowconf{background:#444;color:#ddd}
</style></head><body>
<header>
 <b>Page-number review</b>
 <div class="bar"><i id="prog" style="width:0%"></i></div>
 <span id="count" class="meta"></span>
</header>
<div class="wrap">
 <div class="left">
   <div class="meta" id="fileinfo"></div>
   <div id="probbanner"></div>
   <div class="imgbox"><img id="pic" src=""></div>
   <div class="controls">
     <button class="toggle" id="toggle">show full page</button>
     <a class="toggle" id="raw" href="#" target="_blank">open full-res ↗</a>
     <span class="meta">Type the page number (Arabic <kbd>13</kbd> or Roman <kbd>xiii</kbd>), <kbd>Enter</kbd> to save & next.</span>
   </div>
 </div>
 <div class="right">
   <div class="card">
     <div style="display:flex;gap:10px;align-items:center">
       <input id="pn" autocomplete="off" placeholder="">
       <button class="primary" id="savebtn">Save &amp; Next ⏎</button>
       <button class="none" id="nonebtn">No number (n)</button>
       <button id="skipbtn">Skip (→)</button>
       <button id="prevbtn">◀ Prev</button>
     </div>
     <div class="hint" id="hint"></div>
     <div id="suggest"></div>
     <div class="meta" id="aiinfo"></div>
   </div>
   <div class="card">
     <b>Sequence context</b> <span class="meta">(other pages of this assembled table)</span>
     <div id="ctx"></div>
   </div>
 </div>
</div>
<script>
let cur=null, mode='full_small';
const RSN={dup:'duplicate page # in this table',null:'no number read — type it if legible',outlier:'far from the rest of this table',lowconf:'low confidence'};
const $=s=>document.querySelector(s);
function roman(s){const M={m:1000,d:500,c:100,l:50,x:10,v:5,i:1};s=s.trim().toLowerCase();if(!s||[...s].some(c=>!M[c]))return null;let t=0,p=0;for(const c of [...s].reverse()){const v=M[c];t+=v<p?-v:v;p=Math.max(p,v);}return t>0&&t<4000?t:null;}
function preview(){const v=$('#pn').value.trim();if(!v){$('#hint').textContent='';return;}
  if(/^\\d{1,4}$/.test(v)){$('#hint').textContent='= '+parseInt(v)+' (arabic)';return;}
  const r=roman(v);$('#hint').textContent=r?('= '+r+' (roman '+v.toLowerCase()+')'):("can't parse '"+v+"'");}
async function load(i){const r=await fetch('/item'+(i!=null?('?i='+i):''));const d=await r.json();
  if(d.empty){$('#fileinfo').textContent='Queue empty.';return;} cur=d; mode='full_small';
  $('#pic').src='/img/'+d.hash+'?mode=full_small&_='+Date.now();
  $('#raw').href='/img/'+d.hash+'?mode=raw';
  $('#toggle').textContent='show top strip';
  $('#fileinfo').innerHTML='#'+(d.i+1)+' / '+d.total+' &nbsp; <span class="meta">'+d.filename+' · in '+d.n_tables+' table(s)</span>';
  // problem banner
  if(d.problem){ let h='';
    for(const r of d.problem.reasons) h+='<span class="rsn '+r+'">'+r+'</span> <span class="meta">'+(RSN[r]||'')+'</span> ';
    if(d.problem.table_pages) h+='<div class="meta" style="margin-top:4px">this table\\'s pages: ['+d.problem.table_pages.join(', ')+']</div>';
    $('#probbanner').innerHTML=h;
  } else $('#probbanner').innerHTML='';
  $('#count').textContent=d.decided_count+' decided · '+(d.total-d.decided_count)+' left';
  $('#prog').style.width=(100*d.decided_count/d.total)+'%';
  let ai=''; if(d.used_fallback) ai+=' [fallback]';
  $('#aiinfo').innerHTML = (d.ai_read!=null?('<span class="ai">first-pass read: '+d.ai_read+'</span>'):'<span class="ai">first-pass: none (Arabic-only)</span>')+(d.continuation?(' · marker: '+d.continuation):'')+ai;
  // Roman-aware suggestion
  const sg=d.suggestion;
  if(sg){
    let badges='';
    if(sg.conflict) badges+='<span class="badge conflict">⚠ number already used in this table</span>';
    else if(sg.fits) badges+='<span class="badge fit">✓ fits sequence</span>';
    if(sg.table_number) badges+='<span class="badge tbl">title table # = '+sg.table_number+'</span>';
    $('#suggest').innerHTML='<div class="sg">AI (Roman-aware) suggests <b>'+sg.page_number+'</b> '
      +'<span class="meta">('+(sg.numeral||'')+" '"+(sg.raw||'')+"', "+(sg.confidence||'')+')</span>'
      +badges+'<button class="use" id="usebtn">use ⤵</button>'
      +'<div class="meta" style="margin-top:4px">title: '+(sg.title?String(sg.title).replace(/</g,'&lt;'):'—')+'</div></div>';
    document.getElementById('usebtn').onclick=()=>{$('#pn').value=sg.page_number;preview();$('#pn').focus();};
  } else { $('#suggest').innerHTML=''; }
  // pre-fill: prior decision wins, else the suggestion (so you just verify + Enter)
  if(d.decision&&d.decision.page_number!=null) $('#pn').value=d.decision.page_number;
  else if(sg) $('#pn').value=sg.page_number;
  else $('#pn').value='';
  preview(); $('#pn').focus(); $('#pn').select();
  let html='';
  for(const t of d.context){ html+='<div style="margin-top:8px" class="meta">'+t.asm_id+'</div><div class="seq">';
    for(const s of t.seq){ const cls='pg'+(s.is_current?' cur':'')+(s.src==='you'?' you':'')+(s.page==null?' none':'');
      html+='<span class="'+cls+'" title="'+s.hash.slice(0,10)+'">'+(s.page!=null?s.page:'—')+(s.src==='you'?'*':'')+'</span>'; }
    html+='</div>'; }
  $('#ctx').innerHTML=html||'<div class="meta">no context</div>';
}
async function save(none){ if(!cur)return; const raw=$('#pn').value;
  const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({hash:cur.hash,raw:raw,none:!!none})});
  const j=await r.json(); if(!j.ok){$('#hint').textContent=j.error||'error';return;}
  load(cur.i+1); }
$('#pn').addEventListener('input',preview);
$('#pn').addEventListener('keydown',e=>{ if(e.key==='Enter'){e.preventDefault();save(false);} });
$('#savebtn').onclick=()=>save(false);
$('#nonebtn').onclick=()=>save(true);
$('#skipbtn').onclick=()=>{ if(cur)load(cur.i+1); };
$('#prevbtn').onclick=()=>{ if(cur)load(Math.max(0,cur.i-1)); };
$('#toggle').onclick=()=>{ if(!cur)return; mode=(mode==='top'?'full_small':'top'); $('#pic').src='/img/'+cur.hash+'?mode='+mode+'&_='+Date.now(); $('#toggle').textContent=(mode==='top'?'show full page':'show top strip'); };
document.addEventListener('keydown',e=>{ if(e.target.id==='pn')return; if(e.key==='n')save(true); if(e.key==='ArrowRight'&&cur)load(cur.i+1); if(e.key==='ArrowLeft'&&cur)load(Math.max(0,cur.i-1)); });
load(null);
</script></body></html>"""


INDEX_TABLE = """<!doctype html><html><head><meta charset="utf-8"><title>Table-by-table page numbering</title>
<style>
 *{box-sizing:border-box} body{font:14px -apple-system,Segoe UI,sans-serif;margin:0;background:#111;color:#eee}
 header{position:sticky;top:0;background:#1b1b1f;border-bottom:1px solid #333;padding:10px 16px;display:flex;gap:14px;align-items:center;z-index:9;flex-wrap:wrap}
 header b{font-size:15px} .bar{flex:1;min-width:120px;height:8px;background:#333;border-radius:5px;overflow:hidden} .bar i{display:block;height:100%;background:#3ad07a}
 button{font-size:14px;padding:8px 13px;border-radius:8px;border:1px solid #444;background:#2a2a30;color:#eee;cursor:pointer}
 button.primary{background:#3ad07a;color:#022;border-color:#3ad07a;font-weight:700}
 .meta{color:#9aa;font-size:12px}
 #thead{padding:14px 16px 4px} #thead h2{margin:0;font-size:17px}
 .seq{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0}
 .pg{padding:4px 9px;border-radius:6px;background:#26262c;border:1px solid #3a3a42;font-variant-numeric:tabular-nums}
 .pg.dup{background:#5a2a2a;border-color:#a55;color:#fbb} .pg.none{color:#888}
 .badge{padding:2px 8px;border-radius:10px;font-size:11px;margin-left:8px} .badge.prob{background:#5a2a2a;color:#fbb}
 .members{padding:0 16px 80px}
 .mem{display:flex;gap:14px;background:#1b1b1f;border:1px solid #333;border-radius:10px;padding:12px;margin:12px 0}
 .mem.done{border-color:#2e6b44}
 .mem .imgwrap{flex:1.5;min-width:0;background:#000;border:1px solid #333;border-radius:8px;overflow:auto;max-height:62vh}
 .mem .imgwrap img{width:100%;display:block}
 .mem .side{flex:1;min-width:240px}
 .mem input.pn{font-size:28px;width:140px;padding:6px 10px;text-align:center;border-radius:8px;border:2px solid #3ad07a;background:#000;color:#fff}
 .ord{font-size:22px;font-weight:700;color:#3ad07a;margin-right:8px}
 .src{font-size:11px;padding:2px 7px;border-radius:10px;background:#333;margin-left:6px}
 .src.human{background:#1e4d2b;color:#9f9} .src.roman{background:#2a3a55;color:#9cf} .src.firstpass{background:#3a3320;color:#fd9}
 .src.volume{background:#16465c;color:#9fe3ff} .src.section{background:#3a2a55;color:#cbf}
 .hint{color:#9ad;font-size:13px;min-height:18px;margin-top:6px}
 .aihint{color:#fb9;font-size:12px;margin-top:6px}
 .ctl{font-size:11px;padding:3px 8px;margin-right:6px}
 a{color:#7bf} kbd{background:#000;border:1px solid #444;border-radius:4px;padding:1px 5px;font-size:11px}
</style></head><body>
<header>
 <b>Table-by-table</b>
 <button id="prev">◀ table</button>
 <span id="pos" class="meta"></span>
 <button id="next">table ▶</button>
 <button id="nextun">next unfinished ▸</button>
 <div class="bar"><i id="prog"></i></div>
 <span id="done" class="meta"></span>
 <input id="jump" placeholder="jump to table #" style="width:120px;padding:6px;border-radius:6px;border:1px solid #444;background:#000;color:#eee">
 <a href="/scan" class="meta">single-scan mode →</a>
</header>
<div id="thead"></div>
<div class="seq" id="seq" style="padding:0 16px"></div>
<div class="members" id="members"></div>
<script>
let T=null, idx=0;
const $=s=>document.querySelector(s);
function roman(s){const M={m:1000,d:500,c:100,l:50,x:10,v:5,i:1};s=(s||'').trim().toLowerCase();if(!s||[...s].some(c=>!M[c]))return null;let t=0,p=0;for(const c of [...s].reverse()){const v=M[c];t+=v<p?-v:v;p=Math.max(p,v);}return t>0&&t<4000?t:null;}
function parse(v){v=(v||'').trim().replace(/[.\\[\\]() ]/g,'');if(v==='')return null;if(/^\\d{1,4}$/.test(v))return parseInt(v);return roman(v);}
function seqHTML(mems){const vals=mems.map(m=>m.page);const counts={};vals.forEach(v=>{if(v!=null)counts[v]=(counts[v]||0)+1;});
  return mems.map(m=>{const dup=m.page!=null&&counts[m.page]>1;return '<span class="pg'+(dup?' dup':'')+(m.page==null?' none':'')+'">'+(m.page!=null?m.page:'—')+'</span>';}).join(' ');}
async function loadTable(i){const r=await fetch('/tabledetail?t='+i);const d=await r.json();
  if(d.empty){$('#thead').innerHTML='No tables in queue.';return;} T=d; idx=d.idx;
  $('#pos').textContent='table '+(d.idx+1)+' / '+d.total;
  $('#done').textContent=d.tables_done+' tables fully done';
  $('#prog').style.width=(100*d.tables_done/d.total)+'%';
  $('#thead').innerHTML='<h2>'+(d.title?d.title.replace(/</g,'&lt;'):'(no title)')+(d.is_problem?'<span class="badge prob">needs attention</span>':'')+'</h2>'
    +'<div class="meta">'+d.asm_id+' · doc '+d.dochash+' · '+d.members.length+' pages</div>';
  renderSeq();
  $('#members').innerHTML=d.members.map((m,k)=>memHTML(m,k)).join('');
  d.members.forEach((m,k)=>wire(m,k));
}
function renderSeq(){$('#seq').innerHTML='<span class="meta" style="margin-right:6px">current order:</span>'+seqHTML(T.members);}
function memHTML(m,k){
  let ai='';
  if(m.section_page!=null) ai='volume page (bottom Arabic) = '+m.page+' · section (Roman top) = '+m.section_page;
  else if(m.ai_raw) ai='AI read: '+m.ai_raw+' ('+(m.ai_numeral||'')+')';
  let extra=[];
  if(m.top_raw||m.bottom_raw) extra.push('top='+(m.top_raw||'—')+' bottom='+(m.bottom_raw||'—'));
  if(m.ai_table_no) extra.push('title table #='+m.ai_table_no); if(m.continuation) extra.push('marker: '+m.continuation);
  if(m.ai_title) extra.push('“'+String(m.ai_title).slice(0,48).replace(/</g,'&lt;')+'”');
  return '<div class="mem'+(m.decided?' done':'')+'" id="mem'+k+'">'
   +'<div class="imgwrap"><img loading="lazy" id="img'+k+'" src="/img/'+m.hash+'?mode=full_small"></div>'
   +'<div class="side">'
   +'<div><span class="ord" id="ord'+k+'">'+(m.page!=null?m.page:'—')+'</span>'
   +'<span class="src '+(m.source||'')+'" id="src'+k+'">'+(m.source||'unset')+'</span></div>'
   +'<div style="margin-top:8px"><input class="pn" id="pn'+k+'" autocomplete="off" value="'+(m.page!=null?m.page:'')+'">'
   +' <button class="none" data-k="'+k+'" id="none'+k+'">no #</button></div>'
   +'<div class="hint" id="hint'+k+'"></div>'
   +'<div class="aihint">'+ai+(extra.length?(' · '+extra.join(' · ')):'')+'</div>'
   +'<div style="margin-top:8px"><button class="ctl" data-k="'+k+'" data-m="top" >top strip</button>'
   +'<button class="ctl" data-k="'+k+'" data-m="full_small">full page</button>'
   +'<a class="ctl" href="/img/'+m.hash+'?mode=raw" target="_blank">full-res ↗</a></div>'
   +'<div class="meta" style="margin-top:6px">'+m.filename+'</div>'
   +'</div></div>';
}
function wire(m,k){
  const inp=document.getElementById('pn'+k);
  const prev=()=>{const v=parse(inp.value);document.getElementById('hint'+k).textContent=inp.value.trim()===''?'':(v!=null?('= '+v):("can't parse '"+inp.value.trim()+"'"));};
  inp.addEventListener('input',prev); prev();
  const save=async(none)=>{const raw=inp.value;
    const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({hash:m.hash,raw:raw,none:!!none})});
    const j=await r.json(); if(!j.ok){document.getElementById('hint'+k).textContent=j.error||'error';return;}
    m.page=j.page_number; m.source='human'; m.decided=true;
    document.getElementById('ord'+k).textContent=(j.page_number!=null?j.page_number:'—');
    const s=document.getElementById('src'+k); s.textContent='human'; s.className='src human';
    document.getElementById('mem'+k).classList.add('done'); renderSeq();
  };
  inp.addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();save(false);const n=document.getElementById('pn'+(k+1));if(n){n.focus();n.select();}}});
  inp.addEventListener('blur',()=>{if(inp.value.trim()!==''&&parse(inp.value)!=null)save(false);});
  document.getElementById('none'+k).onclick=()=>save(true);
}
document.addEventListener('click',e=>{const b=e.target.closest('.ctl[data-m]'); if(b){const k=b.dataset.k;document.getElementById('img'+k).src='/img/'+T.members[k].hash+'?mode='+b.dataset.m+'&_='+Date.now();}});
$('#prev').onclick=()=>loadTable(Math.max(0,idx-1));
$('#next').onclick=()=>loadTable(idx+1);
$('#nextun').onclick=async()=>{const r=await fetch('/tables');const d=await r.json();
  let start=idx+1;for(let s=0;s<d.total;s++){const j=(start+s)%d.total;if(!d.tables[j].all_done){loadTable(j);return;}}alert('All tables done!');};
$('#jump').addEventListener('keydown',e=>{if(e.key==='Enter'){const n=parseInt(e.target.value);if(n>=1)loadTable(n-1);}});
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;if(e.key===']')loadTable(idx+1);if(e.key==='[')loadTable(Math.max(0,idx-1));});
loadTable(0);
</script></body></html>"""


@app.route("/")
def index():
    return INDEX_TABLE


@app.route("/scan")
def scan_index():
    return INDEX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="problems", choices=["null", "all", "fallback", "problems"])
    ap.add_argument("--tables", default="attention", choices=["attention", "problems", "all"],
                    help="table-mode queue: attention=problem tables first then all multi-page; problems=only flagged; all=everything")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    build_state(args.queue, args.tables)
    print(f"scan-queue='{args.queue}' ({len(QUEUE)})  table-queue='{args.tables}' ({len(TABLE_QUEUE)} tables)")
    print(f"decisions -> {os.path.relpath(DECISIONS, ROOT)}")
    print(f"open http://{args.host}:{args.port}   (table-by-table; single-scan at /scan)")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
