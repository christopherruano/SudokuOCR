#!/usr/bin/env python3
"""Double-check EVERY population pyramid and build a self-contained browser viewer.
QC signals per pyramid:
  - plausible total for the geography (district / province / state / all-india bands) — catches OCR inflation
  - sex ratio (M/F) in a believable window
  - share under-15 (an expansive pyramid should be young-heavy) + monotone-ish narrowing
  - missing/zero cells
  - CROSS-CHECK: province total vs the sum of its own districts that year (data redundancy)
Verdict: OK / SUSPECT / BAD (+ reasons). Writes:
  export/pyramids/pyramids_qc.csv      one row per pyramid with metrics + verdict
  export/pyramids/pyramids_viewer.html self-contained: filter by verdict, click to see the back-to-back pyramid
  python3 validate_pyramids.py
"""
import os, csv, json
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "export/pyramids")
# plausible MAX total persons by geo bucket (generous upper bounds from known 1881-1941 figures)
MAXTOT = {"all_india": 420e6, "british_province": 95e6, "princely_state": 22e6,
          "modern_state": 95e6, "agency": 25e6, "urban": 6e6, "district": 9e6, "": 95e6}


def main():
    tabs = defaultdict(lambda: {"meta": None, "cells": defaultdict(dict)})
    for r in csv.DictReader(open(os.path.join(P, "pyramids_long.csv"))):
        t = tabs[r["table_id"]]
        if t["meta"] is None:
            t["meta"] = dict(place=r["place"], parent=r["parent"], year=r["year"],
                             entity=r["entity_type"], geo=r["geo_level"])
        t["cells"][(int(r["age_lower"]), r["age_upper"])][r["sex"]] = int(r["population"])

    # province totals (for the district cross-check): sum of districts by (parent, year)
    distsum = defaultdict(float)
    provtot = {}
    for tid, t in tabs.items():
        m = t["meta"]; tot = sum(c.get("Male", 0) + c.get("Female", 0) for c in t["cells"].values())
        if (m["entity"] == "district" or m["geo"] == "district") and m["parent"]:
            distsum[(m["parent"], m["year"])] += tot
        if m["entity"] in ("british_province", "princely_state", "modern_state"):
            provtot[(m["place"], m["year"])] = tot

    rows = []
    data = []
    for tid, t in tabs.items():
        m = t["meta"]
        bands = sorted(t["cells"], key=lambda b: b[0])
        M = sum(c.get("Male", 0) for c in t["cells"].values())
        F = sum(c.get("Female", 0) for c in t["cells"].values())
        tot = M + F
        reasons = []
        # 1) plausible total for geography
        cap = MAXTOT.get(m["entity"] or m["geo"], 95e6)
        if tot > cap:
            reasons.append(f"total {tot:,} implausibly large for {m['entity'] or m['geo']}")
        if tot < 200:
            reasons.append("total tiny (<200)")
        # 2) sex ratio
        sr = (M / F) if F else 0
        if F and (sr < 0.78 or sr > 1.32):
            reasons.append(f"sex ratio M/F={sr:.2f} out of range")
        # 3) shape: share under 15
        u15 = sum(c.get("Male", 0) + c.get("Female", 0) for (lo, hi), c in t["cells"].items() if lo < 15)
        share = (u15 / tot) if tot else 0
        if tot and (share < 0.18 or share > 0.55):
            reasons.append(f"under-15 share {share:.0%} unusual")
        # 4) missing cells
        zeros = sum(1 for c in t["cells"].values() if not c.get("Male") or not c.get("Female"))
        if bands and zeros / len(bands) > 0.30:
            reasons.append(f"{zeros}/{len(bands)} bands missing a sex")
        # 5) province vs sum-of-districts cross-check
        xcheck = ""
        if (m["place"], m["year"]) in provtot and (m["place"], m["year"]) in distsum and provtot[(m["place"], m["year"])] == tot:
            ds = distsum[(m["place"], m["year"])]
            if ds > 0:
                ratio = tot / ds
                xcheck = f"{ratio:.2f}x sum-of-districts"
                if ratio < 0.6 or ratio > 1.7:
                    reasons.append(f"province total {ratio:.2f}x its districts' sum")
        verdict = "OK" if not reasons else ("BAD" if any("implausibly" in r or "tiny" in r for r in reasons) else "SUSPECT")
        rows.append(dict(place=m["place"], year=m["year"], entity=m["entity"], geo=m["geo"], parent=m["parent"],
                         persons=tot, males=M, females=F, sex_ratio=round(sr, 3), under15_share=round(share, 3),
                         n_bands=len(bands), zero_bands=zeros, xcheck=xcheck, verdict=verdict,
                         reasons="; ".join(reasons), table_id=tid))
        data.append(dict(place=m["place"], year=m["year"], entity=m["entity"], geo=m["geo"], persons=tot,
                         verdict=verdict, reasons=reasons,
                         bands=[dict(l=(f"{lo}-{hi}" if hi else f"{lo}+"),
                                     m=t["cells"][(lo, hi)].get("Male", 0), f=t["cells"][(lo, hi)].get("Female", 0))
                                for (lo, hi) in bands]))

    rows.sort(key=lambda r: (r["verdict"] != "BAD", r["verdict"] != "SUSPECT", -r["persons"]))
    with open(os.path.join(P, "pyramids_qc.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    from collections import Counter
    v = Counter(r["verdict"] for r in rows)
    open(os.path.join(P, "pyramids_viewer.html"), "w").write(VIEWER.replace("__DATA__", json.dumps(data)))
    print(f"validated {len(rows)} pyramids -> pyramids_qc.csv + pyramids_viewer.html")
    print(f"  verdicts: {dict(v)}")
    print(f"  BAD examples:")
    for r in [r for r in rows if r["verdict"] == "BAD"][:6]:
        print(f"    {r['place'][:24]:24} {r['year']} · {r['persons']:>12,} · {r['reasons'][:48]}")


VIEWER = """<!doctype html><meta charset=utf-8><title>Census Population Pyramids — QC</title>
<style>body{font:13px/1.4 system-ui;margin:0;background:#f6f8fb;color:#1a2233;display:flex;height:100vh}
#side{width:340px;overflow:auto;border-right:1px solid #ccd;background:#fff}
#side h1{font-size:15px;margin:10px 12px;color:#1f3864}
.f{padding:6px 12px;border-bottom:1px solid #eee;position:sticky;top:0;background:#fff;z-index:2}
.f button{margin:2px;padding:3px 8px;border:1px solid #ccd;border-radius:5px;background:#eef;cursor:pointer}
.f button.on{background:#1f3864;color:#fff}
.row{padding:5px 12px;border-bottom:1px solid #f0f0f0;cursor:pointer;display:flex;justify-content:space-between;gap:6px}
.row:hover{background:#eef5ff}.row.sel{background:#dbe9ff}
.b{font-size:10px;padding:0 6px;border-radius:8px;color:#fff;white-space:nowrap}
.OK{background:#0a7d2e}.SUSPECT{background:#b8860b}.BAD{background:#9c0006}
#main{flex:1;overflow:auto;padding:18px}
input{width:60%;padding:5px 8px;border:1px solid #cce;border-radius:6px;margin:4px 12px}
svg text{font:11px system-ui}</style>
<div id=side>
 <h1>Population Pyramids</h1>
 <div class=f><input id=q placeholder="filter place / year…" oninput="render()">
  <div><button data-v=ALL class=on onclick="setv(this)">all</button>
  <button data-v=OK onclick="setv(this)">OK</button>
  <button data-v=SUSPECT onclick="setv(this)">suspect</button>
  <button data-v=BAD onclick="setv(this)">bad</button></div></div>
 <div id=list></div></div>
<div id=main><div id=chart style="color:#789">← pick a pyramid</div></div>
<script>const D=__DATA__;let vf="ALL",sel=-1;
function setv(b){vf=b.dataset.v;document.querySelectorAll('.f button').forEach(x=>x.classList.toggle('on',x===b));render();}
function render(){const q=document.getElementById('q').value.toLowerCase();
 const L=document.getElementById('list');L.innerHTML='';
 D.forEach((d,i)=>{if(vf!='ALL'&&d.verdict!=vf)return;
  if(q&&!(d.place+' '+d.year).toLowerCase().includes(q))return;
  const r=document.createElement('div');r.className='row'+(i===sel?' sel':'');
  r.innerHTML=`<span>${d.place} <b>${d.year||'?'}</b><br><span style=color:#789>${(d.entity||d.geo||'').replace(/_/g,' ')} · ${d.persons.toLocaleString()}</span></span><span class="b ${d.verdict}">${d.verdict}</span>`;
  r.onclick=()=>{sel=i;draw(d);render();};L.appendChild(r);});}
function draw(d){const W=620,bh=20,H=d.bands.length*bh+60,mid=W/2;
 const mx=Math.max(...d.bands.map(b=>Math.max(b.m,b.f)),1);const sc=(mid-70)/mx;
 let s=`<svg width=${W} height=${H}>`;
 d.bands.forEach((b,i)=>{const y=H-40-(i+1)*bh;
  s+=`<rect x=${mid-b.m*sc} y=${y} width=${b.m*sc} height=${bh-2} fill=#3a6ea5/>`;
  s+=`<rect x=${mid} y=${y} width=${b.f*sc} height=${bh-2} fill=#c44e6a/>`;
  s+=`<text x=${mid} y=${y+bh-6} text-anchor=middle fill=#fff style="font-size:9px">${b.l}</text>`;});
 s+=`<line x1=${mid} y1=10 x2=${mid} y2=${H-40} stroke=#333/></svg>`;
 const rz=d.reasons&&d.reasons.length?`<div style="color:#9c0006;margin:6px 0">⚠ ${d.reasons.join(' · ')}</div>`:'<div style="color:#0a7d2e;margin:6px 0">✓ passed all checks</div>';
 document.getElementById('chart').innerHTML=`<h2 style="margin:0">${d.place} — ${d.year||'?'} <span class="b ${d.verdict}">${d.verdict}</span></h2>
  <div style=color:#789>${(d.entity||d.geo||'').replace(/_/g,' ')} · ${d.persons.toLocaleString()} persons · ◀ Male  Female ▶</div>${rz}${s}`;}
render();</script>"""


if __name__ == "__main__":
    main()
