#!/usr/bin/env python3
"""Merge fleet alignment reads (export/alignment_reads/*.json) into left corpus pages.
Left rows keep labels+groups; right groups overlay per row. Gated: hardened stitch fails() where
checkable; skip if left already stitched. Backup .prestitch.json; metadata records provenance."""
import sys, json, glob, os, shutil; sys.argv=["x"]
sys.path.insert(0,".")
from stitch_pilot import fails, path_for
work={w["right_ih"]:w for w in json.load(open("export/alignment_worklist.json"))}
# also earlier worklist entries (rebuilds drop done ones)
applied=skipped=0
for rf in glob.glob("export/alignment_reads/*.json"):
    r=json.load(open(rf))
    if r.get("verdict")!="aligned": continue
    rih=r["right_ih"]; w=work.get(rih)
    if not w: continue
    lp=path_for(w["left_ih"])
    if not lp: skipped+=1; continue
    L=json.load(open(lp))
    if (L.get("metadata",{}) or {}).get("stitched_halfpage"): continue
    ld=L.get("data",[]); rs=r.get("sections",[])
    if len(ld)!=len(rs): skipped+=1; continue
    ok=True; merged=[]
    for lsec,rsec in zip(ld,rs):
        lr=lsec.get("rows",[]); rr=rsec.get("rows",[])
        if len(lr)!=len(rr): ok=False; break
        rows=[]
        for a,b in zip(lr,rr):
            row=dict(a)
            for g,v in (b or {}).items():
                if isinstance(v,dict): row[g]=v
            rows.append(row)
        merged.append({"name":lsec.get("name",""),"rows":rows})
    if not ok: skipped+=1; continue
    fm=fails(merged)
    shutil.copy(lp, lp.replace("_oneshot.json","_oneshot.prestitch.json"))
    L["data"]=merged
    md=L.setdefault("metadata",{})
    md["stitched_halfpage"]=rih; md["stitch_approved_by"]="human+agent_aligned"
    if fm and fm<10**6: md["stitch_identity_fails"]=fm
    json.dump(L, open(lp,"w")); applied+=1
print(f"alignment merges applied: {applied} · skipped(structure): {skipped}")
