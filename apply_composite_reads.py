#!/usr/bin/env python3
"""Apply composite re-stitch reads, KEEPING THE BEST version per table: apply a read whenever the merged
result has FEWER fail-rows than what's currently in the corpus — even if the table was already re-stitched
(a later, better redo wins). Records metadata.composite_fails so future redos compare against the best-so-far."""
import sys, json, glob, shutil, copy, os; sys.argv=["x"]
sys.path.insert(0,".")
import census_review_app as A
applied=0; improved=[]
for rf in glob.glob("export/composite_reads/*.json"):
    r=json.load(open(rf))
    if r.get("verdict")!="fixed": continue
    lih=r["left_ih"]
    hits=glob.glob(f"export/ocr_results_referenced/digitized_tables/*/{lih}_oneshot.json")
    if not hits: continue
    lp=hits[0]; d=json.load(open(lp))
    current=A.table_fail_rows(d)                      # best achieved so far lives in the live corpus
    nd=copy.deepcopy(d)
    ok=True
    for si,sec in enumerate(r["sections"]):
        if si>=len(nd["data"]): ok=False; break
        old=nd["data"][si]["rows"]
        rh=next((k for k in old[0] if not isinstance(old[0].get(k),dict)),None) if old else None
        new=sec["rows"]
        for i,nrow in enumerate(new):
            if i>=len(old): break
            rebuilt={rh:old[i].get(rh,"")}
            for g,v in nrow.items():
                if isinstance(v,dict): rebuilt[g]=v
            old[i]=rebuilt
        nd["data"][si]["rows"]=old[:len(new)]
    if not ok: continue
    if r.get("data_type")=="proportional": nd.setdefault("metadata",{})["data_type"]="proportional"
    after=A.table_fail_rows(nd)
    # KEEP THE BEST: apply only if this redo is strictly better than the current corpus version
    if after<current:
        if not os.path.exists(lp.replace("_oneshot.json","_oneshot.precomposite.json")):
            shutil.copy(lp, lp.replace("_oneshot.json","_oneshot.precomposite.json"))
        md=nd.setdefault("metadata",{})
        md["composite_restitched"]=True; md["composite_fails"]=after
        json.dump(nd, open(lp,"w")); applied+=1
        improved.append((lih[:8],current,after))
print(f"kept-best composite applies: {applied}")
for x in sorted(improved,key=lambda x:x[1]-x[2],reverse=True)[:20]: print(f"  {x[0]}: {x[1]} -> {x[2]} fails")
