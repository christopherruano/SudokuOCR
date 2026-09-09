#!/usr/bin/env python3
"""Apply subagent re-read corrections (export/agent_rereads/*.json) to the corpus, GATED: standard
constraint-fails must not increase. Backup *_oneshot.prereread.json; marks metadata.agent_reread."""
import sys, json, glob, shutil, copy; sys.argv=["x"]
import census_review_app as A
applied=cells=skipped=0
for rf in glob.glob("export/agent_rereads/*.json"):
    r=json.load(open(rf)); ih=r.get("imghash",""); cor=r.get("corrections") or []
    if not cor: continue
    hits=glob.glob(f"export/ocr_results_referenced/digitized_tables/*/{ih}_oneshot.json")
    if not hits: continue
    f=hits[0]; d=json.load(open(f))
    if (d.get("metadata",{}) or {}).get("agent_reread"): continue     # already applied
    before=sum(len(A.check_constraints(s.get("rows",[]))) for s in d.get("data",[]))
    nd=copy.deepcopy(d); n=0
    for c in cor:
        try:
            row=nd["data"][c["sec_idx"]]["rows"][c["row_idx"]]
            if c["group"] in row and isinstance(row[c["group"]],dict):
                row[c["group"]][c["subkey"]]=c["new"]; n+=1
        except Exception: pass
    after=sum(len(A.check_constraints(s.get("rows",[]))) for s in nd.get("data",[]))
    if n and after<=before:
        shutil.copy(f, f.replace("_oneshot.json","_oneshot.prereread.json"))
        nd.setdefault("metadata",{})["agent_reread"]=True
        json.dump(nd, open(f,"w")); applied+=1; cells+=n
    else: skipped+=1
print(f"agent rereads applied: {applied} tables · {cells} cells · gated-out {skipped}")
