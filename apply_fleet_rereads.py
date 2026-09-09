#!/usr/bin/env python3
"""Apply fleet *_full.json rereads with NORMALIZED group/section matching (trailing periods, case,
spacing differ between agent reads and corpus keys). Gated: layout-aware fail-rows must not increase."""
import sys, json, glob, copy, shutil, re; sys.argv=["x"]
import census_review_app as A
def norm(s): return re.sub(r"[\s.·]+"," ",str(s)).strip().lower()
tot=cells=0
for rf in glob.glob("export/agent_rereads/*_full.json"):
    r=json.load(open(rf))
    if r.get("verdict")!="fixed": continue
    ih=r["imghash"]
    hits=glob.glob(f"export/ocr_results_referenced/digitized_tables/*/{ih}_oneshot.json")
    if not hits: continue
    f=hits[0]; d=json.load(open(f))
    if (d.get("metadata",{}) or {}).get("targeted_reread_user_note"): continue
    before=A.table_fail_rows(d)
    # normalized section lookup; corpus sections may be unnamed ("") -> fall back to ORDER
    rsecs=list(r.get("sections",{}).items())
    nsec={norm(k):v for k,v in rsecs}
    nd=copy.deepcopy(d); n=0
    for si,s in enumerate(nd["data"]):
        new=nsec.get(norm(s.get("name","")))
        if new is None and len(rsecs)==len(nd["data"]):
            new=rsecs[si][1]                    # positional fallback (same section count)
        if not new: continue
        for old,nr in zip(s["rows"],new):
            if not isinstance(nr,dict): continue
            ngr={norm(k):v for k,v in nr.items() if isinstance(v,dict)}
            if not ngr:
                # FLAT read row: keys are "Group. Subkey" ("15 to 19. Males.": 54867) or joined names.
                # Split each flat key against the corpus row's group names (longest match), then subkey.
                groups=[(g,norm(g)) for g,v in old.items() if isinstance(v,dict)]
                groups.sort(key=lambda t:-len(t[1]))
                matched=0
                for fk,x in nr.items():
                    if not isinstance(x,(int,float)): continue
                    nfk=norm(fk)
                    for g,ng in groups:
                        if ng and nfk.startswith(ng):
                            rest=nfk[len(ng):].strip()
                            sk=next((k2 for k2 in old[g] if norm(k2)==rest or norm(k2) in rest or rest in norm(k2)),None)
                            if sk is not None:
                                old[g][sk]=x; n+=1; matched+=1
                            break
                if not matched:
                    # last resort: positional (gated anyway)
                    vals=[x for k,x in nr.items() if isinstance(x,(int,float))]
                    slots=[(g,k2) for g,v in old.items() if isinstance(v,dict) for k2 in v]
                    if vals and len(vals)==len(slots):
                        for (g,k2),x in zip(slots,vals):
                            old[g][k2]=x; n+=1
                # derive persons where flat read gave only M+F
                for g,v in old.items():
                    if not isinstance(v,dict): continue
                    pk=next((k for k in v if re.search(r"person|both",k,re.I)),None)
                    mk=next((k for k in v if re.search(r"male",k,re.I) and not re.search(r"female",k,re.I)),None)
                    fk2=next((k for k in v if re.search(r"female",k,re.I)),None)
                    if pk and mk and fk2 and matched:
                        provided=any(norm(pk) in norm(kk) for kk in nr)
                        if not provided and isinstance(v.get(mk),(int,float)) and isinstance(v.get(fk2),(int,float)):
                            v[pk]=v[mk]+v[fk2]
                continue
            for g,v in old.items():
                if not isinstance(v,dict): continue
                nv=ngr.get(norm(g))
                if not isinstance(nv,dict): continue
                nsub={norm(k):x for k,x in nv.items()}
                touched=False
                for k2 in v:
                    x=nsub.get(norm(k2))
                    if x is not None: v[k2]=x; n+=1; touched=True
                # read gave males/females but not persons -> persons is derived; recompute it
                if touched:
                    pk=next((k for k in v if re.search(r"person|total|both",k,re.I)),None)
                    mk=next((k for k in v if re.search(r"male",k,re.I) and not re.search(r"female",k,re.I)),None)
                    fk=next((k for k in v if re.search(r"female",k,re.I)),None)
                    if pk and mk and fk and nsub.get(norm(pk)) is None:
                        try:
                            m_,f_=v.get(mk),v.get(fk)
                            if isinstance(m_,(int,float)) and isinstance(f_,(int,float)):
                                v[pk]=m_+f_
                        except Exception: pass
    after=A.table_fail_rows(nd)
    if n and after<=before:
        shutil.copy(f, f.replace("_oneshot.json","_oneshot.prefloodfix.json"))
        nd.setdefault("metadata",{})["targeted_reread_user_note"]=True
        json.dump(nd, open(f,"w")); tot+=1; cells+=n
        print(f"{ih[:8]}: fail-rows {before}->{after} · {n} cells")
    elif n:
        print(f"{ih[:8]}: GATED OUT ({before}->{after})")
print(f"applied {tot} pages · {cells} cells")
