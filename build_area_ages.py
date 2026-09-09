#!/usr/bin/env python3
"""Per-AREA age dataset + graphs: every area (province/district/taluk/city/state) × census year gets its
closest-to-an-age-table surfaced: age-band rows with male/female (and persons) values, rendered as
client-side population-pyramid SVGs in areas.html. Emits ONE json + ONE html + ONE zip (file-count safe)."""
import csv, glob, re, json, os, io, zipfile, sys
sys.path.insert(0, ".")
from canonical_places import canonicalize

ROOT = os.path.dirname(os.path.abspath(__file__))
AGE = re.compile(r"age|table vii|civil condition|by age", re.I)
AGELBL = re.compile(r"^\s*(under\s*\d|\d+\s*(-|to|–|—)\s*\d+|\d+\s*(year|and (over|upwards|above))|total|all ages|\d+\+?)\s*(years?|months?)?\s*[.\s]*$", re.I)
CENSUS = {"1871","1872","1881","1891","1901","1911","1921","1931","1941","1951","1961"}
PCTK = re.compile(r"percent|per\s*cent|proportion|per\s*mille|ratio|rate", re.I)
AREAH = re.compile(r"district|province|state|division|section|countr|taluk|tahsil|town|city|circle|pergunnah|village|place", re.I)
RELCOND = (r"hindu|musl|mahomedan|muhammadan|mahammedan|musalman|mussalman|christian|sikh|jain|buddh|parsi|brahmo|\bjew\b"
    r"|animist|worship|aborig|religion|caste"
    r"|\bsingle\b|married|unmarried|widow|divorced|civil condition"
    r"|blind|insane|deaf|leper|infirm|elephantiasis|\bbirths?\b|\bdeaths?\b"
    r"|\bworkers?\b|cultivat|\bcommerce\b|\btransport\b|\bservices?\b|\bmining\b|quarrying|dependant|occupation"
    r"|smoothed|arith|unadjusted|\bhalf\b|all ages|^(under|below|over|above)\b"
    r"|jamaica|england|wales|france|prussia|austria|germany|italy|ireland|scotland|sweden|norway|denmark|america|australia|japan|ceylon|europe")
NOTPLACE = re.compile(
    r"\bmales?\b|\bfemales?\b|persons|total|popu|census|life|table|famine|average|percent|proportion"
    r"|\btribal\b|\baryas?\b|\bjats\b|\bbhils\b|\brajputs\b"
    r"|"+RELCOND+
    r"|unspecified|not stated|rural|urban|^row\b|\d", re.I)
BADBLOCK = re.compile(r"unspecified|not stated|total|rural|urban|\bmales?\b|\bfemales?\b|persons|"+RELCOND, re.I)
JUNKNAME = re.compile(r"^(east|west|north|south|central|northern|southern|eastern|western|plain|hilly|inland|mountainous|sub-montane|sea-coast|actual|others?|province|state|total|upper|lower|middle|english|arabic|gujarati|malayalee|non-malayalee|hindi|urdu)[.,\s]*$", re.I)

def num(x):
    if isinstance(x,(int,float)): return x
    try: return float(str(x).replace(",",""))
    except: return None


def _agev(l):
    l=str(l).strip().rstrip('.').lower()
    m=re.match(r"^(?:under\s*)?(\d+)",l)
    if m: return int(m.group(1))
    if re.match(r"^infant",l): return 0
    return None

def _agekey(k):
    """Age value if a column-group KEY is a pure age band ('0-9.','10 to 12 Years.','60 AND UPWARDS.',
    'Under 1 year.','-30.'), else None. Rejects totals/unspecified/percentages/year-numbers."""
    l=str(k).strip().lower()
    if not re.search(r"\d",l): return None
    if re.search(r"total|all ages|unspecified|not stated|popul|census|percent|proportion",l): return None
    rem=re.sub(r"\b(years?|months?|days?|to|and|over|under|above|upwards|exceeding)\b|[^a-z]"," ",l)
    if rem.strip(): return None
    v=int(re.search(r"(\d+)",l).group(1))
    return v if v<130 else None

def _sexkeys(v):
    """(male_keys, female_keys, persons_key) of a group dict, ignoring percentage-ish keys.
    Multiple male/female tiers (Adults+Minors) are kept for summing unless a Total tier exists."""
    mks=[k for k in v if re.search(r"male",k,re.I) and not re.search(r"female",k,re.I) and not PCTK.search(k)]
    fks=[k for k in v if re.search(r"female",k,re.I) and not PCTK.search(k)]
    tm=[k for k in mks if re.search(r"total",k,re.I)]
    tf=[k for k in fks if re.search(r"total",k,re.I)]
    if tm: mks=tm[:1]
    if tf: fks=tf[:1]
    pk=next((k for k in v if re.search(r"person|both",k,re.I) and not re.search(r"male",k,re.I) and not PCTK.search(k)),None)
    return mks,fks,pk

def _sexval(v,keys):
    vals=[num(v.get(k)) for k in keys]
    vals=[x for x in vals if x is not None]
    return sum(vals) if vals else None

def _transposed(s,rh,rows):
    """Rows = areas (or castes etc.), column groups = age bands with male/female subkeys — the dominant
    1872/1881 layout. Returns (good, series, areaish); areaish series = concatenated [area-header]+age-rows
    blocks (split downstream), else just the table's Total row as one series."""
    if re.search(r"single|married|widow|unspecified|deaf|blind|insane|infirm|leper",s.get("name","") or "",re.I):
        return None                                    # per-condition / infirmity slice, not a population pyramid
    r0=rows[0]
    ag=[k for k in r0 if isinstance(r0.get(k),dict) and _agekey(k) is not None]
    if len(ag)<5: return None
    mks=fks=pk=None
    for k in ag:
        mm,ff,pp=_sexkeys(r0[k])
        if mm and ff: mks,fks,pk=mm,ff,pp; break
    if not mks: return None
    areaish=bool(AREAH.search(rh or ""))
    # merge 'X - Males' / 'X - Females' row pairs into one area row
    def _base(l): return re.sub(r"[\s.,\-–—]*(males?|females?)[\s.,]*$","",l,flags=re.I).strip(" .-—–,")
    rws=[]; i=0
    while i<len(rows):
        l=str(rows[i].get(rh,"")).strip()
        if re.search(r"males?[\s.,]*$",l,re.I) and i+1<len(rows):
            l2=str(rows[i+1].get(rh,"")).strip()
            if re.search(r"females?[\s.,]*$",l2,re.I) and _base(l)==_base(l2) and _base(l):
                rws.append((_base(l),rows[i],rows[i+1])); i+=2; continue
        rws.append((l,rows[i],rows[i])); i+=1
    out=[]; totgood=0
    for name,rm,rf in rws:
        if not areaish and not re.match(r"total",name,re.I): continue
        vals=[]
        for k in ag:
            vm,vf=rm.get(k),rf.get(k)
            m=_sexval(vm,mks) if isinstance(vm,dict) else None
            f=_sexval(vf,fks) if isinstance(vf,dict) else None
            if m is None and f is None: continue
            p=num(vm.get(pk)) if (pk and isinstance(vm,dict)) else None
            if p is None and m is not None and f is not None: p=m+f
            vals.append([str(k).strip().rstrip('.'),m,f,p])
        good=sum(1 for x in vals if (x[1] or 0)>0 and (x[2] or 0)>0)
        if good<6: continue
        _chk=[x for x in vals if x[3] is not None and x[1] is not None and x[2] is not None]
        _bad=sum(1 for x in _chk if abs((x[1]+x[2])-x[3])>max(2,0.02*abs(x[3])))
        if _chk and _bad/len(_chk)>0.3: continue       # persons != M+F -> mislabeled columns
        tots=[(x[1] or 0)+(x[2] or 0) for x in vals]
        if sum(tots)<=0 or max(tots)/sum(tots)>0.6: continue   # one band dominating = not an age spread
        if areaish:
            if len(re.sub(r"[^A-Za-z]","",name))<4 or NOTPLACE.search(name) or JUNKNAME.match(name): continue
            out.append([name,None,None,None])
        out+=vals; totgood+=good
        if not areaish: break
    if not out or totgood<6: return None
    return totgood,out,areaish

def split_blocks(series):
    """Split a concatenated multi-district series into named blocks. A row whose label has >=4 letters and
    is not an age label starts a new block (its label = the block/district name)."""
    blocks=[]; cur=None; curname=None
    for row in series:
        l=str(row[0])
        if _agev(l) is None and len(re.sub(r"[^A-Za-z]","",l))>=4 and not re.search(r"total\s*[.\d]*$",l.strip(),re.I) or \
           (_agev(l) is None and re.search(r"[A-Za-z]{4}.*total",l,re.I)):
            if cur: blocks.append((curname,cur))
            curname=re.sub(r"^\d+\s*","",l).replace("... Total","").replace("Total","").strip(" .")
            cur=[]
            # header rows often carry the block TOTAL values; skip the values, keep name
            continue
        if cur is None: cur=[]; curname=None
        cur.append(row)
    if cur: blocks.append((curname,cur))
    return blocks

def extract_series(d):
    """From a corpus table: age rows -> (label, males, females, persons). Uses the first group that has
    male+female subkeys (POPULATION preferred); falls back to flat male/female columns."""
    best=None
    for s in d.get("data", []):
        rows=s.get("rows",[])
        if len(rows)<4: continue
        rh=next((k for k in rows[0] if not isinstance(rows[0].get(k),dict)),None)
        if not rh: continue
        labs=[str(r.get(rh,"")) for r in rows]
        agey=sum(1 for l in labs if AGELBL.match(l))
        if agey/len(rows)<0.4:
            # rows are NOT ages -> maybe a transposed table (rows=areas, columns=age bands; 1872/1881 style)
            tr=_transposed(s,rh,rows)
            if tr:
                import math
                tgood,tseries,tareaish=tr
                mag=sum((x[1] or 0)+(x[2] or 0) for x in tseries)
                secpri = 40 if re.search(r"total|all religions|general", s.get("name","") or "", re.I) else 0
                ge=tgood*10+secpri+(math.log10(mag+1)*8)
                if best is None or ge>best[0]:
                    best=(ge,s.get("name","")[:40],"@transposed" if tareaish else "@transposed-total",tseries)
            continue
        # find the M/F group
        r0=rows[0]
        # CASE: top-level groups ARE the sexes (subkeys are mislabeled year/civil columns) — take each
        # group's FIRST numeric subkey as that sex's value.
        gm=next((g for g in r0 if isinstance(r0[g],dict) and re.match(r"^males?\.?$",g.strip(),re.I)),None)
        gf=next((g for g in r0 if isinstance(r0[g],dict) and re.match(r"^females?\.?$",g.strip(),re.I)),None)
        if gm and gf and re.search(r"\bsingle\b(?!\s*year)|married|widow|civil condition",
                                   (s.get("name","") or "")+" "+str((d.get("metadata") or {}).get("title","")),re.I):
            gm=gf=None                                 # Males/Females of ONE civil condition, not the population
        if gm and gf:
            def first_sub(v):
                for k in v:
                    if PCTK.search(k): continue
                    n=num(v.get(k))
                    if n is not None: return n
                return None
            series=[]
            for r in rows:
                l=str(r.get(rh,"")).strip()
                vm,vf=r.get(gm),r.get(gf)
                if not isinstance(vm,dict) or not isinstance(vf,dict): continue
                m_,f_=first_sub(vm),first_sub(vf)
                if m_ is None and f_ is None: continue
                series.append([l,m_,f_, (m_+f_) if (m_ is not None and f_ is not None) else None])
            good=sum(1 for x in series if (x[1] or 0)>0 and (x[2] or 0)>0)
            if good>=6:
                import math
                mag=sum((x[1] or 0)+(x[2] or 0) for x in series)
                ge=good*10+(math.log10(mag+1)*8)
                if best is None or ge>best[0]:
                    best=(ge,s.get("name","")[:40],f"{gm}+{gf}",series)
            continue
        cand=[]
        for g,v in r0.items():
            if isinstance(v,dict):
                mks,fks,pk=_sexkeys(v)
                pk=pk or next((k for k in v if re.search(r"total",k,re.I) and not re.search(r"male",k,re.I) and not PCTK.search(k)),None)
                if mks and fks:
                    pri=0 if re.search(r"population|total",g,re.I) else 1
                    cand.append((pri,g,mks,fks,pk))
        if not cand: continue
        cand.sort()
        _,g,mks,fks,pk=cand[0]
        # column groups that are THEMSELVES areas (age rows x province columns): emit one block per place
        sane=[c for c in cand if len(re.sub(r"[^A-Za-z]","",c[1]))>=4 and not NOTPLACE.search(c[1]) and not JUNKNAME.match(c[1].strip(" ."))]
        if cand[0][0]>0 and len(sane)>=2:
            import math
            series=[]; tgood=0
            for _,g2,mk2,fk2,pk2 in sane:
                brows=[]
                for r in rows:
                    v=r.get(g2)
                    if not isinstance(v,dict): continue
                    m,f=_sexval(v,mk2),_sexval(v,fk2)
                    if m is None and f is None: continue
                    p=num(v.get(pk2)) if pk2 else (m+f if m is not None and f is not None else None)
                    brows.append([str(r.get(rh,"")).strip(),m,f,p])
                bgood=sum(1 for x in brows if (x[1] or 0)>0 and (x[2] or 0)>0)
                tots=[(x[1] or 0)+(x[2] or 0) for x in brows]
                if bgood<6 or sum(tots)<=0 or max(tots)/sum(tots)>0.6: continue
                series.append([g2.strip(" ."),None,None,None]); series+=brows; tgood+=bgood
            if series:
                mag=sum((x[1] or 0)+(x[2] or 0) for x in series)
                ge=tgood*10+(math.log10(mag+1)*8)
                if best is None or ge>best[0]:
                    best=(ge,s.get("name","")[:40],"@transposed",series)
            continue
        series=[]
        for r in rows:
            l=str(r.get(rh,"")).strip()
            v=r.get(g)
            if not isinstance(v,dict): continue
            m,f=_sexval(v,mks),_sexval(v,fks)
            p=num(v.get(pk)) if pk else (m+f if m is not None and f is not None else None)
            if m is None and f is None: continue
            series.append([l,m,f,p])
        good=sum(1 for x in series if (x[1] or 0)>0 and (x[2] or 0)>0)   # rows with REAL data both sides
        fill = good/max(1,len(series))
        import math
        mag = sum((x[1] or 0)+(x[2] or 0) for x in series)
        msum = sum(x[1] or 0 for x in series); fsum = sum(x[2] or 0 for x in series)
        if msum<=0 or fsum<=0:
            continue                                   # a one-sided column is not an age pyramid
        _chk=[x for x in series if x[3] is not None and x[1] is not None and x[2] is not None]
        _bad=sum(1 for x in _chk if abs((x[1]+x[2])-x[3])>max(2,0.02*abs(x[3])))
        if _chk and _bad/len(_chk)>0.3:
            continue                                   # persons != M+F on many rows -> not sex columns
        secpri = 40 if re.search(r"total|all religions|general", s.get("name","") or "", re.I) else 0
        good_eff = good*10 + secpri + (math.log10(mag+1)*8)
        if good>=6 and fill>=0.5 and (best is None or good_eff>best[0]):
            best=(good_eff,s.get("name","")[:40],g,series)
    return best

def main():
    rows=list(csv.DictReader(open("export/deep_place_tags.csv")))
    # index corpus files ONCE
    fidx={}
    for f in glob.glob("export/ocr_results_referenced/digitized_tables/*/*_oneshot.json"):
        m=re.search(r"([0-9a-f]{40})",f)
        if m: fidx.setdefault(m.group(1), f)
    ext_cache={}
    groups={}   # (canon, year) -> list of (score, ih, level, series_info)
    for r in rows:
        yr=r.get("year","")
        if yr not in CENSUS: continue
        m=re.search(r"([0-9a-f]{40})",r.get("table_id",""))
        if not m: continue
        ih=m.group(1)
        if ih not in fidx: continue
        if ih in ext_cache:
            cached=ext_cache[ih]
            if cached is None: continue
            md,t,good,secname,gname,series=cached
        else:
            try: d=json.load(open(fidx[ih]))
            except: ext_cache[ih]=None; continue
            md=d.get("metadata",{}) or {}
            t=str(md.get("title",""))
            if not AGE.search(t): ext_cache[ih]=None; continue
            ext=extract_series(d)
            if not ext: ext_cache[ih]=None; continue
            good,secname,gname,series=ext
            ext_cache[ih]=(md,t,good,secname,gname,series)
        # attribute to: volume_place always; plus tagged sub-areas get the same table if the table IS about them
        vp=r.get("volume_place","")
        canon,kind=canonicalize(vp)
        targets=[]
        if canon and kind!="junk": targets.append((canon,"province" if kind in("province","national") else kind))
        # a table titled/attributed at sub-level (deep tags put district/taluk names in volume_place for sub-tables)
        import math as _m
        _mag=sum((x[1] or 0)+(x[2] or 0) for x in series)
        score=good + _m.log10(_mag+1)*15 + (50 if md.get("data_type")!="proportional" else 0) \
              + (20 if md.get("human_verified") else 0) \
              + (30 if re.search(r"total|all religions", secname, re.I) else 0) \
              - (80 if re.search(r"caste|supplementary table e|literacy|infirm|occupation", t, re.I) else 0)
        # if the series concatenates multiple district blocks, split and attribute each block
        ages=[a for a in (_agev(x[0]) for x in series) if a is not None]
        resets=sum(1 for i in range(1,len(ages)) if ages[i]<ages[i-1])
        if resets>=2 or gname=="@transposed":
            for bname,brows in split_blocks(series):
                if not bname or len(brows)<5: continue
                bgood=sum(1 for x in brows if (x[1] or 0)>0 and (x[2] or 0)>0)
                if bgood<5: continue
                if BADBLOCK.search(bname) or JUNKNAME.match(bname): continue
                bname=re.sub(r"^.*?\.{3}\s*","",bname).strip(" .")   # 'Bilaspur ... Kawardha' -> 'Kawardha'
                bms=sum(x[1] or 0 for x in brows); bfs=sum(x[2] or 0 for x in brows)
                if bms<=0 or bfs<=0 or bms+bfs<60 or not (0.25<=bms/bfs<=4): continue   # subset/truncated slice
                bages=[a for a in (_agev(x[0]) for x in brows) if a is not None]
                blabs=sum(1 for x in brows if _agev(x[0]) is not None or AGELBL.match(str(x[0])))
                if blabs/len(brows)<0.6 or not bages or min(bages)>10: continue   # not a full age spread
                bc,bk=canonicalize(bname)
                label = bc if (bc and bk!="junk" and len(bname)>=4) else bname.title()[:30]
                if len(re.sub(r"[^A-Za-z]","",label))<4: continue
                bmag=sum((x[1] or 0)+(x[2] or 0) for x in brows)
                bscore=bgood+30+_m.log10(bmag+1)*8
                groups.setdefault((label,yr),[]).append((bscore,ih,"district",t[:60],bname[:36],brows))
            continue
        for canon2,lvl in targets:
            key=(canon2,yr)
            groups.setdefault(key,[]).append((score,ih,lvl,t[:60],secname,series))
    # pick best per (area,year)
    out={}
    for (area,yr),cands in groups.items():
        cands.sort(reverse=True)
        score,ih,lvl,t,secname,series=cands[0]
        out.setdefault(area,{})[yr]={"ih":ih,"level":lvl,"title":t,"section":secname,
                                     "rows":series,"n_alt":len(cands)-1}
    # emit json + zip of csvs
    os.makedirs("census_site/data",exist_ok=True)
    json.dump(out, open("census_site/data/area_ages.json","w"))
    zbuf=zipfile.ZipFile("census_site/data/area_ages.zip","w",zipfile.ZIP_DEFLATED)
    n_csv=0
    for area,years in out.items():
        for yr,rec in years.items():
            s=io.StringIO(); w=csv.writer(s)
            w.writerow(["age","males","females","persons"])
            for row in rec["rows"]: w.writerow(row)
            safe=re.sub(r"[^A-Za-z0-9]+","_",area)[:40]
            zbuf.writestr(f"area_ages/{safe}_{yr}.csv", s.getvalue()); n_csv+=1
    zbuf.close()
    ny={}
    for a,ys in out.items():
        for y in ys: ny[y]=ny.get(y,0)+1
    print(f"area-age dataset: {len(out)} areas · {n_csv} area-year tables -> area_ages.json ({os.path.getsize('census_site/data/area_ages.json')//1024}KB) + zip")
    print("by year:", dict(sorted(ny.items())))

if __name__=="__main__":
    main()
