#!/usr/bin/env python3
"""Extremely thorough QC on facts_confirmed.cleaned.csv before sendout."""
import csv, re, json, math
from collections import defaultdict, Counter
csv.field_size_limit(10**7)
F='export_final/sendout_final/data/facts_confirmed.cleaned.csv'
rows=list(csv.DictReader(open(F)))
def num(x):
    try: return float(x)
    except: return None
report=[]
def chk(name, bad, total=None, sample=None):
    report.append((name, len(bad) if hasattr(bad,'__len__') else bad, total, sample))

# 1 structural validity
bad_sub=[r for r in rows if r['sub_column_std'] not in ('persons','males','females','')]
bad_unit=[r for r in rows if r['measure_unit'] not in ('count','rate','')]
bad_val=[r for r in rows if num(r['value']) is None and r['value'] not in ('',None)]
sexmap={'persons':'All','males':'Male','females':'Female'}
bad_sexcons=[r for r in rows if r['sub_column_std'] in sexmap and r['sex'] and r['sex']!=sexmap[r['sub_column_std']]]
chk('non-standard sub_column', bad_sub)
chk('invalid measure_unit', bad_unit)
chk('non-numeric value (non-blank)', bad_val, sample=[r['value'] for r in bad_val[:5]])
chk('sub_column vs sex mismatch', bad_sexcons, sample=[(r['sub_column_std'],r['sex']) for r in bad_sexcons[:5]])

# 2 negatives (count facts shouldn't be negative)
neg=[r for r in rows if r['measure_unit']=='count' and (num(r['value']) or 0)<0]
chk('negative COUNT values', neg, sample=[r['value'] for r in neg[:5]])

# 3 duplicate facts (same cell, conflicting value)
cellmap=defaultdict(set)
for r in rows:
    k=(r['table_id'],r['section'],r['row_label_std'],r['column_group_std'],r['sub_column_std'])
    cellmap[k].add(r['value'])
dups=[k for k,v in cellmap.items() if len(v)>1]
chk('cells with conflicting duplicate values', dups, total=len(cellmap), sample=[k[-3:] for k in dups[:5]])

# 4 value outliers
big_count=[r for r in rows if r['measure_unit']=='count' and (num(r['value']) or 0)>5e8]
big_rate=[r for r in rows if r['measure_unit']=='rate' and (num(r['value']) or 0)>1e6]
chk('COUNT > 500M (implausible)', big_count, sample=[r['value'] for r in big_count[:5]])
chk('RATE > 1,000,000 (residual decimal?)', big_rate, sample=[r['value'] for r in big_rate[:5]])

# 5 empty tables (no facts) / tables that lost all data
tbl_facts=Counter(r['table_id'] for r in rows)
chk('distinct tables', len(tbl_facts))
tiny=[t for t,n in tbl_facts.items() if n<3]
chk('tables with <3 facts', tiny)

# 6 missing both M and F and persons for a cell row (all-null)
# 7 age resolution sanity (facts have no year/province cols)
bad_age=[r for r in rows if r['resolved_age_lower'] and r['resolved_age_upper'] and num(r['resolved_age_lower']) is not None and num(r['resolved_age_upper']) is not None and num(r['resolved_age_upper'])<num(r['resolved_age_lower'])]
chk('age range upper<lower', bad_age)

# 8 P=M+F (count)
cell=defaultdict(dict)
for r in rows:
    if r['measure_unit']!='count': continue
    s=r['sub_column_std'] or {'All':'persons','Male':'males','Female':'females'}.get(r['sex'],'')
    if s in ('persons','males','females'):
        v=num(r['value'])
        if v is not None: cell[(r['table_id'],r['section'],r['row_label_std'],r['column_group_std'])][s]=v
pmf=[k for k,d in cell.items() if all(x in d for x in ('persons','males','females')) and abs(d['persons']-d['males']-d['females'])>0.5]
chk('P!=M+F failures', pmf, total=sum(1 for d in cell.values() if all(x in d for x in ('persons','males','females'))))

# 9 zero-flood: tables where >90% of values are 0 (suspicious)
tz=defaultdict(lambda:[0,0])
for r in rows:
    v=num(r['value'])
    if v is None: continue
    tz[r['table_id']][0]+= (1 if v==0 else 0); tz[r['table_id']][1]+=1
zf=[t for t,(z,n) in tz.items() if n>=10 and z/n>0.9]
chk('tables >90% zeros', zf)

print('='*64); print('FINAL QC REPORT — facts_confirmed.cleaned.csv'); print('='*64)
print(f'total facts: {len(rows):,}')
for name,count,total,sample in report:
    flag='OK ' if (isinstance(count,int) and count==0) else ('-- ' if name in ('distinct tables',) else 'XX ')
    extra=f' / {total:,}' if total else ''
    s=f'  e.g. {sample}' if sample else ''
    print(f'  [{flag}] {name}: {count}{extra}{s}')
