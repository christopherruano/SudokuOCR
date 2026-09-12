#!/usr/bin/env python3
"""Phase 1 of the district concordance pipeline: classify every geography.

Deterministic pass over geo_reference.csv + keeper_catalog.csv. Every row gets
an entity_class plus the rule id and evidence that produced it; names no rule
can settle go to out/unresolved_queue.csv for the reviewed-override pass.
Manual decisions belong in overrides.csv (with justification), never here.

Outputs (concordance/out/):
  geo_classified.csv    every canonical name with entity_class, rule, evidence
  district_universe.csv one row per district: province, years, table/fact counts
  unresolved_queue.csv  names needing human/LLM review
"""
import csv, collections, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
SV1 = os.path.join(HERE, "..", "export_final", "sendout_final", "sendout_v1")

# entity types in geo_reference v1 that we accept as-is (provenance: prior catalog)
CARRY = {"district", "urban", "sub_district", "division", "british_province",
         "princely_state", "agency", "all_india", "modern_state", "native_states_agg"}

SUFFIX_RULES = [  # (rule_id, regex on name, entity_class)
    # a row naming two districts ("X District / Y District", "X and Y District")
    # is a printed aggregate, not a unit — must precede the district suffix rule
    ("multi_unit_agg",   r"districts?\b.*(/| and | & ).*\bdistricts?\b", "aggregate"),
    ("slash_agg",        r"\S\s*/\s*\S", "aggregate"),
    ("suffix_district",  r"\bdistricts?$", "district"),
    ("suffix_city",      r"\b(city|town|municipality|cantonment|port|suburbs)$", "urban"),
    ("suffix_subdist",   r"\b(taluk|taluka|tehsil|tahsil|thana|pargana|mahal|island)s?$", "sub_district"),
    ("suffix_division",  r"\bdivision$", "division"),
    ("suffix_state",     r"\b(state|residency|estate)s?$", "princely_state"),
    ("suffix_agency",    r"\bagency$", "agency"),
    ("suffix_province",  r"\bprovinces?$", "british_province"),
]

def load(name):
    with open(os.path.join(SV1, name), newline="") as f:
        return list(csv.DictReader(f))

def apply_overrides(rows):
    path = os.path.join(HERE, "overrides.csv")
    if not os.path.exists(path): return rows, 0
    n = 0
    ov = {}
    with open(path, newline="") as f:
        for o in csv.DictReader(f):
            ov[o["canonical_name"]] = o
    for r in rows:
        o = ov.get(r["canonical_name"])
        if o and o.get("entity_class"):
            r.update(entity_class=o["entity_class"], class_rule="override",
                     class_evidence=o.get("justification", ""))
            n += 1
    return rows, n

def main():
    geo = load("geo_reference.csv")
    cat = load("keeper_catalog.csv")

    # usage per canonical name from the accepted-table catalog
    use = collections.defaultdict(lambda: {"years": set(), "tables": 0, "facts": 0})
    for t in cat:
        if t["status"] != "keeper_accepted": continue
        u = use[t["canonical_name"].strip()]
        if t["year"]: u["years"].add(t["year"])
        u["tables"] += 1
        u["facts"] += int(t["n_facts"] or 0)

    out, queue = [], []
    for g in geo:
        name = g["canonical_name"].strip() or g["raw_geo_name"].strip()
        et = (g["entity_type"] or "").strip()
        row = dict(canonical_name=name, raw_geo_name=g["raw_geo_name"],
                   parent=g["parent"], broad_region=g["broad_region"],
                   n_tables_ref=g["n_tables"])
        low0 = name.lower()
        if g.get("is_geography", "").strip() == "0" or et == "not_geography":
            row.update(entity_class="not_geography", class_rule="carry_not_geography",
                       class_evidence="flagged not a geography in geo_reference.csv")
        elif re.search(r"districts?\b.*(/| and | & ).*\bdistricts?\b", low0) or re.search(r"\S\s*/\s*\S", low0):
            # multi-unit printed rows outrank carried v1 types, which mislabel them
            row.update(entity_class="aggregate", class_rule="multi_unit_agg",
                       class_evidence="row names multiple units")
        elif re.search(r"\bdistrict\b[ .\-]*\b(urban|rural)\b", low0) or \
             re.search(r"\b(brahmin?s?|males|females)\b", low0):
            # urban/rural or caste/sex splits of a district are data slices, not units
            row.update(entity_class="district_subset", class_rule="district_subset",
                       class_evidence="urban/rural or subpopulation slice of a district")
        elif et in CARRY:
            row.update(entity_class=et, class_rule="carry_geo_reference_v1",
                       class_evidence=f"entity_type={et} in geo_reference.csv")
        else:
            low = name.lower()
            for rule, pat, cls in SUFFIX_RULES:
                if re.search(pat, low):
                    row.update(entity_class=cls, class_rule=rule,
                               class_evidence=f"name matches /{pat}/")
                    break
            else:
                row.update(entity_class="UNRESOLVED", class_rule="none",
                           class_evidence=f"entity_type was '{et or 'blank'}'")
        out.append(row)

    out, n_ov = apply_overrides(out)
    for row in out:
        if row["entity_class"] == "UNRESOLVED": queue.append(row)

    def dump(fname, rows, cols):
        with open(os.path.join(HERE, "out", fname), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)

    cols = ["canonical_name", "raw_geo_name", "entity_class", "class_rule",
            "class_evidence", "parent", "broad_region", "n_tables_ref"]
    dump("geo_classified.csv", out, cols)
    dump("unresolved_queue.csv", queue, cols)

    seen = set(); uni = []
    for row in out:
        if row["entity_class"] != "district" or row["canonical_name"] in seen: continue
        seen.add(row["canonical_name"])
        u = use.get(row["canonical_name"], {"years": set(), "tables": 0, "facts": 0})
        uni.append(dict(canonical_name=row["canonical_name"], parent=row["parent"],
                        broad_region=row["broad_region"],
                        years=";".join(sorted(u["years"])),
                        n_years=len(u["years"]), n_tables=u["tables"], n_facts=u["facts"]))
    uni.sort(key=lambda r: (r["parent"], r["canonical_name"]))
    dump("district_universe.csv", uni,
         ["canonical_name", "parent", "broad_region", "years", "n_years", "n_tables", "n_facts"])

    tally = collections.Counter(r["entity_class"] for r in out)
    print("classified", len(out), "geographies;", n_ov, "via overrides")
    for k, v in tally.most_common(): print(f"  {k}: {v}")
    print(f"districts in universe: {len(uni)}; unresolved queue: {len(queue)}")

if __name__ == "__main__":
    main()
