#!/usr/bin/env python3
"""Merge per-chunk place curation, validate every row, write atlas/out/gazetteer.csv.

Checks (each failure is recorded in `flags`, never silently fixed):
  schema    kind in the allowed set; key present in input
  gadm      every modern code exists in gadm_reference.tsv
  region    modern country/admin1 plausible for the strings' broad_region context
"""
import csv, glob, os, collections
HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(HERE, "curation")
KINDS = {"district","sub_district","town","city_section","princely_state","province","division","agency",
         "natural_division","aggregate","country","not_place","unsure"}
REGION_OK = {  # broad_region -> plausible modern (country, admin1-prefix) ; '' admin1 = any
 "South India": [("IND","TamilNadu"),("IND","Kerala"),("IND","Karnataka"),("IND","AndhraPradesh"),("IND","Telangana"),("IND","Puducherry"),("IND","Odisha"),("IND","Maharashtra"),("IND","Goa"),("IND","Lakshadweep")],
 "Burma": [("MMR","")],
 "North India": [("IND","UttarPradesh"),("IND","Uttarakhand"),("IND","Punjab"),("IND","Haryana"),("IND","HimachalPradesh"),("IND","NCTofDelhi"),("IND","JammuandKashmir"),("IND","Ladakh"),("IND","Chandigarh"),("PAK",""),("IND","Rajasthan"),("IND","MadhyaPradesh")],
 "West India": [("IND","Maharashtra"),("IND","Gujarat"),("IND","Karnataka"),("IND","Goa"),("IND","DadraandNagarHaveliandDamanandDiu"),("PAK","Sind"),("IND","MadhyaPradesh"),("IND","Rajasthan")],
 "Central India": [("IND","MadhyaPradesh"),("IND","Chhattisgarh"),("IND","Maharashtra"),("IND","Odisha"),("IND","Rajasthan"),("IND","UttarPradesh"),("IND","Jharkhand"),("IND","Telangana")],
 "Northeast India": [("IND","Assam"),("IND","Meghalaya"),("IND","Nagaland"),("IND","Mizoram"),("IND","Manipur"),("IND","Tripura"),("IND","ArunachalPradesh"),("BGD",""),("IND","WestBengal")],
 "East India": [("IND","WestBengal"),("IND","Bihar"),("IND","Jharkhand"),("IND","Odisha"),("BGD",""),("IND","Sikkim"),("IND","Assam")],
 "Rajputana": [("IND","Rajasthan"),("IND","MadhyaPradesh"),("IND","Gujarat")],
 "Northwest Frontier": [("PAK","")],
 "Andaman & Nicobar": [("IND","AndamanandNicobar")],
}
def main():
    gadm = set()
    for r in csv.DictReader(open(os.path.join(C, "gadm_reference.tsv")), delimiter="\t"):
        gadm.add(f'{r["country"]}/{r["admin1"]}/{r["district"]}')
    rows, flags = [], collections.Counter()
    for fin in sorted(glob.glob(os.path.join(C, "in_*.csv"))):
        chunk = os.path.basename(fin)[3:-4]
        inp = {r["key"]: r for r in csv.DictReader(open(fin))}
        outp = os.path.join(C, f"out_{chunk}.csv")
        got = {r["key"]: r for r in csv.DictReader(open(outp))} if os.path.exists(outp) else {}
        for k, i in inp.items():
            o = got.get(k)
            f = []
            if o is None:
                o = {"kind": "unsure", "hist_name": "", "hist_parent": "", "modern": "", "confidence": "", "note": "missing from curation output"}
                f.append("missing")
            if o["kind"] not in KINDS:
                f.append("bad_kind")
            mods = [m.strip() for m in (o.get("modern") or "").split(";") if m.strip()]
            bad = [m for m in mods if m not in gadm]
            if bad:
                f.append("gadm_unknown:" + "|".join(bad))
            reg = i["context"].split(" || ")[0].split(" | ")[0].strip()
            ok = REGION_OK.get(reg)
            if ok and mods and o["kind"] not in ("country",):
                if not any(any(m.startswith(c + "/" + a) for c, a in ok) for m in mods):
                    f.append(f"region_mismatch:{reg}")
            for x in f:
                flags[x.split(":")[0]] += 1
            rows.append({**{c: i[c] for c in ("key","label","variants","n_tables","sources","context")},
                         **{c: o.get(c, "") for c in ("kind","hist_name","hist_parent","modern","confidence","note")},
                         "chunk": chunk, "flags": ";".join(f)})
    cols = list(rows[0].keys())
    with open(os.path.join(HERE, "out", "gazetteer.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    kc = collections.Counter(r["kind"] for r in rows)
    print(f"{len(rows)} strings; kinds: {dict(kc.most_common())}")
    print("flags:", dict(flags))
    print("with modern:", sum(1 for r in rows if r["modern"]),
          "| confidence:", dict(collections.Counter(r["confidence"] for r in rows if r["modern"])))
if __name__ == "__main__":
    main()
