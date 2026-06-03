#!/usr/bin/env python3
"""Build a canonical geography reference for the historical Indian Census corpus
(1872-1941, extending to 1961 reprints), cross-referenced against the actual
administrative hierarchy of British India + the Princely States.

Input : /tmp/geo_universe.json  {raw_geo_name: n_tables}
Output: export_final/sendout_final/sendout_v1/geo_reference.csv
        columns: raw_geo_name, n_tables, canonical_name, entity_type,
                 parent, broad_region, is_geography, note

entity_type taxonomy
  all_india        - whole-India / multi-province aggregate (India, Country)
  british_province - directly-administered province / presidency / chief-commissionership
  princely_state   - native/feudatory state
  agency           - political agency grouping many states (Rajputana, Central India)
  division         - revenue division within a province (parent often inferable)
  district         - district within a province/state
  sub_district     - taluk / taluq / tahsil / township
  urban            - city / town / municipality / ward / section (incl. Bombay City wards)
  native_states_agg- "Native States" style aggregate, not a single unit
  unspecified      - generic placeholder (Province, State, Total, Unknown, blank)
  not_geography    - a religion/caste/community/table token mis-parsed as a place
"""
import csv, json, os, re

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "export_final/sendout_final/sendout_v1/geo_reference.csv")

# ---------------------------------------------------------------- canonical units
# canonical_name -> (entity_type, parent, broad_region)
CANON = {
    # ---- All-India aggregates ----
    "India": ("all_india", "", "All-India"),
    # ---- British provinces / presidencies / chief commissionerships ----
    "Madras": ("british_province", "British India", "South India"),
    "Bombay": ("british_province", "British India", "West India"),
    "Bengal": ("british_province", "British India", "East India"),
    "Assam": ("british_province", "British India", "Northeast India"),
    "Punjab": ("british_province", "British India", "North India"),
    "Burma": ("british_province", "British India", "Burma"),
    "Central Provinces and Berar": ("british_province", "British India", "Central India"),
    "Berar": ("british_province", "Central Provinces and Berar", "Central India"),
    "United Provinces": ("british_province", "British India", "North India"),
    "Bihar": ("british_province", "British India", "East India"),
    "Orissa": ("british_province", "British India", "East India"),
    "Bihar and Orissa": ("british_province", "British India", "East India"),
    "North-West Frontier Province": ("british_province", "British India", "Northwest Frontier"),
    "Baluchistan": ("british_province", "British India", "Northwest Frontier"),
    "Coorg": ("british_province", "British India", "South India"),
    "Ajmer-Merwara": ("british_province", "British India", "Rajputana"),
    "Delhi": ("british_province", "British India", "North India"),
    "Andaman and Nicobar Islands": ("british_province", "British India", "Andaman & Nicobar"),
    "Sind": ("british_province", "British India", "West India"),
    # ---- Post-1947 successor state names (appear in 1951/1961 reprints) ----
    "Uttar Pradesh": ("modern_state", "Republic of India", "North India"),
    "Madhya Pradesh": ("modern_state", "Republic of India", "Central India"),
    "Rajasthan": ("modern_state", "Republic of India", "Rajputana"),
    "Bombay State": ("modern_state", "Republic of India", "West India"),
    # ---- Princely / native states ----
    "Hyderabad": ("princely_state", "Deccan", "South India"),
    "Mysore": ("princely_state", "Native States", "South India"),
    "Baroda": ("princely_state", "Native States", "West India"),
    "Travancore": ("princely_state", "Native States", "South India"),
    "Cochin": ("princely_state", "Native States", "South India"),
    "Gwalior": ("princely_state", "Central India Agency", "Central India"),
    "Indore": ("princely_state", "Central India Agency", "Central India"),
    "Bhopal": ("princely_state", "Central India Agency", "Central India"),
    "Jammu and Kashmir": ("princely_state", "Native States", "North India"),
    "Chamba": ("princely_state", "Punjab States", "North India"),
    "Kolhapur": ("princely_state", "Deccan States", "West India"),
    "Ratlam": ("princely_state", "Central India Agency", "Central India"),
    "Jaora": ("princely_state", "Central India Agency", "Central India"),
    "Sailana": ("princely_state", "Central India Agency", "Central India"),
    "Sitamau": ("princely_state", "Central India Agency", "Central India"),
    "Jhabua": ("princely_state", "Central India Agency", "Central India"),
    "Banganapalle": ("princely_state", "Madras States", "South India"),
    "Sandur": ("princely_state", "Madras States", "South India"),
    # ---- Political agencies (groupings of many states) ----
    "Rajputana Agency": ("agency", "Native States", "Rajputana"),
    "Central India Agency": ("agency", "Native States", "Central India"),
    "Bundelkhand Agency": ("agency", "Central India Agency", "Central India"),
}

# raw alias -> canonical_name (handles spelling/abbrev variants)
ALIAS = {
    "country": "India", "burma and india": "India", "british india": "India",
    "madras presidency": "Madras", "madras province": "Madras", "bombay presidency": "Bombay",
    "bombay presidency including native states": "Bombay", "bengal presidency": "Bengal",
    "central provinces": "Central Provinces and Berar",
    "north-western provinces and oudh": "United Provinces",
    "north western provinces oudh": "United Provinces", "north-west provinces": "United Provinces",
    "north western provinces": "United Provinces", "n.-w. p. and oudh": "United Provinces",
    "n.-w. provinces and oudh": "United Provinces", "n.-w. provinces": "United Provinces",
    "united provinces (british territory)": "United Provinces",
    "n.-w. f. p": "North-West Frontier Province", "nwfp": "North-West Frontier Province",
    "burma proper": "Burma", "divisional burma": "Burma",
    "ajmer": "Ajmer-Merwara", "ajmer merwara": "Ajmer-Merwara", "merwara": "Ajmer-Merwara",
    "rajputana": "Rajputana Agency", "central india": "Central India Agency",
    "rajputana and ajmer-merwara": "Rajputana Agency", "rajputana / ajmer-merwara": "Rajputana Agency",
    "rajasthan and ajmer": "Rajputana Agency", "simla hill states": "Punjab",
    "south bihar": "Bihar", "lower provinces": "Bengal", "trans-frontier posts": "North-West Frontier Province",
    "andamans and nicobars": "Andaman and Nicobar Islands", "british empire": "India",
    "rewa": "Central India Agency", "kekri": "Ajmer-Merwara",
    "kashmir": "Jammu and Kashmir", "kashmir state": "Jammu and Kashmir",
    "jammu and kashmir state": "Jammu and Kashmir", "jammu province": "Jammu and Kashmir",
    "hyderabad state": "Hyderabad", "hyderabad province": "Hyderabad",
    "mysore state": "Mysore", "mysore province": "Mysore",
    "baroda state": "Baroda", "cochin state": "Cochin", "gwalior state": "Gwalior",
    "indore state": "Indore", "jaora state": "Jaora", "ratlam state": "Ratlam",
    "port blair": "Andaman and Nicobar Islands",
}

# district / sub-unit -> parent canonical state/province
DISTRICT_PARENT = {
    # Mysore state
    "bangalore": "Mysore", "kolar": "Mysore", "shimoga": "Mysore", "tumkur": "Mysore",
    "chitaldrug": "Mysore", "hassan": "Mysore", "kadur": "Mysore", "kolar gold fields": "Mysore",
    "civil and military station": "Mysore",
    # Burma
    "pegu": "Burma", "prome": "Burma", "sagaing": "Burma", "sandoway": "Burma",
    "kyaukpyu": "Burma", "lower chindwin": "Burma", "tharrawaddy": "Burma", "thongwa": "Burma",
    "yamethin": "Burma", "hanthawaddy": "Burma", "salween": "Burma", "thaton": "Burma",
    "rangoon": "Burma", "kyauktan": "Burma",
    # Madras
    "south arcot": "Madras", "north arcot": "Madras", "salem": "Madras", "tanjore": "Madras",
    "nellore": "Madras", "cuddapah": "Madras", "kurnool": "Madras", "bellary": "Madras",
    "kistna": "Madras", "godavari": "Madras", "anantapur": "Madras",
    # Bengal
    "calcutta": "Bengal", "hooghly": "Bengal", "darjeeling": "Bengal", "chandernagore": "Bengal",
    # Bombay (province + city wards/sections)
    "karachi": "Bombay", "surat": "Bombay", "poona": "Bombay", "sholapur": "Bombay",
    "sholápur": "Bombay", "ratnagiri": "Bombay", "navsari": "Bombay", "mandvi": "Bombay",
    "byculla": "Bombay", "walkeshwar": "Bombay", "khetwadi": "Bombay", "mahalaxmi": "Bombay",
    "tardeo": "Bombay", "dhobitalao": "Bombay", "bhuleshwar": "Bombay", "market section": "Bombay",
    "dongree": "Bombay", "khara talao": "Bombay", "kumbharwada": "Bombay", "sion": "Bombay",
    "mahim": "Bombay", "fort": "Bombay", "bhopal": "Bombay", "khara": "Bombay",
    # NWFP / Punjab
    "peshawar": "North-West Frontier Province", "lahore": "Punjab", "amritsar": "Punjab",
    # Hyderabad (Deccan) districts
    "aurangabad": "Hyderabad", "bidar": "Hyderabad", "gulbarga": "Hyderabad", "raichur": "Hyderabad",
    "nizamabad": "Hyderabad", "adilabad": "Hyderabad", "karimnagar": "Hyderabad",
    "mahbubnagar": "Hyderabad", "nander": "Hyderabad", "parbhani": "Hyderabad", "medak": "Hyderabad",
    "warangal": "Hyderabad", "nalgonda": "Hyderabad",
    # Travancore taluks
    "trivandrum": "Travancore", "quilon": "Travancore", "kottayam": "Travancore",
    "changanachery": "Travancore", "tiruvalla": "Travancore", "mavelikara": "Travancore",
    "chengannur": "Travancore", "nedumangad": "Travancore", "neyyattinkara": "Travancore",
    "vilavankod": "Travancore", "agastisvaram": "Travancore", "eraniel": "Travancore",
    "kalkulam": "Travancore", "tovala": "Travancore", "shencottah": "Travancore",
    "pattanapuram": "Travancore", "karunagapalli": "Travancore", "kartikappalli": "Travancore",
    "ambalapuzha": "Travancore", "shertallay": "Travancore", "kayankulam": "Travancore",
    "chirayinkil": "Travancore", "alleppy": "Travancore",
    # Cochin taluks
    "kunnatnad": "Cochin", "alangad": "Cochin", "mukundapuram": "Cochin", "trichur": "Cochin",
    "talappilli": "Cochin", "chittur": "Cochin", "kanayannur": "Cochin", "parur": "Cochin",
    "muvattupuzha": "Cochin", "todupuzha": "Cochin", "vaikam": "Cochin", "ettumanur": "Cochin",
    # United Provinces districts
    "etah": "United Provinces", "shahjahanpur": "United Provinces", "dehradun": "United Provinces",
    "muzaffarnagar": "United Provinces", "meerut": "United Provinces", "देहरादून": "United Provinces",
    "मुजफ्फरनगर": "United Provinces",
    # Central India / Gwalior
    "bhind": "Gwalior", "sheopur": "Gwalior",
    # additional known districts -> parent
    "darrang": "Assam",
    "ganjam": "Madras", "vizagapatam": "Madras", "malabar": "Madras", "anjengo": "Travancore",
    "agra": "United Provinces", "आगरा": "United Provinces",
    "hoshiarpur": "Punjab", "jullundur": "Punjab", "kapurthala": "Punjab",
    "mandi": "Punjab", "suket": "Punjab",
    "girgaum": "Bombay", "colaba": "Bombay", "mazagaon": "Bombay", "sewri": "Bombay",
    "fanaswadi": "Bombay", "dongri": "Bombay", "nagpada": "Bombay", "warli": "Bombay",
    "konkan": "Bombay", "ahmadabad": "Bombay", "aden": "Bombay", "byculla": "Bombay",
    "okhamandal": "Baroda", "akyab": "Burma", "koraput": "Madras",
    "मुरादाबाद": "United Provinces", "बरेली": "United Provinces", "रुहेलखंड": "United Provinces",
}

# tokens that are NOT geography (religion/caste/community/table artefacts mis-parsed as a place)
NOT_GEO = {
    "jew", "parsi", "bramho", "brahmo", "anglo-indians", "anglo-indian", "others", "hindu",
    "muslim", "musalman", "christian", "buddhist", "sikh", "jain", "animist",
    "general/sample", "subsidiary", "appendix ii", "state tables vii-a-c", "state tables",
    "provincial total of literacy", "brahmins of kolar district", "wn. and country",
    "selected castes, tribes and races", "selected castes tribes and races", "unsorted",
    "plain", "plain, northern", "syrian christians", "famine and non-famine districts",
}

# generic placeholders -> unspecified
GENERIC = {
    "", "∅", "/", "province", "state", "district", "city", "town", "country.", "total",
    "provincial", "provincial total", "provincial summary", "provincial total of",
    "state summary", "state total", "the state", "whole province", "unknown", "province (districts)",
    "british territory", "british districts", "british districts and natural divisions",
    "hilly", "subsidiary", "provincial and urban", "provinces", "nat. states",
}

DIVISION_RE = re.compile(r"\bdivi|डिवीजन", re.I)  # division / divisi / divi (OCR-truncated) / Hindi
DISTRICT_RE = re.compile(r"\bdistrict\b", re.I)
TALUK_RE = re.compile(r"\b(taluk|taluq|tahsil|township|taluks)\b", re.I)
URBAN_RE = re.compile(r"\b(city|town|municipality|municipal|ward|section|gold fields|suburb|cantonment)\b", re.I)
AGENCY_RE = re.compile(r"\bagency\b", re.I)
PRESIDENCY_RE = re.compile(r"\bpresidency\b", re.I)
STATE_SUFFIX_RE = re.compile(r"\bstate\b", re.I)


def parent_lookup(low):
    for key, parent in DISTRICT_PARENT.items():
        if key in low:
            return parent
    return ""


def region_of(canonical, parent):
    for name in (canonical, parent):
        if name in CANON:
            return CANON[name][2]
    return ""


def classify(raw):
    low = raw.strip().lower()
    # 1) explicit non-geography
    if low in NOT_GEO:
        return ("", "not_geography", "", "", 0, "religion/caste/community or table token, not a place")
    # 2) generic placeholder
    if low in GENERIC or low.startswith("appendix"):
        return ("", "unspecified", "", "", 0, "generic placeholder")
    # 3) native-states aggregate
    if low in ("native states", "nat. states", "feudatories"):
        return ("Native States", "native_states_agg", "British India", "All-India", 1, "aggregate of princely states")
    # 4) direct alias
    if low in ALIAS:
        c = ALIAS[low]; t, p, r = CANON[c]; return (c, t, p, r, 1, "")
    # 5) exact canonical
    if raw in CANON:
        t, p, r = CANON[raw]; return (raw, t, p, r, 1, "")
    # 6) prefix-stripped canonical match ("Mysore State, excluding ...", "10-Etah District")
    base = re.sub(r"^\d+[-.\s]+", "", raw).strip()
    for c in CANON:
        if base.lower().startswith(c.lower()):
            t, p, r = CANON[c]; return (c, t, p, r, 1, "matched canonical prefix")
    # 7) structural suffix rules
    parent = parent_lookup(low)
    region = region_of(parent, parent) or (CANON.get(parent, ("", "", ""))[2] if parent else "")
    if AGENCY_RE.search(raw):
        return (raw, "agency", "Native States", region, 1, "")
    if PRESIDENCY_RE.search(raw):
        c = base.split(" Presidency")[0]
        if c in CANON:
            t, p, r = CANON[c]; return (c, t, p, r, 1, "")
    if TALUK_RE.search(raw):
        return (raw, "sub_district", parent, region, 1, "taluk/township")
    if URBAN_RE.search(raw):
        return (raw, "urban", parent, region, 1, "city/town/ward")
    if DISTRICT_RE.search(raw):
        return (raw, "district", parent, region, 1, "")
    if DIVISION_RE.search(raw):
        return (raw, "division", parent, region, 1, "revenue division")
    if STATE_SUFFIX_RE.search(raw):
        c = base.split(" State")[0]
        if c in CANON:
            t, p, r = CANON[c]; return (c, t, p, r, 1, "")
        return (raw, "princely_state", "Native States", region, 1, "state suffix")
    # 8) bare district/place known by parent
    if parent:
        return (raw, "district", parent, region, 1, "matched by known parent place")
    # 9) unresolved
    return (raw, "unresolved", "", "", 1, "not matched - needs manual geo review")


def main():
    universe = json.load(open("/tmp/geo_universe.json"))
    rows = []
    for raw, n in sorted(universe.items(), key=lambda x: -x[1]):
        canon, etype, parent, region, isgeo, note = classify(raw)
        rows.append({"raw_geo_name": raw, "n_tables": n, "canonical_name": canon,
                     "entity_type": etype, "parent": parent, "broad_region": region,
                     "is_geography": isgeo, "note": note})
    cols = ["raw_geo_name", "n_tables", "canonical_name", "entity_type", "parent", "broad_region", "is_geography", "note"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in rows]

    from collections import Counter
    by_type = Counter()
    tbl_by_type = Counter()
    for r in rows:
        by_type[r["entity_type"]] += 1
        tbl_by_type[r["entity_type"]] += r["n_tables"]
    print(f"geo_reference.csv: {len(rows)} raw names -> {OUT}")
    print("  entity_type (distinct names / tables covered):")
    for t, c in by_type.most_common():
        print(f"    {t:18} {c:4} names   {tbl_by_type[t]:5} tables")
    unresolved = [r for r in rows if r["entity_type"] == "unresolved"]
    print(f"  UNRESOLVED names: {len(unresolved)} ({sum(r['n_tables'] for r in unresolved)} tables)")
    for r in sorted(unresolved, key=lambda x: -x["n_tables"])[:25]:
        print(f"      {r['n_tables']:3}  {r['raw_geo_name']}")


if __name__ == "__main__":
    main()
