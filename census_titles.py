#!/usr/bin/env python3
"""Shared title/grouping helpers for the Census static site.

Three layers:
  1. clean_title(t)      -> de-truncated, continuation-stripped, whitespace-collapsed title
  2. parse_series(t)     -> canonical "Imperial/Subsidiary/Supplementary Table <N>" group key
  3. subject_topic(...)  -> trustworthy subject for grouping (population, literacy, occupation,
                            infirmity, fertility, ... ) derived from the TITLE, because the
                            catalog's canonical_topic only encodes the crossed axes (age/sex/
                            civil/religion) and mislabels subject (e.g. tags "Occupation by Age"
                            as age_distribution).

Run standalone to validate against the catalog:
    python3 census_titles.py [path/to/catalog_tables.csv]
"""
import re
import sys

# ----------------------------------------------------------------------------- layer 1: clean
_CONT = re.compile(r"[\s\-–—.]*\(?\b(continued|concluded|contd?\.?|cont\.?|concld\.?)\b\)?\.?", re.I)
_DASH = re.compile(r"\s*[—–]\s*")


def clean_title(t):
    """Strip continuation markers, normalise dashes/whitespace. Keeps original casing."""
    t = (t or "").strip()
    if not t:
        return ""
    t = _CONT.sub(" ", t)
    # drop orphan brackets/parens left behind by continuation removal ("TABLE C )" -> "TABLE C")
    t = re.sub(r"\s*[()\[\]]\s*", " ", t) if re.search(r"[()\[\]]", t) and not re.search(r"\([^()]+\)", t) else t
    t = _DASH.sub(" — ", t)
    t = re.sub(r"\s+", " ", t).strip(" .,-—–)(")
    return t


def shorten(t, n=78):
    """Truncate on a word boundary with an ellipsis (full text belongs in a tooltip)."""
    t = t or ""
    if len(t) <= n:
        return t
    cut = t[:n].rsplit(" ", 1)[0].rstrip(" .,;:-—–")
    return (cut or t[:n]) + "…"


# ----------------------------------------------------------------------------- layer 2: series
_ROMAN = "(?:[IVXLC]+(?:-?[A-Z])?|[A-E]|\\d+)"
_SERIES = re.compile(
    r"\b(IMPERIAL|SUBSIDIARY|SUPPLEMENTARY|PROVINCIAL|COMPARATIVE)?\s*"
    r"\b(TABLE|STATEMENT|STATEMENT\s+No\.?)\s+(?:No\.?\s*)?(" + _ROMAN + r")\b",
    re.I,
)


def parse_series(t):
    """Return a canonical series key like 'Imperial Table VII' / 'Subsidiary Table V' / ''."""
    m = _SERIES.search(t or "")
    if not m:
        return ""
    pre = (m.group(1) or "").title()
    kind = re.sub(r"\s+no\.?$", "", m.group(2), flags=re.I).title()
    num = (m.group(3) or "").upper()
    return " ".join(x for x in (pre, kind, num) if x)


# ----------------------------------------------------------------------------- layer 3: subject
# "Strong" subjects: things the table is ABOUT that dims_present cannot see. Order = priority
# only used to break a positional tie; primary selection is earliest mention in the title.
_SUBJECTS = [
    ("Fertility",            r"fertilit|age of mother|mother at (?:the )?(?:first )?birth|female births|births reported|children (?:ever )?born|birth of (?:her |the )?first"),
    ("Mortality",            r"\bdeaths?\b|mortalit|\bdied\b|still-?births?|life table|expectation of life|\bsurvivors?\b|death-?rate"),
    ("Infirmity",            r"infirmit|\binfirm\b|insan|\bdeaf|\bmutes?\b|\bblind|lepro|\blepers?\b|lunatic|afflicted"),
    ("Literacy / Education", r"litera|illitera|educat|instruction|\bschools?\b|\blearning\b|reading and writing"),
    ("Language",             r"\blanguages?\b|mother[- ]tongue|bilingual|linguistic|vernacular"),
    ("Occupation",           r"occupation|livelihood|means of (?:living|livelihood)|\bindustr|\bearners?\b|\bworkers?\b|working population|by occupation"),
    ("Migration",            r"birth-?places?|migrat|immigrat|emigrat|\bborn in\b|enumerated (?:in|outside)|where born"),
    ("Growth / Variation",   r"\bgrowth\b|variation|increase and decrease|movement of (?:the )?population|intercensal|percentage (?:increase|decrease|variation)|relative growth"),
]
# "Structural" subjects: leading-noun nouns that overlap with the crossed axes.
_CASTE = re.compile(r"\bcaste|\btribes?\b|\bracial\b|\braces?\b|tribe or race", re.I)
_RELIGION = re.compile(r"\breligion|\bhindus?\b|\bmuslims?\b|musalman|\bchristians?\b|\bcreed|\bparsi|\bjains?\b|\bsikh|buddhis|animist|\bsects?\b", re.I)
_CIVIL = re.compile(r"civil[ -]condition|marital|marriage|\bmarried|\bwidow|unmarried|conjugal", re.I)
_AGESEX = re.compile(r"age,? *(?:and )?sex|sex,? *(?:and )?age|ages of the (?:people|population)|population by (?:sex and )?age|by age-?periods?|age[ -]distribution|age period", re.I)

_SUBJECTS_C = [(name, re.compile(rx, re.I)) for name, rx in _SUBJECTS]


def _earliest(rx, text):
    m = rx.search(text)
    return m.start() if m else 10 ** 6


def subject_topic(title, dims_present="", canonical_topic="", subject=""):
    """Best-guess subject for grouping. Title-first, structural fallback."""
    t = clean_title(title)
    # remove the series prefix so "TABLE VII" digits don't skew position matching
    t = _SERIES.sub(" ", t).strip()
    low = t.lower()
    dims = dims_present or ""

    # 1) strong subjects — earliest mention wins (handles "Literacy by Religion and Age")
    best, best_pos = None, 10 ** 6
    for name, rx in _SUBJECTS_C:
        p = _earliest(rx, low)
        if p < best_pos:
            best, best_pos = name, p
    if best is not None and best_pos < 10 ** 6:
        # but if a caste/religion/civil word leads BEFORE the strong subject, it's a breakdown,
        # not the subject — except literacy/occupation/etc. nearly always ARE the subject.
        return best

    # 2) caste/race as the leading noun
    cpos = _earliest(_CASTE, low)
    if cpos < min(_earliest(_AGESEX, low), _earliest(_CIVIL, low), _earliest(_RELIGION, low)):
        return "Caste / Race / Tribe"

    # 3) structural fallback from dims_present (the core population tables)
    has = lambda d: d in dims
    if has("sex") and has("civil_condition") and has("age"):
        return "Age × Sex × Civil condition"
    if has("religion") and has("age") and not has("civil_condition"):
        return "Age × Religion"
    if has("civil_condition") and has("age"):
        return "Age × Civil condition"
    if has("sex") and has("age"):
        return "Age × Sex"
    if has("religion"):
        return "Religion"

    # 4) title-only structural cues when dims_present is empty
    if _CIVIL.search(low) and _AGESEX.search(low):
        return "Age × Sex × Civil condition"
    if _RELIGION.search(low) and _AGESEX.search(low):
        return "Age × Religion"
    if _CIVIL.search(low):
        return "Civil condition"
    if _RELIGION.search(low):
        return "Religion"
    if _AGESEX.search(low):
        return "Age × Sex"

    # 5) last resort: map the coarse canonical_topic
    CT = {
        "age_sex_population": "Age × Sex",
        "age_sex_civil_condition": "Age × Sex × Civil condition",
        "age_distribution": "Age distribution",
        "age_civil_condition": "Age × Civil condition",
        "age_by_religion": "Age × Religion",
        "age_sex_marital_status": "Age × Sex × Marital",
    }
    return CT.get(canonical_topic, "Other / Unclassified")


# ----------------------------------------------------------------------------- badges
_MEAS = {
    "counts": "", "per_1000_rates": "per 1,000", "distribution_of_1000": "per-1,000 dist.",
    "other_rates": "rate", "per_10000": "per 10,000", "per_100000": "per 100,000",
    "unclassified": "",
}


def badges(row):
    """Small groupable chips for a table row (dict-like with catalog fields)."""
    out = [subject_topic(row.get("title", ""), row.get("dims_present", ""),
                         row.get("canonical_topic", ""), row.get("subject", ""))]
    meas = _MEAS.get(row.get("measure_category", ""), "")
    if not meas and row.get("measure_class") not in ("count", "", None):
        meas = row.get("measure_class")
    if meas:
        out.append(meas)
    ac = (row.get("age_coverage") or "").strip()
    if ac and ac != "0-0":
        out.append("age " + ac)
    ser = parse_series(row.get("title", ""))
    if ser:
        out.append(ser)
    return out


# ----------------------------------------------------------------------------- validation CLI
def _main(argv):
    import csv
    from collections import Counter, defaultdict
    path = argv[1] if len(argv) > 1 else "export_final/sendout_final/sendout_v1/catalog_tables.csv"
    rows = list(csv.DictReader(open(path)))
    topics = Counter()
    samples = defaultdict(list)
    series_hits = 0
    for r in rows:
        tp = subject_topic(r.get("title", ""), r.get("dims_present", ""),
                           r.get("canonical_topic", ""), r.get("subject", ""))
        topics[tp] += 1
        if len(samples[tp]) < 4:
            samples[tp].append(clean_title(r.get("title", ""))[:64])
        if parse_series(r.get("title", "")):
            series_hits += 1
    print(f"catalog: {len(rows)} tables\n")
    print("=== subject_topic distribution ===")
    for tp, n in topics.most_common():
        print(f"  {n:5d}  {tp}")
        for s in samples[tp]:
            print(f"            · {s}")
    print(f"\nseries parsed on {series_hits}/{len(rows)} ({100*series_hits/len(rows):.0f}%)")
    other = topics.get("Other / Unclassified", 0)
    print(f"Other/Unclassified: {other} ({100*other/len(rows):.1f}%)")


if __name__ == "__main__":
    _main(sys.argv)
