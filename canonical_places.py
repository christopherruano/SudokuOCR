#!/usr/bin/env python3
"""Canonical gazetteer for British-Indian census provinces (1871-1951): maps the messy parsed labels
onto canonical provinces, with historical aliases + successor states, and marks which entries are
NATIONAL volumes (whose Table VII typically breaks down BY PROVINCE - so they cover provinces 'via
parent') vs coarse REGION buckets. Used by build_best_dataset.py for coverage + best-selection.

Contract: canonicalize(label) -> (canonical_name, kind), kind in {province, national, region, junk}.

Successor-state unification (one place traceable across census years):
  United Provinces  <- North-Western Provinces, Oudh, N.-W. P. and Oudh, Agra and Oudh, Uttar Pradesh
  Central Provinces <- Saugor territories, Madhya Pradesh (also CP & Berar; Berar stays separate
                       because it reported its own volumes 1881-1901 before merging in 1903)
  Rajputana         <- Rajputana Agency, Jaipur, Rajasthan
  Kashmir           <- Jammu, Jammu and Kashmir
  Bombay            <- Bombay Presidency, Bombay State
  NWFP              <- North-West(ern) Frontier (Province)
  All-India         <- India, India (national), All-India, Census of British India
Bihar & Orissa (1912-1936) is kept distinct from post-1936 Bihar and Orissa: that was a SPLIT,
not a rename, so unifying would conflate different territories."""
import re

# canonical name -> list of alias regexes (matched against the lowercased parsed label)
CANON = {
    "All-India":               [r"^india\b", r"^all[- ]?india\b", r"^india \(national\)", r"^census of british india", r"^vol\.? 1", r"^country$", r"^british empire$"],
    "Madras":                  [r"^madras", r"presidency of madras"],
    "Bombay":                  [r"^bombay", r"^city of bombay",
                                # 'Volume IX.-A. Bombay' (1901 Pt II tables) truncates to '-A' upstream
                                r"^-a$"],
    "Bengal":                  [r"^bengal(?!,? bihar)", r"^lower provinces\b", r"bengal & sikkim|bengal and sikkim", r"bengal and dacca", r"^howrah"],
    "Bihar & Orissa":          [r"bihar,? (and|&) orissa", r"bengal,? bihar,? (and|&) orissa"],
    "Bihar":                   [r"\bbihar\b(?!,? (and|&) orissa)"],
    "Orissa":                  [r"^orissa\b", r"^bolangir", r"^dhenkanal", r"^koraput"],
    "United Provinces":        [r"united provinces", r"north[- ]?western provinces?( and oudh)?", r"^the north-western province", r"^north-?west provinces", r"^agra and oudh", r"^uttar pradesh", r"^n\.?\s?-?\s?w\.?\s?p(rovinces|\b)", r"^n$",
                                r"^oudh\b", r"census of oudh"],
    "Punjab":                  [r"^punjab", r"^the punjab", r"panj[aá]b", r"punjab and delhi"],
    "Burma":                   [r"^burma(?! report)", r"divisional burma"],
    "Central Provinces":       [r"central provinces", r"^madhya pradesh", r"^saugor",
                                # CP district volumes + 'Census of the Central Provinces, 1881. Volume I. Tables,
                                # Imperial and Supplementary' whose title truncates to the tail phrase upstream
                                r"^seoni", r"^tables, imperial and supplementary$"],
    "Assam":                   [r"^assam", r"hill districts and manipur", r"^manipur",
                                # 'Census of India, 1891. Assam. Vol. II.—Tables' / 'Vol. III.--Provincial Tables'
                                # / the 1891 report volume: titles truncate to these fragments upstream
                                r"^[—–-]+\s*provincial tables$", r"^[—–-]+\s*tables$", r"^[—–-]+\s*report$"],
    "Travancore":              [r"^travancore", r"anjengo"],
    "Cochin":                  [r"^cochin"],
    "Mysore":                  [r"^mysore", r"tumkur district"],
    "Hyderabad":               [r"^hyderabad", r"nizam'?s? d", r"^his highness the nizam",
                                # 'Volume XXIII. H.E.H. the Nizam's Dominions (Hyderabad State)' truncates to 'H'
                                r"^h$"],
    "Berar":                   [r"^berar"],
    "Baroda":                  [r"^baroda", r"^navsari"],
    "Kashmir":                 [r"kashmir", r"^jammu"],
    "Rajputana":               [r"^rajputana", r"^rajasthan", r"^jaipur", r"native states of rajputana"],
    "Central India Agency":    [r"central india", r"^gwalior", r"^indore", r"^dvinhya", r"^vindhya"],
    "Ajmer-Merwara":           [r"^ajmer", r"^merwara"],
    "NWFP":                    [r"north-?west frontier", r"^northwest frontier", r"^n\.?\s?-?\s?w\.?\s?f\.?\s?p?\b", r"trans-frontier"],
    "Baluchistan":             [r"^baluchistan"],
    "Coorg":                   [r"^coorg"],
    "Delhi":                   [r"^delhi"],
    "Andamans":                [r"andaman", r"port blair"],
    "Calcutta":                [r"calcutta"],
    "Western India States":    [r"western india states"],
    "French/Portuguese":       [r"french territor", r"portuguese"],
    "Sikkim":                  [r"^sikkim"],
    "Sind":                    [r"^sind\b"],   # separate province from 1936 (a SPLIT from Bombay, kept distinct)
}

# Coarse ANALYTICAL buckets from the other pipeline - OUR groupings, not real census places.
# Keyed by dash/space-normalized label; value is the fixed canonical bucket name. Checked BEFORE
# CANON so a bucket can never be swallowed by a province alias (e.g. 'burma report' vs '^burma').
# 'all-india' is the one bucket that IS a real place: it resolves to the same canonical name as
# the national volumes ('India', 'India (national)') so All-India stays one traceable place.
REGION_CANON = {
    "south india":     "South India",
    "north india":     "North India",
    "west india":      "West India",
    "east india":      "East India",
    "northeast india": "Northeast India",
    "north east india": "Northeast India",
    "all india":       "All-India",
    "burma report":    "Burma Report",
    # sub-region bucket seen in run metadata; analytical, not a census volume place
    "eastern himalayan sub region": "Eastern Himalayan Sub-Region",
}
# kept for back-compat: the raw bucket spellings used by the other pipeline
REGIONS = {"south india", "north india", "west india", "east india", "northeast india", "all-india", "burma report"}

# junk labels that carry no place information (placeholders, bare volume-part fragments)
JUNK = re.compile(r"^(pt|state|state summary|province|provincial|provincial total|unknown|\(unknown\)|,|/|-|)$")


def _region_key(l):
    """Normalize dashes/whitespace so 'All-India', 'ALL INDIA', 'North-East India' all hit REGION_CANON."""
    return re.sub(r"[\s–—-]+", " ", l).strip()


def canonicalize(label):
    """-> (canonical_name, kind) where kind: province | national | region | junk"""
    l = (label or "").strip().lower().rstrip(".")
    if JUNK.match(l):
        return None, "junk"
    rk = _region_key(l)
    if rk in REGION_CANON:
        name = REGION_CANON[rk]
        # 'All-India' the bucket == 'India'/'India (national)' the place: same canonical name,
        # kind 'region' here marks it as our bucket label; downstream treats region==national.
        return name, "region"
    for canon, pats in CANON.items():
        for p in pats:
            if re.search(p, l):
                if canon == "All-India":
                    return canon, "national"
                return canon, "province"
    return label.strip().title()[:30], "province"     # unmapped -> keep as its own (rare)
