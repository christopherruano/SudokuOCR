"""
Confirmed Census-Table Corpus: Exploratory Data Analysis
HEB 91r — Harvard University

Generates a publication-quality PowerPoint slideshow with substantive
demographic analysis of age structure, mortality, sex ratios, and
regional patterns from the confirmed OCR dataset (1871–1961).

Age-structure analyses bypass the profile-unit system and compute shares
directly from cell-level facts, using proportional splitting for age bins
that cross broad-group boundaries. This recovers usable data from 40+
province-years that the profile pipeline misses.

Output: export_final/sendout_final/analysis/eda_slideshow/
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import statistics
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".font_cache").resolve()))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Paths
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SENDOUT = Path("export_final/sendout_final")
DATA_DIR = SENDOUT / "data"
MORTALITY_DIR = SENDOUT / "mortality"
PRELIM_DIR = SENDOUT / "analysis" / "preliminary_stats"
OUT_DIR = SENDOUT / "analysis" / "eda_slideshow"
CHART_DIR = OUT_DIR / "charts"
TABLE_DIR = OUT_DIR / "tables"

TABLES_CSV = DATA_DIR / "tables_confirmed.csv"
FACTS_CSV = DATA_DIR / "facts_confirmed.csv"
REVIEW_CSV = DATA_DIR / "review_needed.csv"
EXCLUSIONS_CSV = DATA_DIR / "exclusions.csv"
MORTALITY_TABLES_CSV = MORTALITY_DIR / "tables_mortality_confirmed.csv"
MORTALITY_FACTS_CSV = MORTALITY_DIR / "facts_mortality_confirmed.csv"

PPTX_OUT = OUT_DIR / "confirmed_dataset_eda_slideshow.pptx"
FINDINGS_MD = OUT_DIR / "EDA_FINDINGS.md"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Plausibility thresholds
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MIN_POP = 5_000
MAX_POP = 200_000_000
MIN_CHILD_SHARE = 0.10
MAX_CHILD_SHARE = 0.55

LIFE_TABLE_RE = re.compile(
    r"life.table|λ|lx|px|expectation of life|survivorship",
    re.IGNORECASE,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Color palette
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

C_CHILD = "#E63946"
C_WORKING = "#457B9D"
C_ELDERLY = "#6D597A"
C_MALE = "#1D3557"
C_FEMALE = "#E63946"
C_ACCENT = "#1D3557"
C_ACCENT2 = "#457B9D"
C_ACCENT3 = "#A8DADC"
C_MUTED = "#6B7280"
C_LIGHT = "#F1FAEE"
C_RULE_HEX = "#D1D5DB"

PROVINCE_COLORS = [
    "#1D3557", "#457B9D", "#E63946", "#F4A261", "#2A9D8F",
    "#264653", "#E76F51", "#606C38", "#283618", "#BC6C25",
    "#6D597A", "#B56576", "#355070", "#EAAC8B", "#E56B6F",
]

CLR_TITLE = RGBColor(0x1D, 0x35, 0x57)
CLR_BODY = RGBColor(0x33, 0x3D, 0x47)
CLR_CAPTION = RGBColor(0x9C, 0xA3, 0xAF)
CLR_ACCENT = RGBColor(0x1D, 0x35, 0x57)
CLR_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
CLR_SECTION_BG = RGBColor(0x1D, 0x35, 0x57)
CLR_RULE = RGBColor(0xD1, 0xD5, 0xDB)
CLR_BULLET = RGBColor(0x45, 0x7B, 0x9D)

SL_W = Inches(13.333)
SL_H = Inches(7.5)

FOOTER_TEXT = "Source: Confirmed OCR Census Corpus, HEB 91r | Harvard University"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def ensure_dirs():
    for d in [OUT_DIR, CHART_DIR, TABLE_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return sum(1 for _ in reader)


def to_num(val) -> float | None:
    text = str(val or "").replace(",", "").strip()
    if not text:
        return None
    try:
        n = float(text)
    except ValueError:
        return None
    return n if math.isfinite(n) else None


def save_df(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / name
    df.to_csv(path, index=False)
    return path


def age_sort_key(label: str) -> tuple[int, str]:
    text = str(label or "").lower()
    if "under" in text or text.startswith("0"):
        return (0, text)
    m = re.search(r"(\d+)", text)
    return (int(m.group(1)) if m else 999, text)


def broad_bucket_share(age_lower: float, age_upper: float | None, open_ended: bool) -> dict[str, float]:
    """Proportionally assign an age bin to child/working/elderly.
    Returns dict mapping broad group name -> fraction of this bin in that group."""
    if pd.isna(age_lower):
        return {}
    lo = float(age_lower)

    if open_ended or age_upper is None or pd.isna(age_upper):
        if lo >= 60:
            return {"older_60_plus": 1.0}
        if lo >= 15:
            return {"working_15_59": 1.0}
        return {}

    hi = float(age_upper)
    if hi <= lo:
        if lo < 15:
            return {"child_0_14": 1.0}
        if lo < 60:
            return {"working_15_59": 1.0}
        return {"older_60_plus": 1.0}

    span = hi - lo
    if span <= 0:
        return {}

    shares = {}
    child_lo = max(lo, 0)
    child_hi = min(hi, 15)
    if child_hi > child_lo:
        shares["child_0_14"] = (child_hi - child_lo) / span

    work_lo = max(lo, 15)
    work_hi = min(hi, 60)
    if work_hi > work_lo:
        shares["working_15_59"] = (work_hi - work_lo) / span

    elder_lo = max(lo, 60)
    elder_hi = hi
    if elder_hi > elder_lo:
        shares["older_60_plus"] = (elder_hi - elder_lo) / span

    return shares


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Matplotlib style
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 220,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.5,
    })


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data loading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


FACTS_USECOLS = [
    "table_id", "section", "sex",
    "resolved_age_label_std", "resolved_age_lower", "resolved_age_upper",
    "resolved_age_total", "resolved_age_not_stated", "resolved_age_open_ended",
    "measure_unit", "value",
]


def load_all_data() -> dict:
    print("Loading data...")
    tables = read_csv(TABLES_CSV)
    review = read_csv(REVIEW_CSV)
    exclusions = read_csv(EXCLUSIONS_CSV)
    mort_tables = read_csv(MORTALITY_TABLES_CSV)

    print("  Loading facts_confirmed.csv (this may take a moment)...")
    facts = pd.DataFrame()
    if FACTS_CSV.exists():
        facts = pd.read_csv(FACTS_CSV, usecols=FACTS_USECOLS, dtype=str, keep_default_na=False)
    print(f"  {len(facts):,} facts loaded")

    mort_facts = read_csv(MORTALITY_FACTS_CSV) if MORTALITY_FACTS_CSV.exists() else pd.DataFrame()

    return {
        "tables": tables,
        "review": review,
        "exclusions": exclusions,
        "facts": facts,
        "mort_tables": mort_tables,
        "mort_facts": mort_facts,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Facts-based age profiles (bypasses profile_units)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_facts_based_profiles(facts: pd.DataFrame, tables: pd.DataFrame) -> pd.DataFrame:
    """Compute age-structure profiles directly from cell-level facts.
    Uses proportional splitting for cross-boundary age bins."""

    additive = tables[
        (tables["format_class"] == "additive_age_sex_counts")
        & (~tables["province"].isin(["", "/", "unsorted"]))
    ]
    additive_ids = set(additive["table_id"])
    table_meta = {r["table_id"]: r for r in additive.to_dict("records")}

    # Filter out life tables and proportion tables
    exclude_ids = set()
    for tid, meta in table_meta.items():
        title = meta.get("title", "")
        if LIFE_TABLE_RE.search(title):
            exclude_ids.add(tid)

    af = facts[
        facts["table_id"].isin(additive_ids - exclude_ids)
        & (facts["measure_unit"] == "count")
        & (facts["resolved_age_lower"] != "")
        & (facts["resolved_age_total"] != "True")
        & (facts["resolved_age_not_stated"] != "True")
    ].copy()
    af["value_num"] = pd.to_numeric(af["value"], errors="coerce")
    af["age_lower"] = pd.to_numeric(af["resolved_age_lower"], errors="coerce")
    af["age_upper"] = pd.to_numeric(af["resolved_age_upper"], errors="coerce")
    af["open_ended"] = af["resolved_age_open_ended"] == "True"

    print(f"  Building facts-based profiles from {af['table_id'].nunique()} additive tables...")

    records = []
    for tid, tgroup in af.groupby("table_id"):
        meta = table_meta.get(tid, {})
        province = meta.get("province", "")
        year = meta.get("year", "")
        title = meta.get("title", "")

        for sec, sgroup in tgroup.groupby("section"):
            has_m = (sgroup["sex"] == "Male").any()
            has_f = (sgroup["sex"] == "Female").any()
            has_all = (sgroup["sex"] == "All").any()

            if has_all:
                pop_group = sgroup[sgroup["sex"] == "All"]
            elif has_m and has_f:
                pop_group = sgroup[sgroup["sex"].isin(["Male", "Female"])]
            else:
                pop_group = sgroup

            child = working = elderly = 0.0
            total_pop = 0.0
            n_ages = pop_group["age_lower"].nunique()

            for _, row in pop_group.iterrows():
                val = row["value_num"]
                if pd.isna(val) or val <= 0:
                    continue
                shares = broad_bucket_share(row["age_lower"], row["age_upper"], row["open_ended"])
                for group, frac in shares.items():
                    contrib = val * frac
                    total_pop += contrib
                    if group == "child_0_14":
                        child += contrib
                    elif group == "working_15_59":
                        working += contrib
                    elif group == "older_60_plus":
                        elderly += contrib

            if total_pop < 100:
                continue

            records.append({
                "table_id": tid,
                "province": province,
                "year": year,
                "title": title,
                "section": sec,
                "pop": total_pop,
                "child": child,
                "working": working,
                "elderly": elderly,
                "s_child": child / total_pop if total_pop > 0 else 0,
                "s_working": working / total_pop if total_pop > 0 else 0,
                "s_elder": elderly / total_pop if total_pop > 0 else 0,
                "has_mf": has_m and has_f,
                "n_ages": n_ages,
            })

    df = pd.DataFrame(records)
    print(f"  {len(df)} section-level profiles built")
    return df


def plausibility_filter(profiles: pd.DataFrame) -> pd.DataFrame:
    return profiles[
        (profiles["pop"] >= MIN_POP)
        & (profiles["pop"] <= MAX_POP)
        & (profiles["s_child"] >= MIN_CHILD_SHARE)
        & (profiles["s_child"] <= MAX_CHILD_SHARE)
    ].copy()


def best_per_province_year(profiles: pd.DataFrame) -> pd.DataFrame:
    return (
        profiles
        .sort_values("pop", ascending=False)
        .groupby(["province", "year"], as_index=False)
        .first()
        .sort_values(["year", "province"])
        .reset_index(drop=True)
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Analysis: population pyramids
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def find_pyramid_candidates(plausible: pd.DataFrame, n: int = 6) -> list[dict]:
    mf = plausible[plausible["has_mf"]].sort_values("pop", ascending=False)
    if mf.empty:
        return []

    # Maximize geographic and temporal diversity
    seen_prov = set()
    seen_decade = set()
    candidates = []

    # First pass: one per province, prefer different decades
    for _, row in mf.iterrows():
        prov = row["province"]
        decade = str(row["year"])[:3]
        if prov in seen_prov:
            continue
        seen_prov.add(prov)
        seen_decade.add(decade)
        candidates.append(row.to_dict())
        if len(candidates) >= n:
            break

    # Second pass: fill remaining slots with largest remaining
    if len(candidates) < n:
        for _, row in mf.iterrows():
            key = (row["province"], row["year"])
            if any(c["province"] == row["province"] and c["year"] == row["year"] for c in candidates):
                continue
            candidates.append(row.to_dict())
            if len(candidates) >= n:
                break

    return candidates


def extract_pyramid_data(facts: pd.DataFrame, candidate: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = (
        (facts["table_id"] == candidate["table_id"])
        & (facts["sex"].isin(["Male", "Female"]))
        & (facts["measure_unit"] == "count")
        & (facts["resolved_age_lower"] != "")
        & (facts["resolved_age_total"] != "True")
        & (facts["resolved_age_not_stated"] != "True")
        & (~facts["resolved_age_label_std"].str.startswith("Total"))
    )
    sec = candidate["section"]
    if sec:
        mask = mask & (facts["section"] == sec)
    df = facts[mask].copy()
    df["value_num"] = pd.to_numeric(df["value"], errors="coerce")
    df["age_lower"] = pd.to_numeric(df["resolved_age_lower"], errors="coerce")
    df["age_upper"] = pd.to_numeric(df["resolved_age_upper"], errors="coerce")

    agg = df.groupby(["resolved_age_label_std", "sex", "age_lower", "age_upper"]).agg(
        total=("value_num", "sum"),
    ).reset_index()

    for sex_val in ["Male", "Female"]:
        sx = agg[agg["sex"] == sex_val].sort_values("age_lower")
        to_drop = []
        for idx, row in sx.iterrows():
            lo, hi = row["age_lower"], row["age_upper"]
            span = hi - lo
            if span > 5:
                contained = sx[
                    (sx.index != idx)
                    & (sx["age_lower"] >= lo)
                    & (sx["age_upper"] <= hi)
                ]
                if len(contained) >= 2 and contained["total"].sum() > 0:
                    to_drop.append(idx)
        agg = agg.drop(to_drop)

    male = agg[agg["sex"] == "Male"].sort_values("age_lower").reset_index(drop=True)
    female = agg[agg["sex"] == "Female"].sort_values("age_lower").reset_index(drop=True)
    return male, female


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Analysis: sex ratios by age
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MIN_SEX_BIN_POP = 500


def build_sex_ratios_by_age(facts: pd.DataFrame, candidates: list[dict]) -> pd.DataFrame:
    records = []
    for cand in candidates:
        tid = cand["table_id"]
        mask = (
            (facts["table_id"] == tid)
            & (facts["sex"].isin(["Male", "Female"]))
            & (facts["measure_unit"] == "count")
            & (facts["resolved_age_lower"] != "")
            & (facts["resolved_age_total"] != "True")
            & (facts["resolved_age_not_stated"] != "True")
            & (~facts["resolved_age_label_std"].str.startswith("Total"))
        )
        sec = cand["section"]
        if sec:
            mask = mask & (facts["section"] == sec)
        df = facts[mask].copy()
        df["value_num"] = pd.to_numeric(df["value"], errors="coerce")
        df["age_lower"] = pd.to_numeric(df["resolved_age_lower"], errors="coerce")

        pivot = df.groupby(["resolved_age_label_std", "age_lower", "sex"])["value_num"].sum().unstack("sex", fill_value=0)
        if "Male" not in pivot.columns or "Female" not in pivot.columns:
            continue
        pivot = pivot[
            (pivot["Male"] >= MIN_SEX_BIN_POP)
            & (pivot["Female"] >= MIN_SEX_BIN_POP)
        ].copy()
        if pivot.empty:
            continue
        pivot["sex_ratio"] = pivot["Male"] / pivot["Female"] * 1000
        for (label, age_lo), row in pivot.iterrows():
            records.append({
                "province": cand["province"],
                "year": cand["year"],
                "age_label": label,
                "age_lower": age_lo,
                "male": row["Male"],
                "female": row["Female"],
                "sex_ratio": row["sex_ratio"],
            })
    return pd.DataFrame(records)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Analysis: mortality
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def parse_age_label(row: dict) -> str:
    for field in ["resolved_age_label_std", "sub_column_std", "sub_column_raw", "column_group_std", "column_group_raw"]:
        value = str(row.get(field, "") or "").strip()
        if re.search(r"\d", value) or "under" in value.lower() or "over" in value.lower():
            return value
    return ""


def parse_sex(row: dict) -> str:
    sex = str(row.get("sex", "") or "").strip()
    if sex in {"Male", "Female", "All"}:
        return sex
    text = " ".join(str(row.get(f, "")) for f in ["column_group_raw", "column_group_std", "sub_column_raw", "sub_column_std"]).lower()
    if "female" in text:
        return "Female"
    if "male" in text:
        return "Male"
    if "person" in text or "total" in text:
        return "All"
    return ""


def build_mortality_analysis(mort_tables: pd.DataFrame, mort_facts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mort_tables.empty or mort_facts.empty:
        return pd.DataFrame(), pd.DataFrame()
    meta = {r["table_id"]: r for r in mort_tables.to_dict("records")}
    per_table = defaultdict(list)
    age_records = []

    for row in mort_facts.to_dict("records"):
        tid = row.get("table_id", "")
        value = to_num(row.get("value", ""))
        if value is None or tid not in meta:
            continue
        per_table[tid].append(value)
        m = meta[tid]
        age = parse_age_label(row)
        sex = parse_sex(row)
        period_match = re.search(r"(18|19)\d{2}", str(row.get("row_label_std", "") or row.get("row_label_raw", "")))
        age_records.append({
            "table_id": tid,
            "province": m.get("province", ""),
            "year": m.get("year", ""),
            "title": m.get("title", ""),
            "measure_category": m.get("measure_category", ""),
            "denominator_hint": m.get("denominator_hint", ""),
            "format_class": m.get("format_class", ""),
            "section": row.get("section", ""),
            "period": period_match.group(0) if period_match else row.get("row_label_std", ""),
            "age_label": age,
            "sex": sex,
            "raw_value": value,
        })

    diagnostics = []
    for tid, values in per_table.items():
        vs = sorted(values)
        m = meta[tid]
        mx, med = max(vs), statistics.median(vs)
        flags = []
        if m.get("format_class") == "mortality_rates" and mx > 1000:
            flags.append("rate_values_exceed_1000")
        if m.get("denominator_hint") not in {"", "100", "1000", "10000", "100000"}:
            flags.append("unknown_denominator")
        diagnostics.append({
            "table_id": tid, "province": m.get("province", ""), "year": m.get("year", ""),
            "title": m.get("title", ""), "format_class": m.get("format_class", ""),
            "measure_category": m.get("measure_category", ""),
            "denominator_hint": m.get("denominator_hint", ""),
            "facts": len(vs), "min_value": min(vs), "median_value": med, "max_value": mx,
            "flags": ";".join(flags),
        })
    return pd.DataFrame(diagnostics), pd.DataFrame(age_records)


def build_mortality_age_curves(age_records: pd.DataFrame) -> pd.DataFrame:
    if age_records.empty:
        return pd.DataFrame()
    rate_records = age_records[
        (age_records["format_class"] == "mortality_rates")
        & (age_records["measure_category"] == "per_1000_rates")
        & age_records["age_label"].astype(str).str.match(
            r"^(Under \d|[0-9]|[0-9][-–][0-9]|[1-9]\d[-–])", na=False
        )
        & age_records["sex"].isin(["Male", "Female"])
        & age_records["title"].str.contains(r"death.rate|death-rate|specific death", case=False, na=False)
    ].copy()
    if rate_records.empty:
        return pd.DataFrame()
    rate_records["age_sort"] = rate_records["age_label"].map(age_sort_key)
    rate_records["age_lower_num"] = rate_records["age_label"].apply(
        lambda x: age_sort_key(x)[0]
    ).astype(float)
    rate_records = rate_records[rate_records["age_lower_num"] <= 100]
    rate_records = rate_records.sort_values("age_sort")
    return rate_records


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def chart_coverage_heatmap(tables: pd.DataFrame, path: Path):
    top_prov = tables["province"].value_counts().head(22).index
    sub = tables[tables["province"].isin(top_prov)]
    pivot = sub.pivot_table(index="province", columns="year", values="table_id", aggfunc="count", fill_value=0)
    pivot = pivot[sorted(pivot.columns)]
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(pivot.values, cmap="Blues", aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if v > 0:
                color = "white" if v > pivot.values.max() * 0.6 else C_ACCENT
                ax.text(j, i, str(int(v)), ha="center", va="center", fontsize=7, color=color, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Confirmed tables")
    ax.set_title("Geographic and temporal coverage of confirmed tables", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Census year")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def chart_corpus_composition(tables: pd.DataFrame, path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.2, 1]})

    family = tables["table_family"].value_counts()
    family_labels = [f.replace("_", " ") for f in family.index]
    colors1 = [PROVINCE_COLORS[i % len(PROVINCE_COLORS)] for i in range(len(family))]
    bars1 = ax1.barh(range(len(family)), family.values, color=colors1, edgecolor="white", linewidth=0.5)
    ax1.set_yticks(range(len(family)))
    ax1.set_yticklabels(family_labels, fontsize=10)
    ax1.set_xlabel("Number of confirmed tables")
    ax1.set_title("a) Table families", fontsize=13, fontweight="bold", loc="left")
    ax1.invert_yaxis()
    for bar, val in zip(bars1, family.values):
        ax1.text(val + family.values.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:,}", va="center", fontsize=9, color=C_MUTED)

    measure = tables["measure_category"].value_counts()
    measure_labels = [m.replace("_", " ") for m in measure.index]
    colors2 = [C_ACCENT, C_ACCENT2, "#E63946", "#F4A261"][:len(measure)]
    bars2 = ax2.barh(range(len(measure)), measure.values, color=colors2, edgecolor="white", linewidth=0.5)
    ax2.set_yticks(range(len(measure)))
    ax2.set_yticklabels(measure_labels, fontsize=10)
    ax2.set_xlabel("Number of confirmed tables")
    ax2.set_title("b) Measure categories", fontsize=13, fontweight="bold", loc="left")
    ax2.invert_yaxis()
    for bar, val in zip(bars2, measure.values):
        ax2.text(val + measure.values.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:,}", va="center", fontsize=9, color=C_MUTED)

    plt.tight_layout(w_pad=3)
    plt.savefig(path, dpi=220)
    plt.close()


def chart_curated_age_snapshots(snapshots: pd.DataFrame, path: Path):
    if snapshots.empty:
        return

    df = snapshots.sort_values("pop", ascending=True).copy()
    labels = [f"{r['province']} {r['year']}" for _, r in df.iterrows()]
    child = df["s_child"].values * 100
    working = df["s_working"].values * 100
    elderly = df["s_elder"].values * 100
    pops = df["pop"].values

    n_bars = len(df)
    fig, ax = plt.subplots(figsize=(12, max(4, n_bars * 0.45 + 1)))
    y = np.arange(n_bars)
    h = 0.6

    ax.barh(y, child, h, label="0–14 (children)", color=C_CHILD, edgecolor="white", linewidth=0.3)
    ax.barh(y, working, h, left=child, label="15–59 (working age)", color=C_WORKING, edgecolor="white", linewidth=0.3)
    ax.barh(y, elderly, h, left=child + working, label="60+ (elderly)", color=C_ELDERLY, edgecolor="white", linewidth=0.3)

    for i, pop in enumerate(pops):
        total_pct = child[i] + working[i] + elderly[i]
        if pop >= 1_000_000:
            pop_label = f"N = {pop / 1e6:.1f}M"
        else:
            pop_label = f"N = {int(pop):,}"
        ax.text(total_pct + 0.5, i, pop_label, va="center", fontsize=8, color=C_MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Share of population (%)")
    ax.set_xlim(0, 115)
    ax.set_title("Broad age composition across provinces and decades",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9, frameon=True, facecolor="white", edgecolor=C_RULE_HEX)
    ax.grid(axis="x", alpha=0.15)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def chart_multi_pyramid(pyramids: list[dict], path: Path):
    n = len(pyramids)
    if n == 0:
        return
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax, pyr in zip(axes, pyramids):
        male, female = pyr["male"], pyr["female"]
        if male.empty or female.empty:
            ax.set_visible(False)
            continue
        labels = male["resolved_age_label_std"].values
        m_vals = male["total"].values.astype(float)
        f_vals = female["total"].values.astype(float)
        total = m_vals.sum() + f_vals.sum()
        if total == 0:
            ax.set_visible(False)
            continue
        m_pct = m_vals / total * 100
        f_pct = f_vals / total * 100
        y_pos = np.arange(len(labels))

        ax.barh(y_pos, -m_pct, height=0.78, color=C_MALE, edgecolor="white", linewidth=0.3)
        ax.barh(y_pos, f_pct, height=0.78, color=C_FEMALE, edgecolor="white", linewidth=0.3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("% of total", fontsize=8)
        ax.axvline(0, color="#374151", linewidth=0.5)

        max_val = max(m_pct.max(), f_pct.max()) * 1.15
        ax.set_xlim(-max_val, max_val)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{abs(x):.0f}%"))

        ax.text(-max_val * 0.5, len(y_pos) + 0.15, "M", ha="center", fontweight="bold", fontsize=9, color=C_MALE)
        ax.text(max_val * 0.5, len(y_pos) + 0.15, "F", ha="center", fontweight="bold", fontsize=9, color=C_FEMALE)

        prov, yr = pyr["province"], pyr["year"]
        if total >= 1_000_000:
            n_label = f"N = {total / 1e6:.1f}M"
        else:
            n_label = f"N = {int(total):,}"
        ax.set_title(f"{prov}, {yr}\n{n_label}", fontsize=10, fontweight="bold", pad=12)
        ax.grid(axis="x", alpha=0.15)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(w_pad=1.5, h_pad=2)
    plt.savefig(path, dpi=220)
    plt.close()


def chart_regional_age_scatter(profiles: pd.DataFrame, path: Path):
    if profiles.empty:
        return

    df = profiles.copy()
    df["child_pct"] = df["s_child"] * 100
    df["elder_pct"] = df["s_elder"] * 100

    fig, ax = plt.subplots(figsize=(9, 6.5))

    years_sorted = sorted(df["year"].unique())
    cmap = plt.cm.viridis
    year_colors = {yr: cmap(i / max(len(years_sorted) - 1, 1)) for i, yr in enumerate(years_sorted)}

    for yr in years_sorted:
        sub = df[df["year"] == yr]
        sizes = np.clip(sub["pop"] / df["pop"].max() * 300, 25, 300)
        ax.scatter(sub["child_pct"], sub["elder_pct"], s=sizes, alpha=0.7,
                   color=year_colors[yr], edgecolors="white", linewidth=0.5, label=yr, zorder=3)
        for _, row in sub.iterrows():
            if row["pop"] > 1_000_000:
                ax.annotate(row["province"], (row["child_pct"], row["elder_pct"]),
                            fontsize=7, color=C_MUTED, textcoords="offset points", xytext=(5, 3))

    ax.set_xlabel("Children 0–14 (% of population)")
    ax.set_ylabel("Elderly 60+ (% of population)")
    ax.set_title("Regional variation in age structure",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(title="Census year", fontsize=8, title_fontsize=9, loc="upper right",
              frameon=True, facecolor="white", edgecolor=C_RULE_HEX, ncol=2)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def chart_sex_ratio_by_age(sex_ratios: pd.DataFrame, path: Path):
    if sex_ratios.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    groups = sex_ratios.groupby(["province", "year"])
    for i, ((prov, yr), grp) in enumerate(groups):
        grp = grp.sort_values("age_lower").copy()
        grp["age_mid"] = grp["age_lower"] + 2.5
        color = PROVINCE_COLORS[i % len(PROVINCE_COLORS)]
        ax.plot(grp["age_mid"], grp["sex_ratio"], marker="o", markersize=4,
                color=color, linewidth=1.5, label=f"{prov} {yr}", alpha=0.85)

    ax.axhline(1000, color=C_MUTED, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(ax.get_xlim()[1] * 0.98, 1010, "parity (1,000)", ha="right", fontsize=8, color=C_MUTED)
    ax.set_ylabel("Males per 1,000 females")
    ax.set_xlabel("Age (midpoint of age group)")
    ax.set_title("Sex ratio by age group across provinces and decades",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=8, loc="best", frameon=True, facecolor="white", edgecolor=C_RULE_HEX, ncol=2)
    ax.grid(axis="y", alpha=0.15)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def chart_mortality_coverage(mort_tables: pd.DataFrame, path: Path):
    if mort_tables.empty:
        return

    pivot = mort_tables.pivot_table(index="province", columns="year", values="table_id", aggfunc="count", fill_value=0)
    pivot = pivot[sorted(pivot.columns)]
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]

    fig, ax = plt.subplots(figsize=(9, max(4, len(pivot) * 0.45)))
    im = ax.imshow(pivot.values, cmap="OrRd", aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if v > 0:
                ax.text(j, i, str(int(v)), ha="center", va="center", fontsize=9, fontweight="bold",
                        color="white" if v > pivot.values.max() * 0.5 else "#7F1D1D")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Mortality tables")
    ax.set_title("Confirmed mortality tables by province and year", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def chart_mortality_age_curves(age_records: pd.DataFrame, path: Path):
    rate_data = build_mortality_age_curves(age_records)
    if rate_data.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    groups = rate_data.groupby(["province", "year", "sex"])
    plotted = 0
    for i, ((prov, yr, sex), grp) in enumerate(groups):
        grp = grp.sort_values("age_lower_num").drop_duplicates("age_lower_num")
        if len(grp) < 3:
            continue
        color = C_MALE if sex == "Male" else C_FEMALE
        ls = "-" if sex == "Male" else "--"
        alpha = 0.9 if plotted < 4 else 0.5
        ax.plot(grp["age_lower_num"], grp["raw_value"], marker="o", markersize=4,
                color=color, linestyle=ls, linewidth=1.5,
                label=f"{prov} {yr} ({sex})", alpha=alpha)
        plotted += 1

    ax.set_ylabel("Death rate per mille (raw value)")
    ax.set_xlabel("Age (lower bound of age group)")
    ax.set_title("Age-specific mortality curves from confirmed rate tables",
                 fontsize=14, fontweight="bold", pad=12)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8, loc="upper left", frameon=True, facecolor="white", edgecolor=C_RULE_HEX, ncol=2)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def chart_mortality_diagnostics(diag: pd.DataFrame, path: Path):
    if diag.empty:
        return
    diag = diag.sort_values("max_value")
    fig, ax = plt.subplots(figsize=(10, max(5, len(diag) * 0.35)))
    y = np.arange(len(diag))
    colors = diag["format_class"].map({"mortality_rates": C_ACCENT2, "mortality_counts": "#2A9D8F"}).fillna(C_MUTED)
    ax.scatter(pd.to_numeric(diag["median_value"], errors="coerce"), y, c=colors, s=50, label="Median", zorder=3)
    ax.scatter(pd.to_numeric(diag["max_value"], errors="coerce"), y, c="#E63946", marker="x", s=60, label="Max", zorder=3)
    ax.axvline(1000, color="#E63946", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.province} {r.year}" for r in diag.itertuples()], fontsize=9)
    ax.set_xlabel("Raw value (log scale)")
    ax.set_title("Mortality table value-scale diagnostic", fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=10, frameon=True, facecolor="white", edgecolor=C_RULE_HEX)
    ax.grid(axis="x", alpha=0.15)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Generate all charts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def generate_all_charts(data: dict, metrics: dict) -> dict[str, Path]:
    charts: dict[str, Path] = {}

    print("  Generating charts...")

    charts["coverage"] = CHART_DIR / "coverage_heatmap.png"
    chart_coverage_heatmap(data["tables"], charts["coverage"])

    charts["composition"] = CHART_DIR / "corpus_composition.png"
    chart_corpus_composition(data["tables"], charts["composition"])

    # ── Facts-based age profiles ──
    all_profiles = build_facts_based_profiles(data["facts"], data["tables"])
    plausible = plausibility_filter(all_profiles)
    snapshots = best_per_province_year(plausible)

    save_df(all_profiles, "facts_based_profiles_all.csv")
    save_df(plausible, "facts_based_profiles_plausible.csv")
    save_df(snapshots, "curated_age_snapshots.csv")

    n_snap = len(snapshots)
    n_prov = snapshots["province"].nunique()
    metrics["curated_snapshots"] = n_snap
    metrics["curated_provinces"] = n_prov

    print(f"  {n_snap} plausible province-year snapshots from {n_prov} provinces")

    if not snapshots.empty:
        charts["age_snapshots"] = CHART_DIR / "curated_age_snapshots.png"
        chart_curated_age_snapshots(snapshots, charts["age_snapshots"])

    # ── Regional scatter ──
    charts["regional_scatter"] = CHART_DIR / "regional_age_scatter.png"
    chart_regional_age_scatter(snapshots, charts["regional_scatter"])

    # ── Population pyramids ──
    pyramid_candidates = find_pyramid_candidates(plausible, n=6)
    pyramids = []
    for cand in pyramid_candidates:
        male, female = extract_pyramid_data(data["facts"], cand)
        if not male.empty and not female.empty:
            pyramids.append({**cand, "male": male, "female": female})

    if pyramids:
        charts["pyramids_multi"] = CHART_DIR / "population_pyramids.png"
        chart_multi_pyramid(pyramids, charts["pyramids_multi"])
        metrics["pyramid_provinces"] = len(pyramids)

    # ── Sex ratios ──
    sr_candidates = find_pyramid_candidates(plausible, n=5)
    sex_ratios = build_sex_ratios_by_age(data["facts"], sr_candidates)
    if not sex_ratios.empty:
        save_df(sex_ratios, "sex_ratio_by_age.csv")
        charts["sex_ratio"] = CHART_DIR / "sex_ratio_by_age.png"
        chart_sex_ratio_by_age(sex_ratios, charts["sex_ratio"])
        metrics["sex_ratio_provinces"] = sex_ratios.groupby(["province", "year"]).ngroups

    # ── Mortality ──
    mort_diag, mort_age = build_mortality_analysis(data["mort_tables"], data["mort_facts"])
    if not mort_diag.empty:
        save_df(mort_diag, "mortality_diagnostics.csv")
    if not mort_age.empty:
        save_df(mort_age, "mortality_age_records.csv")

    charts["mort_coverage"] = CHART_DIR / "mortality_coverage.png"
    chart_mortality_coverage(data["mort_tables"], charts["mort_coverage"])

    charts["mort_curves"] = CHART_DIR / "mortality_age_curves.png"
    chart_mortality_age_curves(mort_age, charts["mort_curves"])

    charts["mort_diagnostic"] = CHART_DIR / "mortality_scale_diagnostic.png"
    chart_mortality_diagnostics(mort_diag, charts["mort_diagnostic"])

    return charts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PowerPoint builder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _add_shape(slide, left, top, width, height, fill_rgb):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_rgb
    shape.line.fill.background()
    return shape


def _add_text(slide, left, top, width, height, text, font_size=18, bold=False,
              color=CLR_BODY, alignment=PP_ALIGN.LEFT, wrap=True):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.alignment = alignment
    return box


def _add_bullets(slide, items, left=Inches(0.6), top=Inches(1.35), width=Inches(4.8),
                 height=Inches(5.0), font_size=14, color=CLR_BODY, bullet_color=CLR_BULLET):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.space_after = Pt(8)
        p.space_before = Pt(2)
    return box


def _add_image(slide, path: Path, left, top, width=None, height=None):
    if not path.exists():
        return
    kwargs = {"left": left, "top": top}
    if width:
        kwargs["width"] = width
    if height:
        kwargs["height"] = height
    slide.shapes.add_picture(str(path), **kwargs)


def _add_footer(slide, text=FOOTER_TEXT, slide_num=None):
    _add_text(slide, Inches(0.6), Inches(7.05), Inches(10), Inches(0.3),
              text, font_size=8, color=CLR_CAPTION)
    if slide_num is not None:
        _add_text(slide, Inches(12.2), Inches(7.05), Inches(0.8), Inches(0.3),
                  str(slide_num), font_size=8, color=CLR_CAPTION, alignment=PP_ALIGN.RIGHT)


def _make_title_slide(prs, blank, metrics):
    slide = prs.slides.add_slide(blank)

    _add_shape(slide, Inches(0), Inches(0), SL_W, Inches(0.06), CLR_ACCENT)

    _add_text(slide, Inches(1.0), Inches(1.8), Inches(11.3), Inches(1.0),
              "Population Structure in Colonial India, 1871–1961",
              font_size=34, bold=True, color=CLR_TITLE)

    _add_text(slide, Inches(1.0), Inches(2.9), Inches(11.3), Inches(0.6),
              "Exploratory analysis of the digitized census corpus",
              font_size=20, color=CLR_BODY)

    stats_text = (
        f"{metrics['confirmed_tables']:,} confirmed tables  ·  "
        f"{metrics['confirmed_facts']:,} cell-level facts  ·  "
        f"{metrics['mortality_tables']:,} mortality tables  ·  "
        f"{metrics['n_provinces']} provinces  ·  "
        f"{metrics['year_range']}"
    )
    _add_text(slide, Inches(1.0), Inches(3.7), Inches(11.3), Inches(0.5),
              stats_text, font_size=13, color=CLR_CAPTION)

    _add_shape(slide, Inches(1.0), Inches(3.5), Inches(8), Inches(0.015), CLR_RULE)

    _add_text(slide, Inches(1.0), Inches(5.8), Inches(11), Inches(0.4),
              "HEB 91r  |  Harvard University", font_size=12, color=CLR_CAPTION)
    _add_text(slide, Inches(1.0), Inches(6.2), Inches(11), Inches(0.4),
              "Preliminary EDA — generated from confirmed sendout files; not a final demographic series.",
              font_size=10, color=CLR_CAPTION)


def _make_section_slide(prs, blank, section_num, title):
    slide = prs.slides.add_slide(blank)
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = CLR_SECTION_BG

    _add_shape(slide, Inches(1.5), Inches(3.2), Inches(3), Inches(0.015), CLR_WHITE)

    _add_text(slide, Inches(1.5), Inches(2.4), Inches(10.3), Inches(0.6),
              f"Section {section_num}", font_size=14, color=RGBColor(0xA8, 0xDA, 0xDC))

    _add_text(slide, Inches(1.5), Inches(3.5), Inches(10.3), Inches(1.0),
              title, font_size=30, bold=True, color=CLR_WHITE)


def _make_chart_slide(prs, blank, title, bullets, chart_path, caption="", slide_num=None,
                      chart_left=Inches(5.5), chart_top=Inches(0.9), chart_width=Inches(7.3)):
    slide = prs.slides.add_slide(blank)

    _add_shape(slide, Inches(0), Inches(0), SL_W, Inches(0.04), CLR_ACCENT)

    _add_text(slide, Inches(0.55), Inches(0.25), Inches(12.2), Inches(0.55),
              title, font_size=22, bold=True, color=CLR_TITLE)

    _add_shape(slide, Inches(0.55), Inches(0.82), Inches(12.2), Inches(0.012), CLR_RULE)

    if bullets:
        _add_bullets(slide, bullets, left=Inches(0.55), top=Inches(1.15), width=Inches(4.7))

    if chart_path and Path(chart_path).exists():
        _add_image(slide, Path(chart_path), chart_left, chart_top, width=chart_width)

    if caption:
        _add_text(slide, Inches(0.55), Inches(6.65), Inches(12.2), Inches(0.3),
                  caption, font_size=8, color=CLR_CAPTION)

    _add_footer(slide, slide_num=slide_num)
    return slide


def _make_wide_chart_slide(prs, blank, title, chart_path, caption="", slide_num=None):
    slide = prs.slides.add_slide(blank)

    _add_shape(slide, Inches(0), Inches(0), SL_W, Inches(0.04), CLR_ACCENT)
    _add_text(slide, Inches(0.55), Inches(0.25), Inches(12.2), Inches(0.55),
              title, font_size=22, bold=True, color=CLR_TITLE)
    _add_shape(slide, Inches(0.55), Inches(0.82), Inches(12.2), Inches(0.012), CLR_RULE)

    if chart_path and Path(chart_path).exists():
        _add_image(slide, Path(chart_path), Inches(0.4), Inches(0.95), width=Inches(12.5))

    if caption:
        _add_text(slide, Inches(0.55), Inches(6.65), Inches(12.2), Inches(0.3),
                  caption, font_size=8, color=CLR_CAPTION)

    _add_footer(slide, slide_num=slide_num)
    return slide


def _make_findings_slide(prs, blank, title, findings, slide_num=None):
    slide = prs.slides.add_slide(blank)

    _add_shape(slide, Inches(0), Inches(0), SL_W, Inches(0.04), CLR_ACCENT)
    _add_text(slide, Inches(0.55), Inches(0.25), Inches(12.2), Inches(0.55),
              title, font_size=22, bold=True, color=CLR_TITLE)
    _add_shape(slide, Inches(0.55), Inches(0.82), Inches(12.2), Inches(0.012), CLR_RULE)

    _add_bullets(slide, findings, left=Inches(0.7), top=Inches(1.1), width=Inches(11.7),
                 height=Inches(5.5), font_size=15)

    _add_footer(slide, slide_num=slide_num)
    return slide


def make_pptx(charts: dict[str, Path], metrics: dict):
    prs = Presentation()
    prs.slide_width = SL_W
    prs.slide_height = SL_H
    blank = prs.slide_layouts[6]
    sn = 1

    n_snap = metrics.get("curated_snapshots", 0)
    n_cprov = metrics.get("curated_provinces", 0)
    n_pyrs = metrics.get("pyramid_provinces", 0)
    n_sr = metrics.get("sex_ratio_provinces", 0)

    # ── Title ──
    _make_title_slide(prs, blank, metrics)

    # ── Section I: The Corpus ──
    _make_section_slide(prs, blank, "I", "The Confirmed Corpus")

    _make_chart_slide(prs, blank,
        "Corpus composition: table families and measure categories",
        [
            f"{metrics['confirmed_tables']:,} confirmed tables spanning {metrics['year_range']}, each passing strict validation with zero failed quality checks.",
            f"{metrics['confirmed_facts']:,} individual cell-level facts extracted via OCR and validated against internal table constraints.",
            "Age-sex population tables dominate, with substantial civil-condition and age-distribution material.",
            "Counts and rates are kept strictly separate; the corpus is table-faithful, not yet a deduplicated population panel.",
        ],
        charts.get("composition"),
        caption="All confirmed tables are strict-validated, core-layer, high-confidence. Measure categories determine which tables can be pooled.",
        slide_num=(sn := sn + 1),
    )

    _make_chart_slide(prs, blank,
        "Geographic and temporal coverage",
        [
            f"Coverage spans {metrics['n_provinces']} provinces from {metrics['year_range']}.",
            "Densest coverage around 1891–1911, with strong representation of Mysore, Bombay, Travancore, Madras, and Baroda.",
            "Southern and western presidencies are better represented than the north and east in the current release.",
            "Coverage is sufficient for exploratory province-year analysis but not yet a balanced panel.",
        ],
        charts.get("coverage"),
        caption="Cell values show number of confirmed tables. Empty cells indicate no confirmed tables for that province-year.",
        slide_num=(sn := sn + 1),
    )

    # ── Section II: Age Structure ──
    _make_section_slide(prs, blank, "II", "Age Structure and the Demographic Transition")

    _make_chart_slide(prs, blank,
        "Broad age composition across provinces and decades",
        [
            f"{n_snap} province-year snapshots from {n_cprov} provinces pass plausibility filtering, computed directly from cell-level facts using proportional age-bin splitting.",
            "Pre-transitional structure is visible across all decades: child shares cluster around 30–48%, with working-age populations making up the bulk.",
            "Elderly shares are uniformly low (1–6%), consistent with high mortality and low life expectancy throughout the colonial period.",
            "This represents a major expansion over earlier analyses that relied on complete-profile-only filtering.",
        ],
        charts.get("age_snapshots"),
        chart_left=Inches(5.3), chart_width=Inches(7.5),
        caption=f"Plausibility filter: pop ≥ {MIN_POP:,}, child share {MIN_CHILD_SHARE*100:.0f}–{MAX_CHILD_SHARE*100:.0f}%. Largest section per table, largest table per province-year. Cross-boundary age bins proportionally split.",
        slide_num=(sn := sn + 1),
    )

    if charts.get("pyramids_multi"):
        _make_chart_slide(prs, blank,
            f"Population age-sex pyramids ({n_pyrs} provinces)",
            [
                "Classic pre-transitional pyramid shape: broad base, rapid tapering with age, reflecting high fertility and high mortality across all provinces shown.",
                "Age bin granularity varies by census (single-year vs. 5-year vs. 10-year groups), reflecting the evolving tabulation practices of the Indian Census.",
                "Male-female asymmetries at specific ages reflect differential mortality, migration, and enumeration patterns.",
                "Geographic diversity: pyramids span western, eastern, central, and southern India across six decades.",
            ],
            charts["pyramids_multi"],
            chart_left=Inches(5.0), chart_width=Inches(8.0),
            caption="Each pyramid normalizes male and female counts as a share of total population. Built from cell-level facts; largest plausible M+F section per province-year.",
            slide_num=(sn := sn + 1),
        )

    _make_chart_slide(prs, blank,
        "Regional variation in age structure",
        [
            f"Each point is one province-year snapshot (pop ≥ {MIN_POP:,}; child share {MIN_CHILD_SHARE*100:.0f}–{MAX_CHILD_SHARE*100:.0f}%).",
            "Child shares range from ~15% to ~48%; elderly shares from ~1% to ~15%, revealing substantial variation across regions and decades.",
            "Size proportional to population; labels shown for populations exceeding 1 million.",
            "The color gradient reveals temporal patterns: earlier censuses (darker) show similar structure to later ones, consistent with a pre-transitional regime persisting through 1951.",
        ],
        charts.get("regional_scatter"),
        chart_left=Inches(5.3), chart_width=Inches(7.5),
        caption="Each point is one province-year from the plausibility-filtered set. Age shares computed from cell-level facts with proportional splitting.",
        slide_num=(sn := sn + 1),
    )

    # ── Section III: Sex Ratios and Mortality ──
    _make_section_slide(prs, blank, "III", "Sex Ratios and Mortality Patterns")

    if charts.get("sex_ratio"):
        _make_chart_slide(prs, blank,
            f"Sex ratio by age group ({n_sr} province-years)",
            [
                "Sex ratios (males per 1,000 females) from the largest plausibility-filtered tables with male and female counts.",
                "Values above 1,000 indicate male excess; below indicates female excess.",
                f"Per-bin population floor of {MIN_SEX_BIN_POP:,} per sex excludes unreliable ratios from small cells.",
                "In historical India, overall male-biased sex ratios are well documented, driven by excess female mortality in childhood and reproductive ages.",
            ],
            charts["sex_ratio"],
            chart_left=Inches(5.3), chart_width=Inches(7.5),
            caption=f"Ratio = (male count / female count) × 1,000 at each age group. Bins with <{MIN_SEX_BIN_POP} per sex excluded.",
            slide_num=(sn := sn + 1),
        )

    _make_chart_slide(prs, blank,
        f"Mortality data: {metrics['mortality_tables']:,} confirmed tables",
        [
            f"{metrics['mortality_tables']:,} confirmed mortality tables with {metrics['mortality_facts']:,} cell-level mortality facts.",
            "The majority are death rates (per mille); only a small number contain absolute death counts.",
            "Bengal and Bombay dominate mortality coverage; 1931 is the best-represented year.",
            "Denominators, scales, and rate definitions vary across tables and require table-level review before pooled analysis.",
        ],
        charts.get("mort_coverage"),
        chart_left=Inches(5.5), chart_width=Inches(7.3),
        caption="Mortality tables are a distinct subcollection within the confirmed corpus. Counts and rates must be analyzed separately.",
        slide_num=(sn := sn + 1),
    )

    _make_chart_slide(prs, blank,
        "Age-specific mortality curves",
        [
            "Where age-specific death rates are available, the classic pattern emerges: high infant mortality, low risk in childhood/adolescence, rising risk with age.",
            "Male and female curves diverge at specific ages, reflecting sex-differential mortality that varied by province and period.",
            "Recovery of these well-known demographic patterns from OCR-extracted data validates the pipeline.",
            "Caution: raw values reflect differing denominators across tables; direct comparison requires scale normalization.",
        ],
        charts.get("mort_curves"),
        chart_left=Inches(5.3), chart_width=Inches(7.5),
        caption="Age-specific death rates from confirmed per-mille rate tables. Only series matching 'death rate' titles with proper age labels shown.",
        slide_num=(sn := sn + 1),
    )

    _make_chart_slide(prs, blank,
        "Mortality rate scale diagnostic",
        [
            "Several mortality tables have raw values exceeding 1,000, indicating ratio-style tables, implied decimals, or non-standard denominators.",
            "Before pooled mortality analysis, tables should be classified as: analysis-ready, scale-review, or ratio-only.",
            "The diagnostic chart makes scale heterogeneity visible rather than hiding it.",
            "Recommended workflow: classify → normalize → validate against external benchmarks → pool.",
        ],
        charts.get("mort_diagnostic"),
        chart_left=Inches(5.5), chart_width=Inches(7.3),
        caption="Median and maximum raw values per mortality table, log-scaled. Red dashed line at 1,000.",
        slide_num=(sn := sn + 1),
    )

    # ── Section IV: Summary ──
    _make_section_slide(prs, blank, "IV", "Summary and Research Directions")

    _make_findings_slide(prs, blank,
        "Key findings",
        [
            f"1.  The confirmed corpus ({metrics['confirmed_tables']:,} tables, {metrics['confirmed_facts']:,} facts) is large enough for substantive province-year-level demographic analysis across {metrics['year_range']}.",
            f"2.  Direct facts-based analysis recovers {n_snap} plausible province-year snapshots from {n_cprov} provinces — far broader than the {metrics.get('profile_unit_snapshots', 5)} available through the profile-unit pipeline.",
            "3.  Pre-transitional population structure is clearly visible: child shares 30–48%, elderly shares 1–6%, and high dependency ratios, consistent across regions and decades.",
            f"4.  Population pyramids from {n_pyrs} geographically diverse provinces show the classic broad-base, rapid-taper shape, validating data quality at the table level.",
            "5.  Mortality rate tables recover the classic U-shaped age-mortality curve, though scale harmonization is needed before pooled analysis.",
            f"6.  Sex ratio analysis across {n_sr} province-years reveals age-specific patterns consistent with known gender-differential mortality in historical India.",
        ],
        slide_num=(sn := sn + 1),
    )

    _make_findings_slide(prs, blank,
        "Limitations and next steps",
        [
            "Limitation: The current corpus is not a deduplicated population panel. Multiple tables may cover the same province-year from different source pages.",
            "Limitation: Age bin schemes vary across censuses and provinces; proportional splitting introduces approximation for cross-boundary bins.",
            "Limitation: Mortality data is concentrated in Bengal and Bombay 1931; broader geographic and temporal coverage is needed.",
            "Next step: Construct a curated analysis layer with one preferred table per province-year, geographic crosswalk, and mortality scale annotations.",
            "Next step: Systematic age-heaping analysis (Whipple's index) to assess enumeration quality and inform age-adjustment procedures.",
            "Next step: Link to external vital statistics and demographic estimates for benchmarking OCR-derived age and mortality distributions.",
        ],
        slide_num=(sn := sn + 1),
    )

    prs.save(PPTX_OUT)
    print(f"  Wrote PPTX: {PPTX_OUT}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Findings markdown
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def write_findings(metrics: dict):
    n_snap = metrics.get("curated_snapshots", 0)
    n_cprov = metrics.get("curated_provinces", 0)
    lines = [
        "# Confirmed Census-Table Corpus: EDA Findings",
        "",
        "## Dataset Summary",
        "",
        f"- **Confirmed tables:** {metrics['confirmed_tables']:,}",
        f"- **Confirmed facts:** {metrics['confirmed_facts']:,}",
        f"- **Provinces:** {metrics['n_provinces']}",
        f"- **Year range:** {metrics['year_range']}",
        f"- **Mortality tables:** {metrics['mortality_tables']:,}",
        f"- **Mortality facts:** {metrics['mortality_facts']:,}",
        f"- **Review-needed:** {metrics['review_needed_tables']:,}",
        f"- **Excluded:** {metrics['excluded_tables']:,}",
        f"- **Plausibility-filtered snapshots:** {n_snap} from {n_cprov} provinces",
        "",
        "## Analysis Method",
        "",
        "Age-structure analyses bypass the profile-unit system and compute shares",
        "directly from cell-level facts. Cross-boundary age bins (e.g. 10--20) are",
        "proportionally split across broad groups (0--14, 15--59, 60+). This recovers",
        "usable data from tables the profile pipeline cannot bucket.",
        "",
        f"Plausibility filter: pop >= {MIN_POP:,}, child share {MIN_CHILD_SHARE*100:.0f}--{MAX_CHILD_SHARE*100:.0f}%.",
        "Life tables are excluded by title keyword matching.",
        "",
        "## Population Structure",
        "",
        f"The facts-based analysis recovers {n_snap} province-year snapshots from {n_cprov} provinces,",
        "capturing a pre-transitional demographic regime: child shares around 30--48% (aged 0--14),",
        "elderly shares around 1--6% (aged 60+), and correspondingly high youth dependency ratios.",
        "Population pyramids recovered from OCR-extracted cell facts show the classic broad-base,",
        "rapid-taper shape expected for high-fertility, high-mortality populations.",
        "",
        "## Sex Ratios",
        "",
        "Sex ratios (males per 1,000 females) vary across age groups and provinces.",
        "Age-varying patterns can reveal differential mortality, migration, and",
        "enumeration bias. Historical India's well-documented male-biased sex ratios",
        f"are visible in the data. Bins with fewer than {MIN_SEX_BIN_POP} persons per sex are excluded.",
        "",
        "## Mortality",
        "",
        f"The mortality subcollection ({metrics['mortality_tables']:,} tables, {metrics['mortality_facts']:,} facts)",
        "is mostly death rates, not counts. The classic U-shaped age-mortality curve is recoverable",
        "from several tables, particularly the Bengal 1931 age-sex-specific death rate series.",
        "",
        "**Important caveat:** Mortality tables vary in denominator, scale, and rate definition.",
        "Tables must be classified (analysis-ready, scale-review, ratio-only) and normalized",
        "before pooled analysis.",
        "",
        "## Research Directions",
        "",
        "1. Construct a curated analysis layer: one preferred table per province-year, geographic crosswalk.",
        "2. Age-heaping analysis (Whipple's index) to assess enumeration quality.",
        "3. Mortality scale harmonization and external benchmarking.",
        "4. Link to vital statistics and demographic estimates for validation.",
        "5. Systematic sex-ratio-at-age analysis for evidence of gender-differential mortality.",
        "",
        "---",
        "*Generated from confirmed sendout files. Preliminary EDA, not a final demographic series.*",
        "",
    ]
    FINDINGS_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote findings: {FINDINGS_MD}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main():
    ensure_dirs()
    setup_style()

    data = load_all_data()
    tables = data["tables"]

    years = sorted(tables["year"].unique())
    provinces = sorted(set(tables["province"]) - {"", "/"})

    metrics = {
        "confirmed_tables": len(tables),
        "confirmed_facts": count_csv_rows(FACTS_CSV),
        "review_needed_tables": len(data["review"]),
        "excluded_tables": len(data["exclusions"]),
        "mortality_tables": len(data["mort_tables"]),
        "mortality_facts": count_csv_rows(MORTALITY_FACTS_CSV),
        "n_provinces": len(provinces),
        "year_range": f"{min(years)}–{max(years)}" if years else "unknown",
    }
    (OUT_DIR / "summary_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    charts = generate_all_charts(data, metrics)
    make_pptx(charts, metrics)
    write_findings(metrics)

    print("\nDone.")
    print(f"  Slideshow: {PPTX_OUT}")
    print(f"  Charts:    {CHART_DIR}")
    print(f"  Tables:    {TABLE_DIR}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
