#!/usr/bin/env python3
"""Dogfood the Census Explorer through its HTTP API (not the DB directly) — run the
standard population-study questions, build charts, and flag weird results.

Outputs to sendout_v1/dogfood/:  *.png charts  +  anomalies.txt
Requires the server running at http://127.0.0.1:5000 (census_explorer.py).
"""
import io, os, urllib.parse, urllib.request
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "http://127.0.0.1:5000"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "export_final/sendout_final/sendout_v1/dogfood")
os.makedirs(OUT, exist_ok=True)
ANOM = []


def pull(**filters):
    """Pull a combined CSV from the explorer (the exact thing a user downloads)."""
    qs = urllib.parse.urlencode(filters, doseq=True)
    with urllib.request.urlopen(f"{BASE}/facts.csv?{qs}", timeout=120) as r:
        df = pd.read_csv(io.BytesIO(r.read()), low_memory=False)
    yr = pd.to_numeric(df["year"], errors="coerce")
    df["decade"] = (yr // 10 * 10).astype("Int64").astype(str) + "s"
    return df


def flag(msg):
    ANOM.append(msg)
    print("  ⚠", msg)


def numeric(df):
    df = df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


# Q1 — corpus coverage: facts by region x decade -----------------------------
def q1_coverage():
    df = pull(topic=["age_sex_population", "age_sex_civil_condition", "age_distribution",
                     "age_by_religion", "age_civil_condition", "age_sex_marital_status"])
    print(f"Q1 coverage: pulled {len(df):,} facts")
    piv = df.pivot_table(index="broad_region", columns="decade", values="fact_id",
                         aggfunc="count", fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_title("Corpus coverage — facts by region × decade"); fig.colorbar(im)
    fig.tight_layout(); fig.savefig(f"{OUT}/q1_coverage.png", dpi=110); plt.close(fig)
    return df


def best_single_table(df):
    """The KEY correction: facts are NOT additive across tables (overlapping age/
    civil-condition/religion breakdowns of the same people). Pick ONE table — the one
    with the most distinct mutually-exclusive age bands — and exclude total rows."""
    df = df[df["sex"].isin(["Male", "Female"]) & df["resolved_age_lower"].notna()].copy()
    lab = df["resolved_age_label_std"].astype(str).str.lower()
    df = df[~lab.str.contains("total|all ages|all-ages|grand", na=False)]  # drop totals
    if df.empty:
        return None, df
    # one section per table to avoid civil-condition double counts: keep the modal section
    tid = df.groupby("table_id")["resolved_age_lower"].nunique().idxmax()
    sub = df[df["table_id"] == tid]
    top_sec = sub["section"].value_counts().idxmax() if sub["section"].notna().any() else None
    if top_sec is not None:
        sub = sub[sub["section"] == top_sec]
    return tid, sub


# Q2 — age pyramid from ONE canonical table (no cross-table summing) ----------
def q2_pyramid(df_all):
    df = numeric(pull(topic="age_sex_population", measure="count"))
    tid, sub = best_single_table(df)
    reg = sub["canonical_name"].iloc[0]; yr = sub["year"].iloc[0]
    g = sub.groupby(["resolved_age_lower", "sex"])["value"].sum().unstack(fill_value=0)
    g = g.sort_index()
    fig, ax = plt.subplots(figsize=(8, 7))
    if "Male" in g: ax.barh(g.index, -g["Male"], height=4, color="#3b7", label="Male")
    if "Female" in g: ax.barh(g.index, g.get("Female", 0), height=4, color="#c69", label="Female")
    ax.set_title(f"Age pyramid — {reg} {yr}"); ax.set_xlabel("← males   |   females →")
    ax.set_ylabel("age (lower bound)"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/q2_pyramid.png", dpi=110); plt.close(fig)
    # sanity: pyramid should taper with age
    if "Male" in g and g["Male"].iloc[0] < g["Male"].iloc[-1]:
        flag(f"Q2 {reg} {yr}: youngest age band smaller than oldest — inverted pyramid, suspicious")
    return reg, yr


# Q3 — sex ratio: computed WITHIN each table over mutually-exclusive bands -----
def q3_sexratio():
    df = numeric(pull(topic="age_sex_population", measure="count"))
    df = df[df["sex"].isin(["Male", "Female"]) & df["resolved_age_lower"].notna()]
    lab = df["resolved_age_label_std"].astype(str).str.lower()
    df = df[~lab.str.contains("total|all ages|grand", na=False)]
    # one ratio PER TABLE (M vs F summed over that table's own bands), then median per region-year
    per = df.groupby(["table_id", "broad_region", "year", "sex"])["value"].sum().unstack()
    per = per.dropna(subset=["Male", "Female"]); per = per[per["Male"] > 0]
    per["ratio"] = 1000 * per["Female"] / per["Male"]
    per = per[(per["ratio"] > 400) & (per["ratio"] < 1500)]  # drop single-sex/partial tables
    g = per.reset_index().groupby(["broad_region", "year"])["ratio"].median().reset_index()
    bad = g[(g["ratio"] < 800) | (g["ratio"] > 1150)]
    for _, row in bad.iterrows():
        flag(f"Q3 median sex ratio unusual: {row['broad_region']} {int(row['year'])} = {row['ratio']:.0f} F/1000M")
    fig, ax = plt.subplots(figsize=(11, 6))
    for reg, sub in g.groupby("broad_region"):
        sub = sub.sort_values("year")
        ax.plot(sub["year"], sub["ratio"], marker="o", label=reg)
    ax.axhline(1000, color="#888", ls="--", lw=.8)
    ax.set_ylim(800, 1150)
    ax.set_title("Sex ratio (females per 1,000 males) — median of per-table ratios, by region")
    ax.set_ylabel("F per 1000 M"); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(f"{OUT}/q3_sexratio.png", dpi=110); plt.close(fig)
    return g


# Q4 — age structure: % under 15 / 15-59 / 60+ (per-table, then median) ------
def q4_agestructure():
    df = numeric(pull(topic="age_sex_population", measure="count"))
    df = df[df["resolved_age_lower"].notna() & df["sex"].isin(["Male", "Female", "All"])]
    lab = df["resolved_age_label_std"].astype(str).str.lower()
    df = df[~lab.str.contains("total|all ages|grand", na=False)]

    def band(a):
        return "0-14" if a < 15 else ("15-59" if a < 60 else "60+")
    df["band"] = df["resolved_age_lower"].apply(band)
    # share within each table, then median share per region (avoids cross-table double count)
    pt = df.groupby(["table_id", "broad_region", "band"])["value"].sum().unstack(fill_value=0)
    pt = pt[pt.sum(axis=1) > 0]
    pt = pt.div(pt.sum(axis=1), axis=0) * 100
    g = pt.reset_index().groupby("broad_region")[["0-14", "15-59", "60+"]].median()
    g = g[g.sum(axis=1) > 0]
    ax = g[["0-14", "15-59", "60+"]].plot(kind="bar", stacked=True, figsize=(11, 6),
                                          color=["#e67", "#6a8", "#469"])
    ax.set_title("Age structure (% of population) by region"); ax.set_ylabel("%")
    plt.tight_layout(); plt.savefig(f"{OUT}/q4_agestructure.png", dpi=110); plt.close()
    weird = g[(g["0-14"] > 60) | (g["60+"] > 25)]
    for reg, row in weird.iterrows():
        flag(f"Q4 age structure odd: {reg} 0-14={row['0-14']:.0f}% 60+={row['60+']:.0f}%")
    return g


# Q5 — global value sanity across the whole queryable set --------------------
def q5_sanity():
    df = numeric(pull())  # everything
    print(f"Q5 sanity: {len(df):,} facts total")
    n_neg = (df["value"] < 0).sum()
    counts = df[df["measure_unit"] == "count"]
    huge = counts[counts["value"] > 5e8]
    if len(huge):
        flag(f"Q5 {len(huge)} count cells exceed 500M (India 1901 ~ 238M) — possible decimal/scale error")
    pcts = df[df["measure_unit"] == "percentage"]
    badpct = pcts[(pcts["value"] > 100) | (pcts["value"] < -100)]
    if len(badpct):
        flag(f"Q5 {len(badpct)} percentage cells outside [-100,100]")
    print(f"  negatives={n_neg} (variation tables OK)  count>500M={len(huge)}  bad%={len(badpct)}")
    # biggest counts (eyeball)
    top = counts.nlargest(8, "value")[["year", "canonical_name", "value", "title"]]
    print("  largest count values:")
    for _, r in top.iterrows():
        print(f"     {r['year']} {str(r['canonical_name'])[:20]:20} {r['value']:>14,.0f}  {str(r['title'])[:40]}")
    return df


def main():
    df = q1_coverage()
    q2_pyramid(df)
    q3_sexratio()
    q4_agestructure()
    q5_sanity()
    open(f"{OUT}/anomalies.txt", "w").write("\n".join(ANOM) or "no anomalies flagged")
    print(f"\nDONE — {len(ANOM)} anomalies flagged. charts + anomalies.txt -> {OUT}")


if __name__ == "__main__":
    main()
