#!/usr/bin/env python3
"""Coverage navigation (v0): region -> sub-region -> (year x topic) availability, from the explorer DB.
Delivers the 'click a region, see sub-regions, see which years/tables we have' view the user wants,
using existing metadata (broad_region / canonical_name / year / table_family). Chapters/combined-tables
get layered on later. Writes census_site/coverage.html + a homepage link.
  python3 build_coverage_nav.py
"""
import os, sqlite3, html, collections, hashlib, re, datetime
STAMP=datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

ROOT = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(ROOT, "export_final/sendout_final/sendout_v1/census_explorer.sqlite")
SITE = os.path.join(ROOT, "census_site")
SHOWN = ("confirmed_sent", "recovered", "keeper_accepted", "keeper_review", "assembled")

# readable topic labels for the family codes (the 'Imperial Table' topics ~ chapters)
TOPIC = {
    "age_civil_condition": "Age, Sex & Civil Condition", "age_sex_civil_condition": "Age, Sex & Civil Condition",
    "age_sex_population": "Age & Sex of Population", "age_by_religion": "Age by Religion",
    "age_civil_condition_religion": "Civil Condition by Religion & Age", "age_sex_marital_status": "Age, Sex & Marital Status",
    "age_distribution": "Age Distribution", "education_age": "Education by Age", "caste_class_age": "Age by Caste/Class",
    "proportional_age": "Proportional Age", "birth_death": "Birth & Death",
}
PRIORITY = ["age_civil_condition", "age_sex_civil_condition", "age_by_religion", "age_civil_condition_religion",
            "age_sex_marital_status", "age_sex_population"]  # Table VII family first


def topic_label(f):
    return TOPIC.get(f, (f or "other").replace("_", " ").title())


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    q = ("SELECT broad_region, parent_geo, canonical_name, entity_type, year, table_family, COUNT(*) n "
         "FROM tables WHERE status IN (%s) GROUP BY broad_region, canonical_name, year, table_family"
         % ",".join("?" * len(SHOWN)))
    # region -> subregion -> year -> {family: count}
    tree = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))
    fam_seen = collections.Counter()
    for r in con.execute(q, SHOWN):
        reg = r["broad_region"] or "(unspecified)"
        sub = r["canonical_name"] or "(unknown)"
        y = str(r["year"] or "")
        fam = r["table_family"] or "other"
        if not y:
            continue
        tree[reg][sub][y][fam] = (tree[reg][sub][y].get(fam, 0) + r["n"])
        fam_seen[fam] += r["n"]
    # order families: priority first, then by frequency
    fams = [f for f in PRIORITY if f in fam_seen] + [f for f, _ in fam_seen.most_common() if f not in PRIORITY]

    css = """
    body{font:14px/1.5 system-ui,sans-serif;margin:0;background:#fbfdff;color:#1a2233}
    header{background:#1f3864;color:#fff;padding:12px 20px} header a{color:#cfe0ff}
    h1{margin:0;font-size:18px}.sub{opacity:.9;font-size:12px;margin-top:3px}
    details{margin:0 14px}details.reg>summary{font-size:16px;font-weight:700;color:#1f3864;padding:8px 6px;cursor:pointer;border-bottom:2px solid #e3e9f5}
    details.sub{margin-left:18px}details.sub>summary{font-weight:600;padding:5px 6px;cursor:pointer;color:#24406b}
    table.cov{border-collapse:collapse;margin:6px 0 12px 26px;font-size:12px}
    table.cov th,table.cov td{border:1px solid #e2e6ee;padding:3px 7px;text-align:center}
    table.cov th{background:#eef3fb;color:#1f3864;font-weight:600}
    table.cov th.yr{background:#1f3864;color:#fff}
    td.has{background:#c6efce;color:#0a5d1e;font-weight:700}td.no{color:#cdd5e0}
    .legend{margin:8px 20px;color:#566;font-size:12px}
    """
    out = [f"<!doctype html><meta charset=utf-8><title>Coverage — region × year × topic</title><style>{css}</style>",
           "<header><h1>Coverage — what we have, by region, year &amp; topic</h1>"
           "<div class=sub>click a region &rarr; sub-regions &rarr; the year &times; table grid &middot; "
           "Age/Sex/Civil-Condition topics first &middot; <a href='index.html'>&larr; map</a> &middot; "
           "<a href='highlights.html'>highlights</a> &middot; <b>Last updated " + STAMP + "</b></div></header>",
           f"<div class=legend>Topics shown (columns): {', '.join(topic_label(f) for f in fams[:8])}…</div>"]

    def last(s):
        return (s.startswith("("), s)
    for reg in sorted(tree, key=last):
        nsub = len(tree[reg])
        out.append(f"<details class=reg><summary>{html.escape(reg)} &middot; {nsub} sub-regions</summary>")
        for sub in sorted(tree[reg], key=last):
            years = sorted(tree[reg][sub], reverse=True)
            ntab = sum(sum(tree[reg][sub][y].values()) for y in years)
            out.append(f"<details class=sub><summary>{html.escape(sub)} &middot; {len(years)} years &middot; {ntab} tables</summary>")
            out.append("<table class=cov><tr><th class=yr>year</th>"
                       + "".join(f"<th title='{html.escape(f)}'>{html.escape(topic_label(f))}</th>" for f in fams) + "</tr>")
            for y in years:
                cells = []
                for f in fams:
                    c = tree[reg][sub][y].get(f, 0)
                    cells.append(f"<td class=has>{c}</td>" if c else "<td class=no>·</td>")
                out.append(f"<tr><th class=yr>{html.escape(y)}</th>" + "".join(cells) + "</tr>")
            out.append("</table></details>")
        out.append("</details>")
    open(os.path.join(SITE, "coverage.html"), "w", encoding="utf-8").write("\n".join(out))

    # homepage link (idempotent marker)
    ip = os.path.join(SITE, "index.html"); ih = open(ip, encoding="utf-8").read()
    ih = re.sub(r"<!--COVNAV-->.*?<!--/COVNAV-->", "", ih, flags=re.S)
    link = ('<!--COVNAV--><div style="margin:6px 16px;padding:8px 14px;background:#24406b;color:#fff;border-radius:8px">'
            '<b>Coverage view</b> &middot; browse what exists by region &times; year &times; topic. '
            '<a href="coverage.html" style="color:#ffd966;font-weight:700;text-decoration:none">Open coverage &rarr;</a></div><!--/COVNAV-->')
    anchor = "click a region to drill into the columns</span></header>"
    ih = ih.replace(anchor, anchor + link, 1) if anchor in ih else ih.replace("</header>", "</header>" + link, 1)
    open(ip, "w", encoding="utf-8").write(ih)
    print(f"wrote {SITE}/coverage.html: {len(tree)} regions, "
          f"{sum(len(v) for v in tree.values())} sub-regions, {len(fams)} topic columns")


if __name__ == "__main__":
    main()
