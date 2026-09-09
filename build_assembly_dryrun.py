#!/usr/bin/env python3
"""ASSEMBLY DRY-RUN on what we already have. Now that crops have TRUE PDF pages,
re-order every assembled multi-page Table VII by PDF page and compare to the
book-page order it used before. Surfaces, before any gap pages arrive:
  - tables whose page ORDER changes under true PDF pages (old ordering was wrong)
  - interior gaps in the sequence (missing pages inside the run)
  - schema/template coverage (can we enforce the full grid at assembly time)
Outputs assembly_dryrun.csv + a summary.
"""
import os, re, csv, json
from collections import defaultdict

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
HEX = re.compile(r"([0-9a-f]{40})")


def main():
    members = json.load(open(os.path.join(V1, "assembled_members.json")))
    crop_pdf = {r["imghash"]: int(r["pdf_page"]) for r in csv.DictReader(open(os.path.join(V1, "crop_pdf_pages.csv")))
                if str(r["pdf_page"]).lstrip("-").isdigit()}
    book, val, vii = {}, {}, {}
    for r in csv.DictReader(open(os.path.join(V1, "master_tracking.csv"))):
        ih = r["imghash"]
        if not ih:
            continue
        if str(r["printed_page"]).lstrip("-").isdigit():
            book[ih] = int(r["printed_page"])
        val[ih] = r["validation"]; vii[ih] = r["is_tableVII"]
    tmpl = set(json.load(open(os.path.join(V1, "group_schema_templates.json"))).keys()) \
        if os.path.exists(os.path.join(V1, "group_schema_templates.json")) else set()

    rows = []
    for asm, mem in members.items():
        # dedup member imghashes, first-appearance order
        seen, ihs = set(), []
        for m in mem:
            h = HEX.search(m)
            if h and h.group(1) not in seen:
                seen.add(h.group(1)); ihs.append(h.group(1))
        if len(ihs) < 2:
            continue
        with_pdf = [h for h in ihs if h in crop_pdf]
        with_book = [h for h in ihs if h in book]
        # order by book page (what assembly used) vs by true PDF page
        common = [h for h in ihs if h in crop_pdf and h in book]
        ord_book = sorted(common, key=lambda h: book[h])
        ord_pdf = sorted(common, key=lambda h: crop_pdf[h])
        order_changed = ord_book != ord_pdf and len(common) >= 2
        # interior gaps in the true-PDF sequence
        pages = sorted(crop_pdf[h] for h in with_pdf)
        interior_gaps = (pages[-1] - pages[0] + 1 - len(set(pages))) if pages else 0
        n_perfect = sum(1 for h in ihs if val.get(h) == "perfect")
        n_tmpl = sum(1 for h in ihs if h in tmpl)
        rows.append({"asm": asm, "n_pages": len(ihs), "n_with_pdf": len(with_pdf),
                     "pdf_range": f"{pages[0]}-{pages[-1]}" if pages else "",
                     "order_changed": "YES" if order_changed else "",
                     "interior_gaps": interior_gaps, "n_perfect": n_perfect,
                     "has_template": n_tmpl, "templated_all": "yes" if n_tmpl == len(ihs) and ihs else ""})

    rows.sort(key=lambda r: (r["order_changed"] != "YES", -r["interior_gaps"]))
    cols = ["asm", "n_pages", "n_with_pdf", "pdf_range", "order_changed",
            "interior_gaps", "n_perfect", "has_template", "templated_all"]
    with open(os.path.join(V1, "assembly_dryrun.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); [w.writerow(r) for r in rows]

    n = len(rows)
    reordered = sum(1 for r in rows if r["order_changed"] == "YES")
    withgaps = sum(1 for r in rows if r["interior_gaps"] > 0)
    fulltmpl = sum(1 for r in rows if r["templated_all"])
    print(f"assembled multi-page tables examined: {n}")
    print(f"  ORDER CHANGES under true PDF pages (old book-page order was wrong): {reordered}")
    print(f"  have interior gaps (missing pages inside the run): {withgaps}")
    print(f"  fully covered by a schema template (full-grid enforceable): {fulltmpl}")
    print(f"  gap pages inside existing assembled runs: {sum(r['interior_gaps'] for r in rows)}")
    print(f"\nwrote {V1}/assembly_dryrun.csv (sorted: reordered + most-gappy first)")


if __name__ == "__main__":
    main()
