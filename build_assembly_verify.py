#!/usr/bin/env python3
"""ASSEMBLY VERIFICATION GATE (phase 1) — give every assembled multi-page Table VII
a verdict from objective checks, so nothing is called "done" on faith:
  - ORDER     : members ascending by true PDF page (and did the old book-order disagree)
  - CONTIGUOUS: no interior gap pages inside the run
  - CELLS     : every page passed the per-page check (Pop=U+M+W, 3 cols, age bands)
  - TEMPLATE  : covered by a schema template (full grid enforceable)
Verdict:
  VERIFIED   complete + ordered + all pages cell-valid
  INCOMPLETE interior gap pages (cannot verify totals until filled)  <-- needs cluster OCR
  REORDER    page order was wrong under true PDF pages
  PARTIAL    complete+ordered but some pages not yet cell-valid
  REVIEW     anything else
Phase-2 checks (cross-page Total=Sum(religions), completeness) run AFTER the gap fill.
Outputs assembly_verify.csv + summary.
"""
import os, re, csv, json
from collections import Counter

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
HEX = re.compile(r"([0-9a-f]{40})")


def main():
    members = json.load(open(os.path.join(V1, "assembled_members.json")))
    crop_pdf = {r["imghash"]: int(r["pdf_page"]) for r in csv.DictReader(open(os.path.join(V1, "crop_pdf_pages.csv")))
                if str(r["pdf_page"]).lstrip("-").isdigit()}
    book, val = {}, {}
    for r in csv.DictReader(open(os.path.join(V1, "master_tracking.csv"))):
        ih = r["imghash"]
        if not ih:
            continue
        if str(r["printed_page"]).lstrip("-").isdigit():
            book[ih] = int(r["printed_page"])
        val[ih] = r["validation"]
    tmpl = set(json.load(open(os.path.join(V1, "group_schema_templates.json"))).keys()) \
        if os.path.exists(os.path.join(V1, "group_schema_templates.json")) else set()

    rows = []
    for asm, mem in members.items():
        seen, ihs = set(), []
        for m in mem:
            h = HEX.search(m)
            if h and h.group(1) not in seen:
                seen.add(h.group(1)); ihs.append(h.group(1))
        if len(ihs) < 2:
            continue
        n = len(ihs)
        common = [h for h in ihs if h in crop_pdf and h in book]
        order_wrong = (len(common) >= 2 and
                       sorted(common, key=lambda h: book[h]) != sorted(common, key=lambda h: crop_pdf[h]))
        pages = sorted(crop_pdf[h] for h in ihs if h in crop_pdf)
        interior_gaps = (pages[-1] - pages[0] + 1 - len(set(pages))) if pages else 0
        n_perfect = sum(1 for h in ihs if val.get(h) == "perfect")
        n_tmpl = sum(1 for h in ihs if h in tmpl)
        unpaged = sum(1 for h in ihs if h not in crop_pdf)

        if unpaged:
            verdict, reason = "REVIEW", f"{unpaged} pages lack a PDF page"
        elif interior_gaps > 0:
            verdict, reason = "INCOMPLETE", f"{interior_gaps} interior gap pages"
        elif order_wrong:
            verdict, reason = "REORDER", "book-order != PDF-order"
        elif n_perfect == n:
            verdict, reason = "VERIFIED", ""
        elif n_perfect > 0:
            verdict, reason = "PARTIAL", f"{n-n_perfect}/{n} pages not cell-valid"
        else:
            verdict, reason = "REVIEW", "no cell-valid pages"

        rows.append({"asm": asm, "verdict": verdict, "n_pages": n,
                     "pdf_range": f"{pages[0]}-{pages[-1]}" if pages else "",
                     "interior_gaps": interior_gaps, "order_wrong": "YES" if order_wrong else "",
                     "cell_valid": f"{n_perfect}/{n}", "templated": f"{n_tmpl}/{n}", "reason": reason})

    order = {"VERIFIED": 0, "PARTIAL": 1, "REORDER": 2, "INCOMPLETE": 3, "REVIEW": 4}
    rows.sort(key=lambda r: (order.get(r["verdict"], 9), -r["interior_gaps"]))
    cols = ["asm", "verdict", "n_pages", "pdf_range", "interior_gaps", "order_wrong",
            "cell_valid", "templated", "reason"]
    with open(os.path.join(V1, "assembly_verify.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); [w.writerow(r) for r in rows]

    c = Counter(r["verdict"] for r in rows)
    print(f"assembled tables verified: {len(rows)}")
    for v in ("VERIFIED", "PARTIAL", "REORDER", "INCOMPLETE", "REVIEW"):
        print(f"  {v:<11} {c.get(v,0)}")
    print(f"\n  -> VERIFIED today (complete+ordered+all cell-valid): {c.get('VERIFIED',0)}")
    print(f"  -> INCOMPLETE only because of {sum(r['interior_gaps'] for r in rows if r['verdict']=='INCOMPLETE')} "
          f"gap pages — these become VERIFIABLE after the cluster fill")
    print(f"\nwrote {V1}/assembly_verify.csv  (phase-2 cross-page arithmetic runs after the gap fill)")


if __name__ == "__main__":
    main()
