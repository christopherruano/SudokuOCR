#!/usr/bin/env python3
"""Review queue for the core-money constraint-fails: for each failing page, show the
scan, the pinpointed failing check, and WHAT THE DETERMINISTIC FIX WOULD HAVE BEEN
(trust-the-parts: set the aggregate = sum of its parts), so a human can compare their
own read against it. Easiest-first; the high-failure tail is flagged RE-OCR (don't
hand-fix). Output: core_money_review_queue.html
"""
import json, csv, re, os, html
V1 = "export_final/sendout_final/sendout_v1"
IMG = "/Users/chrisruano/Downloads/final_tableVII"
FIXRE = re.compile(r"(-?\d+)\s*!=\s*sum\(.*\)\s*=\s*(-?\d+)")


def det_fix(detail):
    """(current_aggregate, parts_sum) -> the deterministic fix sets aggregate = parts_sum."""
    m = FIXRE.search(detail or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def main():
    rows = [r for r in csv.DictReader(open(f"{V1}/core_money_ocr_results.csv"))
            if r["category"] == "review_constraints"]
    items = []
    for r in rows:
        d = json.load(open(r["json"]))
        fails = d.get("constraints", {}).get("failures", []) or []
        items.append({"saoa": r["saoa"], "page": r["pdf_page"], "n": len(fails),
                      "title": d.get("metadata", {}).get("title", "")[:80],
                      "fails": fails, "img": f"{IMG}/{r['saoa']}_p{r['pdf_page']}.png"})
    items.sort(key=lambda x: (x["n"], x["saoa"]))

    css = """
    body{font:13px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f4f5f7;color:#1a1a1a}
    header{position:sticky;top:0;background:#1f3864;color:#fff;padding:10px 16px;z-index:5}
    .card{display:flex;gap:16px;background:#fff;margin:10px 16px;border:1px solid #e2e4e8;border-radius:8px;padding:12px;align-items:flex-start}
    .card.reocr{border-color:#d99;background:#fff6f6}
    .scan{flex:0 0 460px}.scan img{width:460px;border:1px solid #ccc;border-radius:4px}
    .info{flex:1}
    h3{margin:0 0 2px;font-size:14px}.ttl{color:#555;font-size:12px;margin-bottom:8px}
    .chk{margin:8px 0;padding:8px 10px;border-radius:6px;background:#fbfbfd;border:1px solid #eee}
    .where{font-weight:700}.detail{font-variant-numeric:tabular-nums;color:#333}
    .fix{margin-top:4px;padding:5px 8px;border-radius:5px;background:#e8f5e9;border:1px solid #b6dcb9;display:inline-block}
    .fix b{color:#0a5d1e}.alt{color:#7a5b00;font-size:12px;margin-top:3px}
    .tag{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;font-weight:700}
    .tag.easy{background:#c6efce;color:#0a5d1e}.tag.mid{background:#ffeb9c;color:#7a5b00}.tag.bad{background:#ffc7ce;color:#9c0017}
    """
    out = [f"<!doctype html><meta charset=utf-8><title>Core money — review queue</title><style>{css}</style>"]
    n_easy = sum(1 for x in items if x["n"] <= 4)
    out.append(f"<header><b>Core-money constraint-fail review</b> — {len(items)} pages, easiest-first. "
               f"{n_easy} are ≤4 failures (one column each). Green = deterministic fix (trust the row-sum); "
               f"amber = the other possibility (a data cell is the mis-read) — that's what your eye decides.</header>")
    for x in items:
        tag = "easy" if x["n"] <= 4 else ("mid" if x["n"] <= 30 else "bad")
        reocr = x["n"] > 30
        out.append(f'<div class="card {"reocr" if reocr else ""}">')
        out.append(f'<div class=scan><a href="file://{html.escape(x["img"])}" target=_blank>'
                   f'<img loading=lazy src="file://{html.escape(x["img"])}"></a></div>')
        out.append('<div class=info>')
        out.append(f'<h3>{x["saoa"]}_p{x["page"]} <span class="tag {tag}">{x["n"]} fail</span>'
                   f'{" — RE-OCR (do not hand-fix)" if reocr else ""}</h3>')
        out.append(f'<div class=ttl>{html.escape(x["title"])}</div>')
        for f in x["fails"][:8]:
            where = f"{f.get('group','')}/{f.get('col','')}".strip("/")
            out.append('<div class=chk>')
            out.append(f'<span class=where>{html.escape(where)}</span> '
                       f'<span class=detail>{html.escape(f.get("detail",""))}</span>')
            fx = det_fix(f.get("detail", ""))
            if fx:
                cur, parts = fx
                out.append(f'<div class=fix>deterministic fix: set total <b>{cur:,} → {parts:,}</b> '
                           f'(Δ {parts-cur:+,}, trust the parts)</div>')
                out.append(f'<div class=alt>…or a data cell in this column is mis-read by {abs(parts-cur):,} '
                           f'(then the total {cur:,} is right) — check the scan.</div>')
            out.append('</div>')
        if x["n"] > 8:
            out.append(f'<div class=ttl>…and {x["n"]-8} more failing checks</div>')
        out.append('</div></div>')
    p = f"{V1}/core_money_review_queue.html"
    open(p, "w").write("\n".join(out))
    print(f"wrote {p}")
    print(f"  {len(items)} pages | {n_easy} easy (<=4 fails) | "
          f"{sum(1 for x in items if x['n']>30)} flagged RE-OCR")
    print(f"  open: file://{os.path.abspath(p)}")


if __name__ == "__main__":
    main()
