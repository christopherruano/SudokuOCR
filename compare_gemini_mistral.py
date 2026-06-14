#!/usr/bin/env python3
"""Head-to-head on imperfect Table VII scans: GEMINI structured read (our oneshot
pipeline, already produced) vs MISTRAL OCR (mistral-ocr-latest -> markdown).
For each sampled scan, run Mistral OCR and save its markdown next to the Gemini read,
plus a side-by-side manifest so an AI judge can compare both against the actual image.
Writes comparison/<imghash>.{mistral.md,gemini.json,scan.txt} + compare_manifest.json."""
import os, re, csv, json, base64, sqlite3
import census_explorer as ce

csv.field_size_limit(10 ** 7)
V1 = os.path.dirname(ce.DB)
OUT = os.path.join(V1, "comparison")
HEX = re.compile(r"([0-9a-f]{40})")


def gemini_read_for(imghash):
    """Find the fresh Gemini oneshot.json for this imghash and summarize its grid."""
    for root, _, files in os.walk("export/ocr_results_referenced/digitized_tables"):
        for fn in files:
            if fn.startswith(imghash) and fn.endswith("_oneshot.json"):
                try:
                    d = json.load(open(os.path.join(root, fn)))
                except Exception:
                    return None
                md = d.get("metadata", {})
                rows = [r for s in d.get("data", []) for r in s.get("rows", [])]
                return {"title": md.get("title", ""), "column_groups": md.get("column_groups", []),
                        "sections": [s.get("name", "") for s in d.get("data", [])],
                        "n_rows": len(rows), "data": d.get("data", [])}
    return None


def main():
    from mistralai import Mistral
    os.makedirs(OUT, exist_ok=True)
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    cats = list(csv.DictReader(open(os.path.join(V1, "imperfect_categorized.csv"))))
    # sample across categories
    by = {}
    for r in cats:
        by.setdefault(r["category"], []).append(r)
    sample = []
    for cat, n in (("genuine_reocr", 5), ("transposed", 4), ("half_page_split", 3), ("partial_page", 2)):
        sample += by.get(cat, [])[:n]

    con = sqlite3.connect(f"file:{ce.DB}?mode=ro", uri=True); con.row_factory = sqlite3.Row
    manifest = []
    for r in sample:
        row = con.execute("SELECT source_image,member_images FROM tables WHERE table_id=?", (r["table_id"],)).fetchone()
        raw = ((row["member_images"] or "") + ";" + (row["source_image"] or "")) if row else ""
        scan = None
        for p in re.split(r"[;,|]\s*", raw):
            loc = ce.resolve_img(os.path.basename(p.strip())) if p.strip().endswith(".png") else None
            if loc:
                scan = loc; break
        if not scan:
            continue
        ih = HEX.search(os.path.basename(scan)).group(1)
        gem = gemini_read_for(ih)
        # Mistral OCR
        try:
            data_uri = "data:image/png;base64," + base64.b64encode(open(scan, "rb").read()).decode()
            resp = client.ocr.process(model="mistral-ocr-latest",
                                      document={"type": "image_url", "image_url": {"url": data_uri}},
                                      include_image_base64=False, timeout_ms=120000)
            mistral_md = "\n".join(getattr(pg, "markdown", "") or "" for pg in (getattr(resp, "pages", []) or []))
        except Exception as e:
            mistral_md = f"[mistral error: {str(e)[:120]}]"
        open(os.path.join(OUT, f"{ih}.mistral.md"), "w").write(mistral_md)
        json.dump(gem or {}, open(os.path.join(OUT, f"{ih}.gemini.json"), "w"), indent=1)
        manifest.append({"table_id": r["table_id"], "category": r["category"], "issues": r["issues"],
                         "title": (r["title"] or "")[:60], "imghash": ih, "scan": scan,
                         "gemini_cols": (gem or {}).get("column_groups", []),
                         "gemini_rows": (gem or {}).get("n_rows", 0),
                         "mistral_md_chars": len(mistral_md)})
        print(f"  {r['category']:16s} {ih[:10]} | gemini cols={len((gem or {}).get('column_groups',[]))} rows={(gem or {}).get('n_rows',0)} | mistral md={len(mistral_md)} chars")
    json.dump(manifest, open(os.path.join(V1, "compare_manifest.json"), "w"), indent=1)
    print(f"\ncompared {len(manifest)} scans -> {OUT}/  (+ compare_manifest.json)")


if __name__ == "__main__":
    main()
