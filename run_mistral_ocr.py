#!/usr/bin/env python3
"""Bulk Mistral-OCR the hard imperfect Table VII scans (transposed / genuine_reocr /
half_page) into cached markdown, in parallel. Safe to run alongside the Gemini
verdict loop (separate API, writes only to mistral_md/). The validated INGEST
(parse -> cross-check -> constraint gate -> corrections overlay) happens later.
Resumable: skips scans already cached."""
import os, re, csv, base64, sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
import census_explorer as ce

csv.field_size_limit(10 ** 7)
V1 = os.path.dirname(ce.DB)
OUT = os.path.join(V1, "mistral_md")
HEX = re.compile(r"([0-9a-f]{40})")
HARD = {"transposed", "genuine_reocr", "half_page_split"}


def scan_list():
    cats = list(csv.DictReader(open(os.path.join(V1, "imperfect_categorized.csv"))))
    con = sqlite3.connect(f"file:{ce.DB}?mode=ro", uri=True); con.row_factory = sqlite3.Row
    scans = {}
    for r in cats:
        if r["category"] not in HARD:
            continue
        tids = {r["table_id"]}
        m = re.search(r"sibling::(.+?)::", r["detail"] or "")
        if m:
            tids.add(m.group(1))
        for t in tids:
            row = con.execute("SELECT source_image,member_images FROM tables WHERE table_id=?", (t,)).fetchone()
            if not row:
                continue
            for p in re.split(r"[;,|]\s*", ((row["member_images"] or "") + ";" + (row["source_image"] or ""))):
                if p.strip().endswith(".png"):
                    loc = ce.resolve_img(os.path.basename(p.strip())) or ce.resolve_img(p.strip())
                    if loc and os.path.exists(loc):
                        m = HEX.search(os.path.basename(loc))
                        if m:                       # render-named files (e.g. 25352797_p133.png) have
                            scans[m.group(1)] = loc  # no 40-hex basename — skip, don't crash scan_list
    return scans


def ocr_one(client, ih, path):
    dst = os.path.join(OUT, f"{ih}.mistral.md")
    if os.path.exists(dst) and os.path.getsize(dst) > 20:
        return ih, "cached"
    try:
        data_uri = "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()
        resp = client.ocr.process(model="mistral-ocr-latest",
                                  document={"type": "image_url", "image_url": {"url": data_uri}},
                                  include_image_base64=False, timeout_ms=120000)
        md = "\n".join(getattr(pg, "markdown", "") or "" for pg in (getattr(resp, "pages", []) or []))
        open(dst, "w").write(md)
        return ih, f"ok({len(md)})"
    except Exception as e:
        return ih, f"ERR:{str(e)[:60]}"


def main():
    from mistralai import Mistral
    os.makedirs(OUT, exist_ok=True)
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    scans = scan_list()
    print(f"Mistral-OCR {len(scans)} hard-table scans (concurrency 6) -> {OUT}/")
    done = err = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(ocr_one, client, ih, p) for ih, p in scans.items()]
        for i, f in enumerate(as_completed(futs), 1):
            ih, st = f.result()
            if st.startswith("ERR"):
                err += 1
                print(f"  [{i}/{len(scans)}] {ih[:10]} {st}")
            else:
                done += 1
            if i % 25 == 0:
                print(f"  ...{i}/{len(scans)} (ok={done} err={err})")
    print(f"done: {done} cached/ok, {err} errors -> {OUT}/")


if __name__ == "__main__":
    main()
