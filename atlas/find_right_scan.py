#!/usr/bin/env python3
"""For tables whose linked scan does not contain their data (column check verdict wrong_scan),
look for the right page among the neighbouring pages of the same PDF (offsets -3..+3).

Gemini is shown the candidate pages (labelled by offset) with the table's title, columns and
first values, and names the page that holds the data (or none).
Output out/right_scan.jsonl: page, img, pdf, pdf_page, found_offset (int or null), confidence, note.
"""
import os, re, sys, json, time, random, argparse, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gemini_column_check as g

OFFSETS = [-3, -2, -1, 1, 2, 3]
PROMPT = """The digitized census table below was linked to the wrong scanned page. You are shown candidate pages from the same volume, each preceded by its label (offset from the linked page). Decide which candidate page, if any, is the page this table's data was read from: the title, column headers and the first values listed must appear on it.

Table: "{title}" ({year}, {place}); {n_rows} rows, {n_columns} columns.
Digitized columns with first values:
{columns}

Answer with the offset label of the matching page, or null if none matches."""
SCHEMA = {"type": "object", "properties": {
    "found_offset": {"type": "integer", "nullable": True},
    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    "note": {"type": "string"}}, "required": ["found_offset", "confidence", "note"]}


def render(pdf, pg):
    import fitz
    doc = fitz.open(pdf)
    if not (1 <= pg <= doc.page_count):
        return None
    page = doc.load_page(pg - 1); ims = page.get_images()
    if len(ims) == 1:
        x = doc.extract_image(ims[0][0])
        if x["ext"] in ("jpeg", "jpg", "png"):
            return x["image"], "image/png" if x["ext"] == "png" else "image/jpeg"
    return page.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72), colorspace=fitz.csGRAY).tobytes("png"), "image/png"


def find(client, model, it):
    from google.genai import types
    img = it["imgs"][0].rsplit(".", 1)[0]
    src = g.pdf_index().get(img)
    if not src:
        return dict(page=it["page"], img=img, found_offset=None, confidence="low", note="no PDF mapping")
    pdf, pg = src
    parts = []
    for off in OFFSETS:
        r = render(pdf, pg + off)
        if r:
            parts += [types.Part.from_text(text=f"Candidate page offset {off:+d}:"), types.Part.from_bytes(data=r[0], mime_type=r[1])]
    parts.append(types.Part.from_text(text=PROMPT.format(title=it["title"], year=it["year"], place=it["place"], n_rows=it["n_rows"],
                                                          n_columns=it["n_columns"], columns="\n".join(it["columns"][:40]))))
    cfg = types.GenerateContentConfig(response_mime_type="application/json", response_schema=SCHEMA, temperature=0)
    for attempt in range(6):
        try:
            d = json.loads(client.models.generate_content(model=model, contents=parts, config=cfg).text)
            d.update(page=it["page"], img=img, pdf=pdf, pdf_page=pg)
            return d
        except Exception as e:
            time.sleep(2 ** attempt + random.random())
            err = str(e)[:150]
    return dict(page=it["page"], img=img, found_offset=None, confidence="low", note="error " + err)


def main():
    from google import genai
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="gemini-3.8-flash"); ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    V = {}
    for l in open(os.path.join(HERE, "out", "gemini_column_check_gemini-3.8-flash.jsonl")):
        d = json.loads(l)
        if d.get("verdict") != "error":
            V[d["page"]] = d
    pages = [p for p, d in V.items() if d["verdict"] == "wrong_scan"]
    items = g.table_items(pages)
    out = os.path.join(HERE, "out", "right_scan.jsonl")
    done = {json.loads(l)["page"] for l in open(out)} if os.path.exists(out) else set()
    items = [i for i in items if i["page"] not in done]
    print(f"{len(items)} wrong-scan tables to search", flush=True)
    client = genai.Client(api_key=g.env_key())
    c = collections.Counter()
    with open(out, "a") as f, ThreadPoolExecutor(a.workers) as ex:
        for fu in as_completed([ex.submit(find, client, a.model, i) for i in items]):
            d = fu.result(); f.write(json.dumps(d) + "\n"); f.flush()
            c[d.get("found_offset")] += 1
    print("found offsets:", dict(c), flush=True)


if __name__ == "__main__":
    main()
