#!/usr/bin/env python3
"""OCR every page image used by a published table once (atlas/grounding.py, digits only), in
parallel and resumable. Output atlas/out/page_ocr_numbers.jsonl: {"img": ..., "nums": {number: count}}.
column_audit.py then scores tables, columns and rows against these numbers."""
import os, re, sys, json, glob
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
SITE = os.path.join(HERE, "..", "census_site")
OUT = os.path.join(HERE, "out", "page_ocr_numbers.jsonl")


def work(img):
    import grounding as gr
    try:
        return img, dict(gr.page_numbers(os.path.join(SITE, "img", img)))
    except Exception as e:
        return img, {"__error__": str(e)[:100]}


def main():
    imgs = set()
    for f in glob.glob(os.path.join(SITE, "tables", "*.html")):
        imgs.update(re.findall(r'src="\.\./img/([^"]+)"', open(f, errors="ignore").read()))
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["img"] for l in open(OUT)}
    todo = sorted(imgs - done)
    print(f"images: {len(imgs)}; done {len(done)}; to do {len(todo)}", flush=True)
    with open(OUT, "a") as f, ProcessPoolExecutor(max(1, os.cpu_count() - 1)) as ex:
        for k, (img, nums) in enumerate(ex.map(work, todo, chunksize=2)):
            f.write(json.dumps({"img": img, "nums": nums}) + "\n"); f.flush()
            if k % 250 == 0:
                print(f"{k}/{len(todo)}", flush=True)
    print("OCR_DONE", flush=True)


if __name__ == "__main__":
    main()
