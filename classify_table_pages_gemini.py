#!/usr/bin/env python3
"""Classify census page images before expensive full-table OCR.

This is intentionally a one-call Gemini-only preflight. It does not extract cell
values; it decides whether a page should enter the full Table VII OCR pipeline.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from pipeline import call_gemini, encode_image


PROMPT = """You are classifying one scanned census page before OCR.

Return ONLY compact JSON with these fields:
{
  "page_class": one of [
    "main_tablevii_numeric",
    "tablevii_context_or_title",
    "derivative_civil_condition",
    "offtarget_table",
    "blank_or_empty",
    "uncertain_review"
  ],
  "contains_numeric_grid": true/false,
  "printed_title": "short exact title/header if visible",
  "why": "short reason"
}

Definitions:
- main_tablevii_numeric: the page contains actual numeric rows from Table VII / Imperial Table VII / C.III / Table VIII civil-condition-by-age, i.e. age/sex/civil-condition or civil-condition-by-age data that belongs to the main assembled civil-condition tables.
- tablevii_context_or_title: title page, divider page, continuation header, notes page, or otherwise Table VII context without numeric table rows.
- derivative_civil_condition: civil-condition subsidiary/summary/percentage/rate/selected-caste table, including per-1,000/per-10,000/percentage/females-per-males/distribution-only derivatives.
- offtarget_table: education, literacy, lepers, area/houses/population-only, infirmities, language/parent-tongue, or any table not part of civil-condition assembly.
- blank_or_empty: blank/near-empty page or watermark-only page.
- uncertain_review: cannot decide from the image.

Be conservative: if the page has a numeric grid but it is not clearly main Table VII/civil-condition-by-age, mark derivative_civil_condition or offtarget_table, not main_tablevii_numeric.
"""


def parse_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines()
                         if not line.strip().startswith("```"))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def iter_manifest(path):
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        yield Path(line).expanduser()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-list", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--total-chunks", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    images = [p.resolve() for p in iter_manifest(args.image_list) if p.exists()]
    if args.chunk is not None or args.total_chunks is not None:
        if args.chunk is None or args.total_chunks is None:
            raise SystemExit("--chunk and --total-chunks must be used together")
        size = (len(images) + args.total_chunks - 1) // args.total_chunks
        start = args.chunk * size
        images = images[start:start + size]

    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out.exists():
        for line in out.read_text(errors="ignore").splitlines():
            try:
                done.add(json.loads(line).get("image_path"))
            except Exception:
                pass

    print(f"Classifying {len(images)} images -> {out}")
    with out.open("a", encoding="utf-8") as fh:
        for i, image in enumerate(images, 1):
            image_s = str(image)
            if image_s in done:
                continue
            t0 = time.time()
            rec = {"image_path": image_s, "status": "unknown"}
            try:
                raw = call_gemini(encode_image(image_s), PROMPT, max_attempts=3)
                parsed = parse_json(raw)
                rec.update({
                    "status": "ok" if isinstance(parsed, dict) else "parse_failed",
                    "parsed": parsed,
                    "raw": raw[:2000] if not isinstance(parsed, dict) else "",
                })
            except Exception as e:
                rec.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
            rec["elapsed_seconds"] = round(time.time() - t0, 1)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            label = (rec.get("parsed") or {}).get("page_class") if rec.get("parsed") else rec["status"]
            print(f"[{i}/{len(images)}] {image.name}: {label} ({rec['elapsed_seconds']}s)")
            if args.sleep:
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()
