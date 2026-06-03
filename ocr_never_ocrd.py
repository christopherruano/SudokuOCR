#!/usr/bin/env python3
"""
Run the local oneshot OCR pipeline over the never-OCR'd backlog (priority order).

Reads export/never_ocrd_images/_core_worklist.json, runs a slice through
oneshot.extract_and_verify, writes per-image results to
export/never_ocrd_ocr/results/, and a run summary CSV with quality + cost.

Usage:
  python3 ocr_never_ocrd.py --limit 10                 # measure a slice
  python3 ocr_never_ocrd.py --offset 0 --limit 3712    # full core run
"""
import argparse, json, os, sys, time, csv, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "export/never_ocrd_images/_core_worklist.json"
OUT = ROOT / "export/never_ocrd_ocr/results"
SUMMARY = ROOT / "export/never_ocrd_ocr/run_summary.csv"


def n_rows(result):
    try:
        df = result.get("df")
        if df is not None:
            return len(df)
    except Exception:
        pass
    p = (result or {}).get("parsed") or {}
    data = p.get("data") if isinstance(p, dict) else None
    if isinstance(data, list):
        return sum(len(s.get("rows", [])) for s in data if isinstance(s, dict))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()

    import oneshot  # loads .env, sets up clients

    OUT.mkdir(parents=True, exist_ok=True)
    work = json.load(open(WORK))[args.offset:args.offset + args.limit]
    print(f"Running OCR on {len(work)} pages (offset {args.offset}) -> {OUT}", flush=True)

    rows = []
    t0 = time.time()
    for i, w in enumerate(work):
        img = w["img"]
        rec = {"img": img, "bucket": w["bucket"], "year": w["year"],
               "topic": w["topic"], "ev": w["ev"]}
        t = time.time()
        try:
            res = oneshot.extract_and_verify(img, output_dir=str(OUT), fallback=False)
            if not res:
                rec.update(status="no_result", rows=0, passed="", api_calls="", elapsed=round(time.time()-t, 1))
            else:
                rep = res.get("report") or {}
                acc = res.get("acceptance") or {}
                # api_calls written into the per-image json
                api = ""
                for op in (res.get("output_paths") or {}).values() if isinstance(res.get("output_paths"), dict) else []:
                    if str(op).endswith(".json") and os.path.exists(op):
                        try:
                            api = json.load(open(op)).get("api_calls", "")
                        except Exception:
                            pass
                rec.update(
                    status="ok",
                    rows=n_rows(res),
                    passed=bool(rep.get("all_passed")) if "all_passed" in rep else "",
                    accepted=acc.get("status", "") if isinstance(acc, dict) else "",
                    api_calls=api,
                    elapsed=round(res.get("elapsed", time.time()-t), 1),
                )
        except Exception as exc:
            rec.update(status="error", rows=0, passed="", api_calls="",
                       elapsed=round(time.time()-t, 1), err=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
        rows.append(rec)
        print(f"  [{i+1}/{len(work)}] {rec['status']:9} rows={rec.get('rows')} "
              f"passed={rec.get('passed')} api={rec.get('api_calls')} {rec['elapsed']}s "
              f"{os.path.basename(img)[:12]}", flush=True)

    # summary
    cols = ["img", "bucket", "year", "topic", "ev", "status", "rows", "passed",
            "accepted", "api_calls", "elapsed", "err"]
    write_header = not SUMMARY.exists()
    with open(SUMMARY, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)

    ok = [r for r in rows if r["status"] == "ok"]
    passed = [r for r in ok if r.get("passed") is True]
    total_api = sum(int(r["api_calls"]) for r in ok if str(r.get("api_calls")).isdigit())
    wall = time.time() - t0
    print("\n=== SLICE SUMMARY ===")
    print(f"pages: {len(rows)}  ok: {len(ok)}  constraint-passed: {len(passed)}  "
          f"errors: {sum(1 for r in rows if r['status']=='error')}")
    print(f"total rows extracted: {sum(r.get('rows',0) for r in ok)}")
    print(f"total api calls: {total_api}  (~{total_api/max(1,len(ok)):.1f}/page)")
    print(f"wall time: {wall:.0f}s  (~{wall/max(1,len(rows)):.1f}s/page)")
    print(f"summary appended -> {SUMMARY}")


if __name__ == "__main__":
    main()
