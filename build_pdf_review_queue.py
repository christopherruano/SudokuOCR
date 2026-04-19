"""Build focused manual-review queues from a census audit manifest.

Usage:
    python3 build_pdf_review_queue.py --manifest /tmp/census_audit_full.json
    python3 build_pdf_review_queue.py --manifest /tmp/census_audit_full.json --mode cautious
"""

import argparse
import json
from pathlib import Path


def load_rows(path):
    with open(path) as fh:
        data = json.load(fh)
    return data["pdfs"]


def choose_rows(rows, mode):
    by_name = {row["filename"]: row for row in rows}
    selected = []

    for row in rows:
        name = row["filename"]
        low = name.lower()
        if name.startswith("._") or name.endswith("_text.pdf"):
            continue

        if mode == "strict":
            if row["classification"] not in {"no_age_signal", "unreadable"}:
                continue
            if row.get("pages", 0) <= 0:
                continue
            if "scheduled-tribes-in-india-as-revealed-in-census-2011" in low:
                continue
            if "census of housing" in low:
                continue
            if "encrypted" in low:
                continue
            if not (low.startswith("saoa.crl.") or ("census" in low and "united states" not in low)):
                continue
            twin = by_name.get(name[:-4] + "_text.pdf")
            if twin and twin["classification"] in {"age_table_evidence", "age_table_candidate"}:
                continue
            selected.append(row)
            continue

        if mode == "cautious":
            if row["classification"] in {"age_table_candidate", "table_volume_only"}:
                selected.append(row)
                continue
            if row["classification"] not in {"no_age_signal", "unreadable"}:
                continue
            if row.get("pages", 0) <= 0:
                continue
            if "scheduled-tribes-in-india-as-revealed-in-census-2011" in low:
                continue
            if "census of housing" in low:
                continue
            if "encrypted" in low:
                continue
            if not (low.startswith("saoa.crl.") or ("census" in low and "united states" not in low)):
                continue
            twin = by_name.get(name[:-4] + "_text.pdf")
            if twin and twin["classification"] in {"age_table_evidence", "age_table_candidate"}:
                continue
            selected.append(row)

    selected.sort(key=lambda r: (r.get("year") or 0, r.get("province") or "", r["filename"]))
    return selected


def main():
    parser = argparse.ArgumentParser(description="Build focused PDF review queues")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mode", choices=["strict", "cautious"], default="strict")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = choose_rows(load_rows(args.manifest), args.mode)
    out = Path(args.output or f"/tmp/census_review_{args.mode}.json")
    with open(out, "w") as fh:
        json.dump({"pdfs": rows}, fh, indent=2)

    print(f"mode={args.mode}")
    print(f"count={len(rows)}")
    print(f"output={out}")
    for row in rows:
        print(f"{row['classification']:<14} {row.get('year') or '?':>4} {row.get('province') or '?':<30} {row['filename']}")


if __name__ == "__main__":
    main()
