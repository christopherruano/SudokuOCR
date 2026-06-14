#!/usr/bin/env python3
"""Fold human decisions from llm_plausibility_review_app.py (failing_decisions.csv)
back into the corpus:
  - accept / image_compared_accept  -> table is human-VERIFIED correct. Written to
    human_verified.csv; validate_tableVII then treats it as PERFECT even if the
    automated U+M+W=Population gate flagged it (the printed source may not sum, or
    it's a faithful read). This is the path to 100% on genuine source-limited cases.
  - rerun                            -> human_rerun_queue.csv (table_id + the note as
    re-OCR guidance) for the next pass.
  - reject / image_compared_reject   -> human_rejected.csv (genuinely wrong, excluded).
Idempotent; merges with any prior decisions."""
import os, csv
import census_explorer as ce

csv.field_size_limit(10 ** 7)
V1 = os.path.dirname(ce.DB)
DEC = "export_final/final_validated/llm_plausibility_review/failing/failing_decisions.csv"


def main():
    if not os.path.exists(DEC):
        print(f"no decisions yet at {DEC} (review some tables first)"); return
    rows = list(csv.DictReader(open(DEC)))
    verified, rerun, rejected = [], [], []
    for r in rows:
        d = (r.get("human_decision") or "").strip()
        tid, note = r.get("table_id", ""), (r.get("human_notes") or "").strip()
        if not tid or not d:
            continue
        if d in ("accept", "image_compared_accept"):
            verified.append({"table_id": tid, "note": note, "decision": d})
        elif d == "rerun":
            rerun.append({"table_id": tid, "note": note})
        elif d in ("reject", "image_compared_reject"):
            rejected.append({"table_id": tid, "note": note})

    def write(path, rows_, cols):
        with open(os.path.join(V1, path), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
            [w.writerow(x) for x in rows_]
    write("human_verified.csv", verified, ["table_id", "note", "decision"])
    write("human_rerun_queue.csv", rerun, ["table_id", "note"])
    write("human_rejected.csv", rejected, ["table_id", "note"])
    print(f"folded {len(rows)} human decisions:")
    print(f"  VERIFIED correct (-> perfect, gate-exempt): {len(verified)}")
    print(f"  rerun (-> human_rerun_queue.csv): {len(rerun)}")
    print(f"  rejected (-> human_rejected.csv): {len(rejected)}")
    print("Run validate_tableVII.py + build_verdicts.py to reflect verifications.")


if __name__ == "__main__":
    main()
