#!/usr/bin/env python3
"""Bridge human reviews (review_app.py -> review_annotations/*.review.json) back into
the corpus. For each annotation:
  - status 'corrected' with edited reviewed_data -> write that data as the scan's
    oneshot.json (human read wins), originals kept as *_oneshot.prereview.json
  - status 'verified' -> tag the existing read trusted (left as-is; it's already good)
  - status 'flagged'  -> recorded in review_flags.csv for follow-up
Run before reingest.sh so the standard extract picks up the corrected reads.
The U+M+W=Population gate still validates downstream — human edits aren't exempt,
but a careful human correction should pass it."""
import os, re, json, glob, shutil, csv

V1 = "export_final/sendout_final/sendout_v1"
ANN = "review_annotations"
HEX = re.compile(r"([0-9a-f]{40})")


def find_oneshot(ih):
    for root, _, files in os.walk("export/ocr_results_referenced/digitized_tables"):
        if f"{ih}_oneshot.json" in files:
            return os.path.join(root, f"{ih}_oneshot.json")
    return None


def main():
    anns = glob.glob(os.path.join(ANN, "**", "*.review.json"), recursive=True)
    corrected = verified = flagged = no_target = skipped = 0
    flags = []
    for ap in anns:
        try:
            a = json.load(open(ap))
        except Exception:
            continue
        status = (a.get("status") or "").strip()
        ih = HEX.search(os.path.basename(ap))
        ih = ih.group(1) if ih else None
        if not ih:
            skipped += 1; continue
        if status == "flagged":
            flagged += 1
            flags.append({"imghash": ih, "notes": a.get("notes", ""),
                          "issues": json.dumps(a.get("issues", []))})
            continue
        if status == "verified":
            verified += 1; continue
        if status == "corrected" and a.get("reviewed_data") is not None:
            op = find_oneshot(ih)
            if not op:
                no_target += 1; continue
            bak = op.replace("_oneshot.json", "_oneshot.prereview.json")
            if not os.path.exists(bak):
                shutil.copy(op, bak)
            d = json.load(open(op))
            d["data"] = a["reviewed_data"]          # human-edited sections/rows
            d["extraction_method"] = "human_corrected"
            d.setdefault("acceptance", {})["decision"] = "accepted_high_confidence"
            d["acceptance"]["trust_score"] = 100
            json.dump(d, open(op, "w"), indent=1)
            corrected += 1
        else:
            skipped += 1
    if flags:
        with open(os.path.join(V1, "review_flags.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["imghash", "notes", "issues"]); w.writeheader()
            [w.writerow(x) for x in flags]
    print(f"review annotations applied: corrected {corrected} (-> oneshot.json) | verified {verified} | "
          f"flagged {flagged} (-> review_flags.csv) | no target {no_target} | skipped {skipped}")
    if corrected:
        print("Run reingest.sh to fold the human-corrected reads into the corpus.")


if __name__ == "__main__":
    main()
