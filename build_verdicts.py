#!/usr/bin/env python3
"""Per-table FINAL VERDICT ledger for Table VII. After each re-OCR+reingest round,
classify every table as:
  - perfect       : passes validation (locked, final)
  - unrecoverable : proven it cannot be completed from what exists, with a REASON
  - retrying      : still fixable, queued for another (escalating) re-OCR pass
and emit the requery queue (local scans of the 'retrying' tables) for the next round.

A table becomes unrecoverable when:
  * partial_page with no stitchable sibling      -> source scan is incomplete (data not on page)
  * half_page whose sibling is already merged     -> the other half/columns aren't a separate readable source
  * re-OCR plateaued: same issues N rounds running, or >= MAX_ATTEMPTS attempts
Writes tableVII_verdicts.csv (persistent) + requery_local.txt (next round) + prints the tally."""
import os, re, csv, sqlite3
from collections import Counter
import census_explorer as ce

csv.field_size_limit(10 ** 7)
V1 = os.path.dirname(ce.DB)
MAX_ATTEMPTS = 3            # base + 2 MoE escalations
STABLE_LIMIT = 2           # same issues this many rounds in a row -> give up


def main():
    score = {r["table_id"]: r for r in csv.DictReader(open(os.path.join(V1, "tableVII_scorecard.csv")))}
    cats = {r["table_id"]: r for r in csv.DictReader(open(os.path.join(V1, "imperfect_categorized.csv")))}
    led_path = os.path.join(V1, "tableVII_verdicts.csv")
    prev = {r["table_id"]: r for r in csv.DictReader(open(led_path))} if os.path.exists(led_path) else {}

    ledger = []
    for tid, s in score.items():
        p = prev.get(tid, {})
        attempts = int(p.get("attempts", "0") or 0)
        prev_issues = p.get("last_issues", "")
        cat = cats.get(tid, {})
        category = cat.get("category", "")
        detail = cat.get("detail", "")
        if s["perfect"]:
            ledger.append({"table_id": tid, "status": "perfect", "attempts": attempts,
                           "category": "", "last_issues": "", "stable_rounds": 0, "reason": "",
                           "year": s["year"], "province": s["province"], "title": s["title"]})
            continue
        # imperfect this round -> it was just re-OCR'd, so count an attempt
        attempts += 1
        issues = s["issues"]
        stable = int(p.get("stable_rounds", "0") or 0)
        stable = stable + 1 if issues == prev_issues and prev_issues else 0

        status, reason = "retrying", ""
        # immediate unrecoverable: incomplete source
        if category == "partial_page":
            status, reason = "unrecoverable", "source scan incomplete — age bands not present on the page; no sibling page to stitch"
        elif category == "half_page_split" and "sibling::" in detail and attempts >= 2:
            status, reason = "unrecoverable", "missing columns live only on an already-merged sibling; no separate readable source for them"
        elif stable >= STABLE_LIMIT:
            status, reason = "unrecoverable", f"re-OCR not improving — identical defects {stable} rounds running ({issues})"
        elif attempts >= MAX_ATTEMPTS:
            status, reason = "unrecoverable", f"plateaued after {attempts} re-OCR attempts (incl. MoE) — likely illegible/structurally non-standard"
        ledger.append({"table_id": tid, "status": status, "attempts": attempts,
                       "category": category, "last_issues": issues, "stable_rounds": stable,
                       "reason": reason, "year": s["year"], "province": s["province"], "title": s["title"]})

    with open(led_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["table_id", "status", "attempts", "category", "last_issues",
                                          "stable_rounds", "reason", "year", "province", "title"])
        w.writeheader(); [w.writerow(r) for r in ledger]

    # build the requery queue: local scans of 'retrying' tables (+ half-page siblings)
    retrying = [r for r in ledger if r["status"] == "retrying"]
    con = sqlite3.connect(f"file:{ce.DB}?mode=ro", uri=True); con.row_factory = sqlite3.Row
    HEX = re.compile(r"([0-9a-f]{40})")
    tids = set()
    for r in retrying:
        tids.add(r["table_id"])
        d = cats.get(r["table_id"], {}).get("detail", "")
        m = re.search(r"sibling::(.+?)::", d)
        if m:
            tids.add(m.group(1))
    local = set()
    for t in tids:
        row = con.execute("SELECT source_image,member_images FROM tables WHERE table_id=?", (t,)).fetchone()
        if not row:
            continue
        for p in re.split(r"[;,|]\s*", ((row["member_images"] or "") + ";" + (row["source_image"] or ""))):
            p = p.strip()
            if p.endswith(".png"):
                loc = ce.resolve_img(os.path.basename(p)) or ce.resolve_img(p)
                if loc and os.path.exists(loc):
                    local.add(loc)
    open(os.path.join(V1, "requery_local.txt"), "w").write("\n".join(sorted(local)) + "\n")

    tally = Counter(r["status"] for r in ledger)
    reasons = Counter(r["reason"][:55] for r in ledger if r["status"] == "unrecoverable")
    print(f"VERDICTS over {len(ledger)} Table VII:")
    print(f"  perfect:       {tally['perfect']}")
    print(f"  unrecoverable: {tally['unrecoverable']}")
    print(f"  retrying:      {tally['retrying']}  -> requery_local.txt ({len(local)} scans)")
    if reasons:
        print("  unrecoverable reasons:")
        for rs, n in reasons.most_common():
            print(f"    {n:3d}  {rs}")
    nxt = max((int(r["attempts"]) for r in retrying), default=0)
    print(f"  next round escalation: {'MoE (3 Gemini + Mistral)' if nxt >= 1 else 'base'}  (set MOE=1)")


if __name__ == "__main__":
    main()
