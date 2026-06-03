#!/usr/bin/env python3
"""Apply the human decimal-review decisions, then rebuild the send/hold split.

Human decisions (decimal_rollout/decimal_decision_review_decisions.csv):
  already_correct (78) -> values fine as-is, release
  divide          (23) -> value / 10^human_dp (human-verified uniform tables)
  mixed           (27) -> rerun_or_hand_fix, stay held for re-OCR

Effect: 101 tables leave decimal-HOLD for the clean set; 27 stay held.
Writes facts_confirmed.cleaned.csv (divides applied, backup kept), then regenerates
sendout_v1/tables_status.csv + facts_sendout.csv.
"""
import csv, os, glob, shutil, json
from collections import defaultdict

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
SF = os.path.join(ROOT, "export_final/sendout_final")
V1 = os.path.join(SF, "sendout_v1")
DR = os.path.join(SF, "review/decimal_rollout")
FACTS = os.path.join(SF, "data/facts_confirmed.cleaned.csv")


def main():
    # 1) divide map  table_id -> dp
    divmap = {}
    for r in csv.DictReader(open(os.path.join(DR, "decimal_decision_apply_divides.csv"))):
        hd = r.get("human_dp", "").strip()
        dp = {"1": 1, "2": 2, "3+": int(r.get("opus_dp") or 3)}.get(hd, None)
        if dp:
            divmap[r["table_id"]] = dp
    # decision sets
    mixed = {r["table_id"] for r in csv.DictReader(open(os.path.join(DR, "decimal_decision_rerun_or_hand_fix.csv")))}

    # 2) apply divides to facts
    shutil.copy(FACTS, FACTS + ".predecimal.bak")
    rows = list(csv.DictReader(open(FACTS)))
    cols = list(rows[0].keys())
    n_div = 0
    for r in rows:
        dp = divmap.get(r["table_id"])
        if dp and r["value"] not in ("", "None", None):
            try:
                v = float(r["value"]) / (10 ** dp)
                r["value"] = str(int(v)) if v == int(v) else f"{v:.{dp}f}"
                n_div += 1
            except ValueError:
                pass
    with open(FACTS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in rows]
    print(f"divides applied: {n_div:,} cells across {len(divmap)} tables (backup -> .predecimal.bak)")

    # 3) recompute send/hold (decimal now resolved: only `mixed` stays held for decimal reason)
    reason = defaultdict(set)
    for t in mixed:
        reason[t].add("decimal_mixed_reocr")
    res = {}
    for f in glob.glob(os.path.join(SF, "review/mf_relabel/result_mf_b*.json")):
        res.update(json.load(open(f)))
    for t, v in res.items():
        if isinstance(v, dict) and v.get("data_loss_risk"):
            reason[t].add("mf_data_loss")
    try:
        for t in json.load(open(os.path.join(SF, "review/validation_candidates/failure_annotations.json"))):
            reason[t].add("reconciliation_candidate")
    except Exception:
        pass

    alltbl = set(r["table_id"] for r in rows)
    HOLD = {"decimal_mixed_reocr", "mf_data_loss", "reconciliation_candidate"}

    def status(t):
        return "HOLD" if reason.get(t, set()) & HOLD else "SEND_clean"
    st = {t: status(t) for t in alltbl}

    with open(os.path.join(V1, "tables_status.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["table_id", "status", "reasons"])
        for t in sorted(alltbl):
            w.writerow([t, st[t], "|".join(sorted(reason.get(t, [])))])
    send = [r for r in rows if st[r["table_id"]] == "SEND_clean"]
    with open(os.path.join(V1, "facts_sendout.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in send]

    from collections import Counter
    print("send/hold:", dict(Counter(st.values())))
    print("HOLD reasons:", dict(Counter(r for t in alltbl if st[t] == "HOLD" for r in reason[t])))
    print(f"facts in SEND: {len(send):,}  (confirmed tables sent: {sum(1 for v in st.values() if v=='SEND_clean')})")


if __name__ == "__main__":
    main()
