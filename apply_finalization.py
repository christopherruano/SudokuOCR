#!/usr/bin/env python3
"""Consolidated finalization apply — run AFTER review is done.
Folds together: UMW vision verdicts, the user's /review flags, and the verified
multi-page orderings, into the config the build consumes. Does NOT rebuild
(run build_query_dataset -> build_explorer_db -> build_static_site afterwards).

Inputs (sendout_v1/):
  umw_verdicts.json        relabel(10) / reocr(140) / remove(8) / keep(55)
  review_decisions.json    user flags from /review  (action: ok|flag)
  order_verdicts.json      vision-verified page orderings (90 reordered)
  multipage_groups.json    deterministic groups (order updated in place)
Outputs:
  manual_overrides.json    status overrides merged in
  relabel_overrides.json   {table_id: {current_sub -> correct_label}}
  multipage_groups.json    member order corrected from vision
"""
import json, os
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")


def load(name, default):
    p = os.path.join(V1, name)
    return json.load(open(p)) if os.path.exists(p) else default


def main():
    ov = load("manual_overrides.json", {"title": {}, "note": {}, "status": {}})
    ov.setdefault("status", {}); ov.setdefault("note", {})
    relabel = {}

    # 1) UMW vision verdicts
    umw = load("umw_verdicts.json", [])
    uc = Counter()
    for v in umw:
        t, verd = v["table_id"], v["verdict"]
        uc[verd] += 1
        if verd == "reocr":
            ov["status"][t] = "needs_reocr"
            ov["note"][t] = "vision: column dropped (" + (v.get("note", "")[:80]) + ")"
        elif verd == "remove":
            ov["status"][t] = "excluded_malformed"
            ov["note"][t] = "vision: malformed (" + (v.get("note", "")[:70]) + ")"
        elif verd == "relabel" and v.get("relabel_map"):
            relabel[t] = v["relabel_map"]
            ov["note"][t] = "vision: relabeled M/F->U/M/W"
        # keep -> leave as-is

    # 2) user /review decisions (categorized by reason)
    rev = load("review_decisions.json", {})
    catz = load("review_categorized.json", {})
    reocr_ids = set(catz.get("reocr", []))
    reject_ids = set(catz.get("reject", []))
    rc = Counter()
    for t, d in rev.items():
        a = d.get("action", ""); r = (d.get("reason") or "").strip()[:90]; rc[a] += 1
        if t in reject_ids:
            ov["status"][t] = "excluded_rejected"; ov["note"][t] = "review-reject: " + r
        elif t in reocr_ids:
            ov["status"][t] = "needs_reocr"; ov["note"][t] = "review-reocr: " + r
        elif r:  # multipage-context / metadata / uncat -> stay on site, note for later passes
            ov["note"][t] = "review: " + r

    # 3) verified multi-page order -> rewrite group member order (match by member set)
    mp = load("multipage_groups.json", {"groups": {}, "page_order": {}})
    ords = load("order_verdicts.json", [])
    by_set = {frozenset(m): h for h, m in mp["groups"].items()}
    reordered = 0
    for o in ords:
        ids = [t for t in o.get("ordered_table_ids", [])]
        s = frozenset(ids)
        h = by_set.get(s)
        if h and ids and o.get("changed"):
            mp["groups"][h] = ids   # corrected reading order
            for i, t in enumerate(ids):
                mp["page_order"].setdefault(t, {})["order"] = i
            reordered += 1

    json.dump(ov, open(os.path.join(V1, "manual_overrides.json"), "w"), indent=1)
    json.dump(relabel, open(os.path.join(V1, "relabel_overrides.json"), "w"), indent=1)
    json.dump(mp, open(os.path.join(V1, "multipage_groups.json"), "w"))

    print("UMW verdicts applied:", dict(uc))
    print("review flags:", dict(rc))
    print(f"relabel maps: {len(relabel)}")
    print(f"status overrides now: {dict(Counter(ov['status'].values()))}")
    print(f"multi-page groups reordered from vision: {reordered}")
    print("staged -> manual_overrides.json, relabel_overrides.json, multipage_groups.json")
    print("\nNEXT: python3 build_query_dataset.py && python3 build_explorer_db.py && python3 build_static_site.py")


if __name__ == "__main__":
    main()
