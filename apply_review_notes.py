#!/usr/bin/env python3
"""Apply the reviewer's per-table notes (decimal_decision_review_decisions human_notes).

Deterministic edits (reading is right) -> applied to facts_confirmed.cleaned.csv:
  discard            : drop the table entirely (reviewer: "discard / delete")
  drop_subcol        : drop a spurious sub-column (reviewer: "get rid of males/females/persons")
  section_override   : fix a wrong section label
  title_override     : fix a wrong title (catalog)
  measure=percentage : mark as percentage table
Re-OCR marks (reading wrong / columns lost) -> status=needs_reocr with the note,
  removed from the clean send set so they don't ship until re-OCR'd.

Writes facts_confirmed.cleaned.csv, sendout_v1/tables_status.csv, facts_sendout.csv,
and sendout_v1/manual_overrides.json (title/note/status for the catalog build).
"""
import csv, os, re, json, shutil
from collections import defaultdict, Counter

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
SF = os.path.join(ROOT, "export_final/sendout_final")
V1 = os.path.join(SF, "sendout_v1")
FACTS = os.path.join(SF, "data/facts_confirmed.cleaned.csv")

DISCARD = {"e1842e22e4f04611fbadf65175a246cac3017332", "497e39b4d5e4883619d9bc15594999c25b46c6de",
           "0636014fb515a12ffe77420deb70d380bd0af458"}
DROP_SUBCOL = {"10cc0967a777f03a0196ff348688f180a9f49fb9": "males",
               "9b3a870402ee52f5e16c2d4c47e0edf174bfd9a5": "males",
               "dd15806998a4649b22e7b9e95705ff21c8f2d49d": "females",
               "782e4d57d0daa2509be88e2f23be720ca26d7efa": "persons"}
SECTION_OVERRIDE = {"90275614a5eae02ec467c1474f1653d5ea463402":
                    "Statement showing the strength of castes having various percentages of married females"}
TITLE_OVERRIDE = {"34ac6a69744c89d6944491271317da16b699ae59": "Age distribution of 1,000 persons in selected castes",
                  "2dc907f387f1c37cb1835c5d3e0feca23a08c75e": "Statement No. 28-continued. Proportion of the sexes by age periods"}
MEASURE_PERCENT = {"2b39d93b54525f6703d58b598c8677881988cee7", "ac59666a3fefd7e6d1628cec6ed5d028934c9319"}
NOTE_ONLY = {"470a4eb509ad38de6cdb3f54902dfb065231b9d0": "proportion of widows in the population per 1000",
             "21a07d9edc8b8e414b8c6307508105dec4cb16de": "multi-page table"}
REOCR = {
    "f6c98d05b3ee8626407c9356cf15a8adbe78187c": "MF dichotomy wrongly imposed; reread",
    "e50e19a74246860aa38bd5c159669eb0db6137a6": "not picking up marriage status of M/F; rerun",
    "530faed432f60229a9a4273e0ba953fe2d32dc67": "widows by age at widowhood; no M/F split; reread",
    "bb6731636ec1616feda2025059afed57c0d71559": "split by all-India and Mysore; reread",
    "19492305a38692663c8c0a5956d2c19df6e665c3": "misread",
    "42495c65ab22b50f6f7c06095f1164e9abd9d25c": "MF dominance over correct structure; misread",
    "75586ab24edff00c83f2bc4d6fe3a4fe847d740b": "reading right but UMW imposed as M/F",
    "de910aa1daad8450fb4a1e24f12a342938942d3f": "% males who are widowers; fix M/F reading",
    "2d0db7b39ef5833af9867f280899ccb37d437e20": "M/F issue; reading right; relabel to UMW",
    "428b6553b0748f751134d4a2895c76644f0cf632": "missing first column; M/F autoimposed",
    "f4c2bfc2ef6ff85eddcb33b044d46dbd153f27e6": "missing first column",
    "98ce83d9feb0318ef3a9a4363897011064b0aed2": "females with three age-segment columns",
    "5862c388f6dd57ea8e585b4c233f45adb6ade4cb": "UMW split M/F up top; imposed",
    "e9378c9de7b52c560df1886293b93d199e3ff158": "percentage unmarried/married; reread",
    "bc2264f94764d17359f5f8f4428cb3cf971edb14": "MF designation; need UMW",
    "b344b0c018e761086353edd86a598a573cec5815": "UMW with M/F imposition",
    "6e3a91de43b2517acac958c6d5f24b905f314cb5": "literacy by age/sex; reread",
    "a539d9e42a65e79564e67185ad1f9aba89328e40": "% women over 15 who are widows; M/F distinction problem",
    "7909bf45ca0256f8e137e4e0929a9bca33c39620": "% increase/decrease in population; reread",
    "4cabf8c0046c4b42955fae381e3e8e0c6d4a6937": "UMW; remove M/F formatting",
    "e6728619347c1f84f1870a6ef5796e466cd0c9f8": "actual vs natural population; constraint relook (not M/F)",
    "6c4279740af36c3ab61e5c19744ae109f1f5d9b9": "% variation; drop or reread",
    "3165d642816f7bcb0522496f68e9b84e15709d59": "also want total population read",
    "7f8c8db225210e6728879d7be61bce05f3b825a7": "read town names not the town label",
}
HEX = re.compile(r"[0-9a-f]{40}")


def h(tid):
    m = HEX.search(tid)
    return m.group(0) if m else ""


def subof(r):
    return r["sub_column_std"] or {"All": "persons", "Male": "males", "Female": "females"}.get(r["sex"], "")


def main():
    shutil.copy(FACTS, FACTS + ".prenotes.bak")
    rows = list(csv.DictReader(open(FACTS)))
    cols = list(rows[0].keys())

    hash2tid = {}
    out = []
    n_disc = n_drop = n_sec = n_meas = 0
    for r in rows:
        hh = h(r["table_id"])
        hash2tid[hh] = r["table_id"]
        if hh in DISCARD:
            n_disc += 1; continue
        if hh in DROP_SUBCOL and subof(r) == DROP_SUBCOL[hh]:
            n_drop += 1; continue
        if hh in SECTION_OVERRIDE:
            r["section"] = SECTION_OVERRIDE[hh]; n_sec += 1
        if hh in MEASURE_PERCENT:
            r["measure_unit"] = "percentage"; n_meas += 1
        out.append(r)
    with open(FACTS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in out]
    print(f"facts edits: discarded {n_disc} cells, dropped {n_drop} subcol cells, "
          f"{n_sec} section fixes, {n_meas} -> percentage")

    # status: re-OCR + discard tids leave the clean send set
    reocr_tids = {hash2tid[hx] for hx in REOCR if hx in hash2tid}
    discard_tids = {hash2tid[hx] for hx in DISCARD if hx in hash2tid}
    st_rows = list(csv.DictReader(open(os.path.join(V1, "tables_status.csv"))))
    bumped = 0
    for r in st_rows:
        if r["table_id"] in reocr_tids:
            r["status"] = "HOLD"; r["reasons"] = (r["reasons"] + "|review_reocr").strip("|"); bumped += 1
        elif r["table_id"] in discard_tids:
            r["status"] = "HOLD"; r["reasons"] = (r["reasons"] + "|review_discard").strip("|")
    with open(os.path.join(V1, "tables_status.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=st_rows[0].keys()); w.writeheader(); [w.writerow(r) for r in st_rows]
    send_tids = {r["table_id"] for r in st_rows if r["status"] == "SEND_clean"}
    send = [r for r in out if r["table_id"] in send_tids]
    with open(os.path.join(V1, "facts_sendout.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in send]
    print(f"re-OCR marked: {bumped} tables  | discards: {len(discard_tids)}  | facts in SEND: {len(send):,}")

    # overrides for the catalog build (title/note/status)
    ov = {"title": {}, "note": {}, "status": {}}
    for hx, t in TITLE_OVERRIDE.items():
        if hx in hash2tid: ov["title"][hash2tid[hx]] = t
    for hx, note in {**NOTE_ONLY, **REOCR}.items():
        if hx in hash2tid: ov["note"][hash2tid[hx]] = note
    for t in reocr_tids: ov["status"][t] = "needs_reocr"
    for t in discard_tids: ov["status"][t] = "discarded_by_review"
    json.dump(ov, open(os.path.join(V1, "manual_overrides.json"), "w"), indent=1)
    print(f"manual_overrides.json: {len(ov['title'])} titles, {len(ov['note'])} notes, {len(ov['status'])} status")


if __name__ == "__main__":
    main()
