#!/usr/bin/env python3
"""Verify-and-fix the existing Table VII assemblies IN PLACE (NOT a regenerate). Conservative:

  PHASE 1  SPLIT only runs whose pages genuinely span >1 table family (TABLE VII -> TABLE VIII), using
           the master's per-page title (the member table_ids carry garbage OCR titles). A run that is a
           single family is left BYTE-IDENTICAL. A lone off-family page flanked by the run family is
           bridged, not split. Each >=2-page segment keeps its original member objects.
  PHASE 2  BACKWARD-BRIDGE continued-start runs to their missing opening pages (same family, stop at
           region dividers); new pages get members rebuilt from the master.
  PHASE 3  ASSEMBLE ORPHANS: contiguous (>=2pp) groups of content-classified Table VII pages (incl.
           non-standard titles) that are in no run become new runs.

Default = DRY RUN. Pass --apply to write (backs up tableVII_runs.json.prefix first).
  python3 fix_assemblies.py [--apply]
"""
import json, csv, os, re, sys, collections, shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(ROOT, "export_final/sendout_final/sendout_v1")
RUNS = os.path.join(ROOT, "export/tableVII_runs.json")
MASTER = os.path.join(V1, "page_number_status.csv")
APPLY = "--apply" in sys.argv

CONT = re.compile(r"contd|continued|\(cont|concld|conclud|-cont\b", re.I)
TAB = re.compile(r"TABLE\s+(?:No\.?\s*)?((?:[A-Z]-)?[IVXLC]+|[A-Z])\b", re.I)
# caste age/civil tables count as VII-family (Neil: Table VII by religion OR caste). Include them even
# when the content classifier (which lists 'caste' as an OTHER topic) didn't flag them — but exclude the
# caste *subsidiary* tables (literacy / proportion / size-of-family by caste).
CASTE = re.compile(r"\bcaste|\btribe|selected\s+caste|castes? and sections", re.I)
AGECIV = re.compile(r"\bages?\b|civil\s*condition|marital|\bsex\b", re.I)
CASTE_SUBSID = re.compile(r"literacy|proportion|size of famil|density|migration|variation|occupation|"
                          r"means of livelihood|education", re.I)


def caste_age_table(t):
    return bool(CASTE.search(t or "") and AGECIV.search(t or "") and not CASTE_SUBSID.search(t or ""))


# Agent-verified: these topics are NOT age/sex/civil-condition POPULATION tables even when the title
# mentions civil condition. Used to keep orphan assembly from pulling in prose/derived/other tables.
ORPHAN_EXCLUDE = re.compile(
    r"unsound mind|lunatic|insane|insanity|\bleper|deaf[- ]?mut|\bblind\b|infirm|"
    r"means of livelihood|\bworkers?\b|\boccupation|diagram", re.I)


def family(title):
    t = re.sub(r"[-.\s]*(cont(?:d|inued)?|concl(?:d|uded)?)\.?", " ", title or "", flags=re.I)
    m = TAB.search(t)
    return m.group(1).upper() if m else ""


def seg_indices(pages, fams):
    """Group consecutive indices into family-consistent segments (bridge <=2-page gaps & lone blips)."""
    segs, segfam = [[0]], [fams[0]]
    for i in range(1, len(pages)):
        cur, f, brk = segfam[-1], fams[i], False
        if pages[i] - pages[i - 1] > 3:
            brk = True
        elif f and cur and f != cur:
            nxt = fams[i + 1] if i + 1 < len(pages) else None
            if not (nxt == cur and pages[i + 1] - pages[i] <= 2):
                brk = True
        if brk:
            segs.append([i]); segfam.append(f)
        else:
            segs[-1].append(i)
            if f and not segfam[-1]:
                segfam[-1] = f
    return segs


def main():
    page = {}
    for r in csv.DictReader(open(MASTER)):
        dh, pg = r.get("dochash", ""), r.get("pdf_page", "")
        if not (dh and str(pg).isdigit()):
            continue
        pg = int(pg); t = r.get("title", "")
        prev = page.get((dh, pg))
        if prev is None or len(t) > len(prev[0]):
            page[(dh, pg)] = (t, r.get("is_tableVII") == "1", r.get("table_dir", ""), r.get("imghash", ""))

    def mtitle(dh, p):
        return page.get((dh, p), ("", False, "", ""))[0]

    def member_for(dh, p):
        rec = page.get((dh, p))
        if not rec or not rec[2]:
            return None
        return {"page": p, "table_id": f"digitized_tables/{rec[2]}/{rec[3]}_oneshot"}

    runs = json.load(open(RUNS))
    tmpl = {r["dochash"]: r for r in runs}
    out, log_split, log_bridge, log_orphan = [], [], [], []

    # ---- PHASE 1: split only genuinely multi-family runs ----
    for run in runs:
        dh = run["dochash"]
        members = sorted(run.get("members", []), key=lambda m: m["page"])
        if not members:
            out.append(run); continue
        pages = [m["page"] for m in members]
        fams = [family(mtitle(dh, p)) for p in pages]
        distinct = set(f for f in fams if f)
        if len(distinct) <= 1:
            out.append(run); continue                       # clean -> untouched
        segs = seg_indices(pages, fams)
        kept = [s for s in segs if len(s) >= 2]
        fcount = collections.Counter(f for f in fams if f)
        log_split.append((run["saoa"], run["page_range"], dict(fcount),
                          [[pages[s[0]], pages[s[-1]]] for s in segs],
                          [[pages[s[0]], pages[s[-1]]] for s in kept]))
        for s in kept:
            seg_members = [members[i] for i in s]
            nr = dict(run)
            nr["members"] = seg_members
            nr["page_range"] = [seg_members[0]["page"], seg_members[-1]["page"]]
            nr["n_pages"] = len(seg_members)
            nr["origin"] = "split_mixed_family"
            out.append(nr)

    # ---- PHASE 2: backward-bridge continued-start runs ----
    for r in out:
        dh = r["dochash"]; s, e = r["page_range"]
        if not CONT.search(mtitle(dh, s)):
            continue
        fam = family(mtitle(dh, s))
        if not fam:
            continue
        ns, blanks, p = s, 0, s - 1
        while p >= 1 and (s - p) <= 40:
            rec = page.get((dh, p))
            if rec is None:
                blanks += 1
                if blanks > 2:
                    break
                p -= 1; continue
            if family(rec[0]) == fam:
                ns = p; blanks = 0; p -= 1; continue
            break
        if ns < s:
            add = [member_for(dh, p) for p in range(ns, s)]
            r["members"] = [m for m in add if m] + sorted(r.get("members", []), key=lambda m: m["page"])
            r["page_range"][0] = ns
            r["n_pages"] = e - ns + 1
            r["origin"] = "backward_bridged"
            log_bridge.append((r["saoa"], [s, e], [ns, e]))

    # ---- PHASE 3: assemble orphan multi-page VII tables ----
    owned = set()
    for r in out:
        for p in range(r["page_range"][0], r["page_range"][1] + 1):
            owned.add((r["dochash"], p))
    orphans = collections.defaultdict(list)
    excl_path = os.path.join(ROOT, "export/orphan_exclude.json")
    EXCL = {k: set(v) for k, v in json.load(open(excl_path)).items()} if os.path.exists(excl_path) else {}
    for (dh, pg), (t, isv, td, ih) in page.items():
        if pg in EXCL.get(dh, ()):                 # agent-verified non-table page
            continue
        if ORPHAN_EXCLUDE.search(t or "") and not re.search(r"civil\s*condition|marital", t or "", re.I):
            continue                               # never-VII topic AND no civil-condition column present
        if (isv or caste_age_table(t)) and (dh, pg) not in owned:
            orphans[dh].append(pg)
    for dh, ps in orphans.items():
        ps = sorted(ps); i = 0
        while i < len(ps):
            j = i
            while j + 1 < len(ps) and ps[j + 1] - ps[j] <= 2:
                j += 1
            grp = ps[i:j + 1]
            fams = [family(mtitle(dh, p)) for p in grp]
            for s in seg_indices(grp, fams):
                if len(s) < 2:
                    continue
                gp = [grp[k] for k in s]
                mem = [member_for(dh, p) for p in gp]
                mem = [m for m in mem if m]
                if len(mem) < 2:
                    continue
                t0 = tmpl.get(dh, {})
                out.append({"dochash": dh, "saoa": t0.get("saoa", dh), "year": t0.get("year", ""),
                            "region": t0.get("region", ""), "tnum": "VII",
                            "page_range": [gp[0], gp[-1]], "n_pages": gp[-1] - gp[0] + 1,
                            "origin": "orphan_assembled", "members": mem})
                log_orphan.append((t0.get("saoa", dh), [gp[0], gp[-1]], len(mem), mtitle(dh, gp[0])[:38]))
            i = j + 1

    # ---- merge same-dochash overlaps (from bridging), dedup members ----
    out.sort(key=lambda r: (r["dochash"], r["page_range"][0]))
    merged = []
    for r in out:
        if merged and r["dochash"] == merged[-1]["dochash"] and r["page_range"][0] <= merged[-1]["page_range"][1]:
            pr = merged[-1]
            pr["page_range"][1] = max(pr["page_range"][1], r["page_range"][1])
            seen = {m["page"] for m in pr["members"]}
            pr["members"] += [m for m in r["members"] if m["page"] not in seen]
            pr["members"].sort(key=lambda m: m["page"])
            pr["n_pages"] = pr["page_range"][1] - pr["page_range"][0] + 1
        else:
            merged.append(r)

    # ---- PHASE 4: merge ADJACENT same-family fragments within a volume (one Table VII per place/year
    # is split only at blank dividers -> the pieces are one table). Merge consecutive same-dominant-family
    # runs with a <=3-page gap where the intervening pages carry NO different explicit table number. This
    # consolidates over-fragmented tables (e.g. 'Ages by Religion' shattered into 3-page chunks) WITHOUT
    # touching genuinely-separate per-region tables (their gaps are larger / a divider sits between).
    def runfam(r):
        fc = collections.Counter(family(mtitle(r["dochash"], p))
                                 for p in range(r["page_range"][0], r["page_range"][1] + 1)
                                 if family(mtitle(r["dochash"], p)))
        return fc.most_common(1)[0][0] if fc else ""
    for r in merged:
        r["_fam"] = runfam(r)
    merged.sort(key=lambda r: (r["dochash"], r["page_range"][0]))
    frag, log_merge = [], []
    for r in merged:
        if frag and r["dochash"] == frag[-1]["dochash"]:
            pr = frag[-1]
            gap = r["page_range"][0] - pr["page_range"][1]
            interv_ok = all(family(mtitle(r["dochash"], p)) in ("", pr["_fam"])
                            for p in range(pr["page_range"][1] + 1, r["page_range"][0]))
            if 1 <= gap <= 3 and pr["_fam"] and pr["_fam"] == r["_fam"] and interv_ok:
                log_merge.append((pr["saoa"], list(pr["page_range"]), list(r["page_range"]), pr["_fam"]))
                pr["page_range"][1] = r["page_range"][1]
                seen = {m["page"] for m in pr["members"]}
                add = [m for m in r["members"] if m["page"] not in seen]
                for p in range(pr["page_range"][0], pr["page_range"][1] + 1):  # pick up crops on bridged blanks
                    if p not in seen and not any(m["page"] == p for m in add):
                        mm = member_for(r["dochash"], p)
                        if mm:
                            add.append(mm)
                pr["members"] = sorted(pr["members"] + add, key=lambda m: m["page"])
                pr["n_pages"] = pr["page_range"][1] - pr["page_range"][0] + 1
                pr["origin"] = (pr.get("origin", "original") + "+merged_frags").replace("original+", "")
                continue
        frag.append(r)
    for r in frag:
        r.pop("_fam", None)
    merged = frag

    print(f"=== FIX ASSEMBLIES (dry-run={not APPLY}) ===")
    print(f"\n[PHASE 1] {len(log_split)} runs SPLIT at real VII/other table boundaries (clean runs untouched):")
    for sa, rng, fams, segs, kept in log_split[:40]:
        print(f"   {sa} {rng} families={fams} -> kept(>=2){kept}")
    print(f"\n[PHASE 2] {len(log_bridge)} continued-start runs extended backward:")
    for sa, old, new in log_bridge:
        print(f"   {sa} {old} -> {new}")
    print(f"\n[PHASE 3] {len(log_orphan)} NEW runs from orphan tables (top by size):")
    for sa, rng, n, t in sorted(log_orphan, key=lambda x: -x[2])[:25]:
        print(f"   {sa} {rng} ({n}pp) {t!r}")
    print(f"\n[PHASE 4] {len(log_merge)} adjacent same-family fragment-merges (consolidating over-split tables):")
    for sa, a, b, f in log_merge[:30]:
        print(f"   {sa} fam={f}: {a} + {b}")
    print(f"\nRUN COUNT {len(runs)} -> {len(merged)} | split {len(log_split)} | bridged {len(log_bridge)} | "
          f"+{len(log_orphan)} orphan | -{len(log_merge)} fragment-merges")
    if APPLY:
        shutil.copy(RUNS, RUNS + ".prefix")
        json.dump(merged, open(RUNS, "w"), indent=1)
        print(f"\nAPPLIED -> {RUNS} (backup .prefix). Next: assembly_audit.py, build_assembled_view.py")
    else:
        print("\n(dry run — re-run with --apply to write)")


if __name__ == "__main__":
    main()
