#!/usr/bin/env python3
"""Reclaim disk: the corpus accumulated ~8x duplicate table_dirs (same imghash, different t-index) from
repeated adapter runs. Keep ONE table_dir per imghash — the one whose read has the MOST rows (best
extraction; ties -> shortest/earliest path) — and delete the duplicates.

Safety: only deletes a table_dir if (a) it contains exactly one *_oneshot.json, (b) that imghash has
another kept copy. Run-member references are re-pointed afterward (rebuild members from the deduped
corpus). DRY-RUN by default.
  python3 dedup_corpus.py --dry-run     # report only
  python3 dedup_corpus.py --apply       # actually delete duplicates
"""
import glob, os, re, json, sys, shutil, collections

ROOT = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(ROOT, "export/ocr_results_referenced/digitized_tables")
APPLY = "--apply" in sys.argv


def rows_of(jf):
    try:
        return sum(len(s.get("rows", [])) for s in (json.load(open(jf)).get("data", []) or []))
    except Exception:
        return -1


def main():
    # imghash -> list of (table_dir, json_path, nrows)
    by_ih = collections.defaultdict(list)
    multi = 0
    for d in glob.glob(os.path.join(CORPUS, "*")):
        if not os.path.isdir(d):
            continue
        js = glob.glob(os.path.join(d, "*_oneshot.json"))
        if len(js) != 1:          # safety: skip dirs that aren't the simple one-json shape
            multi += 1
            continue
        m = re.search(r"([0-9a-f]{40})_oneshot", os.path.basename(js[0]))
        if not m:
            continue
        by_ih[m.group(1)].append((d, js[0], rows_of(js[0])))

    keepers = {}
    losers = []
    for ih, lst in by_ih.items():
        # keep most rows; tie -> shortest dir path then lexicographic (stable)
        lst.sort(key=lambda t: (-t[2], len(t[0]), t[0]))
        keepers[ih] = lst[0][0]
        losers.extend(t[0] for t in lst[1:])

    reclaim = 0
    for d in losers:
        try:
            reclaim += sum(os.path.getsize(os.path.join(d, x)) for x in os.listdir(d) if os.path.isfile(os.path.join(d, x)))
        except Exception:
            pass
    print(f"distinct imghashes (keepers): {len(keepers)}")
    print(f"  duplicate dirs to delete: {len(losers)}  | est reclaim: {reclaim/1024/1024/1024:.1f} GB")
    print(f"  skipped non-simple dirs (kept untouched): {multi}")

    if "--list" in sys.argv:
        outf = sys.argv[sys.argv.index("--list") + 1]
        with open(outf, "w") as fh:
            for d in losers:
                fh.write(os.path.relpath(d, CORPUS) + "\n")
        print(f"  wrote {len(losers)} loser dir paths (relative to corpus) -> {outf}")
        return

    if not APPLY:
        print("DRY-RUN — nothing deleted. Re-run with --apply to delete duplicates.")
        # show one example
        ex = next((ih for ih, l in by_ih.items() if len(l) > 1), None)
        if ex:
            print(f"  example imghash {ex[:12]}: {len(by_ih[ex])} copies, keeping rows={max(t[2] for t in by_ih[ex])}")
        return

    deleted = 0
    for d in losers:
        try:
            shutil.rmtree(d); deleted += 1
        except Exception:
            pass
    print(f"DELETED {deleted} duplicate table_dirs.")


if __name__ == "__main__":
    main()
