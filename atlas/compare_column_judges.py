#!/usr/bin/env python3
"""Agreement between column judges on the calibration sample (Claude reviewers vs Gemini)."""
import json, re, sys, os, collections
HERE = os.path.dirname(os.path.abspath(__file__))


def norm(d):
    v = d.get("verdict")
    if v == "columns_missing":
        cols = [str(x) for x in d.get("problem_columns") or []]
        if cols and all(re.search(r"(?i)persons|total|both sexes", c) for c in cols):
            return "persons_only"
    return {"persons_only_missing": "persons_only", "columns_missing": "problem", "columns_invented": "problem",
            "columns_shifted": "problem", "columns_mislabelled": "problem"}.get(v, v)


def load(p):
    out = {}
    for l in open(p):
        try:
            d = json.loads(l)
        except Exception:
            continue
        if d.get("verdict") not in (None, "error"):
            out[d["page"]] = d
    return out


def kappa(pairs):
    n = len(pairs); po = sum(a == b for a, b in pairs) / n
    ca, cb = collections.Counter(a for a, _ in pairs), collections.Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in ca) / n / n
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


ref = load(os.path.join(HERE, "out", "column_review_sample.jsonl"))
for p in sys.argv[1:]:
    g = load(p)
    common = [k for k in ref if k in g]
    pairs = [(norm(ref[k]), norm(g[k])) for k in common]
    act = lambda c: "act" if c in ("problem", "wrong_scan") else "fine"
    bp = [(act(a), act(b)) for a, b in pairs]
    tp = sum(a == b == "act" for a, b in bp); fn = sum(a == "act" and b == "fine" for a, b in bp); fp = sum(a == "fine" and b == "act" for a, b in bp)
    print(f"\n{os.path.basename(p)}: {len(common)} tables in common")
    print(f"  4-way agreement {sum(a == b for a, b in pairs) / len(pairs):.0%} (kappa {kappa(pairs):.2f}); "
          f"needs-action agreement {sum(a == b for a, b in bp) / len(bp):.0%} (kappa {kappa(bp):.2f})")
    print(f"  of Claude's {tp + fn} needs-action tables Gemini catches {tp} ({tp / (tp + fn):.0%}); Gemini flags {fp} Claude calls fine")
    cm = collections.Counter(pairs)
    labs = ["ok", "persons_only", "problem", "wrong_scan", "unsure"]
    print("  rows=Claude cols=Gemini " + " ".join(f"{l[:9]:>9}" for l in labs))
    for a in labs:
        print(f"  {a:22}" + " ".join(f"{cm[(a, b)]:>9}" for b in labs))
