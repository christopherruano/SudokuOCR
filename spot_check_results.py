"""Spot-check OCR results: random sample + quality summary."""
import json, os, random

root = os.path.expanduser("~/census_toc/ocr_results")
files = []
for p, ds, fs in os.walk(root):
    for f in fs:
        if f.endswith("_oneshot.json"):
            files.append(os.path.join(p, f))

print(f"Total results: {len(files)}")
print()

# Random sample of 5
random.seed(42)
samples = random.sample(files, min(5, len(files)))
for f in samples:
    d = json.load(open(f))
    c = d["constraints"]
    a = d["acceptance"]
    meta = d.get("metadata", {})
    print(f"File: {os.path.basename(f)}")
    print(f"  Year: {meta.get('year')}, Region: {meta.get('region')}")
    print(f"  Checks: {c['total_checks']}, Failed: {c['failed']}")
    print(f"  Status: {a['decision']}, Trust: {a.get('trust_score')}")
    print(f"  API calls: {a.get('signals',{}).get('api_calls')}")
    sections = d.get("data", [])
    if sections:
        rows = sections[0].get("rows", [])
        print(f"  Sections: {len(sections)}, Rows in first: {len(rows)}")
        if rows:
            print(f"  Sample row: {json.dumps(rows[0])[:150]}")
    print()

# Summary by status
stats = {}
zero_checks = 0
corrupt = 0
for f in files:
    try:
        d = json.load(open(f))
        status = d["acceptance"]["decision"]
        stats[status] = stats.get(status, 0) + 1
        if d["constraints"]["total_checks"] == 0:
            zero_checks += 1
    except (json.JSONDecodeError, KeyError):
        corrupt += 1

print("=== Quality Summary ===")
for k, v in sorted(stats.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v} ({100*v/len(files):.1f}%)")
print(f"  Zero-check results: {zero_checks}")
print(f"  Corrupt/unreadable: {corrupt}")
