"""Quick stats on digitized table outputs."""
import os, json, sys

root = os.path.expanduser("~/census_toc/digitized_tables")
dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
total_png = 0
for d in dirs:
    pngs = [f for f in os.listdir(os.path.join(root, d)) if f.endswith(".png")]
    total_png += len(pngs)

print(f"Table directories: {len(dirs)}")
print(f"Total PNGs: {total_png}")
if dirs:
    print(f"Avg pages/table: {total_png/len(dirs):.1f}")

# Show sample metadata
for d in dirs[:3]:
    mpath = os.path.join(root, d, "metadata.json")
    if os.path.exists(mpath):
        m = json.load(open(mpath))
        print(f"\n--- {d[:40]}... ---")
        for k in ("source_pdf", "year", "province", "table_title", "canonical_topic", "pages_rendered"):
            if k in m:
                print(f"  {k}: {m[k]}")
