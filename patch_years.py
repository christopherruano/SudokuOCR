"""Patch year metadata in organized results from directory structure."""
import json, os, sys

root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/Research/SudokuOCR/organized_results')
patched = 0
skipped = 0

for p, ds, fs in os.walk(root):
    for f in fs:
        if not f.endswith('_oneshot.json'):
            continue
        fp = os.path.join(p, f)
        d = json.load(open(fp))
        meta = d.get('metadata', {})
        existing_year = meta.get('year')

        if existing_year and existing_year != 0:
            continue

        rel = os.path.relpath(p, root)
        parts = rel.split('/')
        if len(parts) >= 2:
            try:
                dir_year = int(parts[1])
            except ValueError:
                skipped += 1
                continue
        else:
            skipped += 1
            continue

        meta['year'] = dir_year
        d['metadata'] = meta
        with open(fp, 'w') as out:
            json.dump(d, out, indent=2, ensure_ascii=False)
        patched += 1

print(f'Patched: {patched}')
print(f'Skipped (no year in path): {skipped}')
