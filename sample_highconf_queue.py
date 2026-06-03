"""Sample 20 high-confidence rows to verify they're real age tables."""
import json, random, pathlib

root = pathlib.Path('~/census_toc').expanduser()
rows = [json.loads(line) for line in (root / 'pending_age_images.jsonl').open()]
failed = set(line.strip() for line in (root / 'rerun_failed_age_images.txt').open() if line.strip())

keep = {
    'age_sex_population',
    'age_sex_civil_condition',
    'age_sex_marital_status',
}

pool = [
    r for r in rows
    if r.get('image_path') not in failed
    and r.get('canonical_topic') in keep
    and r.get('dir_name_match')
]

print('high_conf sample pool:', len(pool))
random.seed(42)
for r in random.sample(pool, min(20, len(pool))):
    print(r.get('year'), '|', r.get('province'), '|', r.get('canonical_topic'), '|', r.get('table_dir'))
