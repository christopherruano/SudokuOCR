"""Build high-confidence pending queue: primary topics + dir_name_match only."""
import json, pathlib

root = pathlib.Path('~/census_toc').expanduser()
rows = [json.loads(line) for line in (root / 'pending_age_images.jsonl').open()]
failed = set(line.strip() for line in (root / 'rerun_failed_age_images.txt').open() if line.strip())

keep = {
    'age_sex_population',
    'age_sex_civil_condition',
    'age_sex_marital_status',
}

high = [
    r['image_path']
    for r in rows
    if r.get('image_path') not in failed
    and r.get('canonical_topic') in keep
    and r.get('dir_name_match')
]

(root / 'pending_age_images_highconf.txt').write_text(
    ''.join(x + '\n' for x in high),
    encoding='utf-8'
)

print('high_conf:', len(high))
