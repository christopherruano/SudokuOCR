"""Split fresh pending images into primary (core age tables) and secondary (looser topics)."""
import json, pathlib

root = pathlib.Path('~/census_toc').expanduser()
pending_jsonl = root / 'pending_age_images.jsonl'
failed = set(line.strip() for line in (root / 'rerun_failed_age_images.txt').open() if line.strip())

primary_topics = {
    'age_sex_population',
    'age_sex_civil_condition',
    'age_sex_marital_status',
}
secondary_topics = {
    'age_distribution',
    'age_by_religion',
    'age_civil_condition',
}

primary = []
secondary = []

for line in pending_jsonl.open():
    row = json.loads(line)
    img = row.get('image_path')
    topic = row.get('canonical_topic')
    if not img or img in failed:
        continue
    if topic in primary_topics:
        primary.append(img)
    elif topic in secondary_topics:
        secondary.append(img)

(root / 'pending_age_images_primary.txt').write_text(''.join(x + '\n' for x in primary), encoding='utf-8')
(root / 'pending_age_images_secondary.txt').write_text(''.join(x + '\n' for x in secondary), encoding='utf-8')

print('primary:', len(primary))
print('secondary:', len(secondary))
