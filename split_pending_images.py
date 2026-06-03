"""Split pending OCR images into failed-age reruns and fresh work."""
import json, pathlib, collections

root = pathlib.Path('~/census_toc').expanduser()
manifest = root / 'organized_manifest.jsonl'
pending_txt = root / 'pending_age_images.txt'
pending_jsonl = root / 'pending_age_images.jsonl'

failed = set()
for line in manifest.open():
    row = json.loads(line)
    if row.get('classification') == 'age_table_failed' and row.get('source_image'):
        failed.add(row['source_image'])

pending = [line.strip() for line in pending_txt.open() if line.strip()]
fresh = [img for img in pending if img not in failed]

(root / 'rerun_failed_age_images.txt').write_text(
    ''.join(img + '\n' for img in sorted(failed)),
    encoding='utf-8',
)
(root / 'pending_age_images_fresh.txt').write_text(
    ''.join(img + '\n' for img in fresh),
    encoding='utf-8',
)

topic = collections.Counter()
year = collections.Counter()
source = collections.Counter()
for line in pending_jsonl.open():
    row = json.loads(line)
    topic[row.get('canonical_topic') or 'unknown'] += 1
    year[str(row.get('year') or 'unknown')] += 1
    source['dir_name_match' if row.get('dir_name_match') else 'metadata_only'] += 1

print('failed age images:', len(failed))
print('pending fresh images:', len(fresh))
print('\nTop topics:')
for k, v in topic.most_common(12):
    print(v, k)
print('\nTop years:')
for k, v in year.most_common(12):
    print(v, k)
print('\nMatch source:')
for k, v in source.items():
    print(v, k)
