"""Run Google Cloud Vision API on all 52 images."""
import json, time, re
from pathlib import Path
from google.cloud import vision

RESULTS_DIR = Path(__file__).parent / "results"
BASELINE_DIR = RESULTS_DIR / "baselines"
BASELINE_DIR.mkdir(exist_ok=True)

client = vision.ImageAnnotatorClient()


def normalize_age(a):
    s = str(a).strip().lower()
    s = re.sub(r'\s+', '', s)
    s = s.replace('&', 'and').replace('\u2013', '-').replace('\u2014', '-')
    if s == 'total':
        return 'total'
    s = re.sub(r'^total', '', s)
    return s


def extract_numbers(text):
    numbers = set()
    for match in re.finditer(r'\b(\d[\d,]*\d)\b|\b(\d+)\b', text):
        num_str = (match.group(1) or match.group(2)).replace(',', '')
        try:
            numbers.add(int(num_str))
        except ValueError:
            pass
    return numbers


total_matched = 0
total_cells = 0
results_list = []

for f in sorted(RESULTS_DIR.glob('*_oneshot.json')):
    if f.name.startswith('_'):
        continue
    with open(f) as fp:
        data = json.load(fp)
    src = data.get('source_image', '')
    if not src:
        continue
    img_path = Path(src) if Path(src).is_absolute() else Path(__file__).parent / src
    if not img_path.exists():
        img_path = Path(__file__).parent / src
    if not img_path.exists():
        continue
    name = f.stem.replace('_oneshot', '')

    gt_cells = []
    for sec in data.get('data', []):
        for row in sec.get('rows', []):
            age = row.get('age', '')
            for k, v in row.items():
                if k == 'age' or not isinstance(v, dict):
                    continue
                for col, val in v.items():
                    if val is not None:
                        gt_cells.append({
                            'age_norm': normalize_age(age),
                            'group': k, 'col': col, 'value': int(val)
                        })

    cache_file = BASELINE_DIR / f'{name}_vision.json'
    if cache_file.exists():
        with open(cache_file) as fp:
            cached = json.load(fp)
        m, t = cached['matched'], cached['total']
        total_matched += m
        total_cells += t
        results_list.append(cached)
        print(f"  [{len(results_list):2d}/52] {name}: {m}/{t} ({m/t*100:.1f}%) [cached]")
        continue

    print(f"  [{len(results_list)+1:2d}/52] {name}...", end=' ', flush=True)
    t0 = time.time()
    try:
        with open(img_path, 'rb') as img_file:
            content = img_file.read()
        image = vision.Image(content=content)
        response = client.document_text_detection(image=image)
        elapsed = time.time() - t0

        if response.error.message:
            raise Exception(response.error.message)

        raw_text = response.full_text_annotation.text if response.full_text_annotation else ""
        (BASELINE_DIR / f'{name}_vision_raw.txt').write_text(raw_text)

        numbers = extract_numbers(raw_text)
        m = sum(1 for c in gt_cells if c['value'] in numbers)
        t = len(gt_cells)
        acc = m / t * 100 if t > 0 else 0
        print(f"{m}/{t} ({acc:.1f}%) [{elapsed:.1f}s]")
        result = {
            'name': name, 'matched': m, 'total': t,
            'accuracy': round(acc, 1), 'elapsed': round(elapsed, 1), 'status': 'ok'
        }
    except Exception as e:
        elapsed = time.time() - t0
        print(f"ERROR: {str(e)[:100]} [{elapsed:.1f}s]")
        t = len(gt_cells)
        result = {
            'name': name, 'matched': 0, 'total': t,
            'accuracy': 0, 'elapsed': round(elapsed, 1),
            'status': f'error: {str(e)[:100]}'
        }
        m = 0

    total_matched += m
    total_cells += t
    results_list.append(result)
    with open(cache_file, 'w') as fp:
        json.dump(result, fp, indent=2)

overall = total_matched / total_cells * 100 if total_cells > 0 else 0
print(f"\nGOOGLE VISION TOTAL: {total_matched}/{total_cells} = {overall:.1f}%")

with open(BASELINE_DIR / '_vision_summary.json', 'w') as fp:
    json.dump({
        'total_matched': total_matched, 'total_cells': total_cells,
        'overall_accuracy': round(overall, 2), 'per_table': results_list
    }, fp, indent=2)
