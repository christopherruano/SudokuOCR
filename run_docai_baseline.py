"""Run Google Document AI Form Parser on all 52 images."""
import sys, json, time, re
from pathlib import Path
from google.cloud import documentai_v1 as documentai

RESULTS_DIR = Path(__file__).parent / "results"
BASELINE_DIR = RESULTS_DIR / "baselines"
BASELINE_DIR.mkdir(exist_ok=True)

PROJECT_ID = "823398653757"
LOCATION = "us"
PROCESSOR_ID = "f7e1db20f1694974"

client = documentai.DocumentProcessorServiceClient()
processor_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}"


def normalize_age(a):
    s = str(a).strip().lower()
    s = re.sub(r'\s+', '', s)
    s = s.replace('&', 'and').replace('\u2013', '-').replace('\u2014', '-')
    if s == 'total':
        return 'total'
    s = re.sub(r'^total', '', s)
    return s


def process_image(image_path):
    """Send image to Document AI and return raw text + any table data."""
    with open(image_path, "rb") as f:
        content = f.read()

    mime = "image/png" if str(image_path).endswith('.png') else "image/jpeg"
    raw_document = documentai.RawDocument(content=content, mime_type=mime)
    request = documentai.ProcessRequest(name=processor_name, raw_document=raw_document)
    result = client.process_document(request=request)
    return result.document


def extract_numbers_from_text(text):
    """Extract all numbers from Document AI text output (generous scoring like Tesseract)."""
    numbers = set()
    for match in re.finditer(r'\b(\d[\d,]*\d)\b|\b(\d+)\b', text):
        num_str = (match.group(1) or match.group(2)).replace(',', '')
        try:
            numbers.add(int(num_str))
        except ValueError:
            pass
    return numbers


def score_generous(numbers, gt_cells):
    """Generous scoring: check if GT value appears in extracted numbers."""
    matched = 0
    for cell in gt_cells:
        if cell['value'] in numbers:
            matched += 1
    return matched, len(gt_cells)


def score_table_aware(document, gt_cells):
    """Try to score using Document AI's table detection if available."""
    # First try table-based scoring
    tables_found = False
    pred_by_age = {}

    for page in document.pages:
        for table in page.tables:
            tables_found = True
            # Extract header row
            headers = []
            if table.header_rows:
                for cell in table.header_rows[0].cells:
                    text = get_cell_text(cell, document.text).strip()
                    headers.append(text.lower())

            # Extract body rows
            for body_row in table.body_rows:
                row_vals = {}
                age_val = None
                for j, cell in enumerate(body_row.cells):
                    text = get_cell_text(cell, document.text).strip()
                    if j == 0:
                        age_val = normalize_age(text)
                    else:
                        try:
                            val = int(text.replace(',', '').replace(' ', ''))
                            col_name = headers[j] if j < len(headers) else f'col{j}'
                            row_vals[col_name] = val
                        except (ValueError, IndexError):
                            pass
                if age_val and row_vals:
                    if age_val in pred_by_age:
                        pred_by_age[age_val].update(row_vals)
                    else:
                        pred_by_age[age_val] = row_vals

    if not tables_found or not pred_by_age:
        return None  # Fall back to generous scoring

    # Score against GT
    matched = 0
    for cell in gt_cells:
        age_norm = cell['age_norm']
        gt_val = cell['value']
        pred_row = pred_by_age.get(age_norm, {})
        if not pred_row:
            continue
        col = cell['col'].lower()
        group = cell['group'].lower()
        candidates = [col, f'{group}_{col}', f'{group} {col}', f'{col}_{group}']
        found = False
        for cand in candidates:
            if cand in pred_row and pred_row[cand] == gt_val:
                found = True
                break
        if not found and gt_val in pred_row.values():
            found = True
        if found:
            matched += 1

    return matched, len(gt_cells)


def get_cell_text(cell, full_text):
    """Extract text from a table cell using text anchors."""
    text = ""
    if cell.layout and cell.layout.text_anchor and cell.layout.text_anchor.text_segments:
        for segment in cell.layout.text_anchor.text_segments:
            start = int(segment.start_index) if segment.start_index else 0
            end = int(segment.end_index)
            text += full_text[start:end]
    return text


# Run
total_matched = 0
total_cells = 0
results = []

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

    # Load GT cells
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

    # Check cache
    cache_file = BASELINE_DIR / f'{name}_docai.json'
    if cache_file.exists():
        with open(cache_file) as fp:
            cached = json.load(fp)
        m, t = cached['matched'], cached['total']
        total_matched += m
        total_cells += t
        results.append(cached)
        print(f"  [{len(results):2d}/52] {name}: {m}/{t} ({m/t*100:.1f}%) [cached]")
        continue

    print(f"  [{len(results)+1:2d}/52] {name}...", end=' ', flush=True)
    t0 = time.time()
    try:
        document = process_image(img_path)
        elapsed = time.time() - t0

        # Save raw text
        raw_text = document.text
        (BASELINE_DIR / f'{name}_docai_raw.txt').write_text(raw_text)

        # Try table-aware scoring first, fall back to generous
        table_result = score_table_aware(document, gt_cells)
        if table_result is not None:
            m, t = table_result
            scoring = 'table'
        else:
            numbers = extract_numbers_from_text(raw_text)
            m, t = score_generous(numbers, gt_cells)
            scoring = 'text'

        acc = m / t * 100 if t > 0 else 0
        print(f"{m}/{t} ({acc:.1f}%) [{elapsed:.1f}s] [{scoring}]")
        result = {
            'name': name, 'matched': m, 'total': t,
            'accuracy': round(acc, 1), 'elapsed': round(elapsed, 1),
            'status': 'ok', 'scoring': scoring
        }
    except Exception as e:
        elapsed = time.time() - t0
        print(f"ERROR: {e} [{elapsed:.1f}s]")
        t = len(gt_cells)
        result = {
            'name': name, 'matched': 0, 'total': t,
            'accuracy': 0, 'elapsed': round(elapsed, 1),
            'status': f'error: {str(e)[:100]}'
        }
        m = 0

    total_matched += m
    total_cells += t
    results.append(result)
    with open(cache_file, 'w') as fp:
        json.dump(result, fp, indent=2)
    time.sleep(0.5)

overall = total_matched / total_cells * 100 if total_cells > 0 else 0
print(f"\nDOCUMENT AI TOTAL: {total_matched}/{total_cells} = {overall:.1f}%")

with open(BASELINE_DIR / '_docai_summary.json', 'w') as fp:
    json.dump({
        'total_matched': total_matched, 'total_cells': total_cells,
        'overall_accuracy': round(overall, 2), 'per_table': results
    }, fp, indent=2)
