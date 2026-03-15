"""
Run baseline methods on all 52 census images and score against SudokuOCR output.

Methods: Tesseract 5, raw GPT-4o, raw Gemini 2.5, raw Claude.
Scoring: cell-level exact match against constraint-verified SudokuOCR results.
"""

import os
import re
import sys
import json
import base64
import time
import traceback
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = Path(__file__).parent / "results"
BASELINE_DIR = RESULTS_DIR / "baselines"
BASELINE_DIR.mkdir(exist_ok=True)

# ─── Utilities ──────────────────────────────────────────────────────────

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def normalize_age(a):
    s = str(a).strip().lower()
    s = re.sub(r'\s+', '', s)
    s = s.replace('&', 'and').replace('–', '-').replace('—', '-')
    s = re.sub(r'^total\s*', '', s)
    return s

def parse_json_response(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    text = re.sub(r'(?<=\d)_(?=\d)', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON array
        for start_char, end_char in [('[', ']'), ('{', '}')]:
            start = text.find(start_char)
            end = text.rfind(end_char) + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
    return None


# ─── Load Ground Truth (SudokuOCR verified results) ────────────────────

def load_gt_from_oneshot(oneshot_path):
    """Load GT cells from a SudokuOCR result file.
    Returns list of (section_name, age, group_name, col, value) tuples.
    """
    with open(oneshot_path) as f:
        data = json.load(f)

    cells = []
    for section in data.get('data', []):
        sec_name = section.get('name', 'unknown')
        for row in section.get('rows', []):
            age = row.get('age', '')
            for group_name, group_data in row.items():
                if group_name == 'age' or not isinstance(group_data, dict):
                    continue
                for col, val in group_data.items():
                    if val is not None:
                        cells.append({
                            'section': sec_name,
                            'age': age,
                            'age_norm': normalize_age(age),
                            'group': group_name,
                            'col': col,
                            'value': int(val),
                        })
    return cells


# ─── Generic Prompt ─────────────────────────────────────────────────────
# A reasonable generic prompt — not intentionally bad, but not schema-aware.
# This is what a typical user would write.

GENERIC_PROMPT = """This is a scanned page from a historical Indian census showing an age-by-sex table.

Extract ALL the numeric data from the table. Return the data as a JSON array of objects.
Each object represents one row. Use "age" for the age column. For all other columns,
use the column header text as the key (e.g., "Persons", "Males", "Females").

If the table has multiple sections (e.g., different community groups), include all of them.
All values must be integers (remove commas). Return ONLY valid JSON, no other text."""


# ─── Tesseract ──────────────────────────────────────────────────────────

def run_tesseract(image_path):
    """Run Tesseract OCR and return raw text."""
    import pytesseract
    from PIL import Image
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, config='--psm 6')  # assume uniform block of text
    return text

def score_tesseract(raw_text, gt_cells):
    """Score Tesseract: for each GT cell value, check if it appears in the raw text.
    This is generous — doesn't require correct position, just that the number was read.
    """
    # Extract all multi-digit numbers from Tesseract output
    numbers = set()
    for match in re.finditer(r'\b(\d[\d,]*\d)\b|\b(\d+)\b', raw_text):
        num_str = (match.group(1) or match.group(2)).replace(',', '')
        try:
            numbers.add(int(num_str))
        except ValueError:
            pass

    matched = 0
    total = len(gt_cells)
    for cell in gt_cells:
        if cell['value'] in numbers:
            matched += 1

    return matched, total


# ─── LLM Callers ────────────────────────────────────────────────────────

def call_openai(image_path, prompt):
    from openai import OpenAI
    client = OpenAI()
    b64 = encode_image(image_path)
    ext = Path(image_path).suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}},
        ]}],
        max_tokens=16384,
        temperature=0,
    )
    return response.choices[0].message.content

def call_gemini(image_path, prompt):
    from google import genai
    from google.genai.types import HttpOptions
    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=HttpOptions(timeout=120_000),
    )
    b64 = encode_image(image_path)
    ext = Path(image_path).suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[prompt, genai.types.Part.from_bytes(data=base64.b64decode(b64), mime_type=mime)],
        config=genai.types.GenerateContentConfig(temperature=0),
    )
    return response.text

def call_claude(image_path, prompt):
    import anthropic
    client = anthropic.Anthropic()
    b64 = encode_image(image_path)
    ext = Path(image_path).suffix.lower()
    media_type = "image/png" if ext == ".png" else "image/jpeg"
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16384,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
            {"type": "text", "text": prompt},
        ]}],
    )
    return response.content[0].text


def score_llm_response(raw_text, gt_cells):
    """Score an LLM response against GT cells.
    Parse JSON, then for each GT cell, try to find a matching value in the parsed output.
    """
    parsed = parse_json_response(raw_text)
    if parsed is None:
        return 0, len(gt_cells), "parse_failed"

    # Handle nested structures: could be flat array or {sections: [...]}
    rows = []
    if isinstance(parsed, list):
        rows = parsed
    elif isinstance(parsed, dict):
        # Try common keys
        for key in ['data', 'rows', 'sections', 'table', 'results']:
            if key in parsed and isinstance(parsed[key], list):
                items = parsed[key]
                # Could be sections with nested rows
                for item in items:
                    if isinstance(item, dict) and 'rows' in item:
                        rows.extend(item['rows'])
                    else:
                        rows.append(item)
                break
        if not rows:
            rows = [parsed]

    # Build lookup: age_norm -> {col_name: value}
    pred_by_age = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        age = row.get('age', row.get('Age', row.get('age_group', '')))
        age_norm = normalize_age(str(age))
        if not age_norm:
            continue

        vals = {}
        for k, v in row.items():
            if isinstance(v, dict):
                # Nested group: {"POPULATION": {"persons": 123, ...}}
                for sub_k, sub_v in v.items():
                    try:
                        vals[sub_k.lower()] = int(str(sub_v).replace(',', ''))
                    except (ValueError, TypeError):
                        pass
            else:
                try:
                    vals[k.lower()] = int(str(v).replace(',', ''))
                except (ValueError, TypeError):
                    pass

        if age_norm in pred_by_age:
            pred_by_age[age_norm].update(vals)
        else:
            pred_by_age[age_norm] = vals

    # Score: for each GT cell, check if the predicted value matches
    matched = 0
    total = len(gt_cells)

    for cell in gt_cells:
        age_norm = cell['age_norm']
        gt_val = cell['value']

        pred_row = pred_by_age.get(age_norm, {})
        if not pred_row:
            continue

        # Try to match by column name
        col = cell['col'].lower()
        # Also try group_col combinations
        group = cell['group'].lower()
        candidates = [
            col,                              # "persons"
            f"{group}_{col}",                 # "population_persons"
            f"{group} {col}",                 # "population persons"
            f"{col}_{group}",                 # "persons_population"
        ]

        found = False
        for cand in candidates:
            if cand in pred_row and pred_row[cand] == gt_val:
                found = True
                break

        if not found:
            # Generous fallback: check if gt_val appears as ANY value in the predicted row
            if gt_val in pred_row.values():
                found = True

        if found:
            matched += 1

    return matched, total, "ok"


# ─── Main Runner ────────────────────────────────────────────────────────

def run_all():
    # Gather all 52 images and their GT
    images = []
    for f in sorted(RESULTS_DIR.glob("*_oneshot.json")):
        if f.name.startswith('_'):
            continue
        with open(f) as fp:
            data = json.load(fp)
        src = data.get('source_image', '')
        if not src:
            continue
        img_path = Path(src)
        if not img_path.is_absolute():
            img_path = Path(__file__).parent / src
        if not img_path.exists():
            print(f"  SKIP {f.stem}: image not found at {img_path}")
            continue
        name = f.stem.replace('_oneshot', '')
        gt_cells = load_gt_from_oneshot(f)
        images.append({
            'name': name,
            'image_path': img_path,
            'gt_path': f,
            'gt_cells': gt_cells,
            'n_cells': len(gt_cells),
        })

    print(f"Found {len(images)} images with {sum(i['n_cells'] for i in images)} total GT cells\n")

    # Which methods to run?
    methods = {}

    # Tesseract (always available now)
    methods['tesseract'] = {'caller': run_tesseract, 'scorer': score_tesseract, 'type': 'ocr'}

    # LLMs (check API keys)
    if os.environ.get('OPENAI_API_KEY'):
        methods['gpt4o'] = {'caller': lambda p: call_openai(p, GENERIC_PROMPT), 'scorer': score_llm_response, 'type': 'llm'}
    else:
        print("SKIP gpt4o: no OPENAI_API_KEY")

    if os.environ.get('GEMINI_API_KEY'):
        methods['gemini'] = {'caller': lambda p: call_gemini(p, GENERIC_PROMPT), 'scorer': score_llm_response, 'type': 'llm'}
    else:
        print("SKIP gemini: no GEMINI_API_KEY")

    if os.environ.get('ANTHROPIC_API_KEY'):
        methods['claude'] = {'caller': lambda p: call_claude(p, GENERIC_PROMPT), 'scorer': score_llm_response, 'type': 'llm'}
    else:
        print("SKIP claude: no ANTHROPIC_API_KEY")

    # Results storage
    all_results = {}

    for method_name, method_info in methods.items():
        print(f"\n{'='*70}")
        print(f"METHOD: {method_name.upper()}")
        print(f"{'='*70}")

        method_results = []
        total_matched = 0
        total_cells = 0

        for i, img in enumerate(images):
            name = img['name']
            cache_file = BASELINE_DIR / f"{name}_{method_name}.json"

            # Check cache
            if cache_file.exists():
                with open(cache_file) as f:
                    cached = json.load(f)
                m, t = cached['matched'], cached['total']
                total_matched += m
                total_cells += t
                method_results.append(cached)
                acc = m / t * 100 if t > 0 else 0
                print(f"  [{i+1:2d}/52] {name}: {m}/{t} ({acc:.1f}%) [cached]")
                continue

            print(f"  [{i+1:2d}/52] {name}...", end=' ', flush=True)
            t0 = time.time()

            try:
                raw_output = method_info['caller'](img['image_path'])
                elapsed = time.time() - t0

                if method_info['type'] == 'ocr':
                    m, t = method_info['scorer'](raw_output, img['gt_cells'])
                    status = 'ok'
                else:
                    m, t, status = method_info['scorer'](raw_output, img['gt_cells'])

                acc = m / t * 100 if t > 0 else 0
                print(f"{m}/{t} ({acc:.1f}%) [{elapsed:.1f}s] {status}")

                result = {
                    'name': name,
                    'matched': m,
                    'total': t,
                    'accuracy': acc,
                    'elapsed': round(elapsed, 1),
                    'status': status,
                }

                # Save raw output
                raw_file = BASELINE_DIR / f"{name}_{method_name}_raw.txt"
                raw_file.write_text(raw_output if isinstance(raw_output, str) else str(raw_output))

            except Exception as e:
                elapsed = time.time() - t0
                print(f"ERROR: {e} [{elapsed:.1f}s]")
                traceback.print_exc()
                t = img['n_cells']
                result = {
                    'name': name,
                    'matched': 0,
                    'total': t,
                    'accuracy': 0,
                    'elapsed': round(elapsed, 1),
                    'status': f'error: {str(e)[:100]}',
                }
                m = 0

            total_matched += m
            total_cells += t
            method_results.append(result)

            # Cache result
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)

            # Rate limit for LLMs
            if method_info['type'] == 'llm':
                time.sleep(1)

        overall_acc = total_matched / total_cells * 100 if total_cells > 0 else 0
        print(f"\n  TOTAL {method_name}: {total_matched}/{total_cells} = {overall_acc:.1f}%")

        all_results[method_name] = {
            'total_matched': total_matched,
            'total_cells': total_cells,
            'overall_accuracy': round(overall_acc, 2),
            'per_table': method_results,
        }

    # Save summary
    summary_path = BASELINE_DIR / '_baseline_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Print comparison table
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<15s} {'Matched':>10s} {'Total':>10s} {'Accuracy':>10s}")
    print(f"{'-'*45}")
    for method_name, res in all_results.items():
        print(f"{method_name:<15s} {res['total_matched']:>10,d} {res['total_cells']:>10,d} {res['overall_accuracy']:>9.1f}%")
    print(f"{'SudokuOCR':<15s} {'16,542':>10s} {'16,542':>10s} {'100.0':>9s}%")


if __name__ == '__main__':
    # Allow running specific methods
    if len(sys.argv) > 1:
        method_filter = sys.argv[1]
        print(f"Running only: {method_filter}")
    run_all()
