"""Run a single baseline method on all 52 images."""
import sys, os, re, json, base64, time, traceback
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

RESULTS_DIR = Path(__file__).parent / "results"
BASELINE_DIR = RESULTS_DIR / "baselines"
BASELINE_DIR.mkdir(exist_ok=True)

method_name = sys.argv[1]  # tesseract, gemini, gpt4o, claude

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

GENERIC_PROMPT = """This is a scanned page from a historical Indian census showing an age-by-sex table.

Extract ALL the numeric data from the table. Return the data as a JSON array of objects.
Each object represents one row. Use "age" for the age column. For all other columns,
use the column header text as the key (e.g., "Persons", "Males", "Females").

If the table has multiple sections (e.g., different community groups), include all of them.
All values must be integers (remove commas). Return ONLY valid JSON, no other text."""

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
        for sc, ec in [('[', ']'), ('{', '}')]:
            start = text.find(sc)
            end = text.rfind(ec) + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
    return None

def normalize_age(a):
    s = str(a).strip().lower()
    s = re.sub(r'\s+', '', s)
    s = s.replace('&', 'and').replace('\u2013', '-').replace('\u2014', '-')
    # Keep "total" as-is if it's the entire string, only strip "total " prefix
    if s == 'total':
        return 'total'
    s = re.sub(r'^total', '', s)
    return s

def score_llm_response(raw_text, gt_cells):
    parsed = parse_json_response(raw_text)
    if parsed is None:
        return 0, len(gt_cells), "parse_failed"
    rows = []
    if isinstance(parsed, list):
        rows = parsed
    elif isinstance(parsed, dict):
        for key in ['data', 'rows', 'sections', 'table', 'results']:
            if key in parsed and isinstance(parsed[key], list):
                items = parsed[key]
                for item in items:
                    if isinstance(item, dict) and 'rows' in item:
                        rows.extend(item['rows'])
                    else:
                        rows.append(item)
                break
        if not rows:
            rows = [parsed]

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
                for sub_k, sub_v in v.items():
                    try: vals[sub_k.lower()] = int(str(sub_v).replace(',', ''))
                    except: pass
            else:
                try: vals[k.lower()] = int(str(v).replace(',', ''))
                except: pass
        if age_norm in pred_by_age:
            pred_by_age[age_norm].update(vals)
        else:
            pred_by_age[age_norm] = vals

    matched = 0
    total = len(gt_cells)
    for cell in gt_cells:
        age_norm = cell['age_norm']
        gt_val = cell['value']
        pred_row = pred_by_age.get(age_norm, {})
        if not pred_row:
            continue
        col = cell['col'].lower()
        group = cell['group'].lower()
        candidates = [col, f"{group}_{col}", f"{group} {col}", f"{col}_{group}"]
        found = False
        for cand in candidates:
            if cand in pred_row and pred_row[cand] == gt_val:
                found = True
                break
        if not found and gt_val in pred_row.values():
            found = True
        if found:
            matched += 1
    return matched, total, "ok"

# Callers
def call_openai(image_path):
    from openai import OpenAI
    client = OpenAI()
    b64 = encode_image(image_path)
    mime = "image/png" if str(image_path).endswith('.png') else "image/jpeg"
    response = client.chat.completions.create(
        model="gpt-4.1", messages=[{"role": "user", "content": [
            {"type": "text", "text": GENERIC_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}},
        ]}], max_tokens=16384, temperature=0)
    return response.choices[0].message.content

def call_gemini(image_path):
    from google import genai
    from google.genai.types import HttpOptions
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=HttpOptions(timeout=120_000))
    b64 = encode_image(image_path)
    mime = "image/png" if str(image_path).endswith('.png') else "image/jpeg"
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[GENERIC_PROMPT, genai.types.Part.from_bytes(data=base64.b64decode(b64), mime_type=mime)],
        config=genai.types.GenerateContentConfig(temperature=0))
    return response.text

def call_claude(image_path):
    import anthropic
    client = anthropic.Anthropic()
    b64 = encode_image(image_path)
    mime = "image/png" if str(image_path).endswith('.png') else "image/jpeg"
    response = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=16384,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
            {"type": "text", "text": GENERIC_PROMPT},
        ]}])
    return response.content[0].text

CALLERS = {'gpt4o': call_openai, 'gemini': call_gemini, 'claude': call_claude}

# Run
caller = CALLERS[method_name]
total_matched = 0
total_cells = 0
results = []

for f in sorted(RESULTS_DIR.glob('*_oneshot.json')):
    if f.name.startswith('_'): continue
    with open(f) as fp:
        data = json.load(fp)
    src = data.get('source_image', '')
    if not src: continue
    img_path = Path(src) if Path(src).is_absolute() else Path(__file__).parent / src
    if not img_path.exists():
        img_path = Path(__file__).parent / src
    if not img_path.exists(): continue
    name = f.stem.replace('_oneshot', '')

    # Load GT cells
    gt_cells = []
    for sec in data.get('data', []):
        for row in sec.get('rows', []):
            age = row.get('age', '')
            for k, v in row.items():
                if k == 'age' or not isinstance(v, dict): continue
                for col, val in v.items():
                    if val is not None:
                        gt_cells.append({'age_norm': normalize_age(age), 'group': k, 'col': col, 'value': int(val)})

    # Check cache
    cache_file = BASELINE_DIR / f'{name}_{method_name}.json'
    if cache_file.exists():
        with open(cache_file) as fp:
            cached = json.load(fp)
        m, t = cached['matched'], cached['total']
        total_matched += m; total_cells += t
        results.append(cached)
        print(f"  [{len(results):2d}/52] {name}: {m}/{t} ({m/t*100:.1f}%) [cached]")
        continue

    print(f"  [{len(results)+1:2d}/52] {name}...", end=' ', flush=True)
    t0 = time.time()
    try:
        raw = caller(img_path)
        elapsed = time.time() - t0
        m, t, status = score_llm_response(raw, gt_cells)
        acc = m / t * 100 if t > 0 else 0
        print(f"{m}/{t} ({acc:.1f}%) [{elapsed:.1f}s]")
        result = {'name': name, 'matched': m, 'total': t, 'accuracy': round(acc, 1), 'elapsed': round(elapsed, 1), 'status': status}
        (BASELINE_DIR / f'{name}_{method_name}_raw.txt').write_text(raw)
    except Exception as e:
        elapsed = time.time() - t0
        print(f"ERROR: {e} [{elapsed:.1f}s]")
        t = len(gt_cells)
        result = {'name': name, 'matched': 0, 'total': t, 'accuracy': 0, 'elapsed': round(elapsed, 1), 'status': f'error: {str(e)[:100]}'}
        m = 0

    total_matched += m; total_cells += t
    results.append(result)
    with open(cache_file, 'w') as fp:
        json.dump(result, fp, indent=2)
    time.sleep(1)  # rate limit

overall = total_matched / total_cells * 100 if total_cells > 0 else 0
print(f"\n{method_name.upper()} TOTAL: {total_matched}/{total_cells} = {overall:.1f}%")

with open(BASELINE_DIR / f'_{method_name}_summary.json', 'w') as fp:
    json.dump({'total_matched': total_matched, 'total_cells': total_cells,
               'overall_accuracy': round(overall, 2), 'per_table': results}, fp, indent=2)
