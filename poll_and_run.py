#!/usr/bin/env python3
"""Poll for gemini-3.1-pro-preview availability, then run the 4 remaining files.
Retries each file up to 3 times if it fails (e.g., 503/504 mid-extraction)."""
import time, os, subprocess, sys, json

os.environ['GEMINI_MODEL'] = 'gemini-3.1-pro-preview'

# Import after setting env
import base64
from pipeline import call_gemini

img = base64.b64encode(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100).decode()

def check_available():
    """Return True if 3.1-pro is responding."""
    try:
        call_gemini(img, 'Say OK')
        return True  # 200 = up
    except Exception as e:
        msg = str(e)
        if msg[:3] == '400':
            return True  # 400 = bad image = model is up
        return False

print("Polling for gemini-3.1-pro-preview availability...")
print(f"Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

while not check_available():
    print(f"{time.strftime('%H:%M:%S')} — still down, retrying in 2 min...")
    time.sleep(120)

print(f"\n*** 3.1-PRO IS AVAILABLE at {time.strftime('%H:%M:%S')} ***\n")

# Now run the 4 remaining files sequentially, with retries
files = [
    "age_tables/Mysore/1891/Mysore_age_1891-10.png",
    "age_tables/Hyderabad/1931/Parbhani.png",
    "age_tables/Hyderabad/1931/Warangal.png",
    "age_tables/Hyderabad/1941/Atraf-i-balda.png",
]

MAX_RETRIES = 3

for img_path in files:
    name = os.path.basename(img_path)
    # Determine the expected output JSON name
    json_name = img_path.split('/')[-1].replace('.png', '')
    # Get year from path
    parts = img_path.split('/')
    year = None
    for p in parts:
        if p.isdigit() and len(p) == 4:
            year = p
            break

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n{'='*60}")
        print(f"Running: {name} (attempt {attempt}/{MAX_RETRIES})")
        print(f"{'='*60}")

        # Wait briefly between retries to let model recover
        if attempt > 1:
            wait = 60 * attempt
            print(f"  Waiting {wait}s before retry...")
            time.sleep(wait)
            # Re-check availability
            if not check_available():
                print(f"  Model went down, waiting for recovery...")
                while not check_available():
                    time.sleep(120)
                print(f"  Model recovered at {time.strftime('%H:%M:%S')}")

        result = subprocess.run(
            [sys.executable, "oneshot.py", img_path],
            env={**os.environ, 'GEMINI_MODEL': 'gemini-3.1-pro-preview'},
            capture_output=False,
            timeout=1200,  # 20 min max per file
        )

        if result.returncode != 0:
            print(f"  FAILED with exit code {result.returncode}")
            continue

        # Check if the result has 0 failures
        # Find the result JSON
        found = False
        for rf in os.listdir('results'):
            if rf.endswith('_oneshot.json') and json_name.replace('.png','') in rf:
                with open(f'results/{rf}') as fh:
                    data = json.load(fh)
                c = data.get('constraints', {})
                failed = c.get('failed', 0)
                print(f"  Result: {failed} failures")
                if failed == 0:
                    print(f"  PERFECT!")
                    found = True
                break

        if found:
            break
        elif attempt < MAX_RETRIES:
            print(f"  Still has failures, will retry...")
    else:
        print(f"  Exhausted {MAX_RETRIES} attempts for {name}")

# Final scoreboard
print(f"\n{'='*60}")
print("FINAL SCOREBOARD")
print(f"{'='*60}")
perfect = 0
failing = []
total = 0
for f in sorted(os.listdir('results')):
    if not f.endswith('_oneshot.json'):
        continue
    total += 1
    with open(f'results/{f}') as fh:
        data = json.load(fh)
    c = data.get('constraints', {})
    failed = c.get('failed', 0)
    if failed > 0:
        failing.append((f, failed))
    else:
        perfect += 1

print(f"Perfect: {perfect}/{total}")
if failing:
    print(f"Failing: {len(failing)} files, {sum(x[1] for x in failing)} total failures")
    for f, n in failing:
        print(f"  {n:3d} failures: {f}")
else:
    print("ALL 52 FILES PERFECT!")
