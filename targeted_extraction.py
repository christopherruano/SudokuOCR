"""
Targeted extraction: zoom into specific table regions and enhance image quality
before sending to vision APIs. Tests whether higher-res cropped regions
produce better digit recognition on the stubborn error cells.
"""

import os
import json
import base64
import io
import time
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
AGE_TABLES = PROJECT_ROOT / "age_tables"
RESULTS_DIR = PROJECT_ROOT / "results"


def encode_pil_image(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def enhance_image(img, sharpen=2.0, contrast=1.5, upscale=2):
    """Enhance image: upscale, sharpen, increase contrast."""
    if upscale > 1:
        w, h = img.size
        img = img.resize((w * upscale, h * upscale), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(sharpen)
    return img


def binarize_image(img, threshold=140):
    """Convert to clean black and white."""
    gray = img.convert("L")
    return gray.point(lambda p: 255 if p > threshold else 0).convert("RGB")


def call_api(model_name, b64_image, prompt, mime="image/png"):
    if model_name == "openai":
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_image}", "detail": "high"}},
            ]}],
            max_tokens=4096, temperature=0,
        )
        return response.choices[0].message.content
    elif model_name == "gemini":
        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[prompt, genai.types.Part.from_bytes(data=base64.b64decode(b64_image), mime_type=mime)],
            config=genai.types.GenerateContentConfig(temperature=0),
        )
        return response.text
    elif model_name == "claude":
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64_image}},
                {"type": "text", "text": prompt},
            ]}],
        )
        return response.content[0].text


def test_travancore_targeted():
    """Zoom into the specific rows where all models fail on Travancore."""
    img_path = AGE_TABLES / "Travancore/1901/Eastern_division_age_1901.png"
    img = Image.open(img_path)
    w, h = img.size
    print(f"Original image: {w}x{h}")

    # The problem rows are roughly in the middle of the table
    # Row 20-25 is approximately at y=55-60% of image height
    # Row 30-35 is approximately at y=65-70%
    # Let's crop a region covering rows 15-20 through 40-45

    regions = {
        "rows_20_35": (0.0, 0.48, 1.0, 0.72),  # left, top, right, bottom as fractions
        "rows_45_60over": (0.0, 0.72, 1.0, 0.92),
        "full_data": (0.0, 0.20, 1.0, 0.95),
    }

    # Image enhancement variations
    enhancements = {
        "original": lambda img: img,
        "enhanced": lambda img: enhance_image(img, sharpen=2.0, contrast=1.5, upscale=2),
        "binarized": lambda img: binarize_image(img, threshold=140),
        "enhanced_binarized": lambda img: binarize_image(enhance_image(img, sharpen=2.0, contrast=1.8, upscale=3), threshold=128),
    }

    available_models = []
    for m, env in [("gemini", "GEMINI_API_KEY"), ("claude", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY")]:
        if os.environ.get(env):
            available_models.append(m)

    prompt_targeted = """You are an expert at reading numbers in historical census tables. This image shows rows from an Indian census table (1901).

The POPULATION section has columns: Persons, Males, Females.

Read EVERY DIGIT with extreme care. These old typefaces have confusing characters:
- The digit 6 often looks like 5 (look for the curved bottom of 6)
- The digit 0 can look like 9 (check if the top is fully closed)
- The digit 4 can look like 9 or 1
- Check: for each row, Persons MUST equal Males + Females

Extract the SUMMARY age group rows (like "0-5", "5-10", "10-15", "15-20", "20-25", etc.).
Do NOT extract individual year rows (0-1, 1-2, etc.).

For EACH number, before writing it down, mentally verify:
1. Count the digits - is it the right number of digits?
2. Does Persons = Males + Females?

Return ONLY a JSON array: [{"age": "...", "persons": N, "males": N, "females": N}]
All integers, no commas. Return ONLY valid JSON."""

    results = {}

    # Test 1: Full image with different enhancements
    print("\n=== Test: Full image with enhancements ===")
    for enh_name, enh_fn in enhancements.items():
        print(f"\n  Enhancement: {enh_name}")
        processed = enh_fn(img)
        b64 = encode_pil_image(processed)
        print(f"    Image size: {processed.size[0]}x{processed.size[1]}, b64 len: {len(b64)}")

        for model_name in available_models[:2]:  # test with 2 models to save costs
            try:
                t0 = time.time()
                raw = call_api(model_name, b64, prompt_targeted)
                elapsed = time.time() - t0

                import re
                raw_clean = re.sub(r'(?<=\d)_(?=\d)', '', raw)
                if "```" in raw_clean:
                    lines = raw_clean.split("\n")
                    lines = [l for l in lines if not l.strip().startswith("```")]
                    raw_clean = "\n".join(lines)

                start = raw_clean.find("[")
                end = raw_clean.rfind("]") + 1
                if start >= 0 and end > start:
                    parsed = json.loads(raw_clean[start:end])
                else:
                    parsed = json.loads(raw_clean)

                # Check the specific problem cells
                key = f"{enh_name}_{model_name}"
                results[key] = {}
                for row in parsed:
                    age = str(row.get("age", "")).replace("\u2013", "-").lower().strip()
                    if "20" in age and "25" in age:
                        results[key]["20-25"] = {
                            "persons": row.get("persons"),
                            "males": row.get("males"),
                            "females": row.get("females"),
                        }
                        # GT: persons=112040, males=53092, females=58948
                        p_ok = row.get("persons") == 112040
                        m_ok = row.get("males") == 53092
                        f_ok = row.get("females") == 58948
                        status = "PERFECT" if (p_ok and m_ok and f_ok) else "ERRORS"
                        print(f"    [{model_name}] {elapsed:.1f}s | 20-25: P={row.get('persons')} M={row.get('males')} F={row.get('females')} → {status}")

                    if "30" in age and "35" in age:
                        results[key]["30-35"] = {
                            "persons": row.get("persons"),
                            "males": row.get("males"),
                            "females": row.get("females"),
                        }
                        # GT: persons=96124, males=49257, females=46867
                        p_ok = row.get("persons") == 96124
                        m_ok = row.get("males") == 49257
                        f_ok = row.get("females") == 46867
                        status = "PERFECT" if (p_ok and m_ok and f_ok) else "ERRORS"
                        print(f"    [{model_name}]         | 30-35: P={row.get('persons')} M={row.get('males')} F={row.get('females')} → {status}")

            except Exception as e:
                print(f"    [{model_name}] ERROR: {e}")

    # Test 2: Cropped + enhanced regions
    print("\n=== Test: Cropped + enhanced regions ===")
    for region_name, (left, top, right, bottom) in regions.items():
        crop = img.crop((int(w*left), int(h*top), int(w*right), int(h*bottom)))
        enhanced = enhance_image(crop, sharpen=2.5, contrast=1.8, upscale=3)
        b64 = encode_pil_image(enhanced)
        print(f"\n  Region: {region_name} ({enhanced.size[0]}x{enhanced.size[1]})")

        for model_name in available_models[:2]:
            try:
                t0 = time.time()
                raw = call_api(model_name, b64, prompt_targeted)
                elapsed = time.time() - t0

                import re
                raw_clean = re.sub(r'(?<=\d)_(?=\d)', '', raw)
                if "```" in raw_clean:
                    lines = raw_clean.split("\n")
                    lines = [l for l in lines if not l.strip().startswith("```")]
                    raw_clean = "\n".join(lines)
                start = raw_clean.find("[")
                end = raw_clean.rfind("]") + 1
                if start >= 0 and end > start:
                    parsed = json.loads(raw_clean[start:end])
                else:
                    parsed = json.loads(raw_clean)

                for row in parsed:
                    age = str(row.get("age", "")).replace("\u2013", "-").lower().strip()
                    if "20" in age and "25" in age:
                        p_ok = row.get("persons") == 112040
                        m_ok = row.get("males") == 53092
                        f_ok = row.get("females") == 58948
                        status = "PERFECT" if (p_ok and m_ok and f_ok) else "ERRORS"
                        print(f"    [{model_name}] {elapsed:.1f}s | 20-25: P={row.get('persons')} M={row.get('males')} F={row.get('females')} → {status}")
                    if "30" in age and "35" in age:
                        p_ok = row.get("persons") == 96124
                        m_ok = row.get("males") == 49257
                        f_ok = row.get("females") == 46867
                        status = "PERFECT" if (p_ok and m_ok and f_ok) else "ERRORS"
                        print(f"    [{model_name}]         | 30-35: P={row.get('persons')} M={row.get('males')} F={row.get('females')} → {status}")

            except Exception as e:
                print(f"    [{model_name}] ERROR: {e}")

    print("\n" + json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    test_travancore_targeted()
