#!/usr/bin/env python3
"""Ping every API key in .env once and report which are dead/exhausted.
SECURITY: only ever prints a masked label (var name + last 4 chars) + status. Never the key.
"""
import os, json, urllib.request, urllib.error

ENV = {}
for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        ENV[k.strip()] = v.strip().strip('"').strip("'")


def mask(k):
    return f"…{k[-4:]}" if k and len(k) >= 4 else "(empty)"


def http(url, headers=None, timeout=20):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, ""
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode()[:200]
        except Exception:
            pass
        return e.code, body
    except Exception as e:
        return None, str(e)[:120]


def classify(code, body):
    b = (body or "").lower()
    if code == 200:
        return "OK"
    if code in (401, 403) or "api key not valid" in b or "api_key_invalid" in b or "permission_denied" in b or "unauthorized" in b or "invalid" in b:
        return "INVALID (refresh)"
    if code == 429 or "resource_exhausted" in b or "quota" in b or "exhausted" in b or "rate" in b:
        return "EXHAUSTED / rate-limited"
    return f"OTHER (http {code})"


def test_harvard(key, base_url, label):
    try:
        from google import genai
        client = genai.Client(api_key=key, http_options=genai.types.HttpOptions(base_url=base_url, timeout=30000))
        model = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
        client.models.generate_content(model=model, contents=["ping"],
                                       config=genai.types.GenerateContentConfig(temperature=0))
        return "OK"
    except Exception as e:
        s = str(e).lower()
        if "api key not valid" in s or "api_key_invalid" in s or "permission_denied" in s or "403" in s or "401" in s:
            return "INVALID (refresh)"
        if "resource_exhausted" in s or "429" in s or "quota" in s or "exhausted" in s:
            return "EXHAUSTED / rate-limited"
        if "not_found" in s or "404" in s:
            return "OK-ish (auth fine; model 404)"
        return f"OTHER ({str(e)[:70]})"


def main():
    print("=" * 64)
    print("API KEY HEALTH CHECK  (labels masked — last 4 chars only)")
    print("=" * 64)

    # --- provider keys via REST ---
    rows = []
    if ENV.get("OPENAI_API_KEY"):
        c, b = http("https://api.openai.com/v1/models", {"Authorization": f"Bearer {ENV['OPENAI_API_KEY']}"})
        rows.append(("OPENAI_API_KEY", mask(ENV["OPENAI_API_KEY"]), classify(c, b)))
    if ENV.get("ANTHROPIC_API_KEY"):
        c, b = http("https://api.anthropic.com/v1/models",
                    {"x-api-key": ENV["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"})
        rows.append(("ANTHROPIC_API_KEY", mask(ENV["ANTHROPIC_API_KEY"]), classify(c, b)))
    if ENV.get("MISTRAL_API_KEY"):
        c, b = http("https://api.mistral.ai/v1/models", {"Authorization": f"Bearer {ENV['MISTRAL_API_KEY']}"})
        rows.append(("MISTRAL_API_KEY", mask(ENV["MISTRAL_API_KEY"]), classify(c, b)))
    if ENV.get("GEMINI_API_KEY"):
        c, b = http(f"https://generativelanguage.googleapis.com/v1beta/models?key={ENV['GEMINI_API_KEY']}")
        rows.append(("GEMINI_API_KEY (direct)", mask(ENV["GEMINI_API_KEY"]), classify(c, b)))

    for name, m, st in rows:
        print(f"  {name:26} {m:8}  {st}")

    # --- Harvard gateway keys (dedup across the three vars) ---
    base = ENV.get("HARVARD_GEMINI_BASE_URL", "")
    seen = {}
    def add(varlabel, val):
        for i, k in enumerate([x.strip() for x in val.split(",") if x.strip()]):
            seen.setdefault(k, []).append(f"{varlabel}[{i}]")
    add("KEYS", ENV.get("HARVARD_GEMINI_API_KEYS", ""))
    if ENV.get("HARVARD_GEMINI_API_KEY"):
        seen.setdefault(ENV["HARVARD_GEMINI_API_KEY"], []).append("KEY")
    add("EXHAUSTED_LIST", ENV.get("HARVARD_GEMINI_API_KEYS_EXHAUSTED", ""))

    print(f"\n  Harvard HUIT gateway ({base}) — {len(seen)} distinct keys:")
    dead = []
    for k, where in seen.items():
        st = test_harvard(k, base, where)
        print(f"  {','.join(where):24} {mask(k):8}  {st}")
        if "OK" not in st:
            dead.append((mask(k), ",".join(where), st))

    print("\n" + "=" * 64)
    print("NEEDS REFRESH:")
    for m, where, st in dead:
        print(f"  {m}  ({where})  -> {st}")
    if not dead:
        print("  none — all Harvard keys OK")


if __name__ == "__main__":
    main()
