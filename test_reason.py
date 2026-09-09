#!/usr/bin/env python3
"""Ask Gemini to REASON the correct values from the constraint problem (not just re-read pixels) for a
few small-failure pages. Shows its reasoning + whether the deduced fix actually clears the failures.
  python3 test_reason.py [N]
"""
import sys, os, json
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
sys.argv = ["x"]
import constraint_guided_reocr as C
import census_review_app as A

IMG = "census_site/img"


def reason_prompt(d):
    lines = ["A row from a historical Indian census Table VII (Age, Sex & Civil Condition). The values we",
             "OCR'd are below. Identities that MUST hold: within each civil-condition group Males+Females=",
             "Persons; and across groups, per column, Unmarried+Married+Widowed=Total.\nOCR'd values:"]
    for s in d.get("data", []):
        rows = s.get("rows", [])
        if A.is_transposed(rows):
            continue
        rh = next((k for k in rows[0] if not isinstance(rows[0].get(k), dict)), None) if rows else None
        viol = A.check_constraints(rows)
        for ri in sorted({k[0] for k in viol}):
            lbl = (str(rows[ri].get(rh, "")) if rh else f"r{ri}") or f"r{ri}"
            cells = [f"{g}{{" + ", ".join(f"{k}={v.get(k)}" for k in v) + "}" for g, v in rows[ri].items() if isinstance(v, dict)]
            lines.append(f'  Row "{lbl}": ' + " ; ".join(cells))
    lines.append("\nREASON step by step: which cell(s) are most likely misread, and what value(s) make ALL")
    lines.append("identities hold? (A frequent error is a males/females mis-split — Persons is right but the")
    lines.append("split is wrong, so BOTH sex cells change by the same amount.) Use the scan image to confirm")
    lines.append("the actual printed digits. Explain briefly, THEN end with the fix as JSON on its own line:")
    lines.append('  {"<row label>": {"<group>": {"<subkey>": <int>}}}')
    return "\n".join(lines)


def gemini_reason(client, types, prompt, img):
    cfg = types.GenerateContentConfig(temperature=0.0)   # NO json mode -> free reasoning text
    part = types.Part.from_bytes(data=img, mime_type="image/jpeg")
    return client.models.generate_content(model=C.MODEL, contents=[part, prompt], config=cfg).text or ""


def main():
    keys = C.load_keys()
    targets = []
    for run in A.assembly_runs():
        for m in run.get("members", []):
            d, ih, _ = A.load_table(m.get("table_id", ""))
            if not d or (d.get("metadata", {}) or {}).get("human_corrected"):
                continue
            if not (ih and os.path.exists(f"{IMG}/{ih}.jpg")):
                continue
            nf, nrows = C.page_fail_rows(d)
            if 1 <= nf <= 4 and (nrows == 0 or nf / nrows < 0.25):
                targets.append((ih, d, m.get("page")))
        if len(targets) >= N:
            break
    for i, (ih, d, page) in enumerate(targets[:N]):
        client, types = C.make_client(keys[i % len(keys)])
        txt = gemini_reason(client, types, reason_prompt(d), open(f"{IMG}/{ih}.jpg", "rb").read())
        print(f"\n{'='*70}\nPAGE p{page}  (failing rows: {C.page_fail_rows(d)[0]})")
        print("GEMINI REASONING:\n" + txt.strip()[:1400])
        cor = C.parse_json(txt)
        if cor:
            nd, ap = C.apply_fix(d, cor)
            old, new = C.page_fail_rows(d)[0], C.page_fail_rows(nd)[0]
            print(f"\n>>> deduced fix: {json.dumps(cor)}  ->  failing rows {old} -> {new}  "
                  + ("✅ FIXED" if new < old else "✗ not resolved"))
        else:
            print("\n>>> no JSON correction parsed")


if __name__ == "__main__":
    main()
