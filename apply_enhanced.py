#!/usr/bin/env python3
"""Merge enhanced-image re-OCR results back over the originals, but only when the
enhanced read is NOT worse (>= column groups and >= rows). Original is backed up as
<imghash>_oneshot.preenh.json (keep old copies). The downstream U+M+W=Population gate
in validate_tableVII is the final arbiter of whether the enhanced read is accepted."""
import os, re, json, glob, shutil

V1 = "export_final/sendout_final/sendout_v1"
ENH = "/tmp/enh_results"
HEX = re.compile(r"([0-9a-f]{40})")


def shape(path):
    try:
        d = json.load(open(path))
    except Exception:
        return (0, 0)
    cg = len(d.get("metadata", {}).get("column_groups", []))
    rows = sum(len(s.get("rows", [])) for s in d.get("data", []))
    return (cg, rows)


def find_orig(ih):
    for root, _, files in os.walk("export/ocr_results_referenced/digitized_tables"):
        if f"{ih}_oneshot.json" in files:
            return os.path.join(root, f"{ih}_oneshot.json")
    return None


def main():
    enh = glob.glob(f"{ENH}/**/*_oneshot.json", recursive=True)
    replaced = kept = no_orig = 0
    for ep in enh:
        ih = HEX.search(os.path.basename(ep))
        if not ih:
            continue
        ih = ih.group(1)
        op = find_orig(ih)
        if not op:
            no_orig += 1; continue
        ec, er = shape(ep)
        oc, orr = shape(op)
        # accept enhanced only if not structurally worse (gate decides correctness later)
        if (ec, er) >= (oc, orr) and er >= 4:
            if not os.path.exists(op.replace("_oneshot.json", "_oneshot.preenh.json")):
                shutil.copy(op, op.replace("_oneshot.json", "_oneshot.preenh.json"))
            # rewrite source_image to the original (not the enhanced temp path) for provenance
            d = json.load(open(ep))
            d["extraction_method"] = "oneshot_enhanced"
            json.dump(d, open(op, "w"), indent=1)
            replaced += 1
        else:
            kept += 1
    print(f"enhanced reads merged: replaced {replaced} | kept original (enhanced not better) {kept} | no original {no_orig}")
    print("Next: bash reingest.sh -> U+M+W gate validates the enhanced reads, then build_verdicts.")


if __name__ == "__main__":
    main()
