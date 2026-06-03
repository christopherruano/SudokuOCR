"""Assemble a local research bundle from synced recovery/export artifacts.

The bundle is a reproducible local snapshot of:
  - inference exports
  - rescue outputs
  - never-OCR queue planning
  - recovery status summaries
  - optional audit/guide/manifests

Outputs:
  - bundle_manifest.json
  - bundle_manifest.csv
  - README.md
  - copied artifact directories/files
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path


def ensure_copy(src: Path, dst: Path, mode: str = "copy"):
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        if mode == "copy":
            shutil.copytree(src, dst)
        elif mode == "hardlink":
            shutil.copytree(src, dst, copy_function=os.link)
        elif mode == "symlink":
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(src.resolve(), dst, target_is_directory=True)
        else:
            raise ValueError(f"Unsupported materialization mode: {mode}")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if mode == "copy":
            shutil.copy2(src, dst)
        elif mode == "hardlink":
            os.link(src, dst)
        elif mode == "symlink":
            os.symlink(src.resolve(), dst)
        else:
            raise ValueError(f"Unsupported materialization mode: {mode}")


def file_digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_csv_rows(path: Path):
    try:
        with path.open(encoding="utf-8", newline="") as fh:
            return max(sum(1 for _ in fh) - 1, 0)
    except Exception:
        return ""


def iter_files(root: Path):
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def build_manifest(bundle_root: Path, manifest_mode: str = "full"):
    rows = []
    for path in iter_files(bundle_root):
        if path.name in {"bundle_manifest.json", "bundle_manifest.csv"}:
            continue
        rel = path.relative_to(bundle_root).as_posix()
        if path.is_symlink():
            target = path.resolve()
            rows.append({
                "relative_path": rel,
                "bytes": target.stat().st_size if target.exists() and target.is_file() else "",
                "sha256": file_digest(target) if (manifest_mode == "full" and target.exists() and target.is_file()) else "",
                "csv_rows": count_csv_rows(target) if (manifest_mode == "full" and target.exists() and target.suffix.lower() == ".csv") else "",
            })
            continue
        rows.append({
            "relative_path": rel,
            "bytes": path.stat().st_size,
            "sha256": file_digest(path) if manifest_mode == "full" else "",
            "csv_rows": count_csv_rows(path) if (manifest_mode == "full" and path.suffix.lower() == ".csv") else "",
        })
    return rows


def write_readme(path: Path, copied_items, manifest_rows):
    lines = []
    lines.append("# Local Research Bundle")
    lines.append("")
    lines.append("This bundle is a local snapshot of the current best OCR/inference corpus and recovery planning artifacts.")
    lines.append("")
    lines.append("## Included Artifacts")
    for label, src, dst in copied_items:
        lines.append(f"- `{label}`: `{dst}`")
        lines.append(f"  source: `{src}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("- `export_inference_final_core_curated_v1` is the current best curated main inference corpus when present.")
    lines.append("- `export_inference_plus_core_rescue_v5` is the pre-curation merged core baseline.")
    lines.append("- `export_inference_tail_rescue_unreviewed_v1` is a separate exploratory tail layer and should not be mixed into the main paper dataset.")
    lines.append("- `assembled_final_core_curated_v1` is the multi-page assembly layer built from the curated core export.")
    lines.append("- `assembled_final_core_curated_v1_improved` is the stricter subset where cross-page assembly improves constraint consistency.")
    lines.append("- `non_age_rescue_v5` contains the current rescue split: `core_rescue`, `core_review`, `tail_rescue`.")
    lines.append("- `never_ocrd_queues` contains the next OCR queues once API budget returns.")
    lines.append("- `bundle_manifest.json` and `bundle_manifest.csv` provide checksums and row counts for reproducibility.")
    lines.append("")
    lines.append(f"## File Count")
    lines.append(f"- files tracked: {len(manifest_rows)}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build a local research bundle from synced artifacts")
    parser.add_argument("--inference-dir", action="append", default=[],
                        help="Inference export directory to include (can be passed multiple times)")
    parser.add_argument("--assembled-dir", action="append", default=[],
                        help="Assembled multi-page output directory to include (can be passed multiple times)")
    parser.add_argument("--recovery-status-dir", default="")
    parser.add_argument("--non-age-rescue-dir", default="")
    parser.add_argument("--never-ocrd-queues-dir", default="")
    parser.add_argument("--age-recall-audit-dir", default="")
    parser.add_argument("--organized-manifest", default="")
    parser.add_argument("--data-guide", default="")
    parser.add_argument("--gap-fillability-guide", default="")
    parser.add_argument(
        "--materialization",
        choices=["copy", "hardlink", "symlink"],
        default="copy",
        help="How to place artifacts in the bundle. Use hardlink to avoid duplicating large local data.",
    )
    parser.add_argument(
        "--manifest-mode",
        choices=["full", "shallow"],
        default="full",
        help="Use shallow to skip SHA256 hashing and CSV row counts for very large bundles.",
    )
    parser.add_argument("--output-dir", default="export/local_research_bundle")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_items = []

    def maybe_copy(label, raw_path):
        if not raw_path:
            return
        src = Path(raw_path).expanduser()
        if not src.exists():
            return
        dst = output_dir / src.name
        ensure_copy(src, dst, mode=args.materialization)
        copied_items.append((label, str(src), dst.relative_to(output_dir).as_posix()))

    for inference_dir in args.inference_dir:
        maybe_copy("inference_dir", inference_dir)
    for assembled_dir in args.assembled_dir:
        maybe_copy("assembled_dir", assembled_dir)

    maybe_copy("recovery_status", args.recovery_status_dir)
    maybe_copy("non_age_rescue", args.non_age_rescue_dir)
    maybe_copy("never_ocrd_queues", args.never_ocrd_queues_dir)
    maybe_copy("age_recall_audit", args.age_recall_audit_dir)
    maybe_copy("organized_manifest", args.organized_manifest)
    maybe_copy("data_guide", args.data_guide)
    maybe_copy("gap_fillability_guide", args.gap_fillability_guide)

    manifest_rows = build_manifest(output_dir, manifest_mode=args.manifest_mode)

    (output_dir / "bundle_manifest.json").write_text(
        json.dumps({"files": manifest_rows}, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "bundle_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["relative_path", "bytes", "sha256", "csv_rows"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    write_readme(output_dir / "README.md", copied_items, manifest_rows)

    print(f"Wrote bundle to: {output_dir}")
    print(f"Tracked files:   {len(manifest_rows)}")


if __name__ == "__main__":
    main()
