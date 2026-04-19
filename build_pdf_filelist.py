"""Build a stable PDF filelist for the TOC-manifest workflow."""

import argparse
from pathlib import Path

from toc_job_utils import write_filelist


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a stable newline-delimited PDF filelist")
    parser.add_argument("root", help="Root directory containing PDFs")
    parser.add_argument("--output", required=True, help="Destination .txt file")
    args = parser.parse_args()

    count = write_filelist(Path(args.root), Path(args.output))
    print(f"Wrote {count} PDF paths to {args.output}")


if __name__ == "__main__":
    main()
