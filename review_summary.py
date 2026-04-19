"""Summarize human review annotations."""

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Summarize review annotations")
    parser.add_argument("--annotations", default="review_annotations",
                        help="Annotations directory")
    args = parser.parse_args()

    root = Path(args.annotations)
    files = sorted(root.rglob("*.review.json"))
    counts = Counter()

    print("REVIEW SUMMARY")
    print("=" * 70)
    print(f"annotations_root: {root}")
    print(f"files: {len(files)}")

    for path in files:
        with open(path) as f:
            data = json.load(f)
        status = data.get("status", "unreviewed")
        counts[status] += 1

    for status in ["verified", "corrected", "flagged", "unreviewed"]:
        if counts[status]:
            print(f"{status}: {counts[status]}")

    if files:
        print("\nRecent:")
        for path in files[-10:]:
            with open(path) as f:
                data = json.load(f)
            print(f"{data.get('status','?'):<10} {path}")


if __name__ == "__main__":
    main()
