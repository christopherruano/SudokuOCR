"""
Automated table row detection and cropping for historical census table images.

Uses PIL/Pillow and numpy only (no OpenCV dependency).

Designed for scanned tables from the Indian Census (1872-1941) that have
horizontal ruled lines separating rows.  Works on both clean PNGs and
noisier scanned documents.

Typical workflow
----------------
>>> from image_processing import crop_rows, crop_vertical_sections, crop_region
>>> rows = crop_rows("age_tables/Hyderabad/Hyderabad_state_summary_age_1901.png",
...                   output_dir="output/rows")
>>> sections = crop_vertical_sections(
...     "age_tables/Hyderabad/Hyderabad_state_summary_age_1901.png",
...     num_sections=4, output_dir="output/sections")
>>> region = crop_region(
...     "age_tables/Hyderabad/Hyderabad_state_summary_age_1901.png",
...     top_frac=0.1, bottom_frac=0.5, left_frac=0.25, right_frac=0.75)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

# Pixels darker than this (0-255 scale) count as "ink".  Works well for
# both black-on-white and slightly faded scans.
_DEFAULT_DARK_THRESHOLD = 128

# A scanline counts as part of a horizontal rule when at least this
# fraction of its pixels are dark.
_DEFAULT_LINE_RATIO = 0.30

# After finding individual dark scanlines we cluster them; any gap smaller
# than this many pixels is considered part of the same rule.
_DEFAULT_CLUSTER_GAP = 6

# Minimum height (px) for a detected row strip to be kept.  Avoids
# emitting tiny slivers caused by double-rules.
_DEFAULT_MIN_ROW_HEIGHT = 15

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_grayscale(image_path: Union[str, Path]) -> Tuple[Image.Image, np.ndarray]:
    """Open an image and return (original PIL Image, grayscale numpy array)."""
    img = Image.open(image_path).convert("RGB")
    gray = np.array(img.convert("L"))  # shape (H, W), uint8
    return img, gray


def _dark_row_mask(
    gray: np.ndarray,
    dark_threshold: int = _DEFAULT_DARK_THRESHOLD,
    min_line_length_ratio: float = _DEFAULT_LINE_RATIO,
) -> np.ndarray:
    """Return a boolean 1-D array of length H.

    ``mask[y]`` is True when scanline *y* has enough dark pixels to look
    like a horizontal rule.
    """
    height, width = gray.shape
    # Count dark pixels per scanline
    dark_counts = np.sum(gray < dark_threshold, axis=1)  # shape (H,)
    min_dark = int(width * min_line_length_ratio)
    return dark_counts >= min_dark


def _cluster_line_positions(
    mask: np.ndarray,
    cluster_gap: int = _DEFAULT_CLUSTER_GAP,
) -> List[Tuple[int, int]]:
    """Group contiguous (or nearly contiguous) True runs into clusters.

    Returns a list of (start_y, end_y) pairs, each describing one
    horizontal rule.
    """
    # Find indices where the mask is True
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return []

    clusters: List[Tuple[int, int]] = []
    start = int(indices[0])
    prev = start
    for idx in indices[1:]:
        idx = int(idx)
        if idx - prev > cluster_gap:
            clusters.append((start, prev))
            start = idx
        prev = idx
    clusters.append((start, prev))
    return clusters


def _cluster_midpoints(clusters: List[Tuple[int, int]]) -> List[int]:
    """Return the midpoint y-coordinate for each cluster."""
    return [(s + e) // 2 for s, e in clusters]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_row_boundaries(
    image_path: Union[str, Path],
    min_line_length_ratio: float = 0.30,
    dark_threshold: int = _DEFAULT_DARK_THRESHOLD,
    cluster_gap: int = _DEFAULT_CLUSTER_GAP,
) -> List[int]:
    """Detect horizontal line boundaries in a table image.

    Parameters
    ----------
    image_path : str or Path
        Path to a PNG / JPEG table image.
    min_line_length_ratio : float
        Minimum fraction of the image width that must be dark for a
        scanline to count as part of a horizontal rule (default 0.30).
    dark_threshold : int
        Grayscale value (0-255) below which a pixel is considered dark.
    cluster_gap : int
        Maximum vertical gap (px) between dark scanlines that should
        still be grouped into a single rule.

    Returns
    -------
    list[int]
        Sorted y-coordinates (midpoints of each detected horizontal rule).
    """
    _, gray = _load_grayscale(image_path)
    mask = _dark_row_mask(gray, dark_threshold, min_line_length_ratio)
    clusters = _cluster_line_positions(mask, cluster_gap)
    return _cluster_midpoints(clusters)


def crop_rows(
    image_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    padding: int = 5,
    min_line_length_ratio: float = 0.30,
    dark_threshold: int = _DEFAULT_DARK_THRESHOLD,
    cluster_gap: int = _DEFAULT_CLUSTER_GAP,
    min_row_height: int = _DEFAULT_MIN_ROW_HEIGHT,
) -> List[Union[Image.Image, str]]:
    """Crop a table image into individual row strips.

    Detects horizontal rules, then extracts the region between each
    consecutive pair of rules as a separate image.

    Parameters
    ----------
    image_path : str or Path
        Path to a PNG / JPEG table image.
    output_dir : str, Path, or None
        If provided, each strip is saved as a PNG in this directory and
        the function returns a list of file paths.  Otherwise the
        function returns a list of PIL Image objects.
    padding : int
        Pixels to add above and below each row strip (clipped to image
        bounds).  Positive values include a sliver of the ruled line for
        visual context; negative values crop into the row content.
    min_line_length_ratio, dark_threshold, cluster_gap
        Forwarded to :func:`detect_row_boundaries`.
    min_row_height : int
        Strips shorter than this (px) are discarded.

    Returns
    -------
    list[Image.Image] or list[str]
        Row-strip images (or saved file paths when *output_dir* is set).
    """
    img, gray = _load_grayscale(image_path)
    height, width = gray.shape

    mask = _dark_row_mask(gray, dark_threshold, min_line_length_ratio)
    clusters = _cluster_line_positions(mask, cluster_gap)

    if not clusters:
        # No horizontal lines detected -- return the whole image.
        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            dest = out / f"row_000.png"
            img.save(str(dest))
            return [str(dest)]
        return [img]

    # Build cut points: top edge of each rule cluster.
    # We slice *between* the rules, so the edges of each strip are at the
    # cluster boundaries (start of next rule = end of current row).
    cut_tops = [s for s, _ in clusters]
    cut_bots = [e for _, e in clusters]

    # Each row strip goes from (end of rule N) to (start of rule N+1).
    strips: List[Tuple[int, int]] = []

    # Region above the first rule (header area, if any)
    if cut_tops[0] > min_row_height:
        strips.append((0, cut_tops[0]))

    for i in range(len(clusters) - 1):
        y_top = cut_bots[i] + 1
        y_bot = cut_tops[i + 1]
        strips.append((y_top, y_bot))

    # Region below the last rule
    if height - cut_bots[-1] > min_row_height:
        strips.append((cut_bots[-1] + 1, height))

    # Apply padding and filter tiny strips
    results: List[Union[Image.Image, str]] = []
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

    stem = Path(image_path).stem
    for idx, (y_top, y_bot) in enumerate(strips):
        y_top_padded = max(0, y_top - padding)
        y_bot_padded = min(height, y_bot + padding)
        if y_bot_padded - y_top_padded < min_row_height:
            continue
        cropped = img.crop((0, y_top_padded, width, y_bot_padded))

        if output_dir is not None:
            dest = out / f"{stem}_row_{idx:03d}.png"
            cropped.save(str(dest))
            results.append(str(dest))
        else:
            results.append(cropped)

    return results


def crop_vertical_sections(
    image_path: Union[str, Path],
    num_sections: int = 4,
    section_boundaries: Optional[List[float]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> List[Union[Image.Image, str]]:
    """Split an image vertically into N column sections.

    Useful for separating the major column groups in wide census tables
    (e.g. Population | Unmarried | Married | Widowed).

    Parameters
    ----------
    image_path : str or Path
        Path to a PNG / JPEG table image.
    num_sections : int
        Number of equal-width vertical slices (used when
        *section_boundaries* is None).
    section_boundaries : list[float] or None
        Explicit fractional boundaries, e.g. ``[0, 0.15, 0.40, 0.65, 1.0]``
        defines four sections.  Fractions are relative to image width.
        When provided, *num_sections* is ignored.
    output_dir : str, Path, or None
        If provided, each section is saved as a PNG and the function
        returns file paths; otherwise PIL Image objects are returned.

    Returns
    -------
    list[Image.Image] or list[str]
        Vertical section images (or saved file paths).
    """
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    if section_boundaries is not None:
        bounds = section_boundaries
    else:
        bounds = [i / num_sections for i in range(num_sections + 1)]

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

    stem = Path(image_path).stem
    results: List[Union[Image.Image, str]] = []

    for idx in range(len(bounds) - 1):
        x_left = int(round(bounds[idx] * width))
        x_right = int(round(bounds[idx + 1] * width))
        cropped = img.crop((x_left, 0, x_right, height))

        if output_dir is not None:
            dest = out / f"{stem}_sec_{idx:03d}.png"
            cropped.save(str(dest))
            results.append(str(dest))
        else:
            results.append(cropped)

    return results


def crop_region(
    image_path: Union[str, Path],
    top_frac: float,
    bottom_frac: float,
    left_frac: float = 0.0,
    right_frac: float = 1.0,
) -> Image.Image:
    """Crop a fractional region of an image.

    Parameters
    ----------
    image_path : str or Path
        Path to a PNG / JPEG image.
    top_frac, bottom_frac : float
        Vertical bounds as fractions of image height (0 = top, 1 = bottom).
    left_frac, right_frac : float
        Horizontal bounds as fractions of image width (0 = left, 1 = right).

    Returns
    -------
    PIL.Image.Image
        The cropped region.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    box = (
        int(round(left_frac * w)),
        int(round(top_frac * h)),
        int(round(right_frac * w)),
        int(round(bottom_frac * h)),
    )
    return img.crop(box)


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

def _cli():
    """Minimal command-line interface for quick experiments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect and crop rows/columns in scanned census table images."
    )
    sub = parser.add_subparsers(dest="command")

    # --- detect ---
    p_det = sub.add_parser("detect", help="Print detected row boundary y-coordinates.")
    p_det.add_argument("image", help="Path to table image")
    p_det.add_argument("--ratio", type=float, default=0.30,
                       help="Min dark-pixel ratio for a scanline to count as a rule")
    p_det.add_argument("--threshold", type=int, default=_DEFAULT_DARK_THRESHOLD,
                       help="Grayscale darkness threshold (0-255)")

    # --- rows ---
    p_row = sub.add_parser("rows", help="Crop image into row strips.")
    p_row.add_argument("image", help="Path to table image")
    p_row.add_argument("-o", "--output", default="cropped_rows",
                       help="Output directory")
    p_row.add_argument("--padding", type=int, default=5)
    p_row.add_argument("--ratio", type=float, default=0.30)
    p_row.add_argument("--threshold", type=int, default=_DEFAULT_DARK_THRESHOLD)

    # --- sections ---
    p_sec = sub.add_parser("sections", help="Split image into vertical sections.")
    p_sec.add_argument("image", help="Path to table image")
    p_sec.add_argument("-n", "--num", type=int, default=4,
                       help="Number of equal vertical sections")
    p_sec.add_argument("-o", "--output", default="cropped_sections",
                       help="Output directory")

    # --- region ---
    p_reg = sub.add_parser("region", help="Crop a fractional region of the image.")
    p_reg.add_argument("image", help="Path to table image")
    p_reg.add_argument("top", type=float, help="Top fraction (0-1)")
    p_reg.add_argument("bottom", type=float, help="Bottom fraction (0-1)")
    p_reg.add_argument("--left", type=float, default=0.0)
    p_reg.add_argument("--right", type=float, default=1.0)
    p_reg.add_argument("-o", "--output", default="cropped_region.png",
                       help="Output file path")

    args = parser.parse_args()

    if args.command == "detect":
        boundaries = detect_row_boundaries(
            args.image,
            min_line_length_ratio=args.ratio,
            dark_threshold=args.threshold,
        )
        print(f"Detected {len(boundaries)} horizontal rules at y-coordinates:")
        for y in boundaries:
            print(f"  y = {y}")

    elif args.command == "rows":
        paths = crop_rows(
            args.image,
            output_dir=args.output,
            padding=args.padding,
            min_line_length_ratio=args.ratio,
            dark_threshold=args.threshold,
        )
        print(f"Saved {len(paths)} row strips to {args.output}/")
        for p in paths:
            print(f"  {p}")

    elif args.command == "sections":
        paths = crop_vertical_sections(
            args.image,
            num_sections=args.num,
            output_dir=args.output,
        )
        print(f"Saved {len(paths)} vertical sections to {args.output}/")
        for p in paths:
            print(f"  {p}")

    elif args.command == "region":
        region = crop_region(args.image, args.top, args.bottom,
                             left_frac=args.left, right_frac=args.right)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        region.save(args.output)
        print(f"Saved region to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
