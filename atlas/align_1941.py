#!/usr/bin/env python3
"""Rubber-sheet the 1941 census-map polygons onto GADM before they are used to assign tehsils.

The 1941 digitization is displaced from true position by a median ~15 km in varying
directions. Control points are the 1941 units whose name matches a modern GADM district of
similar area (0.7-1.4x), i.e. districts whose extent barely changed; for each, the
displacement is GADM centroid minus 1941 centroid. Each 1941 polygon is then translated by
the inverse-distance-weighted displacement of its 8 nearest control points (leave-one-out
for the control units themselves when measuring error).

align(ref_ea, l2_ea) -> GeoSeries of shifted polygons; report() prints before/after error.
"""
import re, unicodedata, math
import numpy as np
from shapely.affinity import translate


def _nm(s):
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return re.sub(r"[^a-z]", "", re.sub(r"\b(district|state|the)\b", " ", s))


def control_points(ref, l2):
    g2 = {}
    for n, g in zip(l2["name"], l2.geometry):
        g2.setdefault(_nm(re.sub(r"([a-z])([A-Z])", r"\1 \2", n)), g)
    pts = []
    for i, r in enumerate(ref.itertuples()):
        k = _nm(r.Districts)
        g = g2.get(k)
        if g is None or r.geometry is None:
            continue
        a = r.geometry.area / g.area
        if 0.7 <= a <= 1.4:
            c1, c2 = r.geometry.centroid, g.centroid
            pts.append((i, c1.x, c1.y, c2.x - c1.x, c2.y - c1.y))
    return np.array(pts)


def field(pts, x, y, exclude=None, k=8):
    d = np.hypot(pts[:, 1] - x, pts[:, 2] - y)
    if exclude is not None:
        d = np.where(pts[:, 0] == exclude, np.inf, d)
    idx = np.argsort(d)[:k]
    w = 1.0 / np.maximum(d[idx], 5000.0) ** 2
    return float((pts[idx, 3] * w).sum() / w.sum()), float((pts[idx, 4] * w).sum() / w.sum())


def align(ref, l2):
    pts = control_points(ref, l2)
    out = []
    for i, g in enumerate(ref.geometry):
        if g is None or g.is_empty:
            out.append(g)
            continue
        c = g.centroid
        dx, dy = field(pts, c.x, c.y)
        out.append(translate(g, dx, dy))
    return out, pts


def report(ref, l2):
    pts = control_points(ref, l2)
    before = np.hypot(pts[:, 3], pts[:, 4])
    after = []
    for i, x, y, dx, dy in pts:
        fx, fy = field(pts, x, y, exclude=i)
        after.append(math.hypot(dx - fx, dy - fy))
    print(f"control points: {len(pts)}; displacement median {np.median(before)/1000:.1f} km -> "
          f"residual after alignment (leave-one-out) {np.median(after)/1000:.1f} km")
