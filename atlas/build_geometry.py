#!/usr/bin/env python3
"""Simplify GADM 4.1 modern districts (IND/PAK/BGD/MMR) into SVG paths.

Output atlas/out/districts_svg.json: {viewBox, units:[{gid,name,admin1,country,d,cx,cy}]}
Projection: equirectangular scaled by cos(22N) so shapes are not stretched.
"""
import json, math, os
import geopandas as gpd

HERE = os.path.dirname(os.path.abspath(__file__))
MAPD = os.path.join(HERE, "..", "mapdata")
SRC = [("IND", 2), ("PAK", 3), ("BGD", 2), ("MMR", 2)]
K = math.cos(math.radians(22))
TOL = 0.025  # degrees; ~2.5 km — keeps 900 units under ~1 MB

def load_l2():
    frames = []
    for c, lvl in SRC:
        g = gpd.read_file(os.path.join(MAPD, f"gadm41_{c}_{lvl}.json"))
        g = g.rename(columns={f"NAME_{lvl}": "name", f"GID_{lvl}": "gid"})
        g["country"], g["admin1"] = c, g["NAME_1"]
        frames.append(g[["gid", "name", "admin1", "country", "geometry"]])
    return gpd.pd.concat(frames, ignore_index=True)


def projection(g):
    """equirectangular x cos(22N), scaled to 1000 px wide; shared by every map layer"""
    minx, miny, maxx, maxy = g.total_bounds
    return dict(minx=minx, maxy=maxy, K=K, S=1000 / ((maxx - minx) * K), maxx=maxx, miny=miny)


def svg_path(geom, proj, min_area=1e-5):
    def xy(x, y):
        return (x - proj["minx"]) * proj["K"] * proj["S"], (proj["maxy"] - y) * proj["S"]
    def ring(coords):
        return "M" + "L".join(f"{a:.1f},{b:.1f}" for a, b in (xy(x, y) for x, y in coords)) + "Z"
    polys = [geom] if geom.geom_type == "Polygon" else [p for p in getattr(geom, "geoms", []) if p.geom_type == "Polygon"]
    if not polys:
        return "", (0, 0)
    d = "".join(ring(p.exterior.coords) for p in polys if p.area > min_area) or ring(max(polys, key=lambda p: p.area).exterior.coords)
    rp = geom.representative_point()
    return d, tuple(round(v, 1) for v in xy(rp.x, rp.y))


def main():
    g = load_l2()
    proj = projection(g)
    g["geometry"] = g.geometry.simplify(TOL, preserve_topology=True)
    units = []
    for r in g.itertuples():
        if r.geometry is None or r.geometry.is_empty:
            continue
        d, (cx, cy) = svg_path(r.geometry, proj)
        units.append(dict(gid=r.gid, name=r.name, admin1=r.admin1, country=r.country, d=d, cx=cx, cy=cy))
    W = (proj["maxx"] - proj["minx"]) * proj["K"] * proj["S"]
    H = (proj["maxy"] - proj["miny"]) * proj["S"]
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    out = os.path.join(HERE, "out", "districts_svg.json")
    json.dump({"viewBox": f"0 0 {W:.0f} {H:.0f}", "units": units, "proj": proj}, open(out, "w"), separators=(",", ":"))
    print(f"{len(units)} units, {os.path.getsize(out)/1e6:.2f} MB -> {out}")


if __name__ == "__main__":
    main()
