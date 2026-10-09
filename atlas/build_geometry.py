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

def main():
    frames = []
    for c, lvl in SRC:
        g = gpd.read_file(os.path.join(MAPD, f"gadm41_{c}_{lvl}.json"))
        g = g.rename(columns={f"NAME_{lvl}": "name", f"GID_{lvl}": "gid"})
        g["country"], g["admin1"] = c, g["NAME_1"]
        frames.append(g[["gid", "name", "admin1", "country", "geometry"]])
    g = gpd.pd.concat(frames, ignore_index=True)
    g["geometry"] = g.geometry.simplify(TOL, preserve_topology=True)
    minx, miny, maxx, maxy = g.total_bounds
    S = 1000 / ((maxx - minx) * K)
    def xy(x, y):
        return (x - minx) * K * S, (maxy - y) * S
    def ring(coords):
        pts = [xy(x, y) for x, y in coords]
        return "M" + "L".join(f"{a:.1f},{b:.1f}" for a, b in pts) + "Z"
    units = []
    for r in g.itertuples():
        geom = r.geometry
        if geom is None or geom.is_empty:
            continue
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        d = "".join(ring(p.exterior.coords) for p in polys if p.area > 1e-5) or ring(polys[0].exterior.coords)
        cx, cy = xy(geom.representative_point().x, geom.representative_point().y)
        units.append(dict(gid=r.gid, name=r.name, admin1=r.admin1, country=r.country,
                          d=d, cx=round(cx, 1), cy=round(cy, 1)))
    W, H = xy(maxx, miny)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    out = os.path.join(HERE, "out", "districts_svg.json")
    json.dump({"viewBox": f"0 0 {W:.0f} {H:.0f}", "units": units}, open(out, "w"), separators=(",", ":"))
    print(f"{len(units)} units, {os.path.getsize(out)/1e6:.2f} MB -> {out}")

if __name__ == "__main__":
    main()
