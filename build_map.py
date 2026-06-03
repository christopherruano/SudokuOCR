#!/usr/bin/env python3
"""Build a real (geographic) SVG map of British-India-era regions for the explorer.

Dissolves modern India states into the project's broad_regions, adds Pakistan (=>
Northwest Frontier) and Myanmar (=> Burma) from a world layer, plus neighbour
outlines as grey context. Projects lon/lat to an SVG viewBox and writes compact
path strings to mapdata/regions_svg.json (consumed by /map).
"""
import json, math
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon

STATE_REGION = {
    "Andaman and Nicobar": "Andaman & Nicobar",
    "Andhra Pradesh": "South India", "Telangana": "South India", "Karnataka": "South India",
    "Kerala": "South India", "Tamil Nadu": "South India", "Puducherry": "South India",
    "Arunachal Pradesh": "Northeast India", "Assam": "Northeast India", "Manipur": "Northeast India",
    "Meghalaya": "Northeast India", "Mizoram": "Northeast India", "Nagaland": "Northeast India",
    "Sikkim": "Northeast India", "Tripura": "Northeast India",
    "Bihar": "East India", "Jharkhand": "East India", "Orissa": "East India", "Odisha": "East India",
    "West Bengal": "East India",
    "Chandigarh": "North India", "Delhi": "North India", "Haryana": "North India",
    "Himachal Pradesh": "North India", "Jammu and Kashmir": "North India", "Punjab": "North India",
    "Uttar Pradesh": "North India", "Uttaranchal": "North India", "Uttarakhand": "North India",
    "Rajasthan": "Rajputana",
    "Dadra and Nagar Haveli": "West India", "Daman and Diu": "West India", "Goa": "West India",
    "Gujarat": "West India", "Maharashtra": "West India",
    "Chhattisgarh": "Central India", "Madhya Pradesh": "Central India",
}
NEIGHBOUR_REGION = {"Pakistan": "Northwest Frontier", "Myanmar": "Burma"}
CONTEXT = {"Bangladesh", "Nepal", "Sri Lanka", "Afghanistan", "Bhutan", "China", "Iran"}

LON0, LON1, LAT0, LAT1 = 60.0, 101.5, 5.0, 38.5
SCALE = 1000.0 / (LON1 - LON0)
H = (LAT1 - LAT0) * SCALE


def project(lon, lat):
    return (lon - LON0) * SCALE, (LAT1 - lat) * SCALE


def geom_to_path(geom, tol=0.04):
    geom = geom.simplify(tol, preserve_topology=True)
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    out = []
    for poly in polys:
        if not isinstance(poly, Polygon) or poly.is_empty:
            continue
        if poly.area < 0.05:   # drop tiny slivers/islands for a clean shape
            continue
        xy = [project(x, y) for x, y in poly.exterior.coords]
        out.append("M" + " ".join(f"{x:.1f},{y:.1f}" for x, y in xy) + "Z")
    return "".join(out)


def name_field(gdf):
    for k in ("ADMIN", "name", "NAME", "SOVEREIGNT", "NAME_EN"):
        if k in gdf.columns:
            return k
    raise SystemExit("no country-name field in world layer")


def main():
    regions = {}            # region -> {path, cx, cy}
    context = []

    states = gpd.read_file("mapdata/india_states.geojson")
    states["region"] = states["NAME_1"].map(STATE_REGION)
    miss = sorted(set(states.loc[states["region"].isna(), "NAME_1"]))
    if miss:
        print("unmapped states (-> dropped):", miss)
    dissolved = states.dropna(subset=["region"]).dissolve(by="region")
    for region, row in dissolved.iterrows():
        c = row.geometry.representative_point()
        cx, cy = project(c.x, c.y)
        regions[region] = {"path": geom_to_path(row.geometry), "cx": round(cx, 1), "cy": round(cy, 1)}

    world = gpd.read_file("mapdata/world.geojson")
    nf = name_field(world)
    for _, row in world.iterrows():
        nm = row[nf]
        if nm in NEIGHBOUR_REGION:
            region = NEIGHBOUR_REGION[nm]
            c = row.geometry.representative_point()
            cx, cy = project(c.x, c.y)
            regions[region] = {"path": geom_to_path(row.geometry, 0.06), "cx": round(cx, 1), "cy": round(cy, 1)}
        elif nm in CONTEXT:
            p = geom_to_path(row.geometry, 0.08)
            if p:
                context.append(p)

    out = {"viewBox": f"0 0 1000 {H:.0f}", "regions": regions, "context": context}
    json.dump(out, open("mapdata/regions_svg.json", "w"))
    print(f"regions mapped: {sorted(regions)}")
    print(f"context shapes: {len(context)}  viewBox: {out['viewBox']}")
    print("wrote mapdata/regions_svg.json", round(len(json.dumps(out)) / 1024), "KB")


if __name__ == "__main__":
    main()
