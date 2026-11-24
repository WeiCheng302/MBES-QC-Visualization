"""Georeferenced scene-layer adapters for the offshore wind farm viewer.

Extension points: add photogrammetry/GCP adapters or richer turbine meshes here.
Vector inputs must carry a CRS (.prj for shapefiles; GeoJSON defaults to EPSG:4326).
"""
from pathlib import Path
import json

import numpy as np
import pyvista as pv
from pyproj import CRS, Transformer


def read_vector_geometries(path: str):
    """Return line and point coordinates in source CRS from GeoJSON or Shapefile."""
    p = Path(path)
    if p.suffix.lower() in {".json", ".geojson"}:
        doc = json.loads(p.read_text(encoding="utf-8-sig"))
        source_crs = CRS.from_user_input(doc.get("crs", {}).get("properties", {}).get("name", "EPSG:4326"))
        features = doc.get("features", [])
        geoms = [f.get("geometry") for f in features if f.get("geometry")]
        return source_crs, geoms
    if p.suffix.lower() == ".shp":
        import shapefile
        prj = p.with_suffix(".prj")
        if not prj.exists():
            raise ValueError("Shapefile 缺少 .prj 座標系資訊，請補上 .prj 或轉成 GeoJSON。")
        source_crs = CRS.from_wkt(prj.read_text(encoding="utf-8"))
        reader = shapefile.Reader(str(p))
        geoms = []
        for shape in reader.shapes():
            pts = [list(xy) for xy in shape.points]
            if shape.shapeType in (1, 8, 11, 18, 21, 28):
                geoms.extend({"type": "Point", "coordinates": pt} for pt in pts)
            else:
                starts = list(shape.parts) + [len(pts)]
                geoms.extend({"type": "LineString", "coordinates": pts[starts[i]:starts[i+1]]}
                              for i in range(len(starts)-1) if starts[i+1]-starts[i] > 1)
        return source_crs, geoms
    raise ValueError("支援 GeoJSON 或含 .prj 的 Shapefile。")


def make_vector_actor(path: str, target_crs: str, origin_xy: tuple[float, float], cell_size: float,
                      z_scene: float, color: str, radius: float = 0.7):
    """Create a scene-space actor; all input XY are transformed into terrain CRS."""
    source_crs, geometries = read_vector_geometries(path)
    transformer = Transformer.from_crs(source_crs, CRS.from_user_input(target_crs), always_xy=True)
    actors = []
    for geom in geometries:
        kind, coords = geom.get("type"), geom.get("coordinates", [])
        lines = [coords] if kind == "LineString" else coords if kind == "MultiLineString" else []
        for line in lines:
            if len(line) < 2:
                continue
            x, y = transformer.transform([p[0] for p in line], [p[1] for p in line])
            points = np.column_stack(((np.asarray(x)-origin_xy[0])/cell_size,
                                      (np.asarray(y)-origin_xy[1])/cell_size,
                                      np.full(len(x), z_scene)))
            poly = pv.PolyData(points)
            poly.lines = np.r_[len(points), np.arange(len(points))]
            actors.append((poly, color, 3))
        points_list = [coords] if kind == "Point" else coords if kind == "MultiPoint" else []
        for pt in points_list:
            x, y = transformer.transform(pt[0], pt[1])
            center = ((x-origin_xy[0])/cell_size, (y-origin_xy[1])/cell_size, z_scene)
            actors.append((pv.Sphere(radius=radius, center=center), color, None))
    return actors
