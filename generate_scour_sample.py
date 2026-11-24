"""Generate continuous, corridor-shaped offshore terrain survey test data."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer

OUT = Path(__file__).parent
N, CELL = 1600, 5.0                 # 8 km x 8 km, 5 m pixels
CRS = "EPSG:32651"
EASTING0, NORTHING0 = 500_000.0, 2_600_000.0
HALF = N * CELL / 2
X = EASTING0 - HALF + (np.arange(N) + 0.5) * CELL
Y = NORTHING0 + HALF - (np.arange(N) + 0.5) * CELL
XX, YY = np.meshgrid(X, Y)
XLOC, YLOC = XX - EASTING0, YY - NORTHING0
TRANSFORM = from_origin(EASTING0 - HALF, NORTHING0 + HALF, CELL, CELL)
NODATA = -9999.0

# Ten turbines in two strings. Coordinates are local metres from the site centre.
TURBINES = [(x, y, f"WTG-{i:02d}")
            for i, (x, y) in enumerate(
                [(x, y) for y in (650, -650) for x in (-1600, -800, 0, 800, 1600)], 1)]
OSS = (2550.0, 0.0)

# The survey covers circles within 50 m of every turbine and continuous 25 m
# corridors on either side of each cable centerline. Everything else is nodata.
CABLES = []
for row_y, row in [(650.0, TURBINES[:5]), (-650.0, TURBINES[5:])]:
    row_sorted = sorted(row, key=lambda t: t[0])
    for turbine in row_sorted:
        tx, ty, name = turbine
        CABLES.append((f"{name} spur", [(tx, ty), (tx, row_y)]))
    CABLES.append((f"Array string y={row_y:g}", [(row_sorted[0][0], row_y), (row_sorted[-1][0], row_y)]))
# Join the two strings and connect the array to the offshore substation/export route.
CABLES.extend([
    ("Inter-array link", [(1600, 650), (1900, 650), (1900, -650), (1600, -650)]),
    ("Export cable", [(1900, -650), (2200, -650), OSS, (3800, 0)]),
])


def distance_to_polyline(x, y, points):
    best = np.full(x.shape, np.inf, dtype=np.float32)
    for a, b in zip(points[:-1], points[1:]):
        ax, ay = a[0], a[1]
        vx, vy = b[0]-ax, b[1]-ay
        t = np.clip(((x-ax)*vx + (y-ay)*vy) / max(vx*vx + vy*vy, 1e-12), 0, 1)
        dist = np.hypot(x-(ax+t*vx), y-(ay+t*vy))
        np.minimum(best, dist, out=best)
    return best


valid = np.zeros((N, N), dtype=bool)
for tx, ty, _ in TURBINES:
    valid |= np.hypot(XLOC-tx, YLOC-ty) <= 50.0
for _, points in CABLES:
    valid |= distance_to_polyline(XLOC, YLOC, points) <= 25.0

# Synthetic seabed depth spans ~5-35 m and varies smoothly across the field.
base = (20.0 + 12.5 * XLOC / HALF + 2.5 * YLOC / HALF
        + 0.45*np.sin(XLOC/510)*np.cos(YLOC/680)
        + 0.18*np.sin((XLOC+YLOC)/125))
period1 = base.copy()
for tx, ty, _ in TURBINES:
    radius = np.hypot(XLOC-tx, YLOC-ty)
    period1 += -0.75*np.exp(-(radius/18)**2) + 0.16*np.exp(-((radius-34)/8)**2)

# Period 2 has intentionally varied local erosion/deposition at scour aprons
# plus small along-cable changes, so the difference layer is easy to inspect.
period2 = period1.copy()
for i, (tx, ty, _) in enumerate(TURBINES):
    radius = np.hypot(XLOC-tx, YLOC-ty)
    sign = -1 if i % 2 == 0 else 1
    period2 += sign*(0.42+0.04*i)*np.exp(-(radius/24)**2)
    period2 += -sign*0.20*np.exp(-((radius-39)/7)**2)
for i, (_, pts) in enumerate(CABLES):
    d = distance_to_polyline(XLOC, YLOC, pts)
    period2 += (0.035 if i % 2 else -0.035)*np.exp(-(d/16)**2)


def write_tif(path, values, description):
    raster = np.where(valid, values, NODATA).astype(np.float32)
    with rasterio.open(path, "w", driver="GTiff", width=N, height=N, count=1,
                       dtype="float32", crs=CRS, transform=TRANSFORM,
                       nodata=NODATA, compress="deflate", predictor=3,
                       tiled=True, blockxsize=256, blockysize=256) as dst:
        dst.write(raster, 1)
        dst.set_band_description(1, description)


write_tif(OUT/"sample_scour_protection.tif", period1, "Synthetic seabed depth period 1 (m)")
write_tif(OUT/"sample_scour_protection_period2.tif", period2, "Synthetic seabed depth period 2 (m)")

# Export vector layout as standard WGS84 GeoJSON for interoperability.
to_wgs84 = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
def geo(pt):
    return list(to_wgs84.transform(EASTING0+pt[0], NORTHING0+pt[1]))

features = []
for x, y, name in TURBINES:
    features.append({"type":"Feature", "properties":{"name":name,"layer":"turbine"},
                     "geometry":{"type":"Point","coordinates":geo((x,y))}})
features.append({"type":"Feature", "properties":{"name":"OSS-01","layer":"substation"},
                 "geometry":{"type":"Point","coordinates":geo(OSS)}})
for name, pts in CABLES:
    features.append({"type":"Feature", "properties":{"name":name,"layer":"cable","width_m":50},
                     "geometry":{"type":"LineString","coordinates":[geo(pt) for pt in pts]}})
layout = OUT/"sample_windfarm_layout.geojson"
layout.write_text(json.dumps({"type":"FeatureCollection","name":"Synthetic 10-turbine wind farm layout",
                              "features":features}, indent=2), encoding="utf-8")

# Overview preview: valid survey corridors are shown against the full site extent.
fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
im = ax.imshow(np.where(valid, period1, np.nan), extent=[X.min()-EASTING0, X.max()-EASTING0,
                Y.min()-NORTHING0, Y.max()-NORTHING0], origin="upper", cmap="terrain")
for x, y, name in TURBINES:
    ax.plot(x, y, "k^", ms=5)
ax.set_title("Synthetic continuous survey coverage: 10 turbines and cable corridors")
ax.set_xlabel("Local Easting (m)"); ax.set_ylabel("Local Northing (m)"); ax.set_aspect("equal")
fig.colorbar(im, ax=ax, shrink=.78, label="Synthetic seabed depth (m)")
fig.tight_layout(); fig.savefig(OUT/"sample_windfarm_coverage_preview.png", bbox_inches="tight"); plt.close(fig)

diff = np.where(valid, period2-period1, np.nan)
limit = float(np.nanpercentile(np.abs(diff[valid]), 98))
fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
im = ax.imshow(diff, extent=[X.min()-EASTING0, X.max()-EASTING0,
                Y.min()-NORTHING0, Y.max()-NORTHING0], origin="upper",
                cmap="RdBu_r", vmin=-limit, vmax=limit)
ax.set_title("Synthetic seabed change: period 2 − period 1")
ax.set_xlabel("Local Easting (m)"); ax.set_ylabel("Local Northing (m)"); ax.set_aspect("equal")
fig.colorbar(im, ax=ax, shrink=.78, label="Depth change (m)")
fig.tight_layout(); fig.savefig(OUT/"sample_scour_change_preview.png", bbox_inches="tight"); plt.close(fig)

print(f"Wrote two aligned {N}x{N} GeoTIFFs; {valid.mean():.1%} valid survey coverage")
print(f"Wrote 10 turbines, {len(CABLES)} cable segments/routes, and {layout.name}")
