"""Wind-farm site layers for the terrain viewer: WTG coordinates, subsea-cable SHP folder and
buffer-zone geometry.

Everything below the "geometry" marker is pure numpy (no Qt / VTK) so it can be tested alone.
All geometry functions work in *scene units* (see terrain_viewer.py: 1 unit = one native raster
pixel); the viewer converts metres to scene units before calling them.

WTG coordinate file (WTG_coordinates.csv, next to terrain_viewer.py)
-------------------------------------------------------------------
    # crs: EPSG:3826            <- optional; default is TWD97 / TM2 zone 121 (EPSG:3826)
    name,easting,northing
    WTG-01,250123.4,2650123.4

Header names are flexible (name/id, e/east/easting/x, n/north/northing/y); without a header the
columns are name,E,N (or just E,N). Excel "CSV" saved in Big5/cp950 is fine too.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

DEFAULT_WTG_CRS = "EPSG:3826"

_NAME_KEYS = {"name", "id", "wtg", "turbine", "wtg_id", "風機", "編號", "名稱", "風機編號"}
_E_KEYS = {"e", "east", "easting", "x", "twd97_e", "e(m)", "東", "東距", "e座標"}
_N_KEYS = {"n", "north", "northing", "y", "twd97_n", "n(m)", "北", "北距", "n座標"}


# --------------------------------------------------------------------------- loaders
def _read_text(path) -> str:
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp950", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def parse_turbine_csv(text: str):
    """Return ([(name, x, y), ...], crs_string)."""
    crs = DEFAULT_WTG_CRS
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            body = s.lstrip("#").strip()
            if body.lower().startswith("crs"):
                rest = body[3:].lstrip()
                if rest[:1] in (":", "="):
                    rest = rest[1:]
                if rest.strip():
                    crs = rest.strip()
            continue
        rows.append(next(csv.reader([s])))
    if not rows:
        raise ValueError("風機座標檔是空的。")
    cells = [c.strip() for c in rows[0]]
    has_header = len(cells) >= 2 and not (_is_number(cells[-1]) and _is_number(cells[-2]))
    name_i, e_i, n_i = 0, -2, -1
    if has_header:
        low = [c.lower() for c in cells]
        e_found = next((i for i, c in enumerate(low) if c in _E_KEYS), None)
        n_found = next((i for i, c in enumerate(low) if c in _N_KEYS), None)
        name_found = next((i for i, c in enumerate(low) if c in _NAME_KEYS), None)
        if e_found is not None and n_found is not None:
            e_i, n_i = e_found, n_found
            name_i = name_found if name_found is not None else None
        rows = rows[1:]
    elif len(cells) == 2:
        name_i = None
    out = []
    for k, row in enumerate(rows, 1):
        row = [c.strip() for c in row]
        try:
            x, y = float(row[e_i]), float(row[n_i])
        except (ValueError, IndexError):
            raise ValueError(f"第 {k} 筆風機座標無法解析：{','.join(row)}")
        name = row[name_i] if name_i is not None and name_i < len(row) and row[name_i] else f"WTG-{k:02d}"
        out.append((name, x, y))
    if not out:
        raise ValueError("風機座標檔沒有資料列。")
    return out, crs


def load_turbines(path, target_crs: str) -> list[dict]:
    """Read the CSV and return [{'name', 'x', 'y'}] in `target_crs`."""
    from pyproj import CRS, Transformer

    rows, crs = parse_turbine_csv(_read_text(path))
    tr = Transformer.from_crs(CRS.from_user_input(crs), CRS.from_user_input(target_crs), always_xy=True)
    xs, ys = tr.transform([r[1] for r in rows], [r[2] for r in rows])
    return [{"name": r[0], "x": float(x), "y": float(y)} for r, x, y in zip(rows, xs, ys)]


def load_cables(folder, target_crs: str):
    """Read every .shp under `folder`. Returns (cables, errors, n_files).

    cables = [{'name': file stem, 'pts': (n, 2) array in target_crs}]
    """
    from pyproj import CRS, Transformer
    from scene_layers import read_vector_geometries

    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(str(folder))
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() == ".shp")
    target = CRS.from_user_input(target_crs)
    cables, errors = [], []
    for f in files:
        try:
            source_crs, geoms = read_vector_geometries(str(f))
            tr = Transformer.from_crs(source_crs, target, always_xy=True)
            for g in geoms:
                kind, coords = g.get("type"), g.get("coordinates", [])
                lines = [coords] if kind == "LineString" else coords if kind == "MultiLineString" else []
                for line in lines:
                    arr = np.asarray(line, dtype=float)
                    if arr.ndim != 2 or len(arr) < 2:
                        continue
                    x, y = tr.transform(arr[:, 0], arr[:, 1])
                    pts = np.column_stack((x, y))
                    pts = pts[np.isfinite(pts).all(axis=1)]
                    if len(pts) >= 2:
                        cables.append({"name": f.stem, "pts": pts})
        except Exception as exc:
            errors.append(f"{f.name}：{exc}")
    return cables, errors, len(files)


# --------------------------------------------------------------------------- geometry
def polyline_length(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    d = np.diff(pts, axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def densify(pts: np.ndarray, step: float) -> np.ndarray:
    """Insert points so no segment is longer than `step`; original vertices are kept."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return pts
    seg = np.diff(pts, axis=0)
    length = np.hypot(seg[:, 0], seg[:, 1])
    n = np.maximum(1, np.ceil(length / max(step, 1e-9)).astype(np.int64))
    idx = np.repeat(np.arange(len(seg)), n)
    first = np.repeat(np.cumsum(n) - n, n)
    k = np.arange(idx.size) - first + 1                     # 1..n within each segment
    new = pts[idx] + (k / n[idx])[:, None] * seg[idx]
    return np.vstack((pts[:1], new))


def clip_polyline_to_box(pts: np.ndarray, box) -> list[np.ndarray]:
    """Parts of a polyline inside box = (xmin, xmax, ymin, ymax). Vertices are preserved."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return []
    a, b = pts[:-1], pts[1:]
    d = b - a
    t0, t1 = np.zeros(len(a)), np.ones(len(a))
    ok = np.hypot(d[:, 0], d[:, 1]) > 1e-12
    for axis, lo, hi in ((0, box[0], box[1]), (1, box[2], box[3])):
        da, pa = d[:, axis], a[:, axis]
        par = np.abs(da) < 1e-12
        ok &= ~(par & ((pa < lo) | (pa > hi)))
        with np.errstate(divide="ignore", invalid="ignore"):
            ta, tb = (lo - pa) / da, (hi - pa) / da
        t0 = np.maximum(t0, np.where(par, -np.inf, np.minimum(ta, tb)))
        t1 = np.minimum(t1, np.where(par, np.inf, np.maximum(ta, tb)))
    ok &= t0 < t1 - 1e-12
    runs, cur, last = [], None, -2
    for i in np.flatnonzero(ok):
        p0, p1 = a[i] + t0[i] * d[i], a[i] + t1[i] * d[i]
        if cur is not None and last == i - 1 and t1[last] >= 1 - 1e-12 and t0[i] <= 1e-12:
            cur.append(p1)
        else:
            if cur is not None:
                runs.append(np.array(cur))
            cur = [p0, p1]
        last = i
    if cur is not None:
        runs.append(np.array(cur))
    return runs


def ribbon_quads(pts: np.ndarray, half_width: float, n_across: int = 5):
    """Quad strip of total width 2*half_width around a polyline, with mitered corners.

    Returns (xy (m*n_across, 2), quads (k, 4)). `n_across` columns allow the strip to follow
    terrain across its width when heights are sampled per vertex.
    """
    pts = np.asarray(pts, dtype=float)
    seg = np.diff(pts, axis=0)
    length = np.hypot(seg[:, 0], seg[:, 1])
    pts = pts[np.r_[True, length > 1e-9]]                   # drop duplicate points
    m = len(pts)
    if m < 2:
        return np.empty((0, 2)), np.empty((0, 4), dtype=np.int64)
    seg = np.diff(pts, axis=0)
    d = seg / np.hypot(seg[:, 0], seg[:, 1])[:, None]
    nrm = np.column_stack((-d[:, 1], d[:, 0]))              # left normals, (m-1, 2)
    vn = np.empty((m, 2))
    vn[0], vn[-1] = nrm[0], nrm[-1]
    if m > 2:
        s = nrm[:-1] + nrm[1:]
        sl = np.hypot(s[:, 0], s[:, 1])
        mdir = s / np.maximum(sl, 1e-12)[:, None]
        degenerate = sl < 1e-9
        mdir[degenerate] = nrm[1:][degenerate]
        vn[1:-1] = mdir / np.maximum(sl / 2, 1 / 3)[:, None]  # miter, capped at 3x half-width
    t = np.linspace(-1.0, 1.0, n_across)
    xy = pts[:, None, :] + (t[None, :, None] * half_width) * vn[:, None, :]
    idx = np.arange(m * n_across, dtype=np.int64).reshape(m, n_across)
    quads = np.column_stack((idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(),
                             idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()))
    return xy.reshape(-1, 2), quads


def disc_quads(cx: float, cy: float, radius: float, n_rings: int = 8, n_seg: int = 64):
    """Polar grid covering a disc (ring 0 is the degenerate centre)."""
    ang = np.linspace(0.0, 2 * np.pi, n_seg, endpoint=False)
    rad = np.linspace(0.0, radius, n_rings + 1)
    x = cx + rad[:, None] * np.cos(ang)[None, :]
    y = cy + rad[:, None] * np.sin(ang)[None, :]
    idx = np.arange((n_rings + 1) * n_seg, dtype=np.int64).reshape(n_rings + 1, n_seg)
    j1 = (np.arange(n_seg) + 1) % n_seg
    quads = np.stack((idx[:-1, :], idx[:-1, :][:, j1], idx[1:, :][:, j1], idx[1:, :]), axis=-1).reshape(-1, 4)
    return np.column_stack((x.ravel(), y.ravel())), quads


def bilinear_sample(z: np.ndarray, col, row) -> np.ndarray:
    """Bilinear lookup in grid `z` at fractional (col, row) *index* coordinates.

    NaN if outside the grid or if any of the four neighbours is NaN (same rule as the terrain
    mesh, which only keeps quads whose four corners are valid).
    """
    col = np.atleast_1d(np.asarray(col, dtype=float))
    row = np.atleast_1d(np.asarray(row, dtype=float))
    h, w = z.shape
    out = np.full(col.shape, np.nan, dtype=np.float32)
    ok = (col >= 0) & (col <= w - 1) & (row >= 0) & (row <= h - 1)
    if not ok.any():
        return out
    c, r = col[ok], row[ok]
    c0 = np.minimum(np.floor(c).astype(np.int64), max(w - 2, 0))
    r0 = np.minimum(np.floor(r).astype(np.int64), max(h - 2, 0))
    c1, r1 = np.minimum(c0 + 1, w - 1), np.minimum(r0 + 1, h - 1)
    fc, fr = c - c0, r - r0
    out[ok] = (z[r0, c0] * (1 - fc) * (1 - fr) + z[r0, c1] * fc * (1 - fr)
               + z[r1, c0] * (1 - fc) * fr + z[r1, c1] * fc * fr)
    return out


def compact_quads(xy: np.ndarray, z: np.ndarray, quads: np.ndarray):
    """Drop quads touching a NaN height and compact the points.

    Returns (points (n, 3) float32, faces flat int64 in VTK format) or (None, None).
    """
    good = np.isfinite(z)
    keep = good[quads].all(axis=1)
    q = quads[keep]
    if q.size == 0:
        return None, None
    used = np.unique(q)
    remap = np.full(len(z), -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    pts = np.column_stack((xy[used], z[used])).astype(np.float32)
    faces = np.empty((q.shape[0], 5), dtype=np.int64)
    faces[:, 0] = 4
    faces[:, 1:] = remap[q]
    return pts, faces.ravel()
