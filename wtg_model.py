"""Parametric monopile wind-turbine model (pure numpy) + PLY I/O + JSON configuration.

Model coordinates (metres): +X = East, +Y = North, +Z = up, origin on the tower axis at mean sea
level (MSL). The model contains the pile stub, transition piece (TP), boat landing, ladders, work
platform, tower, nacelle, hub and three blades, with per-vertex colours.

The viewer loads `WTG Models/WTG_<type>.ply`. If a file is missing it is generated from
`WTG Models/wtg_models.json`, and you can replace any PLY with your own model as long as it uses
the same conventions (metres, Z up, origin at MSL on the tower axis, one vertex colour per point).

Vertical references (see the JSON "_notes"):
  * hub_height_m is measured from MSL (sea level).
  * boat landing bottom = LAT = lat_below_msl_m below MSL.
  * sea_level_in_dem_m tells the viewer where MSL sits in the DEM's own vertical datum
    (2.92 if the DEM is referenced to LAT, 0 if it is referenced to MSL).
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

YELLOW = (248, 190, 20)         # TP / pile / foundation (all yellow)
WHITE = (238, 240, 243)
LIGHT_GREY = (205, 208, 212)
STEEL = (112, 120, 130)
BL_RED = (200, 40, 38)          # boat landing: contrasts with the yellow TP and the orange cable
LADDER_WHITE = (250, 250, 250)
CRANE_BLUE = (70, 110, 170)
MODEL_VERSION = "v2"

DEFAULT_CONFIG = {
    "_notes": [
        "高度皆以海平面 (MSL) 為 0。hub_height_m = 輪轂中心離海平面高度。",
        "sea_level_in_dem_m：海平面在 DEM 高程系統中的數值。DEM 以 MSL 為基準時 = 0（目前設定）；DEM 以 LAT 為基準時填 2.92（海平面 = LAT + 2.92 m）。",
        "lat_below_msl_m：LAT 在海平面下方的距離；Boat landing 最低點放在 LAT。",
        "rotor_yaw_deg / boat_landing_azimuth_deg：羅盤方位（北 = 0°，順時針）。所有風機共用。",
        "修改尺寸後，按檢視器的「重新產生風機模型」才會重做 PLY；你也可以直接用自己的 PLY 取代（公尺、Z 向上、原點在塔中心的 MSL）。",
    ],
    "sea_level_in_dem_m": 0.0,
    "lat_below_msl_m": 2.92,
    "rotor_yaw_deg": 45.0,
    "boat_landing_azimuth_deg": 135.0,
    "blade_azimuth_deg": 0.0,
    "default_type": "D6",
    "type_by_name": {"D21": "G4", "D28": "G4"},
    "types": {
        "D6": {
            "label": "Siemens Gamesa D6 (SWT-6.0-154)",
            "hub_height_m": 105.8, "rotor_diameter_m": 154.0, "hub_radius_m": 2.0,
            "blade_root_chord_m": 3.4, "blade_max_chord_m": 4.4,
            "nacelle_length_m": 13.0, "nacelle_width_m": 4.8, "nacelle_height_m": 4.6,
            "rotor_overhang_m": 6.0, "rotor_tilt_deg": 6.0,
            "tower_base_dia_m": 6.3, "tower_top_dia_m": 4.2,
            "tp_dia_m": 7.2, "pile_dia_m": 6.5, "tp_top_m": 20.0,
        },
        "G4": {
            "label": "Siemens Gamesa G4 (SWT-4.0-120)",
            "hub_height_m": 105.8, "rotor_diameter_m": 120.0, "hub_radius_m": 1.5,
            "blade_root_chord_m": 2.6, "blade_max_chord_m": 3.5,
            "nacelle_length_m": 12.0, "nacelle_width_m": 4.2, "nacelle_height_m": 4.2,
            "rotor_overhang_m": 5.2, "rotor_tilt_deg": 6.0,
            "tower_base_dia_m": 5.4, "tower_top_dia_m": 3.4,
            "tp_dia_m": 6.4, "pile_dia_m": 5.7, "tp_top_m": 20.0,
        },
    },
}


# --------------------------------------------------------------------------- mesh helpers
class Mesh:
    def __init__(self):
        self._v, self._f, self._c, self.n = [], [], [], 0

    def add(self, V, F, color):
        V = np.asarray(V, dtype=float)
        F = np.asarray(F, dtype=np.int64)
        if len(V) == 0 or len(F) == 0:
            return
        self._v.append(V)
        self._f.append(F + self.n)
        self._c.append(np.tile(np.asarray(color, dtype=np.uint8), (len(V), 1)))
        self.n += len(V)

    def arrays(self):
        return (np.vstack(self._v).astype(np.float32), np.vstack(self._f).astype(np.int32),
                np.vstack(self._c).astype(np.uint8))


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _perp_basis(axis):
    a = _unit(axis)
    h = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = _unit(np.cross(a, h))
    return a, u, np.cross(a, u)


def lathe(profile, n=32):
    """Solid of revolution about the Z axis from a (radius, z) profile (radius 0 closes the ends)."""
    prof = np.asarray(profile, dtype=float)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.column_stack((np.cos(t), np.sin(t)))
    V = np.concatenate([np.column_stack((r * ring, np.full(n, z))) for r, z in prof])
    i, j = np.arange(n), (np.arange(n) + 1) % n
    F = []
    for k in range(len(prof) - 1):
        a, b = k * n, (k + 1) * n
        F.append(np.column_stack((a + i, a + j, b + j)))
        F.append(np.column_stack((a + i, b + j, b + i)))
    return V, np.vstack(F)


def frustum(p0, p1, r0, r1, n=12, caps=True):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    _, u, v = _perp_basis(p1 - p0)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.cos(t)[:, None] * u + np.sin(t)[:, None] * v
    V = np.vstack((p0 + r0 * ring, p1 + r1 * ring))
    i, j = np.arange(n), (np.arange(n) + 1) % n
    F = [np.column_stack((i, j, n + j)), np.column_stack((i, n + j, n + i))]
    if caps:
        c0, c1 = len(V), len(V) + 1
        V = np.vstack((V, p0, p1))
        F += [np.column_stack((np.full(n, c0), j, i)), np.column_stack((np.full(n, c1), n + i, n + j))]
    return V, np.vstack(F)


def box(center, size, axes=None):
    """Axis-aligned (or rotated by the 3x3 `axes`, columns = local x/y/z) box."""
    sx, sy, sz = (np.asarray(size, float) / 2)
    corners = np.array([[x, y, z] for z in (-sz, sz) for y in (-sy, sy) for x in (-sx, sx)])
    if axes is not None:
        corners = corners @ np.asarray(axes, float).T
    V = corners + np.asarray(center, float)
    F = np.array([[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4],
                  [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]])
    return V, F


def _sgnpow(x, p):
    return np.sign(x) * np.abs(x) ** p


def superellipsoid(center, axes, radii, n1=2.0, n2=2.0, nu=28, nv=40):
    """Rounded box / ellipsoid; `axes` columns are the local x, y, z directions."""
    th = np.linspace(-np.pi / 2, np.pi / 2, nu)[:, None]
    ph = np.linspace(-np.pi, np.pi, nv, endpoint=False)[None, :]
    ct, st, cp, sp = np.cos(th), np.sin(th), np.cos(ph), np.sin(ph)
    x = radii[0] * _sgnpow(ct, 2 / n1) * _sgnpow(cp, 2 / n2)
    y = radii[1] * _sgnpow(ct, 2 / n1) * _sgnpow(sp, 2 / n2)
    z = radii[2] * _sgnpow(st, 2 / n1) * np.ones_like(cp)
    P = np.stack((x, y, z), axis=-1).reshape(-1, 3) @ np.asarray(axes, float).T + np.asarray(center, float)
    idx = np.arange(nu * nv).reshape(nu, nv)
    j1 = (np.arange(nv) + 1) % nv
    a, b, c, d = idx[:-1, :], idx[:-1, :][:, j1], idx[1:, :][:, j1], idx[1:, :]
    F = np.vstack((np.stack((a, b, c), -1).reshape(-1, 3), np.stack((a, c, d), -1).reshape(-1, 3)))
    return P, F


def _compass(az_deg):
    a = math.radians(az_deg)
    radial = np.array([math.sin(a), math.cos(a), 0.0])      # East, North
    tangent = np.array([math.cos(a), -math.sin(a), 0.0])
    return radial, tangent


# --------------------------------------------------------------------------- parts
def _ladder(m, radial, tangent, r_surface, z0, z1, color=LADDER_WHITE, spacing=0.3):
    rail_r = r_surface + 0.28
    for off in (-0.28, 0.28):
        c = radial * rail_r + tangent * off
        m.add(*box((c[0], c[1], (z0 + z1) / 2), (0.07, 0.07, z1 - z0)), color)
    for z in np.arange(z0 + 0.2, z1 - 0.1, spacing):
        c = radial * rail_r
        axes = np.column_stack((tangent, radial, [0, 0, 1]))
        m.add(*box((c[0], c[1], z), (0.56, 0.04, 0.04), axes), color)


def _railing_ring(m, z, radius, color=YELLOW, posts=28, tube=0.03, segs=84):
    for k in range(posts):
        a = 2 * np.pi * k / posts
        m.add(*box((radius * np.cos(a), radius * np.sin(a), z + 0.55), (0.06, 0.06, 1.1)), color)
    for zr in (z + 0.55, z + 1.1):
        pts = [(radius * np.cos(a), radius * np.sin(a), zr) for a in np.linspace(0, 2 * np.pi, segs + 1)]
        for p, q in zip(pts[:-1], pts[1:]):
            m.add(*frustum(p, q, tube, tube, n=4, caps=False), color)


def _boat_landing(m, radial, tangent, r_tp, z_bottom, z_platform, tp_top):
    r_post = r_tp + 1.5
    z_top = z_platform + 2.0
    for off in (-2.1, 2.1):
        c = radial * r_post + tangent * off
        m.add(*frustum((c[0], c[1], z_bottom), (c[0], c[1], z_top), 0.42, 0.42, n=14), BL_RED)
        for z in np.arange(z_bottom + 0.6, z_top, 2.6):            # struts back to the TP
            s = radial * (r_tp - 0.05) + tangent * off
            m.add(*frustum((s[0], s[1], z), (c[0], c[1], z), 0.11, 0.11, n=8, caps=False), BL_RED)
    for z in np.arange(z_bottom + 0.6, z_top, 2.6):                # cross beams between the posts
        a = radial * r_post + tangent * -2.1
        b = radial * r_post + tangent * 2.1
        m.add(*frustum((a[0], a[1], z), (b[0], b[1], z), 0.09, 0.09, n=8, caps=False), BL_RED)
    c = radial * (r_tp + 0.85)                                     # landing platform + rails
    axes = np.column_stack((tangent, radial, [0, 0, 1]))
    m.add(*box((c[0], c[1], z_platform), (4.4, 1.6, 0.1), axes), BL_RED)
    for zr in (z_platform + 0.55, z_platform + 1.1):
        o = radial * (r_post + 0.1)
        m.add(*box((o[0], o[1], zr), (4.4, 0.05, 0.05), axes), BL_RED)
    _ladder(m, radial, tangent, r_tp, z_bottom + 0.3, tp_top)      # main ladder up to the work platform


def _blade(m, hub, a, up, azimuth_deg, r_root, radius, root_chord, max_chord, n_st=44, nx=18):
    side = np.cross(a, up)
    phi = math.radians(azimuth_deg)
    rdir = _unit(up * math.cos(phi) + side * math.sin(phi))
    ddir = _unit(np.cross(rdir, a))                                # rotation direction (clockwise from the front)
    span = radius - r_root
    s = np.linspace(0.0, 1.0, n_st)
    chord = np.interp(s, [0, 0.04, 0.12, 0.30, 0.60, 0.85, 1.0],
                      [root_chord, root_chord * 1.03, max_chord, max_chord * 0.84, max_chord * 0.57,
                       max_chord * 0.34, 0.25])
    thick = np.interp(s, [0, 0.06, 0.15, 0.30, 0.60, 1.0], [1.0, 0.55, 0.34, 0.25, 0.18, 0.10])
    twist = np.radians(np.interp(s, [0, 0.1, 0.3, 0.6, 1.0], [18, 12, 6, 2.5, 0.0]))
    w = np.clip(1.0 - s / 0.07, 0.0, 1.0) ** 1.5                    # blend into a round root
    beta = np.linspace(0, np.pi, nx + 1)
    xu, xl = 0.5 * (1 + np.cos(beta)), 0.5 * (1 - np.cos(beta))     # upper TE->LE, lower LE->TE
    x = np.concatenate((xu, xl[1:-1]))
    surf = np.concatenate((np.ones(len(xu)), -np.ones(len(xl) - 2)))
    N = len(x)
    ang = 2 * np.pi * np.arange(N) / N
    rings = []
    for k in range(n_st):
        yt = 5 * thick[k] * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
        px, py = x, surf * yt
        cx, cy = 0.3 + 0.5 * np.cos(ang), 0.5 * np.sin(ang)
        px, py = (1 - w[k]) * px + w[k] * cx, (1 - w[k]) * py + w[k] * cy
        c_dir = _unit(-ddir * math.cos(twist[k]) - a * math.sin(twist[k]))   # leading edge -> trailing edge
        n_up = np.cross(c_dir, rdir)                                          # suction side (downwind)
        centre = hub + rdir * (r_root + span * s[k])
        rings.append(centre + np.outer((px - 0.3) * chord[k], c_dir) + np.outer(py * chord[k], n_up))
    V = np.vstack(rings)
    i, j = np.arange(N), (np.arange(N) + 1) % N
    F = []
    for k in range(n_st - 1):
        A, B = k * N, (k + 1) * N
        F.append(np.column_stack((A + i, A + j, B + j)))
        F.append(np.column_stack((A + i, B + j, B + i)))
    m.add(V, np.vstack(F), WHITE)


# --------------------------------------------------------------------------- full turbine
def build_wtg_parts(spec: dict, cfg: dict):
    """Build one turbine as (static, rotor); each is (vertices float32, faces int32, colours uint8)."""
    m, mr = Mesh(), Mesh()
    lat = float(cfg["lat_below_msl_m"])
    r_tp, r_pile = spec["tp_dia_m"] / 2, spec["pile_dia_m"] / 2
    z_tp = float(spec["tp_top_m"])
    hub_h = float(spec["hub_height_m"])
    r_tw0, r_tw1 = spec["tower_base_dia_m"] / 2, spec["tower_top_dia_m"] / 2

    # pile stub + TP (grout flare at the bottom)
    m.add(*lathe([(0, -8), (r_pile, -8), (r_pile, -5), (r_tp + 0.8, -0.5), (r_tp, 2.0), (r_tp, z_tp), (0, z_tp)], n=48),
          YELLOW)

    # work platform with railing
    r_deck = r_tp + 1.7
    m.add(*lathe([(r_tw0, z_tp - 0.12), (r_deck, z_tp - 0.12), (r_deck, z_tp), (r_tw0, z_tp)], n=64), YELLOW)
    _railing_ring(m, z_tp, r_deck - 0.08)

    # boat landing + a second ladder + davit crane
    bl_dir, bl_tan = _compass(cfg["boat_landing_azimuth_deg"])
    _boat_landing(m, bl_dir, bl_tan, r_tp, -lat, 6.5, z_tp)
    ld_dir, ld_tan = _compass(cfg["boat_landing_azimuth_deg"] + 165.0)
    _ladder(m, ld_dir, ld_tan, r_tp, 1.5, z_tp)
    cr_dir, _ = _compass(cfg["boat_landing_azimuth_deg"] + 90.0)
    base = cr_dir * (r_tp + 0.9)
    top = (base[0], base[1], z_tp + 3.2)
    m.add(*frustum((base[0], base[1], z_tp), top, 0.16, 0.14, n=10), CRANE_BLUE)
    tip = cr_dir * (r_tp + 5.2)
    m.add(*frustum(top, (tip[0], tip[1], z_tp + 4.4), 0.11, 0.07, n=8), CRANE_BLUE)

    # tower
    tower_top = hub_h - 2.9
    m.add(*lathe([(0, z_tp), (r_tw0, z_tp), (r_tw1, tower_top), (0, tower_top)], n=48), WHITE)

    # nacelle, hub, blades
    yaw = math.radians(cfg["rotor_yaw_deg"])
    f = np.array([math.sin(yaw), math.cos(yaw), 0.0])                # rotor faces this compass direction
    tilt = math.radians(spec["rotor_tilt_deg"])
    z_hat = np.array([0.0, 0.0, 1.0])
    a = f * math.cos(tilt) + z_hat * math.sin(tilt)                  # rotor axis (tilted up at the front)
    up = z_hat * math.cos(tilt) - f * math.sin(tilt)
    right = np.cross(a, up)
    overhang = spec["rotor_overhang_m"]
    hub = f * overhang + np.array([0.0, 0.0, hub_h])
    nl, nw, nh = spec["nacelle_length_m"], spec["nacelle_width_m"], spec["nacelle_height_m"]
    n_centre = f * (overhang - 1.2 - nl / 2) + np.array([0.0, 0.0, hub_h - 0.2])
    m.add(*superellipsoid(n_centre, np.column_stack((a, right, up)), (nl / 2, nw / 2, nh / 2), 3.0, 3.0), WHITE)
    rh = spec["hub_radius_m"]
    mr.add(*superellipsoid(hub + a * (rh * 0.35), np.column_stack((a, right, up)), (rh * 1.45, rh * 1.05, rh * 1.05), 2.0, 2.0,
                          nu=24, nv=32), WHITE)
    for k in range(3):
        _blade(mr, hub, a, up, cfg.get("blade_azimuth_deg", 0.0) + 120.0 * k, rh * 0.9, spec["rotor_diameter_m"] / 2,
               spec["blade_root_chord_m"], spec["blade_max_chord_m"])
    return m.arrays(), mr.arrays()


def build_wtg(spec: dict, cfg: dict):
    """Whole turbine in one mesh (static + rotor)."""
    (v1, f1, c1), (v2, f2, c2) = build_wtg_parts(spec, cfg)
    return np.vstack((v1, v2)), np.vstack((f1, f2 + len(v1))).astype(np.int32), np.vstack((c1, c2))


def rotor_pivot(spec: dict, cfg: dict):
    """(hub centre, unit rotor axis pointing downwind-to-front) in model coordinates (metres)."""
    yaw, tilt = math.radians(cfg["rotor_yaw_deg"]), math.radians(spec["rotor_tilt_deg"])
    f = np.array([math.sin(yaw), math.cos(yaw), 0.0])
    a = f * math.cos(tilt) + np.array([0.0, 0.0, 1.0]) * math.sin(tilt)
    return f * spec["rotor_overhang_m"] + np.array([0.0, 0.0, float(spec["hub_height_m"])]), a


# --------------------------------------------------------------------------- PLY I/O
def write_ply(path, V, F, C):
    V, F, C = np.asarray(V, np.float32), np.asarray(F, np.int32), np.asarray(C, np.uint8)
    header = (f"ply\nformat binary_little_endian 1.0\ncomment wtg_model_generated {MODEL_VERSION} (m, Z up, origin at MSL)\n"
              f"element vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n")
    vd = np.empty(len(V), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    vd["p"], vd["c"] = V, C
    fd = np.empty(len(F), dtype=[("n", "u1"), ("i", "<i4", 3)])
    fd["n"], fd["i"] = 3, F
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(vd.tobytes())
        fh.write(fd.tobytes())


_PLY_TYPES = {"char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2", "int": "<i4", "uint": "<u4",
              "float": "<f4", "double": "<f8", "int8": "i1", "uint8": "u1", "int16": "<i2", "uint16": "<u2",
              "int32": "<i4", "uint32": "<u4", "float32": "<f4", "float64": "<f8"}


def read_ply(path):
    """Read a triangle PLY (binary little endian or ascii) -> (V float32, F int32, C uint8 or None).

    Raises ValueError for layouts it does not understand (the viewer then falls back to pyvista).
    """
    raw = Path(path).read_bytes()
    end = raw.find(b"end_header")
    if end < 0:
        raise ValueError("not a PLY file")
    nl = raw.find(b"\n", end) + 1
    lines = raw[:nl].decode("ascii", "replace").splitlines()
    fmt = next((l.split()[1] for l in lines if l.startswith("format")), "")
    elements, cur = [], None
    for l in lines:
        t = l.split()
        if t[:1] == ["element"]:
            cur = {"name": t[1], "count": int(t[2]), "props": []}
            elements.append(cur)
        elif t[:1] == ["property"] and cur is not None:
            cur["props"].append(t[1:])
    vert = next(e for e in elements if e["name"] == "vertex")
    face = next((e for e in elements if e["name"] == "face"), None)
    body = raw[nl:]
    names = [p[-1] for p in vert["props"]]
    if fmt == "ascii":
        rows = body.decode("ascii").split("\n")
        v = np.array([r.split() for r in rows[:vert["count"]]], dtype=float)
        V = v[:, [names.index("x"), names.index("y"), names.index("z")]].astype(np.float32)
        C = (v[:, [names.index("red"), names.index("green"), names.index("blue")]].astype(np.uint8)
             if "red" in names else None)
        F = np.array([[int(x) for x in r.split()[1:4]] for r in rows[vert["count"]:vert["count"] + face["count"]]
                      if r.split()[0] == "3"], dtype=np.int32)
        return V, F, C
    if fmt != "binary_little_endian":
        raise ValueError(f"unsupported PLY format {fmt}")
    if any(p[0] == "list" for p in vert["props"]):
        raise ValueError("list property in vertex element")
    vdt = np.dtype([(p[1], _PLY_TYPES[p[0]]) for p in vert["props"]])
    v = np.frombuffer(body, dtype=vdt, count=vert["count"])
    V = np.column_stack((v["x"], v["y"], v["z"])).astype(np.float32)
    C = np.column_stack((v["red"], v["green"], v["blue"])).astype(np.uint8) if "red" in names else None
    off = vdt.itemsize * vert["count"]
    if face is None:
        return V, np.empty((0, 3), np.int32), C
    prop = face["props"][0]
    if prop[0] != "list" or len(face["props"]) != 1:
        raise ValueError("unsupported face layout")
    cnt_t, idx_t = _PLY_TYPES[prop[1]], _PLY_TYPES[prop[2]]
    n_vert = np.frombuffer(body, dtype=cnt_t, count=1, offset=off)[0]
    fdt = np.dtype([("n", cnt_t), ("i", idx_t, int(n_vert))])
    f = np.frombuffer(body, dtype=fdt, count=face["count"], offset=off)
    if not (f["n"] == n_vert).all():
        raise ValueError("mixed face sizes")
    idx = f["i"].astype(np.int32)
    if n_vert == 4:                                         # split quads
        idx = np.vstack((idx[:, [0, 1, 2]], idx[:, [0, 2, 3]]))
    elif n_vert != 3:
        raise ValueError("only triangles or quads are supported")
    return V, idx, C


# --------------------------------------------------------------------------- configuration
def load_config(folder) -> dict:
    """Read `wtg_models.json` (written with defaults if it does not exist)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "wtg_models.json"
    if not path.exists():
        path.write_text(json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")
        return copy.deepcopy(DEFAULT_CONFIG)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key, value in DEFAULT_CONFIG.items():              # fill in keys added by newer versions
        cfg.setdefault(key, copy.deepcopy(value))
    return cfg


def turbine_type(name: str, cfg: dict) -> str:
    return cfg["type_by_name"].get(name, cfg["default_type"])


def rotor_path(path) -> Path:
    path = Path(path)
    return path.with_name(path.stem + "_rotor.ply")


def _is_stale_generated(path: Path) -> bool:
    """True for a PLY this module wrote in an older version (user-supplied models are never touched)."""
    try:
        head = path.read_bytes()[:400]
    except OSError:
        return False
    if b"wtg_model_generated" in head:
        return f"wtg_model_generated {MODEL_VERSION}".encode() not in head
    return b"comment wind turbine model" in head                       # v1 files


def ensure_models(folder, cfg: dict, force: bool = False) -> dict:
    """Make sure `WTG_<type>.ply` (+ `_rotor.ply`) exist for every type. -> {type: static path}"""
    folder = Path(folder)
    out = {}
    for tname, spec in cfg["types"].items():
        path = folder / f"WTG_{tname}.ply"
        rpath = rotor_path(path)
        if force or not path.exists() or _is_stale_generated(path):
            static, rotor = build_wtg_parts(spec, cfg)
            write_ply(path, *static)
            write_ply(rpath, *rotor)
        out[tname] = path
    return out
