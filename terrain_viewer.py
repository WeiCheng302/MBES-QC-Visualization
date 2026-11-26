"""Interactive 3D viewer for GeoTIFF terrain rasters (tuned for multi-GB DEMs).

Design notes
------------
* Surface = quad PolyData built from valid pixels only (C-order point ids, so no
  dependence on VTK structured-grid ordering).
* Scene coordinates are fixed per file: X/Y = (easting/northing - raster origin) / native
  cell size, Z = (elevation - zref) / native cell size * exaggeration. They do not change
  when the sampling window or resolution changes, so the camera stays valid while the
  mesh is refined.
* A pyramid (average-downsampled GeoTIFF levels, built once into ~/.terrain_viewer_cache)
  makes coarse overview reads instant. Source TIFFs are never modified.
* Level of detail: when the camera settles, the visible ground window is re-sampled at up to
  `顯示解析度` cells, so zooming in reaches the native resolution.
* Never replace the windowed/out_shape reads with src.read(1): a multi-GB DEM must only be
  read at preview resolution.
"""

from __future__ import annotations

import hashlib
import time
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window, transform as window_transform
from PySide6 import QtCore, QtWidgets
from pyvistaqt import QtInteractor
import pyvista as pv
from scene_layers import make_vector_actor
import site_layers as sl
import wtg_model as wm

FOOTPRINT_PX = 2048                  # coarse size used to find the surveyed footprint
PYRAMID_MIN_PIXELS = 25_000_000      # build a pyramid automatically above this size ...
PYRAMID_MIN_BYTES = 200 * 1024**2    # ... or this file size
PYRAMID_STEP = 4                     # each pyramid level is 4x smaller than the previous one
LEVEL_NODATA = -9999.0
CACHE_ROOT = Path.home() / ".terrain_viewer_cache"
BASE_DIR = Path(__file__).resolve().parent
WTG_CSV = BASE_DIR / "WTG_coordinates.csv"          # name,easting,northing (TWD97 by default)
MODEL_DIR = BASE_DIR / "WTG Models"                 # WTG_<type>.ply + wtg_models.json (auto-generated)
CABLE_DIR = BASE_DIR / "Subsea Cable SHP"           # every .shp in here is loaded as a cable layer


# --------------------------------------------------------------------------- helpers
def build_quad_faces(mask: np.ndarray):
    """Quads for every 2x2 block of valid pixels.

    Returns (used_point_ids, faces) or (None, None) when nothing is valid. Point ids are
    C-order raster indices (row * width + col); `faces` is the flat VTK array
    [4, a, b, c, d, 4, ...] re-indexed to the compacted `used_point_ids` list.
    """
    h, w = mask.shape
    ok = mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]
    rows, cols = np.nonzero(ok)
    if rows.size == 0:
        return None, None
    p00 = rows.astype(np.int64) * w + cols
    quads = np.column_stack((p00, p00 + w, p00 + w + 1, p00 + 1))  # CCW for a north-up raster
    used = np.zeros(h * w, dtype=bool)
    used[quads.ravel()] = True
    remap = np.cumsum(used, dtype=np.int64) - 1
    quads = remap[quads]
    faces = np.empty((quads.shape[0], 5), dtype=np.int64)
    faces[:, 0] = 4
    faces[:, 1:] = quads
    return np.flatnonzero(used), faces.ravel()


def block_mean(z: np.ndarray, valid: np.ndarray, f: int) -> np.ndarray:
    """Nodata-aware f x f block average (edges are padded with invalid pixels)."""
    h, w = z.shape
    ph, pw = -h % f, -w % f
    if ph or pw:
        z = np.pad(z, ((0, ph), (0, pw)))
        valid = np.pad(valid, ((0, ph), (0, pw)))
    hh, ww = z.shape[0] // f, z.shape[1] // f
    zr = z.reshape(hh, f, ww, f)
    vr = valid.reshape(hh, f, ww, f)
    cnt = vr.sum(axis=(1, 3))
    tot = np.where(vr, zr, 0).sum(axis=(1, 3), dtype=np.float64)
    out = np.full((hh, ww), LEVEL_NODATA, dtype=np.float32)
    m = cnt > 0
    out[m] = (tot[m] / cnt[m]).astype(np.float32)
    return out


def ground_hits(matrix: np.ndarray, zplane: float, n: int = 5) -> np.ndarray:
    """Intersect an n x n grid of screen rays with the plane z = zplane.

    `matrix` is the camera's composite projection matrix (world -> clip). Returns (k, 2)
    world XY hits that lie between the near and far planes.
    """
    try:
        inv = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.empty((0, 2))
    hits = []
    for sx in np.linspace(-1, 1, n):
        for sy in np.linspace(-1, 1, n):
            p0 = inv @ np.array([sx, sy, -1.0, 1.0])
            p1 = inv @ np.array([sx, sy, 1.0, 1.0])
            p0, p1 = p0[:3] / p0[3], p1[:3] / p1[3]
            dz = p1[2] - p0[2]
            if abs(dz) < 1e-12:
                continue
            t = (zplane - p0[2]) / dz
            if 0.0 <= t <= 1.0:
                hits.append((p0 + t * (p1 - p0))[:2])
    return np.array(hits).reshape(-1, 2)


def needs_pyramid(path: str) -> bool:
    try:
        with rasterio.open(path) as src:
            pixels = src.width * src.height
        return pixels >= PYRAMID_MIN_PIXELS or Path(path).stat().st_size >= PYRAMID_MIN_BYTES
    except Exception:
        return False


class Pyramid:
    """Average-downsampled copies of a GeoTIFF stored in the user cache."""

    def __init__(self, directory: Path, levels: list[tuple[int, str]]):
        self.directory = directory
        self.levels = levels  # [(factor, path)] ascending factor

    @staticmethod
    def cache_dir(path: str) -> Path:
        p = Path(path)
        st = p.stat()
        tag = hashlib.sha1(f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]
        return CACHE_ROOT / f"{p.stem}_{tag}"

    @classmethod
    def load(cls, path: str):
        try:
            d = cls.cache_dir(path)
            meta = d / "levels.json"
            if not meta.exists():
                return None
            info = json.loads(meta.read_text(encoding="utf-8"))
            levels = [(int(e["factor"]), str(d / e["file"])) for e in info["levels"]]
            if not all(Path(p).exists() for _, p in levels):
                return None
            return cls(d, sorted(levels))
        except Exception:
            return None

    def pick(self, scale: float):
        """Coarsest level that is still no coarser than `scale` native pixels per output pixel."""
        best = (1, None)
        for f, p in self.levels:
            if f <= scale:
                best = (f, p)
        return best


class _Cancelled(Exception):
    pass


class PyramidWorker(QtCore.QThread):
    """Builds the pyramid in a background thread, reporting progress."""

    progress = QtCore.Signal(int, str)
    finished_with = QtCore.Signal(str)  # "" = ok, "cancelled", or an error text

    def __init__(self, path: str, out_dir: Path, parent=None):
        super().__init__(parent)
        self.path = path
        self.out_dir = out_dir
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            self._build()
        except _Cancelled:
            shutil.rmtree(self.out_dir, ignore_errors=True)
            self.finished_with.emit("cancelled")
            return
        except Exception as exc:
            shutil.rmtree(self.out_dir, ignore_errors=True)
            self.finished_with.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished_with.emit("")

    def _build(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with rasterio.open(self.path) as src:
            width, height = src.width, src.height
        factors, f = [], PYRAMID_STEP
        while len(factors) < 4 and max(width, height) // f >= 400:
            factors.append(f)
            f *= PYRAMID_STEP
        stages, sw, sh = [], width, height
        for _ in factors:
            stages.append((sw, sh))
            sw, sh = -(-sw // PYRAMID_STEP), -(-sh // PYRAMID_STEP)
        total = float(sum(w * h for w, h in stages)) or 1.0
        done, src_path, entries = 0.0, self.path, []
        for fac in factors:
            dst_path = self.out_dir / f"level_{fac}.tif"
            done = self._stage(src_path, dst_path, done, total, fac)
            entries.append({"factor": fac, "file": dst_path.name})
            src_path = str(dst_path)
        st = Path(self.path).stat()
        (self.out_dir / "levels.json").write_text(json.dumps(
            {"source": str(Path(self.path).resolve()), "size": st.st_size, "levels": entries}),
            encoding="utf-8")
        self.progress.emit(100, "完成")

    def _stage(self, src_path, dst_path, done, total, level_factor):
        step = PYRAMID_STEP
        with rasterio.open(src_path) as src:
            sw, sh = src.width, src.height
            ow, oh = -(-sw // step), -(-sh // step)
            profile = dict(driver="GTiff", width=ow, height=oh, count=1, dtype="float32",
                           crs=src.crs, transform=src.transform * rasterio.Affine.scale(step),
                           nodata=LEVEL_NODATA, tiled=True, blockxsize=256, blockysize=256,
                           compress="deflate", predictor=3, BIGTIFF="IF_SAFER")
            strip = 512  # source rows per pass; multiple of `step`
            with rasterio.open(dst_path, "w", **profile) as dst:
                for r in range(0, sh, strip):
                    if self._cancel:
                        raise _Cancelled()
                    rows = min(strip, sh - r)
                    blk = src.read(1, window=Window(0, r, sw, rows), masked=True)
                    z = np.ma.filled(blk.astype(np.float32), 0.0)
                    valid = (~np.ma.getmaskarray(blk)) & np.isfinite(z)
                    z = np.where(valid, z, 0.0).astype(np.float32)
                    small = block_mean(z, valid, step)
                    dst.write(small, 1, window=Window(0, r // step, ow, small.shape[0]))
                    done += rows * sw
                    self.progress.emit(min(99, int(100 * done / total)),
                                       f"正在建立金字塔（縮小 {level_factor} 倍層）…")
        return done


# --------------------------------------------------------------------------- viewer
class TerrainViewer(QtWidgets.QMainWindow):
    def __init__(self, initial_file: str | None = None):
        super().__init__()
        self.setWindowTitle("離岸風場地形 3D 檢視器")
        self.resize(1420, 900)
        self.setAcceptDrops(True)
        self.raster: dict | None = None
        self.actor = None
        self.mesh = None
        self.comparison_path: str | None = None
        self.comparison_aligned = False
        self.comparison_enabled = False
        self.vector_layers: list[dict] = []
        self.vector_actors = []
        self.scene_meta = None
        self.clim: dict[str, tuple[float, float] | None] = {"elev": None, "delta": None}
        self.pyramids: dict[str, Pyramid] = {}
        self.view_window: tuple[int, int, int, int] | None = None   # None = whole footprint
        self._current_window: tuple[int, int, int, int, int] | None = None  # r0, r1, c0, c1, stride
        self._sample: dict | None = None
        self._delta: dict | None = None
        self._fresh_view = True
        self._home_top = False
        self._building = False
        self._pending = False
        self._quiet = False
        self._pyr_job: dict | None = None
        self._cam_state = None
        self._cam_settled = True
        self.turbines: list[dict] = []       # [{'name', 'x', 'y'}] in the terrain CRS
        self.cables: list[dict] = []         # [{'name', 'pts' (n, 2)}] in the terrain CRS
        self.site_actors = []
        self.env_actors = []
        self._mesh_stride = 1
        self._sand_cache = None
        self._focus: tuple[float, float] | None = None   # scene XY of the turbine used as rotation centre
        self._wtg_busy = False

        self._rebuild_timer = QtCore.QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(350)
        self._rebuild_timer.timeout.connect(self.rebuild)
        self._exag_timer = QtCore.QTimer(self)
        self._exag_timer.setSingleShot(True)
        self._exag_timer.setInterval(60)
        self._exag_timer.timeout.connect(self._apply_exaggeration)
        self._cam_timer = QtCore.QTimer(self)       # detects "camera stopped moving"
        self._cam_timer.setInterval(350)
        self._cam_timer.timeout.connect(self._poll_camera)
        self._cam_timer.start()
        self._spin_timer = QtCore.QTimer(self)      # rotor animation
        self._spin_timer.setInterval(40)
        self._spin_timer.timeout.connect(self._spin_tick)
        self._spin_timer.start()
        self._rotors = None
        self._waves = None
        self._spin_theta = 0.0
        self._spin_clock = time.perf_counter()
        self._site_timer = QtCore.QTimer(self)
        self._site_timer.setSingleShot(True)
        self._site_timer.setInterval(80)
        self._site_timer.timeout.connect(self._draw_site_layers)

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QHBoxLayout(root)
        controls = QtWidgets.QWidget()
        controls.setFixedWidth(270)
        side = QtWidgets.QVBoxLayout(controls)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(295)
        scroll.setWidget(controls)
        layout.addWidget(scroll)

        heading = QtWidgets.QLabel("地形資料")
        heading.setStyleSheet("font-size:18px;font-weight:600")
        side.addWidget(heading)
        self.file_label = QtWidgets.QLabel("拖放 GeoTIFF 到視窗，或選取檔案")
        self.file_label.setWordWrap(True)
        side.addWidget(self.file_label)
        open_btn = QtWidgets.QPushButton("開啟 TIFF…")
        open_btn.clicked.connect(self.choose_file)
        side.addWidget(open_btn)

        side.addSpacing(16)
        self.margin = self._slider(side, "場址外圍留白", 0, 100, 8, "%")
        self.margin.valueChanged.connect(self._on_margin_changed)
        self.exaggeration = self._slider(side, "垂直誇大（整數倍）", 0, 5, 1, "×")
        self.exaggeration.valueChanged.connect(lambda _v: self._exag_timer.start())
        self.resolution = self._slider(side, "顯示解析度（每視窗格數）", 200, 2400, 1200, " 格")
        self.resolution.valueChanged.connect(lambda _v: self._rebuild_timer.start())
        self.auto_lod = QtWidgets.QCheckBox("拉近時自動提高解析度")
        self.auto_lod.setChecked(True)
        self.auto_lod.setToolTip("相機停止移動後，依畫面範圍重新取樣；拉近到一定程度就會是原始解析度。\n垂直俯視或接近俯視的視角效果最好。")
        side.addWidget(self.auto_lod)

        side.addSpacing(8)
        self.info = QtWidgets.QLabel("尚未載入資料")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color:#555")
        side.addWidget(self.info)
        view_row = QtWidgets.QHBoxLayout()
        home_btn = QtWidgets.QPushButton("重設視角")
        home_btn.setToolTip("整個場址，從南往北看（北朝上）")
        home_btn.clicked.connect(lambda: self.go_home(top=False))
        top_btn = QtWidgets.QPushButton("俯視（北朝上）")
        top_btn.clicked.connect(lambda: self.go_home(top=True))
        view_row.addWidget(home_btn)
        view_row.addWidget(top_btn)
        side.addLayout(view_row)
        self.color_btn = QtWidgets.QPushButton("依目前畫面自適應色階")
        self.color_btn.setToolTip("依目前視角內的地形高程重新調整色階，並以 2–98 百分位降低極端值影響。")
        self.color_btn.clicked.connect(self.adapt_color_range)
        self.color_btn.setEnabled(False)
        side.addWidget(self.color_btn)
        self.pyr_btn = QtWidgets.QPushButton("重建加速金字塔…")
        self.pyr_btn.setToolTip("為大型 TIFF 建立縮圖金字塔（存在使用者資料夾的快取，不修改原檔）。載入大檔案時會自動建立一次。")
        self.pyr_btn.clicked.connect(self.rebuild_pyramid)
        self.pyr_btn.setEnabled(False)
        side.addWidget(self.pyr_btn)
        self.compare_btn = QtWidgets.QPushButton("載入比較時期 TIFF…")
        self.compare_btn.clicked.connect(self.choose_comparison)
        side.addWidget(self.compare_btn)
        self.compare_toggle = QtWidgets.QCheckBox("顯示時期高程差")
        self.compare_toggle.setEnabled(False)
        self.compare_toggle.toggled.connect(self.set_comparison_visible)
        side.addWidget(self.compare_toggle)
        self.vector_btn = QtWidgets.QPushButton("加入風場圖層（SHP / GeoJSON）…")
        self.vector_btn.clicked.connect(self.choose_vector)
        side.addWidget(self.vector_btn)
        self.layers_box = QtWidgets.QVBoxLayout()
        side.addLayout(self.layers_box)

        side.addSpacing(8)
        wtg_box = QtWidgets.QGroupBox("風機 WTG")
        wtg_lay = QtWidgets.QVBoxLayout(wtg_box)
        self.wtg_chk = QtWidgets.QCheckBox("顯示風機標記與名稱")
        self.wtg_chk.setChecked(True)
        self.wtg_chk.toggled.connect(lambda _on: self._draw_site_layers())
        wtg_lay.addWidget(self.wtg_chk)
        self.model_chk = QtWidgets.QCheckBox("顯示 3D 風機模型（單樁 + TP + Boat landing）")
        self.model_chk.setChecked(True)
        self.model_chk.toggled.connect(lambda _on: self._draw_site_layers())
        wtg_lay.addWidget(self.model_chk)
        self.spin_chk = QtWidgets.QCheckBox("風機旋轉")
        self.spin_chk.setChecked(True)
        wtg_lay.addWidget(self.spin_chk)
        self.spin_rpm = self._slider(wtg_lay, "轉速", 0, 30, 10, " rpm")
        self.model_regen_btn = QtWidgets.QPushButton("重新產生風機模型（依 wtg_models.json）")
        self.model_regen_btn.clicked.connect(self._regen_models)
        wtg_lay.addWidget(self.model_regen_btn)
        self.wtg_list = QtWidgets.QListWidget()
        self.wtg_list.setFixedHeight(170)
        self.wtg_list.itemChanged.connect(self._on_wtg_item_changed)
        wtg_lay.addWidget(self.wtg_list)
        wtg_hint = QtWidgets.QLabel("勾選＝平移到該風機，並以它作為旋轉中心；取消勾選＝回到全場視角。")
        wtg_hint.setWordWrap(True)
        wtg_hint.setStyleSheet("color:#777")
        wtg_lay.addWidget(wtg_hint)
        side.addWidget(wtg_box)

        cable_box = QtWidgets.QGroupBox("海纜")
        cable_lay = QtWidgets.QVBoxLayout(cable_box)
        self.cable_chk = QtWidgets.QCheckBox("顯示海纜（貼合海床）")
        self.cable_chk.setChecked(True)
        self.cable_chk.toggled.connect(lambda _on: self._draw_site_layers())
        cable_lay.addWidget(self.cable_chk)
        side.addWidget(cable_box)

        buffer_box = QtWidgets.QGroupBox("Buffer zone")
        buffer_lay = QtWidgets.QVBoxLayout(buffer_box)
        self.buffer_chk = QtWidgets.QCheckBox("顯示半透明 Buffer zone")
        self.buffer_chk.toggled.connect(lambda _on: self._draw_site_layers())
        buffer_lay.addWidget(self.buffer_chk)
        self.cable_buffer = self._slider(buffer_lay, "海纜 Buffer（單側距離）", 0, 50, 25, " m")
        self.cable_buffer.valueChanged.connect(lambda _v: self._site_timer.start())
        self.wtg_buffer = self._slider(buffer_lay, "風機 Buffer 半徑", 0, 200, 50, " m")
        self.wtg_buffer.valueChanged.connect(lambda _v: self._site_timer.start())
        side.addWidget(buffer_box)

        env_box = QtWidgets.QGroupBox("環境")
        env_lay = QtWidgets.QVBoxLayout(env_box)
        self.sand_chk = QtWidgets.QCheckBox("砂質海床紋理（鋪滿整個海床，DEM 疊在上面）")
        self.sand_chk.toggled.connect(lambda _on: self._on_sand_toggled())
        env_lay.addWidget(self.sand_chk)
        self.sky_chk = QtWidgets.QCheckBox("天空背景")
        self.sky_chk.setChecked(True)
        self.sky_chk.toggled.connect(lambda _on: self._apply_background(render=True))
        env_lay.addWidget(self.sky_chk)
        self.water_chk = QtWidgets.QCheckBox("海水（半透明水面）")
        self.water_chk.toggled.connect(self._on_water_toggled)
        env_lay.addWidget(self.water_chk)
        self.water_opacity = self._slider(env_lay, "海水不透明度", 10, 90, 45, " %")
        self.water_opacity.valueChanged.connect(lambda _v: self._site_timer.start())
        self.wave_chk = QtWidgets.QCheckBox("海浪（風浪 + 湧浪，需先開海水）")
        self.wave_chk.setEnabled(False)
        self.wave_chk.toggled.connect(self._on_water_toggled)
        env_lay.addWidget(self.wave_chk)
        self.wind_dir = self._slider(env_lay, "風向（風從此方位吹來，北 0°、順時針）", 0, 359, 45, "°")
        self.windsea_h = self._slider(env_lay, "風浪高度 Hs", 0, 50, 10, " m", scale=10)
        self.swell_h = self._slider(env_lay, "湧浪高度 Hs", 0, 50, 8, " m", scale=10)
        self.swell_T = self._slider(env_lay, "湧浪週期", 6, 14, 9, " s")
        self.swell_off = self._slider(env_lay, "湧浪方向相對風向偏角", -90, 90, 20, "°")
        self.wave_gain = self._slider(env_lay, "海浪視覺放大（高度倍率）", 1, 10, 3, " ×")
        self._wave_sliders = [self.wind_dir, self.windsea_h, self.swell_h, self.swell_T, self.swell_off, self.wave_gain]
        for sl_ in self._wave_sliders:
            sl_.setEnabled(False)
        side.addWidget(env_box)

        self.site_info = QtWidgets.QLabel("")
        self.site_info.setWordWrap(True)
        self.site_info.setStyleSheet("color:#555")
        side.addWidget(self.site_info)
        reload_btn = QtWidgets.QPushButton("重新讀取風機／海纜資料")
        reload_btn.clicked.connect(self._reload_site_data)
        side.addWidget(reload_btn)
        side.addStretch(1)
        hint = QtWidgets.QLabel("滑鼠左鍵：旋轉\n滑鼠中鍵：平移\n滾輪：縮放\n\n僅繪製有效像元；稀疏資料會依有效地形範圍自動取景。")
        hint.setWordWrap(True)
        side.addWidget(hint)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)

        self.plotter = QtInteractor(root)
        layout.addWidget(self.plotter.interactor, 1)
        self._apply_background()
        self._add_axes()
        self.plotter.enable_trackball_style()
        self.statusBar().showMessage("就緒")
        if initial_file:
            QtCore.QTimer.singleShot(0, lambda: self.load_file(initial_file))

    @staticmethod
    def _slider(parent, title, minimum, maximum, value, suffix, scale=1):
        label = QtWidgets.QLabel()
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        value_label = QtWidgets.QLabel()
        value_label.setMinimumWidth(45)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(slider, 1)
        row.addWidget(value_label)
        parent.addWidget(label)
        parent.addLayout(row)

        def update(v):
            label.setText(title)
            shown = v / scale
            value_label.setText(f"{shown:g}{suffix}")

        slider.valueChanged.connect(update)
        update(value)
        return slider

    def _add_axes(self):
        # Scene axes: X = Easting, Y = Northing, Z = Height. ASCII only (VTK default font has no CJK).
        self.plotter.add_axes(xlabel="E", ylabel="N", zlabel="H")

    def _step(self, value: int, text: str):
        """Advance the status-bar progress bar (determinate) and keep the UI alive."""
        self.progress.setVisible(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(value)
        self.statusBar().showMessage(text)
        QtWidgets.QApplication.processEvents()

    # ------------------------------------------------------------------ pyramids
    def _ensure_pyramid(self, path: str, on_done, force: bool = False):
        """Make sure `path` has a pyramid (building one with a progress dialog), then call on_done()."""
        if self._pyr_job is not None:
            return
        if not force:
            pyr = Pyramid.load(path)
            if pyr is not None:
                self.pyramids[path] = pyr
                on_done()
                return
            if not needs_pyramid(path):
                on_done()
                return
        out_dir = Pyramid.cache_dir(path)
        shutil.rmtree(out_dir, ignore_errors=True)
        dlg = QtWidgets.QProgressDialog("正在建立加速金字塔（每個檔案只需做一次）…", "取消", 0, 100, self)
        dlg.setWindowTitle("建立影像金字塔")
        dlg.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setValue(0)
        worker = PyramidWorker(path, out_dir, self)
        worker.progress.connect(self._on_pyramid_progress)
        worker.finished_with.connect(self._on_pyramid_finished)
        dlg.canceled.connect(worker.cancel)
        self._pyr_job = {"path": path, "dlg": dlg, "worker": worker, "on_done": on_done}
        worker.start()

    def _on_pyramid_progress(self, value: int, text: str):
        if self._pyr_job:
            self._pyr_job["dlg"].setValue(value)
            self._pyr_job["dlg"].setLabelText(text)

    def _on_pyramid_finished(self, error: str):
        job, self._pyr_job = self._pyr_job, None
        if job is None:
            return
        job["dlg"].hide()
        job["dlg"].deleteLater()
        job["worker"].wait()
        if error == "cancelled":
            self.statusBar().showMessage("已取消建立金字塔，將直接讀取原檔（較慢）")
        elif error:
            QtWidgets.QMessageBox.warning(self, "金字塔建立失敗", f"{error}\n\n將直接讀取原檔（較慢）。")
        else:
            self.pyramids[job["path"]] = Pyramid.load(job["path"])
        job["on_done"]()

    def rebuild_pyramid(self):
        if not self.raster:
            return

        def done():
            self._sample = None
            self._delta = None
            self.rebuild()

        self.pyramids.pop(self.raster["path"], None)
        self._ensure_pyramid(self.raster["path"], done, force=True)

    # ------------------------------------------------------------------ files
    def choose_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "選取地形 GeoTIFF", "", "GeoTIFF (*.tif *.tiff);;所有檔案 (*)"
        )
        if path:
            self.load_file(path)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in {".tif", ".tiff"}:
                self.load_file(path)
                break

    def load_file(self, path: str):
        if self._pyr_job is not None:
            return
        self._ensure_pyramid(path, lambda: self._finish_load(path))

    def _footprint(self, path: str, width: int, height: int):
        """Bounding rows/cols (native pixels) of the valid data and a coverage estimate."""
        pyr = self.pyramids.get(path)
        if pyr is not None and pyr.levels:
            cand = [(f, p) for f, p in pyr.levels if max(width, height) / f <= FOOTPRINT_PX]
            f, level_path = cand[0] if cand else pyr.levels[-1]
            with rasterio.open(level_path) as lsrc:
                mask = lsrc.read_masks(1) > 0
            if not mask.any():
                raise ValueError("影像沒有可顯示的有效像元。")
            rows, cols = np.where(mask)
            return (max(0, int(rows.min()) * f - f), min(height, (int(rows.max()) + 1) * f + f),
                    max(0, int(cols.min()) * f - f), min(width, (int(cols.max()) + 1) * f + f),
                    float(mask.mean()))
        with rasterio.open(path) as src:
            # Average on the 0/255 mask retains partially covered coarse pixels.
            factor = min(1.0, FOOTPRINT_PX / max(height, width))
            mh, mw = max(1, int(height * factor)), max(1, int(width * factor))
            coarse = src.read_masks(1, out_shape=(mh, mw), resampling=Resampling.average) > 0
        if not coarse.any():
            raise ValueError("影像沒有可顯示的有效像元。")
        rows, cols = np.where(coarse)
        return (max(0, int(rows.min() * height / mh) - 1),
                min(height, int(np.ceil((rows.max() + 1) * height / mh)) + 1),
                max(0, int(cols.min() * width / mw) - 1),
                min(width, int(np.ceil((cols.max() + 1) * width / mw)) + 1),
                float(coarse.mean()))

    def _finish_load(self, path: str):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            self._step(10, "正在讀取有效範圍…")
            with rasterio.open(path) as src:
                height, width = src.height, src.width
                t, crs, nodata = src.transform, src.crs, src.nodata
            r0, r1, c0, c1, frac = self._footprint(path, width, height)
            corners = [t * (c, r) for c, r in ((0, 0), (width, 0), (0, height), (width, height))]
            self.raster = {
                "path": path, "width": width, "height": height, "transform": t,
                "crs": str(crs) if crs else "未定義", "crs_obj": crs,
                "origin_xy": (min(p[0] for p in corners), min(p[1] for p in corners)),
                "cell": float(max(np.hypot(t.a, t.d), np.hypot(t.b, t.e), 1e-12)),
                "bounds": (r0, r1, c0, c1), "nodata": nodata, "zref": None,
                "valid_fraction_estimate": frac,
            }
            self.file_label.setText(Path(path).name)
            self.comparison_path = None
            self.comparison_enabled = False
            self._sample = None
            self._delta = None
            self.view_window = None
            self._current_window = None
            self.clim = {"elev": None, "delta": None}
            self._fresh_view = True
            self._home_top = False
            self.compare_toggle.blockSignals(True)
            self.compare_toggle.setChecked(False)
            self.compare_toggle.setEnabled(False)
            self.compare_toggle.blockSignals(False)
            self.pyr_btn.setEnabled(True)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "無法載入 TIFF", str(exc))
            return
        finally:
            self.progress.setVisible(False)
            QtWidgets.QApplication.restoreOverrideCursor()
        self._load_site_data()
        self.rebuild()

    # --------------------------------------------------------------- sampling
    def _read_window(self, path: str, window: Window, out_h: int, out_w: int):
        """Resampled read -> (float32 array with NaN holes, valid mask). Uses the pyramid if present."""
        src_path, f = path, 1
        pyr = self.pyramids.get(path)
        if pyr is not None:
            f, level_path = pyr.pick(min(window.width / out_w, window.height / out_h))
            if level_path:
                src_path = level_path
                window = Window(window.col_off / f, window.row_off / f, window.width / f, window.height / f)
        with rasterio.open(src_path) as src:
            sampled = src.read(1, window=window, out_shape=(out_h, out_w),
                               resampling=Resampling.bilinear, masked=True)
        z = np.ma.filled(sampled.astype(np.float32), np.nan)  # also safe for int16 DEMs
        mask = (~np.ma.getmaskarray(sampled)) & np.isfinite(z)
        return z, mask

    def _ensure_sample(self, key, window, out_h, out_w):
        if self._sample is not None and self._sample["key"] == key:
            return self._sample
        r = self.raster
        z, mask = self._read_window(r["path"], window, out_h, out_w)
        if not mask.any():
            raise ValueError("此範圍沒有有效地形資料。請增加外圍留白或提高顯示解析度。")
        used, faces = build_quad_faces(mask)
        if used is None:
            raise ValueError("有效區域太稀疏，請提高『顯示解析度』或拉近視角。")
        transform = window_transform(window, r["transform"]) * rasterio.Affine.scale(
            window.width / out_w, window.height / out_h)
        cgrid = np.arange(out_w) + 0.5
        rgrid = (np.arange(out_h) + 0.5)[:, None]
        x = transform.a * cgrid + transform.b * rgrid + transform.c   # Affine form also covers rotated rasters
        y = transform.d * cgrid + transform.e * rgrid + transform.f
        cell = r["cell"]                      # fixed scene unit = native pixel size
        ox, oy = r["origin_xy"]
        xy = np.empty((used.size, 2), dtype=np.float32)
        xy[:, 0] = ((x.ravel()[used] - ox) / cell).astype(np.float32)
        xy[:, 1] = ((y.ravel()[used] - oy) / cell).astype(np.float32)
        elev = z.ravel()[used]
        if r["zref"] is None:                 # fixed per file so heights stay consistent across refinements
            r["zref"] = float(elev.min())
        self._sample = {
            "key": key, "window": window, "out_hw": (out_h, out_w), "transform": transform,
            "z": z, "used": used, "faces": faces, "xy": xy, "elev": elev,
            "zbase": r["zref"], "cell": cell,
        }
        self._delta = None
        return self._sample

    def _ensure_delta(self, s):
        dkey = (s["key"], self.comparison_path)
        if self._delta is not None and self._delta["key"] == dkey:
            return self._delta["values"]
        out_h, out_w = s["out_hw"]
        if self.comparison_aligned:
            z2, _ = self._read_window(self.comparison_path, s["window"], out_h, out_w)
        else:
            z2 = np.full((out_h, out_w), np.nan, dtype=np.float32)
            src_path = self.comparison_path
            pyr = self.pyramids.get(self.comparison_path)
            if pyr is not None:
                with rasterio.open(self.comparison_path) as cs:
                    comp_cell = max(np.hypot(cs.transform.a, cs.transform.d), 1e-12)
                _, level_path = pyr.pick(np.hypot(s["transform"].a, s["transform"].d) / comp_cell)
                if level_path:
                    src_path = level_path
            with rasterio.open(src_path) as comp_src:
                reproject(source=rasterio.band(comp_src, 1), destination=z2,
                          src_transform=comp_src.transform, src_crs=comp_src.crs,
                          dst_transform=s["transform"], dst_crs=self.raster["crs_obj"],
                          src_nodata=comp_src.nodata, dst_nodata=np.nan,
                          resampling=Resampling.bilinear)
        values = (z2 - s["z"]).ravel()[s["used"]].astype(np.float32)
        self._delta = {"key": dkey, "values": values}
        return values

    # ---------------------------------------------------------------- rebuild
    def _footprint_bounds(self):
        r = self.raster
        r0, r1, c0, c1 = r["bounds"]
        margin = self.margin.value() / 100
        pad_r, pad_c = round((r1 - r0) * margin), round((c1 - c0) * margin)
        return (max(0, r0 - pad_r), min(r["height"], r1 + pad_r),
                max(0, c0 - pad_c), min(r["width"], c1 + pad_c))

    def rebuild(self, *_) -> bool:
        if not self.raster:
            return False
        if self._building:           # a slider moved mid-build: run again afterwards
            self._pending = True
            return False
        self._building = True
        ok = False
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            r = self.raster
            fr0, fr1, fc0, fc1 = self._footprint_bounds()
            r0, r1, c0, c1 = self.view_window or (fr0, fr1, fc0, fc1)
            r0, r1, c0, c1 = max(fr0, r0), min(fr1, r1), max(fc0, c0), min(fc1, c1)
            if r1 - r0 < 2 or c1 - c0 < 2:
                r0, r1, c0, c1 = fr0, fr1, fc0, fc1
            maxdim = self.resolution.value()
            stride = max(1, int(np.ceil(max(r1 - r0, c1 - c0) / maxdim)))
            self._mesh_stride = stride
            out_h, out_w = max(2, int(np.ceil((r1 - r0) / stride))), max(2, int(np.ceil((c1 - c0) / stride)))
            key = (r0, r1, c0, c1, out_h, out_w)
            window = Window(c0, r0, c1 - c0, r1 - r0)

            self._step(5, "正在讀取取樣視窗…")
            s = self._ensure_sample(key, window, out_h, out_w)
            self._step(40, "正在建立 3D 網格…")
            points = np.empty((s["elev"].size, 3), dtype=np.float32)
            points[:, :2] = s["xy"]
            points[:, 2] = 0.0
            mesh = pv.PolyData(points, s["faces"])
            mesh.point_data["elev"] = s["elev"]
            if self.comparison_path:
                self._step(55, "正在計算比較期高程差…")
                mesh.point_data["delta"] = self._ensure_delta(s)
            self.mesh = mesh
            self._apply_height()

            self._step(75, "正在繪製…")
            self.plotter.clear()
            self.actor = None
            self.vector_actors = []
            self._apply_background()
            if self.clim["elev"] is None:
                self.clim["elev"] = (float(s["elev"].min()), float(s["elev"].max()))
            self._draw_surface()
            self._draw_vector_layers()
            self._current_window = (r0, r1, c0, c1, stride)   # the site layers need the grid spacing
            self._draw_site_layers()
            self.color_btn.setEnabled(True)
            self._add_axes()
            if self._fresh_view:
                self._home_view(self._home_top)
                self._fresh_view = False
            else:
                self._refocus()
            self.plotter.render()

            self._current_window = (r0, r1, c0, c1, stride)
            self._cam_settled = False       # let the LOD logic re-check the view
            width, height = r["width"], r["height"]
            self.info.setText(
                f"尺寸：{width:,} × {height:,}\n"
                f"有效覆蓋估計：約 {r['valid_fraction_estimate']:.1%}\n"
                f"高程：{float(s['elev'].min()):g} – {float(s['elev'].max()):g}\n"
                f"座標系：{r['crs']}\n"
                f"原始像元：{self._fmt_len(r['cell'])}　目前網格間距：{self._fmt_len(r['cell'] * stride)}\n"
                f"目前視窗：{out_w:,} × {out_h:,} 格（原始像元約 {stride} 倍取樣）"
            )
            self.statusBar().showMessage(f"已載入：{Path(r['path']).name}")
            ok = True
        except Exception as exc:
            if self._quiet:
                self.statusBar().showMessage(f"略過自動細化：{exc}")
            else:
                QtWidgets.QMessageBox.critical(self, "無法建立地形", str(exc))
        finally:
            self.progress.setVisible(False)
            QtWidgets.QApplication.restoreOverrideCursor()
            self._building = False
            if self._pending:
                self._pending = False
                self._rebuild_timer.start()
        return ok

    def _fmt_len(self, value: float) -> str:
        try:
            crs = self.raster["crs_obj"]
            if crs is not None and crs.is_projected and "metre" in (crs.linear_units or "").lower():
                return f"{value:g} m"
        except Exception:
            pass
        return f"{value:g}"

    def _apply_height(self):
        """Set vertex heights from cached elevations (no disk access)."""
        s = self._sample
        k = float(self.exaggeration.value())
        heights = (s["elev"] - s["zbase"]) / s["cell"] * k
        pts = np.array(self.mesh.points, dtype=np.float32)
        pts[:, 2] = heights
        self.mesh.points = pts
        self.scene_meta = {
            "crs": self.raster["crs"], "origin_xy": self.raster["origin_xy"], "cell": s["cell"],
            "z_mean": float(heights.mean()), "z_scene": float(heights.mean()) + 0.3,
        }

    def _apply_exaggeration(self):
        if self.mesh is None or self._sample is None:
            return
        self._apply_height()
        self._draw_surface()
        self._draw_vector_layers()
        self._draw_site_layers()
        self._refocus()
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def _on_margin_changed(self, _value):
        self.view_window = None
        self._rebuild_timer.start()

    # ---------------------------------------------------------------- camera
    def _home_view(self, top: bool = False):
        """Whole mesh, north up: oblique from the south, or straight down."""
        if top:
            direction, up = np.array([0.0, 0.0, 1.0]), (0.0, 1.0, 0.0)
        else:
            el = np.radians(50.0)
            direction, up = np.array([0.0, -np.cos(el), np.sin(el)]), (0.0, 0.0, 1.0)
        b = self.mesh.bounds
        center = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
        self.plotter.camera_position = [tuple(center + direction), tuple(center), up]
        self.plotter.reset_camera()

    def go_home(self, top: bool = False):
        if not self.raster:
            return
        self._home_top = top
        self._fresh_view = True
        self.view_window = None
        self._focus = None
        self._clear_wtg_checks()
        self.rebuild()

    def _camera_matrix(self):
        camera = self.plotter.renderer.GetActiveCamera()
        width, height = self.plotter.ren_win.GetSize()
        if width < 2 or height < 2:
            return None, None
        m = camera.GetCompositeProjectionTransformMatrix(width / height, -1.0, 1.0)
        return camera, np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])

    def _poll_camera(self):
        """When the camera has stopped moving, refine the mesh to what is on screen."""
        if (not self.raster or self.mesh is None or self._building or self._pyr_job is not None
                or not self.auto_lod.isChecked()):
            return
        camera = self.plotter.renderer.GetActiveCamera()
        state = tuple(np.round(camera.GetPosition() + camera.GetFocalPoint(), 3)) + (
            round(camera.GetViewAngle(), 3), round(camera.GetParallelScale(), 3))
        if state != self._cam_state:
            self._cam_state = state
            self._cam_settled = False
            return
        if self._cam_settled:
            return
        self._cam_settled = True
        self._refine_to_view()

    def _visible_window(self):
        """Ground area on screen as native-pixel bounds (r0, r1, c0, c1), or None."""
        if self.scene_meta is None:
            return None
        _, matrix = self._camera_matrix()
        if matrix is None:
            return None
        hits = ground_hits(matrix, self.scene_meta["z_mean"])
        if len(hits) < 3:
            return None
        r = self.raster
        ox, oy = r["origin_xy"]
        X = ox + hits[:, 0] * r["cell"]
        Y = oy + hits[:, 1] * r["cell"]
        inv = ~r["transform"]
        cols = inv.a * X + inv.b * Y + inv.c
        rows = inv.d * X + inv.e * Y + inv.f
        return float(rows.min()), float(rows.max()), float(cols.min()), float(cols.max())

    def _refine_to_view(self):
        vis = self._visible_window()
        if vis is None or self._current_window is None:
            return
        fr0, fr1, fc0, fc1 = self._footprint_bounds()
        vr0, vr1 = max(vis[0], fr0), min(vis[1], fr1)
        vc0, vc1 = max(vis[2], fc0), min(vis[3], fc1)
        if vr1 - vr0 < 2 or vc1 - vc0 < 2:
            return
        pad_r, pad_c = 0.3 * (vr1 - vr0), 0.3 * (vc1 - vc0)     # keep a context border while panning
        wr0, wr1 = max(fr0, int(np.floor(vr0 - pad_r))), min(fr1, int(np.ceil(vr1 + pad_r)))
        wc0, wc1 = max(fc0, int(np.floor(vc0 - pad_c))), min(fc1, int(np.ceil(vc1 + pad_c)))
        stride_new = max(1, int(np.ceil(max(wr1 - wr0, wc1 - wc0) / self.resolution.value())))
        cr0, cr1, cc0, cc1, stride_cur = self._current_window
        inside = cr0 <= vr0 and vr1 <= cr1 and cc0 <= vc0 and vc1 <= cc1
        if inside and stride_new * 1.5 > stride_cur:
            return                                              # already fine enough
        previous = self.view_window
        self.view_window = (wr0, wr1, wc0, wc1)
        self._quiet = True
        try:
            ok = self.rebuild()
        finally:
            self._quiet = False
        if not ok:
            self.view_window = previous

    def reset_camera(self):          # kept for compatibility
        self.go_home(self._home_top)

    # ---------------------------------------------------------------- drawing
    def _draw_surface(self):
        for title in list(self.plotter.scalar_bars.keys()):
            self.plotter.remove_scalar_bar(title)
        if self.actor is not None:
            try:
                self.plotter.remove_actor(self.actor)
            except Exception:
                pass
        # NOTE: keep 3-D text ASCII; VTK's default font has no CJK glyphs.
        if self.comparison_enabled and self.mesh is not None and "delta" in self.mesh.point_data:
            if self.clim["delta"] is None:
                values = np.asarray(self.mesh.point_data["delta"])
                finite = values[np.isfinite(values)]
                limit = max(float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0, 1e-6)
                self.clim["delta"] = (-limit, limit)
            self.actor = self.plotter.add_mesh(
                self.mesh, scalars="delta", cmap="RdBu_r", clim=self.clim["delta"],
                nan_color="#c8c8c8", reset_camera=False,
                scalar_bar_args={"title": "Elevation change (m): compare - base"})
        else:
            self.actor = self.plotter.add_mesh(
                self.mesh, scalars="elev", cmap="terrain", clim=self.clim["elev"],
                reset_camera=False, scalar_bar_args={"title": "Elevation (m)"})
        self.plotter.render()

    # ------------------------------------------------------------- comparison
    def choose_comparison(self):
        if not self.raster:
            QtWidgets.QMessageBox.information(self, "請先載入基準地形", "先載入基準年度的 GeoTIFF。")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "選取比較時期 GeoTIFF", "", "GeoTIFF (*.tif *.tiff)")
        if not path:
            return
        try:
            base = self.raster
            with rasterio.open(path) as src:
                if not src.crs or not base["crs_obj"]:
                    raise ValueError("基準期與比較期都必須有 CRS 座標系，才能正確對齊相減。")
                aligned = bool(
                    src.crs == base["crs_obj"] and src.width == base["width"]
                    and src.height == base["height"]
                    and np.allclose(tuple(src.transform)[:6], tuple(base["transform"])[:6],
                                    rtol=0, atol=1e-6))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "比較期載入失敗", str(exc))
            return

        def done():
            self.comparison_aligned = aligned
            self.comparison_path = path
            self.clim["delta"] = None
            self._delta = None
            self.compare_toggle.setEnabled(True)
            self.rebuild()
            self.statusBar().showMessage(f"比較期已對齊：{Path(path).name}；差值方向為比較期 − 基準期")

        self._ensure_pyramid(path, done)

    def set_comparison_visible(self, checked):
        self.comparison_enabled = bool(checked)
        self._draw_surface()

    # ------------------------------------------------------------ site layers
    def _reload_site_data(self):
        if not self.raster:
            return
        self._load_site_data()
        self._draw_site_layers()

    def _load_site_data(self):
        """Read WTG coordinates (CSV) and the subsea-cable SHP folder, converted to the terrain CRS."""
        self.turbines, self.cables = [], []
        notes = []
        crs = self.raster["crs"] if self.raster else None
        if not crs or crs == "未定義":
            notes.append("地形沒有座標系，無法對齊風機／海纜。")
        else:
            try:
                self.turbines = sl.load_turbines(WTG_CSV, crs)
                notes.append(f"風機：{len(self.turbines)} 座")
            except FileNotFoundError:
                notes.append(f"找不到風機座標檔 {WTG_CSV.name}")
            except Exception as exc:
                notes.append(f"風機座標讀取失敗：{exc}")
            try:
                self.cables, errors, n_files = sl.load_cables(CABLE_DIR, crs)
                notes.append(f"海纜：{n_files} 個 SHP、{len(self.cables)} 條線")
                notes += [f"略過 {e}" for e in errors]
            except FileNotFoundError:
                notes.append(f"找不到資料夾「{CABLE_DIR.name}」")
            except Exception as exc:
                notes.append(f"海纜讀取失敗：{exc}")
        self.site_info.setText("\n".join(notes))
        self._focus = None
        self._wtg_busy = True
        self.wtg_list.clear()
        for t in self.turbines:
            item = QtWidgets.QListWidgetItem(t["name"])
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.wtg_list.addItem(item)
        self._wtg_busy = False

    def _clear_wtg_checks(self):
        self._wtg_busy = True
        for i in range(self.wtg_list.count()):
            self.wtg_list.item(i).setCheckState(QtCore.Qt.CheckState.Unchecked)
        self._wtg_busy = False

    def _on_wtg_item_changed(self, item):
        if self._wtg_busy:
            return
        index = self.wtg_list.row(item)
        if item.checkState() == QtCore.Qt.CheckState.Checked:
            self._wtg_busy = True                      # only one turbine can be the rotation centre
            for i in range(self.wtg_list.count()):
                if i != index:
                    self.wtg_list.item(i).setCheckState(QtCore.Qt.CheckState.Unchecked)
            self._wtg_busy = False
            self.focus_turbine(index)
        elif not any(self.wtg_list.item(i).checkState() == QtCore.Qt.CheckState.Checked
                     for i in range(self.wtg_list.count())):
            self.go_home(self._home_top)

    def _scene_xy(self, x: float, y: float):
        ox, oy = self.raster["origin_xy"]
        cell = self.raster["cell"]
        return (x - ox) / cell, (y - oy) / cell

    def focus_turbine(self, index: int):
        """Pan to a turbine (zoom and view angle unchanged) and make it the rotation centre."""
        if self.mesh is None or not (0 <= index < len(self.turbines)):
            return
        t = self.turbines[index]
        sx, sy = self._scene_xy(t["x"], t["y"])
        camera = self.plotter.camera
        focal = np.array(camera.focal_point, dtype=float)
        position = np.array(camera.position, dtype=float)
        h = self._height_at(sx, sy)
        target = np.array([sx, sy, focal[2] if h is None else h])
        camera.position = tuple(position + (target - focal))
        camera.focal_point = tuple(target)
        self._focus = (sx, sy)
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()
        self._cam_settled = False          # let auto-refinement load the area around the turbine
        self.statusBar().showMessage(f"已移到 {t['name']}（E {t['x']:.1f}, N {t['y']:.1f}），旋轉中心設為此風機")

    def _refocus(self):
        """Keep the rotation centre on the terrain surface after the mesh or exaggeration changed."""
        if self._focus is None or self._sample is None:
            return
        camera = self.plotter.camera
        focal = np.array(camera.focal_point, dtype=float)
        if np.hypot(focal[0] - self._focus[0], focal[1] - self._focus[1]) > 1.0:
            self._focus = None             # the user panned away from the turbine
            return
        h = self._height_at(*self._focus)
        if h is None or abs(h - focal[2]) < 1e-6:
            return
        shift = np.array([0.0, 0.0, h - focal[2]])
        camera.position = tuple(np.array(camera.position, dtype=float) + shift)
        camera.focal_point = tuple(focal + shift)

    def _height_sampler(self):
        """sample(scene_x, scene_y) -> scene height (NaN where the terrain has no data)."""
        s, r = self._sample, self.raster
        if s is None:
            return None
        inv = ~s["transform"]
        ox, oy = r["origin_xy"]
        cell, zbase, z = s["cell"], s["zbase"], s["z"]
        k = float(self.exaggeration.value())

        def sample(sx, sy):
            X = ox + np.asarray(sx, dtype=float) * cell
            Y = oy + np.asarray(sy, dtype=float) * cell
            col = inv.a * X + inv.b * Y + inv.c - 0.5     # pixel centres sit at +0.5
            row = inv.d * X + inv.e * Y + inv.f - 0.5
            return (sl.bilinear_sample(z, col, row) - zbase) / cell * k

        return sample

    def _height_at(self, sx: float, sy: float):
        sample = self._height_sampler()
        if sample is None:
            return None
        v = float(sample(sx, sy)[0])
        return v if np.isfinite(v) else None

    def _cable_runs(self, to_scene, box, step):
        parts = []
        for c in self.cables:
            parts.extend(sl.clip_polyline_to_box(to_scene(c["pts"]), box))
        total = sum(sl.polyline_length(p) for p in parts)
        step = max(step, total / 120_000)             # hard cap on the number of vertices
        return [sl.densify(p, step) for p in parts]

    @staticmethod
    def _lines_polydata(runs, sample, lift):
        pieces, lines, offset = [], [], 0
        for run in runs:
            z = sample(run[:, 0], run[:, 1])
            good = np.isfinite(z)
            edges = np.flatnonzero(np.diff(np.r_[False, good, False].astype(np.int8)))
            for a, b in zip(edges[::2], edges[1::2]):   # contiguous stretches that have terrain
                if b - a < 2:
                    continue
                pieces.append(np.column_stack((run[a:b], z[a:b] + lift)))
                lines.append(np.r_[b - a, np.arange(offset, offset + b - a)])
                offset += b - a
        if not pieces:
            return None
        poly = pv.PolyData(np.vstack(pieces).astype(np.float32))
        poly.lines = np.concatenate(lines)
        return poly

    @staticmethod
    def _ribbon_arrays(runs, half_width, sample, lift):
        xys, quads, offset = [], [], 0
        for run in runs:
            xy, q = sl.ribbon_quads(run, half_width)
            if len(xy):
                xys.append(xy)
                quads.append(q + offset)
                offset += len(xy)
        if not xys:
            return None, None
        xy = np.vstack(xys)
        return sl.compact_quads(xy, sample(xy[:, 0], xy[:, 1]) + lift, np.vstack(quads))

    @staticmethod
    def _disc_arrays(centers, radius, stride, sample, lift):
        n_rings = int(np.clip(np.ceil(radius / (3.0 * stride)), 3, 40))
        xys, quads, offset = [], [], 0
        for cx, cy in centers:
            xy, q = sl.disc_quads(cx, cy, radius, n_rings)
            xys.append(xy)
            quads.append(q + offset)
            offset += len(xy)
        xy = np.vstack(xys)
        return sl.compact_quads(xy, sample(xy[:, 0], xy[:, 1]) + lift, np.vstack(quads))

    def _load_models(self, force=False):
        """Read wtg_models.json and WTG_<type>.ply / WTG_<type>_rotor.ply (generated on demand).

        Returns (cfg, {type: {"static": (V,F,C), "rotor": (V,F,C) | None, "hub": xyz, "axis": xyz}}).
        """
        if force or getattr(self, "_wtg_cfg", None) is None:
            self._wtg_cfg = wm.load_config(MODEL_DIR)
            self._wtg_meshes = {}
            try:
                paths = wm.ensure_models(MODEL_DIR, self._wtg_cfg, force=force)
            except Exception as exc:
                print("風機模型檔寫入失敗，改用記憶體內模型：", exc)
                paths = {}
            for tname, spec in self._wtg_cfg["types"].items():
                try:
                    static = wm.read_ply(paths[tname])
                    rp = wm.rotor_path(paths[tname])
                    rotor = wm.read_ply(rp) if rp.exists() else None
                except Exception as exc:
                    print(f"讀取 {tname} PLY 失敗，改用內建模型：", exc)
                    static, rotor = wm.build_wtg_parts(spec, self._wtg_cfg)
                fix = lambda m: None if m is None else (
                    m[0].astype(np.float64), m[1],
                    m[2] if m[2] is not None else np.full((len(m[0]), 3), 230, np.uint8))
                hub, axis = wm.rotor_pivot(spec, self._wtg_cfg)
                self._wtg_meshes[tname] = {"static": fix(static), "rotor": fix(rotor), "hub": hub, "axis": axis}
        return self._wtg_cfg, self._wtg_meshes

    def _regen_models(self):
        try:
            self._load_models(force=True)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "風機模型", f"重新產生失敗：{exc}")
            return
        self._draw_site_layers()

    def _model_polydata(self, tpos, z_sea, seabed, cell, cfg, meshes):
        """All turbine models (+ pile down to the seabed) merged into one RGB PolyData in scene units."""
        Vs, Fs, Cs, n = [], [], [], 0
        for i, t in enumerate(self.turbines):
            if not np.isfinite(tpos[i]).all():
                continue
            V0, F0, C0 = meshes[wm.turbine_type(t["name"], cfg)]["static"]
            V = V0 / cell + np.array([tpos[i, 0], tpos[i, 1], z_sea])
            Vs.append(V); Fs.append(F0 + n); Cs.append(C0); n += len(V)
            spec = cfg["types"][wm.turbine_type(t["name"], cfg)]
            z_bed = seabed[i]
            z_bot = z_sea - 8.0 / cell                         # the PLY pile stub ends 8 m below MSL
            if not np.isfinite(z_bed):
                z_bed = z_sea - 30.0 / cell                    # no DEM here: assume 30 m of water
            if z_bed < z_bot:
                r = spec["pile_dia_m"] / 2 / cell
                P, F = wm.frustum((tpos[i, 0], tpos[i, 1], z_bed), (tpos[i, 0], tpos[i, 1], z_bot), r, r, n=24, caps=False)
                Vs.append(P); Fs.append(F + n); Cs.append(np.tile(np.array(wm.YELLOW, np.uint8), (len(P), 1))); n += len(P)
        if not Vs:
            return None
        V = np.vstack(Vs).astype(np.float32)
        F = np.vstack(Fs).astype(np.int64)
        faces = np.column_stack((np.full(len(F), 3), F)).ravel()
        poly = pv.PolyData(V, faces)
        poly.point_data["RGB"] = np.vstack(Cs).astype(np.uint8)
        try:                                                    # crisp edges, smooth curved surfaces
            poly = poly.compute_normals(cell_normals=False, point_normals=True, split_vertices=True,
                                        feature_angle=50.0)
        except Exception:
            pass
        return poly

    def _rotor_data(self, tpos, z_sea, cell, cfg, meshes):
        """Rotor (hub + blades) of every turbine as one mesh, plus the arrays needed to spin it."""
        Vs, Fs, Cs, H, A, T, n = [], [], [], [], [], [], 0
        for i, t in enumerate(self.turbines):
            if not np.isfinite(tpos[i]).all():
                continue
            m = meshes[wm.turbine_type(t["name"], cfg)]
            if m["rotor"] is None:
                continue
            V0, F0, C0 = m["rotor"]
            off = np.array([tpos[i, 0], tpos[i, 1], z_sea])
            V = V0 / cell + off
            Vs.append(V); Fs.append(F0 + n); Cs.append(C0); n += len(V)
            H.append(np.broadcast_to(m["hub"] / cell + off, V.shape))
            A.append(np.broadcast_to(m["axis"], V.shape))
            T.append(np.full(len(V), i))
        if not Vs:
            return None
        V = np.vstack(Vs)
        F = np.vstack(Fs).astype(np.int64)
        H, A, T = np.vstack(H), np.vstack(A), np.concatenate(T)
        R = V - H
        poly = pv.PolyData(V.astype(np.float32), np.column_stack((np.full(len(F), 3), F)).ravel())
        poly.point_data["RGB"] = np.vstack(Cs).astype(np.uint8)
        phase = (T * 2.399963) % (2 * np.pi)                    # golden-angle offsets: turbines do not spin in sync
        rot = {"poly": poly, "H": H, "R": R, "AxR": np.cross(A, R), "AdR": A * (A * R).sum(1)[:, None],
               "phase": phase}
        poly.points = self._rotor_points(rot, 0.0)               # start already phase-shifted
        return rot

    @staticmethod
    def _rotor_points(rot, theta_deg):
        ang = -(np.radians(theta_deg) + rot["phase"])            # clockwise seen from upwind
        c, sn = np.cos(ang)[:, None], np.sin(ang)[:, None]
        return (rot["H"] + c * rot["R"] + sn * rot["AxR"] + (1 - c) * rot["AdR"]).astype(np.float32)

    def _spin_tick(self):
        now = time.perf_counter()
        dt, self._spin_clock = min(now - self._spin_clock, 0.2), now
        changed = False
        rot = self._rotors
        if rot is not None and self.spin_chk.isChecked() and self.spin_rpm.value() > 0:
            self._spin_theta = (self._spin_theta + self.spin_rpm.value() * 6.0 * dt) % 360.0
            rot["poly"].points = self._rotor_points(rot, self._spin_theta)
            changed = True
        if self._waves is not None:
            try:
                self._wave_update()
                changed = True
            except Exception as exc:                                # never let the animation kill the timer
                print("海浪更新失敗，已停用：", exc)
                self._waves = None
        if changed:
            self.plotter.render()

    @staticmethod
    def _sand_rgb(xw, yw, spacing):
        """Procedural sandy seabed colours from world coordinates (metres); stable when the mesh is refined."""
        def hash2(a, b):
            return np.modf(np.sin(a * 127.1 + b * 311.7) * 43758.5453)[0] % 1.0

        def vnoise(x, y):
            xi, yi = np.floor(x), np.floor(y)
            fx, fy = x - xi, y - yi
            ux, uy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
            a, b = hash2(xi, yi), hash2(xi + 1, yi)
            c, d = hash2(xi, yi + 1), hash2(xi + 1, yi + 1)
            return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy

        l1, l2, l3 = max(1.5, 6 * spacing), max(4.0, 16 * spacing), max(12.0, 40 * spacing)
        tone = 0.5 * vnoise(xw / l1, yw / l1) + 0.3 * vnoise(xw / l2 + 7.3, yw / l2 - 3.1) \
            + 0.2 * vnoise(xw / l3 - 5.7, yw / l3 + 9.9)
        lam = max(1.2, 5 * spacing)
        ang = 0.6
        warp = 2.5 * vnoise(xw / (6 * lam), yw / (6 * lam))
        ripple = 0.5 + 0.5 * np.sin(2 * np.pi * (xw * np.cos(ang) + yw * np.sin(ang)) / lam + warp)
        base1, base2 = np.array([204, 182, 140.0]), np.array([166, 142, 104.0])
        col = base1 + (base2 - base1) * np.clip(tone * 1.6 - 0.3, 0, 1)[:, None]
        col *= (0.93 + 0.14 * ripple)[:, None]
        return np.clip(col, 0, 255).astype(np.uint8)

    def _on_sand_toggled(self):
        if self.mesh is not None:
            self._draw_water(render=True)

    def _apply_background(self, render=False):
        if self.sky_chk.isChecked():
            self.plotter.set_background("#d8ebf6", top="#2f6db3")      # horizon haze -> blue sky
        else:
            self.plotter.set_background("#e8f1f5")
        if render:
            self.plotter.render()

    def _add_seabed(self, n=1000):
        """Sand-textured plane under the DEM that covers the whole seabed (also where the DEM has no data)."""
        s, mesh = self._sample, self.mesh
        elev = s["elev"][np.isfinite(s["elev"])]
        if elev.size == 0:
            return
        b = mesh.bounds
        w, h = b[1] - b[0], b[3] - b[2]
        extent = max(w, h, 1.0)
        pad = 0.5 * extent                                       # seabed continues past the DEM window
        x0, x1, y0, y1 = b[0] - pad, b[1] + pad, b[2] - pad, b[3] + pad
        k = float(self.exaggeration.value())
        lift = max(0.3, 0.0003 * extent)
        z = (float(elev.min()) - s["zbase"]) / s["cell"] * k - lift     # always below the DEM surface
        key = (round(x0, 1), round(x1, 1), round(y0, 1), round(y1, 1), n)
        if self._sand_cache is None or self._sand_cache[0] != key:
            xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
            X, Y = np.meshgrid(xs, ys)
            ox, oy = self.raster["origin_xy"]
            spacing = (x1 - x0) / (n - 1) * s["cell"]               # metres per grid step
            rgb = self._sand_rgb(X.ravel() * s["cell"] + ox, Y.ravel() * s["cell"] + oy, spacing)
            idx = np.arange(n * n).reshape(n, n)
            quads = np.column_stack((np.full((n - 1) ** 2, 4), idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(),
                                     idx[1:, 1:].ravel(), idx[1:, :-1].ravel())).ravel()
            self._sand_cache = (key, X.ravel().astype(np.float32), Y.ravel().astype(np.float32), rgb, quads)
        _, X, Y, rgb, quads = self._sand_cache
        plane = pv.PolyData(np.column_stack((X, Y, np.full(len(X), z, np.float32))), quads)
        plane.point_data["RGB"] = rgb
        self.env_actors.append(self.plotter.add_mesh(
            plane, scalars="RGB", rgb=True, lighting=False, reset_camera=False, pickable=False))

    def _on_water_toggled(self, _on=False):
        water = self.water_chk.isChecked()
        self.wave_chk.setEnabled(water)
        for sl_ in self._wave_sliders:
            sl_.setEnabled(water and self.wave_chk.isChecked())
        self._draw_water(render=True)

    # ---- ocean waves: wind sea + swell, deep-water dispersion, animated on a camera-centred grid
    def _wave_components(self):
        g = 9.81
        wind = float(self.wind_dir.value())
        hw, hs = self.windsea_h.value() / 10.0, self.swell_h.value() / 10.0
        T, off = float(self.swell_T.value()), float(self.swell_off.value())
        rng = np.random.default_rng(7)
        comps = []                                                  # (from-direction deg, wavelength, amplitude)
        if hw > 0:
            lam_p = max(8.0, 18.0 * hw + 10.0)
            lams = lam_p * np.array([0.45, 0.6, 0.8, 1.0, 1.3, 1.7])
            spread = np.array([-35.0, 0.0, 35.0, -18.0, 18.0, 0.0])
            wgt = np.exp(-np.log(lams / lam_p) ** 2 / (2 * 0.35 ** 2)) * np.cos(np.radians(spread)) ** 2
            amp = (hw / 4.0) * np.sqrt(2.0 * wgt / wgt.sum())          # sum of a^2/2 = (Hs/4)^2
            comps += [(wind + d, l, a) for d, l, a in zip(spread, lams, amp)]
        if hs > 0:
            lam_s = g * T * T / (2 * np.pi)
            for f, d in ((0.92, -4.0), (1.1, 4.0)):
                comps.append((wind + off + d, lam_s * f, (hs / 4.0) * 1.0))
        out = []
        for i, (frm, lam, a) in enumerate(comps):
            k = 2 * np.pi / lam
            th = np.radians(frm + 180.0)                                # waves travel towards frm + 180
            out.append((k * np.sin(th), k * np.cos(th), k, np.sqrt(g * k), a, lam, float(rng.uniform(0, 2 * np.pi))))
        return out

    def _wave_grid(self):
        """Screen-projected grid: camera rays hit the sea plane, so detail is dense near the camera."""
        w = self._waves
        nx, ny, cell = w["nx"], w["ny"], w["cell"]
        cam = self.plotter.camera
        pos = np.array(cam.position, float)
        d = np.array(cam.focal_point, float) - pos
        d /= np.linalg.norm(d)
        up = np.array(cam.up, float)
        right = np.cross(d, up)
        if np.linalg.norm(right) < 1e-9:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        upv = np.cross(right, d)
        try:
            width, height = self.plotter.window_size
            aspect = max(width, 1) / max(height, 1)
        except Exception:
            aspect = 1.5
        gx, gy = np.meshgrid(np.linspace(-1.15, 1.15, nx), np.linspace(-1.15, 1.15, ny))   # 15 % overscan
        if cam.parallel_projection:
            ps = float(cam.parallel_scale)
            org = pos + gx[..., None] * aspect * ps * right + gy[..., None] * ps * upv
            ray = np.broadcast_to(d, org.shape)
        else:
            t = np.tan(np.radians(cam.view_angle) / 2.0)
            org = np.broadcast_to(pos, gx.shape + (3,))
            ray = d + gx[..., None] * t * aspect * right + gy[..., None] * t * upv
        zsea, dmax = w["zsea"], w["half"]
        rz = ray[..., 2]
        hit = rz < -1e-6
        tt = np.where(hit, (zsea - org[..., 2]) / np.where(hit, rz, -1.0), 0.0)
        pt = org + ray * tt[..., None]
        horiz = ray[..., :2] / np.maximum(np.linalg.norm(ray[..., :2], axis=-1), 1e-9)[..., None]
        dist = np.linalg.norm(pt[..., :2] - org[..., :2], axis=-1)
        far = ~hit | (dist > dmax) | (tt <= 0)
        pt[..., :2] = np.where(far[..., None], org[..., :2] + horiz * dmax, pt[..., :2])
        X, Y = pt[..., 0], pt[..., 1]
        sp = np.maximum(np.hypot(np.gradient(X * cell, axis=1), np.gradient(Y * cell, axis=1)),
                        np.hypot(np.gradient(X * cell, axis=0), np.gradient(Y * cell, axis=0)))
        w.update(Xs=X.ravel().astype(np.float32), Ys=Y.ravel().astype(np.float32),
                 Xm=X.ravel() * cell, Ym=Y.ravel() * cell, sp=np.maximum(sp.ravel(), 0.05), cam=pos)

    def _wave_update(self, force=False):
        w = self._waves
        if w is None:
            return
        self._wave_grid()
        cell = w["cell"]
        t = time.perf_counter() - w["t0"]
        gain = float(self.wave_gain.value())                             # visual height exaggeration
        Xm, Ym, sp = w["Xm"], w["Ym"], w["sp"]
        zm = np.zeros(len(Xm)); sx = np.zeros(len(Xm)); sy = np.zeros(len(Xm))
        var = 0.0
        for kx, ky, k, om, a, lam, ph in self._wave_components():
            fade = np.clip((lam / sp - 2.0) / 2.0, 0.0, 1.0)              # drop waves the grid cannot resolve
            fade = fade * fade * (3 - 2 * fade)
            psi = kx * Xm + ky * Ym - om * t + ph
            af = a * fade * gain
            zm += af * np.cos(psi)
            ds = af * np.sin(psi)
            sx -= ds * kx
            sy -= ds * ky
            var += (a * gain) ** 2 / 2.0
        n = np.stack((-sx, -sy, np.ones_like(sx)), axis=1)
        n /= np.linalg.norm(n, axis=1)[:, None]
        light = np.array([0.4, 0.3, 0.86]); light /= np.linalg.norm(light)
        P = np.column_stack((w["Xs"], w["Ys"], (w["zsea"] + zm / cell).astype(np.float32))).astype(np.float32)
        view = w["cam"] - P.astype(np.float64)
        view /= np.maximum(np.linalg.norm(view, axis=1), 1e-9)[:, None]
        ndv = np.clip((n * view).sum(1), 0.0, 1.0)
        diff = np.clip(n @ light, 0.0, 1.0)
        deep, sky = np.array([22.0, 96.0, 150.0]), np.array([150.0, 195.0, 225.0])
        fres = ((1.0 - ndv) ** 3 * 0.75)[:, None]                         # grazing angles reflect the sky
        col = deep * (0.55 + 0.6 * diff)[:, None] * (1 - fres) + sky * fres
        refl = 2.0 * ndv[:, None] * n - view                              # sun glint
        glint = np.clip((refl * light).sum(1), 0.0, 1.0) ** 40
        col = col + 255.0 * (0.8 * glint)[:, None]
        sigma = np.sqrt(var)
        if sigma > 1e-6:                                                  # whitecaps on the highest crests
            foam = np.clip((zm / (2.2 * sigma) - 0.8) * 4.0, 0.0, 1.0) * 0.6
            col = col * (1 - foam[:, None]) + 255.0 * foam[:, None]
        w["poly"].points = P
        w["poly"].point_data["RGB"] = np.clip(col, 0, 255).astype(np.uint8)
        try:
            self.plotter.reset_camera_clipping_range()
        except Exception:
            pass

    def _draw_water(self, render=False):
        for actor in self.env_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self.env_actors = []
        self._waves = None
        if self.sand_chk.isChecked() and self.mesh is not None and self._sample is not None:
            self._add_seabed()
        if self.water_chk.isChecked() and self.mesh is not None and self._sample is not None:
            try:
                sea = float(self._load_models()[0].get("sea_level_in_dem_m", 0.0))
            except Exception:
                sea = 0.0
            s = self._sample
            z = (sea - s["zbase"]) / s["cell"] * float(self.exaggeration.value())
            b = self.mesh.bounds
            cx, cy = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2
            half = 2.5 * max(b[1] - b[0], b[3] - b[2], 1.0)       # reaches well past the terrain edge
            if self.wave_chk.isChecked():
                nx, ny = 280, 180
                idx = np.arange(nx * ny).reshape(ny, nx)
                faces = np.column_stack((np.full((nx - 1) * (ny - 1), 4), idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(),
                                         idx[1:, 1:].ravel(), idx[1:, :-1].ravel())).ravel()
                poly = pv.PolyData(np.zeros((nx * ny, 3), np.float32), faces)
                poly.point_data["RGB"] = np.zeros((nx * ny, 3), np.uint8)
                self._waves = {"poly": poly, "nx": nx, "ny": ny, "cell": s["cell"], "half": half, "zsea": z,
                               "t0": time.perf_counter()}
                self._wave_update(force=True)
                self.env_actors.append(self.plotter.add_mesh(
                    poly, scalars="RGB", rgb=True, opacity=self.water_opacity.value() / 100.0, lighting=False,
                    reset_camera=False, pickable=False))
                if render:
                    self.plotter.render()
                return
            quad = pv.PolyData(np.array([[cx - half, cy - half, z], [cx + half, cy - half, z],
                                         [cx + half, cy + half, z], [cx - half, cy + half, z]], np.float32),
                               np.array([4, 0, 1, 2, 3]))
            self.env_actors.append(self.plotter.add_mesh(
                quad, color="#1f78a8", opacity=self.water_opacity.value() / 100.0, lighting=False,
                reset_camera=False, pickable=False))
        if render:
            self.plotter.render()

    def _add_buffer(self, arrays, color):
        pts, faces = arrays
        if pts is None:
            return
        self.site_actors.append(self.plotter.add_mesh(
            pv.PolyData(pts, faces), color=color, opacity=0.35, lighting=False,
            reset_camera=False, pickable=False))

    def _draw_site_layers(self):
        """Turbine markers, draped cables and translucent buffer zones for the current mesh window."""
        for actor in self.site_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self.site_actors = []
        self._rotors = None
        self._draw_water()
        if self._sample is None or self.mesh is None or self._current_window is None:
            return
        want_wtg = self.wtg_chk.isChecked() and bool(self.turbines)
        want_cable = self.cable_chk.isChecked() and bool(self.cables)
        want_buffer = self.buffer_chk.isChecked() and bool(self.turbines or self.cables)
        want_model = self.model_chk.isChecked() and bool(self.turbines)
        if not (want_wtg or want_cable or want_buffer or want_model):
            self.plotter.render()
            return
        r, s = self.raster, self._sample
        cell, stride = s["cell"], self._current_window[4]
        sample = self._height_sampler()
        b = self.mesh.bounds
        extent = max(b[1] - b[0], b[3] - b[2], 1.0)
        lift = max(0.3, 0.0006 * extent)                  # keeps overlays above the terrain (no z-fighting)
        xy = s["xy"]
        box = (float(xy[:, 0].min()), float(xy[:, 0].max()), float(xy[:, 1].min()), float(xy[:, 1].max()))
        ox, oy = r["origin_xy"]

        def to_scene(p):
            return np.column_stack(((p[:, 0] - ox) / cell, (p[:, 1] - oy) / cell))

        tpos = (to_scene(np.array([[t["x"], t["y"]] for t in self.turbines]))
                if self.turbines else np.empty((0, 2)))
        cable_d, wtg_d = self.cable_buffer.value(), self.wtg_buffer.value()   # metres (TWD97)
        runs = []
        if self.cables and (want_cable or (want_buffer and cable_d > 0)):
            runs = self._cable_runs(to_scene, box, step=max(1.0, 2.0 * stride))

        if want_buffer:
            if runs and cable_d > 0:
                self._add_buffer(self._ribbon_arrays(runs, cable_d / cell, sample, lift), "#ff9f1c")
            if len(tpos) and wtg_d > 0:
                self._add_buffer(self._disc_arrays(tpos, wtg_d / cell, stride, sample, lift), "#d81b60")
        if want_cable and runs:
            poly = self._lines_polydata(runs, sample, 2.0 * lift)
            if poly is not None:
                self.site_actors.append(self.plotter.add_mesh(
                    poly, color="#ff9f1c", line_width=3, render_lines_as_tubes=True,
                    reset_camera=False, pickable=False))
        if want_model and len(tpos):
            try:
                cfg, meshes = self._load_models()
                k = float(self.exaggeration.value())
                z_sea = (cfg["sea_level_in_dem_m"] - s["zbase"]) / cell * k
                inside = ((tpos[:, 0] >= b[0]) & (tpos[:, 0] <= b[1]) & (tpos[:, 1] >= b[2]) & (tpos[:, 1] <= b[3]))
                tp_ = np.where(inside[:, None], tpos, np.nan)      # skip turbines far outside the loaded window
                poly = self._model_polydata(tp_, z_sea, sample(tpos[:, 0], tpos[:, 1]), cell, cfg, meshes)
                if poly is not None:
                    self.site_actors.append(self.plotter.add_mesh(
                        poly, scalars="RGB", rgb=True, smooth_shading=False, reset_camera=False, pickable=False,
                        specular=0.2))
                rot = self._rotor_data(tp_, z_sea, cell, cfg, meshes)
                if rot is not None:
                    self.site_actors.append(self.plotter.add_mesh(
                        rot["poly"], scalars="RGB", rgb=True, smooth_shading=False, reset_camera=False,
                        pickable=False, specular=0.2))
                    self._rotors = rot
            except Exception as exc:
                print("風機模型顯示失敗：", exc)
                want_model = False
        if want_wtg and len(tpos):
            z = sample(tpos[:, 0], tpos[:, 1])
            z = np.where(np.isfinite(z), z, self.scene_meta["z_mean"])
            radius = max(0.7, 0.004 * extent)
            if not want_model:
                cloud = pv.PolyData(np.column_stack((tpos, z + lift + radius)).astype(np.float32))
                glyphs = cloud.glyph(geom=pv.Sphere(radius=radius, theta_resolution=20, phi_resolution=20),
                                     scale=False, orient=False)
                self.site_actors.append(self.plotter.add_mesh(
                    glyphs, color="#d81b60", reset_camera=False, pickable=False))
            names = [t["name"] if t["name"].isascii() else f"#{i + 1}"   # VTK's default font has no CJK
                     for i, t in enumerate(self.turbines)]
            label_z = (z_sea + 45.0 / cell) if want_model else (z + lift + 2.5 * radius)
            label_pts = np.column_stack((tpos, np.broadcast_to(label_z, len(tpos)))).astype(np.float32)
            self.site_actors.append(self.plotter.add_point_labels(
                label_pts, names, font_size=14, text_color="black", shape=None,
                show_points=False, always_visible=True, reset_camera=False))
        self.plotter.render()

    # ---------------------------------------------------------------- vectors
    def choose_vector(self):
        if not self.raster:
            QtWidgets.QMessageBox.information(self, "請先載入地形", "先載入基準地形，向量圖層會轉換到它的 CRS。")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "加入風場向量圖層", "", "向量圖層 (*.shp *.geojson *.json)"
        )
        if not path:
            return
        box = QtWidgets.QCheckBox(Path(path).stem)
        box.setChecked(True)
        layer = {"path": path, "visible": True, "checkbox": box}
        box.toggled.connect(lambda on, item=layer: self._toggle_layer(item, on))
        self.layers_box.addWidget(box)
        self.vector_layers.append(layer)
        self._draw_vector_layers()

    def _toggle_layer(self, layer, visible):
        layer["visible"] = visible
        self._draw_vector_layers()

    def _draw_vector_layers(self):
        for actor in self.vector_actors:
            self.plotter.remove_actor(actor)
        self.vector_actors = []
        if not self.scene_meta or self.mesh is None:
            return
        b = self.mesh.bounds
        radius = max(0.7, 0.004 * max(b[1] - b[0], b[3] - b[2]))   # keep point markers visible at any zoom
        for layer in self.vector_layers:
            if not layer["visible"]:
                continue
            try:
                color = "#e63b2e" if layer["path"].lower().endswith(".shp") else "#1e78b4"
                specs = make_vector_actor(layer["path"], self.scene_meta["crs"],
                                          self.scene_meta["origin_xy"], self.scene_meta["cell"],
                                          self.scene_meta["z_scene"], color, radius=radius)
                for mesh, shade, width in specs:
                    self.vector_actors.append(self.plotter.add_mesh(mesh, color=shade, line_width=width or 1,
                                                                    render_lines_as_tubes=True,
                                                                    reset_camera=False))
            except Exception as exc:
                layer["checkbox"].setEnabled(False)
                self.statusBar().showMessage(f"圖層載入失敗：{exc}")
        self.plotter.render()

    # ------------------------------------------------------------- colour map
    def adapt_color_range(self):
        """Fit the colour scale to valid terrain points inside the current camera view."""
        if self.actor is None or self.mesh is None:
            return
        try:
            _, transform = self._camera_matrix()
            if transform is None:
                return
            points = np.asarray(self.mesh.points, dtype=np.float64)
            clip = points @ transform[:, :3].T + transform[:, 3]
            w = clip[:, 3]
            safe_w = np.where(np.abs(w) > 1e-12, w, np.nan)
            ndc = clip[:, :3] / safe_w[:, None]
            visible = (
                np.isfinite(ndc).all(axis=1)
                & (ndc[:, 0] >= -1) & (ndc[:, 0] <= 1)
                & (ndc[:, 1] >= -1) & (ndc[:, 1] <= 1)
                & (ndc[:, 2] >= -1) & (ndc[:, 2] <= 1)
                & (w > 0)
            )
            use_delta = self.comparison_enabled and "delta" in self.mesh.point_data
            values = np.asarray(self.mesh.point_data["delta" if use_delta else "elev"])[visible]
            values = values[np.isfinite(values)]
            if values.size < 2:
                raise ValueError("目前畫面內的有效地形點太少，請稍微縮小或移動視角後再試。")
            if use_delta:
                limit = float(np.percentile(np.abs(values), 98))
                low, high = -limit, limit
            else:
                low, high = np.percentile(values, [2, 98])
            if not np.isfinite([low, high]).all():
                raise ValueError("無法從目前畫面計算有效色階。")
            if high <= low:
                midpoint = float(values[0])
                low, high = midpoint - 0.5, midpoint + 0.5
            self.clim["delta" if use_delta else "elev"] = (float(low), float(high))
            mapper = self.actor.mapper
            mapper.scalar_range = (float(low), float(high))
            mapper.lookup_table.SetRange(float(low), float(high))
            mapper.lookup_table.Modified()
            mapper.Modified()
            self.plotter.render()
            self.statusBar().showMessage(
                f"色階已依畫面更新：{low:.2f} – {high:.2f}（{values.size:,} 個可視地形點）"
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "無法調整色階", str(exc))

    def closeEvent(self, event):
        if self._pyr_job is not None:
            self._pyr_job["worker"].cancel()
            self._pyr_job["worker"].wait(15000)
        self.plotter.close()
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = TerrainViewer(sys.argv[1] if len(sys.argv) > 1 else None)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
