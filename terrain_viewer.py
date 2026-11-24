"""Interactive 3D viewer for GeoTIFF terrain rasters."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from PySide6 import QtCore, QtWidgets
from pyvistaqt import QtInteractor
import pyvista as pv
from scene_layers import make_vector_actor


class TerrainViewer(QtWidgets.QMainWindow):
    def __init__(self, initial_file: str | None = None):
        super().__init__()
        self.setWindowTitle("離岸風場地形 3D 檢視器")
        self.resize(1420, 900)
        self.setAcceptDrops(True)
        self.raster: dict | None = None
        self.actor = None
        self.mesh = None
        self.full_scalar_range: tuple[float, float] | None = None
        self.comparison_path: str | None = None
        self.comparison_enabled = False
        self.vector_layers: list[dict] = []
        self.vector_actors = []
        self.scene_meta = None
        self._building = False

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QHBoxLayout(root)
        controls = QtWidgets.QWidget()
        controls.setFixedWidth(280)
        side = QtWidgets.QVBoxLayout(controls)
        layout.addWidget(controls)

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
        self.margin.valueChanged.connect(self.rebuild)
        self.exaggeration = self._slider(side, "垂直誇大", 1, 300, 50, "×", scale=10)
        self.exaggeration.valueChanged.connect(self.rebuild)
        self.resolution = self._slider(side, "顯示解析度", 200, 1800, 900, " 格")
        self.resolution.valueChanged.connect(self.rebuild)

        side.addSpacing(8)
        self.info = QtWidgets.QLabel("尚未載入資料")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color:#555")
        side.addWidget(self.info)
        fit_btn = QtWidgets.QPushButton("重設視角")
        fit_btn.clicked.connect(self.reset_camera)
        side.addWidget(fit_btn)
        self.color_btn = QtWidgets.QPushButton("依目前畫面自適應色階")
        self.color_btn.setToolTip("依目前視角內的地形高程重新調整色階，並以 2–98 百分位降低極端值影響。")
        self.color_btn.clicked.connect(self.adapt_color_range)
        self.color_btn.setEnabled(False)
        side.addWidget(self.color_btn)
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
        side.addStretch(1)
        hint = QtWidgets.QLabel("滑鼠左鍵：旋轉\n滑鼠中鍵：平移\n滾輪：縮放\n\n僅繪製有效像元；稀疏資料會依有效地形範圍自動取景。")
        hint.setWordWrap(True)
        side.addWidget(hint)

        self.plotter = QtInteractor(root)
        layout.addWidget(self.plotter.interactor, 1)
        self.plotter.set_background("#e8f1f5")
        self.plotter.add_axes()
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
        try:
            self.statusBar().showMessage("正在讀取 GeoTIFF 有效範圍摘要…")
            QtWidgets.QApplication.processEvents()
            with rasterio.open(path) as src:
                height, width = src.height, src.width
                # Read only a bounded overview mask to find the survey footprint.
                # Average on the 0/255 mask retains partially covered coarse pixels;
                # max is not supported by GDAL's read_masks operation.
                factor = min(1.0, 2048 / max(height, width))
                mh, mw = max(1, int(height*factor)), max(1, int(width*factor))
                coarse_mask = src.read_masks(1, out_shape=(mh, mw), resampling=Resampling.average) > 0
                if not coarse_mask.any():
                    raise ValueError("影像沒有可顯示的有效像元。")
                rows, cols = np.where(coarse_mask)
                r0 = max(0, int(rows.min()*height/mh)-1)
                r1 = min(height, int(np.ceil((rows.max()+1)*height/mh))+1)
                c0 = max(0, int(cols.min()*width/mw)-1)
                c1 = min(width, int(np.ceil((cols.max()+1)*width/mw))+1)
                self.raster = {
                    "path": path,
                    "width": width,
                    "height": height,
                    "transform": src.transform,
                    "crs": str(src.crs) if src.crs else "未定義",
                    "bounds": (r0, r1, c0, c1),
                    "nodata": src.nodata,
                    "valid_fraction_estimate": float(coarse_mask.mean()),
                }
            self.file_label.setText(Path(path).name)
            self.comparison_path = None
            self.comparison_enabled = False
            self.compare_toggle.blockSignals(True)
            self.compare_toggle.setChecked(False)
            self.compare_toggle.setEnabled(False)
            self.compare_toggle.blockSignals(False)
            self.rebuild()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "無法載入 TIFF", str(exc))

    def rebuild(self, *_):
        if not self.raster or self._building:
            return
        self._building = True
        try:
            self.statusBar().showMessage("正在讀取視窗取樣並建立 3D 網格…")
            QtWidgets.QApplication.processEvents()
            r = self.raster
            height, width = r["height"], r["width"]
            r0, r1, c0, c1 = r["bounds"]
            # Crop around valid pixels, then add a small, adjustable context border.
            margin = self.margin.value() / 100
            rh, cw = r1 - r0, c1 - c0
            pad_r, pad_c = round(rh * margin), round(cw * margin)
            r0, r1 = max(0, r0 - pad_r), min(height, r1 + pad_r)
            c0, c1 = max(0, c0 - pad_c), min(width, c1 + pad_c)
            maxdim = self.resolution.value()
            stride = max(1, int(np.ceil(max(r1-r0, c1-c0) / maxdim)))
            out_h, out_w = max(2, int(np.ceil((r1-r0)/stride))), max(2, int(np.ceil((c1-c0)/stride)))
            window = Window(c0, r0, c1-c0, r1-r0)
            # Keep memory bounded by the preview grid, regardless of source TIFF size.
            # Do not replace this with src.read(1) without a window/out_shape.
            with rasterio.open(r["path"]) as src:
                sampled = src.read(1, window=window, out_shape=(out_h, out_w),
                                   resampling=Resampling.bilinear, masked=True)
                z = np.asarray(sampled.filled(np.nan), dtype=np.float32)
                mask = (~np.ma.getmaskarray(sampled)) & np.isfinite(z)
                transform = src.window_transform(window) * rasterio.Affine.scale(
                    (c1-c0)/out_w, (r1-r0)/out_h)
            if not mask.any():
                raise ValueError("此範圍沒有有效地形資料。請增加外圍留白或提高顯示解析度。")
            # Affine coordinates support rotated rasters as well as north-up grids.
            cgrid, rgrid = np.meshgrid(np.arange(out_w)+0.5, np.arange(out_h)+0.5)
            x = transform.a*cgrid + transform.b*rgrid + transform.c
            y = transform.d*cgrid + transform.e*rgrid + transform.f
            finite_z = z[mask]
            zbase = float(np.nanmin(finite_z))
            zsafe = np.where(mask, z, zbase)
            zrange = max(float(np.nanmax(finite_z) - zbase), 1e-6)
            # Normalize horizontal coordinates to the raster cell size to keep vertical
            # exaggeration interpretable in projected and geographic coordinate systems.
            cell = max(np.hypot(transform.a, transform.d), np.hypot(transform.b, transform.e), 1e-12)
            xx = (x - np.nanmin(x)) / cell
            yy = (y - np.nanmin(y)) / cell
            zz = (zsafe - zbase) / cell * (self.exaggeration.value() / 10)
            grid = pv.StructuredGrid(xx, yy, zz)
            grid.point_data["高程"] = z.ravel(order="F")
            if self.comparison_path is not None:
                comparison = np.full((out_h, out_w), np.nan, dtype=np.float32)
                with rasterio.open(self.comparison_path) as comp_src:
                    reproject(source=rasterio.band(comp_src, 1), destination=comparison,
                              src_transform=comp_src.transform, src_crs=comp_src.crs,
                              dst_transform=transform, dst_crs=r["crs"],
                              src_nodata=comp_src.nodata, dst_nodata=np.nan,
                              resampling=Resampling.bilinear)
                delta = comparison - z
                delta[~mask] = np.nan
                grid.point_data["高程差"] = delta.ravel(order="F")
            # Drop quads touching nodata so missing pixels remain transparent holes.
            cell_mask = (mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]).ravel(order="C")
            mesh = grid.extract_cells(np.flatnonzero(cell_mask)).extract_surface()
            if mesh.n_cells == 0:
                raise ValueError("有效區域太稀疏，降低『顯示解析度』以保留更多像元。")
            self.plotter.clear()
            self.actor = None
            self.vector_actors = []
            self.plotter.set_background("#e8f1f5")
            self.mesh = mesh
            self.scene_meta = {
                "crs": r["crs"], "origin_xy": (float(np.nanmin(x)), float(np.nanmin(y))),
                "cell": cell, "z_scene": float(np.nanmean((z[mask]-zbase)/cell*(self.exaggeration.value()/10))) + 0.3,
            }
            self.full_scalar_range = (float(np.nanmin(finite_z)), float(np.nanmax(finite_z)))
            self._draw_surface()
            self._draw_vector_layers()
            self.color_btn.setEnabled(True)
            self.plotter.add_axes()
            self.plotter.reset_camera()
            self.plotter.render()
            self.info.setText(
                f"尺寸：{width:,} × {height:,}\n"
                f"有效覆蓋估計：約 {r['valid_fraction_estimate']:.1%}\n"
                f"高程：{zbase:g} – {float(np.nanmax(finite_z)):g}\n"
                f"座標系：{r['crs']}\n預覽網格：{out_w:,} × {out_h:,}（原始像元約 {stride} 倍取樣）"
            )
            self.statusBar().showMessage(f"已載入：{Path(r['path']).name}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "無法建立地形", str(exc))
        finally:
            self._building = False

    def reset_camera(self):
        self.plotter.reset_camera()
        self.plotter.render()

    def _draw_surface(self):
        for title in list(self.plotter.scalar_bars.keys()):
            self.plotter.remove_scalar_bar(title)
        if self.actor is not None:
            try:
                self.plotter.remove_actor(self.actor)
            except Exception:
                pass
        if self.comparison_enabled and self.mesh is not None and "高程差" in self.mesh.point_data:
            values = np.asarray(self.mesh.point_data["高程差"])
            finite = values[np.isfinite(values)]
            limit = float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0
            clim = (-max(limit, 1e-6), max(limit, 1e-6))
            self.actor = self.plotter.add_mesh(self.mesh, scalars="高程差", cmap="RdBu_r", clim=clim,
                                               scalar_bar_args={"title": "高程差 (比較期 − 基準期)"})
        else:
            self.actor = self.plotter.add_mesh(self.mesh, scalars="高程", cmap="terrain",
                                               clim=self.full_scalar_range,
                                               scalar_bar_args={"title": "高程 (m)"})
        self.plotter.render()

    def choose_comparison(self):
        if not self.raster:
            QtWidgets.QMessageBox.information(self, "請先載入基準地形", "先載入基準年度的 GeoTIFF。")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "選取比較時期 GeoTIFF", "", "GeoTIFF (*.tif *.tiff)")
        if not path:
            return
        try:
            with rasterio.open(path) as src:
                if not src.crs or not self.raster["crs"] or self.raster["crs"] == "未定義":
                    raise ValueError("基準期與比較期都必須有 CRS 座標系，才能正確對齊相減。")
                if not src.crs:
                    raise ValueError("比較期 TIFF 缺少 CRS 座標系資訊。")
            self.comparison_path = path
            self.compare_toggle.setEnabled(True)
            self.statusBar().showMessage(f"比較期已對齊：{Path(path).name}；差值方向為比較期 − 基準期")
            self.rebuild()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "比較期載入失敗", str(exc))

    def set_comparison_visible(self, checked):
        self.comparison_enabled = bool(checked)
        self._draw_surface()

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
        if not self.scene_meta:
            return
        for layer in self.vector_layers:
            if not layer["visible"]:
                continue
            try:
                color = "#e63b2e" if layer["path"].lower().endswith(".shp") else "#1e78b4"
                specs = make_vector_actor(layer["path"], self.scene_meta["crs"],
                                          self.scene_meta["origin_xy"], self.scene_meta["cell"],
                                          self.scene_meta["z_scene"], color)
                for mesh, shade, width in specs:
                    self.vector_actors.append(self.plotter.add_mesh(mesh, color=shade, line_width=width or 1,
                                                                    render_lines_as_tubes=True))
            except Exception as exc:
                layer["checkbox"].setEnabled(False)
                self.statusBar().showMessage(f"圖層載入失敗：{exc}")
        self.plotter.render()

    def adapt_color_range(self):
        """Fit the colour scale to valid terrain points inside the current camera view."""
        if self.actor is None or self.mesh is None:
            return
        try:
            renderer = self.plotter.renderer
            camera = renderer.GetActiveCamera()
            width, height = self.plotter.ren_win.GetSize()
            aspect = width / max(height, 1)
            matrix = camera.GetCompositeProjectionTransformMatrix(aspect, -1.0, 1.0)
            transform = np.array([[matrix.GetElement(row, col) for col in range(4)] for row in range(4)])
            points = np.asarray(self.mesh.points)
            homogeneous = np.column_stack((points, np.ones(len(points))))
            clip = homogeneous @ transform.T
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
            scalar_name = "高程差" if self.comparison_enabled else "高程"
            values = np.asarray(self.mesh.point_data[scalar_name])[visible]
            values = values[np.isfinite(values)]
            if values.size < 2:
                raise ValueError("目前畫面內的有效地形點太少，請稍微縮小或移動視角後再試。")
            if self.comparison_enabled:
                limit = float(np.percentile(np.abs(values), 98))
                low, high = -limit, limit
            else:
                low, high = np.percentile(values, [2, 98])
            if not np.isfinite([low, high]).all():
                raise ValueError("無法從目前畫面計算有效色階。")
            if high <= low:
                midpoint = float(values[0])
                low, high = midpoint - 0.5, midpoint + 0.5
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
        self.plotter.close()
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = TerrainViewer(sys.argv[1] if len(sys.argv) > 1 else None)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
