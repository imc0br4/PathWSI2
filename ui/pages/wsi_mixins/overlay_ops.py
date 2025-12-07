from __future__ import annotations
from typing import Optional, Tuple
import os, json, numpy as np
from PySide6.QtWidgets import QMessageBox, QFileDialog
from PIL import Image
from ui.viewer.overlay_item import OverlayItem
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGraphicsItem

_OVERLAY_FILTER = "Overlay (*.png);;All (*)"
_MAX_PNG_SIDE = 12000
class OverlayOpsMixin:
    def _create_overlay_item(self):
        if self.overlay_item is None:
            self.overlay_item = OverlayItem()
        sc = self.view.scene()
        if sc is None:
            return
        if self.overlay_item.scene() is not sc:
            try:
                if self.overlay_item.scene():
                    self.overlay_item.scene().removeItem(self.overlay_item)
            except Exception:
                pass
            sc.addItem(self.overlay_item)
        self.overlay_item.setVisible(False)
        self.overlay_item.setPos(0, 0)
        self.overlay_item.setScale(1.0)
        self.overlay_item.setZValue(10)
        self._set_overlay_opacity(self._overlay_opacity)
        self.overlay_item.setCacheMode(QGraphicsItem.DeviceCoordinateCache)  # 平移复用缓存
        self.overlay_item.setAcceptedMouseButtons(Qt.NoButton)  # 不吃鼠标事件
        self.overlay_item.setTransformationMode(Qt.FastTransformation)  # 缩放用快速采样

    def _destroy_overlay_item(self):
        try:
            if self.overlay_item and self.overlay_item.scene():
                self.overlay_item.scene().removeItem(self.overlay_item)
        except Exception:
            pass
        self.overlay_item = None

    def _clear_overlay_state(self):
        self._overlay_rgba = None
        self._overlay_ds = 1.0
        self._overlay_pos = (0, 0)
        self._overlay_meta = None
        self._grid_edit = False
        self._grid_backup = None

    def _set_overlay_opacity(self, a: float):
        self._overlay_opacity = max(0.0, min(1.0, float(a)))
        try:
            if self.overlay_item:
                self.overlay_item.setOpacity(self._overlay_opacity)
        except Exception:
            pass

    def on_load_overlay(self):
        if not self.reader:
            QMessageBox.information(self, "提示", "请先打开一张 WSI。")
            return

        png_path, _ = QFileDialog.getOpenFileName(self, "加载叠加图", "", _OVERLAY_FILTER)
        if not png_path:
            return

        try:
            ov_rgba = np.array(Image.open(png_path).convert("RGBA"), dtype=np.uint8)
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"读取叠加图失败：{e}")
            return

        # 尝试载入 _meta.json
        meta = None
        mpath = os.path.splitext(png_path)[0] + "_meta.json"
        if os.path.isfile(mpath):
            try:
                with open(mpath, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = None

        # 计算 level/downsample/ROI
        try:
            level, ds, roi = self._resolve_overlay_alignment(ov_rgba.shape[1], ov_rgba.shape[0], meta)
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"计算对齐参数失败：{e}")
            return

        # 创建/刷新 overlay item
        try:
            if self.overlay_item is None or self.overlay_item.scene() is None:
                self._create_overlay_item()
            sc = self.view.scene()
            if sc is None:
                QMessageBox.critical(self, "加载失败", "场景未就绪，无法加入叠加图。")
                return
            if self.overlay_item.scene() is not sc:
                sc.addItem(self.overlay_item)
            self.overlay_item.setZValue(100)
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"创建叠加层失败：{e}")
            return

        # 应用到场景
        try:
            self.overlay_item.set_rgba(ov_rgba)
            x0 = int(roi[0]) if isinstance(roi, (list, tuple)) and len(roi) >= 2 else 0
            y0 = int(roi[1]) if isinstance(roi, (list, tuple)) and len(roi) >= 2 else 0
            self.overlay_item.setPos(x0, y0)
            self.overlay_item.setScale(float(ds))
            self.overlay_item.setVisible(True)
            # 透明度
            self.overlay_item.setOpacity(self._overlay_opacity)
        except Exception as e:
            QMessageBox.critical(self, "叠加失败", str(e))
            return

        # 记录状态 & 自动推断格子大小
        self._overlay_rgba = ov_rgba
        self._overlay_ds = float(ds)
        self._overlay_pos = (int(x0), int(y0))
        self._overlay_meta = meta if isinstance(meta, dict) else None
        auto_tile = self._infer_tile_from_meta(self._overlay_meta, default=self._grid_tile_px)
        if auto_tile and auto_tile > 0:
            self._grid_tile_px = int(auto_tile)

        QMessageBox.information(self, "已加载", f"叠加图已对齐：level={level}, downsample={ds:g}, ROI=({x0},{y0},...)")
        if hasattr(self, "btn_edit"):
            self.btn_edit.setEnabled(True)   # ← 新增

    def _on_export_overlay_clicked(self):
        if self._overlay_rgba is None or self.reader is None:
            QMessageBox.information(self, "提示", "请先打开一张 WSI 并加载叠加图。")
            return

        # 选择输出目录
        export_cfg = self.app_cfg.get("export", {}) if isinstance(self.app_cfg, dict) else {}
        default_dir = os.path.abspath(export_cfg.get("default_dir", "./exports"))
        os.makedirs(default_dir, exist_ok=True)
        out_dir = QFileDialog.getExistingDirectory(self, "选择导出目录", default_dir)
        if not out_dir:
            return

        # 对齐/网格参数（优先 _overlay_meta）
        meta = self._overlay_meta or {}
        lvl = int(meta.get("level", 0))
        ds = float(self._overlay_ds)
        ox, oy = self._overlay_pos
        w0 = int(round(self._overlay_rgba.shape[1] * ds))
        h0 = int(round(self._overlay_rgba.shape[0] * ds))
        roi = meta.get("bbox_level0") or meta.get("roi_level0") or [int(ox), int(oy), w0, h0]

        patch_level = meta.get("patch_size_level", None)
        stride_level = meta.get("stride_size_level", None)
        grid_px = int(self._grid_tile_px)  # 我们已在加载/应用时推断过

        # 后台保存（带 *_meta.json）
        from ui.pages.cls_page import SaveOverlayWorker  # 就地导入，避免循环依赖
        worker = SaveOverlayWorker(
            overlay_rgba=self._overlay_rgba,
            slide_path=getattr(self.reader, "path", "") or "",
            out_dir=out_dir,
            level=lvl,
            ds_at_level=ds,
            roi_level0=tuple(roi),
            patch_size_level=int(patch_level) if patch_level else None,
            grid_tile_px_on_overlay=grid_px,
            stride_size_level=int(stride_level) if stride_level else None,
            parent=self
        )

        def _ok(out: dict):
            lines = []
            if "overlay" in out:      lines.append(f"overlay: {out['overlay']}")
            if "overlay_meta" in out: lines.append(f"overlay_meta: {out['overlay_meta']}")
            if not lines: lines.append("（没有文件被保存）")
            QMessageBox.information(self, "导出完成", "已保存：\n" + "\n".join(lines))

        worker.finished_ok.connect(_ok)
        worker.failed.connect(lambda err: QMessageBox.critical(self, "导出失败", err))
        worker.start()

    def on_clear_overlay(self):
        self._set_edit_mode('off')  # ← 新增：先退出编辑
        self._destroy_overlay_item()
        self._clear_overlay_state()
        if hasattr(self, "btn_edit"):
            self.btn_edit.setEnabled(False)  # ← 新增

    def _resolve_overlay_alignment(self, w: int, h: int, meta: Optional[dict]) -> Tuple[int, float, Tuple[int, int, int, int]]:
        ds_list = [float(d) for d in self.reader.level_downsamples]
        dims = list(self.reader.level_dimensions)

        if isinstance(meta, dict):
            lvl = int(meta.get("level", 0))
            ds = float(meta.get("downsample", ds_list[lvl] if 0 <= lvl < len(ds_list) else 1.0))
            roi = meta.get("bbox_level0") or meta.get("roi_level0")
            if not (isinstance(roi, list) and len(roi) == 4):
                w0, h0 = dims[0]
                roi = [0, 0, w0, h0]
            return lvl, ds, (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))

        # 无 meta：根据 w/h 粗匹配
        errs = []
        for i, (W, H) in enumerate(dims):
            err = abs(W - w) / max(W, 1) + abs(H - h) / max(H, 1)
            errs.append((err, i))
        errs.sort()
        lvl = errs[0][1]
        ds = ds_list[lvl]
        w0, h0 = dims[0]
        return lvl, ds, (0, 0, w0, h0)

        # overlay_ops.py 里同一个类中
    def _infer_tile_from_meta(self, meta: dict, default: int = 64) -> int:
        if not isinstance(meta, dict):
            return default

        # 1) 首选：导出时就写在 meta 里的 overlay 像素下网格边长
        v = meta.get("grid_tile_px_on_overlay")
        if isinstance(v, int) and v > 0:
            return int(v)

        # 2) 次选：以「level 像素」描述的 stride / patch（不再除 ds）
        v = meta.get("stride_size_level")
        if isinstance(v, int) and v > 0:
            return int(v)
        v = meta.get("patch_size_level")
        if isinstance(v, int) and v > 0:
            return int(v)

        # 3) 只有 level-0 尺寸：换算到 overlay 像素（除以 downsample）
        ds = float(meta.get("downsample", getattr(self, "_overlay_ds", 1.0) or 1.0))
        for k in ("patch_size", "input_size"):
            v = meta.get(k)
            if isinstance(v, int) and v > 0:
                return max(1, int(round(v / max(ds, 1e-9))))

        return default

    def _mark_dirty(self, x, y, w, h):
        # 聚合多个脏区
        if self._dirty_rect is None:
            self._dirty_rect = [x, y, w, h]
        else:
            x0, y0, w0, h0 = self._dirty_rect
            x1 = min(x0, x);
            y1 = min(y0, y)
            x2 = max(x0 + w0, x + w);
            y2 = max(y0 + h0, y + h)
            self._dirty_rect = [x1, y1, x2 - x1, y2 - y1]
        if not self._flush_timer.isActive():
            self._flush_timer.start()

    def _flush_dirty(self):
        if self._dirty_rect is None or self._overlay_rgba is None:
            return
        x, y, w, h = self._dirty_rect
        self._dirty_rect = None
        try:
            if self.overlay_item and hasattr(self.overlay_item, "update_subrect"):
                sub = self._overlay_rgba[y:y + h, x:x + w]
                self.overlay_item.update_subrect(x, y, w, h, sub)
            elif self.overlay_item:
                # 退化：整图刷新（慢，但保证可用）
                self.overlay_item.set_rgba(self._overlay_rgba)
        except Exception:
            # 任何异常也退化到整图刷新
            if self.overlay_item:
                self.overlay_item.set_rgba(self._overlay_rgba)

    def _downsample_overlay_for_display(self, arr, ds, meta):
        """若 overlay 过大，则按 2/4/8… 等比降采样；并把 ds 与 grid 一并同步。"""
        H, W = arr.shape[:2]
        sf = 1
        side = max(H, W)
        while side > _MAX_PNG_SIDE:
            sf *= 2
            side = (side + 1) // 2
        if sf > 1:
            arr = np.array(
                Image.fromarray(arr, "RGBA").resize((W // sf, H // sf), Image.NEAREST),
                dtype=np.uint8
            )
            ds = float(ds) * sf
            if isinstance(meta, dict):
                g = meta.get("grid_tile_px_on_overlay")
                if isinstance(g, int) and g > 0:
                    # 叠加图像素被缩小了 sf 倍，网格边长也要等比例除以 sf
                    meta["grid_tile_px_on_overlay"] = max(1, int(round(g / sf)))
        return arr, ds, sf, meta
