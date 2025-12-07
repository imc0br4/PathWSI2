# ui/pages/wsi_mixins/roi_ops.py
from __future__ import annotations
from typing import List
from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import QMessageBox, QFileDialog
from PIL import Image
from ui.viewer.roi_tools import RoiRect, MODE_EDIT
import os
import json
import numpy as np

class RoiOpsMixin:
    def _enter_roi(self, draw: bool = True):
        if not self.reader:
            QMessageBox.information(self, "提示", "请先打开一张 WSI。")
            return
        if not self.roi_tool.is_active():
            self.roi_tool.activate(True)
        self.view.set_allow_doubleclick_home(False)
        if draw:
            self.roi_tool.set_mode_draw()
        else:
            self.roi_tool.set_mode_edit()
        self._update_buttons(active=True)

    def _exit_roi(self):
        try:
            self.roi_tool.clear()
        except Exception:
            pass
        if self.roi_tool.is_active():
            self.roi_tool.activate(False)
        self.view.set_allow_doubleclick_home(True)
        self._update_buttons(active=(self.reader is not None))

    def _on_roi_mode_changed(self, mode: int):
        self._update_buttons(active=(self.reader is not None))

    def _on_roi_selection_changed(self):
        self._update_buttons(active=(self.reader is not None))

    def _on_delete_selected_roi_silent(self):
        if not self.roi_tool.is_active() or self.roi_tool.mode() != MODE_EDIT:
            return
        self.roi_tool.delete_selected()

    def _on_escape_clear(self):
        if not self.roi_tool.is_active() or self.roi_tool.mode() != MODE_EDIT:
            return
        self.roi_tool.clear_selection()

    def _on_select_all(self):
        if not self.roi_tool.is_active() or self.roi_tool.mode() != MODE_EDIT:
            return
        self.roi_tool.select_all()

    def _on_delete_selected_roi(self):
        if not self.roi_tool.is_active():
            QMessageBox.information(self, "提示", "请先进入 ROI 模式。")
            return
        if self.roi_tool.mode() != MODE_EDIT:
            QMessageBox.information(self, "提示", "请切换到“编辑 ROI”后再删除。")
            return
        n = self.roi_tool.delete_selected()
        if n == 0:
            QMessageBox.information(self, "提示", "请先单击选择要删除的 ROI（或框选多个），也可直接按 Delete/Backspace。")

    def _on_roi_clear(self):
        if not self.roi_tool.is_active():
            QMessageBox.information(self, "提示", "请先进入 ROI 模式。")
            return
        if self.roi_tool.has_rois():
            self.roi_tool.clear()

    def on_save_roi(self):
        if not self.reader:
            QMessageBox.information(self, "提示", "请先打开一张 WSI。")
            return
        rois: List[RoiRect] = self.roi_tool.rois()
        if not rois:
            QMessageBox.information(self, "提示", "请先框选 ROI。")
            return

        if len(rois) == 1:
            out_path, _ = QFileDialog.getSaveFileName(
                self, "保存 ROI (10×)", "", "PNG (*.png);;TIFF (*.tif *.tiff)"
            )
            if not out_path:
                return
            self._export_roi_10x(rois[0].rect, out_path)
        else:
            out_dir = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
            if not out_dir:
                return
            base = os.path.splitext(os.path.basename(self.reader.path))[0] if getattr(self.reader, "path", None) else "slide"
            for i, r in enumerate(rois):
                out_path = os.path.join(out_dir, f"{base}_roi{i+1}_10x.png")
                self._export_roi_10x(r.rect, out_path)

        QMessageBox.information(self, "完成", "ROI 导出完成。")

    def _export_roi_10x(self, rect_scene: QRectF, out_path: str):
        reader = self.reader
        if not reader:
            raise RuntimeError("No reader opened")

        x0 = int(rect_scene.left())
        y0 = int(rect_scene.top())
        w0 = int(rect_scene.width())
        h0 = int(rect_scene.height())
        if w0 <= 0 or h0 <= 0:
            raise RuntimeError("ROI 尺寸无效")

        obj0 = reader.objective_power()
        ds_list = [float(d) for d in reader.level_downsamples]
        have_phys = obj0 is not None and obj0 > 0
        if have_phys:
            target_scale = obj0 / 10.0
        else:
            target_scale = ds_list[1] if len(ds_list) > 1 else 1.0

        lvl = int(np.argmin([abs(d - target_scale) for d in ds_list]))
        ds = ds_list[lvl]
        wL = max(1, int(round(w0 / ds)))
        hL = max(1, int(round(h0 / ds)))
        rgba = reader.read_region(lvl, x0, y0, wL, hL)

        if abs(ds - target_scale) / max(target_scale, 1e-6) > 1e-3:
            dst_w = max(1, int(round(w0 / target_scale)))
            dst_h = max(1, int(round(h0 / target_scale)))
            im = Image.fromarray(rgba, mode="RGBA").resize((dst_w, dst_h), resample=Image.BILINEAR)
        else:
            im = Image.fromarray(rgba, mode="RGBA")

        ext = os.path.splitext(out_path)[1].lower()
        if ext in (".tif", ".tiff"):
            im.save(out_path, compression="tiff_lzw")
        else:
            im.save(out_path)

        meta = {
            "bbox_level0": [x0, y0, w0, h0],
            "level": int(lvl),
            "downsample": float(ds),
            "objective": float(obj0) if have_phys else None,
            "target_magnification": "10x",
            "path": getattr(reader, "path", None),
        }
        meta_path = os.path.splitext(out_path)[0] + "_meta.json"
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
