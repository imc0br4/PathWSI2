# ui/pages/wsi_mixins/info_status.py
from __future__ import annotations
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox
from ui.viewer.roi_tools import MODE_EDIT
from ui.viewer.wsi_view import select_level_smart



class InfoStatusMixin:
    def _refresh_info(self):
        if self._is_closing or (not self.reader):
            return
        try:
            sc = self.view.scene()
            if sc is None:
                return
            ppd = self.view._current_ppd()
            ds_list = [float(d) for d in self.reader.level_downsamples]
            lvl = select_level_smart(ds_list, ppd, oversample_limit=1.4)
            ds = ds_list[lvl] if 0 <= lvl < len(ds_list) else 1.0
            obj = self.reader.objective_power()
            if obj and obj > 0:
                mag = obj / max(ppd, 1e-9)
                mag_txt = f"{mag:.1f}×"
            else:
                mag_txt = "N/A"
            self.lbl_info.setText(f"倍率: {mag_txt} | ppd: {ppd:.3f} | level: {lvl} | ds: {ds:.3f}")
        except Exception:
            pass

    def _update_buttons(self, active: bool):
        for b in [self.btn_rotl, self.btn_rotr, self.btn_close, self.btn_overlay]:
            b.setEnabled(active)
        self.btn_roi.setEnabled(active)

        roi_on = self.roi_tool.is_active()
        in_edit = roi_on and (self.roi_tool.mode() == MODE_EDIT)
        has_rois = self.roi_tool.has_rois()
        has_sel = (self.roi_tool.selected_count() > 0)

        self.act_roi_draw.setEnabled(active)
        self.act_roi_edit.setEnabled(active)
        self.act_roi_exit.setEnabled(active and roi_on)
        self.act_roi_clear.setEnabled(active and roi_on and has_rois)
        self.act_roi_save.setEnabled(active and has_rois)
        self.act_roi_del.setEnabled(active and in_edit and has_sel)

    def showEvent(self, e):
        super().showEvent(e)
        # 窗口显示后再调一次，避免某些平台上初值不准
        QTimer.singleShot(0, self._tune_split_sizes)
        self._layout_minimap()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._layout_minimap()

    def _tune_split_sizes(self):
        """根据窗口当前宽度，给左缩略图和右浏览区一个合理的初始占比。"""
        try:
            W = max(800, self.width())
            left = min(320, max(260, int(W * 0.18)))  # 左侧约 18%，限制 260~320
            right = max(400, W - left - 16)
            self._split.setSizes([left, right])
        except Exception:
            pass

