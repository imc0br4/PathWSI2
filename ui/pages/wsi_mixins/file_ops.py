from __future__ import annotations
from typing import Optional
from PySide6.QtCore import QTimer, QRectF, Qt
from PySide6.QtWidgets import QFileDialog, QMessageBox
from wsi.reader import WsiReader
from PySide6.QtGui import QImage, QPixmap, QPainter

_OPEN_FILTER = "Slides/Images (*.svs *.ndpi *.scn *.mrxs *.svslide *.tif *.tiff *.png *.jpg *.jpeg *.bmp)"
class FileOpsMixin:
    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开 WSI/图像", "", _OPEN_FILTER)
        if path:
            self._open_slide(path)

    def on_close_wsi(self):
        """关闭当前 WSI，退出 ROI/编辑，清理叠加与小地图，并复位 UI。"""
        self._is_closing = True
        try:
            # 1) 停止定时器
            for t in ("_info_timer", "_hover_timer", "_flush_timer"):
                obj = getattr(self, t, None)
                try:
                    obj and obj.stop()
                except Exception:
                    pass

            # 2) 退出 ROI/编辑态 + 清理临时图元
            try:
                if hasattr(self, "roi_tool") and self.roi_tool:
                    try:
                        self.roi_tool.clear()
                    except Exception:
                        pass
                    if hasattr(self.roi_tool, "is_active") and self.roi_tool.is_active():
                        self.roi_tool.activate(False)
            except Exception:
                pass

            self._edit_mode = 'off'
            for it_name in ("_edit_rect_item", "_hover_preview_item"):
                it = getattr(self, it_name, None)
                if it:
                    try:
                        sc = self.view.scene() if hasattr(self, "view") else None
                        sc and sc.removeItem(it)
                    except Exception:
                        pass
                setattr(self, it_name, None)
            self._panning = False
            self._pan_last_pos = None

            # 3) 叠加层与状态
            try: self._destroy_overlay_item()
            except Exception: pass
            try: self._clear_overlay_state()
            except Exception: pass

            # 4) 小地图（用 MiniMapMixin 的接口）
            try:
                if hasattr(self, "_hide_minimap"):
                    self._hide_minimap()
            except Exception:
                pass

            # 5) 卸载视图与关闭 reader
            try:
                if hasattr(self, "view") and self.view:
                    try:
                        self.view.unload()   # 如果你的 WsiView 提供了
                    except Exception:
                        sc = self.view.scene()
                        if sc:
                            try: sc.clear()
                            except Exception: pass
            except Exception:
                pass

            try:
                if getattr(self, "reader", None):
                    try: self.reader.close()
                    except Exception: pass
            finally:
                self.reader = None

            # 6) 复位 UI
            try: self._update_buttons(active=False)
            except Exception: pass
            try: self.lbl_info.setText("倍率: - | ppd: - | level: - | ds: -")
            except Exception: pass
            try: self.log.info("WSI closed")
            except Exception: pass

        finally:
            self._is_closing = False


    def _open_slide(self, path: str):
        """打开 WSI，并用 MiniMapMixin 显示右上角缩略图。"""
        if not path:
            return
        try:
            getattr(self, "_info_timer", None) and self._info_timer.stop()
        except Exception:
            pass

        # 清理旧叠加与状态
        try: self._destroy_overlay_item()
        except Exception: pass
        try: self._clear_overlay_state()
        except Exception: pass

        try:
            # 1) 打开并加载到视图
            reader = WsiReader().open(path)
            self.reader = reader
            self.view.load_reader(reader)

            # 首屏自适应
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, self._fit_view_to_scene)

            # 叠加层占位
            self._create_overlay_item()
            self._update_buttons(active=True)
            self._is_closing = False
            try: getattr(self, "_info_timer", None) and self._info_timer.start()
            except Exception: pass
            try: self.log.info(f"Opened WSI: {path}")
            except Exception: pass

            # 2) 小地图：交给 MiniMapMixin
            #    只做方法调用，不直接创建/维护 QLabel 或 pixmap
            try:
                if hasattr(self, "_update_minimap_image"):
                    self._update_minimap_image()   # 读取合适 pyramid level，做底图并显示
                if hasattr(self, "_layout_minimap"):
                    self._layout_minimap()         # 放到右上角
                if hasattr(self, "_update_minimap_rect"):
                    self._update_minimap_rect()    # 画红框
            except Exception:
                # 小地图失败不影响 WSI 打开
                pass

        except Exception as e:
            try: self.log.error("Failed to open WSI", exc_info=True)
            except Exception: pass
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "打开失败", str(e))
    def _fit_view_to_scene(self):
        """把视图适配到场景（若 WsiView 没有公有接口，则用通用做法）"""
        try:
            # 优先尝试你组件可能已有的方法
            if hasattr(self.view, "zoom_to_fit"):
                self.view.zoom_to_fit()
                return
            if hasattr(self.view, "zoom_fit_window"):
                self.view.zoom_fit_window()
                return
            # 兜底：直接对 sceneRect 做 fitInView
            sc = self.view.scene()
            if sc:
                r = sc.itemsBoundingRect() if hasattr(sc, "itemsBoundingRect") else sc.sceneRect()
                # 需要 import QGraphicsView
                from PySide6.QtWidgets import QGraphicsView
                self.view.fitInView(r, Qt.KeepAspectRatio)
                self.view.setRenderHints(self.view.renderHints() |
                                         QPainter.Antialiasing |
                                         QPainter.SmoothPixmapTransform)
                # 拖拽模式回到手型
                self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        except Exception:
            pass

    def _on_open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择包含 WSI 的文件夹")
        if not folder:
            return
        try:
            self.thumb.set_folder(folder)  # ← 替换原来的 load_dir
        except Exception as e:
            QMessageBox.critical(self, "加载缩略图失败", str(e))
            return

        # 可选：自动打开第一个
        if self.thumb.count() > 0:
            self.thumb.setCurrentRow(0)
            first = self.thumb.item(0).data(Qt.UserRole)
            if first:
                self._open_wsi_path(first)

    def _open_wsi_path(self, path: str):
        self._open_slide(path)