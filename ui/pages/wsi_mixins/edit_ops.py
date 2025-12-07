# ui/pages/wsi_mixins/edit_ops.py
from __future__ import annotations
from typing import Optional
import numpy as np
import os, json
from PySide6.QtCore import Qt, QEvent, QRectF, QTimer
from PySide6.QtGui import QPen, QColor, QBrush
from PySide6.QtWidgets import QGraphicsRectItem, QFileDialog, QMessageBox, QGraphicsView
from PIL import Image


class EditOpsMixin:

    # ==== 宿主辅助：交互期标记（无感优化的占位实现）====
    def _begin_interaction(self, duration_ms: int = 200):
        """
        一些 mixin/页面会在交互期降低开销；拆分后若宿主未实现，
        给个无害的占位即可防止 AttributeError。
        """
        try:
            if not hasattr(self, "_interact_timer"):
                self._interact_timer = QTimer(self)
                self._interact_timer.setSingleShot(True)
                self._interact_timer.timeout.connect(lambda: None)
            self._interact_timer.start(max(1, int(duration_ms)))
        except Exception:
            pass

    def _mark_dirty(self, x: int, y: int, w: int, h: int):
        """
        把我们改过的 self._overlay_rgba 真正同步到 OverlayItem 上。

        为了兼容你当前的 OverlayItem 实现，这里直接整图 set_rgba，
        再调用 update()/update_region 触发重绘 —— 优先保证“能看到效果”，
        性能以后再优化。
        """
        try:
            if getattr(self, "overlay_item", None) is None:
                return
            if getattr(self, "_overlay_rgba", None) is None:
                return

            # 1）先把整张 numpy 覆盖回去，确保 QImage 与 _overlay_rgba 一致
            if hasattr(self.overlay_item, "set_rgba"):
                self.overlay_item.set_rgba(self._overlay_rgba)

            # 2）再尝试局部刷新（有就用，没有就整图 update）
            try:
                if hasattr(self.overlay_item, "update_region"):
                    self.overlay_item.update_region(x, y, w, h)
                else:
                    # 没有局部刷新接口就整图刷新
                    self.overlay_item.update()
            except Exception:
                # 刷新失败就退化成整图 update
                try:
                    self.overlay_item.update()
                except Exception:
                    pass

        except Exception:
            # 任何异常都不抛出去，避免影响交互
            try:
                if getattr(self, "overlay_item", None) is not None and hasattr(self.overlay_item, "set_rgba") and getattr(self, "_overlay_rgba", None) is not None:
                    self.overlay_item.set_rgba(self._overlay_rgba)
                    try:
                        self.overlay_item.update()
                    except Exception:
                        pass
            except Exception:
                pass

    def _set_edit_mode(self, mode: Optional[str]):
        mode = (mode or 'off').lower()
        if mode not in ('off', 'add', 'erase'):
            mode = 'off'
        if getattr(self, "_edit_mode", 'off') == mode:
            return

        # 必须有 overlay 才能进编辑态
        if mode != 'off' and getattr(self, "_overlay_rgba", None) is None:
            QMessageBox.information(self, "提示", "请先加载一张叠加图（overlay）。")
            return

        # 首次进入编辑态时做备份，便于撤销
        if mode != 'off' and getattr(self, "_grid_backup", None) is None and getattr(self, "_overlay_rgba", None) is not None:
            self._grid_backup = self._overlay_rgba.copy()

        self._edit_mode = mode

        # 视图拖拽/光标
        try:
            self.view.setDragMode(QGraphicsView.NoDrag if mode != 'off' else QGraphicsView.ScrollHandDrag)
        except Exception:
            pass

        try:
            if mode == 'add':
                self.view.viewport().setCursor(Qt.CrossCursor)
            elif mode == 'erase':
                self.view.viewport().setCursor(Qt.PointingHandCursor)
            else:
                self.view.viewport().unsetCursor()
        except Exception:
            pass

        # 动作可用性
        for act, ok in [
            (getattr(self, "act_edit_add", None),   mode != 'add'),
            (getattr(self, "act_edit_erase", None), mode != 'erase'),
            (getattr(self, "act_edit_undo", None),  mode != 'off'),
            (getattr(self, "act_edit_save", None),  mode != 'off'),
            (getattr(self, "act_edit_exit", None),  mode != 'off'),
        ]:
            if act: act.setEnabled(ok)

        self._hover_preview_enabled = (mode == 'erase')

        if mode == 'off':
            # 清除编辑预览（如果宿主实现了该方法）
            if hasattr(self, "_remove_edit_preview"):
                try:
                    self._remove_edit_preview()
                except Exception:
                    pass
            self._edit_origin_scene = None

    def _cancel_overlay_edits(self):
        if getattr(self, "_grid_backup", None) is not None and getattr(self, "_overlay_rgba", None) is not None:
            # 恢复备份
            self._overlay_rgba[...] = self._grid_backup
            try:
                if getattr(self, "overlay_item", None):
                    # 直接整图刷新一次，保证能看到变化
                    self.overlay_item.set_rgba(self._overlay_rgba)
                    try:
                        self.overlay_item.update()
                    except Exception:
                        pass
            except Exception:
                pass
        self._grid_backup = None
        if hasattr(self, "_remove_edit_preview"):
            try:
                self._remove_edit_preview()
            except Exception:
                pass

    def _save_overlay_edits(self):
        if getattr(self, "_overlay_rgba", None) is None:
            QMessageBox.information(self, "提示", "没有可保存的 overlay。")
            return

        # -------- 1. 准备输出路径 --------
        export_cfg = self.app_cfg.get("export", {}) if isinstance(getattr(self, "app_cfg", {}), dict) else {}
        default_dir = os.path.abspath(export_cfg.get("default_dir", "./exports"))
        os.makedirs(default_dir, exist_ok=True)

        slide = getattr(self, "reader", None)
        base_name = os.path.splitext(os.path.basename(slide.path if slide else "slide"))[0]
        out_default = os.path.join(default_dir, f"{base_name}_overlay_edited.png")

        out_path, _ = QFileDialog.getSaveFileName(
            self, "保存修正后的 Overlay", out_default, "PNG (*.png)"
        )
        if not out_path:
            return

        # -------- 2. 写 PNG 图像 --------
        Image.fromarray(self._overlay_rgba, mode="RGBA").save(out_path, optimize=False, compress_level=1)

        # -------- 3. 构造 / 合并 meta --------
        # 当前 overlay 在场景中的位置信息
        ox, oy = getattr(self, "_overlay_pos", (0, 0))
        ds = float(getattr(self, "_overlay_ds", 1.0))  # 通常 = 该 level 的 downsample
        H_ov, W_ov = self._overlay_rgba.shape[0], self._overlay_rgba.shape[1]

        # 在 level0 下的宽高（像素）
        w0 = int(round(W_ov * ds))
        h0 = int(round(H_ov * ds))

        # 原始 meta（分类时生成的），里面可能已经有 bbox_level / patch_size_level / patch_grid 等
        meta_src = getattr(self, "_overlay_meta", {}) or {}

        # 从原 meta 拷贝一份，尽量保留所有字段
        meta = dict(meta_src)

        # level：如果原 meta 没写 level，就默认 0
        level = int(meta_src.get("level", 0))

        # 3.1 重新写 bbox_level0：以当前 overlay 的位置 + 尺寸为准
        bbox_level0 = [int(ox), int(oy), int(w0), int(h0)]
        meta["bbox_level0"] = bbox_level0

        # 3.2 如果原来没有 bbox_level，则根据 downsample 反算一个 level 坐标下的 bbox
        try:
            ds_for_level = float(meta_src.get("downsample", ds)) or ds
            if ds_for_level > 0 and "bbox_level" not in meta:
                meta["bbox_level"] = [
                    int(round(bbox_level0[0] / ds_for_level)),
                    int(round(bbox_level0[1] / ds_for_level)),
                    int(round(bbox_level0[2] / ds_for_level)),
                    int(round(bbox_level0[3] / ds_for_level)),
                ]
        except Exception:
            # 出问题就算了，不强求 bbox_level，一般 bbox_level0 足够
            pass

        # 3.3 基本字段：目标、层级、下采样、路径
        meta["target"] = "classification_overlay_edited"
        meta["level"] = level
        # downsample：优先保留原来的（分类时写入的），没有再用当前 ds
        meta["downsample"] = float(meta_src.get("downsample", ds))

        meta["path"] = slide.path if slide is not None else meta_src.get("path")

        # 3.4 网格边长（在 overlay 像素坐标系下）
        # 供擦除/添加以“方格”为单位对齐
        meta["grid_tile_px_on_overlay"] = int(getattr(self, "_grid_tile_px", 64))

        # 3.5 说明文字（如果原来没写 note 就补一个）
        meta.setdefault(
            "note",
            "医生在 WSI 中以方格为单位修正后的 overlay"
        )

        # 注意：这里**不会**去改动你原先的 patch_size_level / stride_level / patch_grid 等字段，
        # 它们都会保留在 meta 里，供后续检测/数据集脚本继续使用。

        # -------- 4. 写 meta.json --------
        meta_path = os.path.splitext(out_path)[0] + "_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        QMessageBox.information(self, "完成", f"已保存：\n{out_path}\n{meta_path}")


    def _remove_edit_preview(self):
        # 删添加矩形预览
        if getattr(self, "_edit_rect_item", None):
            try:
                sc = self.view.scene()
                if sc:
                    sc.removeItem(self._edit_rect_item)
            except Exception:
                pass
        self._edit_rect_item = None
        # 删橡皮 hover 预览
        if getattr(self, "_hover_preview_item", None):
            try:
                sc = self.view.scene()
                if sc:
                    sc.removeItem(self._hover_preview_item)
            except Exception:
                pass
        self._hover_preview_item = None

    def _eraser_rect_overlay_at_scene(self, scene_x: float, scene_y: float):
        """返回 (x0_ov,y0_ov,w_ov,h_ov) —— 以 overlay 像素为单位的橡皮框."""
        if self._overlay_rgba is None:
            return None
        ds = float(self._overlay_ds) if self._overlay_ds else 1.0
        ox, oy = self._overlay_pos
        x_ov = int((scene_x - ox) / ds)
        y_ov = int((scene_y - oy) / ds)
        H, W = self._overlay_rgba.shape[0], self._overlay_rgba.shape[1]
        if x_ov < 0 or y_ov < 0 or x_ov >= W or y_ov >= H:
            return None
        unit = max(1, int(self._grid_tile_px))
        s = unit * max(1, int(self._eraser_units))
        col = x_ov // s
        row = y_ov // s
        x0 = col * s
        y0 = row * s
        w = min(s, W - x0)
        h = min(s, H - y0)
        if w <= 0 or h <= 0:
            return None
        return x0, y0, w, h

    def _schedule_hover_preview(self, scene_pt):
        """移动时仅记录点位并启动/重启定时器；真正的绘制放到 _on_hover_timer。"""
        self._hover_pending_pt = scene_pt
        if self._hover_preview_item:
            self._hover_preview_item.setVisible(False)  # 移动中先隐藏，避免晃动
        self._hover_timer.start()

    def _on_hover_timer(self):
        """延迟触发：根据最近的悬停点画出半透明弱提示的预览块。"""
        if self._edit_mode != 'erase' or self._overlay_rgba is None or self._hover_pending_pt is None:
            return
        rect = self._eraser_rect_overlay_at_scene(self._hover_pending_pt.x(), self._hover_pending_pt.y())
        if not rect:
            if self._hover_preview_item:
                self._hover_preview_item.setVisible(False)
            return

        x0, y0, w, h = rect
        ds = float(self._overlay_ds) if self._overlay_ds else 1.0
        ox, oy = self._overlay_pos
        xr, yr = ox + x0 * ds, oy + y0 * ds
        wr, hr = w * ds, h * ds

        sc = self.view.scene()
        if sc is None:
            return

        if self._hover_preview_item is None:
            it = QGraphicsRectItem(QRectF(xr, yr, wr, hr))
            # 更弱的半透明提示（避免头晕）
            pen = QPen(QColor(0, 0, 0, 60), 1)
            pen.setCosmetic(True)
            it.setPen(pen)
            it.setBrush(QBrush(QColor(0, 0, 0, 40), Qt.Dense4Pattern))  # 半透明 + 稀疏点阵
            it.setZValue(1e6)
            sc.addItem(it)
            self._hover_preview_item = it
        else:
            self._hover_preview_item.setRect(QRectF(xr, yr, wr, hr))

        self._hover_preview_item.setVisible(True)

    def eventFilter(self, obj, ev):
        # 只处理 view / viewport
        if obj not in (self.view, self.view.viewport()):
            return super().eventFilter(obj, ev)

        # ========= 会触发“视图变换”的事件：先标记为交互中 =========
        # 滚轮缩放（交给 WsiView 自己处理，因此这里不拦截，只做节流提示）
        if ev.type() == QEvent.Wheel:
            self._begin_interaction()
            # 若实现了缩略图，下一轮事件循环更新一次红框
            try:
                QTimer.singleShot(0, getattr(self, "_update_minimap_rect"))
            except Exception:
                pass
            # 不拦截，继续交给原逻辑
            return super().eventFilter(obj, ev)

        # 视口尺寸变化（窗口 resize / 分割条拖动）
        if ev.type() == QEvent.Resize:
            self._begin_interaction()
            try:
                # 有缩略图就重排 + 刷新红框
                if hasattr(self, "_layout_minimap"):
                    self._layout_minimap()
                if hasattr(self, "_update_minimap_rect"):
                    self._update_minimap_rect()
            except Exception:
                pass
            return super().eventFilter(obj, ev)

        # ========= 编辑态下的拦截 =========
        # 编辑态禁用双击回首屏；退出编辑后双击仍可用
        if self._edit_mode != 'off' and ev.type() == QEvent.MouseButtonDblClick:
            return True

        # 键盘：Shift 切换 add/erase；[ / ] 调橡皮大小
        if ev.type() == QEvent.KeyPress and not ev.isAutoRepeat():
            if ev.key() == Qt.Key_Shift:
                if self._edit_mode in ('add', 'erase'):
                    self._toggle_edit_mode()
                    return True
            if self._edit_mode == 'erase':
                if ev.key() == Qt.Key_BracketLeft:  # [
                    self._eraser_units = max(1, int(self._eraser_units) // 2)
                    return True
                if ev.key() == Qt.Key_BracketRight:  # ]
                    self._eraser_units = min(16, int(self._eraser_units) * 2)
                    return True

        # 非编辑态：其余事件放行
        if self._edit_mode == 'off':
            return super().eventFilter(obj, ev)

        # ========= 右键平移（编辑态内有效） =========
        if ev.type() == QEvent.MouseButtonPress and ev.button() == Qt.RightButton:
            self._begin_interaction()  # ← 新增：进入交互期（禁平滑+最小重绘）
            self._panning = True
            self._pan_last_pos = ev.pos()
            self.view.viewport().setCursor(Qt.ClosedHandCursor)
            return True

        if ev.type() == QEvent.MouseMove and self._panning:
            self._begin_interaction()  # ← 新增：持续交互（重置计时器）
            pos = ev.pos()
            dx = pos.x() - self._pan_last_pos.x()
            dy = pos.y() - self._pan_last_pos.y()
            h = self.view.horizontalScrollBar()
            v = self.view.verticalScrollBar()
            h.setValue(h.value() - dx)
            v.setValue(v.value() - dy)
            self._pan_last_pos = pos
            # 缩略图红框跟随
            try:
                if hasattr(self, "_update_minimap_rect"):
                    self._update_minimap_rect()
            except Exception:
                pass
            return True

        if ev.type() == QEvent.MouseButtonRelease and ev.button() == Qt.RightButton and self._panning:
            self._begin_interaction()  # ← 新增：结束交互倒计时，稍后恢复高质量
            self._panning = False
            # 恢复光标
            try:
                if self._edit_mode == 'add':
                    self.view.viewport().setCursor(Qt.CrossCursor)
                elif self._edit_mode == 'erase':
                    self.view.viewport().setCursor(Qt.PointingHandCursor)
                else:
                    self.view.viewport().unsetCursor()
            except Exception:
                pass
            # 缩略图红框最后同步一次
            try:
                if hasattr(self, "_update_minimap_rect"):
                    self._update_minimap_rect()
            except Exception:
                pass
            return True

        # ---- 工具：事件坐标 -> scene 坐标（兼容 position/pos）----
        def _scene_pos(_ev):
            try:
                pos_attr = getattr(_ev, "position", None)
                if callable(pos_attr):
                    p = pos_attr()
                    return self.view.mapToScene(int(p.x()), int(p.y()))
                return self.view.mapToScene(_ev.pos())
            except Exception:
                return None

        # ============== 添加矩形（左键拖拽） ==============
        if self._edit_mode == 'add':
            if ev.type() == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
                pt = _scene_pos(ev)
                if pt is None:
                    return True
                self._edit_origin_scene = pt
                # 清理旧预览并建新框
                if getattr(self, "_edit_rect_item", None):
                    sc = self.view.scene()
                    try:
                        if sc:
                            sc.removeItem(self._edit_rect_item)
                    except Exception:
                        pass
                self._edit_rect_item = None
                it = QGraphicsRectItem(QRectF(pt, pt).normalized())
                pen = QPen(QColor(255, 0, 0, 200), 1, Qt.DashLine)
                pen.setCosmetic(True)
                it.setPen(pen)
                it.setBrush(QColor(255, 0, 0, 40))
                it.setZValue(1e6)
                sc = self.view.scene()
                sc.addItem(it) if sc else None
                self._edit_rect_item = it
                return True

            if ev.type() == QEvent.MouseMove and self._edit_origin_scene is not None and self._edit_rect_item is not None:
                cur = _scene_pos(ev)
                if cur is not None:
                    self._edit_rect_item.setRect(QRectF(self._edit_origin_scene, cur).normalized())
                return True

            if ev.type() == QEvent.MouseButtonRelease and ev.button() == Qt.LeftButton and self._edit_origin_scene is not None:
                cur = _scene_pos(ev) or self._edit_origin_scene
                # 填充：scene 矩形 -> overlay 矩形（硬边）
                self._fill_overlay_scene_rect(self._edit_origin_scene, cur)
                # 清预览
                sc = self.view.scene()
                try:
                    if sc and self._edit_rect_item:
                        sc.removeItem(self._edit_rect_item)
                except Exception:
                    pass
                self._edit_rect_item = None
                self._edit_origin_scene = None
                return True

        # ============== 橡皮擦（单格 + 矩形拖拽） ==============
        if self._edit_mode == 'erase':
            # Hover：0.15s 延迟弱提示（单格）
            if ev.type() == QEvent.MouseMove and self._erase_drag_origin_scene is None:
                if self._hover_preview_enabled:  # ← 新增开关
                    pt = _scene_pos(ev)
                    if pt:
                        self._hover_pending_pt = pt
                        if self._hover_preview_item:
                            self._hover_preview_item.setVisible(False)
                        self._hover_timer.start()
                    return True

            # 左键按下：开始拖拽矩形擦除；若不移动则退化为单格擦
            if ev.type() == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
                self._erase_drag_origin_scene = _scene_pos(ev)
                # 正式拖拽前隐藏 hover 方块，避免混乱
                if self._hover_preview_item:
                    self._hover_preview_item.setVisible(False)
                return True

            # 拖拽中：显示矩形预览（淡色）
            if ev.type() == QEvent.MouseMove and self._erase_drag_origin_scene is not None:
                cur = _scene_pos(ev)
                if cur is None:
                    return True
                r = QRectF(self._erase_drag_origin_scene, cur).normalized()
                sc = self.view.scene()
                if sc:
                    if self._hover_preview_item is None:
                        it = QGraphicsRectItem(r)
                        pen = QPen(QColor(0, 0, 0, 80), 1)
                        pen.setCosmetic(True)
                        it.setPen(pen)
                        it.setBrush(QColor(0, 0, 0, 40))
                        it.setZValue(1e6)
                        sc.addItem(it)
                        self._hover_preview_item = it
                    else:
                        self._hover_preview_item.setRect(r)
                        self._hover_preview_item.setVisible(True)
                return True

            # 鼠标抬起：如果拖拽距离小 -> 单格擦；否则 -> 矩形范围擦
            if ev.type() == QEvent.MouseButtonRelease and ev.button() == Qt.LeftButton and self._erase_drag_origin_scene is not None:
                cur = _scene_pos(ev) or self._erase_drag_origin_scene
                p0 = self._erase_drag_origin_scene
                p1 = cur
                self._erase_drag_origin_scene = None

                # 删除临时预览
                sc = self.view.scene()
                try:
                    if sc and self._hover_preview_item:
                        self._hover_preview_item.setVisible(False)
                except Exception:
                    pass

                # 判断是否当作点击（单格）
                click_like = (abs(p1.x() - p0.x()) < 4) and (abs(p1.y() - p0.y()) < 4)
                if click_like:
                    # 单块擦除
                    self._erase_single_tile_at_scene(p0.x(), p0.y())
                    return True

                # 框选整块擦除（对齐网格）
                self._erase_scene_rect_snap_to_grid(p0, p1)
                return True

        return super().eventFilter(obj, ev)

    def _fill_overlay_scene_rect(self, p0_scene, p1_scene):
        """把场景坐标的矩形映射到 overlay 像素并用填充色着色。"""
        if self._overlay_rgba is None:
            return
        import math
        ds = float(self._overlay_ds) if self._overlay_ds else 1.0
        ox, oy = self._overlay_pos

        x0s = min(p0_scene.x(), p1_scene.x())
        x1s = max(p0_scene.x(), p1_scene.x())
        y0s = min(p0_scene.y(), p1_scene.y())
        y1s = max(p0_scene.y(), p1_scene.y())

        x0 = int(math.floor((x0s - ox) / ds))
        y0 = int(math.floor((y0s - oy) / ds))
        x1 = int(math.ceil((x1s - ox) / ds))
        y1 = int(math.ceil((y1s - oy) / ds))

        H, W = self._overlay_rgba.shape[0], self._overlay_rgba.shape[1]
        x0 = max(0, min(W, x0))
        x1 = max(0, min(W, x1))
        y0 = max(0, min(H, y0))
        y1 = max(0, min(H, y1))
        if x1 <= x0 or y1 <= y0:
            return

        self._overlay_rgba[y0:y1, x0:x1, :] = self._grid_fill_rgba
        if getattr(self, "overlay_item", None):
            self._mark_dirty(x0, y0, x1 - x0, y1 - y0)

    def _erase_single_tile_at_scene(self, sx: float, sy: float):
        """删掉鼠标所在的“一个单位格”，格大小来自 meta 推断（stride/grid）。"""
        if self._overlay_rgba is None:
            return
        import math
        ds = float(self._overlay_ds) if self._overlay_ds else 1.0
        ox, oy = self._overlay_pos

        x_ov = int(math.floor((sx - ox) / ds))
        y_ov = int(math.floor((sy - oy) / ds))

        H, W = self._overlay_rgba.shape[0], self._overlay_rgba.shape[1]
        if x_ov < 0 or y_ov < 0 or x_ov >= W or y_ov >= H:
            return

        s = int(max(1, self._grid_tile_px))  # ← 由 meta/默认确定
        col = x_ov // s
        row = y_ov // s
        x0 = col * s
        y0 = row * s
        x1 = min(W, x0 + s)
        y1 = min(H, y0 + s)
        if x1 <= x0 or y1 <= y0:
            return

        blk = self._overlay_rgba[y0:y1, x0:x1]
        blk[..., :3] = 0
        blk[..., 3] = 0  # 只清 alpha
        if getattr(self, "overlay_item", None):
            self._mark_dirty(x0, y0, x1 - x0, y1 - y0)

    def _erase_scene_rect_snap_to_grid(self, p0_scene, p1_scene):
        """将 scene 矩形投到 overlay，并对齐网格整块清除。"""
        if self._overlay_rgba is None:
            return
        import math
        ds = float(self._overlay_ds) if self._overlay_ds else 1.0
        ox, oy = self._overlay_pos
        s = int(max(1, self._grid_tile_px))  # 网格边长(overlay像素)

        # scene -> overlay 像素范围
        x0s, y0s = min(p0_scene.x(), p1_scene.x()), min(p0_scene.y(), p1_scene.y())
        x1s, y1s = max(p0_scene.x(), p1_scene.x()), max(p0_scene.y(), p1_scene.y())
        x0 = int(math.floor((x0s - ox) / ds))
        y0 = int(math.floor((y0s - oy) / ds))
        x1 = int(math.ceil((x1s - ox) / ds))
        y1 = int(math.ceil((y1s - oy) / ds))

        H, W = self._overlay_rgba.shape[:2]
        x0 = max(0, min(W, x0))
        x1 = max(0, min(W, x1))
        y0 = max(0, min(H, y0))
        y1 = max(0, min(H, y1))
        if x1 <= x0 or y1 <= y0:
            return

        # 对齐到网格：求覆盖到的格子范围
        col0 = x0 // s
        col1 = (x1 + s - 1) // s
        row0 = y0 // s
        row1 = (y1 + s - 1) // s

        # 聚合脏区
        dx0, dy0, dx1, dy1 = W, H, 0, 0
        for r in range(row0, row1):
            yy0 = r * s
            yy1 = min(H, yy0 + s)
            for c in range(col0, col1):
                xx0 = c * s
                xx1 = min(W, xx0 + s)
                block = self._overlay_rgba[yy0:yy1, xx0:xx1]
                block[..., :3] = 0
                block[..., 3] = 0
                if xx0 < dx0:
                    dx0 = xx0
                if yy0 < dy0:
                    dy0 = yy0
                if xx1 > dx1:
                    dx1 = xx1
                if yy1 > dy1:
                    dy1 = yy1

        if dx1 > dx0 and dy1 > dy0:
            self._mark_dirty(dx0, dy0, dx1 - dx0, dy1 - dy0)

    def _toggle_edit_mode(self):
        """在 add / erase 之间持久切换；在 off 时不响应。"""
        if self._edit_mode in ('add', 'erase'):
            self._set_edit_mode('erase' if self._edit_mode == 'add' else 'add')
