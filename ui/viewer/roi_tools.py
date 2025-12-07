# ui/viewer/roi_tools.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from PySide6.QtCore import QObject, Qt, QRectF, QPointF, QEvent, Signal
from PySide6.QtGui import QPen, QBrush, QColor, QCursor
from PySide6.QtWidgets import (
    QGraphicsRectItem, QGraphicsView, QGraphicsScene, QGraphicsItem
)

# 模式常量
MODE_OFF   = 0
MODE_DRAW  = 1  # 框选创建 ROI
MODE_EDIT  = 2  # 选择/编辑 ROI（支持多选）

@dataclass
class RoiRect:
    rect: QRectF
    item: QGraphicsRectItem


class RoiTool(QObject):
    """
    ROI 工具（与 QGraphicsView 协作）：
      - activate(True/False)：进入/退出 ROI 模式；
      - set_mode_draw()：绘制（左键拖框新建；右键拖动画面）；
      - set_mode_edit()：编辑（点选/多选/框选；Delete 删除；右键拖拽平移）；
      - 新快捷键：Shift（轻按）在“绘制/编辑”之间切换；Ctrl 控制多选逻辑：
          · Ctrl+单击命中 ROI：切换该 ROI 的选中状态
          · Ctrl+框选：叠加选择命中的 ROI
          · 无修饰框选：替换选择（仅保留框内 ROI）
      - rois()/clear()/delete_selected()/selected_count()/has_rois()/clear_selection()
      - selectionChanged/modeChanged 信号
    """
    modeChanged = Signal(int)     # 传递 MODE_*
    selectionChanged = Signal()

    def __init__(self, view: QGraphicsView):
        super().__init__(view)
        self.view = view
        self._active = False
        self._mode = MODE_OFF

        # 临时绘制（新建 ROI）
        self._temp_item: Optional[QGraphicsRectItem] = None
        self._press_pos_scene: Optional[QPointF] = None

        # 编辑模式：自定义橡皮筋框选
        self._rb_item: Optional[QGraphicsRectItem] = None
        self._rb_origin: Optional[QPointF] = None

        # 右键平移
        self._panning = False
        self._pan_last_pos = None

        # ROI 列表
        self._rois: List[RoiRect] = []

        # 样式
        self._pen = QPen(QColor(0, 153, 255, 220), 2, Qt.SolidLine)
        self._pen.setCosmetic(True)
        self._brush = QBrush(QColor(0, 153, 255, 60))

        # 橡皮筋样式（编辑态用）
        self._rb_pen = QPen(QColor(0, 180, 255, 220), 1, Qt.DashLine)
        self._rb_pen.setCosmetic(True)
        self._rb_brush = QBrush(QColor(0, 180, 255, 40))

        # 监听场景选中变化
        sc = self._get_scene()
        if sc:
            sc.selectionChanged.connect(self._on_selection_changed)

        # 让视图可接收键盘事件（Delete/Esc/Ctrl+A 等）
        try:
            self.view.setFocusPolicy(Qt.StrongFocus)
        except Exception:
            pass

    # ---------- 场景访问 ----------
    def _get_scene(self) -> Optional[QGraphicsScene]:
        try:
            return self.view.scene()
        except Exception:
            return None

    # ---------- 对外 API ----------
    def activate(self, on: bool):
        if on == self._active:
            return
        self._active = on
        if on:
            # 同时监听 view 和 viewport，这样键鼠事件都能捕获
            self.view.installEventFilter(self)
            self.view.viewport().installEventFilter(self)
            self.view.setFocus()  # 保证 Delete/Esc 能收到
            self.set_mode_draw()  # 默认进入绘制
        else:
            self.view.removeEventFilter(self)
            self.view.viewport().removeEventFilter(self)
            self._end_temp()
            self._end_rubberband()
            self._set_cursor(Qt.ArrowCursor)
            # 退出时恢复视图默认拖拽
            try:
                self.view.setDragMode(QGraphicsView.ScrollHandDrag)
            except Exception:
                pass
            self._mode = MODE_OFF
            self.modeChanged.emit(self._mode)

    def is_active(self) -> bool:
        return self._active

    def mode(self) -> int:
        return self._mode

    def set_mode_draw(self):
        if not self._active:
            self.activate(True)
        self._mode = MODE_DRAW
        self._end_rubberband()
        self._set_items_editable(movable=False)
        self._set_cursor(Qt.CrossCursor)
        try:
            self.view.setDragMode(QGraphicsView.NoDrag)  # 我们接管左键
        except Exception:
            pass
        self.modeChanged.emit(self._mode)

    def set_mode_edit(self):
        if not self._active:
            self.activate(True)
        self._mode = MODE_EDIT
        self._end_temp()
        self._set_items_editable(movable=True)
        self._set_cursor(Qt.ArrowCursor)
        try:
            self.view.setDragMode(QGraphicsView.NoDrag)  # 用自定义橡皮筋，避免与 RubberBandDrag 冲突
        except Exception:
            pass
        self.modeChanged.emit(self._mode)

    def clear(self):
        sc = self._get_scene()
        for r in self._rois:
            try:
                if sc:
                    sc.removeItem(r.item)
            except Exception:
                pass
        self._rois.clear()
        self._end_temp()
        self._end_rubberband()
        self.selectionChanged.emit()

    def rois(self) -> List[RoiRect]:
        """
        返回 ROI 列表；在返回前刷新一次几何（把 item 的当前矩形写回 rect），
        以便导出时使用到移动后的最新位置。
        """
        for r in self._rois:
            try:
                r.rect = r.item.mapRectToScene(r.item.rect())
            except Exception:
                pass
        return list(self._rois)

    def has_rois(self) -> bool:
        return len(self._rois) > 0

    def selected_count(self) -> int:
        return sum(1 for r in self._rois if r.item.isSelected())

    def clear_selection(self):
        for r in self._rois:
            r.item.setSelected(False)
        self.selectionChanged.emit()

    def select_all(self):
        for r in self._rois:
            r.item.setSelected(True)
        self.selectionChanged.emit()

    def delete_selected(self) -> int:
        sc = self._get_scene()
        keep: List[RoiRect] = []
        removed = 0
        for r in self._rois:
            if r.item.isSelected():
                try:
                    if sc:
                        sc.removeItem(r.item)
                except Exception:
                    pass
                removed += 1
            else:
                keep.append(r)
        self._rois = keep
        if removed:
            self.selectionChanged.emit()
        return removed

    # ---------- 事件过滤 ----------
    def eventFilter(self, obj, ev):
        if not self._active:
            return False

        et = ev.type()

        # ---- 键盘 ----
        if et == QEvent.KeyPress:
            key = ev.key()
            mods = ev.modifiers()

            # NEW: 轻按 Shift → 在“绘制/编辑”之间切换（避免长按连发）
            if key == Qt.Key_Shift and not ev.isAutoRepeat():
                if self._mode == MODE_DRAW:
                    self.set_mode_edit()
                elif self._mode == MODE_EDIT:
                    self.set_mode_draw()
                return True

            if key in (Qt.Key_Delete, Qt.Key_Backspace) and self._mode == MODE_EDIT:
                self.delete_selected()
                return True
            if key == Qt.Key_Escape and self._mode == MODE_EDIT:
                self.clear_selection()
                return True
            if (mods & Qt.ControlModifier) and key == Qt.Key_A and self._mode == MODE_EDIT:
                self.select_all()
                return True
            if (mods & Qt.ControlModifier) and key == Qt.Key_D and self._mode == MODE_EDIT:
                self.clear_selection()
                return True

        # ---- 右键平移（两种模式均可用）----
        if et == QEvent.MouseButtonPress and ev.button() == Qt.RightButton:
            self._panning = True
            self._pan_last_pos = ev.pos()
            self._set_cursor(Qt.ClosedHandCursor)
            return True

        if et == QEvent.MouseMove and self._panning:
            pos = ev.pos()
            dx = pos.x() - self._pan_last_pos.x()
            dy = pos.y() - self._pan_last_pos.y()
            h = self.view.horizontalScrollBar()
            v = self.view.verticalScrollBar()
            h.setValue(h.value() - dx)
            v.setValue(v.value() - dy)
            self._pan_last_pos = pos
            return True

        if et == QEvent.MouseButtonRelease and ev.button() == Qt.RightButton:
            self._panning = False
            self._set_cursor(Qt.CrossCursor if self._mode == MODE_DRAW else Qt.ArrowCursor)
            return True

        # ---- 左键逻辑（区分两种模式） ----
        if self._mode == MODE_DRAW:
            # 新建 ROI：左键拖拽
            if et == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
                self._press_pos_scene = self.view.mapToScene(ev.pos())
                self._start_temp(self._press_pos_scene)
                return True

            if et == QEvent.MouseMove and self._press_pos_scene is not None:
                cur = self.view.mapToScene(ev.pos())
                self._update_temp(self._press_pos_scene, cur)
                return True

            if et == QEvent.MouseButtonRelease and ev.button() == Qt.LeftButton and self._press_pos_scene is not None:
                cur = self.view.mapToScene(ev.pos())
                self._finish_temp(self._press_pos_scene, cur)
                self._press_pos_scene = None
                return True

        elif self._mode == MODE_EDIT:
            # 编辑模式：点选 / 橡皮筋框选
            if et == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
                mods = ev.modifiers()
                scene_pos = self.view.mapToScene(ev.pos())
                hit_roi = self._hit_roi(scene_pos)

                if hit_roi:
                    # 命中 ROI
                    if mods & Qt.ControlModifier:
                        # Ctrl: 切换该 ROI 选中态
                        hit_roi.setSelected(not hit_roi.isSelected())
                    else:
                        # 无修饰：仅保留该 ROI 为选中
                        self._select_single(hit_roi)
                    return True
                else:
                    # 点空白：若无 Ctrl，清空选择；随后进入橡皮筋框选
                    if not (mods & Qt.ControlModifier):
                        self.clear_selection()
                    self._rb_origin = scene_pos
                    self._start_rubberband(scene_pos)
                    return True

            if et == QEvent.MouseMove and self._rb_origin is not None:
                cur = self.view.mapToScene(ev.pos())
                self._update_rubberband(self._rb_origin, cur)
                return True

            if et == QEvent.MouseButtonRelease and ev.button() == Qt.LeftButton and self._rb_origin is not None:
                cur = self.view.mapToScene(ev.pos())
                self._apply_rubberband(self._rb_origin, cur, ev.modifiers())
                self._rb_origin = None
                self._end_rubberband()
                return True

        return False

    # ---------- 命中测试 / 选择工具 ----------
    def _hit_roi(self, scene_pos: QPointF) -> Optional[QGraphicsRectItem]:
        sc = self._get_scene()
        if not sc:
            return None
        for it in sc.items(scene_pos):
            # 只认我们创建的 ROI
            if isinstance(it, QGraphicsRectItem) and it.data(0) == "roi":
                return it
        return None

    def _select_single(self, item: QGraphicsRectItem):
        for r in self._rois:
            r.item.setSelected(r.item is item)

    # ---------- 新建 ROI：临时矩形 ----------
    def _start_temp(self, p0: QPointF):
        self._end_temp()
        rect = QRectF(p0, p0).normalized()
        item = QGraphicsRectItem(rect)
        item.setPen(self._pen)
        item.setBrush(self._brush)
        item.setZValue(0)
        item.setData(0, "roi")
        item.setFlag(QGraphicsItem.ItemIsSelectable, True)
        sc = self._get_scene()
        if sc:
            sc.addItem(item)
        self._temp_item = item

    def _update_temp(self, p0: QPointF, p1: QPointF):
        if not self._temp_item:
            return
        rect = QRectF(p0, p1).normalized()
        self._temp_item.setRect(rect)

    def _finish_temp(self, p0: QPointF, p1: QPointF):
        if not self._temp_item:
            return
        rect = QRectF(p0, p1).normalized()
        sc = self._get_scene()
        if rect.width() < 2 or rect.height() < 2:
            if sc:
                sc.removeItem(self._temp_item)
            self._temp_item = None
            return
        self._temp_item.setRect(rect)
        self._temp_item.setZValue(0)
        self._temp_item.setFlag(QGraphicsItem.ItemIsMovable, False)  # 绘制态下不可移动
        self._rois.append(RoiRect(rect=rect, item=self._temp_item))
        self._temp_item = None
        self.selectionChanged.emit()

    def _end_temp(self):
        if self._temp_item:
            try:
                sc = self._get_scene()
                if sc:
                    sc.removeItem(self._temp_item)
            except Exception:
                pass
        self._temp_item = None

    # ---------- 橡皮筋（编辑态用） ----------
    def _start_rubberband(self, p0: QPointF):
        self._end_rubberband()
        rect = QRectF(p0, p0).normalized()
        rb = QGraphicsRectItem(rect)
        rb.setPen(self._rb_pen)
        rb.setBrush(self._rb_brush)
        rb.setZValue(1e6)
        sc = self._get_scene()
        if sc:
            sc.addItem(rb)
        self._rb_item = rb

    def _update_rubberband(self, p0: QPointF, p1: QPointF):
        if self._rb_item:
            self._rb_item.setRect(QRectF(p0, p1).normalized())

    def _apply_rubberband(self, p0: QPointF, p1: QPointF, mods: Qt.KeyboardModifiers):
        rb = QRectF(p0, p1).normalized()
        if rb.width() < 2 or rb.height() < 2:
            return
        # 计算命中
        hits: List[QGraphicsRectItem] = []
        for r in self._rois:
            if r.item.mapRectToScene(r.item.rect()).intersects(rb):
                hits.append(r.item)

        if mods & Qt.ControlModifier:
            # Ctrl + 框选：叠加选择
            for it in hits:
                it.setSelected(True)
        else:
            # 无修饰：替换选择
            for r in self._rois:
                r.item.setSelected(False)
            for it in hits:
                it.setSelected(True)
        self.selectionChanged.emit()

    def _end_rubberband(self):
        if self._rb_item:
            try:
                sc = self._get_scene()
                if sc:
                    sc.removeItem(self._rb_item)
            except Exception:
                pass
        self._rb_item = None

    # ---------- 通用 ----------
    def _set_items_editable(self, movable: bool):
        for r in self._rois:
            r.item.setFlag(QGraphicsItem.ItemIsMovable, movable)
            r.item.setFlag(QGraphicsItem.ItemIsSelectable, True)

    def _set_cursor(self, shape: Qt.CursorShape):
        try:
            self.view.viewport().setCursor(QCursor(shape))
        except Exception:
            pass

    def _on_selection_changed(self):
        self.selectionChanged.emit()
