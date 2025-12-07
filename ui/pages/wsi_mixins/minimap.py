# ui/pages/wsi_mixins/minimap.py
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from PySide6.QtCore import Qt, QRectF, QPointF, QSize, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QPixmap, QImage
from PySide6.QtWidgets import QWidget, QFrame, QLabel


# ------------------------------
# 小部件：绘制缩略图 + 可拖拽的视窗红框
# ------------------------------
class _MiniMapCanvas(QWidget):
    requestCenterOn = Signal(float, float)  # cx, cy in scene coords
    dragStateChanged = Signal(bool)  # True when dragging red box

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._pix: Optional[QPixmap] = None
        self._scene_w = 1.0
        self._scene_h = 1.0

        self._scale = 1.0
        self._ox = 0.0
        self._oy = 0.0

        self._view_rect_scene = QRectF(0, 0, 1, 1)
        self._view_rect_canvas = QRectF(0, 0, 1, 1)

        self._dragging = False
        self._drag_inside_offset = QPointF(0, 0)  # 鼠标相对红框中心的偏移（canvas 坐标）
        self._bg = QColor(245, 245, 245)

    # ---- public APIs ----
    def set_scene_size(self, w: float, h: float):
        w = max(1.0, float(w)); h = max(1.0, float(h))
        if abs(w - self._scene_w) > 1e-3 or abs(h - self._scene_h) > 1e-3:
            self._scene_w, self._scene_h = w, h
            self._recalc_fit()
            self.update()

    def set_pixmap(self, pix: Optional[QPixmap]):
        self._pix = pix
        self._recalc_fit()
        self.update()

    def set_view_rect_scene(self, r: QRectF):
        # r：当前主视口在 scene 的可见矩形
        w = max(0.0, float(r.width()))
        h = max(0.0, float(r.height()))
        w = min(w if w > 0 else self._scene_w, self._scene_w)
        h = min(h if h > 0 else self._scene_h, self._scene_h)

        max_left = max(0.0, self._scene_w - w)
        max_top = max(0.0, self._scene_h - h)
        left = max(0.0, min(max_left, float(r.left())))
        top = max(0.0, min(max_top, float(r.top())))

        self._view_rect_scene = QRectF(left, top, w, h)
        self._view_rect_canvas = self._scene_to_canvas_rect(self._view_rect_scene)
        self.update()

    # ---- helpers ----
    def _recalc_fit(self):
        W = max(1, self.width()); H = max(1, self.height())
        sw, sh = float(self._scene_w), float(self._scene_h)
        if sw <= 0 or sh <= 0:
            self._scale, self._ox, self._oy = 1.0, 0.0, 0.0
            return
        sx = W / sw; sy = H / sh
        self._scale = min(sx, sy)
        draw_w = sw * self._scale
        draw_h = sh * self._scale
        self._ox = (W - draw_w) * 0.5
        self._oy = (H - draw_h) * 0.5
        # 同步一次红框
        self._view_rect_canvas = self._scene_to_canvas_rect(self._view_rect_scene)

    def resizeEvent(self, _e):
        self._recalc_fit()
        super().resizeEvent(_e)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), self._bg)

        # 背景缩略图
        if self._pix is not None and not self._pix.isNull():
            dst = QRectF(self._ox, self._oy, self._scene_w * self._scale, self._scene_h * self._scale)
            p.setRenderHint(QPainter.SmoothPixmapTransform, True)
            p.drawPixmap(dst, self._pix, self._pix.rect())

        # 外框
        p.setPen(QPen(QColor(0, 0, 0, 80), 1))
        p.drawRect(QRectF(self._ox, self._oy, self._scene_w * self._scale, self._scene_h * self._scale))

        # 红框（视口）
        r = self._view_rect_canvas
        if r.width() > 0 and r.height() > 0:
            pen = QPen(QColor(220, 50, 47), 2)  # 红色
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(r)

            # 半透明遮罩（红框外）
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 50))
            full = QRectF(self._ox, self._oy, self._scene_w * self._scale, self._scene_h * self._scale)
            # 上
            p.drawRect(QRectF(full.left(), full.top(), full.width(), r.top() - full.top()))
            # 下
            p.drawRect(QRectF(full.left(), r.bottom(), full.width(), full.bottom() - r.bottom()))
            # 左
            p.drawRect(QRectF(full.left(), r.top(), r.left() - full.left(), r.height()))
            # 右
            p.drawRect(QRectF(r.right(), r.top(), full.right() - r.right(), r.height()))

        p.end()

    # ---- mouse interaction ----
    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return super().mousePressEvent(e)
        pos = e.position() if hasattr(e, "position") else e.pos()
        pos = pos if isinstance(pos, QPointF) else QPointF(float(pos.x()), float(pos.y()))
        if self._view_rect_canvas.contains(pos):
            # 在红框内 -> 进入拖动
            self._dragging = True
            c = self._view_rect_canvas.center()
            self._drag_inside_offset = pos - c
            try:
                self.dragStateChanged.emit(True)
            except Exception:
                pass
        else:
            # 点击到其它地方 -> 直接居中
            cx, cy = self._canvas_to_scene_point(pos)
            self._set_view_center_scene(cx, cy)
            self.requestCenterOn.emit(cx, cy)
        e.accept()

    def mouseMoveEvent(self, e):
        if not self._dragging:
            return super().mouseMoveEvent(e)
        pos = e.position() if hasattr(e, "position") else e.pos()
        pos = pos if isinstance(pos, QPointF) else QPointF(float(pos.x()), float(pos.y()))
        # 把红框中心移动到鼠标位置减去初始偏移
        c_canvas = pos - self._drag_inside_offset
        cx, cy = self._canvas_to_scene_point(c_canvas)
        self._set_view_center_scene(cx, cy)
        self.requestCenterOn.emit(cx, cy)
        e.accept()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            if self._dragging:
                self._dragging = False
                try:
                    self.dragStateChanged.emit(False)
                except Exception:
                    pass
            e.accept()
            return
        super().mouseReleaseEvent(e)

    # ---- coordinate transforms ----
    def _scene_to_canvas_rect(self, r: QRectF) -> QRectF:
        x = self._ox + r.left() * self._scale
        y = self._oy + r.top() * self._scale
        w = r.width() * self._scale
        h = r.height() * self._scale
        return QRectF(x, y, w, h)

    def _canvas_to_scene_point(self, p: QPointF) -> Tuple[float, float]:
        # 把点钳到可绘制区域内部
        x = max(self._ox, min(self._ox + self._scene_w * self._scale, p.x()))
        y = max(self._oy, min(self._oy + self._scene_h * self._scale, p.y()))
        sx = (x - self._ox) / max(1e-6, self._scale)
        sy = (y - self._oy) / max(1e-6, self._scale)

        # 若已知当前视口大小，则把中心钳在[half_w, W-half_w]
        half_w = self._view_rect_scene.width() * 0.5
        half_h = self._view_rect_scene.height() * 0.5
        sx = max(half_w, min(self._scene_w - half_w, sx))
        sy = max(half_h, min(self._scene_h - half_h, sy))
        return float(sx), float(sy)

    def _set_view_center_scene(self, cx: float, cy: float):
        w = max(0.0, self._view_rect_scene.width())
        h = max(0.0, self._view_rect_scene.height())
        if w <= 0 or h <= 0:
            return
        rect = QRectF(cx - w * 0.5, cy - h * 0.5, w, h)
        self.set_view_rect_scene(rect)


# ------------------------------
# Mixin：提供初始化 / 布局 / 同步 / 交互
# ------------------------------
class MiniMapMixin:
    def _init_minimap_widgets(self):
        # 在视图右上角做一个悬浮容器
        self._mini_cont = QFrame(self.view)
        self._mini_cont.setObjectName("MiniMapContainer")
        self._mini_cont.setStyleSheet("""
        QFrame#MiniMapContainer { background: rgba(255,255,255,235); border:1px solid #d0d0d0; border-radius:6px; }
        """)
        self._mini_cont.setFrameShape(QFrame.StyledPanel)
        self._mini_cont.setVisible(False)

        self._mini_canvas = _MiniMapCanvas(self._mini_cont)
        self._mini_canvas.requestCenterOn.connect(self._minimap_center_on_scene)
        self._mini_canvas.dragStateChanged.connect(self._on_minimap_drag_state)

        self._mini_hint = QLabel("minimap", self._mini_cont)
        self._mini_hint.setStyleSheet("color:#777; font-size:11px;")
        self._mini_hint.move(6, 4)

        # 默认尺寸；可在 app_cfg['viewer'] 里覆盖 minimap_w / minimap_h
        cfg = self.app_cfg.get("viewer", {}) if hasattr(self, "app_cfg") else {}
        self._mini_w = int(cfg.get("minimap_w", 200))
        self._mini_h = int(cfg.get("minimap_h", 200))
        self._mini_margin = int(cfg.get("minimap_margin", 10))
        preview_side = int(cfg.get("minimap_max_side", 1024))
        fetch_side = int(cfg.get("minimap_fetch_side", max(preview_side, preview_side * 4)))
        fetch_side = max(preview_side, min(fetch_side, 8192))
        self._minimap_max_side = max(128, preview_side)
        self._minimap_fetch_side = max(self._minimap_max_side, fetch_side)

        self._layout_minimap()

        # 记录一次以便在首次加载后立即显示
        self._minimap_ready = False
        self._minimap_pix: Optional[QPixmap] = None
        self._minimap_drag_active = False

    # 放到右上角；随窗口变化
    def _layout_minimap(self):
        if not hasattr(self, "_mini_cont"):  # 未初始化
            return
        W = max(80, int(self._mini_w))
        H = max(80, int(self._mini_h))

        self._mini_cont.setGeometry(
            self.view.width() - W - self._mini_margin,
            self._mini_margin,
            W, H
        )
        # 让 canvas 充满容器
        self._mini_canvas.setGeometry(0, 0, W, H)

    # ---------- 缩略图生成与刷新 ----------
    def _minimap_make_preview(self) -> Optional[QPixmap]:
        """尽量从 reader 拿缩略图；若没有 thumbnail API，就用较清晰的 pyramid 层生成预览。"""
        r = getattr(self, "reader", None) or getattr(getattr(self, "view", None), "reader", None)
        if not r:
            return None

        preview_side = int(getattr(self, "_minimap_max_side", 1024))
        fetch_side = int(getattr(self, "_minimap_fetch_side", max(preview_side, preview_side * 4)))
        preview_side = max(64, preview_side)
        fetch_side = max(preview_side, fetch_side)

        def _to_qpix(img) -> Optional[QPixmap]:
            if isinstance(img, QPixmap):
                return img
            if isinstance(img, QImage):
                return QPixmap.fromImage(img)
            qimg = _to_qimage(img)
            if isinstance(qimg, QImage):
                return QPixmap.fromImage(qimg)
            return None

        def _normalize_pix(pix: Optional[QPixmap]) -> Optional[QPixmap]:
            if not pix or pix.isNull():
                return None
            if max(pix.width(), pix.height()) > preview_side:
                return pix.scaled(preview_side, preview_side, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            return pix

        # 1) 常见 thumbnail API（返回 QImage / ndarray / PIL）
        thumb_methods = (
            "get_thumbnail_qimage",
            "thumbnail_qimage",
            "thumbnail_qt",
            "get_thumbnail",
            "thumbnail",
        )
        thumb_kwargs = (
            {"max_side": fetch_side},
            {"size": fetch_side},
            {"size": (fetch_side, fetch_side)},
            {"max_dim": fetch_side},
        )
        for name in thumb_methods:
            fn = getattr(r, name, None)
            if callable(fn):
                for kwargs in thumb_kwargs:
                    try:
                        img = fn(**kwargs)
                    except TypeError:
                        continue
                    except Exception:
                        break
                    else:
                        pix = _to_qpix(img)
                        norm = _normalize_pix(pix)
                        if norm is not None:
                            return norm
                # 如果不同参数都失败，则继续尝试其他方法

        # 2) 回退：挑选一个尺寸适中的 pyramid 层
        try:
            dims = list(getattr(r, "level_dimensions", []))
            if not dims:
                return None

            target_level = len(dims) - 1
            for idx, (w_l, h_l) in enumerate(dims):
                if max(int(w_l), int(h_l)) <= fetch_side:
                    target_level = idx
                    break

            w_sel, h_sel = dims[target_level]
            w_sel = max(1, int(w_sel))
            h_sel = max(1, int(h_sel))

            if hasattr(r, "read_region"):
                img = r.read_region(level=target_level, x=0, y=0, w=w_sel, h=h_sel)
            elif hasattr(r, "get_region"):
                img = r.get_region(level=target_level, x=0, y=0, w=w_sel, h=h_sel)
            else:
                img = None
            norm = _normalize_pix(_to_qpix(img))
            if norm is not None:
                return norm
        except Exception:
            pass
        return None

    def _minimap_refresh_preview(self):
        """重建 minimap 预览图并刷新显示。外部可在旋转/换片后调用。"""
        pix = self._minimap_make_preview()
        self._minimap_pix = pix
        self._mini_canvas.set_pixmap(pix)
        ready = bool(pix) and not pix.isNull()
        self._minimap_ready = ready
        self._mini_cont.setVisible(ready)
        # 文本提示：有图就隐藏
        try:
            self._mini_hint.setVisible(not ready)
        except Exception:
            pass

    def _update_minimap_image(self):
        """按照当前 reader 重新生成缩略图。"""
        self._minimap_refresh_preview()

    def _hide_minimap(self):
        """关闭小地图显示，并清空状态。"""
        if not hasattr(self, "_mini_cont"):
            return
        self._minimap_ready = False
        self._minimap_pix = None
        try:
            self._mini_canvas.set_pixmap(None)
            self._mini_canvas.set_scene_size(1.0, 1.0)
            self._mini_canvas.set_view_rect_scene(QRectF(0.0, 0.0, 0.0, 0.0))
        except Exception:
            pass
        try:
            self._mini_cont.setVisible(False)
        except Exception:
            pass
        try:
            self._mini_hint.setVisible(True)
        except Exception:
            pass

    # 绑定/刷新 minimap 内容（在成功打开 WSI 后调用一次）
    def _minimap_bind_reader(self, reader):
        try:
            w0, h0 = reader.level_dimensions[0]
        except Exception:
            # 没有 level 信息就先隐藏
            self._mini_cont.setVisible(False)
            self._minimap_ready = False
            return

        self._mini_canvas.set_scene_size(float(w0), float(h0))
        # 生成/刷新缩略图
        self._update_minimap_image()

        # 首次绑定就立即同步一次红框
        if hasattr(self, "_update_minimap_rect"):
            self._update_minimap_rect()

    # 把当前主视口的可见矩形同步给 minimap
    def _update_minimap_rect(self):
        if not getattr(self, "_minimap_ready", False):
            return
        if not (self.view and getattr(self.view, "item", None)):
            return
        try:
            vr = self.view.viewport().rect()
            vis = self.view.mapToScene(vr).boundingRect().intersected(self.view.item.boundingRect())
            self._mini_canvas.set_view_rect_scene(vis)
        except Exception:
            pass

    # 收到 minimap 的“请求居中”信号
    def _minimap_center_on_scene(self, cx: float, cy: float):
        try:
            hold_capture = not getattr(self, "_minimap_drag_active", False)
            if hasattr(self.view, "begin_pan_transition"):
                self.view.begin_pan_transition(
                    capture=hold_capture,
                    cover_ms=320 if hold_capture else 220,
                    freeze_ms=220 if hold_capture else 120,
                )
        except Exception:
            pass
        try:
            self.view.centerOn(QPointF(cx, cy))
            # 若你实现了视口钳制，顺带调用
            if hasattr(self.view, "_clamp_viewport_to_scene"):
                self.view._clamp_viewport_to_scene()
            if hasattr(self.view, "_prefetch_visible_rect"):
                self.view._prefetch_visible_rect(full_cover=True, extra_margin=2)
        finally:
            if hasattr(self, "_update_minimap_rect"):
                self._update_minimap_rect()

    def _on_minimap_drag_state(self, active: bool):
        self._minimap_drag_active = bool(active)
        try:
            if hasattr(self.view, "begin_pan_transition"):
                self.view.begin_pan_transition(
                    capture=active,
                    cover_ms=360 if active else 200,
                    freeze_ms=260 if active else 120,
                )
        except Exception:
            pass


# ---- helpers: convert numpy / PIL to QImage ----
def _to_qimage(data) -> Optional[QImage]:
    try:
        from PIL import Image as _PILImage
    except Exception:
        _PILImage = None

    if isinstance(data, QImage):
        return data

    if _PILImage is not None and isinstance(data, _PILImage.Image):
        im = data.convert("RGBA")
        w, h = im.size
        buf = im.tobytes("raw", "RGBA")
        q = QImage(buf, w, h, QImage.Format_RGBA8888)
        return q.copy()

    if isinstance(data, np.ndarray):
        arr = data
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        if arr.ndim == 2:
            h, w = arr.shape
            q = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
            return q.copy()
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            h, w, c = arr.shape
            if c == 3:
                q = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
                return q.copy()
            else:
                q = QImage(arr.data, w, h, w * 4, QImage.Format_RGBA8888)
                return q.copy()
    return None
