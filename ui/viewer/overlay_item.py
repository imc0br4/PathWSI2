# ui/viewer/overlay_item.py
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsItem
from PySide6.QtGui import QPixmap, QPainter
from PySide6.QtCore import QRectF, Qt
import numpy as np
from utils.image_ops import ndarray_to_qimage


class OverlayItem(QGraphicsPixmapItem):
    """
    叠加层图元：
      - set_rgba：整图替换
      - update_subrect：原地更新子矩形，避免整图重建
    性能要点：
      - FastTransformation：邻近采样缩放，避免大图平滑插值造成卡顿
      - DeviceCoordinateCache：以设备坐标缓存，平移/轻微缩放重绘开销更低
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pix = QPixmap()
        # 关键：提高交互帧率
        self.setTransformationMode(Qt.FastTransformation)
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)

    def set_rgba(self, arr: np.ndarray):
        """整图替换（arr: uint8 RGBA HxWx4）。"""
        if arr is None or arr.ndim != 3 or arr.shape[2] != 4 or arr.dtype != np.uint8:
            raise ValueError("set_rgba expects uint8 RGBA array of shape (H,W,4)")
        qimg = ndarray_to_qimage(arr)  # RGBA8888
        self._pix = QPixmap.fromImage(qimg)
        self.setPixmap(self._pix)

    def update_subrect(self, x: int, y: int, w: int, h: int, sub_np: np.ndarray):
        """
        原地更新一小块，避免整图 set_rgba。
        (x,y,w,h) 为 overlay 像素坐标；sub_np 为 uint8 RGBA，尺寸=(h,w,4)。
        """
        if self._pix.isNull():
            raise RuntimeError("Overlay pixmap not initialized")

        if sub_np.dtype != np.uint8 or sub_np.ndim != 3 or sub_np.shape[2] != 4:
            raise ValueError("sub_np must be uint8 RGBA (H,W,4)")

        if sub_np.shape[1] != w or sub_np.shape[0] != h:
            raise ValueError("subrect size mismatch")

        # 边界裁剪（防御性处理）
        pw, ph = self._pix.width(), self._pix.height()
        if w <= 0 or h <= 0 or x >= pw or y >= ph or (x + w) <= 0 or (y + h) <= 0:
            return
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(pw, int(x + w))
        y1 = min(ph, int(y + h))
        if x1 <= x0 or y1 <= y0:
            return

        # 如有必要，对 sub_np 做同步裁剪
        if (x0, y0, x1 - x0, y1 - y0) != (x, y, w, h):
            dx0 = x0 - x
            dy0 = y0 - y
            sub_np = sub_np[dy0:dy0 + (y1 - y0), dx0:dx0 + (x1 - x0)]
            w = x1 - x0
            h = y1 - y0
            x = x0
            y = y0

        qsub = ndarray_to_qimage(sub_np)  # RGBA8888
        p = QPainter(self._pix)
        # FastTransformation 已在图元级别设置，这里无需再改 painter 的 render hint
        p.drawImage(int(x), int(y), qsub)
        p.end()

        # 仅重绘该区域
        self.update(QRectF(float(x), float(y), float(w), float(h)))
