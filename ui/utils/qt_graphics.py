# ui/utils/qt_graphics.py
from __future__ import annotations
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap, QPainter, QPen, QImage, QColor


def _hamburger_icon() -> QIcon:
    pix = QPixmap(24, 24)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    pen = QPen(QColor("#1A90FF"))
    pen.setWidth(2)
    p.setPen(pen)
    for y in (6, 12, 18):
        p.drawLine(5, y, 19, y)
    p.end()
    return QIcon(pix)

def _np_rgba_to_qpixmap(rgba: np.ndarray) -> QPixmap:
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("expect RGBA uint8 array")
    h, w = rgba.shape[:2]
    qimg = QImage(rgba.data, w, h, 4*w, QImage.Format_RGBA8888).copy()
    return QPixmap.fromImage(qimg)
