# ui/widgets/minimap.py
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QPixmap, QPen, QColor
from PySide6.QtCore import Qt, QRect, QRectF

class MiniMapOverlay(QWidget):
    def __init__(self, parent=None, palette=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setWindowFlags(Qt.SubWindow)
        self._pix = QPixmap()
        self._scene_wh = (1, 1)
        self._view_rect_scene = QRectF()
        self._margin = 8

        # ★ 从 palette 读取颜色
        pal = palette or {}
        # 背景：略带蓝色的半透明面板，更“医学监视器”一点
        self._bg = QColor(246, 251, 255, 235)
        self._border = QColor(pal.get('border', '#B7CDE6'))

        # 视窗框颜色：默认用医疗蓝，可在 palette 中覆盖
        box_c = pal.get('minimap_box', '#1A90FF')
        self._box = QColor(box_c)

        # 确保半透明度
        self._border.setAlpha(160)
        self._box.setAlpha(220)

    def set_scene_size(self, w0: int, h0: int):
        self._scene_wh = (max(1, int(w0)), max(1, int(h0)))
        self.update()

    def set_pixmap(self, pm: QPixmap):
        self._pix = pm or QPixmap()
        self.update()

    def set_view_rect_scene(self, r: QRectF):
        self._view_rect_scene = QRectF(r)
        self.update()

    def paintEvent(self, ev):
        if self._pix.isNull():
            return
        p = QPainter(self)
        # 背板
        p.fillRect(self.rect(), self._bg)
        p.setPen(QPen(self._border, 1))
        p.drawRect(self.rect().adjusted(0,0,-1,-1))
        # 内容区域（留白边距）
        area = self.rect().adjusted(self._margin, self._margin, -self._margin, -self._margin)
        # 缩略图等比放进 area
        pm = self._pix.scaled(area.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # 居中
        x = area.x() + (area.width() - pm.width())//2
        y = area.y() + (area.height() - pm.height())//2
        p.drawPixmap(x, y, pm)

        # 红框：把 scene 的可视范围映射到 pm 坐标
        w0, h0 = self._scene_wh
        if w0 <= 0 or h0 <= 0:
            return
        sx = pm.width() / float(w0)
        sy = pm.height() / float(h0)
        rx = int(x + self._view_rect_scene.left() * sx)
        ry = int(y + self._view_rect_scene.top()  * sy)
        rw = int(self._view_rect_scene.width()  * sx)
        rh = int(self._view_rect_scene.height() * sy)
        # 取交集，避免越界
        r = QRect(rx, ry, rw, rh).intersected(QRect(x, y, pm.width(), pm.height()))
        if r.width() > 1 and r.height() > 1:
            pen = QPen(self._box, 2)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(r.adjusted(0,0,-1,-1))
        p.end()
