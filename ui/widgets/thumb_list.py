# ui/widgets/thumb_list.py
from __future__ import annotations
import os
from typing import Optional, List

import numpy as np
from PIL import Image

from PySide6.QtCore import Qt, QSize, Signal, QObject, QRunnable, QThreadPool, QTimer
from PySide6.QtGui import QIcon, QPixmap, QImage, QColor,QPainter
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QSizePolicy

from utils.io import list_images

# 可选：OpenSlide 缩略更快更准
try:
    import openslide
except Exception:
    openslide = None


# ---------- 小工具：numpy/PIL -> QImage（线程里安全） ----------
def _np_to_qimage(arr: np.ndarray) -> QImage:
    """返回深拷贝的 QImage（线程安全）"""
    if arr.ndim == 3 and arr.shape[2] == 3:
        h, w = arr.shape[:2]
        qimg = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888).copy()
        return qimg
    if arr.ndim == 3 and arr.shape[2] == 4:
        h, w = arr.shape[:2]
        qimg = QImage(arr.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
        return qimg
    # 灰度兜底
    h, w = arr.shape[:2]
    qimg = QImage(w, h, QImage.Format_Grayscale8)
    for y in range(h):
        qimg.scanLine(y)[:w] = arr[y, :w].tobytes()
    return qimg


def _pil_to_qimage(img: Image.Image) -> QImage:
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    arr = np.array(img, dtype=np.uint8)
    return _np_to_qimage(arr)


# ---------- 在后台线程里做“图像读取与缩略”，仅返回 QImage ----------
def _make_thumbnail_qimage(path: str, max_wh: int = 220) -> QImage:
    # 定义统一的画布尺寸（宽, 高）。建议设为固定比例，如 4:3
    # 这里我们设定为 (220, 165)，您也可以根据侧边栏宽度调整
    CANVAS_W, CANVAS_H = 220, 165

    # 1. 创建透明底板
    canvas = QImage(CANVAS_W, CANVAS_H, QImage.Format_ARGB32)
    canvas.fill(QColor(0, 0, 0, 0))  # 透明背景

    # 辅助函数：读取并缩放图片
    def load_and_scale(p):
        ext = os.path.splitext(p)[1].lower()
        img = None
        # 尝试 OpenSlide
        if openslide and ext in {".svs", ".ndpi", ".mrxs", ".scn", ".svslide"}:
            try:
                slide = openslide.OpenSlide(p)
                # 稍微读大一点，保证清晰度
                pil_thumb = slide.get_thumbnail((CANVAS_W, CANVAS_H))
                slide.close()
                if pil_thumb.mode not in ("RGB", "RGBA"):
                    pil_thumb = pil_thumb.convert("RGBA")
                img = _pil_to_qimage(pil_thumb)
            except Exception:
                pass

        # 尝试 PIL (普通图片)
        if img is None:
            try:
                pil_img = Image.open(p)
                pil_img.thumbnail((CANVAS_W, CANVAS_H), Image.Resampling.LANCZOS)
                img = _pil_to_qimage(pil_img)
            except Exception:
                pass
        return img

    source_img = load_and_scale(path)

    # 2. 如果读取成功，将其居中绘制到 canvas 上
    if source_img and not source_img.isNull():
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # 计算居中坐标
        x = (CANVAS_W - source_img.width()) // 2
        y = (CANVAS_H - source_img.height()) // 2

        painter.drawImage(x, y, source_img)
        painter.end()
    else:
        # 失败显示灰色占位
        canvas.fill(QColor(230, 230, 230))

    return canvas


# ---------- 用信号把结果带回主线程 ----------
class _ThumbEmitter(QObject):
    sigReady = Signal(str, QImage)   # path, qimage


class _ThumbTask(QRunnable):
    def __init__(self, path: str, emitter: _ThumbEmitter, max_wh: int = 220):
        super().__init__()
        self.path = path
        self.emitter = emitter
        self.max_wh = max_wh

    def run(self):
        qimg = _make_thumbnail_qimage(self.path, self.max_wh)
        # 回主线程：仅传 QImage（QPixmap 必须在 GUI 线程创建）
        self.emitter.sigReady.emit(self.path, qimg)


# ---------- 左侧缩略图列表 ----------
class WsiThumbList(QListWidget):
    """显示某个文件夹下的 WSI/图像缩略图；单击/双击发出 sigOpenPath(path)"""

    sigOpenPath = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("WsiThumbList")


        self.pool = QThreadPool.globalInstance()
        self._emitter = _ThumbEmitter()
        self._emitter.sigReady.connect(self._on_thumb_ready)

        # 外观：以大图标网格显示
        self.setViewMode(QListWidget.ListMode)
        # 稍微调整下图标尺寸，让比例更协调
        self.setIconSize(QSize(220, 165))
        # 关闭固定大小，让布局更灵活
        # self.setUniformItemSizes(True)
        self.setResizeMode(QListWidget.Adjust)
        self.setMovement(QListWidget.Static)
        self.setSpacing(12)  # 增加项间距
        self.setWrapping(False)
        self.setWordWrap(True)
        self.setSelectionMode(QListWidget.SingleSelection)

        # ★ 移除旧的硬编码样式表
        # self.setStyleSheet("QListWidget{ border:1px solid #e9e9e9; }")

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # 单击/双击/回车都触发打开
        self.itemClicked.connect(self._on_open_item)
        self.itemActivated.connect(self._on_open_item)

        self._folder: Optional[str] = None

    # ---- 对外：设置文件夹 ----
    def set_folder(self, folder: str):
        self.clear()
        self._folder = folder
        if not folder or not os.path.isdir(folder):
            return

        paths: List[str] = list_images(folder)
        if not paths:
            return

        # 使用透明占位图，比纯黑更好看
        ph = QPixmap(self.iconSize())
        ph.fill(Qt.transparent)

        for p in paths:
            it = QListWidgetItem(os.path.basename(p))
            it.setData(Qt.UserRole, p)
            it.setToolTip(p)
            it.setIcon(QIcon(ph))

            # ★ 关键修改2：明确设置每个项的文本对齐方式为水平居中 + 底部对齐
            it.setTextAlignment(Qt.AlignHCenter | Qt.AlignBottom)

            self.addItem(it)
            # 异步生成缩略... (保持不变)
            self.pool.start(_ThumbTask(p, self._emitter, max_wh=max(self.iconSize().width(),
                                                                    self.iconSize().height())))

    # ---- 缩略生成完成（主线程）----
    def _on_thumb_ready(self, path: str, qimg: QImage):
        # 转为 QPixmap（只能在 GUI 线程）
        pm = QPixmap.fromImage(qimg).scaled(
            self.iconSize(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        # 更新目标 item
        for i in range(self.count()):
            it = self.item(i)
            if it.data(Qt.UserRole) == path:
                it.setIcon(QIcon(pm))
                break

    # ---- 打开选中项 ----
    def _on_open_item(self, item: QListWidgetItem):
        p = item.data(Qt.UserRole)
        if isinstance(p, str) and os.path.exists(p):
            self.sigOpenPath.emit(p)
