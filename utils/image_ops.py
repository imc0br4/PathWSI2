import numpy as np
from PySide6.QtGui import QImage, QPixmap  # QPixmap 保留，避免外部依赖受影响
from PIL import Image


# ndarray (H,W,[C]) -> QImage
# 兼容：
# - 非 uint8 输入会被安全转换为 uint8（clip 到 0..255）
# - 非连续（切片/转置）数组会先转为连续内存，bytesPerLine 使用 strides 更稳
def ndarray_to_qimage(arr: np.ndarray) -> QImage:
    if not isinstance(arr, np.ndarray):
        raise TypeError("ndarray_to_qimage expects a numpy.ndarray")

    # 类型兜底（避免 PIL/Qt 因 dtype 抛错）
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)

    # 连续内存，避免错行/花屏
    arr = np.ascontiguousarray(arr)

    if arr.ndim == 2:
        h, w = arr.shape
        bpl = arr.strides[0]  # bytes per line
        qimg = QImage(arr.data, w, h, bpl, QImage.Format_Grayscale8)
        return qimg.copy()  # 与 numpy 内存解耦

    if arr.ndim == 3:
        h, w, c = arr.shape
        if c == 3:
            bpl = arr.strides[0]
            qimg = QImage(arr.data, w, h, bpl, QImage.Format_RGB888)
            return qimg.copy()
        if c == 4:
            bpl = arr.strides[0]
            qimg = QImage(arr.data, w, h, bpl, QImage.Format_RGBA8888)
            return qimg.copy()

    raise ValueError("Unsupported array shape; expected (H,W), (H,W,3) or (H,W,4) uint8")


def qimage_from_rgba(rgba: np.ndarray) -> QImage:
    return ndarray_to_qimage(rgba)


def colorize_mask(mask: np.ndarray, color=(255, 0, 255), alpha=0.5) -> np.ndarray:
    """
    将二值/布尔掩码上色为 RGBA。
    - mask: 任意可转为 bool 的 2D 数组
    - color: (R, G, B)
    - alpha: [0,1]
    """
    if mask is None:
        raise ValueError("mask is None")
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")

    m = (mask.astype(np.uint8) > 0)
    h, w = m.shape

    a = int(max(0.0, min(1.0, float(alpha))) * 255)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = int(color[0]) & 255
    rgba[..., 1] = int(color[1]) & 255
    rgba[..., 2] = int(color[2]) & 255
    rgba[..., 3] = (m.astype(np.uint8) * a)
    return rgba


def save_png(path: str, arr: np.ndarray):
    """
    安全保存 PNG：
    - 非 uint8 时先 clip 到 0..255 再转 uint8
    - 其他行为与原先保持一致
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("save_png expects a numpy.ndarray")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
    Image.fromarray(arr).save(path)
