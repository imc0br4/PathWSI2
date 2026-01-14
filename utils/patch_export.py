# utils/patch_export.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def iter_positive_cells(
    overlay_rgba,
    grid_px: int,
    mode: str = "center",
    alpha_th: int = 1
) -> Iterator[Tuple[int, int]]:
    """
    遍历 alpha>0 的网格单元，返回 (gx, gy)。

    mode:
      - "center": 仅采样格子中心像素（最快，旧行为）
      - "any":    只要格子内任意像素 alpha>0 就算阳性（更稳）
      - "max":    等价于 any（兼容习惯叫法）
    """
    if overlay_rgba is None:
        return
    if not isinstance(overlay_rgba, np.ndarray):
        overlay_rgba = np.asarray(overlay_rgba)

    H, W = overlay_rgba.shape[:2]
    grid_px = int(grid_px) if grid_px else 1
    if grid_px <= 0:
        grid_px = 1

    mode = (mode or "center").lower().strip()
    if mode == "max":
        mode = "any"

    alpha = overlay_rgba[..., 3] if overlay_rgba.ndim == 3 and overlay_rgba.shape[2] >= 4 else None
    if alpha is None:
        return

    rows = (H + grid_px - 1) // grid_px
    cols = (W + grid_px - 1) // grid_px

    for gy in range(rows):
        y0 = gy * grid_px
        y1 = min(y0 + grid_px, H)
        cy = min(H - 1, y0 + grid_px // 2)

        for gx in range(cols):
            x0 = gx * grid_px
            x1 = min(x0 + grid_px, W)
            cx = min(W - 1, x0 + grid_px // 2)

            if mode == "center":
                if int(alpha[cy, cx]) >= alpha_th:
                    yield (gx, gy)
            else:  # "any"
                if int(alpha[y0:y1, x0:x1].max()) >= alpha_th:
                    yield (gx, gy)


def _to_pil_rgb(x) -> Image.Image:
    """
    将 reader.read_region 返回值转成 PIL RGB。
    兼容：
      - PIL.Image
      - numpy.ndarray: (H,W,4)/(H,W,3)/(H,W)
    """
    if x is None:
        return Image.new("RGB", (1, 1), (255, 255, 255))

    # 1) PIL.Image
    if isinstance(x, Image.Image):
        return x.convert("RGB")

    # 2) numpy.ndarray
    if isinstance(x, np.ndarray):
        arr = x
        if arr.dtype != np.uint8:
            # 常见：float/uint16 -> uint8
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.ndim == 2:
            # 灰度 -> RGB
            return Image.fromarray(arr, mode="L").convert("RGB")

        if arr.ndim == 3:
            c = arr.shape[2]
            if c == 4:
                # RGBA -> RGB
                return Image.fromarray(arr, mode="RGBA").convert("RGB")
            if c == 3:
                return Image.fromarray(arr, mode="RGB")
            # 其他通道数，尽量取前三通道
            if c > 4:
                return Image.fromarray(arr[:, :, :3], mode="RGB")

        # 实在不符合就兜底
        return Image.new("RGB", (arr.shape[1], arr.shape[0]), (255, 255, 255))

    # 3) 其他未知类型兜底
    return Image.new("RGB", (1, 1), (255, 255, 255))


def safe_read_patch_level0(reader, x0: int, y0: int, size: int = 1024, fill: int = 255) -> Image.Image:
    """
    始终返回 size×size 的 PIL.Image (RGB)。
    超边界用白底补齐。
    兼容 reader.read_region 返回 PIL.Image 或 numpy.ndarray。
    """
    W0, H0 = reader.level_dimensions[0]
    x0i, y0i = int(x0), int(y0)

    # 目标 patch 在 level0 的范围
    x1i, y1i = x0i + size, y0i + size

    # 与切片范围求交集
    rx0 = max(0, x0i)
    ry0 = max(0, y0i)
    rx1 = min(W0, x1i)
    ry1 = min(H0, y1i)

    canvas = Image.new("RGB", (size, size), (fill, fill, fill))

    if rx1 <= rx0 or ry1 <= ry0:
        return canvas

    rw = rx1 - rx0
    rh = ry1 - ry0

    # 注意：不同 reader 的 read_region 可能返回 PIL 或 ndarray
    region = reader.read_region(0, rx0, ry0, rw, rh)
    patch = _to_pil_rgb(region)

    # 粘贴到 canvas 的偏移
    ox = rx0 - x0i
    oy = ry0 - y0i
    canvas.paste(patch, (ox, oy))
    return canvas


def draw_detections_on_patch(img: Image.Image, detections: List[Dict[str, Any]], patch_x0: int, patch_y0: int):
    """在 patch 上画检测框。detections 坐标须是 level0 xyxy。"""
    if not detections:
        return img

    draw = ImageDraw.Draw(img)
    for det in detections:
        if "xyxy" in det:
            x1, y1, x2, y2 = det["xyxy"]
        else:
            x1 = det.get("x1", det.get("xmin"))
            y1 = det.get("y1", det.get("ymin"))
            x2 = det.get("x2", det.get("xmax"))
            y2 = det.get("y2", det.get("ymax"))
        if x1 is None:
            continue

        px1 = int(round(float(x1) - patch_x0))
        py1 = int(round(float(y1) - patch_y0))
        px2 = int(round(float(x2) - patch_x0))
        py2 = int(round(float(y2) - patch_y0))

        px1 = max(0, min(img.size[0] - 1, px1))
        py1 = max(0, min(img.size[1] - 1, py1))
        px2 = max(0, min(img.size[0] - 1, px2))
        py2 = max(0, min(img.size[1] - 1, py2))
        if px2 <= px1 or py2 <= py1:
            continue

        draw.rectangle([px1, py1, px2, py2], outline=(0, 120, 255), width=3)

        # --- label 规则：避免出现 “0 0.97” 看成 “00.97” ---
        cls_name = det.get("label", None)  # 优先用你 det_integration 传入的 label（更友好）
        if cls_name is None:
            cls_name = det.get("cls", det.get("class", ""))

        conf = det.get("conf", det.get("score", None))

        label = str(cls_name).strip() if cls_name is not None else ""

        # 如果 label 是纯数字（常见：cls id），就不显示它，只显示置信度
        if label.isdigit():
            label = ""

        # 格式化置信度
        conf_txt = ""
        if conf is not None:
            try:
                conf_txt = f"{float(conf):.2f}"
            except Exception:
                conf_txt = ""

        # 最终显示文本
        txt = conf_txt if (not label) else (f"{label} {conf_txt}".strip())

        if txt:
            draw.text((px1 + 2, max(0, py1 - 14)), txt, fill=(0, 120, 255))


    return img


def filter_detections_for_patch(detections: List[Dict[str, Any]], patch_x0: int, patch_y0: int, size: int = 1024):
    """只保留与 patch 相交的检测框（level0）。"""
    if not detections:
        return []
    patch_x1 = patch_x0 + size
    patch_y1 = patch_y0 + size

    out = []
    for det in detections:
        if "xyxy" in det:
            x1, y1, x2, y2 = det["xyxy"]
        else:
            x1 = det.get("x1", det.get("xmin"))
            y1 = det.get("y1", det.get("ymin"))
            x2 = det.get("x2", det.get("xmax"))
            y2 = det.get("y2", det.get("ymax"))
        if x1 is None:
            continue
        x1 = float(x1); y1 = float(y1); x2 = float(x2); y2 = float(y2)
        if x2 <= patch_x0 or x1 >= patch_x1 or y2 <= patch_y0 or y1 >= patch_y1:
            continue
        out.append(det)
    return out
