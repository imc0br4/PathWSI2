# wsi/roi.py
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, List
from math import ceil
import json
import os

from PySide6.QtCore import QRectF
from PIL import Image
import numpy as np


@dataclass
class RoiRect:
    """以 level0 坐标存储的矩形 ROI"""
    x: int
    y: int
    w: int
    h: int
    angle: float = 0.0  # 预留，多边形/旋转未来扩展

    def to_qrectf(self) -> QRectF:
        return QRectF(float(self.x), float(self.y), float(self.w), float(self.h))


def qrectf_to_bbox(rect: QRectF) -> Tuple[int, int, int, int]:
    """QRectF -> (x, y, w, h)，自动归一化为非负宽高"""
    r = rect.normalized()
    x = int(round(r.x()))
    y = int(round(r.y()))
    w = int(round(r.width()))
    h = int(round(r.height()))
    return x, y, max(0, w), max(0, h)


def clamp_bbox_to_image(bbox: Tuple[int, int, int, int], w0: int, h0: int) -> Tuple[int, int, int, int]:
    """把 bbox 限制在图像范围内"""
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return 0, 0, 0, 0
    x = max(0, min(x, w0))
    y = max(0, min(y, h0))
    w = max(0, min(w, w0 - x))
    h = max(0, min(h, h0 - y))
    return x, y, w, h


def _choose_level_for_10x(reader, obj0: Optional[float]) -> int:
    """
    选择最接近 10× 的金字塔层：
      - 若有 obj0：target_ds = obj0 / 10
      - 没有 obj0：用层列表里“次粗略层”（经验：通常 level=1 很合适），也可直接选最细层 0 做精确缩放
    """
    ds_list = [float(d) for d in getattr(reader, "level_downsamples", [1.0])]
    if not ds_list:
        return 0
    if obj0 and obj0 > 0:
        target_ds = obj0 / 10.0
        # 选与 target_ds 最接近的层
        idx = min(range(len(ds_list)), key=lambda i: abs(ds_list[i] - target_ds))
        return idx
    # 无物理信息：优先第 1 层（若存在），避免一次性读超大区域
    return 1 if len(ds_list) > 1 else 0


def export_roi_10x(reader,
                   bbox_level0: Tuple[int, int, int, int],
                   out_png_path: str,
                   out_meta_path: Optional[str] = None) -> str:
    """
    按“10×”导出 ROI 到 PNG；返回最终 PNG 路径。
    - 读取时先在最接近 10× 的金字塔层上采样，然后做一次线性缩放，保证 10× 精确。
    - meta.json 写入映射信息，便于审计追踪。
    """
    x0, y0, w0, h0 = bbox_level0
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Empty ROI")

    # 读物理信息 & 选择层
    obj0 = None
    try:
        obj0 = reader.objective_power()
    except Exception:
        obj0 = None

    level = _choose_level_for_10x(reader, obj0)
    ds_list = [float(d) for d in getattr(reader, "level_downsamples", [1.0])]
    ds = ds_list[level] if level < len(ds_list) else 1.0

    # 以该层尺寸读取
    wL = int(ceil(w0 / ds))
    hL = int(ceil(h0 / ds))
    rgba = reader.read_region(level, x0, y0, wL, hL)  # RGBA uint8
    # 转 PIL Image
    if rgba.ndim == 3 and rgba.shape[2] == 4:
        img = Image.fromarray(rgba, mode="RGBA").convert("RGB")  # 去 alpha
    else:
        img = Image.fromarray(rgba[..., :3]) if rgba.ndim == 3 else Image.fromarray(rgba)

    # 若有 obj0：目标 10× 的目标宽高 = (w0/target_ds, h0/target_ds)
    # target_ds = obj0/10
    if obj0 and obj0 > 0:
        target_ds = obj0 / 10.0
        target_w = int(round(w0 / target_ds))
        target_h = int(round(h0 / target_ds))
        if target_w > 0 and target_h > 0 and (target_w != wL or target_h != hL):
            img = img.resize((target_w, target_h), Image.BILINEAR)

    # 保存 PNG
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    img.save(out_png_path)

    # 写 meta
    if out_meta_path:
        meta = {
            "roi_bbox_level0": {"x": x0, "y": y0, "w": w0, "h": h0},
            "chosen_level": int(level),
            "ds_at_level": float(ds),
            "objective_power": float(obj0) if obj0 else None,
            "path": getattr(reader, "path", None)
        }
        with open(out_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    return out_png_path
