# inference/engine.py
from __future__ import annotations
import os
from typing import Optional, Tuple, Callable, Dict, List
import numpy as np
import torch
from PIL import Image

from inference.patchify import SlideTiler
from models.manager import ModelRunner

ProgressFn = Optional[Callable[[int, int], None]]  # (done, total)


# ------------------ 公共工具 ------------------
def _choose_level_for_mag(reader, target_mag: Optional[float]) -> int:
    """
    根据目标倍率选择 pyramid level：
      - 若未给目标倍率或未能获取物镜倍率，返回 level 0
      - 否则选择下采样最接近 obj/target_mag 的层
    """
    if target_mag is None:
        return 0
    obj = reader.objective_power()
    if not obj or obj <= 0:
        return 0
    ds_list = [float(d) for d in reader.level_downsamples]
    target_ds = obj / float(target_mag)
    idx = int(np.argmin([abs(d - target_ds) for d in ds_list]))
    return idx


def _post_act_seg(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """分割输出后处理：二类→sigmoid，多类→softmax"""
    if num_classes <= 1 or logits.shape[1] == 1:
        return torch.sigmoid(logits)
    return torch.softmax(logits, dim=1)


# ------------------ 分割：segment_slide + save_mask ------------------
def segment_slide(
    reader,
    runner: ModelRunner,
    num_classes: int = 1,
    level: Optional[int] = None,
    target_mag: Optional[float] = None,
    patch_size: int = 224,
    overlap: int = 32,
    batch_size: int = 4,
    roi_level0: Optional[Tuple[int,int,int,int]] = None,
    progress: ProgressFn = None,
) -> Dict[str, np.ndarray]:
    """
    滑窗分割整张切片（或 ROI）：
    返回:
      {
        'prob': float32 [C,H,W],  # 在所选 level 分辨率下
        'mask': uint8  [H,W],     # 二类为 0/255，多类为类别索引
        'level': int
      }
    """
    # 选层
    if level is None:
        level = _choose_level_for_mag(reader, target_mag)

    tiler = SlideTiler(reader, level, patch=patch_size, overlap=overlap, roi_level0=roi_level0)
    H, W = tiler.hL, tiler.wL
    C = max(1, int(num_classes))
    acc = np.zeros((C, H, W), dtype=np.float32)
    wgt = np.zeros((H, W), dtype=np.float32)
    win = tiler.window  # 汉宁窗融合

    # 统计 tile 总数（用于进度）
    stride = tiler.stride
    total = ((H + stride - 1) // stride) * ((W + stride - 1) // stride)
    done = 0

    batch_tiles: List[torch.Tensor] = []
    batch_metas: List[Tuple[int,int,int,int]] = []

    @torch.no_grad()
    def flush():
        nonlocal batch_tiles, batch_metas, done
        if not batch_tiles:
            return
        x = torch.cat(batch_tiles, dim=0)        # [N,3,P,P]
        y = runner(x)                             # [N,C,P,P] or [N,1,P,P]
        y = _post_act_seg(y, C).numpy()           # 概率
        for i, (xL, yL, pw, ph) in enumerate(batch_metas):
            prob = y[i, :, 0:ph, 0:pw]
            acc[:, yL:yL+ph, xL:xL+pw] += prob * win[0:ph, 0:pw][None, ...]
            wgt[yL:yL+ph, xL:xL+pw] += win[0:ph, 0:pw]
            done += 1
            if progress:
                progress(done, total)
        batch_tiles.clear()
        batch_metas.clear()

    # 迭代所有 tile
    for yL in range(0, H, stride):
        ph = min(tiler.patch, H - yL)
        for xL in range(0, W, stride):
            pw = min(tiler.patch, W - xL)
            rgb = tiler._read_rgb(xL, yL, pw, ph)
            if ph != tiler.patch or pw != tiler.patch:
                canvas = np.zeros((tiler.patch, tiler.patch, 3), dtype=np.float32)
                canvas[0:ph, 0:pw, :] = rgb
                rgb = canvas
            t = tiler._to_tensor(rgb)
            batch_tiles.append(t)
            batch_metas.append((xL, yL, pw, ph))
            if len(batch_tiles) >= batch_size:
                flush()
    flush()

    w = np.maximum(wgt, 1e-6)[None, ...]
    prob = acc / w

    if C == 1:
        mask = (prob[0] >= 0.5).astype(np.uint8) * 255
    else:
        mask = np.argmax(prob, axis=0).astype(np.uint8)

    return {"prob": prob, "mask": mask, "level": int(level)}


def save_mask(mask: np.ndarray, out_path: str):
    """保存 uint8 掩膜到文件（自动创建父目录）。"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(mask).save(out_path)
    return out_path


# ------------------ 分类：classify_slide（供 Cls 页面） ------------------

# ------------------ 分类：classify_slide（供 Cls 页面） ------------------
@torch.no_grad()
def classify_slide(
    reader,
    runner: ModelRunner,
    classes: List[str],
    papillary_ids: List[int],
    threshold: float = 0.5,
    level: Optional[int] = None,
    target_mag: Optional[float] = None,
    patch_size: int = 224,
    overlap: int = 32,
    batch_size: int = 8,
    roi_level0: Optional[Tuple[int,int,int,int]] = None,
    mean=(0.485,0.456,0.406),
    std=(0.229,0.224,0.225),
    progress: ProgressFn = None,
    color=(255, 0, 255),
    use_prob_alpha: bool = False,        # 新增：是否用概率映射 alpha（默认关闭）
    alpha_const: float = 0.4,            # 新增：固定透明度（0..1），来自 palette.yaml
) -> Dict[str, np.ndarray]:
    """
    将 WSI 切成 patch 并做 N 分类，仅对 papillary_ids 上色。
    关键变化：
      - 仅在“stride 区域”上色（stride = patch - overlap），避免重叠覆盖；
      - 默认使用固定透明度，保持每格深浅一致；
      - 输出仍是 RGBA 叠加层。
    返回:
      {'overlay': uint8 [H,W,4], 'level': int}
    """
    if level is None:
        level = _choose_level_for_mag(reader, target_mag)

    tiler = SlideTiler(reader, level, patch=patch_size, overlap=overlap,
                       roi_level0=roi_level0, mean=mean, std=std)
    H, W = tiler.hL, tiler.wL
    overlay = np.zeros((H, W, 4), dtype=np.uint8)

    stride = max(1, int(tiler.stride))  # 防御：避免 stride<=0
    total = ((H + stride - 1) // stride) * ((W + stride - 1) // stride)
    done = 0

    batch_tiles: List[torch.Tensor] = []
    batch_metas: List[Tuple[int,int,int,int]] = []

    # 固定 alpha（若启用 use_prob_alpha 则按概率）
    alpha_const_u8 = int(max(0.0, min(1.0, float(alpha_const))) * 255)

    def _flush_cls():
        nonlocal batch_tiles, batch_metas, done
        if not batch_tiles:
            return
        x = torch.cat(batch_tiles, dim=0)       # [N,3,P,P]
        logits = runner(x)                      # [N,C] 或 [N,C,1,1]
        if logits.ndim == 4 and logits.shape[2] == 1 and logits.shape[3] == 1:
            logits = logits[:, :, 0, 0]
        prob = torch.softmax(logits, dim=1).cpu().numpy()  # [N,C]

        for i, (xL, yL, pw, ph) in enumerate(batch_metas):
            # papillary 概率
            p_pap = 0.0
            for cls_id in papillary_ids:
                if 0 <= cls_id < prob.shape[1]:
                    p_pap = max(p_pap, float(prob[i, cls_id]))

            if p_pap >= float(threshold):
                # 仅在 stride 区域上色（无重叠）
                w_core = min(stride, W - xL)
                h_core = min(stride, H - yL)
                if w_core > 0 and h_core > 0:
                    overlay[yL:yL+h_core, xL:xL+w_core, 0] = color[0]
                    overlay[yL:yL+h_core, xL:xL+w_core, 1] = color[1]
                    overlay[yL:yL+h_core, xL:xL+w_core, 2] = color[2]
                    if use_prob_alpha:
                        oa = int(max(0.0, min(1.0, p_pap)) * 255)
                    else:
                        oa = alpha_const_u8
                    overlay[yL:yL+h_core, xL:xL+w_core, 3] = oa

            done += 1
            if progress:
                progress(done, total)

        batch_tiles.clear()
        batch_metas.clear()

    for yL in range(0, H, stride):
        ph = min(tiler.patch, H - yL)
        for xL in range(0, W, stride):
            pw = min(tiler.patch, W - xL)
            rgb = tiler._read_rgb(xL, yL, pw, ph)
            if ph != tiler.patch or pw != tiler.patch:
                canvas = np.zeros((tiler.patch, tiler.patch, 3), dtype=np.float32)
                canvas[0:ph, 0:pw, :] = rgb
                rgb = canvas
            t = tiler._to_tensor(rgb)
            batch_tiles.append(t)
            batch_metas.append((xL, yL, pw, ph))
            if len(batch_tiles) >= batch_size:
                _flush_cls()
    _flush_cls()

    return {"overlay": overlay, "level": int(level)}
