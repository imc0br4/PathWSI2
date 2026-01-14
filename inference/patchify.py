# inference/patchify.py
from __future__ import annotations

import math
from typing import Iterator, Tuple, Optional, Union

import numpy as np
import torch


def hann2d(h: int, w: int) -> np.ndarray:
    """2D 汉宁窗用于重叠融合"""
    wy = np.hanning(max(h, 2))
    wx = np.hanning(max(w, 2))
    win = np.outer(wy, wx).astype(np.float32)
    win += 1e-6
    return win


class SlideTiler:
    """
    在给定 level 上做滑窗切片（支持 ROI）。
    reader.read_region(level, x0, y0, w, h):
      - x0,y0 是 level0 坐标
      - w,h 是 level 像素尺寸
    """

    def __init__(
        self,
        reader,
        level: int,
        patch: int = 224,
        overlap: int = 32,
        roi_level0: Optional[Tuple[int, int, int, int]] = None,  # x0,y0,w0,h0
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.reader = reader
        self.level = int(level)
        self.patch = int(patch)
        self.overlap = int(overlap)

        # ✅ 缓存 downsample，避免每次 _read_rgb 都去取 list
        self.ds = float(reader.level_downsamples[self.level])

        if roi_level0 is None:
            w0, h0 = reader.level_dimensions[0]
            self.x0 = 0
            self.y0 = 0
            self.wL = int(math.ceil(w0 / self.ds))
            self.hL = int(math.ceil(h0 / self.ds))
        else:
            x0, y0, w0, h0 = map(int, roi_level0)
            self.x0 = x0
            self.y0 = y0
            self.wL = int(math.ceil(w0 / self.ds))
            self.hL = int(math.ceil(h0 / self.ds))

        self.stride = max(1, self.patch - self.overlap)
        self.mean = np.asarray(mean, dtype=np.float32)[None, None, :]
        self.std = np.asarray(std, dtype=np.float32)[None, None, :]
        self.window = hann2d(self.patch, self.patch)

    def _read_rgb(self, xL: int, yL: int, wL: int, hL: int) -> np.ndarray:
        """
        读 level 图块，返回 float32 rgb in [0,1]，shape [hL, wL, 3]
        """
        xx0 = int(round((self.x0 + xL) * self.ds))
        yy0 = int(round((self.y0 + yL) * self.ds))
        rgba = self.reader.read_region(self.level, xx0, yy0, wL, hL)  # [hL,wL,4]
        return rgba[..., :3].astype(np.float32) / 255.0

    def _pad_to_patch(self, rgb: np.ndarray, pw: int, ph: int) -> np.ndarray:
        """
        边缘不足 patch 的 tile 补零到 [patch, patch, 3]
        """
        if ph == self.patch and pw == self.patch:
            return rgb
        canvas = np.zeros((self.patch, self.patch, 3), dtype=np.float32)
        canvas[0:ph, 0:pw, :] = rgb
        return canvas

    def _to_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        """
        rgb: float32 [H,W,3] in [0,1]（通常 H=W=patch）
        return: torch float32 [1,3,H,W] on CPU
        """
        x = (rgb - self.mean) / self.std
        x = np.transpose(x, (2, 0, 1))[None, ...]  # [1,3,H,W]
        # ✅ from_numpy 0 拷贝（共享内存），比先 astype 再拷贝更好
        return torch.from_numpy(x.astype(np.float32, copy=False))

    def __iter__(self):
        """
        兼容一些写法：for ... in tiler
        默认迭代 tiles() 的结果
        """
        return self.tiles()

    def tiles(
        self,
        return_rgb: bool = False,
        return_tensor: bool = True,
        pad: bool = True,
    ) -> Iterator[
        Union[
            Tuple[int, int, int, int, torch.Tensor],
            Tuple[int, int, int, int, torch.Tensor, np.ndarray],
            Tuple[int, int, int, int, np.ndarray],
        ]
    ]:
        """
        默认（兼容旧代码）：yield (xL, yL, pw, ph, tensor)
        若 return_rgb=True：yield (xL, yL, pw, ph, tensor, rgb)
        若 return_tensor=False：yield (xL, yL, pw, ph, rgb)

        注意：
        - rgb 若 pad=True，则返回 [patch,patch,3]；否则返回原始 [ph,pw,3]
        - pw/ph 永远是原始未 pad 的尺寸，便于你在别处做边界处理/统计
        """
        for yL in range(0, self.hL, self.stride):
            ph = min(self.patch, self.hL - yL)
            for xL in range(0, self.wL, self.stride):
                pw = min(self.patch, self.wL - xL)

                rgb = self._read_rgb(xL, yL, pw, ph)
                if pad:
                    rgb_out = self._pad_to_patch(rgb, pw, ph)
                else:
                    rgb_out = rgb

                if return_tensor:
                    t = self._to_tensor(rgb_out)
                    if return_rgb:
                        yield xL, yL, pw, ph, t, rgb_out
                    else:
                        yield xL, yL, pw, ph, t
                else:
                    # 只要 rgb
                    yield xL, yL, pw, ph, rgb_out
