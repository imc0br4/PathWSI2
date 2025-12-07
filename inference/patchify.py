# inference/patchify.py
from __future__ import annotations
import math
from typing import Iterator, Tuple, Optional
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
    reader.read_region(level, x0,y0,w,h): x0,y0 是 level0 坐标，w,h 是 level 像素尺寸。
    """
    def __init__(
        self,
        reader,
        level: int,
        patch: int = 224,
        overlap: int = 32,
        roi_level0: Optional[Tuple[int, int, int, int]] = None,  # x0,y0,w0,h0
        mean=(0.485,0.456,0.406),
        std=(0.229,0.224,0.225),
    ):
        self.reader = reader
        self.level = int(level)
        self.patch = int(patch)
        self.overlap = int(overlap)

        ds = float(reader.level_downsamples[level])
        if roi_level0 is None:
            w0, h0 = reader.level_dimensions[0]
            self.x0 = 0
            self.y0 = 0
            self.wL = int(math.ceil(w0 / ds))
            self.hL = int(math.ceil(h0 / ds))
        else:
            x0,y0,w0,h0 = map(int, roi_level0)
            self.x0 = x0
            self.y0 = y0
            self.wL = int(math.ceil(w0 / ds))
            self.hL = int(math.ceil(h0 / ds))

        self.stride = max(1, self.patch - self.overlap)
        self.mean = np.asarray(mean, dtype=np.float32)[None, None, :]
        self.std  = np.asarray(std,  dtype=np.float32)[None, None, :]
        self.window = hann2d(self.patch, self.patch)

    def _read_rgb(self, xL: int, yL: int, wL: int, hL: int) -> np.ndarray:
        ds = float(self.reader.level_downsamples[self.level])
        xx0 = int(round((self.x0 + xL) * ds))
        yy0 = int(round((self.y0 + yL) * ds))
        rgba = self.reader.read_region(self.level, xx0, yy0, wL, hL)  # [hL,wL,4]
        return rgba[..., :3].astype(np.float32) / 255.0

    def _to_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        x = (rgb - self.mean) / self.std  # [H,W,3]
        x = np.transpose(x, (2,0,1))[None, ...]  # [1,3,H,W]
        return torch.from_numpy(x.astype(np.float32))

    def tiles(self) -> Iterator[Tuple[int,int,int,int, torch.Tensor]]:
        for yL in range(0, self.hL, self.stride):
            ph = min(self.patch, self.hL - yL)
            for xL in range(0, self.wL, self.stride):
                pw = min(self.patch, self.wL - xL)
                rgb = self._read_rgb(xL, yL, pw, ph)
                if ph != self.patch or pw != self.patch:
                    canvas = np.zeros((self.patch, self.patch, 3), dtype=np.float32)
                    canvas[0:ph, 0:pw, :] = rgb
                    rgb = canvas
                yield xL, yL, pw, ph, self._to_tensor(rgb)
