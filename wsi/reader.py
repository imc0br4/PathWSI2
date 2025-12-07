# wsi/reader.py
import os
import numpy as np
from typing import Optional

try:
    import openslide
except Exception:
    openslide = None

import tifffile
from PIL import Image


OPENS_SLIDE_EXTS = {'.svs', '.ndpi', '.mrxs', '.scn', '.svslide'}
TIFF_EXTS = {'.tif', '.tiff'}
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}


class WsiReader:
    """
    统一读取接口：
      kind: 'openslide' | 'tif' | 'img'
      read_region(level, x, y, w, h) -> RGBA np.uint8（x,y 为 level0 坐标）
    """
    def __init__(self):
        self.path = None
        self.kind = None
        self._os_obj = None     # OpenSlide对象
        self._tif = None        # tifffile.TiffFile
        self._img = None        # 普通图（PIL.Image 转 np.array）
        self.level_dimensions = []
        self.level_downsamples = []
        self.level_count = 0
        self.properties = {}

    def close(self):
        if self._os_obj is not None:
            self._os_obj.close()
        if self._tif is not None:
            self._tif.close()
        self.__init__()

    def open(self, path: str):
        self.close()
        self.path = path
        ext = os.path.splitext(path)[1].lower()

        # 1) 优先 OpenSlide（多层金字塔）
        if openslide and ext in OPENS_SLIDE_EXTS:
            self._os_obj = openslide.OpenSlide(path)
            self.kind = 'openslide'
            self.level_count = self._os_obj.level_count
            self.level_dimensions = [self._os_obj.level_dimensions[i] for i in range(self.level_count)]
            self.level_downsamples = [float(self._os_obj.level_downsamples[i]) for i in range(self.level_count)]
            self.properties = dict(self._os_obj.properties)
            return self

        # 2) 再尝试真正的 TIFF/OME-TIFF
        if ext in TIFF_EXTS:
            self._tif = tifffile.TiffFile(path)
            self.kind = 'tif'
            series = self._tif.series[0]
            h, w = series.shape[-2:]
            self.level_count = 1
            self.level_dimensions = [(w, h)]
            self.level_downsamples = [1.0]
            self.properties = {'tiff.series.name': series.name}
            return self

        # 3) 常规图片（PNG/JPG/BMP）—— 用 PIL 读入内存
        if ext in IMG_EXTS:
            img = Image.open(path).convert('RGBA')  # 直接 RGBA
            arr = np.array(img, dtype=np.uint8)
            self._img = arr
            self.kind = 'img'
            h, w = arr.shape[:2]
            self.level_count = 1
            self.level_dimensions = [(w, h)]
            self.level_downsamples = [1.0]
            self.properties = {'format': img.format}
            return self

        # 4) 兜底：尝试 PIL
        try:
            img = Image.open(path).convert('RGBA')
            arr = np.array(img, dtype=np.uint8)
            self._img = arr
            self.kind = 'img'
            h, w = arr.shape[:2]
            self.level_count = 1
            self.level_dimensions = [(w, h)]
            self.level_downsamples = [1.0]
            self.properties = {'format': img.format}
            return self
        except Exception:
            pass

        # 若都失败，抛错
        raise RuntimeError(f'Unsupported image format: {ext}')

    def objective_power(self) -> Optional[float]:
        if self.kind == 'openslide':
            for k in ['aperio.AppMag', 'openslide.objective-power', 'hamamatsu.SourceLens']:
                v = self.properties.get(k)
                if v:
                    try:
                        return float(v)
                    except Exception:
                        pass
        return None

    def read_region(self, level: int, x: int, y: int, w: int, h: int) -> np.ndarray:
        """返回 RGBA uint8；(x, y) 是 level0 坐标；w,h 是 level坐标尺寸。"""
        if self.kind == 'openslide':
            rgba = self._os_obj.read_region((int(x), int(y)), int(level), (int(w), int(h)))
            return np.array(rgba, copy=False)

        if self.kind == 'tif':
            # 单层：直接裁切（level 只有 0）
            page = self._tif.pages[0]
            img = page.asarray()
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            h0, w0, _ = img.shape
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(w0, int(x + w)), min(h0, int(y + h))
            canvas = np.zeros((int(h), int(w), 4), dtype=np.uint8)
            if x0 < x1 and y0 < y1:
                crop = img[y0:y1, x0:x1]
                hh, ww = crop.shape[:2]
                canvas[0:hh, 0:ww, :3] = crop[..., :3]
                canvas[..., 3] = 255
            return canvas

        # 普通图片
        if self.kind == 'img':
            img = self._img
            h0, w0 = img.shape[:2]
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(w0, int(x + w)), min(h0, int(y + h))
            canvas = np.zeros((int(h), int(w), 4), dtype=np.uint8)
            if x0 < x1 and y0 < y1:
                crop = img[y0:y1, x0:x1]
                hh, ww = crop.shape[:2]
                canvas[0:hh, 0:ww, :] = crop
            return canvas

        raise RuntimeError('Reader not opened')
