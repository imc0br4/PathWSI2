# wsi/tiles.py —— 单线程异步加载 + 生命周期安全（精简稳定版）
from collections import OrderedDict
from dataclasses import dataclass
from typing import Set

from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool, Qt
from PySide6.QtGui import QImage
import shiboken6

from utils.image_ops import qimage_from_rgba


class TileCache:
    def __init__(self, max_items=1024):
        self.max_items = max_items
        self._cache = OrderedDict()

    def get(self, key):
        if key in self._cache:
            v = self._cache.pop(key)
            self._cache[key] = v  # LRU：移动到尾部
            return v

    def put(self, key, value):
        if key in self._cache:
            self._cache.pop(key)
        self._cache[key] = value
        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()


@dataclass(frozen=True)
class TileKey:
    level: int
    tx: int
    ty: int


class _TileTask(QRunnable):
    def __init__(self, generation: int, reader, level: int, x0: int, y0: int,
                 tile_px: int, key: TileKey, signal_obj, stopping_flag_ref):
        super().__init__()
        self.generation = generation
        self.reader = reader
        self.level = level
        self.x0 = x0
        self.y0 = y0
        self.tile_px = tile_px
        self.key = key
        self.sig = signal_obj          # TileLoader（QObject）
        self._stopping_flag_ref = stopping_flag_ref

    def run(self):
        try:
            if self._stopping_flag_ref():
                return
            rgba = self.reader.read_region(self.level, self.x0, self.y0,
                                           self.tile_px, self.tile_px)
            qimg = qimage_from_rgba(rgba)

            # 接收者是否还活着；停止中或对象已销毁则不 emit
            if self._stopping_flag_ref() or not shiboken6.isValid(self.sig):
                return
            self.sig.tileReady.emit(self.generation, self.key, qimg)
        except Exception as e:
            if self._stopping_flag_ref() or not shiboken6.isValid(self.sig):
                return
            self.sig.tileFailed.emit(self.generation, self.key, str(e))


class TileLoader(QObject):
    tileReady = Signal(int, TileKey, QImage)
    tileFailed = Signal(int, TileKey, str)

    def __init__(self, reader, tile_px=256, max_workers=1, max_cache=1024, parent=None):
        super().__init__(parent)
        self.reader = reader
        self.tile_px = int(tile_px)

        # 强制串行，稳定优先；如需更快可改为 >1，但需谨慎测试线程安全
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(1 if max_workers is None else max(1, int(max_workers)))

        self.cache = TileCache(max_cache)
        self._inflight: Set[TileKey] = set()
        self.generation = 0
        self._stopping = False

        # 任务完成时（成功/失败）把 key 从 inflight 移除
        self.tileReady.connect(self._on_done, Qt.QueuedConnection)
        self.tileFailed.connect(self._on_done, Qt.QueuedConnection)

    def _on_done(self, *args):
        # 信号形如：(generation, key, qimg/errmsg)
        try:
            key = args[1]
            self._inflight.discard(key)
        except Exception:
            pass

    def bump_generation(self):
        self.generation += 1
        self._inflight.clear()

    def is_stopping(self):
        return self._stopping

    def request(self, level: int, tx: int, ty: int, priority: int | None = None):
        if self._stopping:
            return

        key = TileKey(level, tx, ty)

        # 先查缓存
        img = self.cache.get(key)
        if img is not None:
            self.tileReady.emit(self.generation, key, img)
            return

        # 避免重复请求
        if key in self._inflight:
            return

        self._inflight.add(key)

        # 计算 level0 坐标起点（ds 为该层下采样因子）
        ds = float(self.reader.level_downsamples[level]) if level < len(self.reader.level_downsamples) else 1.0
        x0 = int(round(tx * self.tile_px * ds))
        y0 = int(round(ty * self.tile_px * ds))

        task = _TileTask(self.generation, self.reader, level, x0, y0,
                         self.tile_px, key, self, self.is_stopping)
        if priority is None:
            try:
                priority = self.priority_for_level(level)
            except Exception:
                priority = 0
        self.pool.start(task, int(priority))

    def put_cache(self, key: TileKey, qimg: QImage):
        self.cache.put(key, qimg)

    def clear_all(self):
        self.cache.clear()
        self._inflight.clear()

    def stop_and_wait(self, msec: int = 10_000):
        """进入停止态，阻止新任务，等待线程池清空，然后清理缓存。"""
        self._stopping = True
        try:
            self.pool.waitForDone(msec)
        except Exception:
            pass
        self.clear_all()
    def priority_for_level(self, level: int) -> int:
        """
        层级越细（level 越小）优先级越高。
        返回一个非负整数；数值越大优先级越高。
        """
        try:
            n = len(self.reader.level_downsamples)
        except Exception:
            n = 1
        # 例如: 最细层(0)≈最高优先级 12，最粗层(n-1)≈0
        return max(0, (n - 1 - int(level)) * 2 + 0)
