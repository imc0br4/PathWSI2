# inference/batcher.py
from __future__ import annotations
from typing import List, Tuple, Iterable
import torch

class Batcher:
    """
    简单批处理器：累积 [1,3,P,P] 张量与其元信息，凑够 batch_size 或 flush().
    """
    def __init__(self, batch_size: int = 4):
        self.batch_size = int(batch_size)
        self._tiles: List[torch.Tensor] = []
        self._meta: List[Tuple[int,int,int,int]] = []

    def add(self, t: torch.Tensor, meta: Tuple[int,int,int,int]):
        self._tiles.append(t)
        self._meta.append(meta)
        return len(self._tiles) >= self.batch_size

    def flush(self) -> Tuple[torch.Tensor, List[Tuple[int,int,int,int]]]:
        if not self._tiles:
            return None, []
        x = torch.cat(self._tiles, dim=0)
        meta = self._meta[:]
        self._tiles.clear()
        self._meta.clear()
        return x, meta

    def has_pending(self) -> bool:
        return len(self._tiles) > 0

    def iter_batches(self, iterable: Iterable):
        """
        工具：从外部 (t,meta) 迭代器生成批次。
        """
        for t, meta in iterable:
            full = self.add(t, meta)
            if full:
                yield self.flush()
        if self.has_pending():
            yield self.flush()
