from dataclasses import dataclass
from typing import Tuple


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int




def map_level0_to_level(b: BBox, downsample: float) -> BBox:
    return BBox(int(b.x / downsample), int(b.y / downsample), int(b.w / downsample), int(b.h / downsample))