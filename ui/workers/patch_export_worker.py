# ui/workers/patch_export_worker.py
from __future__ import annotations
import os, json
from typing import Optional, List, Dict, Any, Tuple, Set

from PySide6.QtCore import QThread, Signal

from utils.patch_export import (
    iter_positive_cells,
    safe_read_patch_level0,
    filter_detections_for_patch,
    draw_detections_on_patch
)

class PatchExportWorker(QThread):
    progress = Signal(int, int)        # done, total
    finished_ok = Signal(dict)         # {"out_dir":..., "count":..., "meta":...}
    failed = Signal(str)

    def __init__(
        self,
        reader,
        overlay_rgba,
        overlay_pos: Tuple[int, int],
        overlay_ds: float,
        grid_px: int,
        out_dir: str,
        detections: Optional[List[Dict[str, Any]]] = None,
        patch_size: int = 1024,

        # NEW: 如果传了 cells，就用它作为“要导出的格子列表”（通常用检测时的 dilated cells）
        cells: Optional[List[Tuple[int, int]]] = None,
        include_stain: bool = True,
        # NEW: 额外补齐：按检测框中心导出 patch（解决“检测在染色块外 => 导出遗漏”）
        extra_centers_level0: Optional[List[Tuple[float, float]]] = None,
        stain_cell_mode: str = "center",
        parent=None
    ):
        super().__init__(parent)
        self.reader = reader
        self.overlay_rgba = overlay_rgba
        self.overlay_pos = overlay_pos
        self.overlay_ds = float(overlay_ds) if overlay_ds else 1.0
        self.grid_px = int(grid_px) if grid_px else 1
        self.out_dir = out_dir
        self.detections = detections or []
        self.patch_size = int(patch_size)

        self.cells = cells
        self.extra_centers_level0 = extra_centers_level0 or []
        self.include_stain = bool(include_stain)
        self.stain_cell_mode = str(stain_cell_mode or "center").strip().lower()

    def run(self):
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            ox0, oy0 = int(self.overlay_pos[0]), int(self.overlay_pos[1])

            # 1) 染色块格子：默认用 any/max 防漏（不再只采中心点）
            # if self.cells is not None:
            #     stain_cells = list(self.cells)
            # else:
            #     stain_cells = list(iter_positive_cells(self.overlay_rgba, self.grid_px, mode="any"))
            if not self.include_stain:
                stain_cells = []
            else:
                if self.cells is not None:
                    stain_cells = list(self.cells)
                else:
                    mode = "center" if self.stain_cell_mode not in ("any","max") else "any"
                    stain_cells = list(iter_positive_cells(self.overlay_rgba,self.grid_px,mode=mode))
            # 2) 额外中心点（检测框中心）
            det_centers = list(self.extra_centers_level0) if self.extra_centers_level0 else []

            # 3) 先算总数（去重按 patch 左上角）
            seen: Set[Tuple[int, int]] = set()
            tasks: List[Dict[str, Any]] = []

            # --- 来自染色格子 ---
            for (gx, gy) in stain_cells:
                cx_ov = gx * self.grid_px + self.grid_px * 0.5
                cy_ov = gy * self.grid_px + self.grid_px * 0.5
                cx0 = ox0 + cx_ov * self.overlay_ds
                cy0 = oy0 + cy_ov * self.overlay_ds
                px0 = int(round(cx0 - self.patch_size / 2))
                py0 = int(round(cy0 - self.patch_size / 2))
                k = (px0, py0)
                if k in seen:
                    continue
                seen.add(k)
                tasks.append({
                    "kind": "stain",
                    "gx": gx, "gy": gy,
                    "cx0": float(cx0), "cy0": float(cy0),
                    "x0": px0, "y0": py0
                })

            # --- 来自检测中心（补齐导出遗漏）---
            for (cx0, cy0) in det_centers:
                px0 = int(round(float(cx0) - self.patch_size / 2))
                py0 = int(round(float(cy0) - self.patch_size / 2))
                k = (px0, py0)
                if k in seen:
                    continue
                seen.add(k)
                tasks.append({
                    "kind": "det",
                    "gx": None, "gy": None,
                    "cx0": float(cx0), "cy0": float(cy0),
                    "x0": px0, "y0": py0
                })

            total = len(tasks)

            manifest = []
            for i, t in enumerate(tasks, 1):
                px0, py0 = int(t["x0"]), int(t["y0"])

                img = safe_read_patch_level0(self.reader, px0, py0, size=self.patch_size)

                dets_here = []
                if self.detections:
                    dets_here = filter_detections_for_patch(self.detections, px0, py0, size=self.patch_size)
                    if dets_here:
                        img = draw_detections_on_patch(img, dets_here, px0, py0)

                fname = f"patch_{i:05d}_{t['kind']}_x{px0}_y{py0}.png"
                fpath = os.path.join(self.out_dir, fname)
                img.save(fpath)

                manifest.append({
                    "file": fname,
                    "kind": t["kind"],
                    "x0": px0, "y0": py0,
                    "size": self.patch_size,
                    "grid": (
                        {"gx": t["gx"], "gy": t["gy"], "grid_px_on_overlay": self.grid_px}
                        if t["gx"] is not None else None
                    ),
                    "center_level0": {"cx0": t["cx0"], "cy0": t["cy0"]},
                    "detections": dets_here
                })

                self.progress.emit(i, total)

            meta_path = os.path.join(self.out_dir, "patches_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"count": total, "patches": manifest}, f, ensure_ascii=False, indent=2)

            self.finished_ok.emit({"out_dir": self.out_dir, "count": total, "meta": meta_path})
        except Exception as e:
            self.failed.emit(str(e))
