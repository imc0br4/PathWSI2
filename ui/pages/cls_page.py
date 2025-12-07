# ui/pages/cls_page.py
from __future__ import annotations
import os, json, math
from typing import Optional, List

from PySide6.QtCore import Qt, QRectF, QThread, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QSplitter,
    QPushButton, QFileDialog, QMessageBox, QLabel, QComboBox, QLineEdit,
    QDoubleSpinBox, QSpinBox, QCheckBox, QSlider, QToolButton
)

from utils.hwcheck import pick_device
from wsi.reader import WsiReader
from ui.viewer.wsi_view import WsiView, select_level_smart
from ui.viewer.overlay_item import OverlayItem
from models.manager import load_model

import numpy as np
from PIL import Image
import time

# 导出策略常量
_MAX_PNG_SIDE = 12000          # 超过则自动降采样（2/4/8…）
_PNG_SAVE_KW  = dict(optimize=False, compress_level=1)  # 快速保存


# ===================== 工作线程 =====================
class ClsWorker(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(dict)
    failed = Signal(str)
    message = Signal(str)

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.params = params
        import threading
        self._pause_evt = threading.Event()
        self._pause_evt.set()
        self._stop = False
        print("DEBUG slide_path =", params.get("slide_path"), "wsi_path =", params.get("wsi_path"))
        mi = params.get("model_info") or {}
        print("DEBUG model_path =", params.get("model_path"), params.get("weight"), mi.get("path"), mi.get("weight"))

    def pause(self):  self._pause_evt.clear()
    def resume(self): self._pause_evt.set()
    def is_paused(self): return not self._pause_evt.is_set()
    def cancel(self): self._stop = True; self._pause_evt.set()

    def _wait_if_paused(self):
        self._pause_evt.wait()
        if self._stop:
            raise RuntimeError("Cancelled by user")

    # ===== 新增: 工具函数 - 任意输入变 RGBA =====
    @staticmethod
    def _to_rgba_any(x, color=(255, 0, 255), alpha_const=0.4, pap_ids=None):
        """
        接受：
          - np.ndarray RGBA/RGB/灰度/二值/概率(2D/3D)
          - PIL.Image / 路径字符串
        返回 np.uint8 (H,W,4) 或 None
        """
        import numpy as _np
        from PIL import Image as _Image
        import os as _os

        if x is None:
            return None

        # 路径 -> 打开为 RGBA
        if isinstance(x, str) and _os.path.isfile(x):
            try:
                im = _Image.open(x)
                if im.mode != "RGBA":
                    im = im.convert("RGBA")
                return _np.array(im, dtype=_np.uint8)
            except Exception:
                return None

        # PIL.Image -> RGBA
        try:
            from PIL.Image import Image as _PIL
            if isinstance(x, _PIL):
                im = x.convert("RGBA")
                return _np.array(im, dtype=_np.uint8)
        except Exception:
            pass

        arr = _np.asarray(x)
        if arr.ndim == 3 and arr.shape[2] == 4:
            return arr.astype(_np.uint8)

        if arr.ndim == 3 and arr.shape[2] == 3:
            H, W = arr.shape[:2]
            a = _np.full((H, W, 1), int(alpha_const * 255), dtype=_np.uint8)
            return _np.concatenate([arr.astype(_np.uint8), a], axis=2)

        if arr.ndim == 2:
            # 概率/灰度（0~1 或 0~255） -> 固定颜色 + alpha=prob*alpha_const
            v = arr.astype(_np.float32)
            if v.max() > 1.0:
                v = v / 255.0
            v = _np.clip(v, 0.0, 1.0)
            H, W = v.shape
            rgba = _np.zeros((H, W, 4), dtype=_np.uint8)
            rgba[..., 0] = color[0]
            rgba[..., 1] = color[1]
            rgba[..., 2] = color[2]
            rgba[..., 3] = (v * (alpha_const * 255)).astype(_np.uint8)
            return rgba

        if arr.ndim == 3 and arr.shape[2] > 4:
            # 可能是 per-class 概率/对数几率( H,W,C )
            C = arr.shape[2]
            ids = pap_ids if (isinstance(pap_ids, (list, tuple)) and len(pap_ids) > 0) else list(range(C))
            ids = [i for i in ids if 0 <= int(i) < C]
            if not ids:
                ids = list(range(C))

            v = arr[..., ids].astype(_np.float32)
            # 粗略判断是否 logits：若绝对值很大
            if _np.nanmax(_np.abs(v)) > 5.0:
                v = 1.0 / (1.0 + _np.exp(-v))
            # 合并为一张概率图
            prob = _np.max(v, axis=2)
            return ClsWorker._to_rgba_any(prob, color=color, alpha_const=alpha_const)

        return None

    @staticmethod
    def _extract_overlay_and_meta(res, color, alpha_const, pap_ids):
        """
        兼容各种 classify_slide 返回：
          - dict: 可能含 overlay_rgba/rgba/overlay/mask/prob/heatmap/overlay_path/meta/...
          - (rgba, meta)
        """
        import numpy as _np
        rgba = None
        meta = {}

        if isinstance(res, dict):
            # 先直接拿常见显式 RGBA
            for k in ("overlay_rgba", "mask_rgba", "rgba", "overlay"):
                if k in res:
                    rgba = ClsWorker._to_rgba_any(res[k], color=color, alpha_const=alpha_const, pap_ids=pap_ids)
                    if rgba is not None:
                        break
            # 再试概率/掩码/热图等
            if rgba is None:
                for k in ("prob", "prob_map", "score", "score_map", "heatmap", "mask", "overlay_path"):
                    if k in res:
                        rgba = ClsWorker._to_rgba_any(res[k], color=color, alpha_const=alpha_const, pap_ids=pap_ids)
                        if rgba is not None:
                            break
            meta = res.get("meta") or {}
        elif isinstance(res, (list, tuple)):
            if len(res) >= 1:
                rgba = ClsWorker._to_rgba_any(res[0], color=color, alpha_const=alpha_const, pap_ids=pap_ids)
            if len(res) >= 2 and isinstance(res[1], dict):
                meta = res[1]

        # 保证 RGBA 类型正确
        if rgba is not None:
            rgba = _np.asarray(rgba)
            if rgba.ndim == 3 and rgba.shape[2] == 4:
                rgba = rgba.astype(_np.uint8)
            else:
                rgba = None
        return rgba, (meta or {})

    def run(self):
        try:
            from wsi.reader import WsiReader
            from models.manager import load_model
            from inference.engine import classify_slide
            import time, os

            slide_path = self.params["slide_path"]
            model_info = self.params["model_info"]
            level = int(self.params["level"])
            roi_level0 = self.params["roi_level0"]  # None or (x0,y0,w0,h0)
            patch = int(self.params["patch"])
            overlap = int(self.params["overlap"])
            batch = int(self.params["batch"])
            mean = self.params["mean"]; std = self.params["std"]
            classes = self.params["classes"]
            pap_ids = (
                self.params.get("pap_ids")
                or self.params.get("papillary_ids")
                or self.params.get("papillary_related_ids")
                or []
            )
            thr = float(self.params["threshold"])
            prefer_gpu = bool(self.params.get("prefer_gpu", True))
            amp_pref   = str(self.params.get("amp_pref", "auto")).lower()

            if amp_pref == "auto":
                amp = None
            elif amp_pref in ("true", "cuda", "gpu", "fp16", "half"):
                amp = True
            elif amp_pref in ("false", "cpu", "never", "fp32"):
                amp = False
            else:
                amp = None

            reader = WsiReader().open(slide_path)
            # 这里取权重：优先 model_info.weight/path
            weight_path = (
                model_info.get("weight")
                or model_info.get("path")
                or model_info.get("weights")
                or model_info.get("checkpoint")
            )
            runner, meta = load_model(
                weight_path,
                arch=model_info.get("arch", "cls_res"),
                num_classes=len(classes),
                prefer_gpu=prefer_gpu,
                amp=amp
            )
            try:
                self.message.emit(f"使用设备: {meta.get('device')}  amp={meta.get('amp')}  arch={meta.get('arch')}")
            except Exception:
                pass

            last_pct = -1; last_ts = 0.0
            def on_prog(done, total):
                if self.isInterruptionRequested():
                    raise RuntimeError("用户取消")
                self._wait_if_paused()
                import time as _t
                pct = int(done * 100 / max(total, 1))
                now = _t.perf_counter()
                nonlocal last_pct, last_ts
                if pct != last_pct or (now - last_ts) >= 0.2:
                    self.progress.emit(done, total)
                    last_pct = pct; last_ts = now

            # === 计时开始 ===
            t0 = time.perf_counter()

            res = classify_slide(
                reader, runner,
                classes=classes, papillary_ids=pap_ids, threshold=thr,
                level=level, target_mag=None,
                patch_size=patch, overlap=overlap, batch_size=batch,
                roi_level0=roi_level0,
                mean=mean, std=std, progress=on_prog,
                color=tuple(self.params.get("color", (255, 0, 255))),
                use_prob_alpha=bool(self.params.get("use_prob_alpha", False)),
                alpha_const=float(self.params.get("alpha_const", 0.4)),
            )

            color = tuple(self.params.get("color", (255, 0, 255)))
            alpha_const = float(self.params.get("alpha_const", 0.4))

            # 统一抽取/组装 overlay 和 meta（关键修复）
            overlay_rgba, meta_out = self._extract_overlay_and_meta(res, color, alpha_const, pap_ids)

            # 如果还没有 overlay，则直接报错提示，让你能知道 classify_slide 的返回形态
            if overlay_rgba is None:
                raise RuntimeError("classify_slide 未返回可用 overlay（未找到 overlay_rgba/rgba/overlay/"
                                   "prob/heatmap/mask/overlay_path 等）。请打印 classify_slide 的返回结构进行适配。")

            # === 计时结束 ===
            t1 = time.perf_counter()
            print(f"[CLS] 推理完成，用时 {t1 - t0:.2f}s | device={meta.get('device')} | "
                  f"level={level} patch={patch} overlap={overlap} batch={batch} | slide='{os.path.basename(slide_path)}'")

            # 补充一些 meta 字段
            meta_out = dict(meta_out or {})
            meta_out.setdefault("level", level)
            meta_out.setdefault("threshold", thr)
            meta_out["roi_level0"] = roi_level0
            meta_out["patch_size_level"] = int(patch)
            meta_out["stride_size_level"] = int(max(1, patch - overlap))

            try: reader.close()
            except Exception: pass

            # 计算该 level 的 downsample，并补齐 bbox_level0
            try:
                ds_list = [float(d) for d in reader.level_downsamples]
                ds0 = ds_list[level] if 0 <= level < len(ds_list) else 1.0
            except Exception:
                ds0 = 1.0

            H, W = overlay_rgba.shape[:2]
            meta_out.setdefault("downsample", float(ds0))
            if not meta_out.get("bbox_level0"):
                meta_out["bbox_level0"] = [0, 0, int(round(W * ds0)), int(round(H * ds0))]

            # stride / patch / 网格（overlay 像素坐标下）
            stride_lv = int(meta_out.get("stride_size_level", max(1, patch - overlap)))
            meta_out["stride_size_level"] = stride_lv
            meta_out.setdefault("patch_size_level", int(patch))
            meta_out.setdefault("grid_tile_px_on_overlay", stride_lv)

            # 统一回传 dict，Dialog 就能拿到 overlay 了
            self.finished_ok.emit({"overlay_rgba": overlay_rgba, "meta": meta_out})

        except Exception as e:
            self.failed.emit(str(e))



class SaveOverlayWorker(QThread):
    """
    只导出 overlay，并写入 *_meta.json
    """
    finished_ok = Signal(dict)   # {'overlay': path, 'overlay_meta': path}
    failed = Signal(str)

    def __init__(
        self,
        overlay_rgba: Optional[np.ndarray],
        slide_path: str,          # 用于命名和记录
        out_dir: str,
        level: int,
        ds_at_level: float,       # 该 overlay 对应的 downsample（若显示降采样过，已乘进来）
        roi_level0: Optional[tuple] = None,
        max_side: int = _MAX_PNG_SIDE,
        parent=None,
        patch_size_level: Optional[int] = None,        # 推理时的 patch（像素，针对当前 level）
        grid_tile_px_on_overlay: Optional[int] = None, # 在“导出的 overlay 像素坐标下”的默认网格大小
        stride_size_level: Optional[int] = None,       # 推理步长（像素，针对当前 level）
    ):
        super().__init__(parent)
        self.overlay = overlay_rgba
        self.slide_path = slide_path
        self.out_dir = os.path.abspath(out_dir)
        self.level = int(level)
        self.ds_at_level = float(ds_at_level)
        self.roi = tuple(roi_level0) if roi_level0 else None
        self.max_side = int(max_side)

        # 额外信息（写入 meta）
        self.patch_size_level = int(patch_size_level) if patch_size_level else None
        self.grid_tile_px_on_overlay = int(grid_tile_px_on_overlay) if grid_tile_px_on_overlay else None
        self.stride_size_level = int(stride_size_level) if stride_size_level else None

    def _base_from_slide(self) -> str:
        return os.path.splitext(os.path.basename(self.slide_path or "slide"))[0]

    def _downsample_arr(self, arr: np.ndarray) -> tuple[np.ndarray, int]:
        if arr is None:
            return None, 1

        # 计算 2/4/8… 的缩放因子，限制最大边不超过 self.max_side
        if arr.ndim == 3:
            h, w = arr.shape[:2]
        else:
            h, w = arr.shape
        sf, side = 1, max(h, w)
        while side > self.max_side:
            sf *= 2
            side = (side + 1) // 2

        if sf == 1:
            return arr, 1

        # 使用最近邻（NEAREST）避免双线性带来的半透明“毛边”
        if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
            # 灰度/掩膜
            im = Image.fromarray(arr.squeeze(), mode="L").resize((w // sf, h // sf), Image.NEAREST)
            out = np.array(im, dtype=arr.dtype)
        else:
            # 颜色（期望 RGBA），若是 RGB 则补 alpha
            if arr.ndim == 3 and arr.shape[2] == 3:
                pil = Image.fromarray(arr, mode="RGB").convert("RGBA")
            else:
                pil = Image.fromarray(arr, mode="RGBA")
            im = pil.resize((w // sf, h // sf), Image.NEAREST)
            out = np.array(im, dtype=arr.dtype)

        return out, sf

    def _write_meta(self, png_path: str, ds_out: float, target: str, extra: dict | None = None) -> str:
        meta = {
            "level": self.level,
            "downsample": float(ds_out),
            "bbox_level0": list(self.roi) if self.roi else None,
            "target": target,
            "path": self.slide_path,
        }
        if extra:
            meta.update(extra)
        meta_path = os.path.splitext(png_path)[0] + "_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return meta_path

    def run(self):
        try:
            if self.overlay is None:
                raise RuntimeError("没有可保存的 overlay。")

            os.makedirs(self.out_dir, exist_ok=True)
            base = self._base_from_slide()
            out = {}

            # 处理 overlay（可能降采样）
            ov_ds, sf_ov = self._downsample_arr(self.overlay)
            ds_ov = self.ds_at_level * sf_ov
            p_ov = os.path.join(self.out_dir, f"{base}_overlay_L{self.level}.png")
            Image.fromarray(ov_ds, mode="RGBA").save(p_ov, **_PNG_SAVE_KW)
            out["overlay"] = p_ov

            patch_out  = self.patch_size_level
            stride_out = self.stride_size_level

            # 只有 grid_tile_px_on_overlay 根据 PNG 的缩小倍数调整
            grid_out = None
            if self.grid_tile_px_on_overlay:
                grid_out = max(1, int(round(self.grid_tile_px_on_overlay / sf_ov)))

            extra_ov = {}
            if patch_out:
                extra_ov["patch_size_level"] = int(patch_out)          # level 像素
            if stride_out:
                extra_ov["stride_size_level"] = int(stride_out)        # level 像素
            if grid_out:
                extra_ov["grid_tile_px_on_overlay"] = int(grid_out)    # overlay 像素

            out["overlay_meta"] = self._write_meta(
                p_ov, ds_out=ds_ov, target="classification_overlay", extra=extra_ov
            )

            self.finished_ok.emit(out)
        except Exception as e:
            self.failed.emit(str(e))



class _ExportWorker(QThread):
    done = Signal(str, str)   # (kind, out_path)
    failed = Signal(str)

    def __init__(self, kind: str, array: np.ndarray, out_path: str, meta: dict, mode: str = "auto", parent=None):
        """
        kind: "overlay" | "mask"
        array: RGBA (H,W,4) 或 uint8 mask (H,W)
        out_path: 目标 PNG 路径
        meta: 要写入的 _meta.json
        mode: "auto" 即按 kind 自动挑 RGBA/L 保存
        """
        super().__init__(parent)
        self.kind = kind
        self.array = array
        self.out_path = out_path
        self.meta = dict(meta or {})
        self.mode = mode

    def run(self):
        try:
            os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
            if self.kind == "overlay":
                im = Image.fromarray(self.array, mode="RGBA")
                im.save(self.out_path, **_PNG_SAVE_KW)
            else:
                # 掩膜：保证是 8-bit L
                arr = self.array
                if arr.ndim == 3 and arr.shape[2] == 4:
                    arr = arr[..., 3]  # 容错
                im = Image.fromarray(arr.astype(np.uint8), mode="L")
                im.save(self.out_path, **_PNG_SAVE_KW)

            meta_path = os.path.splitext(self.out_path)[0] + "_meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self.meta, f, ensure_ascii=False, indent=2)

            self.done.emit(self.kind, self.out_path)
        except Exception as e:
            self.failed.emit(str(e))


# ===================== 页面 =====================
class ClsPage(QWidget):
    """
    分类工作台（医生友好版）
      - 左侧：参数区（常用 / 高级折叠 / 导出）
      - 右侧：WSI 预览（支持叠加分类着色）
      - 按钮：打开 / 整图分类 / 暂停-继续 / 关闭；“局部预览(当前视野)”为次要入口
      - 导出：仅导出叠加图 overlay（写入 meta.json）
    """
    def __init__(self, app_cfg, palette=None, models_cfg=None, logger=None, parent=None):
        super().__init__(parent)
        self.app_cfg = app_cfg
        self.palette = palette or {}
        self.models_cfg = models_cfg or []
        self.log = logger or _NullLogger()

        self.reader: Optional[WsiReader] = None
        self.overlay: Optional[OverlayItem] = None
        self.worker: Optional[ClsWorker] = None
        self._last_res: Optional[dict] = None  # 保存最近一次推理结果（导出用）

        # 右：WSI
        self.view = WsiView(app_cfg)
        self.overlay = None  # 推迟到真正加载 WSI 之后再创建

        # 左：控制面板
        left = self._build_left_panel()

        # 整体
        sp = QSplitter()
        sp.addWidget(left)
        sp.addWidget(self.view)
        sp.setStretchFactor(0, 0)
        sp.setStretchFactor(1, 1)
        sp.setSizes([360, 840])
        lay = QVBoxLayout(self)
        lay.addWidget(sp)

        # 快捷键（保持）
        self._shortcut_home = QShortcut(QKeySequence("H"), self.view)
        self._shortcut_home.activated.connect(self.view.zoom_to_fit)

    # ---------- 左侧 UI ----------
    def _build_left_panel(self) -> QWidget:
        host = QWidget()
        root = QVBoxLayout(host)
        root.setContentsMargins(10,10,10,10)
        root.setSpacing(8)

        # 常用
        grp_basic = QGroupBox("常用")
        f = QFormLayout(grp_basic)
        f.setLabelAlignment(Qt.AlignRight)
        f.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        f.setHorizontalSpacing(10)
        f.setVerticalSpacing(8)
        f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        # 模型
        self.cmbModel = QComboBox()
        self._fill_models_combobox()
        self.cmbModel.setMinimumWidth(220)
        self.cmbModel.currentIndexChanged.connect(self._on_model_changed)

        # WSI 路径
        self.edWSI = QLineEdit(); self.edWSI.setPlaceholderText("请选择 .svs / .tif ...")
        self.edWSI.setReadOnly(True); self.edWSI.setMinimumWidth(220)
        btnPickWSI = QPushButton("选择…"); btnPickWSI.clicked.connect(self._on_pick_wsi)
        row_wsi = _hline([self.edWSI, btnPickWSI], stretch_first=True)

        # 分辨率：Level / 目标倍率
        self.useMag = QCheckBox("按目标倍率")
        self.spnLevel = QSpinBox(); self.spnLevel.setRange(0, 16); self.spnLevel.setValue(0); self.spnLevel.setMinimumWidth(100)
        self.dspMag = QDoubleSpinBox(); self.dspMag.setRange(1.0, 80.0); self.dspMag.setDecimals(1); self.dspMag.setValue(10.0); self.dspMag.setSuffix("×"); self.dspMag.setMinimumWidth(120)
        self.useMag.stateChanged.connect(lambda _: self._sync_level_mag_enabled())
        lv_row = _hline(["level", self.spnLevel, self.useMag, "倍率", self.dspMag])

        # 阈值 & 透明度
        self.sldTh = QSlider(Qt.Horizontal); self.sldTh.setRange(0,100); self.sldTh.setValue(50); self.sldTh.setMinimumWidth(180)
        self.lblTh = QLabel("0.50")
        self.sldTh.valueChanged.connect(lambda v: self.lblTh.setText(f"{v/100:.2f}"))

        self.sldAlpha = QSlider(Qt.Horizontal); self.sldAlpha.setRange(0,100); self.sldAlpha.setValue(50); self.sldAlpha.setMinimumWidth(180)
        self.lblAlpha = QLabel("0.50")
        self.sldAlpha.valueChanged.connect(self._on_alpha_changed)

        f.addRow("模型", self.cmbModel)
        f.addRow("WSI", row_wsi)
        f.addRow("分辨率", lv_row)
        f.addRow("阈值", _hline([self.sldTh, self.lblTh]))
        f.addRow("叠加透明", _hline([self.sldAlpha, self.lblAlpha]))

        # ---- 主按钮行 ----
        row_btn1 = _hline([
            QPushButton("打开WSI"),
            QPushButton("整图分类"),
            QPushButton("暂停"),
            QPushButton("关闭WSI"),
        ])
        self.btnOpen, self.btnRun, self.btnPause, self.btnClose = row_btn1.widgets
        self.btnOpen.clicked.connect(self._open_wsi_clicked)
        self.btnRun.clicked.connect(self._on_run_fullslide)
        self.btnPause.clicked.connect(self._on_toggle_pause)
        self.btnClose.clicked.connect(self._on_close_wsi)
        for b in row_btn1.widgets: b.setMinimumWidth(100)

        # 次要入口：局部预览
        self.btnPreview = QToolButton()
        self.btnPreview.setText("局部预览（当前视野）")
        self.btnPreview.clicked.connect(self._on_preview_viewport)
        self.btnPreview.setToolButtonStyle(Qt.ToolButtonTextOnly)

        # 进度与状态
        self.lblStatus = QLabel("就绪"); self.lblStatus.setStyleSheet("color:#444;")
        self.prog = _progress_bar()

        root.addWidget(grp_basic)
        root.addWidget(row_btn1)
        root.addWidget(self.btnPreview, 0, Qt.AlignLeft)
        root.addWidget(self.prog)
        root.addWidget(self.lblStatus)

        # ---- 高级设置（折叠）----
        self.pnlAdv = _CollapsiblePanel("高级设置")
        adv = self.pnlAdv.bodyLayout

        self.spnPatch = QSpinBox(); self.spnPatch.setRange(64, 2048); self.spnPatch.setValue(224); self.spnPatch.setMinimumWidth(120)
        self.spnOverlap = QSpinBox(); self.spnOverlap.setRange(0, 512); self.spnOverlap.setValue(32); self.spnOverlap.setMinimumWidth(120)
        self.spnBatch = QSpinBox(); self.spnBatch.setRange(1, 64); self.spnBatch.setMinimumWidth(120)
        self.edArch = QLineEdit("cls_res"); self.edArch.setReadOnly(True); self.edArch.setMinimumWidth(160)

        infer_cfg = self.app_cfg.get("inference", {}) if isinstance(self.app_cfg, dict) else {}
        dev = pick_device(prefer_gpu=bool(self.app_cfg.get("hw", {}).get("prefer_gpu", True)))
        is_cuda = (getattr(dev, "type", "cpu") == "cuda")
        default_batch = int(infer_cfg.get("batch_size_gpu" if is_cuda else "batch_size_cpu", 8 if is_cuda else 2))
        self.spnBatch.setValue(default_batch)

        g = QFormLayout()
        g.setLabelAlignment(Qt.AlignRight)
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(6)
        g.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        g.addRow("patch", self.spnPatch)
        g.addRow("overlap", self.spnOverlap)
        g.addRow("batch", self.spnBatch)
        g.addRow("arch", self.edArch)
        adv.addLayout(g)

        root.addWidget(self.pnlAdv)

        # ---- 导出（仅 overlay）----
        grp_out = QGroupBox("导出")
        fo = QFormLayout(grp_out)
        fo.setLabelAlignment(Qt.AlignRight)
        fo.setHorizontalSpacing(10)
        fo.setVerticalSpacing(6)

        self.edOutDir = QLineEdit(os.path.abspath("./exports"))  # 默认 exports
        btnPickOut = QPushButton("选择…")
        btnPickOut.clicked.connect(self._on_pick_outdir)
        fo.addRow("导出到", _hline([self.edOutDir, btnPickOut], stretch_first=True))

        btnExportAll = QPushButton("导出叠加图（overlay）")
        btnExportAll.setMinimumWidth(180)
        btnExportAll.clicked.connect(self._export_all_clicked)
        fo.addRow("", _hline([btnExportAll]))
        self.btnExportAll = btnExportAll

        root.addWidget(grp_out)
        root.addStretch(1)

        # --- 初始化一次，让 models.json 的 input_size/threshold 等同步到 UI ---
        if self.cmbModel.count() > 0:
            self._on_model_changed(self.cmbModel.currentIndex())

        # 初始
        self._sync_level_mag_enabled()
        self._update_buttons(active=False)
        return host

    # ---------- 模型列表 ----------
    def _fill_models_combobox(self):
        self.cmbModel.clear()
        for m in self.models_cfg:
            t = (m.get("type") or "").lower()
            if "classif" in t or t == "":
                name = m.get("name", m.get("id","model"))
                self.cmbModel.addItem(name, userData=m)

    def _on_model_changed(self, idx: int):
        info = self.cmbModel.currentData()
        if not isinstance(info, dict):
            return
        sz = info.get("input_size")
        if isinstance(sz, int) and 64 <= sz <= 2048:
            self.spnPatch.setValue(sz)
        th = info.get("threshold")
        if isinstance(th, (int,float)):
            v = max(0.0, min(1.0, float(th)))
            self.sldTh.setValue(int(round(v*100)))
        self.model_mean = info.get("mean") or [0.485,0.456,0.406]
        self.model_std  = info.get("std")  or [0.229,0.224,0.225]
        self.pap_ids    = list(info.get("papillary_related_ids", []))
        self.classes    = list(info.get("classes", []))
        self.edArch.setText(info.get("arch","cls_res"))
        self.lblStatus.setText("模型已切换")

    # ---------- 使能 ----------
    def _sync_level_mag_enabled(self):
        use_mag = self.useMag.isChecked()
        self.spnLevel.setEnabled(not use_mag)
        self.dspMag.setEnabled(use_mag)

    def _update_buttons(self, active: bool):
        # 基本按钮
        for b in (self.btnRun, self.btnClose, self.btnPreview, self.btnPause):
            b.setEnabled(active)
        self.btnOpen.setEnabled(True)
        if not active:
            self.btnPause.setText("暂停")
            self.btnPause.setEnabled(False)

        # 导出按钮：只有当有 _last_res 时可用
        has_res = bool(self._last_res)
        self.btnExportAll.setEnabled(has_res)

    def _on_alpha_changed(self, v: int):
        a = v/100.0
        self.lblAlpha.setText(f"{a:.2f}")
        try:
            if self.overlay:
                self.overlay.setOpacity(a)
        except Exception:
            pass

    def _ensure_overlay(self):
        """确保 overlay 存在并加入当前场景；若已被场景销毁则重建。"""
        try:
            scene = self.view.scene()
        except Exception:
            scene = None
        if scene is None:
            return
        if (self.overlay is None) or (self.overlay.scene() is None):
            try:
                if self.overlay and self.overlay.scene():
                    try:
                        self.overlay.scene().removeItem(self.overlay)
                    except Exception:
                        pass
            except Exception:
                pass
            from ui.viewer.overlay_item import OverlayItem
            self.overlay = OverlayItem()
            self.overlay.setZValue(10)
            scene.addItem(self.overlay)
            self.overlay.setOpacity(self.sldAlpha.value() / 100.0)

    def _discard_overlay(self):
        """把 overlay 从场景移除并置空引用，避免悬空对象被继续调用。"""
        try:
            if self.overlay is not None:
                sc = self.view.scene()
                if sc and (self.overlay.scene() is sc):
                    sc.removeItem(self.overlay)
        except Exception:
            pass
        self.overlay = None

    # ---------- 文件 ----------
    def _on_pick_wsi(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 WSI", "",
            "Slides/Images (*.svs *.tif *.tiff *.mrxs *.ndpi *.scn *.svslide)")
        if path:
            self.edWSI.setText(path)

    def _open_wsi_clicked(self):
        path = self.edWSI.text().strip()
        if not path:
            self._on_pick_wsi(); path = self.edWSI.text().strip()
        if not path: return
        try:
            reader = WsiReader().open(path)
            self.reader = reader
            self.view.load_reader(reader)
            self._discard_overlay()
            self._ensure_overlay()
            if self.overlay:
                self.view.scene().addItem(self.overlay)
                self.overlay.setZValue(10)
                self.overlay.setOpacity(self.sldAlpha.value()/100.0)
                self.overlay.setVisible(False)
                self.overlay.setPos(0, 0); self.overlay.setScale(1.0)
            self._last_res = None
            self._update_buttons(active=True)
            self.lblStatus.setText("WSI 已打开")
        except Exception as e:
            QMessageBox.critical(self, "打开失败", str(e))
            self.lblStatus.setText("打开失败")

    def _on_close_wsi(self):
        try:
            if hasattr(self, "worker") and self.worker and self.worker.isRunning():
                ret = QMessageBox.question(
                    self, "关闭确认",
                    "当前仍在推理（可能处于暂停）。是否中止推理并关闭？",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                if ret != QMessageBox.Yes:
                    return
                try:
                    try:
                        if self.worker and self.worker.is_paused():
                            self.worker.resume()  # 先确保唤醒
                    except Exception:
                        pass
                    try:
                        if self.worker:
                            self.worker.requestInterruption()
                    except Exception:
                        pass
                except Exception:
                    pass
                try:
                    self.worker.requestInterruption()
                except Exception:
                    pass
                self.worker.wait(3000)
                if self.worker.isRunning():
                    self.worker.terminate()
                    self.worker.wait(1000)
                self.worker = None
        except Exception:
            pass

        try:
            if self.overlay:
                try:
                    self.view.scene().removeItem(self.overlay)
                except Exception:
                    pass
        except Exception:
            pass
        if self.view:
            self.view.unload()
        try:
            if self.reader:
                self.reader.close()
        except Exception:
            pass
        self.reader = None
        self._update_buttons(active=False)
        self.lblStatus.setText("已关闭")

    # ---------- 分类 ----------
    def _current_visible_level0_bbox(self) -> Optional[QRectF]:
        vr = self.view.viewport().rect()
        if vr.isEmpty(): return None
        rect = self.view.mapToScene(vr).boundingRect()
        if rect.width() <= 0 or rect.height() <= 0:
            return None
        return rect

    def _resolve_level(self) -> int:
        if not self.reader: return 0
        if self.useMag.isChecked():
            obj = self.reader.objective_power()
            ds_list = [float(d) for d in self.reader.level_downsamples]
            if obj and obj > 0:
                target_ds = obj / float(self.dspMag.value())
                idx = int(min(range(len(ds_list)), key=lambda i: abs(ds_list[i] - target_ds)))
                return idx
        return int(self.spnLevel.value())

    def _run_classify(self, roi_level0_rect: Optional[QRectF]):
        if not self.reader:
            QMessageBox.information(self, "提示", "请先打开一张 WSI。"); return
        info = self.cmbModel.currentData() or {}
        weight = info.get("weight")
        if not weight or not os.path.isfile(weight):
            QMessageBox.warning(self, "提示", "模型权重未找到，请检查 configs/models.json 的 weight 路径。"); return

        classes = list(info.get("classes", [])) or [f"class{i}" for i in range(6)]
        pap_ids = list(info.get("papillary_related_ids", []))
        threshold = self.sldTh.value()/100.0

        level = self._resolve_level()
        patch = int(self.spnPatch.value())
        overlap = int(self.spnOverlap.value())
        batch = int(self.spnBatch.value())
        mean = getattr(self, "model_mean", [0.485,0.456,0.406])
        std  = getattr(self, "model_std",  [0.229,0.224,0.225])

        hw_cfg = self.app_cfg.get("hw", {})
        inf_cfg = self.app_cfg.get("inference", {})

        roi_tuple = None
        if roi_level0_rect is not None:
            x0 = max(0, int(roi_level0_rect.left()))
            y0 = max(0, int(roi_level0_rect.top()))
            w0 = max(1, int(roi_level0_rect.width()))
            h0 = max(1, int(roi_level0_rect.height()))
            roi_tuple = (x0,y0,w0,h0)

        self._update_buttons(active=False)
        self.prog.setRange(0, 100); self.prog.setValue(0)
        self.lblStatus.setText("推理中…")

        params = dict(
            prefer_gpu=bool(hw_cfg.get("prefer_gpu", True)),
            amp_pref=str(inf_cfg.get("use_fp16", "auto")),
            slide_path=self.edWSI.text().strip(),
            model_info=info, level=level, roi_level0=roi_tuple,
            patch=patch, overlap=overlap, batch=batch,
            mean=mean, std=std, classes=classes, pap_ids=pap_ids,
            threshold=threshold
        )

        # 颜色/透明度（固定 alpha）
        pal = self.palette if isinstance(self.palette, dict) else {}
        alpha_const = float(pal.get("papillary_patch_alpha", 0.4))
        params.update(dict(
            color=tuple(pal.get("papillary_patch_color", [255, 0, 255])),
            use_prob_alpha=False,  # 不用概率映射深浅
            alpha_const=alpha_const
        ))

        self.worker = ClsWorker(params, self)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished_ok.connect(self._on_worker_done)
        self.worker.failed.connect(self._on_worker_failed)
        self.btnPause.setText("暂停");
        self.btnPause.setEnabled(True)
        self.worker.start()

    # === 线程回调 ===
    def _on_worker_progress(self, done: int, total: int):
        pct = int(done*100/max(total,1))
        self.prog.setValue(pct)
        self.lblStatus.setText(f"推理中… {pct}% ({done}/{total})")

    def _on_worker_done(self, res: dict):
        try:
            res = res or {}
            rgba = res.get("overlay_rgba")
            meta = res.get("meta") or {}
            if rgba is None:
                raise RuntimeError("工作线程未返回 overlay_rgba")

            # 若未打开WSI，给个友好提示后结束
            if not self.reader:
                self._last_res = {"overlay_rgba": rgba, "meta": dict(meta)}
                self.lblStatus.setText("完成（WSI 已关闭，未显示叠加）")
                self.prog.setRange(0, 100);
                self.prog.setValue(100)
                QMessageBox.information(self, "分类完成", "分类已完成，但当前未打开 WSI，未显示叠加。")
                return

            # 对齐参数
            lv = int(meta.get("level", 0))
            roi = meta.get("roi_level0") or meta.get("bbox_level0")
            try:
                ds_list = [float(d) for d in self.reader.level_downsamples]
                ds0 = ds_list[lv] if 0 <= lv < len(ds_list) else 1.0
            except Exception:
                ds0 = 1.0

            # 为显示/导出做一次“按 2/4/8…”的可视化降采样（边界清晰，用 NEAREST）
            ov_disp, ds_disp, sf_disp = self._downsample_for_export(rgba, ds0)

            # stride/patch（优先线程meta；否则用UI设置计算）
            stride_level = int(
                meta.get("stride_size_level", max(1, int(self.spnPatch.value()) - int(self.spnOverlap.value()))))
            patch_level = int(meta.get("patch_size_level", int(self.spnPatch.value())))

            # “在 overlay 像素坐标下”的默认格子尺寸（便于后续编辑/导出）
            grid_on_overlay = max(1, int(round(stride_level / max(1, sf_disp))))

            # —— 更新显示层 —— #
            self._ensure_overlay()
            if self.overlay and (self.overlay.scene() is not None):
                self.overlay.set_rgba(ov_disp)
                self.overlay.setScale(ds_disp)
                x0 = y0 = 0
                if isinstance(roi, (list, tuple)) and len(roi) >= 2:
                    x0, y0 = int(roi[0]), int(roi[1])
                self.overlay.setPos(x0, y0)
                self.overlay.setVisible(True)
                self.overlay.setOpacity(self.sldAlpha.value() / 100.0)

            # —— 缓存导出所需信息 —— #
            self._last_res = {
                "overlay_rgba": rgba,  # 原始线程返回
                "meta": dict(meta),
                # 显示/导出用：
                "overlay": ov_disp,  # 当前显示的（可能已降采样）
                "display_ds": float(ds_disp),  # 与 overlay 对应的 downsample
                "level": lv,
                "roi_level0": roi,
                "patch_size_level": patch_level,
                "stride_size_level": stride_level,
                "grid_tile_px_on_overlay": grid_on_overlay,
            }

            self.lblStatus.setText("完成")
            self.prog.setRange(0, 100);
            self.prog.setValue(100)
            QMessageBox.information(self, "分类完成", "分类已完成并已叠加到视图。\n可在“导出”区域保存结果。")

        except Exception as e:
            QMessageBox.warning(self, "显示结果出错", str(e))
        finally:
            self._update_buttons(active=True)
            if hasattr(self, "btnPause") and self.btnPause:
                try:
                    self.btnPause.setText("暂停")
                    self.btnPause.setEnabled(False)
                except Exception:
                    pass
            self.worker = None

    def _current_meta_for_export(self) -> dict:
        """导出用 meta：level/downsample/objective/ROI/原图路径"""
        meta = {}
        try:
            lv = int(self._last_res.get("level", 0)) if hasattr(self, "_last_res") else 0
            ds = float(self.reader.level_downsamples[lv]) if self.reader else 1.0
            obj = self.reader.objective_power() if self.reader else None
            roi = self._last_res.get("roi_level0") if hasattr(self, "_last_res") else None
            meta = {
                "level": lv,
                "downsample": ds,
                "objective": float(obj) if (obj is not None) else None,
                "roi_level0": roi,
                "path": getattr(self.reader, "path", None),
            }
        except Exception:
            pass
        return meta

    # 旧的 “仅导出 overlay” 接口，这里直接走一键导出，避免两套逻辑
    def _export_overlay_clicked(self):
        self._export_all_clicked()

    def _on_save_done(self, out: dict):
        self._update_buttons(active=True)
        self.lblStatus.setText("保存完成")
        msg_lines = []
        if 'overlay' in out: msg_lines.append(f"overlay: {out['overlay']}")
        if 'meta' in out: msg_lines.append(f"meta: {out['meta']}")
        QMessageBox.information(self, "保存完成", "已保存：\n" + "\n".join(msg_lines))

    def _on_save_failed(self, err: str):
        self._update_buttons(active=True)
        self.lblStatus.setText("保存失败")
        QMessageBox.critical(self, "保存失败", err)

    def _on_worker_failed(self, msg: str):
        QMessageBox.critical(self, "推理失败", msg)
        self.lblStatus.setText("失败")
        self.prog.setValue(0)
        self._update_buttons(active=True)
        self.btnPause.setText("暂停"); self.btnPause.setEnabled(False)
        self.worker = None

    # 暂停/继续
    def _on_toggle_pause(self):
        if not self.worker: return
        if self.worker.is_paused():
            self.worker.resume()
            self.btnPause.setText("暂停")
            self.lblStatus.setText("继续…")
        else:
            self.worker.pause()
            self.btnPause.setText("继续")
            self.lblStatus.setText("已暂停")

    # ---------- 预览 / 全图 ----------
    def _on_preview_viewport(self):
        rect = self._current_visible_level0_bbox()
        if rect is None:
            QMessageBox.information(self, "提示", "当前没有有效视野。"); return
        self._run_classify(rect)

    def _on_run_fullslide(self):
        self._run_classify(None)

    def _export_all_clicked(self):
        """一键导出 overlay 到 ./exports 或面板指定目录；不再导出 mask"""
        if not getattr(self, "_last_res", None):
            QMessageBox.information(self, "提示", "还没有可导出的结果。")
            return
        res = self._last_res
        ov = res.get("overlay")
        if ov is None:
            QMessageBox.information(self, "提示", "没有可导出的叠加图（overlay）。")
            return
        if not self.reader:
            QMessageBox.warning(self, "提示", "WSI 未打开，无法确认对齐信息。")
            return

        # 层级与 downsample
        lv = int(res.get("level", 0))
        ds_base = float(self.reader.level_downsamples[lv])
        ds_for_export = float(res.get("display_ds", ds_base))  # 若显示时做过降采样，则优先用显示 downsample
        roi = res.get("roi_level0")

        patch_level = int(self._last_res.get("patch_size_level", self.spnPatch.value()))
        stride_level = int(self._last_res.get("stride_size_level", max(1, patch_level - int(self.spnOverlap.value()))))

        # 计算“在导出的 overlay 像素坐标下”的默认网格大小
        sf_disp = max(1e-9, ds_for_export / ds_base)
        tile_on_overlay = max(1, int(round(stride_level / sf_disp)))

        out_dir = self.edOutDir.text().strip() or "./exports"
        out_dir = os.path.abspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        slide_path = self.edWSI.text().strip()

        # 禁用按钮并提示
        self._update_buttons(active=False)
        self.lblStatus.setText("正在保存（后台进行）…")

        self._saveWorker = SaveOverlayWorker(
            overlay_rgba=ov,
            slide_path=slide_path,
            out_dir=out_dir,
            level=lv,
            ds_at_level=ds_for_export,
            roi_level0=roi,
            patch_size_level=patch_level,
            grid_tile_px_on_overlay=tile_on_overlay,
            stride_size_level=stride_level,
            parent=self
        )
        self._saveWorker.finished_ok.connect(self._on_export_all_done)
        self._saveWorker.failed.connect(self._on_export_all_failed)
        self._saveWorker.start()

    def _on_pick_outdir(self):
        d = QFileDialog.getExistingDirectory(self, "选择导出目录", self.edOutDir.text() or "./exports")
        if d:
            self.edOutDir.setText(d)

    def _ensure_outdir(self) -> str:
        out_dir = self.edOutDir.text().strip() or "./exports"
        out_dir = os.path.abspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _wsibase(self) -> str:
        p = self.edWSI.text().strip()
        return os.path.splitext(os.path.basename(p))[0] if p else "slide"

    def _suffix_from_res(self) -> str:
        lv = int(self._last_res.get("level", 0))
        roi = self._last_res.get("roi_level0")
        suf = f"_L{lv}"
        if roi:
            x0,y0,w0,h0 = roi
            suf += f"_roi_{x0}_{y0}_{w0}_{h0}"
        return suf

    def _write_meta(self, base_png_path: str, kind: str) -> str:
        """写一个 _meta.json；返回路径"""
        meta = {
            "kind": kind,  # overlay_rgba 或 mask
            "level": int(self._last_res.get("level", 0)),
            "downsample": float(self.reader.level_downsamples[int(self._last_res.get("level",0))]) if self.reader else None,
            "roi_level0": self._last_res.get("roi_level0"),
            "path": self.edWSI.text().strip()
        }
        meta_path = os.path.splitext(base_png_path)[0] + "_meta.json"
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.warning(self, "写入元数据失败", str(e))
        return meta_path

    def _on_export_all_done(self, out: dict):
        self._update_buttons(active=True)
        self.lblStatus.setText("保存完成")
        lines = []
        if "overlay" in out:      lines.append(f"overlay: {out['overlay']}")
        if "overlay_meta" in out: lines.append(f"overlay_meta: {out['overlay_meta']}")
        if not lines: lines.append("（没有文件被保存）")
        QMessageBox.information(self, "导出完成", "已保存：\n" + "\n".join(lines))

    def _on_export_all_failed(self, err: str):
        self._update_buttons(active=True)
        self.lblStatus.setText("保存失败")
        QMessageBox.critical(self, "导出失败", err)

    def _downsample_for_export(self, arr: np.ndarray, ds: float) -> tuple[np.ndarray, float, int]:
        """
        若图过大，按 2/4/8… 倍降采样，并把 downsample 同步乘以该倍数。
        返回 (arr_out, ds_out, scale_factor)
        """
        h, w = (arr.shape[0], arr.shape[1]) if arr.ndim == 3 else arr.shape
        sf = 1
        side = max(h, w)
        while side > _MAX_PNG_SIDE:
            sf *= 2
            side = (side + 1) // 2

        if sf == 1:
            return arr, float(ds), 1

        # ui/pages/cls_page.py  —— 替换 _downsample_for_export 中的 RGBA resize
        if arr.ndim == 2:
            im = Image.fromarray(arr, mode="L")
            im = im.resize((w // sf, h // sf), resample=Image.NEAREST)
            out = np.array(im, dtype=arr.dtype)
        else:
            im = Image.fromarray(arr, mode="RGBA")
            im = im.resize((w // sf, h // sf), resample=Image.NEAREST)  # ← 原来是 BILINEAR
            out = np.array(im, dtype=arr.dtype)

        return out, float(ds) * sf, sf


# ---------- 小部件工具 ----------
def _hline(widgets: List[object], stretch_first: bool = False) -> QWidget:
    from PySide6.QtWidgets import QLabel, QHBoxLayout
    row = QWidget()
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0,0,0,0)
    lay.setSpacing(6)
    items = []
    for i, w in enumerate(widgets):
        if isinstance(w, str):
            from PySide6.QtWidgets import QLabel as _QLabel
            w = _QLabel(w)
        items.append(w)
        lay.addWidget(w, 1 if (stretch_first and i == 0) else 0)
    row.widgets = items
    return row

def _progress_bar():
    from PySide6.QtWidgets import QProgressBar
    pb = QProgressBar()
    pb.setRange(0,100); pb.setValue(0)
    pb.setTextVisible(True)
    return pb

class _CollapsiblePanel(QGroupBox):
    def __init__(self, title="高级设置"):
        super().__init__(title)
        from PySide6.QtWidgets import QVBoxLayout, QWidget
        self._container = QWidget()
        self.bodyLayout = QVBoxLayout(self._container)
        self.bodyLayout.setContentsMargins(0,6,0,0)
        lay = QVBoxLayout(self); lay.setContentsMargins(6,10,6,6); lay.setSpacing(4)
        lay.addWidget(self._container)
        self.setCheckable(True); self.setChecked(False)
        self.toggled.connect(self._container.setVisible)
        self._container.setVisible(False)

class _NullLogger:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass
    def exception(self, *a, **k): pass
