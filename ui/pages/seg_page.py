# ui/pages/seg_page.py
from __future__ import annotations
import os
from typing import Optional, Tuple

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox, QProgressBar,
    QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal

from wsi.reader import WsiReader
from models.manager import load_model
from inference.engine import segment_slide, save_mask


class SegWorker(QThread):
    progress = Signal(int, int)     # done, total
    message = Signal(str)
    finished_ok = Signal(str)       # out path
    failed = Signal(str)

    def __init__(
        self,
        wsi_path: str,
        model_path: str,
        arch: Optional[str],
        num_classes: int,
        out_dir: str,
        level: Optional[int],
        target_mag: Optional[float],
        patch: int, overlap: int, batch: int,
        roi_level0: Optional[Tuple[int,int,int,int]],
        prefer_gpu: bool = True,
        amp: Optional[bool] = None,
        parent=None
    ):
        super().__init__(parent)
        self.params = {
            "wsi_path": wsi_path, "model_path": model_path, "arch": arch,
            "num_classes": num_classes, "out_dir": out_dir, "level": level,
            "target_mag": target_mag, "patch": patch, "overlap": overlap, "batch": batch,
            "roi_level0": roi_level0, "prefer_gpu": prefer_gpu, "amp": amp
        }

    def run(self):
        try:
            wsi_path = self.params["wsi_path"]
            model_path = self.params["model_path"]
            arch = self.params["arch"]
            num_classes = int(self.params["num_classes"])
            level = self.params["level"]
            target_mag = self.params["target_mag"]
            patch = int(self.params["patch"])
            overlap = int(self.params["overlap"])
            batch = int(self.params["batch"])
            roi = self.params["roi_level0"]
            out_dir = self.params["out_dir"]

            reader = WsiReader().open(wsi_path)
            runner, meta = load_model(
                model_path,
                arch=arch,
                num_classes=num_classes,
                prefer_gpu=self.params["prefer_gpu"],
                amp=self.params["amp"]
            )

            def on_prog(done, total):
                self.progress.emit(done, total)

            res = segment_slide(
                reader, runner, num_classes=num_classes,
                level=level, target_mag=target_mag,
                patch_size=patch, overlap=overlap, batch_size=batch,
                roi_level0=roi, progress=on_prog
            )
            mask = res["mask"]
            lv = res["level"]

            base = os.path.splitext(os.path.basename(wsi_path))[0]
            suffix = f"_L{lv}"
            if roi is not None:
                suffix += f"_roi_{roi[0]}_{roi[1]}_{roi[2]}_{roi[3]}"
            out_path = os.path.join(out_dir, base + suffix + "_mask.png")
            save_mask(mask, out_path)
            self.finished_ok.emit(out_path)
        except Exception as ex:
            self.failed.emit(str(ex))


class SegPage(QWidget):
    """兼容旧构造：SegPage(app_cfg, palette=None, models_cfg=None, logger=None)"""
    def __init__(self, app_cfg, palette=None, models_cfg=None, logger=None):
        super().__init__()
        self.app_cfg = app_cfg
        self.palette = palette
        self.models_cfg = models_cfg or {}
        self.logger = logger or _NullLogger()
        self.worker: Optional[SegWorker] = None

        lay = QVBoxLayout(self)

        # --- 行：WSI 路径 ---
        self.edWSI = QLineEdit()
        btnWSI = QPushButton("选择 WSI")
        btnWSI.clicked.connect(self.on_pick_wsi)
        row1 = QHBoxLayout(); row1.addWidget(QLabel("WSI:")); row1.addWidget(self.edWSI); row1.addWidget(btnWSI)
        lay.addLayout(row1)

        # --- 行：模型路径 & arch/num_classes ---
        self.edModel = QLineEdit()
        btnModel = QPushButton("选择模型(.pth/.pt)")
        btnModel.clicked.connect(self.on_pick_model)
        row2 = QHBoxLayout(); row2.addWidget(QLabel("模型:")); row2.addWidget(self.edModel); row2.addWidget(btnModel)
        lay.addLayout(row2)

        row2b = QHBoxLayout()
        self.edArch = QLineEdit()
        self.edArch.setPlaceholderText("state_dict 时填写，如 unet_small；TorchScript 可留空")
        self.spnClasses = QSpinBox(); self.spnClasses.setRange(1, 20); self.spnClasses.setValue(1)
        row2b.addWidget(QLabel("结构名(可空):")); row2b.addWidget(self.edArch, 1)
        row2b.addWidget(QLabel("类别数:")); row2b.addWidget(self.spnClasses)
        lay.addLayout(row2b)

        # --- 组：分辨率选择（level 或 目标倍率） ---
        grpLv = QGroupBox("分辨率")
        lvLay = QHBoxLayout(grpLv)
        self.spnLevel = QSpinBox(); self.spnLevel.setRange(0, 16); self.spnLevel.setValue(0)
        self.chkUseTargetMag = QCheckBox("按目标倍率"); self.chkUseTargetMag.setChecked(False)
        self.dspTargetMag = QDoubleSpinBox(); self.dspTargetMag.setRange(1.0, 80.0); self.dspTargetMag.setValue(10.0); self.dspTargetMag.setDecimals(1)
        lvLay.addWidget(QLabel("level:")); lvLay.addWidget(self.spnLevel)
        lvLay.addSpacing(12)
        lvLay.addWidget(self.chkUseTargetMag)
        lvLay.addWidget(QLabel("目标倍率:")); lvLay.addWidget(self.dspTargetMag)
        lay.addWidget(grpLv)

        # --- 组：滑窗参数 ---
        grpSW = QGroupBox("滑窗")
        swLay = QHBoxLayout(grpSW)
        infer_cfg = self.app_cfg.get("inference", {}) if isinstance(self.app_cfg, dict) else {}
        self.spnPatch = QSpinBox(); self.spnPatch.setRange(64, 2048); self.spnPatch.setValue(int(infer_cfg.get("patch_size", 224)))
        self.spnOverlap = QSpinBox(); self.spnOverlap.setRange(0, 512); self.spnOverlap.setValue(int(infer_cfg.get("overlap", 32)))
        # 根据是否有 GPU 决定默认 batch，若 app.yaml 没指定则给出 2
        self.spnBatch = QSpinBox(); self.spnBatch.setRange(1, 64); self.spnBatch.setValue(int(infer_cfg.get("batch_size_cpu", 2)))
        swLay.addWidget(QLabel("patch")); swLay.addWidget(self.spnPatch)
        swLay.addWidget(QLabel("overlap")); swLay.addWidget(self.spnOverlap)
        swLay.addWidget(QLabel("batch")); swLay.addWidget(self.spnBatch)
        lay.addWidget(grpSW)

        # --- 行：ROI（可选，level0 坐标） ---
        self.edROI = QLineEdit()
        self.edROI.setPlaceholderText("可选：level0 坐标 x0,y0,w0,h0，例如 12000,8000,4096,4096")
        row3 = QHBoxLayout(); row3.addWidget(QLabel("ROI:")); row3.addWidget(self.edROI)
        lay.addLayout(row3)

        # --- 行：输出目录 ---
        self.edOut = QLineEdit(os.path.abspath("./exports"))
        btnOut = QPushButton("选择输出目录")
        btnOut.clicked.connect(self.on_pick_out)
        row4 = QHBoxLayout(); row4.addWidget(QLabel("输出:")); row4.addWidget(self.edOut); row4.addWidget(btnOut)
        lay.addLayout(row4)

        # --- 行：控制 & 进度 ---
        ctrl = QHBoxLayout()
        self.btnRun = QPushButton("开始分割")
        self.btnRun.clicked.connect(self.on_run)
        self.prog = QProgressBar(); self.prog.setRange(0, 100); self.prog.setValue(0); self.prog.setTextVisible(True)
        ctrl.addWidget(self.btnRun); ctrl.addWidget(self.prog, 1)
        lay.addLayout(ctrl)

        lay.addStretch(1)

    # ----- UI handlers -----
    def on_pick_wsi(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 WSI", "", "Slides/Images (*.svs *.ndpi *.mrxs *.scn *.svslide *.tif *.tiff *.png *.jpg *.jpeg *.bmp)")
        if path:
            self.edWSI.setText(path)

    def on_pick_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型", "", "PyTorch (*.pth *.pt);;All (*)")
        if path:
            self.edModel.setText(path)

    def on_pick_out(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", self.edOut.text() or "./exports")
        if path:
            self.edOut.setText(path)

    def _parse_roi(self) -> Optional[Tuple[int,int,int,int]]:
        txt = (self.edROI.text() or "").strip()
        if not txt:
            return None
        try:
            xs = [int(t.strip()) for t in txt.split(",")]
            if len(xs) != 4:
                raise ValueError
            x0,y0,w0,h0 = xs
            if w0 <= 0 or h0 <= 0:
                raise ValueError
            return x0,y0,w0,h0
        except Exception:
            QMessageBox.warning(self, "ROI 错误", "ROI 需要形如 x0,y0,w0,h0 且为正整数。")
            return None

    def on_run(self):
        wsi = self.edWSI.text().strip()
        model = self.edModel.text().strip()
        out_dir = self.edOut.text().strip() or "./exports"
        if not os.path.isfile(wsi):
            QMessageBox.warning(self, "提示", "请先选择 WSI 文件")
            return
        if not os.path.isfile(model):
            QMessageBox.warning(self, "提示", "请先选择 模型文件 (.pth/.pt)")
            return

        arch = self.edArch.text().strip() or None
        num_classes = int(self.spnClasses.value())

        # 分辨率策略：优先“按目标倍率”复选框
        use_mag = self.chkUseTargetMag.isChecked()
        level = None if use_mag else int(self.spnLevel.value())
        target_mag = float(self.dspTargetMag.value()) if use_mag else None

        patch = int(self.spnPatch.value())
        overlap = int(self.spnOverlap.value())
        batch = int(self.spnBatch.value())

        roi = self._parse_roi()
        # ROI 解析失败会弹窗提示并返回 None，这里允许 None

        self.prog.setValue(0)
        self.btnRun.setEnabled(False)
        self.logger.info(f"[SEG] start: wsi={wsi}, model={model}, arch={arch}, classes={num_classes}, "
                         f"level={level}, target_mag={target_mag}, patch={patch}, overlap={overlap}, batch={batch}, roi={roi}")

        self.worker = SegWorker(
            wsi, model, arch, num_classes, out_dir,
            level, target_mag, patch, overlap, batch, roi,
            prefer_gpu=True, amp=None, parent=self
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_ok.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def on_progress(self, done: int, total: int):
        pct = int(done * 100 / total) if total > 0 else 0
        self.prog.setValue(max(0, min(100, pct)))
        self.prog.setFormat(f"{pct}% ({done}/{total})")

    def on_done(self, out_path: str):
        self.btnRun.setEnabled(True)
        self.prog.setValue(100)
        self.logger.info(f"[SEG] done -> {out_path}")
        QMessageBox.information(self, "完成", f"已保存分割掩膜：\n{out_path}")

    def on_failed(self, msg: str):
        self.btnRun.setEnabled(True)
        self.prog.setValue(0)
        self.logger.error(f"[SEG] failed: {msg}")
        QMessageBox.critical(self, "失败", f"分割失败：\n{msg}")


class _NullLogger:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass
    def exception(self, *a, **k): pass
