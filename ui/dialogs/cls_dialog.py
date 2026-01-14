from __future__ import annotations
from typing import Optional, Dict, Any
import os, json
import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QPushButton, QProgressBar, QMessageBox, QFileDialog,
    QAbstractSpinBox, QToolButton
)

# 复用你已有的 worker
from ui.pages.cls_page import ClsWorker, SaveOverlayWorker


# ---- Robust parsers for normalization ----
def _parse_triplet(v, default):
    """
    接受:
      - [r,g,b] / (r,g,b)
      - "0.485,0.456,0.406" / "0.485 0.456 0.406"
      - {"r":..,"g":..,"b":..} 或 {"mean":[...]}
    返回 [float,float,float]；失败则 default
    """
    if v is None:
        return list(default)
    try:
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            return [float(v[0]), float(v[1]), float(v[2])]
        if isinstance(v, dict):
            if all(k in v for k in ("r", "g", "b")):
                return [float(v["r"]), float(v["g"]), float(v["b"])]
            if "mean" in v and isinstance(v["mean"], (list, tuple)) and len(v["mean"]) >= 3:
                m = v["mean"]
                return [float(m[0]), float(m[1]), float(m[2])]
        if isinstance(v, str):
            s = v.replace(";", " ").replace(",", " ")
            parts = [p for p in s.split() if p.strip()]
            if len(parts) >= 3:
                return [float(parts[0]), float(parts[1]), float(parts[2])]
    except Exception:
        pass
    return list(default)


# ---- 自带「-  +」按钮的 SpinBox（水平排列，更稳不容易错位） ----
class _PlusMinusSpinBox(QSpinBox):
    """右侧用两个小按钮显示 - / +，与系统主题解耦。"""

    def __init__(self, symbol_color: str = "#4B5E6B", parent=None):
        super().__init__(parent)
        self._symbol_color = QColor(symbol_color or "#4B5E6B")
        self._btn_minus = QToolButton(self)
        self._btn_plus = QToolButton(self)
        self._setup_buttons()

        # 关闭原生上下箭头，只保留我们自己的 - / +
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)
        # 给右侧两个按钮腾出空间
        self.setStyleSheet("""
            QSpinBox {
                padding-right: 34px;
            }
        """)

    def _setup_buttons(self):
        color = self._symbol_color.name()
        for btn, txt, slot in (
            (self._btn_minus, "-", self.stepDown),
            (self._btn_plus,  "+", self.stepUp),
        ):
            btn.setText(txt)
            btn.setAutoRepeat(True)
            btn.clicked.connect(slot)
            btn.setCursor(Qt.ArrowCursor)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.setStyleSheet(f"""
                QToolButton {{
                    border: none;
                    padding: 0px;
                    margin: 0px;
                    background-color: transparent;
                    font-weight: bold;
                    color: {color};
                }}
                QToolButton:hover {{
                    background-color: #E6F1FC;
                }}
            """)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 两个按钮水平排布，垂直居中
        w = 16
        h = self.height() - 4
        y = 2
        x_plus = self.width() - w - 2
        x_minus = x_plus - w
        self._btn_plus.setGeometry(x_plus, y, w, h)
        self._btn_minus.setGeometry(x_minus, y, w, h)
        self._btn_plus.raise_()
        self._btn_minus.raise_()


class _PlusMinusDoubleSpinBox(QDoubleSpinBox):
    """DoubleSpinBox 版本的 - / + 按钮。"""

    def __init__(self, symbol_color: str = "#4B5E6B", parent=None):
        super().__init__(parent)
        self._symbol_color = QColor(symbol_color or "#4B5E6B")
        self._btn_minus = QToolButton(self)
        self._btn_plus = QToolButton(self)
        self._setup_buttons()

        self.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.setStyleSheet("""
            QDoubleSpinBox {
                padding-right: 34px;
            }
        """)

    def _setup_buttons(self):
        color = self._symbol_color.name()
        for btn, txt, slot in (
            (self._btn_minus, "-", self.stepDown),
            (self._btn_plus,  "+", self.stepUp),
        ):
            btn.setText(txt)
            btn.setAutoRepeat(True)
            btn.clicked.connect(slot)
            btn.setCursor(Qt.ArrowCursor)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.setStyleSheet(f"""
                QToolButton {{
                    border: none;
                    padding: 0px;
                    margin: 0px;
                    background-color: transparent;
                    font-weight: bold;
                    color: {color};
                }}
                QToolButton:hover {{
                    background-color: #E6F1FC;
                }}
            """)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = 16
        h = self.height() - 4
        y = 2
        x_plus = self.width() - w - 2
        x_minus = x_plus - w
        self._btn_plus.setGeometry(x_plus, y, w, h)
        self._btn_minus.setGeometry(x_minus, y, w, h)
        self._btn_plus.raise_()
        self._btn_minus.raise_()


class ClsRunDialog(QDialog):
    # 把结果抛回 WSI 页面立即显示
    sigOverlayReady = Signal(np.ndarray, dict)  # rgba, meta
    # 告诉外部“导出到了哪个文件”
    sigOverlayExported = Signal(str, dict)      # out_png, meta

    def __init__(self,
                 app_cfg: dict,
                 palette: dict,
                 models_cfg: dict,
                 logger,
                 slide_path: str,
                 parent=None):
        super().__init__(parent)
        self.setObjectName("ClsRunDialog")           # 供样式表精确匹配
        self.setWindowTitle("分类 - 参数与进度")
        self.setModal(True)
        self.app_cfg = app_cfg or {}
        self.palette = palette or {}
        self.models_cfg = models_cfg or {}
        self.log = logger
        self.slide_path = slide_path
        pal = self.palette if isinstance(self.palette, dict) else {}
        self._spin_symbol_color = pal.get("spinbox_symbol_color") or "#4B5E6B"

        self.worker: Optional[ClsWorker] = None
        self._export_worker: Optional[SaveOverlayWorker] = None
        self._last_rgba: Optional[np.ndarray] = None
        self._last_meta: Optional[dict] = None

        self._build_ui()
        self._apply_cls_theme()      # 只做外观，不动逻辑
        self._populate_defaults()

    # ---------- 仅针对分类对话框的局部样式 ----------
    def _apply_cls_theme(self):
        """
        让 cls 对话框：
          - 方角矩形
          - 表单控件边框统一
        """
        primary = self.palette.get("primary", "#1A90FF")
        border_col = "#C7DDF2"

        self.setStyleSheet(f"""
        /* 只影响本对话框 */
        QDialog#ClsRunDialog {{
            border-radius: 0px;
        }}

        QDialog#ClsRunDialog QLabel {{
            color: #253A4F;
        }}

        QDialog#ClsRunDialog QLineEdit,
        QDialog#ClsRunDialog QSpinBox,
        QDialog#ClsRunDialog QDoubleSpinBox,
        QDialog#ClsRunDialog QComboBox {{
            background-color: #FFFFFF;
            border-radius: 3px;
            border: 1px solid {border_col};
            padding-left: 6px;
        }}
        QDialog#ClsRunDialog QLineEdit:focus,
        QDialog#ClsRunDialog QSpinBox:focus,
        QDialog#ClsRunDialog QDoubleSpinBox:focus,
        QDialog#ClsRunDialog QComboBox:focus {{
            border: 1px solid {primary};
        }}

        QDialog#ClsRunDialog QSpinBox,
        QDialog#ClsRunDialog QDoubleSpinBox {{
            min-height: 22px;
        }}

        /* 开始按钮高亮一点（可选） */
        QDialog#ClsRunDialog QPushButton#BtnPrimary {{
            background-color: {primary};
            border: 1px solid {primary};
            color: #FFFFFF;
            font-weight: 600;
            border-radius: 4px;
            padding: 4px 16px;
        }}
        QDialog#ClsRunDialog QPushButton#BtnPrimary:hover {{
            background-color: #147FE0;
        }}
        """)

    # ---------- UI ----------
    def _build_ui(self):
        lay = QVBoxLayout(self)

        # 基本信息
        row = QGridLayout()
        r = 0

        row.addWidget(QLabel("WSI："), r, 0)
        self.edWSI = QLineEdit(self.slide_path or "")
        self.edWSI.setReadOnly(True)
        row.addWidget(self.edWSI, r, 1, 1, 3); r += 1

        # 模型下拉
        row.addWidget(QLabel("模型："), r, 0)
        self.cmbModel = QComboBox()
        self.cmbModel.currentTextChanged.connect(self._on_model_changed)
        row.addWidget(self.cmbModel, r, 1, 1, 3); r += 1

        # level / patch / overlap / batch
        self.spLevel = _PlusMinusSpinBox(symbol_color=self._spin_symbol_color, parent=self)
        self.spLevel.setRange(0, 8)
        self.spLevel.setValue(0)

        self.spPatch = _PlusMinusSpinBox(symbol_color=self._spin_symbol_color, parent=self)
        self.spPatch.setRange(64, 4096)
        self.spPatch.setSingleStep(64)
        self.spPatch.setValue(1024)

        self.spOverlap = _PlusMinusSpinBox(symbol_color=self._spin_symbol_color, parent=self)
        self.spOverlap.setRange(0, 1024)
        self.spOverlap.setSingleStep(16)
        self.spOverlap.setValue(32)

        self.spBatch = _PlusMinusSpinBox(symbol_color=self._spin_symbol_color, parent=self)
        self.spBatch.setRange(1, 128)
        self.spBatch.setValue(8)

        row.addWidget(QLabel("Level："), r, 0);  row.addWidget(self.spLevel, r, 1)
        row.addWidget(QLabel("Patch："), r, 2);  row.addWidget(self.spPatch, r, 3); r += 1
        row.addWidget(QLabel("Overlap："), r, 0);row.addWidget(self.spOverlap, r, 1)
        row.addWidget(QLabel("Batch："), r, 2);  row.addWidget(self.spBatch, r, 3); r += 1

        # 阈值、GPU/FP16
        self.spThresh = _PlusMinusDoubleSpinBox(symbol_color=self._spin_symbol_color, parent=self)
        self.spThresh.setRange(0.0, 1.0)
        self.spThresh.setSingleStep(0.05)
        self.spThresh.setValue(0.5)

        self.ckGPU = QCheckBox("优先用 GPU"); self.ckGPU.setChecked(True)
        self.ckFP16 = QCheckBox("FP16 (GPU)"); self.ckFP16.setChecked(True)

        row.addWidget(QLabel("阈值："), r, 0); row.addWidget(self.spThresh, r, 1)
        row.addWidget(self.ckGPU, r, 2);      row.addWidget(self.ckFP16, r, 3); r += 1

        lay.addLayout(row)

        # 进度区
        self.lblStatus = QLabel("就绪")
        self.pb = QProgressBar(); self.pb.setRange(0, 100); self.pb.setValue(0)
        lay.addWidget(self.lblStatus)
        lay.addWidget(self.pb)

        # 输出目录
        opt = QGridLayout()
        r2 = 0
        opt.addWidget(QLabel("输出目录："), r2, 0)
        self.edOut = QLineEdit("")
        self.edOut.setPlaceholderText("例如：D:/exports；留空则使用默认导出目录")
        self.btnBrowse = QPushButton("浏览…")
        self.edOut.setEnabled(True)
        self.btnBrowse.setEnabled(True)
        self.btnBrowse.clicked.connect(self._browse_out_dir)

        opt.addWidget(self.edOut, r2, 1, 1, 2)
        opt.addWidget(self.btnBrowse, r2, 3)
        r2 += 1
        lay.addLayout(opt)

        # 按钮区
        btns = QHBoxLayout()
        self.btnStart = QPushButton("开始推理")
        self.btnStart.setObjectName("BtnPrimary")
        self.btnExport = QPushButton("导出结果…"); self.btnExport.setEnabled(False)
        self.btnCancel = QPushButton("关闭")
        btns.addStretch(1)
        btns.addWidget(self.btnStart)
        btns.addWidget(self.btnExport)
        btns.addWidget(self.btnCancel)
        lay.addLayout(btns)

        self.btnStart.clicked.connect(self._on_start)
        self.btnExport.clicked.connect(self._on_manual_export)
        self.btnCancel.clicked.connect(self._on_close)

        self.setMinimumWidth(560)

    def _toggle_auto_out_controls(self, on: bool):
        self.edOut.setEnabled(on)
        self.btnBrowse.setEnabled(on)

    def _browse_out_dir(self):
        base = self._default_out_dir()
        cur = self.edOut.text().strip() or base
        out = QFileDialog.getExistingDirectory(self, "选择输出目录", cur)
        if out:
            self.edOut.setText(out)

    # ---------- models 配置统一化 ----------
    def _models_map(self) -> dict:
        """
        把 self.models_cfg 统一成 {name: cfg_dict} 的映射。
        兼容 list / dict / {"models":{}} 多种写法。
        """
        cfg = getattr(self, "models_cfg", None)
        if cfg is None:
            return {}
        if isinstance(cfg, dict) and isinstance(cfg.get("models"), dict):
            return dict(cfg["models"])
        if isinstance(cfg, dict):
            vals = list(cfg.values())
            if vals and all(isinstance(v, dict) for v in vals):
                return dict(cfg)
        if isinstance(cfg, list):
            out = {}
            for i, it in enumerate(cfg):
                if not isinstance(it, dict):
                    continue
                name = it.get("name") or it.get("id") or it.get("title")
                if not name:
                    p = it.get("path") or it.get("weight") or f"model_{i}.pth"
                    name = os.path.splitext(os.path.basename(p))[0]
                out[name] = it
            return out
        return {}

    def _get_model_cfg(self, name: Optional[str]) -> dict:
        m = self._models_map()
        if not m:
            return {}
        if name and name in m:
            return dict(m[name])
        first_key = next(iter(m.keys()))
        return dict(m[first_key])

    def _resolve_weight_path(self, cfg: dict) -> str:
        """
        支持键名: path/weight/weights/checkpoint
        相对路径将按多候选base逐一尝试，谁存在用谁。
        """
        p = cfg.get("path") or cfg.get("weight") or cfg.get("weights") or cfg.get("checkpoint")
        if not p:
            return ""
        if os.path.isabs(p):
            return p

        bases = []
        if isinstance(self.app_cfg, dict):
            b1 = self.app_cfg.get("models_base_dir")
            if b1: bases.append(os.path.abspath(b1))
            b2 = self.app_cfg.get("project_root")
            if b2: bases.append(os.path.abspath(b2))
            mj = self.app_cfg.get("models_json")
            if mj: bases.append(os.path.abspath(os.path.dirname(mj)))

        here = os.path.abspath(os.path.dirname(__file__))  # .../ui/dialogs
        proj = os.path.abspath(os.path.join(here, "..", ".."))  # 项目根
        bases.append(proj)
        bases.append(os.getcwd())

        for base in bases:
            full = os.path.normpath(os.path.join(base, p))
            if os.path.isfile(full):
                return full

        return os.path.normpath(os.path.join(proj, p))

    # ---------- 默认填充 & 切换时回填 ----------
    def _populate_defaults(self):
        models = self._models_map()
        self.cmbModel.blockSignals(True)
        self.cmbModel.clear()
        self.cmbModel.addItems(sorted(models.keys()))
        self.cmbModel.blockSignals(False)
        if models:
            first = next(iter(sorted(models.keys())))
            self._apply_cfg_to_widgets(self._get_model_cfg(first))
        self.edOut.setText(self._default_out_dir())

    def _on_model_changed(self, name: str):
        cfg = self._get_model_cfg(name)
        self._apply_cfg_to_widgets(cfg)

    def _apply_cfg_to_widgets(self, cfg: Dict[str, Any]):
        lvl = cfg.get("level", 0)
        try:
            self.spLevel.setValue(int(lvl))
        except Exception:
            self.spLevel.setValue(0)

        patch = int(cfg.get("patch", cfg.get("input_size", 512)))
        self.spPatch.setValue(patch)

        self.spOverlap.setValue(int(cfg.get("overlap", 32)))
        self.spBatch.setValue(int(cfg.get("batch", 8)))
        self.spThresh.setValue(float(cfg.get("threshold", 0.5)))
        self.ckGPU.setChecked(bool(cfg.get("prefer_gpu", True)))
        amp = str(cfg.get("amp", cfg.get("use_fp16", "auto"))).lower()
        self.ckFP16.setChecked(False if amp in ("off", "false", "0") else True)

    def _expand_param_aliases(self, params: dict, slide_path: str, model_path: str):
        """
        给 ClsWorker 可能使用到的别名都塞上值。
        """
        for k in ("slide_path", "wsi_path", "slide", "image_path", "input_path"):
            params[k] = slide_path
        for k in ("model_path", "weight", "weights", "checkpoint", "model", "model_file"):
            params[k] = model_path
        mi = params.get("model_info") or {}
        for k in ("path", "weight", "weights", "checkpoint"):
            mi[k] = model_path
        params["model_info"] = mi

    # ---------- 启动 ----------
    def _on_start(self):
        if self.worker is not None:
            return

        slide = (self.edWSI.text().strip() if hasattr(self, "edWSI") else "") or ""
        if not slide:
            QMessageBox.information(self, "提示", "WSI 路径为空。请先打开切片。")
            return
        if not os.path.isfile(slide):
            QMessageBox.information(self, "提示", f"WSI 文件不存在：\n{slide}")
            return

        model_name = self.cmbModel.currentText().strip()
        cfg = self._get_model_cfg(model_name)
        model_path = self._resolve_weight_path(cfg)
        if not model_path:
            QMessageBox.information(self, "提示", "模型路径未配置（支持 path/weight/weights/checkpoint）。")
            return
        if not os.path.isfile(model_path):
            QMessageBox.information(self, "提示", f"模型文件不存在：\n{model_path}")
            return

        mean = _parse_triplet(cfg.get("mean"), [0.485, 0.456, 0.406])
        std  = _parse_triplet(cfg.get("std"),  [0.229, 0.224, 0.225])
        classes = cfg.get("classes") or ["neg", "pos"]
        if not (isinstance(classes, (list, tuple)) and len(classes) >= 2):
            classes = ["neg", "pos"]

        pap_ids = (
            cfg.get("pap_ids")
            or cfg.get("papillary_ids")
            or cfg.get("papillary_related_ids")
            or []
        )
        try:
            pap_ids = [int(x) for x in pap_ids]
        except Exception:
            pap_ids = []

        pal = self.palette if isinstance(self.palette, dict) else {}
        rgba = tuple(pal.get("papillary_patch_color", [255, 0, 255]))
        alpha_const = float(pal.get("papillary_patch_alpha", 0.4))

        params = dict(
            prefer_gpu=self.ckGPU.isChecked(),
            amp_pref=("auto" if self.ckFP16.isChecked() else "off"),
            slide_path=slide,
            model_info={
                "name": model_name,
                "arch": cfg.get("arch", ""),
                "type": cfg.get("type", "classification_logits"),
                "num_classes": int(cfg.get("num_classes", len(classes))),
            },
            level=int(self.spLevel.value()),
            roi_level0=None,
            patch=int(self.spPatch.value()),
            overlap=int(self.spOverlap.value()),
            batch=int(self.spBatch.value()),
            threshold=float(self.spThresh.value()),
            mean=mean, std=std, classes=list(classes),
            color=rgba, use_prob_alpha=False, alpha_const=alpha_const,
            pal=pal,
            pap_ids=pap_ids,
            papillary_ids=pap_ids,
            papillary_related_ids=pap_ids,
        )

        self._expand_param_aliases(params, slide, model_path)

        try:
            self.log.info(f"[CLS] slide={params.get('slide_path')}, weight={params.get('model_path')}")
        except Exception:
            pass

        try:
            self.worker = ClsWorker(params, self)
            self.worker.progress.connect(self._on_progress)
            self.worker.finished_ok.connect(self._on_done)
            self.worker.failed.connect(self._on_failed)
            self.btnStart.setEnabled(False)
            self.btnExport.setEnabled(False)
            self.lblStatus.setText("正在推理…")
            self.pb.setValue(0)
            self.worker.start()
        except Exception as e:
            self.worker = None
            QMessageBox.critical(self, "启动失败", str(e))

    # ---------- 回调 ----------
    def _on_progress(self, done: int, total: int):
        total = max(int(total), 1)
        pct = int(max(0, min(100, done * 100 // total)))
        try:
            self.pb.setValue(pct)
        except Exception:
            pass
        self.lblStatus.setText(f"{done}/{total}（{pct}%）")

    def _on_done(self, result: dict):
        self.btnStart.setEnabled(True)
        self.lblStatus.setText("完成")
        self.worker = None

        rgba = None
        meta = {}
        if isinstance(result, dict):
            rgba = result.get("overlay_rgba")
            meta = result.get("meta") or {}
        elif isinstance(result, (list, tuple)) and len(result) >= 2:
            rgba, meta = result[0], (result[1] or {})
        else:
            meta = {}

        if rgba is None:
            QMessageBox.warning(self, "提示", "推理完成，但未返回 overlay。")
            return

        # --- 1) 规范化 meta，补齐关键字段（下游对齐/网格会用） ---
        meta = dict(meta or {})
        meta.setdefault("target", "classification_overlay")

        # 有些 classify_slide 可能没补 bbox_level0/downsample，这里兜底一下
        try:
            h, w = int(rgba.shape[0]), int(rgba.shape[1])
        except Exception:
            h, w = 0, 0

        if "downsample" not in meta:
            # 对话框没有 reader，因此这里无法从 reader.level_downsamples 推 ds
            # 但你的 ClsWorker.run() 已经会 setdefault("downsample", ds0)，正常不走到这里
            meta["downsample"] = 1.0

        if not meta.get("bbox_level0"):
            ds = float(meta.get("downsample", 1.0) or 1.0)
            meta["bbox_level0"] = [0, 0, int(round(w * ds)), int(round(h * ds))]

        # --- 2) 缓存到对话框（导出按钮依赖） ---
        self._last_rgba = rgba
        self._last_meta = meta
        self.btnExport.setEnabled(True)
        self.lblStatus.setText("完成（可点击“导出结果…”保存 PNG + _meta.json）")

        # --- 3) 关键：抛回 WsiPage，让它自动叠加 overlay ---
        try:
            self.sigOverlayReady.emit(rgba, meta)
        except Exception:
            # 不要让 UI 因为 emit 异常卡死
            pass

        self.btnStart.setEnabled(True)
        self.lblStatus.setText("完成")
        self.worker = None

        rgba = None
        meta = {}
        if isinstance(result, dict):
            rgba = result.get("overlay_rgba")
            meta = result.get("meta") or {}
        elif isinstance(result, (list, tuple)) and len(result) >= 2:
            rgba, meta = result[0], result[1]

        if rgba is None:
            QMessageBox.warning(self, "提示", "推理完成，但未返回 overlay。")
            return

        self._last_rgba = rgba
        self._last_meta = dict(meta or {})
        self.btnExport.setEnabled(True)
        self.lblStatus.setText("完成（可点击“导出结果…”保存 PNG + _meta.json）")

    def _on_failed(self, err: str):
        self.btnStart.setEnabled(True)
        self.btnExport.setEnabled(bool(self._last_rgba is not None))
        self.worker = None
        self.lblStatus.setText("失败")
        QMessageBox.critical(self, "推理失败", err)

    def _on_close(self):
        self.reject()

    # ---------- 导出 ----------
    def _on_manual_export(self):
        if self._last_rgba is None or not isinstance(self._last_meta, dict):
            QMessageBox.information(self, "提示", "没有可导出的结果。")
            return
        out_dir = self.edOut.text().strip()
        if not out_dir:
            out_dir = QFileDialog.getExistingDirectory(self, "选择导出目录", self._default_out_dir())
            if not out_dir:
                out_dir = self._default_out_dir()
                QMessageBox.information(self, "提示", f"未选择目录，已使用默认：\n{out_dir}")
        self._start_export_worker(out_dir)

    def _start_export_worker(self, out_dir: str):
        if self._last_rgba is None or self._last_meta is None:
            return

        lvl = int(self._last_meta.get("level", 0))
        ds  = float(self._last_meta.get("downsample", 1.0))
        h, w = int(self._last_rgba.shape[0]), int(self._last_rgba.shape[1])
        roi = self._last_meta.get("bbox_level0") or self._last_meta.get("roi_level0") \
              or [0, 0, int(round(w * ds)), int(round(h * ds))]

        patch_level  = self._last_meta.get("patch_size_level")
        stride_level = self._last_meta.get("stride_size_level")
        grid_px      = self._last_meta.get("grid_tile_px_on_overlay")

        if self._export_worker is not None:
            try:
                self._export_worker.deleteLater()
            except Exception:
                pass
            self._export_worker = None

        self._export_worker = SaveOverlayWorker(
            overlay_rgba=self._last_rgba,
            slide_path=str(self.slide_path),
            out_dir=out_dir,
            level=lvl,
            ds_at_level=ds,
            roi_level0=tuple(roi),
            patch_size_level=int(patch_level) if patch_level else None,
            grid_tile_px_on_overlay=int(grid_px) if grid_px else None,
            stride_size_level=int(stride_level) if stride_level else None,
            parent=self
        )

        def _ok(out: dict):
            msg_lines = []
            if isinstance(out, dict):
                if out.get("overlay"):      msg_lines.append(f"{out['overlay']}")
                if out.get("overlay_meta"): msg_lines.append(f"{out['overlay_meta']}")
            if not msg_lines:
                msg_lines = ["导出完成。"]
            QMessageBox.information(self, "导出完成", "已保存：\n" + "\n".join(msg_lines))

        def _fail(e: str):
            QMessageBox.critical(self, "导出失败", e)

        self._export_worker.finished_ok.connect(_ok)
        self._export_worker.failed.connect(_fail)
        self._export_worker.start()
        self.lblStatus.setText("正在导出…（后台）")

    # ---------- 路径/输出 ----------
    def _default_out_dir(self) -> str:
        base_dir = ""
        if isinstance(self.app_cfg, dict):
            d = (self.app_cfg.get("export") or {}).get("default_dir")
            if d:
                base_dir = os.path.abspath(d)
        if not base_dir:
            try:
                base_dir = os.path.dirname(self.slide_path) if self.slide_path else os.getcwd()
            except Exception:
                base_dir = os.getcwd()
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    def _compose_out_png(self, out_dir: str) -> str:
        name = os.path.splitext(os.path.basename(self.slide_path or "slide"))[0]
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, f"{name}_cls_overlay.png")

    def _pick_out_dir_from_ui_or_default(self) -> str:
        txt = (self.edOut.text().strip() if hasattr(self, "edOut") else "") or ""
        if not txt:
            return self._default_out_dir()
        if os.path.splitext(txt)[1].lower() in (".png", ".tif", ".tiff"):
            d = os.path.dirname(txt) or self._default_out_dir()
            os.makedirs(d, exist_ok=True)
            return d
        os.makedirs(txt, exist_ok=True)
        return txt
