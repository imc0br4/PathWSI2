# ui/pages/wsi_mixins/det_integration.py
from __future__ import annotations
import os, json, math
from dataclasses import dataclass
from typing import List, Tuple
from PIL import ImageFont
import numpy as np
from PIL import Image, ImageDraw, ImageFont


from PySide6.QtCore import QThread, Signal, QRectF, QObject, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QMessageBox, QMenu, QPushButton, QProgressDialog
)

# 检测 overlay 的最大边长（像素），超过则按 2/4/8… 缩小
_DET_MAX_SIDE = 8000


# 医学主题弹窗 / 进度框 / 菜单样式（与主界面保持一致）
MEDICAL_DIALOG_QSS = """
QDialog#MedicalDialog,
QProgressDialog#MedicalProgressDialog,
QMessageBox#MedicalMessageBox {
    background-color: #f7fbff;
    border-radius: 0px;
    border: 1px solid #c5e1f5;
}

/* 标题/文本颜色偏医学蓝 */
QDialog#MedicalDialog QLabel,
QProgressDialog#MedicalProgressDialog QLabel,
QMessageBox#MedicalMessageBox QLabel {
    color: #123b5d;
}

/* 按钮：青蓝色主色，圆角 */
QDialog#MedicalDialog QPushButton,
QProgressDialog#MedicalProgressDialog QPushButton,
QMessageBox#MedicalMessageBox QPushButton {
    background-color: #1da6b8;
    color: #ffffff;
    border-radius: 6px;
    padding: 4px 12px;
    border: none;
}
QDialog#MedicalDialog QPushButton:hover,
QProgressDialog#MedicalProgressDialog QPushButton:hover,
QMessageBox#MedicalMessageBox QPushButton:hover {
    background-color: #1691a1;
}
QDialog#MedicalDialog QPushButton:pressed,
QProgressDialog#MedicalProgressDialog QPushButton:pressed,
QMessageBox#MedicalMessageBox QPushButton:pressed {
    background-color: #0f6b77;
}
"""

MEDICAL_MENU_QSS = """
QMenu#MedicalMenu {
    background-color: #f7fbff;
    border: 1px solid #c5e1f5;
    padding: 4px 0;
}
QMenu#MedicalMenu::item {
    padding: 4px 24px 4px 18px;
}
QMenu#MedicalMenu::item:selected {
    background-color: #d1ecff;
}
"""


def _create_medical_message_box(parent, icon, title, text,
                                buttons=QMessageBox.Ok,
                                default_button=QMessageBox.Ok) -> QMessageBox:
    box = QMessageBox(parent)
    box.setObjectName("MedicalMessageBox")
    box.setIcon(icon)
    box.setWindowTitle(title)
    box.setText(text)
    box.setStandardButtons(buttons)
    box.setDefaultButton(default_button)
    box.setStyleSheet(MEDICAL_DIALOG_QSS)
    return box


def _info_box(parent, title, text):
    _create_medical_message_box(parent, QMessageBox.Information, title, text).exec()


def _warning_box(parent, title, text):
    _create_medical_message_box(parent, QMessageBox.Warning, title, text).exec()


def _critical_box(parent, title, text):
    _create_medical_message_box(parent, QMessageBox.Critical, title, text).exec()


def _question_box(parent, title, text,
                  buttons=QMessageBox.Yes | QMessageBox.No,
                  default_button=QMessageBox.No):
    box = _create_medical_message_box(parent, QMessageBox.Question, title, text, buttons, default_button)
    return box.exec()


# ------------------------- 后台线程：滑窗检测 -------------------------
@dataclass
class _DetParams:
    weight: str
    imgsz: int
    conf: float
    iou: float
    device: str | None
    classes: list[str] | None
    overlap: float           # 0~0.9，滑窗重叠比例
    max_tiles: int           # 限制总切片数，防炸内存/时间
    nms_iou: float           # 合并不同切片间的重复框
    lvl: int                 # 选择的 pyramid level
    ds: float                # 对应 level 的 downsample（level像素 -> level-0像素）
    rois_level0: List[Tuple[int,int,int,int]]


class _DetWorker(QThread):
    progress = Signal(int, int, str)           # cur, total, note
    finished = Signal(list, list)              # boxes, rois
    error = Signal(str)

    def __init__(self, reader, params: _DetParams, parent: QObject | None = None):
        super().__init__(parent)
        self.reader = reader
        self.p = params
        self._stop = False

    def cancel(self):
        self._stop = True

    # 简易 NMS（IoU on level-0坐标）
    @staticmethod
    def _nms(boxes: list[dict], iou_th: float) -> list[dict]:
        if not boxes:
            return []

        # 拆成纯数值数组 + 元数据列表
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        scores = []
        clses = []
        labels = []
        sources = []

        for b in boxes:
            X, Y, W, H = b["bbox"]
            x1.append(float(X))
            y1.append(float(Y))
            x2.append(float(X + W))
            y2.append(float(Y + H))
            scores.append(float(b.get("score", 0.0)))
            clses.append(int(b.get("cls", 0)))
            labels.append(b.get("label", ""))    # 保留到列表，不进 numpy
            sources.append(b.get("source", ""))  # 同上

        x1 = np.asarray(x1, dtype=float)
        y1 = np.asarray(y1, dtype=float)
        x2 = np.asarray(x2, dtype=float)
        y2 = np.asarray(y2, dtype=float)
        scores = np.asarray(scores, dtype=float)
        clses = np.asarray(clses, dtype=int)

        # 按分数从高到低
        order = scores.argsort()[::-1]
        keep_idx = []

        while order.size > 0:
            i = order[0]
            keep_idx.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            area_i = (x2[i] - x1[i]) * (y1[i] - y1[i] + (y2[i] - y1[i]))
            # 上面 area_i 写错了？更稳妥重新写一遍：
            area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
            area_j = (x2[order[1:]] - x1[order[1:]]) * (y2[order[1:]] - y1[order[1:]])
            iou = inter / (area_i + area_j - inter + 1e-6)

            inds = np.where(iou <= iou_th)[0]
            order = order[inds + 1]

        # 组回带 label/source 的输出
        out = []
        for i in keep_idx:
            out.append({
                "bbox": [
                    int(round(x1[i])),
                    int(round(y1[i])),
                    int(round(x2[i] - x1[i])),
                    int(round(y2[i] - y1[i]))
                ],
                "score": float(scores[i]),
                "cls": int(clses[i]),
                "label": str(labels[i]),
                "source": str(sources[i]),
            })
        return out

    def run(self):
        try:
            from ultralytics import YOLO
        except Exception:
            self.error.emit("请先安装 ultralytics：pip install ultralytics")
            return

        try:
            model = YOLO(self.p.weight)
        except Exception as e:
            self.error.emit(f"加载模型失败：{e}")
            return

        # 预估总 tile 数
        stride = max(1, int(round(self.p.imgsz * (1.0 - self.p.overlap))))
        total_tiles = 0
        for (x0, y0, w0, h0) in self.p.rois_level0:
            wL = max(1, int(round(w0 / self.p.ds)))
            hL = max(1, int(round(h0 / self.p.ds)))
            nx = max(1, math.ceil((max(1, wL) - self.p.imgsz) / stride) + 1)
            ny = max(1, math.ceil((max(1, hL) - self.p.imgsz) / stride) + 1)
            total_tiles += nx * ny

        cur = 0
        self.progress.emit(cur, max(1, total_tiles), "准备检测…")

        boxes: List[dict] = []

        for (x0, y0, w0, h0) in self.p.rois_level0:
            if self._stop:
                break
            wL = max(1, int(round(w0 / self.p.ds)))
            hL = max(1, int(round(h0 / self.p.ds)))

            yL = 0
            while yL < hL and not self._stop:
                tile_hL = min(self.p.imgsz, hL - yL)
                xL = 0
                while xL < wL and not self._stop:
                    tile_wL = min(self.p.imgsz, wL - xL)
                    # level-0 起点
                    X0 = x0 + int(round(xL * self.p.ds))
                    Y0 = y0 + int(round(yL * self.p.ds))
                    # 读取该 level 的 tile
                    rgba = self.reader.read_region(self.p.lvl, X0, Y0, tile_wL, tile_hL)
                    rgb = Image.fromarray(rgba, "RGBA").convert("RGB")

                    # YOLO 推理
                    res = model.predict(
                        source=[rgb],
                        imgsz=self.p.imgsz,
                        conf=self.p.conf,
                        iou=self.p.iou,
                        device=self.p.device,
                        verbose=False,
                        classes=None  # 如需筛选类别，可在这里传索引
                    )
                    if res and res[0].boxes is not None:
                        r0 = res[0]
                        for b in r0.boxes:
                            xyxy = b.xyxy[0].tolist()
                            score = float(getattr(b.conf[0], "item", lambda: b.conf[0])())
                            cls_id = int(getattr(b.cls[0], "item", lambda: b.cls[0])())
                            x1, y1, x2, y2 = xyxy

                            # tile(lvl) -> 全局 level-0
                            X = X0 + int(round(x1 * self.p.ds))
                            Y = Y0 + int(round(y1 * self.p.ds))
                            W = int(round((x2 - x1) * self.p.ds))
                            H = int(round((y2 - y1) * self.p.ds))

                            label = (
                                self.p.classes[cls_id]
                                if isinstance(self.p.classes, (list, tuple))
                                and 0 <= cls_id < len(self.p.classes)
                                else str(cls_id)
                            )
                            boxes.append({
                                "bbox": [X, Y, W, H],
                                "score": score,
                                "cls": cls_id,
                                "label": label,
                                "source": "det",
                            })

                    cur += 1
                    # 有时候估算的 total_tiles 会比真实循环次数略小，防止 cur > total
                    if cur % 5 == 0 or cur >= total_tiles:
                        shown_cur = min(cur, total_tiles if total_tiles > 0 else cur)
                        self.progress.emit(shown_cur,
                                           max(1, total_tiles),
                                           f"检测中… {shown_cur}/{max(1, total_tiles)}")

                    xL += stride
                yL += stride

        if self._stop:
            self.error.emit("已取消")
            return

        # DEBUG: NMS 前后数量对比
        print(f"[DET] raw boxes before NMS: {len(boxes)}")
        final_boxes = self._nms(boxes, self.p.nms_iou)
        print(f"[DET] final boxes after NMS: {len(final_boxes)}, nms_iou={self.p.nms_iou}")

        self.finished.emit(final_boxes, self.p.rois_level0)


# ------------------------- 页面 Mixin：按钮/流程/导出 -------------------------
class DetIntegrationMixin:
    def _build_det_button(self, bar_layout):
        self.btn_det = QPushButton("检测", self)
        bar_layout.addWidget(self.btn_det)
        self.btn_det.clicked.connect(self._on_det_clicked)

        # 默认参数（可在 app_cfg["det"] 里覆盖）
        det_cfg = self.app_cfg.setdefault("det", {}) if isinstance(self.app_cfg, dict) else {}
        det_cfg.setdefault("overlap", 0.2)
        det_cfg.setdefault("max_tiles", 1500)
        det_cfg.setdefault("nms_iou", 0.5)
        det_cfg.setdefault("fallback_full_on_empty_roi", True)
        det_cfg.setdefault("det_stroke", 2)
        det_cfg.setdefault("conf", 0.25)
        det_cfg.setdefault("iou", 0.45)

        self._det_worker = None
        self._det_prog = None
        self._last_det_model = None

    def _on_det_clicked(self):
        if not self.reader:
            _info_box(self, "提示", "请先打开一张 WSI。")
            return

        m = QMenu(self)
        m.setObjectName("MedicalMenu")
        m.setStyleSheet(MEDICAL_MENU_QSS)

        a_full = QAction("全图检测", self)
        a_cls = QAction("仅在分类阳性区域检测", self)
        a_roi = QAction("在当前 ROI 检测", self)
        m.addAction(a_full)
        m.addAction(a_cls)
        m.addAction(a_roi)
        qt_exec = getattr(m, "exec", None) or getattr(m, "exec_", None)
        a = qt_exec(self.btn_det.mapToGlobal(self.btn_det.rect().bottomLeft()))
        if a is None:
            return
        mode = "full" if a is a_full else ("cls" if a is a_cls else "roi")
        self._start_detection(mode)

    # ---------- 入口：在后台线程启动 ----------
    def _start_detection(self, mode: str):
        """
        mode: "full" / "cls" / "roi"
        """
        # ---------- 1) 若选择“分类阳性区域检测”，先确认有分类 overlay ----------
        if mode == "cls":
            ov = getattr(self, "_overlay_rgba", None)
            meta = getattr(self, "_overlay_meta", None) or {}
            target = str(meta.get("target", "")).lower()

            # 粗略判断：有 overlay 且不是检测框那种 overlay，才算“分类叠加”
            is_det_overlay = any(k in target for k in ("detect", "detection", "box"))
            has_cls_overlay = (ov is not None) and (not is_det_overlay)

            print(f"[DET] start_detection mode=cls, has_overlay={ov is not None}, "
                  f"meta.target={target}, is_det_overlay={is_det_overlay}")

            if not has_cls_overlay:
                _info_box(
                    self,
                    "提示",
                    "当前没有分类叠加结果，无法仅在“分类阳性区域”进行检测。\n"
                    "请先运行一次分类功能生成 overlay。"
                )
                return

        # ---------- 2) 选择检测模型 ----------
        model_item = self._pick_det_model_item()
        if model_item is None:
            _info_box(
                self,
                "提示",
                "未在 models.json 找到检测模型。\n"
                "请为 YOLO 模型设置 task/type/backend，例如：\n"
                '  "task": "det", "type": "yolo" 或 "backend": "ultralytics"'
            )
            return

        det_cfg = self.app_cfg.get("det", {}) if isinstance(self.app_cfg, dict) else {}

        imgsz = int(
            model_item.get("imgsz")
            or model_item.get("img_size")
            or model_item.get("input_size", 1024)
        )
        conf = float(model_item.get("conf", det_cfg.get("conf", 0.25)))
        iou = float(model_item.get("iou", det_cfg.get("iou", 0.45)))
        device = model_item.get("device", None)
        classes = model_item.get("classes", None)

        # ---------- 3) 根据模式收集 ROI ----------
        rois = self._collect_rois_for_detection(mode)

        # ROI 模式下，如果没 ROI，根据配置决定是否退化为全图
        if mode == "roi" and (not rois):
            fb = bool(det_cfg.get("fallback_full_on_empty_roi", True))
            if fb:
                ret = _question_box(
                    self,
                    "没有 ROI",
                    "未选定 ROI，是否改为全图检测？",
                    buttons=QMessageBox.Yes | QMessageBox.No,
                    default_button=QMessageBox.No,
                )
                if ret == QMessageBox.No:
                    return
                rois = self._collect_rois_for_detection("full")
            else:
                _info_box(self, "提示", "当前无 ROI。")
                return

        # ---------- 4) 选择 level ----------
        ds_list = [float(d) for d in self.reader.level_downsamples]
        if not ds_list:
            ds_list = [1.0]

        overlap_val = float(det_cfg.get("overlap", 0.2))

        def _estimate_tiles_for_level(ds_val: float) -> int:
            stride = max(1, int(round(imgsz * (1.0 - overlap_val))))
            total = 0
            for (x0, y0, w0, h0) in rois:
                wL = max(1, int(round(w0 / ds_val)))
                hL = max(1, int(round(h0 / ds_val)))
                nx = max(1, math.ceil((max(1, wL) - imgsz) / stride) + 1)
                ny = max(1, math.ceil((max(1, hL) - imgsz) / stride) + 1)
                total += nx * ny
            return total

        if mode == "cls":
            # —— 重点：CLS 模式强制用 level-0，保证与训练脚本一致 —— #
            lvl = 0
            ds = ds_list[0]
            est_tiles = _estimate_tiles_for_level(ds)
            print(f"[DET] force lvl=0 for cls; ds={ds}, est_tiles={est_tiles}, overlap={overlap_val}")
        else:
            # 其它模式（全图 / 手动 ROI）仍用原来的“自适应 level”逻辑，兼顾速度
            lvl, ds, est_tiles = self._pick_level_for_detection(
                rois,
                imgsz,
                overlap=overlap_val,
                max_tiles=int(det_cfg.get("max_tiles", 1500)),
            )

        print(f"[DET] use lvl={lvl}, ds={ds}, est_tiles={est_tiles}, mode={mode}")

        # ---------- 5) 进度条 ----------
        self._det_prog = QProgressDialog("准备检测…", "取消", 0, 0, self)
        self._det_prog.setWindowModality(Qt.ApplicationModal)
        self._det_prog.setMinimumDuration(300)
        self._det_prog.setAutoClose(False)
        self._det_prog.setAutoReset(False)
        self._det_prog.setObjectName("MedicalProgressDialog")
        self._det_prog.setStyleSheet(MEDICAL_DIALOG_QSS)
        self._det_prog.show()

        # ---------- 6) 启动后台线程 ----------
        p = _DetParams(
            weight=model_item.get("weight") or model_item.get("path"),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            classes=classes,
            overlap=float(det_cfg.get("overlap", 0.2)),
            max_tiles=int(det_cfg.get("max_tiles", 1500)),
            nms_iou=float(det_cfg.get("nms_iou", 0.5)),
            lvl=lvl,
            ds=ds,
            rois_level0=rois,
        )
        self._last_det_model = model_item

        self._det_worker = _DetWorker(self.reader, p, self)
        self._det_worker.progress.connect(self._on_det_progress)
        self._det_worker.finished.connect(self._on_det_finished)
        self._det_worker.error.connect(self._on_det_error)
        self._det_prog.canceled.connect(self._det_worker.cancel)
        self._det_worker.start()

    def _on_det_progress(self, cur: int, total: int, note: str):
        try:
            total = max(1, int(total))
            cur = max(0, min(int(cur), total))   # 再次夹一下
            self._det_prog.setRange(0, total)
            self._det_prog.setValue(cur)
            self._det_prog.setLabelText(note)
        except Exception:
            pass

    def _on_det_error(self, msg: str):
        try:
            self._det_prog.cancel()
        except Exception:
            pass
        if msg and msg != "已取消":
            _critical_box(self, "检测失败", msg)

    def _on_det_finished(self, boxes: list[dict], rois: list[tuple]):
        try:
            self._det_prog.close()
        except Exception:
            pass
        if not boxes:
            _info_box(self, "提示", "未检出目标。")
            return
        self._export_and_show_detections(boxes, rois)

    # ---------- 选择合适的 pyramid level ----------
    def _pick_level_for_detection(
        self,
        rois_level0: List[Tuple[int, int, int, int]],
        imgsz: int,
        overlap: float,
        max_tiles: int,
    ):
        ds_list = [float(d) for d in self.reader.level_downsamples]
        if not ds_list:
            return 0, 1.0, 1
        stride = max(1, int(round(imgsz * (1.0 - overlap))))
        best = (0, ds_list[0], 10**12)
        chosen = None
        for lvl, ds in enumerate(ds_list):
            total = 0
            for (x0, y0, w0, h0) in rois_level0:
                wL = max(1, int(round(w0 / ds)))
                hL = max(1, int(round(h0 / ds)))
                nx = max(1, math.ceil((max(1, wL) - imgsz) / stride) + 1)
                ny = max(1, math.ceil((max(1, hL) - imgsz) / stride) + 1)
                total += nx * ny
            if total <= max_tiles:
                chosen = (lvl, ds, total)
                break
            if total < best[2]:
                best = (lvl, ds, total)
        return chosen if chosen else best

    # ---------- 选择检测模型（兼容 list / dict / {"models":{}}） ----------
    def _pick_det_model_item(self) -> dict | None:
        cfg = getattr(self, "models_cfg", None)

        candidates = []

        def _add_candidate(it):
            if not isinstance(it, dict):
                return
            task = str(it.get("task", "")).lower()
            t = str(it.get("type", "")).lower()
            backend = str(it.get("backend", "")).lower()

            is_det_task = task in {"det", "detect", "detection"}
            is_det_type = t in {
                "det", "detect", "detection",
                "yolo", "yolov3", "yolov5", "yolov8", "yolov9", "yolov10", "yolov11"
            }
            is_det_backend = backend == "ultralytics"

            if is_det_task or is_det_type or is_det_backend:
                candidates.append(it)

        if isinstance(cfg, list):
            for it in cfg:
                _add_candidate(it)
        elif isinstance(cfg, dict):
            if isinstance(cfg.get("models"), dict):
                for it in cfg["models"].values():
                    _add_candidate(it)
            else:
                for v in cfg.values():
                    _add_candidate(v)

        # fallback: 直接从 models_json 读取一次
        if not candidates and isinstance(self.app_cfg, dict):
            mj = self.app_cfg.get("models_json")
            if mj and os.path.isfile(mj):
                try:
                    with open(mj, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    if isinstance(raw, list):
                        for it in raw:
                            _add_candidate(it)
                    elif isinstance(raw, dict):
                        if isinstance(raw.get("models"), dict):
                            for it in raw["models"].values():
                                _add_candidate(it)
                        else:
                            for v in raw.values():
                                _add_candidate(v)
                except Exception:
                    pass

        return candidates[0] if candidates else None

    # ---------- 根据模式收集 ROI ----------
    def _collect_rois_for_detection(self, mode: str):
        dims0 = list(self.reader.level_dimensions)
        w0, h0 = int(dims0[0][0]), int(dims0[0][1])

        if mode == "full":
            return [(0, 0, w0, h0)]

        if mode == "roi":
            rois = []
            tool = getattr(self, "roi_tool", None)
            if tool and tool.has_rois():
                for r in tool.rois():
                    rr: QRectF = r.rect
                    rois.append((int(rr.left()), int(rr.top()),
                                 int(rr.width()), int(rr.height())))
            return rois

        # mode == "cls"
        ov = getattr(self, "_overlay_rgba", None)
        meta = getattr(self, "_overlay_meta", None) or {}
        if ov is None:
            return [(0, 0, w0, h0)]
        ds = float(getattr(self, "_overlay_ds", 1.0))
        ox, oy = getattr(self, "_overlay_pos", (0, 0))
        tile = int(self._infer_tile_from_meta(meta, default=getattr(self, "_grid_tile_px", 64)))

        H, W = ov.shape[0], ov.shape[1]
        s = tile
        cols = (W + s - 1) // s
        rows = (H + s - 1) // s
        positive = np.zeros((rows, cols), dtype=np.uint8)
        a = ov[:, :, 3]  # alpha>0 视为阳性

        for r in range(rows):
            for c in range(cols):
                y00 = r * s
                x00 = c * s
                if a[y00:min(y00 + s, H), x00:min(x00 + s, W)].max() > 0:
                    positive[r, c] = 1

        pos = positive.copy()
        for r in range(rows):
            for c in range(cols):
                if positive[r, c]:
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            rr, cc = r + dr, c + dc
                            if 0 <= rr < rows and 0 <= cc < cols:
                                pos[rr, cc] = 1

        rois = []
        vis = np.zeros_like(pos)
        for r in range(rows):
            for c in range(cols):
                if pos[r, c] and not vis[r, c]:
                    cc = c
                    while cc < cols and pos[r, cc] and not vis[r, cc]:
                        cc += 1
                    c2 = cc - 1
                    rr = r
                    while rr < rows and pos[rr, c:c2+1].all() and not vis[rr, c]:
                        for t in range(c, c2+1):
                            vis[rr, t] = 1
                        rr += 1
                    r2 = rr - 1
                    x = ox + c * s * ds
                    y = oy + r * s * ds
                    w = (c2 - c + 1) * s * ds
                    h = (r2 - r + 1) * s * ds
                    rois.append((int(x), int(y), int(w), int(h)))
        # DEBUG: 统计阳性格子和生成的 ROI 数
        try:
            print(
                f"[DET] CLS-ROI: overlay_shape={ov.shape}, "
                f"ds={ds}, tile={tile}, "
                f"positive_cells={int(positive.sum())}, rois={len(rois)}"
            )
        except Exception:
            pass

        return rois or [(0, 0, w0, h0)]


    # ---------- 导出与显示（线框 PNG + meta + json） ----------
    def _export_and_show_detections(self, boxes: list[dict], rois: list[tuple]):
        export_cfg = self.app_cfg.get("export", {}) if isinstance(self.app_cfg, dict) else {}
        default_dir = os.path.abspath(export_cfg.get("default_dir", "./exports"))
        os.makedirs(default_dir, exist_ok=True)

        base = os.path.splitext(os.path.basename(getattr(self.reader, "path", "slide")))[0]
        out_png  = os.path.join(default_dir, f"{base}_detections.png")
        out_meta = os.path.join(default_dir, f"{base}_detections_meta.json")
        out_json = os.path.join(default_dir, f"{base}_detections.json")

        # 没有框就直接提示
        if not boxes:
            _info_box(self, "提示", "未检出目标。")
            return

        # ---- 1) 计算 level-0 下的整体包围框 ----
        x_min = min(b["bbox"][0] for b in boxes)
        y_min = min(b["bbox"][1] for b in boxes)
        x_max = max(b["bbox"][0] + b["bbox"][2] for b in boxes)
        y_max = max(b["bbox"][1] + b["bbox"][3] for b in boxes)
        ox, oy = x_min, y_min
        W0, H0 = max(1, x_max - x_min), max(1, y_max - y_min)  # level-0 尺寸

        # ---- 2) 按 _DET_MAX_SIDE 自动用 2/4/8… 缩小 overlay 尺度 ----
        side = max(W0, H0)
        sf = 1  # 缩小倍数：1/2/4/8…
        while side > _DET_MAX_SIDE:
            sf *= 2
            side = (side + 1) // 2

        # overlay 像素尺寸（注意用整除向上取整）
        W = max(1, (W0 + sf - 1) // sf)
        H = max(1, (H0 + sf - 1) // sf)

        # 这里的 downsample 表示“overlay 像素 -> level-0 像素”的比例
        ds_out = float(sf)

        # ---- 3) 画蓝色框 + 文本（在缩小后的 overlay 坐标系下）----
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        im = Image.fromarray(rgba, "RGBA")
        draw = ImageDraw.Draw(im)
        stroke = int(self.app_cfg.get("det", {}).get("det_stroke", 2))

        box_color  = (0, 0, 255, 255)      # 蓝色边框
        text_color = (255, 255, 255, 255)  # 白色文字

        # 尝试加载字体，找不到就用默认字体
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

        def _measure_text(txt: str):
            """兼容不同 Pillow 版本的文字宽高计算。"""
            try:
                if hasattr(font, "getbbox"):
                    l, t, r, b = font.getbbox(txt)
                    return r - l, b - t
                if hasattr(font, "getsize"):
                    return font.getsize(txt)
            except Exception:
                pass
            # 兜底估算
            return max(8, 8 * len(txt)), 14

        for b in boxes:
            X, Y, Wb, Hb = b["bbox"]   # 这里是 level-0 坐标

            # 映射到 overlay 像素坐标（注意除以 sf）
            x1 = (X - ox) / ds_out
            y1 = (Y - oy) / ds_out
            x2 = (X + Wb - ox) / ds_out
            y2 = (Y + Hb - oy) / ds_out

            x1 = int(round(x1))
            y1 = int(round(y1))
            x2 = int(round(x2))
            y2 = int(round(y2))

            # 蓝色方框
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=stroke)

            # 文本内容：类别 + 置信度
            label = str(b.get("label", ""))
            score = float(b.get("score", 0.0))
            text = f"{label} {score:.2f}"

            tw, th = _measure_text(text)

            # 文本背景：蓝底
            bg_x1 = x1
            bg_y1 = max(0, y1 - th - 2)
            bg_x2 = bg_x1 + tw + 4
            bg_y2 = bg_y1 + th + 2
            draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=box_color)

            # 白色文字
            draw.text((bg_x1 + 2, bg_y1 + 1), text, fill=text_color, font=font)

        # ---- 4) 保存 PNG + meta + 结果 json ----
        im.save(out_png, optimize=False)

        meta = {
            "target": "detection_boxes",
            "level": 0,
            "downsample": ds_out,                     # 关键：缩小倍数
            "bbox_level0": [int(ox), int(oy), int(W0), int(H0)],  # 仍然用 level-0 尺寸
            "path": getattr(self.reader, "path", None),
        }
        with open(out_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        model_item = getattr(self, "_last_det_model", None) or (self._pick_det_model_item() or {})
        out = {
            "slide_path": getattr(self.reader, "path", None),
            "model": model_item.get("id"),
            "backend": model_item.get("backend", "ultralytics"),
            "task": model_item.get("task", "detection"),
            "classes": model_item.get("classes"),
            "params": {
                "imgsz":  model_item.get("imgsz", 1024),
                "conf":   model_item.get("conf", 0.25),
                "iou":    model_item.get("iou",  0.45),
                "overlap": float(self.app_cfg.get("det", {}).get("overlap", 0.2)),
                "nms_iou": float(self.app_cfg.get("det", {}).get("nms_iou", 0.5)),
            },
            "roi_level0": self._merge_rois_level0(rois),
            "level": 0,
            "downsample": ds_out,
            "boxes": boxes,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        # ---- 5) 叠加到当前视图（直接用内存中的 im，不再重新读 PNG）----
        try:
            ov = np.array(im, dtype=np.uint8)  # im 已经是 RGBA
            print(f"[DET] auto overlay: shape={ov.shape}, ds_out={ds_out}, origin=({ox},{oy})")

            # 确保 overlay_item 存在
            if getattr(self, "overlay_item", None) is None or self.overlay_item.scene() is None:
                if hasattr(self, "_create_overlay_item"):
                    self._create_overlay_item()

            if getattr(self, "overlay_item", None) is None:
                print("[DET] WARNING: overlay_item is None, cannot auto-apply detection overlay")
            else:
                self.overlay_item.set_rgba(ov)
                self.overlay_item.setPos(int(ox), int(oy))
                self.overlay_item.setScale(ds_out)
                self.overlay_item.setOpacity(self._overlay_opacity)
                self.overlay_item.setVisible(True)

                self._overlay_rgba = ov
                self._overlay_ds   = ds_out
                self._overlay_pos  = (int(ox), int(oy))
                self._overlay_meta = meta

                if hasattr(self, "btn_edit"):
                    self.btn_edit.setEnabled(True)
        except Exception as e:
            print("[DET] auto overlay failed:", repr(e))

        _info_box(
            self,
            "检测完成",
            f"已导出：\n{out_png}\n{out_meta}\n{out_json}"
        )

    def _merge_rois_level0(self, rois):
        if not rois:
            return [0, 0, 0, 0]
        xs = [x for x, _, _, _ in rois]
        ys = [y for _, y, _, _ in rois]
        x2s = [x + w for x, _, w, _ in rois]
        y2s = [y + h for _, y, _, h in rois]
        x0 = min(xs)
        y0 = min(ys)
        x1 = max(x2s)
        y1 = max(y2s)
        return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
