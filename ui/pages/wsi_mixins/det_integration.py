# ui/pages/wsi_mixins/det_integration.py
from __future__ import annotations
import os, json, math, time, importlib, traceback
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtGui import QDoubleValidator
from PySide6.QtCore import QThread, Signal, QRectF, QObject, Qt
from PySide6.QtGui import QAction, QColor, QPen, QFont, QPainter
from PySide6.QtWidgets import (
    QMessageBox, QMenu, QPushButton, QProgressDialog, QFileDialog,
    QGraphicsItem,QDialog, QLabel, QVBoxLayout, QHBoxLayout, QLineEdit,
    QDoubleSpinBox, QDialogButtonBox, QAbstractSpinBox,QToolButton
)

from utils.patch_export import safe_read_patch_level0, iter_positive_cells

_DET_MAX_SIDE = 8000

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

class _PlusMinusDoubleSpinBox(QDoubleSpinBox):
    """cls_dialog 同款：右侧用 - / + 代替上下箭头"""
    def __init__(self, symbol_color: str = "#4B5E6B", parent=None):
        super().__init__(parent)
        self._btn_minus = QToolButton(self)
        self._btn_plus = QToolButton(self)

        self.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.setStyleSheet("QDoubleSpinBox { padding-right: 34px; }")

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
                    color: {symbol_color};
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


class _DetVisConfDialog(QDialog):
    def __init__(self, value: float, palette: dict | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("DetVisConfDialog")
        self.setWindowTitle("显示阈值")
        self.setModal(True)

        pal = palette if isinstance(palette, dict) else {}
        primary = pal.get("primary", "#1A90FF")
        border_col = "#C7DDF2"
        symbol_col = pal.get("spinbox_symbol_color", "#4B5E6B")

        self.setStyleSheet(f"""
        QDialog#DetVisConfDialog {{
            border-radius: 0px;
            background: #FFFFFF;
        }}
        QDialog#DetVisConfDialog QLabel {{
            color: #253A4F;
        }}
        QDialog#DetVisConfDialog QDoubleSpinBox {{
            background-color: #FFFFFF;
            border-radius: 3px;
            border: 1px solid {border_col};
            padding-left: 6px;
            min-height: 22px;
        }}
        QDialog#DetVisConfDialog QDoubleSpinBox:focus {{
            border: 1px solid {primary};
        }}
        QDialog#DetVisConfDialog QPushButton#BtnPrimary {{
            background-color: {primary};
            border: 1px solid {primary};
            color: #FFFFFF;
            font-weight: 600;
            border-radius: 4px;
            padding: 4px 16px;
        }}
        QDialog#DetVisConfDialog QPushButton#BtnPrimary:hover {{
            background-color: #147FE0;
        }}
        """)

        lay = QVBoxLayout(self)

        lab = QLabel("仅显示置信度  ≥")
        lay.addWidget(lab)

        self.sp = _PlusMinusDoubleSpinBox(symbol_color=symbol_col, parent=self)
        self.sp.setRange(0.0, 1.0)
        self.sp.setDecimals(2)
        self.sp.setSingleStep(0.01)
        self.sp.setValue(float(value))
        lay.addWidget(self.sp)

        btns = QHBoxLayout()
        btns.addStretch(1)
        ok = QPushButton("OK")
        ok.setObjectName("BtnPrimary")
        cancel = QPushButton("Cancel")
        btns.addWidget(ok)
        btns.addWidget(cancel)
        lay.addLayout(btns)

        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)

        self.setMinimumWidth(320)

    def value(self) -> float:
        return float(self.sp.value())


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

def _critical_box(parent, title, text):
    _create_medical_message_box(parent, QMessageBox.Critical, title, text).exec()

def _question_box(parent, title, text,
                  buttons=QMessageBox.Yes | QMessageBox.No,
                  default_button=QMessageBox.No):
    box = _create_medical_message_box(parent, QMessageBox.Question, title, text, buttons, default_button)
    return box.exec()

def _get_patch_export_worker_cls():
    last_err = None
    for modname in (
        "ui.workers.patch_export_worker",
        "ui.pages.wsi_mixins.patch_export_worker",
        "patch_export_worker",
    ):
        try:
            m = importlib.import_module(modname)
            return getattr(m, "PatchExportWorker")
        except Exception as e:
            last_err = e
    raise last_err


# ------------------------- 矢量检测叠加层：不会污染编辑，也能“显示自如” -------------------------
class DetBoxesItem(QGraphicsItem):
    """
    用矢量方式画检测框：
      - 不需要把所有框 rasterize 成大图
      - 支持：置信度阈值、低倍隐藏文字、一键隐藏/清除
      - 不会写入 _overlay_rgba/_overlay_meta（因此编辑只针对染色块）
    """
    def __init__(self, parent: QGraphicsItem | None = None):
        super().__init__(parent)
        self._boxes = []
        self._bounds = QRectF(0, 0, 0, 0)

        self._visible = True
        self._conf_th = 0.25
        self._show_labels = True
        self._label_min_scale = 0.6

        # 你原来的“方形显示框”，建议改语义=“聚合边界显示”
        self._square_display = False  # 现在表示：是否显示“簇边界(并集外包框)”
        self._merge_for_square = True
        self._merge_iou_th = 0.0      # 0 表示只要相交就合并（更符合“并集边界”）
        self._merge_ios_th = 0.0

        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.setZValue(9999)

        # ✅ 关键：让 option.exposedRect 真正代表“当前需要重绘的区域”
        self.setFlag(QGraphicsItem.ItemUsesExtendedStyleOption, True)

        # ---- cache：把重活移出 paint ----
        self._boxes_ver = 0
        self._cache_key = None
        self._cache_raw = None       # 过滤阈值后的原始框
        self._cache_merged = None    # 聚合后的簇边界框



    def boundingRect(self) -> QRectF:
        return self._bounds
    
    def _invalidate_cache(self):
        self._cache_key = None
        self._cache_raw = None
        self._cache_merged = None

    def set_boxes(self, boxes):
        self.prepareGeometryChange()
        self._boxes = boxes or []
        self._boxes_ver += 1
        self._invalidate_cache()

        if not self._boxes:
            self._bounds = QRectF(0, 0, 0, 0)
        else:
            x1 = min(b["bbox"][0] for b in self._boxes)
            y1 = min(b["bbox"][1] for b in self._boxes)
            x2 = max(b["bbox"][0] + b["bbox"][2] for b in self._boxes)
            y2 = max(b["bbox"][1] + b["bbox"][3] for b in self._boxes)
            self._bounds = QRectF(float(x1), float(y1), float(x2 - x1), float(y2 - y1))
        self.update()


    def clear(self):
        self.set_boxes([])


    def set_conf_threshold(self, v: float):
        self._conf_th = float(v)
        self._invalidate_cache()
        self.update()

    def set_visible_flag(self, v: bool):
        self._visible = bool(v)
        self.setVisible(self._visible)
        self.update()

    def set_show_labels(self, v: bool):
        self._show_labels = bool(v)
        self.update()

    def set_label_min_scale(self, v: float):
        self._label_min_scale = float(v)
        self.update()

    def set_square_display(self, v: bool, pad: float | None = None):
        # pad 不再需要：你现在要的是“并集边界”，不是正方形放大
        self._square_display = bool(v)
        self._invalidate_cache()
        self.update()

            # ------------------ NEW: 重叠框合并（仅用于显示，不影响导出/结果） ------------------
    # def _merge_boxes_for_display(
    #     self,
    #     boxes: List[dict],
    #     conf_th: float | None = None,
    #     iou_th: float = 0.0,
    #     ios_th: float = 0.0,
    #     exposed: QRectF | None = None,
    #     same_class_only: bool = False,
    #     gap_px: float = 0.0,
    # ) -> List[dict]:
    #     """
    #     显示用合并：把重叠/相邻的框聚成簇，返回每簇的“并集外包矩形”(envelope)。
    #     - 只影响显示，不改检测结果、不影响导出
    #     - 医生视角：更想快速看到“一团可疑区域”的边界，而不是一堆互相压住的框线
    #     """

    #     if not boxes:
    #         return []

    #     # 1) 过滤 + 视野裁剪
    #     cand = []
    #     for b in boxes:
    #         try:
    #             score = float(b.get("score", 0.0))
    #             if conf_th is not None and score < float(conf_th):
    #                 continue

    #             X, Y, W, H = b.get("bbox", [0, 0, 0, 0])
    #             X = float(X); Y = float(Y); W = float(W); H = float(H)
    #             if W <= 0 or H <= 0:
    #                 continue

    #             r = QRectF(X, Y, W, H)
    #             if exposed is not None and (not exposed.isNull()) and (not r.intersects(exposed)):
    #                 continue

    #             cand.append((X, Y, X + W, Y + H, score, int(b.get("cls", 0)), str(b.get("label", "")), b))
    #         except Exception:
    #             continue

    #     n = len(cand)
    #     if n <= 1:
    #         return [cand[0][7]] if n == 1 else []

    #     parent = list(range(n))
    #     rank = [0] * n

    #     def find(i: int) -> int:
    #         while parent[i] != i:
    #             parent[i] = parent[parent[i]]
    #             i = parent[i]
    #         return i

    #     def union(a: int, b: int):
    #         ra, rb = find(a), find(b)
    #         if ra == rb:
    #             return
    #         if rank[ra] < rank[rb]:
    #             parent[ra] = rb
    #         elif rank[ra] > rank[rb]:
    #             parent[rb] = ra
    #         else:
    #             parent[rb] = ra
    #             rank[ra] += 1

    #     # 2) 并查集合并：相交即合并；gap 扩张后相交也合并（更符合“区域”）
    #     g = float(gap_px or 0.0)

    #     for i in range(n):
    #         x1_i, y1_i, x2_i, y2_i, _, cls_i, _, _ = cand[i]
    #         area_i = max(0.0, x2_i - x1_i) * max(0.0, y2_i - y1_i)
    #         if area_i <= 0:
    #             continue

    #         for j in range(i + 1, n):
    #             x1_j, y1_j, x2_j, y2_j, _, cls_j, _, _ = cand[j]

    #             if same_class_only and cls_i != cls_j:
    #                 continue

    #             # 原框是否相交
    #             xx1 = max(x1_i, x1_j)
    #             yy1 = max(y1_i, y1_j)
    #             xx2 = min(x2_i, x2_j)
    #             yy2 = min(y2_i, y2_j)
    #             inter_w = max(0.0, xx2 - xx1)
    #             inter_h = max(0.0, yy2 - yy1)
    #             inter = inter_w * inter_h

    #             if inter > 0:
    #                 union(i, j)
    #                 continue

    #             # gap 扩张后是否相交（相邻也归为同一簇）
    #             if g > 0:
    #                 ex1_i, ey1_i, ex2_i, ey2_i = (x1_i - g, y1_i - g, x2_i + g, y2_i + g)
    #                 ex1_j, ey1_j, ex2_j, ey2_j = (x1_j - g, y1_j - g, x2_j + g, y2_j + g)

    #                 exx1 = max(ex1_i, ex1_j)
    #                 eyy1 = max(ey1_i, ey1_j)
    #                 exx2 = min(ex2_i, ex2_j)
    #                 eyy2 = min(ey2_i, ey2_j)
    #                 if exx2 > exx1 and eyy2 > eyy1:
    #                     union(i, j)
    #                     continue

    #             # 可选：如果你还想保留 IoU/IoS 判定（一般此处不需要）
    #             if (iou_th > 0) or (ios_th > 0):
    #                 area_j = max(0.0, x2_j - x1_j) * max(0.0, y2_j - y1_j)
    #                 if area_j > 0:
    #                     # inter=0 已经 continue 过了，所以这里一般不触发
    #                     iou = 0.0
    #                     ios = 0.0
    #                     if (iou >= float(iou_th)) or (ios >= float(ios_th)):
    #                         union(i, j)

    #     # 3) 每簇输出一个“并集外包矩形”
    #     groups: dict[int, list[int]] = {}
    #     for i in range(n):
    #         r = find(i)
    #         groups.setdefault(r, []).append(i)

    #     merged = []
    #     for idxs in groups.values():
    #         x1 = min(cand[k][0] for k in idxs)
    #         y1 = min(cand[k][1] for k in idxs)
    #         x2 = max(cand[k][2] for k in idxs)
    #         y2 = max(cand[k][3] for k in idxs)

    #         best_k = max(idxs, key=lambda k: cand[k][4])
    #         _, _, _, _, best_score, best_cls, best_label, best_b = cand[best_k]

    #         out = dict(best_b)
    #         out["bbox"] = [int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))]
    #         out["score"] = float(best_score)
    #         out["cls"] = int(best_cls)
    #         out["label"] = str(best_label)
    #         out["_merged_n"] = int(len(idxs))
    #         out["source"] = "merged_vis"
    #         merged.append(out)

    #     return merged



    def _merge_boxes_to_clusters(self, boxes: list[dict]) -> list[dict]:
        """把相交/嵌套的框合成簇级外包框（并集边界的外接矩形）。只算一次，供 paint 快速画。"""
        n = len(boxes)
        if n <= 1:
            return boxes

        # 提取 xyxy
        x1 = [float(b["bbox"][0]) for b in boxes]
        y1 = [float(b["bbox"][1]) for b in boxes]
        x2 = [float(b["bbox"][0] + b["bbox"][2]) for b in boxes]
        y2 = [float(b["bbox"][1] + b["bbox"][3]) for b in boxes]
        sc = [float(b.get("score", 0.0)) for b in boxes]

        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # ✅ “并集边界”最符合你描述：只要相交就归为一簇（不看 IoU 阈值）
        for i in range(n):
            for j in range(i + 1, n):
                if min(x2[i], x2[j]) <= max(x1[i], x1[j]):  # x 不相交
                    continue
                if min(y2[i], y2[j]) <= max(y1[i], y1[j]):  # y 不相交
                    continue
                union(i, j)

        groups = {}
        for i in range(n):
            r = find(i)
            groups.setdefault(r, []).append(i)

        out = []
        for idxs in groups.values():
            xx1 = min(x1[k] for k in idxs)
            yy1 = min(y1[k] for k in idxs)
            xx2 = max(x2[k] for k in idxs)
            yy2 = max(y2[k] for k in idxs)

            best_k = max(idxs, key=lambda k: sc[k])
            b0 = boxes[best_k]
            merged = dict(b0)
            merged["bbox"] = [int(round(xx1)), int(round(yy1)), int(round(xx2 - xx1)), int(round(yy2 - yy1))]
            merged["_merged_n"] = len(idxs)
            out.append(merged)
        return out

    def _rebuild_cache_if_needed(self):
        key = (self._boxes_ver, float(self._conf_th), bool(self._square_display))
        if self._cache_key == key:
            return

        conf_th = float(self._conf_th)
        raw = []
        for b in self._boxes:
            try:
                if float(b.get("score", 0.0)) < conf_th:
                    continue
                X, Y, W, H = b.get("bbox", [0, 0, 0, 0])
                if float(W) <= 0 or float(H) <= 0:
                    continue
                raw.append(b)
            except Exception:
                continue

        self._cache_raw = raw
        self._cache_merged = self._merge_boxes_to_clusters(raw) if (self._square_display and len(raw) > 1) else raw
        self._cache_key = key

    def paint(self, painter, option, widget=None):
        if (not self._visible) or (not self._boxes):
            return

        self._rebuild_cache_if_needed()
        boxes = self._cache_merged or []
        if not boxes:
            return

        # 视图缩放
        try:
            scale = float(painter.worldTransform().m11())
        except Exception:
            scale = 1.0
        scale = max(scale, 1e-6)

        exposed = getattr(option, "exposedRect", None)
        do_cull = (
            exposed is not None and (not exposed.isNull())
            and exposed.width() > 0 and exposed.height() > 0
        )

        painter.save()
        try:
            # ✅ 关键：检测框不需要抗锯齿（OpenGL 下会显著影响 FPS）
            painter.setRenderHint(QPainter.Antialiasing, False)
            painter.setRenderHint(QPainter.TextAntialiasing, False)

            # ✅ cosmetic pen：线宽不随缩放变化（更快、更稳定）
            pen = QPen(QColor(0, 120, 255))
            pen.setCosmetic(True)
            pen.setWidth(2)
            painter.setPen(pen)

            draw_labels = self._show_labels and (scale >= self._label_min_scale)

            if draw_labels:
                painter.setRenderHint(QPainter.TextAntialiasing, True)
                font = QFont("Arial")
                font.setPointSizeF(max(6.0, 12.0 / scale))
                painter.setFont(font)

                for b in boxes:
                    X, Y, W, H = b["bbox"]
                    r = QRectF(float(X), float(Y), float(W), float(H))
                    if do_cull and (not r.intersects(exposed)):
                        continue

                    painter.drawRect(r)

                    score = float(b.get("score", 0.0))
                    label = str(b.get("label", b.get("cls", "")))
                    n = int(b.get("_merged_n", 1) or 1)
                    txt = f"{label} {score:.2f}".strip()
                    if n > 1:
                        txt = f"{txt} (n={n})"

                    x = r.left() + (2.0 / scale)
                    y = r.top() + (12.0 / scale)
                    painter.setPen(QColor(0, 0, 0))
                    painter.drawText(x + (1.0 / scale), y + (1.0 / scale), txt)
                    painter.setPen(QColor(255, 255, 255))
                    painter.drawText(x, y, txt)
                    painter.setPen(pen)

            else:
                # labels 关闭：批量画 rect（减少 Python->Qt 调用）
                rects = []
                for b in boxes:
                    X, Y, W, H = b["bbox"]
                    r = QRectF(float(X), float(Y), float(W), float(H))
                    if do_cull and (not r.intersects(exposed)):
                        continue
                    rects.append(r)
                if rects:
                    painter.drawRects(rects)

        finally:
            painter.restore()



# ------------------------- 后台线程：滑窗检测 -------------------------
@dataclass
class _DetParams:
    weight: str
    imgsz: int
    conf: float
    iou: float
    device: str | None
    classes: list[str] | None
    overlap: float
    max_tiles: int
    nms_iou: float

    # NEW: 专治“套娃框/嵌套框”
    nms_ios: float = 0.90  # inter / min(area) > 0.9 也认为重复

    # NEW: 加速
    batch: int = 8
    half: bool = True

    # 强制 level0
    lvl: int = 0
    ds: float = 1.0

    rois_level0: List[Tuple[int, int, int, int]] = None


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

    @staticmethod
    def _nms(boxes: list[dict], iou_th: float, ios_th: float) -> list[dict]:
        """NMS + IoS（嵌套框抑制）"""
        if not boxes:
            return []

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
            labels.append(b.get("label", ""))
            sources.append(b.get("source", ""))

        x1 = np.asarray(x1, dtype=float)
        y1 = np.asarray(y1, dtype=float)
        x2 = np.asarray(x2, dtype=float)
        y2 = np.asarray(y2, dtype=float)
        scores = np.asarray(scores, dtype=float)
        clses = np.asarray(clses, dtype=int)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            rest = order[1:]
            if rest.size == 0:
                break

            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
            area_j = (x2[rest] - x1[rest]) * (y2[rest] - y1[rest])

            iou = inter / (area_i + area_j - inter + 1e-6)
            ios = inter / (np.minimum(area_i, area_j) + 1e-6)

            # 同时满足：IoU大 或 IoS大 => 认为重复，抑制
            mask_keep = (iou <= iou_th) & (ios <= ios_th)
            order = rest[mask_keep]

        out = []
        for i in keep:
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
            import torch
        except Exception:
            self.error.emit("请先安装 ultralytics：pip install ultralytics")
            return

        try:
            # 性能选项（不影响结果语义）
            try:
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:
                    pass
            except Exception:
                pass

            model = YOLO(self.p.weight)
            try:
                model.fuse()
            except Exception:
                pass

            # warmup
            try:
                dummy = Image.new("RGB", (self.p.imgsz, self.p.imgsz), (0, 0, 0))
                try:
                    model.predict(source=[dummy], imgsz=self.p.imgsz, conf=self.p.conf, iou=self.p.iou,
                                  device=self.p.device, verbose=False, half=bool(self.p.half))
                except TypeError:
                    model.predict(source=[dummy], imgsz=self.p.imgsz, conf=self.p.conf, iou=self.p.iou,
                                  device=self.p.device, verbose=False)
            except Exception:
                pass

            stride = max(1, int(round(self.p.imgsz * (1.0 - self.p.overlap))))
            rois = self.p.rois_level0 or []
            if not rois:
                self.error.emit("没有可用 ROI。")
                return

            # 估算 tile 数
            total_tiles = 0
            for (x0, y0, w0, h0) in rois:
                nx = max(1, math.ceil((max(1, w0) - self.p.imgsz) / stride) + 1)
                ny = max(1, math.ceil((max(1, h0) - self.p.imgsz) / stride) + 1)
                total_tiles += nx * ny

            cur = 0
            self.progress.emit(cur, max(1, total_tiles), "准备检测…")

            boxes: List[dict] = []
            batch_size = max(1, int(self.p.batch or 1))
            use_half = bool(self.p.half)

            batch_imgs: list[Image.Image] = []
            batch_meta: list[tuple[int, int]] = []   # (X0, Y0)

            def _flush_batch():
                nonlocal batch_imgs, batch_meta, boxes
                if not batch_imgs:
                    return
                try:
                    try:
                        res_list = model.predict(
                            source=batch_imgs,
                            imgsz=self.p.imgsz,
                            conf=self.p.conf,
                            iou=self.p.iou,
                            device=self.p.device,
                            verbose=False,
                            classes=None,
                            half=use_half
                        )
                    except TypeError:
                        res_list = model.predict(
                            source=batch_imgs,
                            imgsz=self.p.imgsz,
                            conf=self.p.conf,
                            iou=self.p.iou,
                            device=self.p.device,
                            verbose=False,
                            classes=None
                        )
                except Exception as e:
                    raise RuntimeError(f"YOLO 推理失败：{e}")

                for ri, r0 in enumerate(res_list):
                    X0, Y0 = batch_meta[ri]
                    if r0 is None or getattr(r0, "boxes", None) is None:
                        continue
                    for b in r0.boxes:
                        xyxy = b.xyxy[0].tolist()
                        score = float(getattr(b.conf[0], "item", lambda: b.conf[0])())
                        cls_id = int(getattr(b.cls[0], "item", lambda: b.cls[0])())
                        x1, y1, x2, y2 = xyxy

                        X = X0 + int(round(x1))
                        Y = Y0 + int(round(y1))
                        W = int(round(x2 - x1))
                        H = int(round(y2 - y1))

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

                batch_imgs.clear()
                batch_meta.clear()

            for (x0, y0, w0, h0) in rois:
                if self._stop:
                    break

                yL = 0
                while yL < h0 and not self._stop:
                    xL = 0
                    while xL < w0 and not self._stop:
                        X0 = int(x0 + xL)
                        Y0 = int(y0 + yL)

                        # 强制 level0 + 固定尺寸读取（避免动态shape拖慢）
                        rgb = safe_read_patch_level0(self.reader, X0, Y0, size=self.p.imgsz)

                        batch_imgs.append(rgb)
                        batch_meta.append((X0, Y0))

                        cur += 1
                        if len(batch_imgs) >= batch_size:
                            _flush_batch()

                        if cur % 10 == 0 or cur >= total_tiles:
                            shown_cur = min(cur, total_tiles if total_tiles > 0 else cur)
                            self.progress.emit(
                                shown_cur,
                                max(1, total_tiles),
                                f"检测中… {shown_cur}/{max(1, total_tiles)} (batch={batch_size})"
                            )

                        xL += stride
                    yL += stride

            if self._stop:
                self.error.emit("已取消")
                return

            _flush_batch()

            print(f"[DET] raw boxes before NMS: {len(boxes)}")
            final_boxes = self._nms(boxes, self.p.nms_iou, self.p.nms_ios)
            print(f"[DET] final boxes after NMS: {len(final_boxes)}, nms_iou={self.p.nms_iou}, nms_ios={self.p.nms_ios}")

            self.finished.emit(final_boxes, rois)

        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))


# ------------------------- 页面 Mixin：按钮/流程/导出 -------------------------
class DetIntegrationMixin:
    def _build_det_button(self, bar_layout):
        self.btn_det = QPushButton("检测", self)
        bar_layout.addWidget(self.btn_det)
        self.btn_det.clicked.connect(self._on_det_clicked)

        self.btn_export = QPushButton("导出", self)
        bar_layout.addWidget(self.btn_export)
        self.btn_export.clicked.connect(self._on_export_clicked)


        det_cfg = self.app_cfg.setdefault("det", {}) if isinstance(self.app_cfg, dict) else {}
        det_cfg.setdefault("overlap", 0.2)
        det_cfg.setdefault("max_tiles", 1500)
        det_cfg.setdefault("nms_iou", 0.5)
        det_cfg.setdefault("nms_ios", 0.90)
        det_cfg.setdefault("conf", 0.25)
        det_cfg.setdefault("iou", 0.45)
        det_cfg.setdefault("batch", 8)
        det_cfg.setdefault("half", True)

        # NEW: 显示控制（解决图二“看不清”）
        det_cfg.setdefault("vis_conf", float(det_cfg.get("conf", 0.25)))

        det_cfg.setdefault("vis_show_labels", False)
        det_cfg.setdefault("label_min_scale", 0.6)   # 低倍不显示标签
        det_cfg.setdefault("vis_square", False)

        # NEW: CLS 模式导出补齐（解决“检测框在染色块外 => 导出遗漏”）
        det_cfg.setdefault("export_union_with_det", True)  # 导出=染色块格子 + 检测框中心patch
        det_cfg.setdefault("export_use_cls_dilated_cells", True)  # 导出时用“检测用的膨胀格子”，避免邻格遗漏

        self._det_worker = None
        self._det_prog = None
        self._last_det_model = None
        self._det_t0 = None
        self._last_det_elapsed_s = None
        self._last_det_mode = ""

        # 染色块 overlay 快照（可编辑）
        self._stain_overlay_rgba = None
        self._stain_overlay_pos = (0, 0)
        self._stain_overlay_ds = 1.0
        self._stain_grid_px = 64
        self._stain_overlay_meta = None

        # 检测结果
        self._last_det_boxes = None
        self._last_det_boxes_xyxy = None
        self._last_det_rois = None
        self._last_det_slide_path = None

        # NEW: 记录 CLS 检测时用到的“膨胀格子列表”，导出时复用
        self._cls_cells_dilated = None

        # NEW: 检测显示层
        self._det_boxes_item = None
        self._det_vis_conf = float(det_cfg.get("vis_conf", 0.4))
        self._det_vis_show_labels = bool(det_cfg.get("vis_show_labels", False))
        self._det_label_min_scale = float(det_cfg.get("label_min_scale", 0.6))
        self._det_vis_square = bool(det_cfg.get("vis_square", False))
        self._det_vis_visible = True

        self._patch_export_worker = None
        self._patch_export_prog = None

    def _on_det_clicked(self):
        if not self.reader:
            _info_box(self, "提示", "请先打开一张 WSI。")
            return

        m = QMenu(self)
        m.setObjectName("MedicalMenu")
        m.setStyleSheet(MEDICAL_MENU_QSS)

        a_full = QAction("全图检测（level0，较慢）", self)
        a_cls  = QAction("仅在分类阳性区域检测（level0）", self)
        a_roi  = QAction("在当前 ROI 检测（level0）", self)
        m.addAction(a_full)
        m.addAction(a_cls)
        m.addAction(a_roi)

        m.addSeparator()

        a_toggle = QAction("显示检测框", self)
        a_toggle.setCheckable(True)
        a_toggle.setChecked(bool(getattr(self, "_det_vis_visible", True)))
        m.addAction(a_toggle)

        a_label = QAction("显示标签（高倍更适合）", self)
        a_label.setCheckable(True)
        a_label.setChecked(bool(getattr(self, "_det_vis_show_labels", False)))
        m.addAction(a_label)

        a_square = QAction("合并显示框（并集外框）", self)
        a_square.setCheckable(True)
        a_square.setChecked(bool(getattr(self, "_det_vis_square", False)))
        m.addAction(a_square)

        a_setconf = QAction(f"设置显示阈值（当前≥{float(getattr(self, '_det_vis_conf', 0.4)):.2f}）", self)
        m.addAction(a_setconf)

        a_clear = QAction("清除检测显示", self)
        m.addAction(a_clear)

        qt_exec = getattr(m, "exec", None) or getattr(m, "exec_", None)
        a = qt_exec(self.btn_det.mapToGlobal(self.btn_det.rect().bottomLeft()))
        if a is None:
            return

        if a is a_toggle:
            self._det_vis_visible = bool(a_toggle.isChecked())
            self._apply_det_overlay_settings()
            return

        if a is a_label:
            self._det_vis_show_labels = bool(a_label.isChecked())
            # 用户显式要求显示标签，就不要倍率阈值挡住
            if self._det_vis_show_labels:
                self._det_label_min_scale = 0.0
            else:
                det_cfg = self.app_cfg.get("det", {}) if isinstance(self.app_cfg, dict) else {}
                self._det_label_min_scale = float(det_cfg.get("label_min_scale", 0.6))
            self._apply_det_overlay_settings()
            return

        if a is a_square:
            self._det_vis_square = bool(a_square.isChecked())
            self._apply_det_overlay_settings()
            return

        if a is a_setconf:
            self._edit_det_vis_conf_dialog()
            return

        if a is a_clear:
            self._clear_det_overlay()
            return

        mode = "full" if a is a_full else ("cls" if a is a_cls else "roi")
        self._start_detection(mode)


    def _ensure_det_boxes_item(self):
        if self._det_boxes_item is not None and self._det_boxes_item.scene() is not None:
            return True

        scene = None
        if getattr(self, "overlay_item", None) is not None and self.overlay_item.scene() is not None:
            scene = self.overlay_item.scene()
        else:
            # 尝试创建 overlay_item（很多工程里 scene 由它托管）
            if hasattr(self, "_create_overlay_item"):
                try:
                    self._create_overlay_item()
                except Exception:
                    pass
            if getattr(self, "overlay_item", None) is not None:
                scene = self.overlay_item.scene()

        if scene is None:
            print("[DET] WARNING: cannot find scene, det overlay will not show.")
            return False

        self._det_boxes_item = DetBoxesItem()
        scene.addItem(self._det_boxes_item)
        return True

    def _apply_det_overlay_settings(self):
        # 1) 如果 item 不存在，但我们有历史检测结果：自动创建并恢复
        if self._det_boxes_item is None or self._det_boxes_item.scene() is None:
            if getattr(self, "_last_det_boxes", None):
                if not self._ensure_det_boxes_item():
                    return
                self._det_boxes_item.set_boxes(self._last_det_boxes)
            else:
                return

        # 2) 如果 item 存在但当前 boxes 为空（例如旧版本曾 clear 掉），也从 last 恢复
        try:
            if (not getattr(self._det_boxes_item, "_boxes", None)) and getattr(self, "_last_det_boxes", None):
                self._det_boxes_item.set_boxes(self._last_det_boxes)
        except Exception:
            pass

        # 3) 应用显示设置
        self._det_boxes_item.set_conf_threshold(self._det_vis_conf)
        self._det_boxes_item.set_show_labels(self._det_vis_show_labels)
        self._det_boxes_item.set_label_min_scale(self._det_label_min_scale)
        self._det_boxes_item.set_square_display(self._det_vis_square)
        self._det_boxes_item.set_visible_flag(self._det_vis_visible)

        # 4) 强制刷新
        try:
            if self.view and self.view.scene():
                self.view.scene().update()
            if self.view:
                self.view.viewport().update()
        except Exception:
            pass


    def _clear_det_overlay(self):
        # “清除显示” = 隐藏，不清空结果
        self._det_vis_visible = False

        if self._det_boxes_item is not None:
            # 不要 self._det_boxes_item.clear() ！否则 boxes 没了就无法再显示
            self._det_boxes_item.set_visible_flag(False)
            self._det_boxes_item.update()

        # 尽量强制刷新（避免 Qt 某些情况下不重绘）
        try:
            if self.view and self.view.scene():
                self.view.scene().update()
            if self.view:
                self.view.viewport().update()
        except Exception:
            pass

    def _on_export_clicked(self):
        if not getattr(self, "reader", None):
            _info_box(self, "提示", "请先打开一张 WSI。")
            return

        m = QMenu(self)
        m.setObjectName("MedicalMenu")
        m.setStyleSheet(MEDICAL_MENU_QSS)

        a_stain = QAction("导出染色块Patch(1024)...", self)
        m.addAction(a_stain)

        a_det = QAction("导出检测框Patch(1024)...（仅检测区域）", self)
        try:
            ok_det = bool(getattr(self, "_last_det_boxes", None)) and (
                getattr(self.reader, "path", None) == getattr(self, "_last_det_slide_path", None)
            )
        except Exception:
            ok_det = False
        a_det.setEnabled(ok_det)
        m.addAction(a_det)

        qt_exec = getattr(m, "exec", None) or getattr(m, "exec_", None)
        a = qt_exec(self.btn_export.mapToGlobal(self.btn_export.rect().bottomLeft()))
        if a is None:
            return

        if a is a_stain:
            self._on_export_patches(kind="stain")
        elif a is a_det:
            self._on_export_patches(kind="det")


    def _start_detection(self, mode: str):
        self._last_det_mode = mode

        # cls 模式：确认当前 overlay 是分类染色块（不是检测框）
        if mode == "cls":
            ov = getattr(self, "_overlay_rgba", None)
            meta = getattr(self, "_overlay_meta", None) or {}
            target = str(meta.get("target", "")).lower()
            is_det_overlay = any(k in target for k in ("detect", "detection", "box"))
            has_cls_overlay = (ov is not None) and (not is_det_overlay)

            print(f"[DET] start_detection mode=cls, has_overlay={ov is not None}, meta.target={target}, is_det_overlay={is_det_overlay}")

            if not has_cls_overlay:
                _info_box(self, "提示", "当前没有分类叠加结果，无法仅在“分类阳性区域”进行检测。\n请先运行一次分类功能生成 overlay。")
                return

            # 保存“染色 overlay 快照”（供导出 & 编辑保持一致）
            try:
                self._stain_overlay_rgba = ov
                self._stain_overlay_meta = meta
                self._stain_overlay_pos = tuple(getattr(self, "_overlay_pos", (0, 0)))
                self._stain_overlay_ds = float(getattr(self, "_overlay_ds", 1.0))
                self._stain_grid_px = int(self._infer_tile_from_meta(meta, default=getattr(self, "_grid_tile_px", 64)))
            except Exception:
                pass

        model_item = self._pick_det_model_item()
        if model_item is None:
            _info_box(self, "提示", "未在 models.json 找到检测模型。\n请为 YOLO 模型设置 task/type/backend，例如：\n  \"task\": \"det\", \"type\": \"yolo\" 或 \"backend\": \"ultralytics\"")
            return

        det_cfg = self.app_cfg.get("det", {}) if isinstance(self.app_cfg, dict) else {}

        imgsz = int(model_item.get("imgsz") or model_item.get("img_size") or model_item.get("input_size", 1024))
        conf = float(model_item.get("conf", det_cfg.get("conf", 0.25)))
        iou = float(model_item.get("iou", det_cfg.get("iou", 0.45)))
        device = model_item.get("device", None)
        classes = model_item.get("classes", None)

        rois = self._collect_rois_for_detection(mode)

        if mode == "roi" and (not rois):
            fb = bool(det_cfg.get("fallback_full_on_empty_roi", True))
            if fb:
                ret = _question_box(self, "没有 ROI", "未选定 ROI，是否改为全图检测？",
                                    buttons=QMessageBox.Yes | QMessageBox.No,
                                    default_button=QMessageBox.No)
                if ret == QMessageBox.No:
                    return
                rois = self._collect_rois_for_detection("full")
            else:
                _info_box(self, "提示", "当前无 ROI。")
                return

        # 强制 level0（你要求必须 lvl=0）
        lvl = 0
        ds = 1.0

        overlap_val = float(det_cfg.get("overlap", 0.2))
        print(f"[DET] force lvl=0; ds={ds}, est_tiles=unknown, overlap={overlap_val}, mode={mode}")

        self._det_prog = QProgressDialog("准备检测…", "取消", 0, 0, self)
        self._det_prog.setWindowModality(Qt.ApplicationModal)
        self._det_prog.setMinimumDuration(300)
        self._det_prog.setAutoClose(False)
        self._det_prog.setAutoReset(False)
        self._det_prog.setObjectName("MedicalProgressDialog")
        self._det_prog.setStyleSheet(MEDICAL_DIALOG_QSS)
        self._det_prog.show()

        p = _DetParams(
            weight=model_item.get("weight") or model_item.get("path"),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            classes=classes,
            overlap=overlap_val,
            max_tiles=int(det_cfg.get("max_tiles", 1500)),
            nms_iou=float(det_cfg.get("nms_iou", 0.5)),
            nms_ios=float(det_cfg.get("nms_ios", 0.90)),
            batch=int(det_cfg.get("batch", 8)),
            half=bool(det_cfg.get("half", True)),
            lvl=lvl,
            ds=ds,
            rois_level0=rois,
        )
        self._last_det_model = model_item

        self._det_t0 = time.perf_counter()
        self._last_det_elapsed_s = None

        self._det_worker = _DetWorker(self.reader, p, self)
        self._det_worker.progress.connect(self._on_det_progress)
        self._det_worker.finished.connect(self._on_det_finished)
        self._det_worker.error.connect(self._on_det_error)
        self._det_prog.canceled.connect(self._det_worker.cancel)
        self._det_worker.start()

    def _on_det_progress(self, cur: int, total: int, note: str):
        try:
            total = max(1, int(total))
            cur = max(0, min(int(cur), total))
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

        try:
            if getattr(self, "_det_t0", None) is not None:
                self._last_det_elapsed_s = float(time.perf_counter() - self._det_t0)
                print(f"[DET] elapsed: {self._last_det_elapsed_s:.2f}s (finished)")
        except Exception:
            pass

        if not boxes:
            _info_box(self, "提示", "未检出目标。")
            return

        self._last_det_boxes = boxes
        self._last_det_rois = rois
        self._last_det_slide_path = getattr(self.reader, "path", None)

        dets_xyxy = []
        for b in boxes:
            X, Y, W, H = b.get("bbox", [0, 0, 0, 0])
            dets_xyxy.append({
                "xyxy": [int(X), int(Y), int(X + W), int(Y + H)],
                "cls": b.get("cls", 0),
                "conf": b.get("score", 0.0),
                "label": b.get("label", ""),
                "source": b.get("source", "det"),
            })
        self._last_det_boxes_xyxy = dets_xyxy

        self._export_and_show_detections(boxes, rois)

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

        ds_overlay = float(getattr(self, "_overlay_ds", 1.0))
        ox, oy = getattr(self, "_overlay_pos", (0, 0))
        tile = int(self._infer_tile_from_meta(meta, default=getattr(self, "_grid_tile_px", 64)))

        H, W = ov.shape[0], ov.shape[1]
        s = tile
        cols = (W + s - 1) // s
        rows = (H + s - 1) // s
        a = ov[:, :, 3]

        # pos_any：格子内任意 alpha>0 就算阳性（更稳）
        pos_any = np.zeros((rows, cols), dtype=np.uint8)
        pos_center = np.zeros((rows, cols), dtype=np.uint8)

        for r in range(rows):
            y00 = r * s
            y11 = min(y00 + s, H)
            cy = min(H - 1, y00 + s // 2)
            for c in range(cols):
                x00 = c * s
                x11 = min(x00 + s, W)
                cx = min(W - 1, x00 + s // 2)

                if int(a[cy, cx]) > 0:
                    pos_center[r, c] = 1
                if a[y00:y11, x00:x11].max() > 0:
                    pos_any[r, c] = 1

        # 膨胀一圈（检测会覆盖邻格，所以导出也要覆盖邻格，避免“框在紫块外 => 导出遗漏”）
        pos = pos_any.copy()
        for r in range(rows):
            for c in range(cols):
                if pos_any[r, c]:
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            rr, cc = r + dr, c + dc
                            if 0 <= rr < rows and 0 <= cc < cols:
                                pos[rr, cc] = 1

        # 记录给导出复用（关键）
        self._cls_cells_dilated = [(c, r) for r in range(rows) for c in range(cols) if pos[r, c] == 1]

        # 生成 ROI：用“合并块”方式（与你当前一致），但导出会用 dilated cells 补齐
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
                    while rr < rows and pos[rr, c:c2 + 1].all() and not vis[rr, c]:
                        for t in range(c, c2 + 1):
                            vis[rr, t] = 1
                        rr += 1
                    r2 = rr - 1

                    x = ox + c * s * ds_overlay
                    y = oy + r * s * ds_overlay
                    w = (c2 - c + 1) * s * ds_overlay
                    h = (r2 - r + 1) * s * ds_overlay
                    rois.append((int(x), int(y), int(w), int(h)))

        try:
            print(
                f"[DET] CLS-ROI: overlay_shape={ov.shape}, ds={ds_overlay}, tile={tile}, "
                f"pos_any={int(pos_any.sum())}, pos_center={int(pos_center.sum())}, "
                f"dilated_cells={len(self._cls_cells_dilated)}, rois={len(rois)}"
            )
        except Exception:
            pass

        return rois or [(0, 0, w0, h0)]

    def _on_export_patches(self, kind: str = "stain"):
        if not getattr(self, "reader", None):
            _info_box(self, "提示", "请先打开一张 WSI。")
            return

        # --- 取“染色 overlay”（用于 stain 导出；det 导出可没有）---
        ov = getattr(self, "_stain_overlay_rgba", None)
        ox, oy = getattr(self, "_stain_overlay_pos", (0, 0))
        ds = float(getattr(self, "_stain_overlay_ds", 1.0))
        grid_px = int(getattr(self, "_stain_grid_px", 64))

        if ov is None:
            cur_ov = getattr(self, "_overlay_rgba", None)
            cur_meta = getattr(self, "_overlay_meta", None) or {}
            target = str(cur_meta.get("target", "")).lower()
            is_det_overlay = any(k in target for k in ("detect", "detection", "box"))
            if cur_ov is not None and not is_det_overlay:
                ov = cur_ov
                ox, oy = getattr(self, "_overlay_pos", (0, 0))
                ds = float(getattr(self, "_overlay_ds", 1.0))
                grid_px = int(self._infer_tile_from_meta(cur_meta, default=getattr(self, "_grid_tile_px", 64)))

        # det-only 导出允许没有 ov（给个 dummy）
        if ov is None and kind == "det":
            ov = np.zeros((1, 1, 4), dtype=np.uint8)
            ox, oy, ds, grid_px = 0, 0, 1.0, 1

        if ov is None:
            _info_box(self, "提示", "当前没有可用的“染色 overlay”（分类结果）。请先运行分类生成 overlay。")
            return

        # --- 输出目录 ---
        export_cfg = self.app_cfg.get("export", {}) if isinstance(self.app_cfg, dict) else {}
        default_dir = os.path.abspath(export_cfg.get("default_dir", "./exports"))
        os.makedirs(default_dir, exist_ok=True)

        base_dir = QFileDialog.getExistingDirectory(self, "选择导出目录（将自动创建子文件夹）", default_dir)
        if not base_dir:
            return

        stamp = time.strftime("%Y-%m-%d_%H_%M_%S")
        out_dir = os.path.join(base_dir, f"{stamp}_{kind}_patches_1024")
        os.makedirs(out_dir, exist_ok=True)

        # --- detections（用于在 patch 上画框）---
        dets = getattr(self, "_last_det_boxes_xyxy", None) or []

        # --- 核心：根据 kind 决定导出内容 ---
        include_stain = True
        cells = None
        extra_centers = []

        if kind == "stain":
            # ✅ 严格按中心格子判定（数量≈你看到的染色块数）
            cells = list(iter_positive_cells(ov, int(grid_px), mode="center"))
            include_stain = True
            extra_centers = []      # 不补齐检测中心
            dets = []               # 不需要画框（也避免误画旧检测框）
        elif kind == "det":
            # ✅ 只导出检测区域 patch（去重后按检测框中心选 patch）
            if not (getattr(self, "_last_det_boxes", None) and getattr(self.reader, "path", None) == self._last_det_slide_path):
                _info_box(self, "提示", "当前没有可用的检测结果，请先运行一次检测。")
                return
            include_stain = False   # 关键：不要导出染色格子
            cells = []              # 无所谓，但显式传空更清晰
            for b in (self._last_det_boxes or []):
                X, Y, Wb, Hb = b.get("bbox", [0, 0, 0, 0])
                extra_centers.append((float(X + Wb * 0.5), float(Y + Hb * 0.5)))
        else:
            _info_box(self, "提示", f"未知导出类型：{kind}")
            return

        PatchExportWorker = _get_patch_export_worker_cls()
        self._patch_export_worker = PatchExportWorker(
            reader=self.reader,
            overlay_rgba=ov,
            overlay_pos=(int(ox), int(oy)),
            overlay_ds=float(ds),
            grid_px=int(grid_px),
            out_dir=out_dir,
            detections=dets,
            patch_size=1024,
            cells=cells,
            extra_centers_level0=extra_centers,
            include_stain=include_stain,
            stain_cell_mode="center",
            parent=self
        )

        # progress dialog 保持你原来的
        self._patch_export_prog = QProgressDialog("导出 patch…", None, 0, 0, self)
        self._patch_export_prog.setWindowModality(Qt.ApplicationModal)
        self._patch_export_prog.setMinimumDuration(200)
        self._patch_export_prog.setAutoClose(False)
        self._patch_export_prog.setAutoReset(False)
        self._patch_export_prog.setObjectName("MedicalProgressDialog")
        self._patch_export_prog.setStyleSheet(MEDICAL_DIALOG_QSS)
        try:
            self._patch_export_prog.setCancelButton(None)
        except Exception:
            pass
        self._patch_export_prog.show()

        def _on_prog(done: int, total: int):
            try:
                total = max(1, int(total))
                done = max(0, min(int(done), total))
                self._patch_export_prog.setRange(0, total)
                self._patch_export_prog.setValue(done)
                self._patch_export_prog.setLabelText(f"导出 patch… {done}/{total}")
            except Exception:
                pass

        def _on_ok(res: dict):
            try:
                self._patch_export_prog.close()
            except Exception:
                pass
            _info_box(self, "导出完成",
                    f"已导出 {res.get('count', 0)} 张 patch\n目录：{res.get('out_dir')}\nmeta：{res.get('meta')}")

        def _on_fail(err: str):
            try:
                self._patch_export_prog.close()
            except Exception:
                pass
            _critical_box(self, "导出失败", err)

        self._patch_export_worker.progress.connect(_on_prog)
        self._patch_export_worker.finished_ok.connect(_on_ok)
        self._patch_export_worker.failed.connect(_on_fail)
        self._patch_export_worker.start()


    def _export_and_show_detections(self, boxes: list[dict], rois: list[tuple]):
        export_cfg = self.app_cfg.get("export", {}) if isinstance(self.app_cfg, dict) else {}
        default_dir = os.path.abspath(export_cfg.get("default_dir", "./exports"))
        os.makedirs(default_dir, exist_ok=True)

        base = os.path.splitext(os.path.basename(getattr(self.reader, "path", "slide")))[0]
        out_png = os.path.join(default_dir, f"{base}_detections.png")
        out_meta = os.path.join(default_dir, f"{base}_detections_meta.json")
        out_json = os.path.join(default_dir, f"{base}_detections.json")

        # 仍然导出 png/json（用于离线查看），但显示改用矢量层（更流畅）
        x_min = min(b["bbox"][0] for b in boxes)
        y_min = min(b["bbox"][1] for b in boxes)
        x_max = max(b["bbox"][0] + b["bbox"][2] for b in boxes)
        y_max = max(b["bbox"][1] + b["bbox"][3] for b in boxes)
        ox, oy = x_min, y_min
        W0, H0 = max(1, x_max - x_min), max(1, y_max - y_min)

        side = max(W0, H0)
        sf = 1
        while side > _DET_MAX_SIDE:
            sf *= 2
            side = (side + 1) // 2

        W = max(1, (W0 + sf - 1) // sf)
        H = max(1, (H0 + sf - 1) // sf)
        ds_out = float(sf)

        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        im = Image.fromarray(rgba, "RGBA")
        draw = ImageDraw.Draw(im)
        stroke = 2
        box_color = (0, 0, 255, 255)
        text_color = (255, 255, 255, 255)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

        for b in boxes:
            X, Y, Wb, Hb = b["bbox"]
            x1 = int(round((X - ox) / ds_out))
            y1 = int(round((Y - oy) / ds_out))
            x2 = int(round((X + Wb - ox) / ds_out))
            y2 = int(round((Y + Hb - oy) / ds_out))
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=stroke)

            label = str(b.get("label", ""))
            score = float(b.get("score", 0.0))
            text = f"{label} {score:.2f}".strip()
            if text:
                draw.text((x1 + 2, max(0, y1 - 14)), text, fill=text_color, font=font)

        im.save(out_png, optimize=False)

        meta = {
            "target": "detection_boxes",
            "level": 0,
            "downsample": ds_out,
            "bbox_level0": [int(ox), int(oy), int(W0), int(H0)],
            "path": getattr(self.reader, "path", None),
            "mode": str(getattr(self, "_last_det_mode", "")),
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
            "elapsed_sec": float(getattr(self, "_last_det_elapsed_s", 0.0) or 0.0),
            "params": {
                "imgsz": model_item.get("imgsz", 1024),
                "conf": model_item.get("conf", 0.25),
                "iou": model_item.get("iou", 0.45),
                "overlap": float(self.app_cfg.get("det", {}).get("overlap", 0.2)),
                "nms_iou": float(self.app_cfg.get("det", {}).get("nms_iou", 0.5)),
                "nms_ios": float(self.app_cfg.get("det", {}).get("nms_ios", 0.9)),
            },
            "roi_level0": self._merge_rois_level0(rois),
            "level": 0,
            "downsample": ds_out,
            "boxes": boxes,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        # ----------- 关键：显示不再写 _overlay_rgba，不污染编辑；用矢量层显示 ----------
        try:
            if self._ensure_det_boxes_item():
                self._det_boxes_item.set_boxes(boxes)
                self._det_boxes_item.setVisible(True)
                self._apply_det_overlay_settings()
        except Exception as e:
            print("[DET] vector overlay failed:", repr(e))
            traceback.print_exc()

        elapsed = getattr(self, "_last_det_elapsed_s", None)
        elapsed_txt = f"{elapsed:.2f}s" if isinstance(elapsed, (int, float)) and elapsed is not None else "-"
        _info_box(self, "检测完成", f"耗时：{elapsed_txt}\n\n已导出：\n{out_png}\n{out_meta}\n{out_json}")

    def _merge_rois_level0(self, rois):
        if not rois:
            return [0, 0, 0, 0]
        xs = [x for x, _, _, _ in rois]
        ys = [y for _, y, _, _ in rois]
        x2s = [x + w for x, _, w, _ in rois]
        y2s = [y + h for _, y, _, h in rois]
        x0 = min(xs); y0 = min(ys)
        x1 = max(x2s); y1 = max(y2s)
        return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
    
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
        if not candidates and isinstance(getattr(self, "app_cfg", None), dict):
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

    def _infer_tile_from_meta(self, meta: dict, default: int = 64) -> int:
        """从 overlay meta 推断格子大小（像素），没有就返回 default。"""
        if not isinstance(meta, dict):
            return int(default)

        for k in ("grid_px", "grid", "tile", "tile_px", "tile_size", "cell_px", "patch"):
            v = meta.get(k, None)
            try:
                if v is None:
                    continue
                v = int(float(v))
                if v > 0:
                    return v
            except Exception:
                pass
        return int(default)

    def _edit_det_vis_conf_dialog(self):
        pal = getattr(self, "palette", None)
        if not isinstance(pal, dict):
            pal = {}
        dlg = _DetVisConfDialog(float(getattr(self, "_det_vis_conf", 0.25)), palette=pal, parent=self)
        if dlg.exec() == QDialog.Accepted:
            self._det_vis_conf = dlg.value()
            self._apply_det_overlay_settings()
