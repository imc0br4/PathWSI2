# ui/widgets/custom_title_bar.py
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt, QPoint, QSize
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QToolButton, QStyle
)


class CustomTitleBar(QWidget):
    """
    扁平风格标题栏：
      - 背景色 #F5FAFF（和主界面 TopBar 一致）
      - 使用 Qt 系统图标：最小化 / 最大化 / 关闭
      - 支持拖动、双击最大化/还原
      - show_maximize=False 时用于对话框（隐藏最大化按钮）
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        title: str = "",
        on_close: Optional[Callable[[], None]] = None,
        show_maximize: bool = True,
    ):
        super().__init__(parent)
        self.setObjectName("MainTitleBar")

        self._on_close = on_close
        self._drag_offset: QPoint | None = None
        self._show_maximize = show_maximize

        self.setFixedHeight(32)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 6, 0)
        lay.setSpacing(4)

        self.lbl_title = QLabel(title, self)
        self.lbl_title.setObjectName("MainTitleLabel")
        self.lbl_title.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        lay.addWidget(self.lbl_title)
        lay.addStretch(1)

        # --- 右上三个按钮：统一尺寸 + 系统图标 ---
        style = self.style()

        self.btn_min = QToolButton(self)
        self.btn_min.setObjectName("WinButton")
        self.btn_min.setIcon(style.standardIcon(QStyle.SP_TitleBarMinButton))
        self.btn_min.setIconSize(QSize(10, 10))
        self.btn_min.setFixedSize(32, 22)

        self.btn_max = QToolButton(self)
        self.btn_max.setObjectName("WinButton")
        self.btn_max.setIcon(style.standardIcon(QStyle.SP_TitleBarMaxButton))
        self.btn_max.setIconSize(QSize(10, 10))
        self.btn_max.setFixedSize(32, 22)

        self.btn_close = QToolButton(self)
        self.btn_close.setObjectName("WinButtonClose")
        self.btn_close.setIcon(style.standardIcon(QStyle.SP_TitleBarCloseButton))
        self.btn_close.setIconSize(QSize(10, 10))
        self.btn_close.setFixedSize(32, 22)

        lay.addWidget(self.btn_min)
        if show_maximize:
            lay.addWidget(self.btn_max)
        lay.addWidget(self.btn_close)

        self.btn_min.clicked.connect(self._on_minimize)
        self.btn_max.clicked.connect(self._on_max_restore)
        self.btn_close.clicked.connect(self._on_close_clicked)

        self._apply_style()

    # ---------- 统一样式（浅蓝背景 + 扁平按钮） ----------
    def _apply_style(self):
        self.setStyleSheet(
            """
            QWidget#MainTitleBar {
                background-color: #F5FAFF;
            }
            QLabel#MainTitleLabel {
                color: #223A4E;
                font-size: 10pt;
                font-weight: 600;
            }
            QToolButton#WinButton,
            QToolButton#WinButtonClose {
                border: none;
                background-color: transparent;
                padding: 0;
            }
            QToolButton#WinButton:hover {
                background-color: #E5F0FF;
            }
            QToolButton#WinButtonClose:hover {
                background-color: #FBEAEA;
            }
        """
        )

    def setTitle(self, text: str):
        self.lbl_title.setText(text)

    # ---------- 找到真正的顶层窗口 ----------
    def _parent_window(self):
        w = self.parent()
        while w is not None and not w.isWindow():
            w = w.parent()
        return w

    # ---------- 按钮行为 ----------
    def _on_minimize(self):
        w = self._parent_window()
        if w:
            w.showMinimized()

    def _on_max_restore(self):
        if not self._show_maximize:
            return
        w = self._parent_window()
        if not w:
            return
        if w.isMaximized():
            w.showNormal()
        else:
            w.showMaximized()

    def _on_close_clicked(self):
        if callable(self._on_close):
            self._on_close()
            return
        w = self._parent_window()
        if w:
            w.close()

    # ---------- 拖动标题栏移动窗口 ----------
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            gw = e.globalPosition().toPoint() if hasattr(e, "globalPosition") else e.globalPos()
            w = self._parent_window()
            if w:
                self._drag_offset = gw - w.frameGeometry().topLeft()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if (e.buttons() & Qt.LeftButton) and self._drag_offset is not None:
            gw = e.globalPosition().toPoint() if hasattr(e, "globalPosition") else e.globalPos()
            w = self._parent_window()
            if w:
                w.move(gw - self._drag_offset)
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._drag_offset = None
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.LeftButton and self._show_maximize:
            self._on_max_restore()
        super().mouseDoubleClickEvent(e)
