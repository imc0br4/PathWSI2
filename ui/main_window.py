# ui/main_window.py
from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QFrame
from PySide6.QtCore import Qt

from ui.pages.wsi_page import WsiPage
from ui.widgets.custom_title_bar import CustomTitleBar


class MainWindow(QMainWindow):
    def __init__(self, app_cfg, palette, models_cfg, logger):
        super().__init__()

        self.app_cfg = app_cfg
        self.palette = palette or {}
        self.models_cfg = models_cfg
        self.logger = logger

        # 关键：去掉系统标题栏，避免上面那一条系统的 - □ ×
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_StyledBackground, True)

        # 外层 Frame：统一边框和主背景
        container = QFrame(self)
        container.setObjectName("MainFrame")
        container.setStyleSheet(
            "QFrame#MainFrame { border: 1px solid #C0C4CC; background-color: #F5F7FA; }"
        )

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 自定义标题栏（这里的标题就是左上角那串字）
        self.title_bar = CustomTitleBar(self, title="PathWSI-Seg", on_close=self.close)
        layout.addWidget(self.title_bar)

        # 主页面
        self.page = WsiPage(
            app_cfg=self.app_cfg,
            logger=self.logger,
            parent=self,
            models_cfg=self.models_cfg,
            palette=self.palette,
        )
        layout.addWidget(self.page)

        self.setCentralWidget(container)
        self.resize(1600, 900)

        # 统一“医疗蓝”主题
        self._apply_medical_theme()

    # 统一主题
    def _apply_medical_theme(self):
        primary = self.palette.get("primary", "#1C9CD8")
        primary_dark = self.palette.get("primary_dark", "#0077B6")
        border_col = "#D4D7DE"
        bg_main = "#F4F7FB"
        topbar_bg = "#F5FAFF"   # 标题栏和工具栏统一背景

        self.setStyleSheet(f"""
            /* 整个窗口的外框和背景 */
            QFrame#MainFrame {{
                background-color: {bg_main};
                border: 1px solid {border_col};
                border-radius: 0;
            }}

            /* 顶部标题栏：和工具栏统一用同一种浅蓝背景 */
            QFrame#CustomTitleBar {{
                background-color: {topbar_bg};
                border: none;
                border-bottom: 1px solid #D6E4F3;
            }}
            QLabel#WindowTitleLabel {{
                color: #223A5E;
                font-size: 13px;
                font-weight: 600;
            }}
            QPushButton#TitleBarButton,
            QPushButton#TitleBarButtonClose {{
                border: none;
                background: transparent;
                padding: 0 6px;
                font-size: 12px;
            }}
            QPushButton#TitleBarButton:hover {{
                background-color: #E4F2FF;
                border-radius: 3px;
            }}
            QPushButton#TitleBarButtonClose:hover {{
                background-color: #FF4D4F;
                color: white;
                border-radius: 3px;
            }}

            /* 顶部工具条容器：和标题栏同背景，且不再画大圆角底板 */
            QFrame#TopBar,
            QFrame#TopBarFrame {{
                background-color: {topbar_bg};
                border: none;
                margin: 0;          /* 关键：顶部不留空 */
                padding: 4px 8px;   /* 只保留一点点内边距 */
            }}

            /* TopBar 里的按钮（打开/左旋/右旋/分类/检测…） */
            QPushButton[role="toolbar"],
            QToolButton[role="toolbar"] {{
                min-width: 70px;
                padding: 4px 10px;
                border-radius: 6px;
                border: 1px solid #D6E4F3;
                background-color: #FFFFFF;
                color: #2C3E50;
            }}
            QPushButton[role="toolbar"]:hover,
            QToolButton[role="toolbar"]:hover {{
                background-color: #E7F3FF;
                border-color: #B8D6F5;
            }}
            QPushButton[role="toolbar"]:pressed,
            QToolButton[role="toolbar"]:pressed {{
                background-color: #D2E7FF;
                border-color: #A7C7F1;
            }}

            /* 关闭WSI 按钮换个强调色，但保持同一风格 */
            QPushButton[role="toolbar-close"] {{
                min-width: 90px;
                padding: 4px 10px;
                border-radius: 6px;
                border: 1px solid #F5A3A2;
                background-color: #FFECEC;
                color: #D93025;
            }}
            QPushButton[role="toolbar-close"]:hover {{
                background-color: #FFD6D6;
            }}

            /* 常规主按钮（比如底部的“开始推理”） */
            QPushButton.primary-button {{
                background-color: {primary};
                color: white;
                border-radius: 6px;
                border: 1px solid {primary_dark};
                padding: 6px 14px;
            }}
            QPushButton.primary-button:hover {{
                background-color: {primary_dark};
            }}
        """)
