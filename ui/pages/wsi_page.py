# ui/pages/wsi_page.py
from __future__ import annotations

import os, json
from typing import Optional, Tuple, List
import numpy as np

from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QSizePolicy, QFrame,
    QPushButton, QToolButton, QLabel, QMenu, QStyle, QGraphicsView
)

from wsi.reader import WsiReader
from ui.viewer.wsi_view import WsiView
from ui.viewer.roi_tools import RoiTool
from ui.viewer.overlay_item import OverlayItem
from ui.widgets.thumb_list import WsiThumbList

# 外部工具与 mixins
from ui.utils.qt_graphics import _hamburger_icon
from ui.pages.wsi_mixins.edit_ops import EditOpsMixin
from ui.pages.wsi_mixins.ui_builders import UiBuildersMixin
from ui.pages.wsi_mixins.file_ops import FileOpsMixin
from ui.pages.wsi_mixins.overlay_ops import OverlayOpsMixin
from ui.pages.wsi_mixins.roi_ops import RoiOpsMixin
from ui.pages.wsi_mixins.info_status import InfoStatusMixin
from ui.pages.wsi_mixins.models_cfg import ModelsCfgMixin
from ui.pages.wsi_mixins.cls_integration import ClsIntegrationMixin
from ui.pages.wsi_mixins.minimap import MiniMapMixin
from ui.pages.wsi_mixins.det_integration import DetIntegrationMixin

__all__ = ["WsiPage"]

_OPEN_FILTER = "Slides/Images (*.svs *.ndpi *.scn *.mrxs *.svslide *.tif *.tiff *.png *.jpg *.jpeg *.bmp)"
_OVERLAY_FILTER = "Overlay (*.png);;All (*)"

# —— 导出条带安全阈值（如需做条带级大图导出可用；当前文件未启用条带导出逻辑）——
_EXPORT_WARN_MB = 2000
_STRIPE_MAX_MB = 200
_MIN_STRIPE_H = 64
_MAX_PNG_SIDE = 12000


class WsiPage(
    EditOpsMixin,
    # 组合功能
    UiBuildersMixin,
    FileOpsMixin,
    OverlayOpsMixin,
    RoiOpsMixin,
    InfoStatusMixin,
    ModelsCfgMixin,
    ClsIntegrationMixin,
    MiniMapMixin,
    DetIntegrationMixin,
    # QWidget 放在最后，避免其 eventFilter 抢先命中
    QWidget,
):
    def __init__(self, app_cfg, logger, parent=None, models_cfg=None, palette=None):
        # 直接初始化 QWidget；mixin 通常不需要 __init__ 或内部会使用 super()
        QWidget.__init__(self, parent)

        self.app_cfg = app_cfg
        self.log = logger
        self.models_cfg = models_cfg if models_cfg is not None else None
        self.palette = palette or {}

        # ========== 1) 视图与核心对象（最先建，后续需要引用） ==========
        self.view = WsiView(app_cfg)
        # 开启鼠标跟踪，悬停/拖拽预览才能在不按键时也收到 move 事件
        self.view.setMouseTracking(True)
        self.view.viewport().setMouseTracking(True)

        # 更丝滑的交互设置（基础）
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setFrameShape(QFrame.NoFrame)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        # self.view.setRenderHints(self.view.renderHints() & ~QPainter.SmoothPixmapTransform)

        # 如果启用了 MiniMapMixin 的部件，先建壳
        if hasattr(self, "_init_minimap_widgets"):
            self._init_minimap_widgets()
        self.view.metricsChanged.connect(lambda *_: getattr(self, "_update_minimap_rect", lambda: None)())
        # 2) 第一次有 reader 且 minimap 还未就绪时，自动绑定 minimap（只会触发一次）
        self.view.metricsChanged.connect(
            lambda *_: (getattr(self, "_minimap_bind_reader", lambda r: None)(self.view.reader)
                        if (not getattr(self, "_minimap_ready", False) and getattr(self.view, "reader", None))
                        else None)
        )
        self.reader: Optional[WsiReader] = None
        self.overlay_item: Optional[OverlayItem] = None

        # ========== 2) 所有“状态变量/定时器”先准备好，避免 eventFilter 早到找不到 ==========
        # —— 编辑态状态 —— #
        self._edit_mode = 'off'
        self._edit_rect_item = None
        self._edit_origin_scene = None
        self._erase_drag_origin_scene = None
        self._panning = False
        self._pan_last_pos = None
        self._hover_preview_item = None
        self._eraser_units = 1
        self._hover_preview_enabled = False

        # —— 橡皮 hover 延迟预览 —— #
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(150)
        self._hover_timer.timeout.connect(self._on_hover_timer)
        self._hover_pending_pt = None

        # —— 叠加/网格/脏区合并刷新 —— #
        self._overlay_rgba: Optional[np.ndarray] = None
        self._overlay_ds: float = 1.0
        self._overlay_pos: Tuple[int, int] = (0, 0)
        self._overlay_meta: Optional[dict] = None
        self._overlay_opacity = 0.5

        self._grid_tile_px = 64
        self._grid_fill_rgba = np.array([255, 0, 0, 180], dtype=np.uint8)
        self._grid_backup = None

        self._dirty_rect = None
        self._flush_timer = QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.setInterval(16)
        self._flush_timer.timeout.connect(self._flush_dirty)

        # ========== 3) 顶部工具条 / UI 结构 ==========
        topBar = QWidget(self)
        topBar.setObjectName("TopBar")
        bar = QHBoxLayout(topBar)
        bar.setContentsMargins(8, 4, 8, 4)
        bar.setSpacing(6)
        topBar.setFixedHeight(36)

        topBar.setObjectName("TopBar")
        # self.setStyleSheet("""
        # QWidget#TopBar { background:#fafafa; border-bottom:1px solid #ddd; }
        # QToolButton, QPushButton { padding:2px 8px; min-height:24px; }
        # QSplitter::handle { background:#e5e5e5; width:4px; }
        # """)

        st = self.style()
        self.btn_open = QPushButton("打开");
        self.btn_open.setIcon(st.standardIcon(QStyle.SP_DialogOpenButton));
        self.btn_open.setObjectName("BtnSecondary")     # ★ 新增：次要按钮样式
        bar.addWidget(self.btn_open)

        self.btn_rotl = QPushButton("左旋");
        self.btn_rotl.setIcon(st.standardIcon(QStyle.SP_ArrowBack));
        self.btn_rotl.setObjectName("BtnIcon")         # ★ 新增：图标类按钮
        bar.addWidget(self.btn_rotl)

        self.btn_rotr = QPushButton("右旋");
        self.btn_rotr.setIcon(st.standardIcon(QStyle.SP_ArrowForward));
        self.btn_rotr.setObjectName("BtnIcon")         # ★ 新增
        bar.addWidget(self.btn_rotr)

        self.btn_open_folder = QPushButton("打开文件夹");
        self.btn_open_folder.setIcon(st.standardIcon(QStyle.SP_DirOpenIcon));
        self.btn_open_folder.setObjectName("BtnSecondary")  # ★ 新增
        bar.addWidget(self.btn_open_folder)

        self.btn_cls = QPushButton("分类");
        self.btn_cls.setObjectName("BtnPrimary")       # ★ 新增：高亮主按钮（分类）
        bar.addWidget(self.btn_cls)

        self._build_det_button(bar)  # ← 保持不变（检测按钮使用通用按钮样式）


        # ROI / Overlay / 编辑 菜单（由 UiBuildersMixin 提供）
        self._build_roi_compact(bar)
        self._build_overlay_menu(bar)
        self._build_edit_toolbar(bar)

        # —— 编辑主按钮 / 菜单动作 —— #
        self._main_button_mode = 'add'  # 默认主模式，可改为 'erase'
        self.btn_edit.setPopupMode(QToolButton.MenuButtonPopup)

        # 主按钮：off <-> 当前主模式
        self.btn_edit.clicked.connect(
            lambda: self._set_edit_mode(self._main_button_mode if self._edit_mode == 'off' else 'off')
        )

        def _switch_edit_mode(mode: str):
            self._main_button_mode = mode
            self._set_edit_mode(mode)

        # 菜单：立刻切换到指定模式，并更新主按钮当前模式
        self.act_edit_add.triggered.connect(lambda: _switch_edit_mode('add'))
        self.act_edit_erase.triggered.connect(lambda: _switch_edit_mode('erase'))

        # 菜单：撤销 / 保存 / 退出
        self.act_edit_undo.triggered.connect(self._cancel_overlay_edits)
        self.act_edit_save.triggered.connect(self._save_overlay_edits)
        self.act_edit_exit.triggered.connect(lambda: self._set_edit_mode('off'))

        bar.addStretch(1)
        self.btn_help = QToolButton(self);
        self.btn_help.setIcon(_hamburger_icon());
        self.btn_help.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.btn_help.setObjectName("BtnIcon")  # ★ 新增：统一为图标按钮样式
        help_menu = QMenu(self.btn_help)
        help_menu.addAction("双击：回到首屏视图（适配窗口）")
        help_menu.addAction("滚轮缩放：自动吸附层级，减少模糊")
        help_menu.addSeparator()
        help_menu.addAction("ROI：主按钮单击即可展开菜单")
        help_menu.addAction("叠加：加载 PNG；若有同名 _meta.json 自动对齐")
        help_menu.addAction("编辑：添加矩形/橡皮（Shift 单击切换；[/] 调橡皮大小）")
        self.btn_help.setMenu(help_menu); bar.addWidget(self.btn_help)

        self.lbl_info = QLabel("倍率: - | ppd: - | level: - | ds: -")
        self.lbl_info.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_info.setObjectName("LblStatus")   # ★ 新增：用样式统一控制颜色
        bar.addWidget(self.lbl_info)

        # 主体布局与分割条
        layout = QVBoxLayout(self)
        # 上边距改为 0，这样 TopBar 就紧贴自定义标题栏，不再有空一条
        layout.setContentsMargins(6, 0, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(topBar)

        split = QSplitter(Qt.Horizontal, self)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(4)
        self._split = split

        # 左侧缩略列表
        self.thumb = WsiThumbList(self)
        self.thumb.setMinimumWidth(260)
        self.thumb.setMaximumWidth(360)
        self.thumb.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.thumb.sigOpenPath.connect(self._open_wsi_path)
        split.addWidget(self.thumb)

        # 右侧 WSI 视图
        split.addWidget(self.view)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        layout.addWidget(split)
        QTimer.singleShot(0, self._tune_split_sizes)

        # ========== 4) ROI 工具与快捷键 ==========
        self.roi_tool = RoiTool(self.view)
        self.roi_tool.modeChanged.connect(self._on_roi_mode_changed)
        self.roi_tool.selectionChanged.connect(self._on_roi_selection_changed)

        from PySide6.QtGui import QKeySequence, QShortcut
        self._shortcut_delete    = QShortcut(QKeySequence.Delete, self.view);              self._shortcut_delete.activated.connect(self._on_delete_selected_roi_silent)
        self._shortcut_backspace = QShortcut(QKeySequence(Qt.Key_Backspace), self.view);   self._shortcut_backspace.activated.connect(self._on_delete_selected_roi_silent)
        self._shortcut_esc       = QShortcut(QKeySequence(Qt.Key_Escape), self.view);      self._shortcut_esc.activated.connect(self._on_escape_clear)
        self._shortcut_sel_all   = QShortcut(QKeySequence("Ctrl+A"), self.view);           self._shortcut_sel_all.activated.connect(self._on_select_all)
        self._shortcut_sel_none  = QShortcut(QKeySequence("Ctrl+D"), self.view);           self._shortcut_sel_none.activated.connect(self._on_escape_clear)

        # ========== 5) 信号连接 ==========
        self.btn_open.clicked.connect(self.on_open)
        self.btn_open_folder.clicked.connect(self._on_open_folder)
        self.btn_close = QPushButton("关闭WSI", self);
        self.btn_close.setIcon(self.style().standardIcon(QStyle.SP_DialogCloseButton));
        self.btn_close.setObjectName("BtnDanger")   # ★ 新增：强调关闭/危险操作
        bar.addWidget(self.btn_close)

        self.btn_close.clicked.connect(self.on_close_wsi)

        self.btn_rotl.clicked.connect(self.view.rotate_left)
        self.btn_rotr.clicked.connect(self.view.rotate_right)
        self.btn_cls.clicked.connect(self._on_cls_clicked)

        # ========== 6) 事件过滤安装（最后装，前面属性都准备好了） ==========
        self.view.viewport().installEventFilter(self)
        self.view.installEventFilter(self)

        # 视口滚动时，让缩略图红框（若实现）跟随
        try:
            self.view.horizontalScrollBar().valueChanged.connect(lambda _: getattr(self, "_update_minimap_rect", lambda: None)())
            self.view.verticalScrollBar().valueChanged.connect(lambda _: getattr(self, "_update_minimap_rect", lambda: None)())
        except Exception:
            pass

        # ========== 7) 模型配置 & 初始状态 & 信息定时器 ==========
        if self.models_cfg is None:
            self.models_cfg = self._load_models_cfg_default()
        else:
            self.models_cfg = self._normalize_models_cfg(self.models_cfg)

        self._is_closing = False
        self._update_buttons(active=False)

        self._info_timer = QTimer(self)
        self._info_timer.setInterval(150)
        self._info_timer.timeout.connect(self._refresh_info)

        self._apply_medical_theme()
    # ---- 事件过滤器转发（双保险）：确保调用 EditOpsMixin 的实现 ----
    # 若 EditOpsMixin 内部使用 super().eventFilter 传递，其他 mixin/父类仍可接力。
    def eventFilter(self, obj, ev):
        return EditOpsMixin.eventFilter(self, obj, ev)

    # ---- 工具：缩略列表项打开 WSI 的小封装（让 WsiThumbList -> FileOpsMixin._open_slide）----
    def _open_wsi_path(self, path: str):
        # 由 FileOpsMixin 提供的 _open_slide 执行真正加载
        if hasattr(self, "_open_slide"):
            self._open_slide(path)

    # ---- 初始分割条宽度调优：避免左侧过宽或过窄 ----
    def _tune_split_sizes(self):
        try:
            W = max(800, self.width())
            left = min(320, max(260, int(W * 0.18)))  # 左侧约 18%，限制 260~320
            right = max(400, W - left - 16)
            self._split.setSizes([left, right])
        except Exception:
            pass

    # ---- 窗口事件里顺带调整缩略图/小地图布局 ----
    def showEvent(self, e):
        super().showEvent(e)
        QTimer.singleShot(0, self._tune_split_sizes)
        if hasattr(self, "_layout_minimap"):
            self._layout_minimap()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, "_layout_minimap"):
            self._layout_minimap()

    def _apply_medical_theme(self):
        """
        应用偏医疗影像风格的主题：
          - 冷色调蓝绿色
          - 主界面统一风格
          - 对话框只做背景/按钮配色，不再全局修改 SpinBox/ComboBox 等子控件
        """
        self.setStyleSheet("""
        /* ===========================
         * 全局基础
         * =========================== */
        QWidget {
            background-color: #F3F7FB;
            font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
            font-size: 11pt;
        }

        QLabel {
            color: #2C3E55;
        }

        /* 状态信息文字（倍率/ppd） */
        QLabel#LblStatus {
            color: #36506A;
            font-size: 10pt;
        }

        /* ===========================
         * 顶部工具栏（主界面）
         * 与标题栏背景保持一致
         * =========================== */
        QWidget#TopBar {
            background-color: #F5FAFF;
            border: none;
        }

        QWidget#TopBar QPushButton,
        QWidget#TopBar QToolButton {
            background-color: #FFFFFF;
            border-radius: 5px;
            border: 1px solid #D6E4F2;
            padding: 3px 10px;
            color: #1F3B58;
        }
        QWidget#TopBar QPushButton:hover,
        QWidget#TopBar QToolButton:hover {
            background-color: #E7F2FF;
        }
        QWidget#TopBar QPushButton:pressed,
        QWidget#TopBar QToolButton:pressed {
            background-color: #D6E8FF;
        }

        /* 主操作按钮（分类） */
        QPushButton#BtnPrimary {
            background-color: #1A90FF;
            border-color: #1A90FF;
            color: #FFFFFF;
            font-weight: 600;
        }
        QPushButton#BtnPrimary:hover {
            background-color: #147FE0;
        }
        QPushButton#BtnPrimary:pressed {
            background-color: #0F6AC0;
        }

        /* 次要按钮（打开、打开文件夹等） */
        QPushButton#BtnSecondary {
            background-color: #FFFFFF;
            border-color: #C8D9EE;
            color: #1F3B58;
        }
        QPushButton#BtnSecondary:hover {
            background-color: #EDF4FF;
        }

        /* 危险按钮（关闭WSI） */
        QPushButton#BtnDanger {
            background-color: #FF4D4F;
            border-color: #FF4D4F;
            color: #FFFFFF;
            font-weight: 500;
        }
        QPushButton#BtnDanger:hover {
            background-color: #E64547;
        }

        /* 图标类按钮（左右旋转、帮助等） */
        QPushButton#BtnIcon,
        QToolButton#BtnIcon {
            padding: 3px 6px;
            min-width: 28px;
            background-color: #FFFFFF;
            border-radius: 5px;
            border: 1px solid #D6E4F2;
            color: #1F3B58;
        }
        QPushButton#BtnIcon:hover,
        QToolButton#BtnIcon:hover {
            background-color: #EDF4FF;
        }

        /* 左侧缩略图列表：像病例列表 */
        QListWidget#WsiThumbList {
            background-color: #F7FAFD;
            border-radius: 6px;
            border: 1px solid #DCE7F5;
        }
        QListWidget#WsiThumbList::item {
            padding: 6px 4px;
            margin: 2px 4px;
        }
        QListWidget#WsiThumbList::item:selected {
            background-color: #E3F2FF;
            border-left: 3px solid #1A90FF;
            color: #123456;
        }

        /* 分割条：淡蓝色 */
        QSplitter::handle {
            background-color: #E0ECF8;
        }
        QSplitter::handle:hover {
            background-color: #C9DDF5;
        }

        /* ===========================
         * 对话框 / 弹窗（分类、检测等）
         * 只设置背景/边框/按钮风格
         * =========================== */
        QDialog, QMessageBox {
            background-color: #F9FDFF;
            border-radius: 4px;
            border: 1px solid #C7DDF2;
        }

        QDialog#MedicalDialog {
            background-color: #F9FDFF;
            border-radius: 4px;
            border: 1px solid #9CC3EB;
        }

        QDialog QLabel,
        QMessageBox QLabel {
            color: #253A4F;
        }

        QDialog QGroupBox {
            border: 1px solid #D1E3F4;
            border-radius: 4px;
            margin-top: 8px;
        }
        QDialog QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
            color: #1A90FF;
            font-weight: 500;
            background-color: transparent;
        }

        /* 对话框里的按钮：与主界面次要按钮一致 */
        QDialog QPushButton,
        QMessageBox QPushButton {
            background-color: #FFFFFF;
            border-radius: 4px;
            border: 1px solid #C8D9EE;
            padding: 4px 12px;
            min-width: 72px;
            color: #1F3B58;
        }
        QDialog QPushButton:hover,
        QMessageBox QPushButton:hover {
            background-color: #EDF4FF;
        }
        QDialog QPushButton:pressed,
        QMessageBox QPushButton:pressed {
            background-color: #D6E8FF;
        }

        QDialog QPushButton#BtnPrimary {
            background-color: #1A90FF;
            border-color: #1A90FF;
            color: #FFFFFF;
            font-weight: 600;
        }
        QDialog QPushButton#BtnPrimary:hover {
            background-color: #147FE0;
        }

        QDialog QPushButton#BtnDanger {
            background-color: #FF4D4F;
            border-color: #FF4D4F;
            color: #FFFFFF;
        }
        QDialog QPushButton#BtnDanger:hover {
            background-color: #E64547;
        }

        /* Tab / 表格类弹窗：保留原来的淡蓝风格 */
        QDialog QTabWidget::pane {
            border: 1px solid #C7DDF2;
            border-radius: 4px;
            top: 1px;
        }
        QDialog QTabBar::tab {
            background-color: #E5F0FF;
            border: 1px solid #C7DDF2;
            border-bottom: none;
            padding: 4px 12px;
            min-width: 60px;
            color: #33516C;
        }
        QDialog QTabBar::tab:selected {
            background-color: #FFFFFF;
            color: #1A90FF;
            font-weight: 500;
        }

        QDialog QTableView {
            background-color: #FFFFFF;
            border: 1px solid #C7DDF2;
            gridline-color: #E0ECF8;
            selection-background-color: #E3F2FF;
            selection-color: #102A43;
        }
        QDialog QHeaderView::section {
            background-color: #EDF4FF;
            border: 1px solid #D7E4F4;
            padding: 3px 6px;
            color: #345067;
        }
        """)



