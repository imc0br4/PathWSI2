from __future__ import annotations
from PySide6.QtWidgets import QHBoxLayout,QMenu,QToolButton
from PySide6.QtGui import QKeySequence, QShortcut, QAction
from PySide6.QtCore import Qt
class UiBuildersMixin:
    def _build_roi_compact(self, bar_layout: QHBoxLayout):
        self.act_roi_draw = QAction("绘制 ROI", self)
        self.act_roi_edit = QAction("编辑 ROI", self)
        self.act_roi_del = QAction("删除选中 ROI (Del)", self)
        self.act_roi_save = QAction("保存 ROI(10×)", self)
        self.act_roi_clear = QAction("清空所有 ROI", self)
        self.act_roi_exit = QAction("退出 ROI 模式", self)

        self.act_roi_draw.triggered.connect(lambda: self._enter_roi(draw=True))
        self.act_roi_edit.triggered.connect(lambda: self._enter_roi(draw=False))
        self.act_roi_del.triggered.connect(self._on_delete_selected_roi)
        self.act_roi_save.triggered.connect(self.on_save_roi)
        self.act_roi_clear.triggered.connect(self._on_roi_clear)
        self.act_roi_exit.triggered.connect(self._exit_roi)

        m = QMenu(self)
        m.addAction(self.act_roi_draw)
        m.addAction(self.act_roi_edit)
        m.addSeparator()
        m.addAction(self.act_roi_del)
        m.addAction(self.act_roi_save)
        m.addAction(self.act_roi_clear)
        m.addSeparator()
        m.addAction(self.act_roi_exit)

        self.btn_roi = QToolButton(self)
        self.btn_roi.setText("ROI 菜单")
        self.btn_roi.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.btn_roi.setPopupMode(QToolButton.MenuButtonPopup)
        self.btn_roi.setMenu(m)
        self.btn_roi.setToolTip("单击按钮即可展开 ROI 菜单")
        self.btn_roi.clicked.connect(self.btn_roi.showMenu)
        bar_layout.addWidget(self.btn_roi)

    def _build_overlay_menu(self, bar_layout: QHBoxLayout):
        self.act_ov_load = QAction("加载叠加图…", self)
        self.act_ov_clear = QAction("清除叠加图", self)
        self.act_ov_load.triggered.connect(self.on_load_overlay)
        self.act_ov_clear.triggered.connect(self.on_clear_overlay)
        # self.act_ov_export = QAction("导出分类 Overlay…", self)
        # self.act_ov_export.triggered.connect(self._on_export_overlay_clicked)

        # 透明度子菜单
        sm_alpha = QMenu("透明度", self)
        for pct in (0, 25, 50, 75, 100):
            a = QAction(f"{pct}%", self)
            a.triggered.connect(lambda _, v=pct: self._set_overlay_opacity(v / 100.0))
            sm_alpha.addAction(a)

        m = QMenu(self)
        m.addAction(self.act_ov_load)
        m.addAction(self.act_ov_clear)
        # m.addAction(self.act_ov_export)
        m.addMenu(sm_alpha)

        self.btn_overlay = QToolButton(self)
        self.btn_overlay.setText("叠加")
        self.btn_overlay.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.btn_overlay.setPopupMode(QToolButton.MenuButtonPopup)
        self.btn_overlay.setMenu(m)
        self.btn_overlay.setToolTip("加载/清除叠加图，调整透明度")
        self.btn_overlay.clicked.connect(self.btn_overlay.showMenu)
        bar_layout.addWidget(self.btn_overlay)

    def _build_edit_toolbar(self, bar_layout: QHBoxLayout):
        # 菜单动作
        self.act_edit_add = QAction("添加矩形", self)
        self.act_edit_erase = QAction("橡皮擦（单块）", self)
        self.act_edit_undo = QAction("撤销全部修正", self)
        self.act_edit_save = QAction("保存修正为 Overlay…", self)
        self.act_edit_exit = QAction("退出编辑", self)

        self.act_edit_add.triggered.connect(lambda: self._set_edit_mode('add'))
        self.act_edit_erase.triggered.connect(lambda: self._set_edit_mode('erase'))
        if hasattr(self, "_cancel_overlay_edits"):
            self.act_edit_undo.triggered.connect(self._cancel_overlay_edits)
        else:
            self.act_edit_undo.setEnabled(False)
            self.act_edit_undo.setToolTip("缺少 EditOpsMixin._cancel_overlay_edits")
        if hasattr(self, "_save_overlay_edits"):
            self.act_edit_save.triggered.connect(self._save_overlay_edits)
        else:
            self.act_edit_save.setEnabled(False)
            self.act_edit_save.setToolTip("缺少 EditOpsMixin._save_overlay_edits")
        self.act_edit_exit.triggered.connect(lambda: self._set_edit_mode('off'))

        m = QMenu(self)
        m.addAction(self.act_edit_add)
        m.addAction(self.act_edit_erase)
        m.addSeparator()
        m.addAction(self.act_edit_undo)
        m.addAction(self.act_edit_save)
        m.addSeparator()
        m.addAction(self.act_edit_exit)

        self.btn_edit = QToolButton(self)
        self.btn_edit.setText("编辑")
        self.btn_edit.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.btn_edit.setPopupMode(QToolButton.MenuButtonPopup)
        self.btn_edit.setMenu(m)
        self.btn_edit.setToolTip("进入编辑：添加矩形/单块擦除（Shift 快速切换；右键平移）")
        # self.btn_edit.clicked.connect(self.btn_edit.showMenu)
        self.btn_edit.setEnabled(False)  # 初始禁用，加载 overlay 后再启用
        bar_layout.addWidget(self.btn_edit)