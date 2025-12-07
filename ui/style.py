# ui/style.py
from PySide6.QtGui import QColor


def load_stylesheet() -> str:
    """
    定义并返回全局的“医学专业风格”样式表。
    """
    # 这套配色方案带来了专业、洁净的临床医学感。

    # ---- 医疗风格调色板 ----
    palette = {
        # 背景基调：使用极淡的灰蓝色，比纯白更护眼，更有质感
        "app_bg": "#F5F7FA",
        # 内容区背景：纯白，用于突出显示图像和数据
        "panel_bg": "#FFFFFF",
        # 侧边栏/工具栏背景：稍微深一点的色调，区分层级
        "side_bg": "#EEF1F6",

        # 主色调（医疗蓝）：用于强调、选中状态和主要按钮
        "primary": "#007AFF",
        # 主色调悬停
        "primary_hover": "#0062CC",

        # 文字颜色
        "text_main": "#2C3E50",  # 深灰蓝，比纯黑柔和
        "text_secondary": "#7F8C8D",  # 次要文字

        # 边框与分割线：非常淡的颜色，提供微妙的边界感
        "border_light": "#E4E7ED",
        "border_medium": "#DCDFE6",

        # 列表项状态
        "item_hover_bg": "#EDF6FF",  # 淡淡的蓝色悬停
        "item_selected_bg": "#D4E8FD"  # 稍深的蓝色选中
    }

    # 生成 QSS 字符串
    qss = f"""
    /* ================= 全局设定 ================= */
    QWidget {{
        font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
        font-size: 13px;
        color: {palette["text_main"]};
        background-color: {palette["app_bg"]};
        outline: none; /* 移除焦点虚线框 */
    }}

    /* 禁用状态统一样式 */
    QWidget:disabled {{
        color: {palette["text_secondary"]};
        background-color: #F9FAFC;
    }}

    /* ================= 主窗口结构 ================= */
    QMainWindow {{
        /* 给主窗口一个背景，作为最底层的画布 */
        background-color: {palette["app_bg"]};
    }}

    /* 核心修改：给主窗口的中央部件区域加一个微妙的边框和阴影感，
       创造一种“专业设备面板”被嵌入窗口的感觉 */
    QMainWindow > QWidget {{
         /* 注意：这里需要稍微调整一下，避免影响到浮动窗口等。
            最好的方式是在 MainWindow 代码里给 centralWidget 设置一个 ID */
    }}

    /* 顶部工具栏区域 */
    QWidget#TopBar {{
        background-color: {palette["panel_bg"]};
        /* 底部增加一条微妙的分割线 */
        border-bottom: 1px solid {palette["border_medium"]};
        /* 顶部增加一点内边距，让内容呼吸 */
        padding-top: 4px;
        padding-bottom: 4px;
    }}

    /* 分割器手柄 */
    QSplitter::handle {{
        background-color: {palette["border_light"]};
        /* 让手柄稍微宽一点，更容易抓取，也更有实体感 */
        width: 2px; 
        margin: 1px;
    }}

    /* ================= 左侧缩略图列表 (重点修改) ================= */
    /* 针对 WsiThumbList 的特殊样式，确保在 Python 代码里设置了 objectName */
    QListWidget#WsiThumbList {{
        background-color: {palette["side_bg"]};
        border: none; /* 移除默认边框 */
        /* 右侧加一条分割线，与右侧视图区分 */
        border-right: 1px solid {palette["border_medium"]};
        padding: 8px; /* 增加整体内边距 */
    }}

    /* 列表项样式 */
    QListWidget#WsiThumbList::item {{
        /* 关键：让内部元素（图标和文字）居中对齐 */
        text-align: center; 
        background-color: {palette["panel_bg"]};
        border: 1px solid {palette["border_light"]};
        border-radius: 6px; /* 圆角让界面更友好 */
        padding: 8px;
        margin-bottom: 8px; /* 项与项之间的间距 */
        /* 给每个项加一个微妙的阴影效果，增加立体感 (可选) */
        /* border-bottom: 2px solid {palette["border_light"]}; */
    }}

    /* 列表项悬停态 */
    QListWidget#WsiThumbList::item:hover {{
        background-color: {palette["item_hover_bg"]};
        border-color: {palette["primary"]};
    }}

    /* 列表项选中态 */
    QListWidget#WsiThumbList::item:selected {{
        background-color: {palette["item_selected_bg"]};
        border: 2px solid {palette["primary"]}; /* 加粗边框强调 */
        color: {palette["primary"]};
        /* 选中时移除默认的虚线框 */
        outline: none;
    }}


    /* ================= 按钮与控件 ================= */
    /* 扁平化工具栏按钮 */
    QToolButton, QPushButton {{
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 4px;
        padding: 4px 8px;
        min-height: 22px;
    }}
    QToolButton:hover, QPushButton:hover {{
        background-color: {palette["item_hover_bg"]};
        border-color: {palette["border_light"]};
        color: {palette["primary"]};
    }}
    QToolButton:checked, QPushButton:checked {{
        background-color: {palette["item_selected_bg"]};
        border-color: {palette["primary"]};
        color: {palette["primary"]};
    }}
    /* 菜单按钮带小箭头的样式调整 */
    QToolButton::menu-indicator {{
        image: none; /* 移除默认丑陋的箭头，可以后续用图标替换，或者就这样简洁显示 */
        width: 0px;
    }}

    /* 输入框等通用控件 */
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background-color: {palette["panel_bg"]};
        border: 1px solid {palette["border_medium"]};
        border-radius: 4px;
        padding: 3px;
        min-height: 22px;
    }}
    QLineEdit:focus, QComboBox:focus {{
        border: 1px solid {palette["primary"]};
    }}

    /* 滚动条美化 (更细、更现代) */
    QScrollBar:vertical {{ background: transparent; width: 8px; margin: 0; }}
    QScrollBar::handle:vertical {{ background: #C0C4CC; border-radius: 4px; min-height: 20px; }}
    QScrollBar::handle:vertical:hover {{ background: #909399; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
    QScrollBar:horizontal {{ background: transparent; height: 8px; margin: 0; }}
    QScrollBar::handle:horizontal {{ background: #C0C4CC; border-radius: 4px; min-width: 20px; }}

    /* 标签 */
    QLabel {{
        color: {palette["text_main"]};
    }}
    """
    return qss