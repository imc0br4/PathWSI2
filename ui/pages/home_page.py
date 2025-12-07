# ui/pages/home_page.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton

class HomePage(QWidget):
    def __init__(self, router):
        super().__init__()
        self.router = router
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel('<h2>PathWSI-Seg</h2><p>欢迎使用。左侧进入 WSI 浏览或分类。</p>'))
        btn1 = QPushButton('打开 WSI 浏览')
        btn1.clicked.connect(lambda: self.router.goto('wsi'))
        btn2 = QPushButton('进入分类')
        btn2.clicked.connect(lambda: self.router.goto('cls'))
        lay.addWidget(btn1); lay.addWidget(btn2)
        lay.addStretch()
