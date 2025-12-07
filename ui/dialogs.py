from PySide6.QtWidgets import QFileDialog


def ask_open_wsi(parent):
    return QFileDialog.getOpenFileName(parent, '打开切片/图片', '.', 'Images (*.svs *.tif *.tiff *.png *.jpg)')[0]


def ask_open_files(parent):
    return QFileDialog.getOpenFileNames(parent, '选择图像', '.', 'Images (*.png *.jpg *.tif *.tiff)')[0]


def ask_open_folder(parent):
    return QFileDialog.getExistingDirectory(parent, '选择文件夹', '.')


def ask_save_file(parent, title, filter='PNG (*.png)'):
    return QFileDialog.getSaveFileName(parent, title, '.', filter)[0]