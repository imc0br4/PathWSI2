import sys, json, yaml, os
from PySide6.QtWidgets import QApplication
from utils.logger import get_logger
from ui.main_window import MainWindow
from ui.style import load_stylesheet

APP_DIR = os.path.dirname(__file__)
CFG_APP = os.path.join(APP_DIR, 'configs', 'app.yaml')
CFG_PALETTE = os.path.join(APP_DIR, 'configs', 'palette.yaml')
CFG_MODELS = os.path.join(APP_DIR, 'configs', 'models.json')


def _load_yaml(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_cfg():
    """
    读取配置文件。若缺失或解析失败，抛出可读异常，便于上层记录日志并退出。
    行为与原来一致（成功则返回三元组）。
    """
    if not os.path.exists(CFG_APP):
        raise FileNotFoundError(f"Config file not found: {CFG_APP}")
    if not os.path.exists(CFG_PALETTE):
        raise FileNotFoundError(f"Config file not found: {CFG_PALETTE}")
    if not os.path.exists(CFG_MODELS):
        raise FileNotFoundError(f"Config file not found: {CFG_MODELS}")

    app_cfg = _load_yaml(CFG_APP)
    palette = _load_yaml(CFG_PALETTE)
    models = _load_json(CFG_MODELS)
    return app_cfg, palette, models


def main():
    print("tettedatda")

    print("tettedatda")
    print("tettedatda")
    # 日志目录
    os.makedirs(os.path.join(APP_DIR, 'logs'), exist_ok=True)
    logger = get_logger('app', os.path.join(APP_DIR, 'logs', 'app.log'))

    # 配置加载（失败时记录日志并退出，而不是堆栈）
    try:
        app_cfg, palette, models = load_cfg()
    except Exception as e:
        # 提前打印到 stderr，保证看得到
        sys.stderr.write(f"[PathWSI-Seg] Failed to load configs: {e}\n")
        try:
            logger.exception("Failed to load configs", exc_info=e)
        except Exception:
            pass
        raise SystemExit(2)

    app = QApplication(sys.argv)
    app.setApplicationName('PathWSI-Seg')

    # ★ 在这里应用全局样式表
    # 我们不再需要传递 palette 给 load_stylesheet，因为它内部已经定义好了
    app.setStyleSheet(load_stylesheet())

    # ... (MainWindow 初始化和显示代码保持不变)
    win = MainWindow(app_cfg, palette, models, logger)
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
