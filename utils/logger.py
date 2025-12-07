import logging
from logging.handlers import RotatingFileHandler
import os


def _ensure_log_dir(logfile: str):
    try:
        d = os.path.dirname(os.path.abspath(logfile))
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
    except Exception:
        pass


def get_logger(name: str, logfile: str):
    """
    返回一个已配置好的 logger（可重入、不会重复添加 Handler）。
    行为与原实现一致：INFO 级别、控制台输出 + 轮转文件输出、UTF-8。
    """
    logger = logging.getLogger(name)

    # 若已配置过，直接返回，避免重复添加 handler（多次调用/单测场景）
    if getattr(logger, "_pgd_configured", False):
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')

    # 控制台
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    # 文件（轮转）
    _ensure_log_dir(logfile)
    fh = RotatingFileHandler(
        logfile, maxBytes=2 * 1024 * 1024, backupCount=3, encoding='utf-8'
    )
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    # 先清掉可能存在的旧 handler（防止外部误配置导致重复输出）
    for h in list(logger.handlers):
        logger.removeHandler(h)

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False

    # 打个旗子，防重复配置
    logger._pgd_configured = True  # type: ignore[attr-defined]
    return logger
