import os

# 扩充到与你的打开过滤器一致，兼容更多 WSI 容器格式
IMG_EXTS = {
    '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp',
    '.svs', '.ndpi', '.scn', '.mrxs', '.svslide'
}


def list_images(folder: str):
    files = []
    try:
        names = os.listdir(folder)
    except Exception:
        return files  # 保持静默失败，不影响调用方

    for n in names:
        ext = os.path.splitext(n)[1].lower()
        if ext in IMG_EXTS:
            files.append(os.path.join(folder, n))

    files.sort()
    return files
