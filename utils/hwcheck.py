# utils/hwcheck.py  —— 兼容增强版
import psutil

# torch 可能在“仅浏览器（不做分割）”的发行包里被剔除，这里要安全导入
try:
    import torch
except Exception:  # 没装 torch 也要保证浏览功能可运行
    torch = None


def pick_device(prefer_gpu=True):
    """保持你的原接口：返回 torch.device('cuda'|'cpu')。torch 缺失时返回字符串兼容。"""
    if torch is not None and prefer_gpu and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu') if torch is not None else 'cpu'


# 新增：语义更明确的选择（和上面等价，提供给新代码使用）
def pick_torch_device(prefer_gpu=True) -> str:
    if torch is not None and prefer_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


# 新增：根据设备选择 dtype（GPU 用 fp16，CPU 用 fp32）
def torch_dtype_for(device: str, prefer_fp16: bool = True):
    if torch is None:
        return None
    if device == "cuda" and prefer_fp16:
        return torch.float16
    return torch.float32


def env_meta():
    """保留你的原函数，增加些安全字段。"""
    meta = {
        'torch': getattr(torch, '__version__', None),
        'cuda_available': bool(torch and torch.cuda.is_available()),
        'gpu_name': None,
        'ram_gb': round(psutil.virtual_memory().total / (1024**3), 1),
    }
    if torch and torch.cuda.is_available():
        try:
            meta['gpu_name'] = torch.cuda.get_device_name(0)
            prop = torch.cuda.get_device_properties(0)
            meta['vram_gb'] = round(prop.total_memory / (1024**3), 1)
        except Exception:
            pass
    return meta
