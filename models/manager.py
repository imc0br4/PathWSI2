# models/manager.py
from __future__ import annotations
import os
from typing import Optional, Dict, Any

import torch
from contextlib import nullcontext
from utils.hwcheck import pick_device

# 仅保留你的分类网络
try:
    from models.zoo.cls_res import build_cls_res as _build_cls_res
except Exception:
    _build_cls_res = None


def _fallback_classifier(num_classes: int = 6):
    """兜底：极简分类器，防止你忘了放 cls_res 时代码也能跑（精度很低，仅排障用）。"""
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(True),
        nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(True),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(64, num_classes)
    )


def _build_arch(arch: Optional[str], num_classes: int):
    """根据 arch 构建网络结构（仅分类）"""
    name = (arch or "cls_res").lower()
    if name in ("cls_res", "resunet50_cls", "resunet50"):
        if _build_cls_res is not None:
            # 离线环境：不在这里加载预训练；你的 state_dict 会覆盖
            return _build_cls_res(num_classes=num_classes, pretrained_backbone=False)
        return _fallback_classifier(num_classes=num_classes)
    raise ValueError(f"Unsupported arch: {arch}. Only 'cls_res' is registered.")


class ModelRunner:
    """
    统一推理器：
      输入: float32 [N,3,H,W] 0..1（已按 mean/std 规范化）
      输出: float32 [N,C] 或 [N,C,h,w]
    """
    def __init__(self, model: torch.nn.Module, device: torch.device, amp: bool):
        self.model = model.eval().to(device)
        self.device = device
        self.amp = bool(amp)

    @torch.no_grad()
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.to(self.device, non_blocking=True)
        ctx = (
            torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True)
            if (self.amp and self.device.type == 'cuda')
            else nullcontext()
        )
        with ctx:
            out = self.model(batch)
        return out.float().cpu()


def load_model(
    model_path: str,
    arch: Optional[str] = "cls_res",
    num_classes: int = 6,
    prefer_gpu: bool = True,
    amp: Optional[bool] = None
) -> tuple[ModelRunner, Dict[str, Any]]:
    """
    返回 (runner, meta)
      - runner: 可调用，批量输出
      - meta:   {'arch','device','jit','num_classes','amp'}

    支持：
      1) TorchScript: torch.jit.load(model_path)
      2) 普通 state_dict: 需提供 arch（默认 'cls_res'）
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)

    device = pick_device(prefer_gpu=prefer_gpu)
    if amp is None:
        amp = (hasattr(device, "type") and device.type == 'cuda')

    # 先尝试 TorchScript
    jit_ok = False
    try:
        jit_model = torch.jit.load(model_path, map_location=device)
        runner = ModelRunner(jit_model, device, amp=amp)
        jit_ok = True
        arch_used = "torchscript"
    except Exception:
        # 再走 state_dict 路线
        net = _build_arch(arch, num_classes=num_classes)
        # models/manager.py 里 load_model 的普通 state_dict 分支：
        try:
            sd = torch.load(model_path, map_location='cpu', weights_only=True)  # torch>=2.4
        except TypeError:
            sd = torch.load(model_path, map_location='cpu')  # 老版本回退

        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        _ = net.load_state_dict(sd, strict=False)  # 允许非严格，便于不同 BN/头部命名
        runner = ModelRunner(net, device, amp=amp)
        arch_used = (arch or "cls_res")

    meta = {
        "arch": arch_used,
        "device": getattr(device, "type", str(device)),
        "jit": jit_ok,
        "num_classes": int(num_classes),
        "amp": bool(amp)
    }
    return runner, meta
