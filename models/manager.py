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
    """兜底：极简分类器（精度很低，仅排障用）。"""
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
            return _build_cls_res(num_classes=num_classes, pretrained_backbone=False)
        return _fallback_classifier(num_classes=num_classes)
    raise ValueError(f"Unsupported arch: {arch}. Only 'cls_res' is registered.")


class ModelRunner:
    def __init__(self, model: torch.nn.Module, device: torch.device, amp: bool):
        self.device = device
        self.amp = bool(amp)

        self.model = model.eval().to(device)

        # ✅ CUDA 专属：channels_last 往往能提升吞吐（不影响 CPU）
        if getattr(self.device, "type", "") == "cuda":
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
            except Exception:
                pass

    @torch.no_grad()
    def __call__(self, batch: torch.Tensor, return_cpu: bool = True) -> torch.Tensor:
        # CUDA 专属：确保输入也尽量 channels_last
        if getattr(self.device, "type", "") == "cuda":
            try:
                batch = batch.contiguous(memory_format=torch.channels_last)
            except Exception:
                pass

        batch = batch.to(self.device, non_blocking=True)

        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
            if (self.amp and getattr(self.device, "type", "") == "cuda")
            else nullcontext()
        )
        with ctx:
            out = self.model(batch)

        # 默认分割/其它路径需要 CPU；分类路径可 return_cpu=False 留在 GPU
        if return_cpu:
            return out.float().cpu()
        return out


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
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)

    device = pick_device(prefer_gpu=prefer_gpu)
    if amp is None:
        amp = (hasattr(device, "type") and device.type == "cuda")

    # ✅ CUDA 专属：全局加速开关（CPU 不受影响）
    if hasattr(device, "type") and device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # 先尝试 TorchScript
    jit_ok = False
    try:
        jit_model = torch.jit.load(model_path, map_location=device)
        # 可选：推理优化（不一定所有版本都有）
        try:
            jit_model = torch.jit.optimize_for_inference(jit_model)
        except Exception:
            pass

        runner = ModelRunner(jit_model, device, amp=amp)
        jit_ok = True
        arch_used = "torchscript"
    except Exception:
        net = _build_arch(arch, num_classes=num_classes)
        try:
            sd = torch.load(model_path, map_location="cpu", weights_only=True)  # torch>=2.4
        except TypeError:
            sd = torch.load(model_path, map_location="cpu")  # 老版本回退

        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        _ = net.load_state_dict(sd, strict=False)
        runner = ModelRunner(net, device, amp=amp)
        arch_used = (arch or "cls_res")

    meta = {
        "arch": arch_used,
        "device": getattr(device, "type", str(device)),
        "jit": jit_ok,
        "num_classes": int(num_classes),
        "amp": bool(amp),
    }
    return runner, meta
