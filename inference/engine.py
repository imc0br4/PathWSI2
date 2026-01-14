# inference/engine.py
from __future__ import annotations

import os
import math
import time
from collections import defaultdict
from typing import Optional, Tuple, Callable, Dict, List

import numpy as np
import torch
from PIL import Image

from queue import Queue
from threading import Thread

from inference.patchify import SlideTiler
from models.manager import ModelRunner
from queue import Queue
from threading import Thread
ProgressFn = Optional[Callable[[int, int], None]]  # (done, total)


# ------------------ 公共工具 ------------------
def _choose_level_for_mag(reader, target_mag: Optional[float]) -> int:
    """
    根据目标倍率选择 pyramid level：
      - 若未给目标倍率或未能获取物镜倍率，返回 level 0
      - 否则选择下采样最接近 obj/target_mag 的层
    """
    if target_mag is None:
        return 0
    obj = reader.objective_power()
    if not obj or obj <= 0:
        return 0
    ds_list = [float(d) for d in reader.level_downsamples]
    target_ds = obj / float(target_mag)
    idx = int(np.argmin([abs(d - target_ds) for d in ds_list]))
    return idx


def _post_act_seg(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """分割输出后处理：二类→sigmoid，多类→softmax"""
    if num_classes <= 1 or logits.shape[1] == 1:
        return torch.sigmoid(logits)
    return torch.softmax(logits, dim=1)


# ------------------ 分割：segment_slide + save_mask ------------------
def segment_slide(
    reader,
    runner: ModelRunner,
    num_classes: int = 1,
    level: Optional[int] = None,
    target_mag: Optional[float] = None,
    patch_size: int = 224,
    overlap: int = 32,
    batch_size: int = 4,
    roi_level0: Optional[Tuple[int, int, int, int]] = None,
    progress: ProgressFn = None,
) -> Dict[str, np.ndarray]:
    """
    滑窗分割整张切片（或 ROI）：
    返回:
      {
        'prob': float32 [C,H,W],  # 在所选 level 分辨率下
        'mask': uint8  [H,W],     # 二类为 0/255，多类为类别索引
        'level': int
      }
    """
    if level is None:
        level = _choose_level_for_mag(reader, target_mag)

    tiler = SlideTiler(reader, level, patch=patch_size, overlap=overlap, roi_level0=roi_level0)
    H, W = tiler.hL, tiler.wL
    C = max(1, int(num_classes))

    acc = np.zeros((C, H, W), dtype=np.float32)
    wgt = np.zeros((H, W), dtype=np.float32)
    win = tiler.window  # 汉宁窗融合

    stride = tiler.stride
    total = ((H + stride - 1) // stride) * ((W + stride - 1) // stride)
    done = 0

    batch_tiles: List[torch.Tensor] = []
    batch_metas: List[Tuple[int, int, int, int]] = []

    @torch.no_grad()
    def flush():
        nonlocal batch_tiles, batch_metas, done
        if not batch_tiles:
            return
        x = torch.cat(batch_tiles, dim=0)  # [N,3,P,P]
        y = runner(x)                      # [N,C,P,P] or [N,1,P,P]
        y = _post_act_seg(y, C).numpy()    # 概率
        for i, (xL, yL, pw, ph) in enumerate(batch_metas):
            prob = y[i, :, 0:ph, 0:pw]
            acc[:, yL:yL + ph, xL:xL + pw] += prob * win[0:ph, 0:pw][None, ...]
            wgt[yL:yL + ph, xL:xL + pw] += win[0:ph, 0:pw]
            done += 1
            if progress:
                progress(done, total)
        batch_tiles.clear()
        batch_metas.clear()

    for yL in range(0, H, stride):
        ph = min(tiler.patch, H - yL)
        for xL in range(0, W, stride):
            pw = min(tiler.patch, W - xL)
            rgb = tiler._read_rgb(xL, yL, pw, ph)
            if ph != tiler.patch or pw != tiler.patch:
                canvas = np.zeros((tiler.patch, tiler.patch, 3), dtype=np.float32)
                canvas[0:ph, 0:pw, :] = rgb
                rgb = canvas
            t = tiler._to_tensor(rgb)
            batch_tiles.append(t)
            batch_metas.append((xL, yL, pw, ph))
            if len(batch_tiles) >= batch_size:
                flush()
    flush()

    w = np.maximum(wgt, 1e-6)[None, ...]
    prob = acc / w

    if C == 1:
        mask = (prob[0] >= 0.5).astype(np.uint8) * 255
    else:
        mask = np.argmax(prob, axis=0).astype(np.uint8)

    return {"prob": prob, "mask": mask, "level": int(level)}


def save_mask(mask: np.ndarray, out_path: str):
    """保存 uint8 掩膜到文件（自动创建父目录）。"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(mask).save(out_path)
    return out_path


# ------------------ 分类：prefetch + grid（可选 tissue filter） ------------------
def _keep_tissue_rgb01(
    rgb01: np.ndarray,
    white_thr: float = 0.92,
    std_thr: float = 0.02,
    gray_thr: float = 0.85,
    min_tissue_frac: float = 0.02,
) -> bool:
    """
    极轻量 tissue 判断（rgb01: float32, 0..1）
    - 极白且无纹理 -> 背景
    - 低灰度像素占比过低 -> 背景
    """
    m = float(rgb01.mean())
    s = float(rgb01.std())
    if (m > white_thr) and (s < std_thr):
        return False
    gray = rgb01[..., 0] * 0.299 + rgb01[..., 1] * 0.587 + rgb01[..., 2] * 0.114
    tissue_frac = float((gray < gray_thr).mean())
    return tissue_frac >= min_tissue_frac


@torch.inference_mode()
def classify_slide(
    reader,
    runner,
    classes,
    papillary_ids,
    threshold: float = 0.5,
    level: Optional[int] = None,
    target_mag: Optional[float] = None,
    patch_size: int = 224,
    overlap: int = 32,
    batch_size: int = 8,
    num_workers: int = 4,
    roi_level0: Optional[Tuple[int, int, int, int]] = None,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    progress: ProgressFn = None,
    color=(255, 0, 255),
    use_prob_alpha: bool = False,
    alpha_const: float = 0.4,
    use_tissue_filter: bool = False,   # 是否开启 tissue mask 过滤
    prof: bool = True,                 # ✅ 新增：是否打印耗时统计（不改变逻辑）
    prof_every: int = 0,               # ✅ 新增：>0 则每 N 个 batch 打印一次简要统计（0=不打印）
) -> Dict[str, np.ndarray]:
    """
    WSI 分类（滑窗取 patch -> 模型分类 -> 仅对 papillary_ids 上色）
    输出 overlay_rgba + meta。

    ✅ 本版本仅新增计时打印，不改变推理逻辑与数据流。
    """
    print("[CLS] using PREFETCH-GRID classify_slide")

    # ------------------ profiler（不改变逻辑，仅统计） ------------------
    TIM = defaultdict(float)
    CNT = defaultdict(int)

    def _tic(k: str):
        return (k, time.perf_counter())

    def _toc(tok):
        k, t0 = tok
        TIM[k] += time.perf_counter() - t0
        CNT[k] += 1

    def _report(final: bool = False):
        if not prof:
            return
        keys = sorted(TIM.keys())
        print("\n[CLS PROF] ---- time breakdown ----")
        # CPU 统计（秒）
        for k in keys:
            if k.endswith("_ms"):
                avg = TIM[k] / max(1, CNT[k])
                print(f"{k:26s}: {TIM[k]:.1f} ms (avg {avg:.2f} ms, n={CNT[k]})")
            else:
                avg = TIM[k] / max(1, CNT[k])
                print(f"{k:26s}: {TIM[k]:.3f} s (avg {avg*1000:.2f} ms, n={CNT[k]})")
        if final:
            print("[CLS PROF] ---- end ----\n")

    # ------------------ 原逻辑开始 ------------------
    if level is None:
        level = _choose_level_for_mag(reader, target_mag)

    tiler = SlideTiler(
        reader, level,
        patch=patch_size, overlap=overlap,
        roi_level0=roi_level0,
        mean=mean, std=std
    )

    H, W = tiler.hL, tiler.wL
    stride = max(1, int(getattr(tiler, "stride", patch_size - overlap)))

    gh = int(math.ceil(H / stride))
    gw = int(math.ceil(W / stride))
    grid = np.zeros((gh, gw, 4), dtype=np.uint8)

    r, g, b = int(color[0]), int(color[1]), int(color[2])
    alpha_const_u8 = int(max(0.0, min(1.0, float(alpha_const))) * 255)

    q: Queue = Queue(maxsize=max(8, int(batch_size) * 4))
    STOP = object()

    # def producer():
    #     try:
    #         kept = 0
    #         skipped = 0

    #         # ✅ 一次读图，同时得到 t 和 rgb_out（已经 pad）
    #         for xL, yL, pw, ph, t, rgb in tiler.tiles(return_rgb=True, return_tensor=True, pad=True):
    #             # 这里的计时是 producer 线程内部统计（用于判断读图/预处理压力）
    #             if prof:
    #                 CNT["producer_tiles"] += 1

    #             if use_tissue_filter:
    #                 tt = _tic("producer_tissue_check")
    #                 ok = _keep_tissue_rgb01(rgb)
    #                 _toc(tt)
    #                 if not ok:
    #                     skipped += 1
    #                     continue

    #             tt = _tic("producer_qput_wait")
    #             q.put((int(xL), int(yL), t))
    #             _toc(tt)

    #             kept += 1

    #         if use_tissue_filter:
    #             print(f"[CLS] tissue kept={kept}, skipped={skipped}, keep_ratio={kept/max(1, kept+skipped):.3f}")

    #         q.put(STOP)
    #     except Exception as e:
    #         q.put(("__ERR__", str(e)))
    #         q.put(STOP)

    q: Queue = Queue(maxsize=max(64, int(batch_size) * 16))  # ✅ 稍微加大缓冲
    STOP = object()

    coord_q: Queue = Queue(maxsize=max(64, int(batch_size) * 16))

    def coord_producer():
        # 只生成坐标，极轻
        for yL in range(0, H, stride):
            ph = min(patch_size, H - yL)
            for xL in range(0, W, stride):
                pw = min(patch_size, W - xL)
                coord_q.put((xL, yL, pw, ph))
        for _ in range(int(num_workers)):
            coord_q.put(STOP)

    def worker_producer():
        while True:
            it = coord_q.get()
            if it is STOP:
                q.put(STOP)  # 每个 worker 都往输出队列放一个 STOP
                break
            xL, yL, pw, ph = it

            rgb = tiler._read_rgb(xL, yL, pw, ph)
            if ph != tiler.patch or pw != tiler.patch:
                canvas = np.zeros((tiler.patch, tiler.patch, 3), dtype=np.float32)
                canvas[0:ph, 0:pw, :] = rgb
                rgb = canvas

            if use_tissue_filter and (not _keep_tissue_rgb01(rgb)):
                continue

            # ✅ pin_memory：后续 H2D non_blocking 才真正异步
            t = tiler._to_tensor(rgb).pin_memory()
            q.put((int(xL), int(yL), t))

    Thread(target=coord_producer, daemon=True).start()
    for _ in range(int(num_workers)):
        Thread(target=worker_producer, daemon=True).start()

        # Thread(target=producer, daemon=True).start()

    total = gh * gw
    done = 0

    batch_tiles: List[torch.Tensor] = []
    batch_xy: List[Tuple[int, int]] = []

    # 用于 batch 次数统计（不改变逻辑）
    flush_count = 0

    def _flush_batch():
        nonlocal done, batch_tiles, batch_xy, flush_count
        if not batch_tiles:
            return

        flush_count += 1

        tt = _tic("flush_cat")
        x = torch.cat(batch_tiles, dim=0)  # [B,3,P,P]
        batch_tiles.clear()
        _toc(tt)

        # GPU forward 计时：用 cuda event（不改变逻辑，只测）
        use_cuda = torch.cuda.is_available() and x.is_cuda
        if torch.cuda.is_available() and not x.is_cuda:
            # runner 可能内部会搬到 GPU；这里不改逻辑，仅作为提示统计
            CNT["note_x_on_cpu_batches"] += 1

        starter = ender = None
        if torch.cuda.is_available():
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()

        tt = _tic("flush_runner_call_cpu")
        # logits = runner(x)  # [B,C] or [B,C,1,1]
        # ✅ 让 logits 留在 GPU，避免每批次 .cpu() 同步
        logits = runner(x, return_cpu=False)

        # squeeze
        if logits.ndim == 4 and logits.shape[2] == 1 and logits.shape[3] == 1:
            logits = logits[:, :, 0, 0]

        # softmax / pap / hit 全在 GPU
        prob = torch.softmax(logits, dim=1)
        if papillary_ids:
            pap = prob[:, papillary_ids].amax(dim=1)
        else:
            pap = prob.max(dim=1).values
        hit = (pap >= float(threshold))

        # ✅ 只把 hit（很小）搬回 CPU
        hit_cpu = hit.to("cpu", non_blocking=True).numpy()

        if use_prob_alpha:
            a_cpu = (pap.clamp(0, 1) * 255).to(torch.uint8).to("cpu", non_blocking=True).numpy()
        else:
            a_cpu = None

        _toc(tt)

        if torch.cuda.is_available():
            ender.record()
            torch.cuda.synchronize()
            TIM["flush_runner_forward_ms"] += float(starter.elapsed_time(ender))
            CNT["flush_runner_forward_ms"] += 1

        # tt = _tic("flush_postprocess")
        # if logits.ndim == 4 and logits.shape[2] == 1 and logits.shape[3] == 1:
        #     logits = logits[:, :, 0, 0]
        # prob = torch.softmax(logits, dim=1)
        # _toc(tt)

        # tt = _tic("flush_select_pap")
        # if papillary_ids:
        #     pap = prob[:, papillary_ids].amax(dim=1)  # [B]
        # else:
        #     pap = prob.max(dim=1).values
        # hit = (pap >= float(threshold))
        # _toc(tt)

        # ✅ 这一步通常会触发 GPU->CPU 同步（不改变逻辑，仅计时）
        # tt = _tic("flush_to_cpu_numpy")
        # hit_cpu = hit.cpu().numpy()
        # if use_prob_alpha:
        #     a_cpu = (pap.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        # else:
        #     a_cpu = None
        # _toc(tt)

        tt = _tic("flush_write_grid")
        for i, (xL, yL) in enumerate(batch_xy):
            gx = xL // stride
            gy = yL // stride
            if 0 <= gx < gw and 0 <= gy < gh and hit_cpu[i]:
                grid[gy, gx, 0] = r
                grid[gy, gx, 1] = g
                grid[gy, gx, 2] = b
                grid[gy, gx, 3] = int(a_cpu[i]) if a_cpu is not None else alpha_const_u8
        _toc(tt)

        done += len(batch_xy)

        if progress:
            tt = _tic("flush_progress_cb")
            progress(min(done, total), total)
            _toc(tt)

        batch_xy.clear()

        # 每 N 个 batch 打印一次简报（可选）
        if prof and prof_every and (flush_count % int(prof_every) == 0):
            print(f"[CLS PROF] flush_count={flush_count}, done={done}/{total}")
            _report(final=False)
    stop_seen = 0
    while True:
        tt = _tic("main_qget_wait")
        it = q.get()
        _toc(tt)

        if it is STOP:
            stop_seen += 1
            if stop_seen >= int(num_workers):
                break
            else:
                continue
        if isinstance(it, tuple) and len(it) == 2 and it[0] == "__ERR__":
            raise RuntimeError(it[1])

        xL, yL, t = it
        batch_tiles.append(t)
        batch_xy.append((xL, yL))

        if len(batch_tiles) >= int(batch_size):
            _flush_batch()

    _flush_batch()

    # tt = _tic("overlay_repeat")
    # # overlay = np.repeat(np.repeat(grid, stride, axis=0), stride, axis=1)
    # # overlay = overlay[:H, :W, :]
    # # grid: (gh, gw, 4) uint8
    # overlay = np.array(
    #     Image.fromarray(grid, mode="RGBA").resize((W, H), resample=Image.NEAREST),
    #     dtype=np.uint8
    # )
    tt = _tic("overlay_repeat")
    overlay = np.repeat(np.repeat(grid, stride, axis=0), stride, axis=1)
    overlay = overlay[:H, :W, :]
    _toc(tt)


    try:
        ds0 = float(reader.level_downsamples[int(level)])
    except Exception:
        ds0 = 1.0

    try:
        x0_lv0 = int(getattr(tiler, "x0", 0))
        y0_lv0 = int(getattr(tiler, "y0", 0))
    except Exception:
        x0_lv0 = y0_lv0 = 0

    meta = {
        "level": int(level),
        "downsample": float(ds0),
        "threshold": float(threshold),
        "patch_size_level": int(patch_size),
        "stride_size_level": int(stride),

        # 你现在返回的是已经展开到 (H,W) 的 overlay，所以这里应该是 stride
        "grid_tile_px_on_overlay": int(stride),
        "grid_shape": [int(gh), int(gw)],  # 可选：仅作参考

        "roi_level0": roi_level0,
        "bbox_level0": [
            x0_lv0, y0_lv0,
            int(round(W * ds0)),
            int(round(H * ds0)),
        ],
        # （可选）保留原来的 xyxy 方便排查/兼容
        "bbox_level0_xyxy": [
            x0_lv0, y0_lv0,
            x0_lv0 + int(round(W * ds0)),
            y0_lv0 + int(round(H * ds0)),
        ],

        "use_tissue_filter": bool(use_tissue_filter),
    }


    # 最终汇总打印
    if prof:
        print(f"[CLS PROF] producer_tiles={CNT.get('producer_tiles', 0)}, flush_count={flush_count}, done={done}/{total}")
        _report(final=True)

    return {"overlay": overlay, "overlay_rgba": overlay, "meta": meta}
