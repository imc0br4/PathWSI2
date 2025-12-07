from __future__ import annotations

from math import ceil, floor, log2
from collections import deque
import time as _time
from typing import Optional, Tuple

import math
import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal, QTimer, QPointF, QSize
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap, QBrush, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsScene, QGraphicsView, QFrame, QWidget, QLabel
try:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    HAVE_GL = True
except Exception:  # pragma: no cover - OpenGL is optional
    HAVE_GL = False

from wsi.tiles import TileLoader, TileKey


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def select_level_smart(ds_list, ppd: float, oversample_limit: float = 1.4) -> int:
    if not ds_list or ppd <= 0:
        return 0
    ds = [float(d) for d in ds_list]
    cand = [(i, v) for i, v in enumerate(ds) if v <= ppd]
    if cand:
        return max(cand, key=lambda t: t[1])[0]
    i_min = 0
    if ppd / ds[i_min] <= oversample_limit:
        return i_min
    lppd = log2(ppd)
    return min(range(len(ds)), key=lambda i: abs(log2(ds[i]) - lppd))


def snap_ppd_to_near_ds(ppd_target: float, ds_list, snap_log2_eps: float = 0.08) -> float:
    if not ds_list or ppd_target <= 0:
        return ppd_target
    lppd = log2(ppd_target)
    ds = [float(d) for d in ds_list]
    i_near = min(range(len(ds)), key=lambda i: abs(log2(ds[i]) - lppd))
    if abs(log2(ds[i_near]) - lppd) <= snap_log2_eps:
        return ds[i_near]
    return ppd_target


# ----------------------------------------------------------------------
# Graphics item responsible for drawing tiles
# ----------------------------------------------------------------------
class TiledItem(QGraphicsItem):
    def __init__(self, view, reader, loader: TileLoader):
        super().__init__()
        self.view = view
        self.reader = reader
        self.loader = loader

        w0, h0 = reader.level_dimensions[0]
        self._rect = QRectF(0, 0, w0, h0)

        self._pix_cache: dict[TileKey, QPixmap] = {}
        self._img_cache: dict[TileKey, QImage] = {}

        self._blank_cover: set[TileKey] = set()  # 仅用于“封面/回退层”的空白瓦片标记
        self._blank_cfg = {
            "enabled": True,         # 开启空白抑制
            "gray_thresh": 248,      # 0~255，均值阈值（细胞学推荐 246~252）
            "std_thresh": 3.0,       # 像素标准差阈值（低纹理=空白）
            "sample_step": 8,        # 采样步长，越大越快
        }
        self._blank_placeholder_brush = QBrush(QColor(248, 248, 248))

    def _align_to_device(self, rect: QRectF) -> QRectF:
        try:
            tr = self.view.transform()  # 先到设备坐标
            tl = tr.map(rect.topLeft())
            br = tr.map(rect.bottomRight())
            # 关键：左上取 floor，右下取 ceil —— 只放大不缩小，避免两侧各缩 0.5px 形成缝
            dl = math.floor(tl.x())
            dt = math.floor(tl.y())
            dr = math.ceil(br.x())
            db = math.ceil(br.y())
            inv, ok = tr.inverted()
            if not ok:
                return rect
            tl_s = inv.map(QPointF(dl, dt))
            br_s = inv.map(QPointF(dr, db))
            return QRectF(tl_s, br_s)
        except Exception:
            return rect

    def boundingRect(self):
        return self._rect

    def clear_pix(self):
        self._pix_cache.clear()

    def last_level(self) -> int:
        ds_list = getattr(self.reader, "level_downsamples", [])
        return max(0, len(ds_list) - 1)

    def has_level_coverage_for_rect(self, level: int, rect: QRectF, min_ratio: float = 1.0) -> tuple[bool, float]:
        ds_list = [float(d) for d in getattr(self.reader, "level_downsamples", [])]
        if not ds_list:
            return False, 0.0
        lvl = max(0, min(level, len(ds_list) - 1))
        coords = self._iter_tile_coords(lvl, rect, margin=0, center_first=False, only_center_n=None)
        if not coords:
            return True, 1.0
        total = len(coords)
        hit = 0
        for tx, ty in coords:
            if self._pix_cache.get(TileKey(lvl, tx, ty)) is not None:
                hit += 1
        ratio = hit / max(1, total)
        return (ratio >= min_ratio), ratio

    def current_level(self) -> int:
        vr = self.view.viewport().rect()
        if vr.isEmpty():
            return 0
        vis = self.view.mapToScene(vr).boundingRect().intersected(self._rect)
        if vis.width() <= 0:
            return 0

        # 当前视口的像素比 (Pixels Per Display Pixel)
        ppd = vis.width() / max(1.0, vr.width())
        ds_list = [float(d) for d in self.reader.level_downsamples]
        if not ds_list:
            return 0

        # 1. 找出所有比当前 ppd 更清晰（或相等）的层级
        candidates = [(i, v) for i, v in enumerate(ds_list) if v <= ppd]

        # 2. 获取配置中的偏好比率（配合上面 yaml 改为 1.5 效果最佳）
        prefer_ratio_cfg = float(getattr(self.view, "_prefer_fine_ratio", 1.5))

        # [修改点]：这里不再动态计算 auto_ratio，或者将其设得很低
        # 强制在低倍率下更容易切换到粗层
        prefer_ratio = prefer_ratio_cfg

        if candidates:
            # 找到最接近当前 ppd 的那个粗层 (idx_coarse)
            # 例如 ppd=4.0 (即 10x倍率), Level0 ds=1, Level1 ds=4
            idx_coarse = max(candidates, key=lambda t: t[1])[0]

            # 如果配置允许稍微“超采样”（即用更清晰的层）
            if prefer_ratio > 1.0:
                idx_fine = min(candidates, key=lambda t: t[1])[0]
                ds_fine = ds_list[idx_fine]

                # [核心逻辑修改]
                # 计算如果使用精细层，需要缩放多少倍？
                # 例如 ppd=4, ds_fine=1 (Level0), 那么 down_scale = 4.0
                down_scale = ppd / max(1e-9, ds_fine)

                # 如果缩放倍数太大（超过 prefer_ratio），就放弃精细层，直接用粗层！
                # 这样在 5x-10x 倍率时，就会直接用 Level 1 (ds=4)，渲染数量瞬间减少 16 倍！
                if down_scale <= prefer_ratio + 0.05:
                    return idx_fine

            return idx_coarse

        # 兜底逻辑
        oversample_limit = 1.2
        idx = select_level_smart(ds_list, ppd, oversample_limit=oversample_limit)
        return max(0, min(idx, len(ds_list) - 1))

    def prefetch_level_for_rect(
        self,
        level: int,
        rect: QRectF,
        margin: int = 1,
        center_first: bool = True,
        only_center_n: int | None = None,
    ):
        coords = self._iter_tile_coords(
            level, rect, margin=margin, center_first=center_first, only_center_n=only_center_n
        )
        if not coords:
            return
        prio = self.loader.priority_for_level(level)
        for tx, ty in coords:
            # 走 View 的“带配额”请求入口；若没有则回退
            if hasattr(self.view, "_budgeted_request"):
                self.view._budgeted_request(level, tx, ty, prio)
            else:
                self.loader.request(level, tx, ty, priority=prio)


    def prefetch_coarsest_for_rect(self, rect: QRectF, margin: int = 2, only_center_n: int | None = None):
        ds_list = [float(d) for d in self.reader.level_downsamples]
        if not ds_list:
            return
        level = len(ds_list) - 1

        if only_center_n is None:
            default_n = getattr(self.view, "_coarse_prefetch_idle", None)
            if default_n is not None:
                try:
                    only_center_n = max(0, int(default_n))
                except Exception:
                    only_center_n = None

        try:
            use_margin_cfg = getattr(self.view, "_coarse_prefetch_margin", margin)
        except Exception:
            use_margin_cfg = margin
        use_margin = max(0, int(use_margin_cfg if use_margin_cfg is not None else margin))

        self.prefetch_level_for_rect(
            level, rect, margin=use_margin, center_first=True, only_center_n=only_center_n,
        )

    def _get_cached_tile(self, level: int, tx: int, ty: int):
        ds_list = [float(d) for d in getattr(self.reader, "level_downsamples", [])]
        if not ds_list:
            return None, level, tx, ty
        level = max(0, min(level, len(ds_list) - 1))
        key = TileKey(level, tx, ty)
        pix = self._pix_cache.get(key)
        if pix is not None:
            return pix, level, tx, ty

        tile_px = self.loader.tile_px
        tile_world = tile_px * ds_list[level]
        left = tx * self.loader.tile_px * ds_list[level]
        top = ty * self.loader.tile_px * ds_list[level]

        # 回退到更粗层找缓存，作为替代
        for lv in range(level + 1, len(ds_list)):
            tile_world_lv = tile_px * ds_list[lv]
            c_tx = int(math.floor(left / tile_world_lv))
            c_ty = int(math.floor(top / tile_world_lv))
            alt = self._pix_cache.get(TileKey(lv, c_tx, c_ty))
            if alt is not None:
                return alt, lv, c_tx, c_ty
        return None, level, tx, ty

    def _draw_backdrop_preview(self, painter: QPainter, visible_scene: QRectF):
        """用缩略图做‘底图后备层’，确保任何时候都不会露白。"""
        pix = getattr(self.view, "_cover_backdrop_pix", None)
        if pix is None or pix.isNull():
            return
        try:
            # 在 item 的全尺寸上铺底图（画布有裁剪，只会绘制可见部分）
            painter.drawPixmap(self._rect, pix, pix.rect())
        except Exception:
            pass

    def _draw_cover_with_level(self, painter: QPainter, level2: int, rect: QRectF, seam_eps: float):
        ds_list = [float(d) for d in self.reader.level_downsamples]
        if not ds_list:
            return
        last = len(ds_list) - 1
        level2 = max(0, min(level2, last))
        ds2 = ds_list[level2]
        tile_px = self.loader.tile_px
        tile_world2 = tile_px * ds2

        w0, h0 = self.reader.level_dimensions[0]
        r = rect.intersected(QRectF(0, 0, w0, h0)).toAlignedRect()
        if r.isEmpty() or tile_world2 <= 0:
            return

        tx0 = max(0, int(floor(r.left() / tile_world2)))
        ty0 = max(0, int(floor(r.top() / tile_world2)))
        tx1 = max(0, int(floor((r.right() - 1) / tile_world2)))
        ty1 = max(0, int(floor((r.bottom() - 1) / tile_world2)))

        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                pix, lvl_src, src_tx, src_ty = self._get_cached_tile(level2, tx, ty)
                tile_world_src = tile_px * ds_list[lvl_src]

                src_left = src_tx * tile_world_src
                src_top = src_ty * tile_world_src
                src_right = min((src_tx + 1) * tile_world_src, w0)
                src_bottom = min((src_ty + 1) * tile_world_src, h0)

                clip_left = max(rect.left(), src_left)
                clip_top = max(rect.top(), src_top)
                clip_right = min(rect.right(), src_right)
                clip_bottom = min(rect.bottom(), src_bottom)
                if clip_right <= clip_left or clip_bottom <= clip_top:
                    continue

                is_blank_cover = False
                blank_key = TileKey(lvl_src, src_tx, src_ty)
                try:
                    is_blank_cover = blank_key in self._blank_cover
                except Exception:
                    is_blank_cover = False

                if pix is None:
                    self.loader.request(level2, tx, ty, priority=self.loader.priority_for_level(level2))
                    if is_blank_cover:
                        painter.fillRect(QRectF(clip_left, clip_top, clip_right - clip_left, clip_bottom - clip_top),
                                         self._blank_placeholder_brush)
                    continue

                if is_blank_cover:
                    painter.fillRect(QRectF(clip_left, clip_top, clip_right - clip_left, clip_bottom - clip_top),
                                     self._blank_placeholder_brush)
                    continue

                clip_width = clip_right - clip_left
                clip_height = clip_bottom - clip_top

                src_x = (clip_left - src_left) / tile_world_src * tile_px
                src_y = (clip_top - src_top) / tile_world_src * tile_px
                src_w = clip_width / tile_world_src * tile_px
                src_h = clip_height / tile_world_src * tile_px
                src_rect = QRectF(src_x, src_y, src_w, src_h)

                dst_rect = QRectF(clip_left, clip_top, clip_width, clip_height)
                dst_rect.adjust(0.0, 0.0, seam_eps, seam_eps)

                moving = getattr(self.view, "_panning", False) or getattr(self.view, "_kinetic_active", False)
                if (not moving) and getattr(self.view, "_always_align_device", False):
                    dst_rect = self._align_to_device(dst_rect)

                painter.drawPixmap(dst_rect, pix, src_rect)

    def _draw_multilevel_cover(self, painter: QPainter, visible_scene: QRectF, level_hint: int, seam_eps: float):
        """多级封面覆盖：先画最粗层，再画 level_hint+2 / +1。
           —— 新增：若主层覆盖已达标（≥0.98），直接早停，不再绘制 cover。"""
        # --- 早停：主层覆盖充足就不画 cover ---
        try:
            if self._main_covered_visible(min_ratio=0.98):
                return
        except Exception:
            pass

        ds_list = [float(d) for d in self.reader.level_downsamples]
        if not ds_list:
            return
        last = len(ds_list) - 1
        self._draw_cover_with_level(painter, last, visible_scene, seam_eps)
        self._draw_cover_with_level(painter, min(level_hint + 2, last), visible_scene, seam_eps)
        self._draw_cover_with_level(painter, min(level_hint + 1, last), visible_scene, seam_eps)

    def paint(self, painter: QPainter, option, widget=None):
        vr = self.view.viewport().rect()
        if vr.isEmpty():
            return

        visible_scene = self.view.mapToScene(vr).boundingRect().intersected(self._rect)
        if visible_scene.isEmpty():
            return
        vis = visible_scene.toAlignedRect()

        ds_list = [float(d) for d in self.reader.level_downsamples]
        if not ds_list:
            return

        level = max(0, min(self.current_level(), len(ds_list) - 1))
        ds = ds_list[level]
        tile_px = self.loader.tile_px
        tile_world = tile_px * ds
        if tile_world <= 0:
            return

        ppd_now = self.view._current_ppd()
        down_ratio = ppd_now / ds if ds > 0 else 1.0
        is_zooming_in = getattr(self.view, "_is_zooming_in", lambda: False)()
        cover_active = getattr(self.view, "_cover_mode_active", lambda: False)()

        # 后备底图：先铺缩略图，保证不露白
        self._draw_backdrop_preview(painter, visible_scene)

        # 只有明显降采样时才开平滑
        allow_smooth = bool(getattr(self.view, "_allow_smooth", False))
        smooth_min = float(getattr(self.view, "_smooth_min_down_ratio", 2.6))
        painter.setRenderHint(
            QPainter.SmoothPixmapTransform,
            allow_smooth
            and (down_ratio >= smooth_min)
            and not is_zooming_in
            and not getattr(self.view, "_loading_active", False)
            and not getattr(self.view, "_panning", False)
            and not getattr(self.view, "_kinetic_active", False)
        )

        # 轻微吸附，抑制抖缝
        snap_near = (
            getattr(self.view, "_panning", False)
            or abs(log2(max(ppd_now, 1e-9)) - log2(max(ds, 1e-9))) <= getattr(self.view, "_snap_log2_eps", 0.22)
            or getattr(self.view, "_loading_active", False)
        )


        # 在轻度缩放区间静止也对齐（更稳的防马赛克）
        down_ratio_safe = (down_ratio if ds > 0 else 1.0)
        align_band_ok = (0.6 <= down_ratio_safe <= 1.8)

        moving = getattr(self.view, "_panning", False) or getattr(self.view, "_kinetic_active", False)

        should_align_device = (
                (not moving)  # <--- 关键：只有不移动时才做像素对齐
                and getattr(self.view, "_always_align_device", False)
                and align_band_ok
                and snap_near
        )

        w0, h0 = self.reader.level_dimensions[0]
        max_tx = max(0, int(ceil(w0 / tile_world)) - 1)
        max_ty = max(0, int(ceil(h0 / tile_world)) - 1)
        margin = 0 if is_zooming_in else 1

        tx0 = max(0, min(max_tx, floor(vis.left() / tile_world) - margin))
        ty0 = max(0, min(max_ty, floor(vis.top() / tile_world) - margin))
        tx1 = max(0, min(max_tx, floor((vis.right() - 1) / tile_world) + margin))
        ty1 = max(0, min(max_ty, floor((vis.bottom() - 1) / tile_world) + margin))

        painter.setPen(Qt.NoPen)

        seam_px = float(getattr(self.view, "_seam_pad_px", 1.0))
        seam_frac_base = float(getattr(self.view, "_seam_pad_frac", 0.06))
        seam_frac_local = max(seam_frac_base, 0.12) if getattr(self.view, "_loading_active", False) else seam_frac_base
        seam_cap = max(seam_px, ppd_now, 1.0)
        seam_eps = min(1.0, max(0.9, max(seam_px, min(tile_world * seam_frac_local, seam_cap))))

        # 以缩放焦点为中心排序
        focus = getattr(self.view, "_zoom_focus_scene", None) or visible_scene.center()
        tc = float(focus.x()) / tile_world
        rc = float(focus.y()) / tile_world

        coords = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]
        coords.sort(key=lambda p: (abs(p[0] - tc) + abs(p[1] - rc)))

        # —— 请求/转换/绘制预算
        tiles_needed = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
        budget_zoom_cfg = max(1, int(getattr(self.view, "_request_budget_zoom", 160)))
        budget_idle_cfg = max(1, int(getattr(self.view, "_request_budget_idle", 220)))
        request_budget_cfg = budget_zoom_cfg if is_zooming_in else budget_idle_cfg

        if getattr(self.view, "_panning", False) or getattr(self.view, "_kinetic_active", False):
            request_budget_eff = min(
                len(coords),
                max(32, int(request_budget_cfg * float(getattr(self.view, "_pan_req_factor", 0.55))))
            )
        else:
            request_budget_eff = min(len(coords), max(int(tiles_needed * 0.8), request_budget_cfg))

        request_count = 0  # ← 补上初始化

        convert_budget_cfg = max(1, int(getattr(self.view, "_pixmap_convert_budget", 28)))
        if getattr(self.view, "_panning", False) or getattr(self.view, "_kinetic_active", False):
            convert_budget_eff = int(getattr(self.view, "_pan_convert_budget", 0))
        else:
            convert_budget_eff = min(len(coords), max(64, convert_budget_cfg))
        convert_count = 0

        # 单帧绘制软上限
        if getattr(self.view, "_panning", False) or getattr(self.view, "_kinetic_active", False):
            paint_soft_cap = tiles_needed
        else:
            paint_soft_cap = 260
        if is_zooming_in:
            paint_soft_cap = 160

        tiles_drawn = 0
        fallbacks_drawn = 0
        fallbacks_soft_cap = 48 if (
            getattr(self.view, "_panning", False) or getattr(self.view, "_kinetic_active", False)
        ) else 24

        # 封面层（多级回退）：只要 cover_active 为 True，就先“即时复判主层覆盖”
        if getattr(self.view, "_loading_active", False) or cover_active:
            need_cover = True
            if cover_active:
                try:
                    ok_main, _ = self.has_level_coverage_for_rect(level, visible_scene, min_ratio=0.98)
                    if ok_main:
                        need_cover = False
                        # 触发下一帧重新判定 + 刷新，确保不必等待鼠标事件
                        try:
                            self.view._last_cover_check_t = 0.0
                        except Exception:
                            pass
                        try:
                            self.view.viewport().update()
                        except Exception:
                            pass
                except Exception:
                    need_cover = True

            if need_cover:
                old_hint = painter.testRenderHint(QPainter.SmoothPixmapTransform)
                painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
                self._draw_multilevel_cover(painter, visible_scene, level, seam_eps)
                painter.setRenderHint(QPainter.SmoothPixmapTransform, old_hint)

        # 主绘制循环
        for tx, ty in coords:
            if tiles_drawn >= paint_soft_cap:
                break

            # 先算几何
            left = tx * tile_world
            top = ty * tile_world
            right = min((tx + 1) * tile_world, w0)
            bottom = min((ty + 1) * tile_world, h0)

            if snap_near and ppd_now > 0:
                step = max(1.0, ppd_now)
                left = round(left / step) * step
                top = round(top / step) * step
                right = round(right / step) * step
                bottom = round(bottom / step) * step
                if right <= left:
                    right = left + step
                if bottom <= top:
                    bottom = top + step

            target_w = right - left
            target_h = bottom - top
            if target_w <= 0.0 or target_h <= 0.0:
                continue

            base_rect = QRectF(left, top, target_w, target_h)
            target_rect = QRectF(base_rect)

            # 准备 src_rect（像素坐标）
            src_rect = QRectF(
                0.0, 0.0,
                (base_rect.width() / tile_world) * tile_px,
                (base_rect.height() / tile_world) * tile_px
            )

            # 设备像素对齐（静止/拖动/配置要求时都启用）
            if should_align_device:
                target_rect = self._align_to_device(target_rect)
            # 对齐后再扩边，避免被对齐“收回”
            if tx < max_tx:
                target_rect.setRight(target_rect.right() + seam_eps)
            if ty < max_ty:
                target_rect.setBottom(target_rect.bottom() + seam_eps)

            key = TileKey(level, tx, ty)
            pix = self._pix_cache.get(key)
            if pix is not None:
                painter.drawPixmap(target_rect, pix, src_rect)
                tiles_drawn += 1
            else:
                qimg = self._img_cache.get(key)
                if qimg is not None:
                    painter.drawImage(target_rect, qimg, src_rect)
                    if convert_count < convert_budget_eff:
                        try:
                            if qimg.format() != QImage.Format_RGB32:
                                qimg = qimg.convertToFormat(QImage.Format_RGB32)
                            self._pix_cache[key] = QPixmap.fromImage(qimg)
                        except Exception:
                            pass
                        convert_count += 1
                    tiles_drawn += 1
                else:
                    drawn = False

                    # [修改 1]：移除 interactive 判断。
                    # 无论动静，只要主瓦片没出来，就应该画 fallback，防止露底。
                    # 原代码：
                    # interactive = ( ... )
                    # if interactive and fallbacks_drawn < fallbacks_soft_cap and not cover_active:

                    # 新代码：始终尝试回退绘制，但保持数量上限防止过载
                    if fallbacks_drawn < fallbacks_soft_cap and not cover_active:
                        drawn = self._draw_coarser_fallback(
                            painter, target_rect, left, top, right, bottom, level, snap_near=snap_near
                        )
                        if not drawn:
                            drawn = self._draw_any_fallback(
                                painter, target_rect, left, top, right, bottom, want_level=level, snap_near=snap_near
                            )

                    if drawn:
                        fallbacks_drawn += 1

                    if request_count < request_budget_eff:
                        self.loader.request(level, tx, ty, priority=self.loader.priority_for_level(level))
                        request_count += 1

        # -------------------------------------------------------
        # 有 fallback 被画出来时的补救策略
        #   - 拖动/惯性：只做轻量预取，避免卡顿
        #   - 静止：直接对当前层做一次“整块补齐”，把粗层彻底换掉
        # -------------------------------------------------------
        if fallbacks_drawn > 0:
            try:
                # 静止：提升紧急度，让 sweep / full-cover 更积极一点；
                # 交互中：走轻量预取，避免卡顿。
                moving = getattr(self.view, "_panning", False) or getattr(self.view, "_kinetic_active", False)
                urgency = 0 if moving else 1
                self.view._schedule_fallback_prefetch(urgency=urgency)
            except Exception:
                pass


    def on_tile_ready(self, generation: int, key: TileKey, qimg: QImage):
        if self.loader.is_stopping():
            return
        try:
            if qimg.format() != QImage.Format_RGB32:
                qimg = qimg.convertToFormat(QImage.Format_RGB32)
        except Exception:
            pass

        self.loader.put_cache(key, qimg)
        self._img_cache[key] = qimg
        try:
            self._pix_cache[key] = QPixmap.fromImage(qimg)
        except Exception:
            self._pix_cache.pop(key, None)

        ds = float(self.reader.level_downsamples[key.level])
        tile_world = self.loader.tile_px * ds
        left = key.tx * tile_world
        top = key.ty * tile_world
        self.update(QRectF(left, top, tile_world + 0.01, tile_world + 0.01))

        try:
            if hasattr(self.view, "_on_tile_ready_rect"):
                self.view._on_tile_ready_rect(QRectF(left, top, tile_world, tile_world))
        except Exception:
            pass

        # 现有代码：qimg 已经转成 RGB32，并放进缓存
        key_is_cover_level = (key.level >= len(self.reader.level_downsamples) - 2)  # 最粗两层作为封面层
        try:
            if key_is_cover_level and self._is_blank_for_cover(qimg):
                self._blank_cover.add(key)
            else:
                self._blank_cover.discard(key)
        except Exception:
            pass

        # ★ 新增：有新瓦片时，合并触发一次视口重绘（防止静止时四角一直是粗层/缩略图）
        try:
            view = self.view
            if view is not None and hasattr(view, "_tile_repaint_timer"):
                if not getattr(view, "_tile_repaint_pending", False):
                    view._tile_repaint_pending = True
                    view._tile_repaint_timer.start()
        except Exception:
            pass


    def _draw_coarser_fallback(
        self,
        painter: QPainter,
        target_rect: QRectF,
        left: float,
        top: float,
        right: float,
        bottom: float,
        want_level: int,
        snap_near: bool = False,
    ) -> bool:
        ds_list = [float(d) for d in self.reader.level_downsamples]
        if len(ds_list) <= 1:
            return False

        ppd_now = self.view._current_ppd()
        seam_px = float(getattr(self.view, "_seam_pad_px", 1.0))
        seam_frac = float(getattr(self.view, "_seam_pad_frac", 0.06))
        seam_cap = max(seam_px, ppd_now)

        ok_any = False
        for level2 in range(min(want_level + 1, len(ds_list) - 1), min(want_level + 3, len(ds_list))):
            ds2 = ds_list[level2]
            tile_px = self.loader.tile_px
            tile_world2 = tile_px * ds2

            # 与 paint() 保持一致：上限 1.0px，下限 0.9px（更稳地抑制细缝）
            seam_eps = min(1.0, max(0.9, max(seam_px, min(tile_world2 * seam_frac, seam_cap))))

            # 是否需要设备像素对齐（拖动/接近吸附/静止处于轻度缩放带宽时）
            down_ratio2 = ppd_now / max(ds2, 1e-6)
            align_band_ok = (0.6 <= down_ratio2 <= 1.8)
            moving = getattr(self.view, "_panning", False) or getattr(self.view, "_kinetic_active", False)
            should_align_device = (
                (not moving)
                and getattr(self.view, "_always_align_device", False)
                and align_band_ok
                and snap_near
            )


            tx2_0 = max(0, int(floor(left / tile_world2)))
            ty2_0 = max(0, int(floor(top / tile_world2)))
            tx2_1 = max(0, int(floor((right - 1) / tile_world2)))
            ty2_1 = max(0, int(floor((bottom - 1) / tile_world2)))

            for ty2 in range(ty2_0, ty2_1 + 1):
                for tx2 in range(tx2_0, tx2_1 + 1):
                    key2 = TileKey(level2, tx2, ty2)
                    pix2 = self._pix_cache.get(key2)
                    if pix2 is None:
                        continue

                    l2 = tx2 * tile_world2
                    t2 = ty2 * tile_world2
                    r2 = min((tx2 + 1) * tile_world2, self._rect.width())
                    b2 = min((ty2 + 1) * tile_world2, self._rect.height())

                    ol = max(left, l2)
                    ot = max(top, t2)
                    orr = min(right, r2)
                    ob = min(bottom, b2)
                    if orr <= ol or ob <= ot:
                        continue

                    src_x = (ol - l2) / tile_world2 * tile_px
                    src_y = (ot - t2) / tile_world2 * tile_px
                    src_w = (orr - ol) / tile_world2 * tile_px
                    src_h = (ob - ot) / tile_world2 * tile_px
                    src_rect = QRectF(src_x, src_y, src_w, src_h)

                    dst_rect = QRectF(
                        target_rect.left() + (ol - left),
                        target_rect.top() + (ot - top),
                        (orr - ol),
                        (ob - ot),
                    )

                    # 先对齐，再扩边
                    if should_align_device:
                        dst_rect = self._align_to_device(dst_rect)
                    dst_rect.adjust(0.0, 0.0, seam_eps, seam_eps)

                    # 空白封面填充
                    try:
                        if key2 in self._blank_cover:
                            painter.fillRect(dst_rect, self._blank_placeholder_brush)
                            ok_any = True
                            continue
                    except Exception:
                        pass

                    painter.drawPixmap(dst_rect, pix2, src_rect)
                    ok_any = True

        if ok_any:
            return True
        return False

    def _draw_any_fallback(
        self,
        painter: QPainter,
        target_rect: QRectF,
        left: float,
        top: float,
        right: float,
        bottom: float,
        want_level: int,
        snap_near: bool = False,
    ) -> bool:
        """
        通用回退：优先尝试当前层的缓存；若缺失，则借助 _get_cached_tile()
        自动回退到任意更粗层的已缓存瓦片进行局部填充。
        """
        ds_list = [float(d) for d in getattr(self.reader, "level_downsamples", [])]
        if not ds_list:
            return False

        tile_px = self.loader.tile_px
        ppd_now = self.view._current_ppd()
        seam_px = float(getattr(self.view, "_seam_pad_px", 1.0))
        seam_frac = float(getattr(self.view, "_seam_pad_frac", 0.06))
        ok_any = False

        # 将请求区域按 want_level 的 tile 网格划分，逐格尝试 fallback
        ds = ds_list[max(0, min(want_level, len(ds_list) - 1))]
        tile_world = tile_px * ds
        if tile_world <= 0:
            return False

        tx0 = max(0, int(math.floor(left / tile_world)))
        ty0 = max(0, int(math.floor(top / tile_world)))
        tx1 = max(0, int(math.floor((right - 1) / tile_world)))
        ty1 = max(0, int(math.floor((bottom - 1) / tile_world)))

        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                pix, lvl_src, src_tx, src_ty = self._get_cached_tile(want_level, tx, ty)
                if pix is None:
                    continue

                ds_src = ds_list[lvl_src]
                tile_world_src = tile_px * ds_src

                # seam 计算与对齐策略与 _draw_coarser_fallback 对齐
                seam_cap = max(seam_px, ppd_now)
                seam_eps = min(1.0, max(0.9, max(seam_px, min(tile_world_src * seam_frac, seam_cap))))
                down_ratio_src = ppd_now / max(ds_src, 1e-6)
                align_band_ok = (0.6 <= down_ratio_src <= 1.8)
                moving = getattr(self.view, "_panning", False) or getattr(self.view, "_kinetic_active", False)
                should_align_device = (
                    (not moving)
                    and getattr(self.view, "_always_align_device", False)
                    and align_band_ok
                    and snap_near
                )


                # 源瓦片在场景的边界
                l2 = src_tx * tile_world_src
                t2 = src_ty * tile_world_src
                r2 = min((src_tx + 1) * tile_world_src, self._rect.width())
                b2 = min((src_ty + 1) * tile_world_src, self._rect.height())

                # 与目标区域求交
                ol = max(left, l2)
                ot = max(top, t2)
                orr = min(right, r2)
                ob = min(bottom, b2)
                if orr <= ol or ob <= ot:
                    continue

                # 对应源图像矩形
                src_x = (ol - l2) / tile_world_src * tile_px
                src_y = (ot - t2) / tile_world_src * tile_px
                src_w = (orr - ol) / tile_world_src * tile_px
                src_h = (ob - ot) / tile_world_src * tile_px
                src_rect = QRectF(src_x, src_y, src_w, src_h)

                # 映射到目标矩形中的对应位置
                dst_rect = QRectF(
                    target_rect.left() + (ol - left),
                    target_rect.top() + (ot - top),
                    (orr - ol),
                    (ob - ot),
                )

                if should_align_device:
                    dst_rect = self._align_to_device(dst_rect)
                dst_rect.adjust(0.0, 0.0, seam_eps, seam_eps)

                # 空白封面填充
                try:
                    if TileKey(lvl_src, src_tx, src_ty) in self._blank_cover:
                        painter.fillRect(dst_rect, self._blank_placeholder_brush)
                        ok_any = True
                        continue
                except Exception:
                    pass

                painter.drawPixmap(dst_rect, pix, src_rect)
                ok_any = True

        return ok_any

    def _iter_tile_coords(
        self,
        level: int,
        rect: QRectF,
        margin: int = 1,
        center_first: bool = True,
        only_center_n: int | None = None,
    ):
        ds_list = [float(d) for d in self.reader.level_downsamples]
        if not ds_list:
            return []
        level = max(0, min(level, len(ds_list) - 1))
        ds = float(ds_list[level])
        tile_px = self.loader.tile_px
        tile_world = tile_px * ds

        density = max(1.0, 256.0 / max(1, int(self.loader.tile_px)))
        extra_margin = int(max(0, math.ceil(density) - 1))
        use_margin = margin + extra_margin

        w0, h0 = self.reader.level_dimensions[0]
        r = rect.intersected(QRectF(0, 0, w0, h0)).toAlignedRect()
        if r.isEmpty() or tile_world <= 0:
            return []

        max_tx = max(0, int(ceil(w0 / tile_world)) - 1)
        max_ty = max(0, int(ceil(h0 / tile_world)) - 1)

        tx0 = max(0, min(max_tx, int(floor(r.left() / tile_world)) - use_margin))
        ty0 = max(0, min(max_ty, int(floor(r.top() / tile_world)) - use_margin))
        tx1 = max(0, min(max_tx, int(floor(r.right() / tile_world)) + use_margin))
        ty1 = max(0, min(max_ty, int(floor(r.bottom() / tile_world)) + use_margin))

        coords = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]

        if center_first:
            cx = (r.left() + r.right()) * 0.5
            cy = (r.top() + r.bottom()) * 0.5
            tc = cx / tile_world
            rc = cy / tile_world
            coords.sort(key=lambda p: (abs(p[0] - tc) + abs(p[1] - rc)))

        if only_center_n is not None:
            cap = max(1, int(round(only_center_n * density)))
            coords = coords[:cap]

        return coords

    def prefetch_after_wheel(self, zoom_dir: int):
        if zoom_dir <= 0 or self.view is None or self.reader is None:
            return
        vr = self.view.viewport().rect()
        if vr.isEmpty():
            return
        vis_rect = self.view.mapToScene(vr).boundingRect().intersected(self._rect)
        if vis_rect.isEmpty():
            return

        ds_list = [float(d) for d in self.reader.level_downsamples]
        ppd = vis_rect.width() / max(1.0, vr.width())
        try:
            base_level = select_level_smart(ds_list, ppd, oversample_limit=2.0)
        except Exception:
            base_level = 0
        target = max(0, base_level - 1)

        tile_px = self.loader.tile_px
        ds = ds_list[target]
        tile_world = tile_px * ds

        focus = getattr(self.view, "_zoom_focus_scene", None) or vis_rect.center()
        tc = float(focus.x()) / tile_world
        rc = float(focus.y()) / tile_world

        w0, h0 = self.reader.level_dimensions[0]
        max_tx = max(0, int(math.ceil(w0 / tile_world)) - 1)
        max_ty = max(0, int(math.ceil(h0 / tile_world)) - 1)

        tx0 = max(0, min(max_tx, int(math.floor(vis_rect.left() / tile_world))))
        ty0 = max(0, min(max_ty, int(math.floor(vis_rect.top() / tile_world))))
        tx1 = max(0, min(max_tx, int(math.floor(vis_rect.right() / tile_world))))
        ty1 = max(0, min(max_ty, int(math.floor(vis_rect.bottom() / tile_world))))

        coords = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]
        coords.sort(key=lambda p: (abs(p[0] - tc) + abs(p[1] - rc)))

        total_tiles = len(coords)
        request_cap = max(1, int(getattr(self.view, "_request_budget_zoom", 160)))
        base_budget = min(total_tiles, request_cap)

        coarse_cap_attr = max(base_budget, int(getattr(self.view, "_coarse_prefetch_zoom", 180)))
        coarse_budget = min(total_tiles, coarse_cap_attr)

        neighbor_cap_attr = max(32, int(getattr(self.view, "_coarse_prefetch_idle", 140)))
        neighbor_budget = min(total_tiles, neighbor_cap_attr)

        coarse_margin = max(0, int(getattr(self.view, "_coarse_prefetch_margin", 3)))

        self.prefetch_coarsest_for_rect(
            vis_rect, margin=coarse_margin, only_center_n=coarse_budget,
        )

        last_idx = len(ds_list) - 1
        if base_level < last_idx:
            self.prefetch_level_for_rect(
                min(base_level + 1, last_idx),
                vis_rect,
                margin=1,
                center_first=True,
                only_center_n=neighbor_budget,
            )

        target_cap = max(24, request_cap)
        target_budget = min(total_tiles, target_cap)

        if target == base_level:
            combined_budget = max(base_budget, target_budget)
            self.prefetch_level_for_rect(
                base_level, vis_rect, margin=1, center_first=True, only_center_n=combined_budget,
            )
        else:
            self.prefetch_level_for_rect(
                base_level, vis_rect, margin=1, center_first=True, only_center_n=base_budget,
            )
            self.prefetch_level_for_rect(
                target, vis_rect, margin=1, center_first=True, only_center_n=target_budget,
            )
        if target > 0 and total_tiles:
            finer_budget = min(total_tiles, max(16, target_budget // 2))
            self.prefetch_level_for_rect(
                target - 1, vis_rect, margin=0, center_first=True, only_center_n=finer_budget,
            )

    def build_coarsest_warm_list(self, max_tiles: int = 4096):
        ds_list = [float(d) for d in self.reader.level_downsamples]
        if not ds_list:
            self._warm_level = None
            self._warm_coords = []
            self._warm_idx = 0
            return

        level = len(ds_list) - 1
        ds = ds_list[level]
        tile_px = self.loader.tile_px
        tile_world = tile_px * ds

        w0, h0 = self.reader.level_dimensions[0]
        tx_max = max(0, int(math.ceil(w0 / tile_world)) - 1)
        ty_max = max(0, int(math.ceil(h0 / tile_world)) - 1)

        cx = w0 * 0.5 / tile_world
        cy = h0 * 0.5 / tile_world

        coords = [(tx, ty) for ty in range(0, ty_max + 1) for tx in range(0, tx_max + 1)]
        coords.sort(key=lambda p: (abs(p[0] - cx) + abs(p[1] - cy)))

        if max_tiles and max_tiles > 0:
            coords = coords[:max_tiles]

        self._warm_level = level
        self._warm_coords = coords
        self._warm_idx = 0

    def warmup_step(self, n: int = 64) -> bool:
        if not getattr(self, "_warm_coords", None):
            return False
        level = getattr(self, "_warm_level", None)
        if level is None:
            return False
        end = min(self._warm_idx + max(1, int(n)), len(self._warm_coords))
        for i in range(self._warm_idx, end):
            tx, ty = self._warm_coords[i]
            try:
                self.loader.request(level, tx, ty, priority=self.loader.priority_for_level(level))
            except TypeError:
                self.loader.request(level, tx, ty, priority=self.loader.priority_for_level(level))
        self._warm_idx = end
        return self._warm_idx < len(self._warm_coords)

    def cancel_warmup(self):
        self._warm_coords = []
        self._warm_idx = 0
        self._warm_level = None

    def _request_coarser_for_rect(self, left: float, top: float, right: float, bottom: float, want_level: int):
        ds_list = [float(d) for d in self.reader.level_downsamples]
        if len(ds_list) <= 1:
            return
        for level2 in range(min(want_level + 1, len(ds_list) - 1), min(want_level + 3, len(ds_list))):
            ds2 = ds_list[level2]
            tile_px = self.loader.tile_px
            tile_world2 = tile_px * ds2

            tx2_0 = max(0, int(math.floor(left / tile_world2)))
            ty2_0 = max(0, int(math.floor(top / tile_world2)))
            tx2_1 = max(0, int(math.floor((right - 1) / tile_world2)))
            ty2_1 = max(0, int(math.floor((bottom - 1) / tile_world2)))

            for ty2 in range(ty2_0, ty2_1 + 1):
                for tx2 in range(tx2_0, tx2_1 + 1):
                    self.loader.request(level2, tx2, ty2, priority=self.loader.priority_for_level(level2))

    def _is_blank_for_cover(self, qimg: QImage) -> bool:
        cfg = self._blank_cfg
        if not cfg["enabled"] or qimg is None or qimg.isNull():
            return False

        w, h = qimg.width(), qimg.height()
        if w == 0 or h == 0:
            return True

        step = int(cfg.get("sample_step", 8)) or 8
        fmt = qimg.format()
        # 统一到 RGBA32
        if fmt != QImage.Format_ARGB32 and fmt != QImage.Format_RGBA8888 and fmt != QImage.Format_RGB32:
            qimg = qimg.convertToFormat(QImage.Format_RGB32)

        ptr = qimg.bits()
        ptr.setsize(qimg.sizeInBytes())
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, qimg.bytesPerLine() // 4, 4)  # BGRA

        # 采样 + 转灰度（简单平均足够）
        s = arr[::step, ::step, :3].astype(np.float32)
        gray = (s[..., 0] + s[..., 1] + s[..., 2]) / 3.0
        mean = float(gray.mean())
        std = float(gray.std())
        return (mean >= float(cfg["gray_thresh"])) and (std <= float(cfg["std_thresh"]))


# ----------------------------------------------------------------------
# View implementation
# ----------------------------------------------------------------------
class WsiView(QGraphicsView):
    metricsChanged = Signal(object, float, int, float)

    def __init__(self, app_cfg):
        super().__init__()

        if bool(app_cfg["viewer"].get("use_opengl", False)) and HAVE_GL:
            self.setViewport(QOpenGLWidget())

        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        if app_cfg["viewer"].get("crisp", True):
            self.setRenderHint(QPainter.SmoothPixmapTransform, False)
            self._allow_smooth = False
        else:
            self._allow_smooth = True  # 明显降采样时才会真正启用

        CacheModeFlag = getattr(QGraphicsView, "CacheModeFlag", QGraphicsView)
        ViewportUpdateMode = getattr(QGraphicsView, "ViewportUpdateMode", QGraphicsView)
        OptimizationFlag = getattr(QGraphicsView, "OptimizationFlag", QGraphicsView)

        self.setCacheMode(getattr(CacheModeFlag, "CacheNone", 0))
        self.setViewportUpdateMode(
            getattr(ViewportUpdateMode, "SmartViewportUpdate", getattr(ViewportUpdateMode, "MinimalViewportUpdate", 0))
        )
        for name in ("DontSavePainterState", "DontAdjustForAntialiasing", "DontClipPainter"):
            flag = getattr(OptimizationFlag, name, None)
            if flag is not None:
                try:
                    self.setOptimizationFlag(flag, True)
                except Exception:
                    pass

        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        self.app_cfg = app_cfg
        viewer_cfg = self.app_cfg.get("viewer", {})

        self._snap_log2_eps = float(viewer_cfg.get("snap_log2_eps", 0.22))
        self._always_align_device = bool(viewer_cfg.get("always_align_device", True))

        self._pan_req_factor = float(viewer_cfg.get("pan_request_factor", 0.55))  # 拖动时请求配额比例
        self._pan_convert_budget = int(viewer_cfg.get("pan_convert_budget", 0))  # 拖动时每帧允许的 QImage→QPixmap 转换数
        self._panning_decay_ms = int(viewer_cfg.get("panning_decay_ms", 180))  # ← 先存起来

        self._zoom_factor = float(viewer_cfg.get("wheel_zoom_factor", 1.25))
        self.min_ppd = float(viewer_cfg.get("min_mpp_scale", 0.02))
        self.max_ppd = float(viewer_cfg.get("max_mpp_scale", 32.0))

        # ---- 帧级请求配额（限流）----
        self._frame_request_budget = int(viewer_cfg.get("frame_request_budget", 96))  # 可在 app.yaml 配置
        self._frame_req_budget_left = self._frame_request_budget
        self._frame_req_defer_q = deque()   # (level, tx, ty, priority)
        self._frame_req_posted = False


        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        # ★ 背景略带蓝色，让画布像“片子工作区域”
        self.setBackgroundBrush(QColor(242, 248, 252))


        self.reader = None
        self.loader = None
        self.item = None
        self._home_ppd = None
        self._allow_doubleclick_home = True

        self._last_prefetch_t = 0.0
        self._zoom_locked = False

        self._cover_backdrop_pix: Optional[QPixmap] = None  # 新增：底图后备层
        self._freeze_pix = None
        self._last_scale_t = 0.0
        self._last_scale_dir = 0
        self._zoom_focus_scene = None

        self._prev_update_mode = self.viewportUpdateMode()

        self._pixmap_convert_budget = int(viewer_cfg.get("pixmap_convert_budget", 28))
        self._prefer_fine_ratio = float(viewer_cfg.get("prefer_fine_ratio", 2.8))
        self._seam_pad_px = float(viewer_cfg.get("seam_pad_px", 1.0))
        self._seam_pad_frac = float(viewer_cfg.get("seam_pad_frac", 0.06))
        self._request_budget_zoom = int(viewer_cfg.get("request_budget_zoom", 160))
        self._request_budget_idle = int(viewer_cfg.get("request_budget_idle", 220))

        self._coarse_prefetch_margin = max(0, int(viewer_cfg.get("prefetch_coarse_margin", 3)))
        self._coarse_prefetch_zoom = max(32, int(viewer_cfg.get("prefetch_coarse_zoom", 180)))
        self._coarse_prefetch_pan = max(32, int(viewer_cfg.get("prefetch_coarse_pan", 260)))
        self._coarse_prefetch_idle = max(32, int(viewer_cfg.get("prefetch_coarse_idle", 140)))

        self._coarse_cover_boost_high = max(32, int(viewer_cfg.get("coarse_cover_boost_high", 220)))
        self._coarse_cover_boost_low = max(32, int(viewer_cfg.get("coarse_cover_boost_low", 160)))

        self._freeze_capture_enabled = bool(viewer_cfg.get("freeze_capture_enabled", True))
        self._freeze_capture_min_factor = max(1.0, float(viewer_cfg.get("freeze_capture_min_factor", 1.06)))
        self._freeze_capture_interval = max(0.0, float(viewer_cfg.get("freeze_capture_interval_ms", 140)) / 1000.0)
        self._freeze_last_capture_t = 0.0

        self._loading_active = False
        self._loading_progress = 0.0
        self._loading_text = ""

        self._startup_pending_tiles: set[tuple[int, int, int]] = set()
        self._startup_required_tiles: set[tuple[int, int, int]] = set()
        self._startup_required_total = 0
        self._startup_optional_total = 0
        self._startup_total_tiles = 0

        self._interaction_locked = False
        self._locked_drag_mode = None

        self._loading_soft_ms = int(viewer_cfg.get("loading_soft_unlock_ms", 500))
        self._loading_hard_ms = int(viewer_cfg.get("loading_hard_unlock_ms", 5000))
        self._loading_min_ms = int(viewer_cfg.get("loading_min_duration_ms", 1500))

        self._loading_soft_timer = QTimer(self)
        self._loading_soft_timer.setSingleShot(True)
        self._loading_soft_timer.timeout.connect(self._on_loading_soft_unlock)

        self._loading_hard_timer = QTimer(self)
        self._loading_hard_timer.setSingleShot(True)
        self._loading_hard_timer.timeout.connect(self._on_loading_hard_unlock)

        self._loading_min_timer = QTimer(self)
        self._loading_min_timer.setSingleShot(True)
        self._loading_min_timer.timeout.connect(self._on_loading_min_elapsed)
        self._loading_min_elapsed = False
        self._loading_finish_pending = False

        self._cover_hold_ms = int(viewer_cfg.get("cover_hold_ms", 160))
        self._freeze_hold_ms = int(viewer_cfg.get("freeze_hold_ms", 160))
        self._cover_until_t = 0.0
        self._freeze_deadline_t = 0.0
        self._last_cover_check_t = 0.0
        self._coarse_covered_cached = False
        self._last_main_cover_check_t = 0.0
        self._main_covered_cached = False

        self._zoom_timer = QTimer(self)
        self._zoom_timer.setInterval(30)
        self._zoom_timer.setSingleShot(True)
        self._zoom_timer.timeout.connect(self._apply_deferred_zoom)

        self._pan_prefetch_timer = QTimer(self)
        self._pan_prefetch_timer.setInterval(35)
        self._pan_prefetch_timer.setSingleShot(True)
        self._pan_prefetch_timer.timeout.connect(self._do_pan_prefetch)

        self._warm_timer = QTimer(self)
        self._warm_timer.setInterval(50)
        self._warm_timer.timeout.connect(self._on_warm_tick)

        self._enter_anim_timer = QTimer(self)
        self._enter_anim_timer.setInterval(16)
        self._enter_anim_timer.timeout.connect(self._on_enter_anim_tick)
        self._enter_anim_t = 0.0
        self._enter_anim_deadline = 0.0

        self._complete_on_required = True
        self._loading_started_at = 0.0
        self._last_progress_at = 0.0

        self._loading_watchdog_timer = QTimer(self)
        self._loading_watchdog_timer.setInterval(int(viewer_cfg.get("loading_watchdog_ms", 1200)))
        self._loading_watchdog_timer.timeout.connect(self._on_loading_watchdog_tick)

        self._warm_batch = int(viewer_cfg.get("coarse_warm_batch", viewer_cfg.get("coarse_warmup_step", 48)))

        self._kinetic_enabled = bool(viewer_cfg.get("kinetic_enabled", True))
        self._kinetic_decay = float(viewer_cfg.get("kinetic_decay", 3.0))
        self._kinetic_max_speed = float(viewer_cfg.get("kinetic_max_speed", 6000.0))
        self._kinetic_min_speed = float(viewer_cfg.get("kinetic_min_speed", 25.0))
        self._drag_samples = deque(maxlen=8)
        self._kinetic_timer = QTimer(self)
        self._kinetic_timer.setInterval(16)
        self._kinetic_timer.timeout.connect(self._on_kinetic_tick)
        self._kinetic_vx = 0.0
        self._kinetic_vy = 0.0
        self._kinetic_last_t = 0.0
        self._kinetic_active = False

        self._pan_prefetch_pending = False
        self._last_fallback_prefetch_t = 0.0

        # —— 缩放/平移结束后的“补齐 sweep”，确保整个可视区域被主层瓦片覆盖
        self._post_zoom_sweep_timer = QTimer(self)
        self._post_zoom_sweep_timer.setInterval(110)
        self._post_zoom_sweep_timer.timeout.connect(self._post_zoom_sweep_tick)
        self._post_zoom_sweep_deadline = 0.0

        # 新增：加载期间的“网格吸附”开关
        self._grid_snap_enabled = bool(self.app_cfg.get("viewer", {}).get("grid_snap_enabled", True))

        self._panning = False
        self._panning_decay_timer = QTimer(self)
        self._panning_decay_timer.setSingleShot(True)
        self._panning_decay_timer.setInterval(getattr(self, "_panning_decay_ms", 140))
        self._panning_decay_timer.timeout.connect(lambda: setattr(self, "_panning", False))

        self._smooth_min_down_ratio = float(viewer_cfg.get("smooth_min_down_ratio", 2.6))
        self._always_align_device = bool(viewer_cfg.get("always_align_device", True))

        self._tile_repaint_timer = QTimer(self)
        self._tile_repaint_timer.setInterval(30)   # ~33 FPS 上限
        self._tile_repaint_timer.setSingleShot(True)
        self._tile_repaint_timer.timeout.connect(self._on_tile_repaint_tick)
        self._tile_repaint_pending = False

    # -------------------------------------------------------------
    # 快速生成后备底图（避免露白）：优先用 reader 的缩略图接口，其次降级读取最粗层
    # -------------------------------------------------------------
    def _make_backdrop_pix(self, reader, max_side: int = 2048) -> Optional[QPixmap]:
        def _to_qimage(data) -> Optional[QImage]:
            try:
                from PIL import Image as _PILImage
            except Exception:
                _PILImage = None

            if isinstance(data, QImage):
                return data
            if _PILImage is not None and isinstance(data, _PILImage.Image):
                im = data.convert("RGBA")
                w, h = im.size
                buf = im.tobytes("raw", "RGBA")
                q = QImage(buf, w, h, QImage.Format_RGBA8888)
                return q.copy()
            if isinstance(data, np.ndarray):
                arr = data
                if arr.dtype != np.uint8:
                    arr = arr.astype(np.uint8, copy=False)
                if arr.ndim == 2:
                    h, w = arr.shape
                    q = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
                    return q.copy()
                if arr.ndim == 3 and arr.shape[2] in (3, 4):
                    h, w, c = arr.shape
                    if c == 3:
                        q = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
                        return q.copy()
                    else:
                        q = QImage(arr.data, w, h, w * 4, QImage.Format_RGBA8888)
                        return q.copy()
            return None

        def _to_qpix(img) -> Optional[QPixmap]:
            if isinstance(img, QPixmap):
                return img
            if isinstance(img, QImage):
                return QPixmap.fromImage(img)
            qimg = _to_qimage(img)
            if isinstance(qimg, QImage):
                return QPixmap.fromImage(qimg)
            return None

        def _normalize_pix(pix: Optional[QPixmap]) -> Optional[QPixmap]:
            if not pix or pix.isNull():
                return None
            side = max_side
            if max(pix.width(), pix.height()) > side:
                return pix.scaled(side, side, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            return pix

        methods = (
            "get_thumbnail_qimage",
            "thumbnail_qimage",
            "thumbnail_qt",
            "get_thumbnail",
            "thumbnail",
        )
        kwargs_list = (
            {"max_side": max_side},
            {"size": max_side},
            {"size": (max_side, max_side)},
            {"max_dim": max_side},
        )
        for name in methods:
            fn = getattr(reader, name, None)
            if callable(fn):
                for kwargs in kwargs_list:
                    try:
                        img = fn(**kwargs)
                    except TypeError:
                        continue
                    except Exception:
                        break
                    else:
                        pix = _to_qpix(img)
                        norm = _normalize_pix(pix)
                        if norm is not None:
                            return norm

        try:
            dims = list(getattr(reader, "level_dimensions", []))
            if not dims:
                return None
            target_level = len(dims) - 1
            for idx, (w_l, h_l) in enumerate(dims):
                if max(int(w_l), int(h_l)) <= max_side:
                    target_level = idx
                    break
            w_sel, h_sel = dims[target_level]
            w_sel = max(1, int(w_sel))
            h_sel = max(1, int(h_sel))
            if hasattr(reader, "read_region"):
                img = reader.read_region(level=target_level, x=0, y=0, w=w_sel, h=h_sel)
            elif hasattr(reader, "get_region"):
                img = reader.get_region(level=target_level, x=0, y=0, w=w_sel, h=h_sel)
            else:
                img = None
            norm = _normalize_pix(_to_qpix(img))
            if norm is not None:
                return norm
        except Exception:
            pass
        return None

    def _is_zooming_in(self) -> bool:
        return (_time.perf_counter() - self._last_scale_t) < 0.25 and self._last_scale_dir > 0

    def _coarse_covered_visible(self, min_ratio: float = 0.98) -> bool:
        """改进版：同时参考‘当前层’与‘最粗层’覆盖。
        只要当前主层覆盖≥min_ratio，就认为覆盖达标（即使最粗层有洞，也不再点亮 cover）。"""
        now = _time.perf_counter()
        if (now - self._last_cover_check_t) < 0.06:
            return self._coarse_covered_cached

        if not (self.item and self.reader):
            self._coarse_covered_cached = False
            self._last_cover_check_t = now
            return False

        vr = self.viewport().rect()
        if vr.isEmpty():
            self._coarse_covered_cached = True
            self._last_cover_check_t = now
            return True
        vis = self.mapToScene(vr).boundingRect().intersected(self.item.boundingRect())
        if vis.isEmpty():
            self._coarse_covered_cached = True
            self._last_cover_check_t = now
            return True

        # 先看当前层
        lvl_cur = self.item.current_level()
        ok_cur, ratio_cur = self.item.has_level_coverage_for_rect(lvl_cur, vis, min_ratio=min_ratio)
        if ok_cur:
            self._coarse_covered_cached = True
            self._last_cover_check_t = now
            return True

        # 再看最粗层（用于加载阶段/回退层判定）
        last = self.item.last_level()
        ok_last, ratio_last = self.item.has_level_coverage_for_rect(last, vis, min_ratio=min_ratio)
        self._coarse_covered_cached = ok_last
        self._last_cover_check_t = now

        if not ok_last:
            try:
                boost_hi = max(32, int(getattr(self, "_coarse_cover_boost_high", 220)))
                boost_lo = max(32, int(getattr(self, "_coarse_cover_boost_low", 160)))
                base_ratio = max(ratio_cur, ratio_last)
                budget = boost_hi if base_ratio < 0.6 else boost_lo
                margin = max(0, int(getattr(self, "_coarse_prefetch_margin", 3)))
                self.item.prefetch_coarsest_for_rect(
                    vis, margin=margin, only_center_n=budget,
                )
            except Exception:
                pass
        return self._coarse_covered_cached

    def _main_covered_visible(self, min_ratio: float = 0.98) -> bool:
        """检测“当前主层”对可视区域的覆盖是否足够（带轻缓存）。
           与 coarse 覆盖不同，这里直接看当前层，从而用来早停 cover。"""
        now = _time.perf_counter()
        if (now - getattr(self, "_last_main_cover_check_t", 0.0)) < 0.01:
            return bool(getattr(self, "_main_covered_cached", False))

        if not (self.item and self.reader):
            self._main_covered_cached = False
            self._last_main_cover_check_t = now
            return False

        vr = self.viewport().rect()
        if vr.isEmpty():
            self._main_covered_cached = True
            self._last_main_cover_check_t = now
            return True

        vis = self.mapToScene(vr).boundingRect().intersected(self.item.boundingRect())
        if vis.isEmpty():
            self._main_covered_cached = True
            self._last_main_cover_check_t = now
            return True

        lvl = self.item.current_level()
        ok, _ratio = self.item.has_level_coverage_for_rect(lvl, vis, min_ratio=min_ratio)

        self._main_covered_cached = bool(ok)
        self._last_main_cover_check_t = now
        return bool(ok)

    def _cover_mode_active(self) -> bool:
        # 加载期或缩放瞬时：始终启用 cover
        if getattr(self, "_loading_active", False):
            return True
        if self._is_zooming_in():
            return True

        now = _time.perf_counter()
        if now < getattr(self, "_cover_until_t", 0.0):
            return True

        # —— 关键：先看“主层覆盖”是否已经足够；足够则直接关闭 cover（早停）
        # 同时通过该函数刷新主层覆盖缓存，保证 cover 状态与主层瓦片同步。
        try:
            if self._main_covered_visible(min_ratio=0.98):
                # 立即撤掉 cover，并触发下一帧重新判定与刷新
                try:
                    self._last_cover_check_t = 0.0
                except Exception:
                    pass
                try:
                    self.viewport().update()
                except Exception:
                    pass
                return False
        except Exception:
            pass

        # 若主层尚不足，再看最粗层兜底是否也不足；不足则继续维持 cover
        if not self._coarse_covered_visible(min_ratio=0.98):
            return True

        # 其它情况：关闭 cover
        return False

    def _on_warm_tick(self):
        if (_time.perf_counter() - self._last_scale_t) < 0.4:
            return
        if not self.item:
            self._warm_timer.stop()
            return
        try:
            cont = self.item.warmup_step(self._warm_batch)
        except Exception:
            cont = False
        if not cont:
            self._warm_timer.stop()

    def set_allow_doubleclick_home(self, allowed: bool):
        self._allow_doubleclick_home = bool(allowed)

    def _current_ppd(self) -> float:
        vr = self.viewport().rect()
        if vr.isEmpty():
            return 1.0
        sr = self.mapToScene(vr).boundingRect()
        if sr.width() <= 0:
            return 1.0
        return sr.width() / max(1.0, vr.width())

    def _extract_wheel_delta(self, event) -> float:
        """Normalize wheel delta from angle/pixel based sources."""
        delta_units = 0.0
        if hasattr(event, "angleDelta"):
            try:
                ang = event.angleDelta()
            except Exception:
                ang = None
            if ang is not None:
                try:
                    if hasattr(ang, "y") and ang.y():
                        delta_units = float(ang.y())
                    elif hasattr(ang, "x") and ang.x():
                        delta_units = float(ang.x())
                except Exception:
                    delta_units = 0.0
        if delta_units == 0.0 and hasattr(event, "pixelDelta"):
            try:
                pix = event.pixelDelta()
            except Exception:
                pix = None
            if pix is not None:
                try:
                    if hasattr(pix, "y") and pix.y():
                        delta_units = float(pix.y()) * 0.6
                    elif hasattr(pix, "x") and pix.x():
                        delta_units = float(pix.x()) * 0.6
                except Exception:
                    delta_units = 0.0
        if delta_units == 0.0 and hasattr(event, "delta"):
            try:
                delta_units = float(event.delta())
            except Exception:
                delta_units = 0.0
        return delta_units

    def _scale_to_ppd(self, target_ppd: float, *, bump: bool = True):
        now = self._current_ppd()
        if target_ppd <= 0:
            return
        ratio = max(1e-6, now / target_ppd)
        super().scale(ratio, ratio)
        if self.loader and bump:
            self.loader.bump_generation()
        self._emit_metrics()

    def _apply_scale_clamped(self, f: float):
        if getattr(self, "_loading_active", False):
            return
        if not self.reader:
            return
        ppd_now = self._current_ppd()
        target = ppd_now / f
        lo, hi = self.min_ppd, self.max_ppd
        if self._home_ppd is not None:
            hi = max(hi, self._home_ppd * 1.2)
        target = max(lo, min(hi, target))
        target = snap_ppd_to_near_ds(target, getattr(self.reader, "level_downsamples", []), 0.08)
        self._scale_to_ppd(target)

    def zoom_to_fit(self):
        if getattr(self, "_loading_active", False):
            return
        if self._home_ppd is None:
            return
        self._scale_to_ppd(self._home_ppd)

    def _clamp_viewport_to_scene(self):
        """可选：把视口限制在场景范围内（供迷你地图定位后调用）"""
        if not self.item:
            return
        r = self.item.boundingRect()
        self.centerOn(
            max(r.left(), min(r.right(), self.mapToScene(self.viewport().rect().center()).x())),
            max(r.top(), min(r.bottom(), self.mapToScene(self.viewport().rect().center()).y()))
        )

    # -------------------------------------------------------------
    # 新增：把缩放/平移吸附到 tile 网格
    # -------------------------------------------------------------
    def _snap_view_to_grid(
        self,
        base_level: int | None = None,
        *,
        allow_scale_snap: bool = True,
        allow_translate_snap: bool = True,
        scale_eps_log2: float = 0.02,
        translate_eps: float = 0.5,
    ):
        if not self._grid_snap_enabled:
            return
        if not (self.item and self.reader and self.loader):
            return

        ds_list = [float(d) for d in getattr(self.reader, "level_downsamples", [])]
        if not ds_list:
            return

        if base_level is None:
            try:
                base_level = self.item.current_level()
            except Exception:
                base_level = 0
        base_level = max(0, min(base_level, len(ds_list) - 1))

        # 1) 缩放吸附
        if allow_scale_snap:
            ppd_now = self._current_ppd()
            if ppd_now > 0:
                target = snap_ppd_to_near_ds(ppd_now, ds_list, snap_log2_eps=scale_eps_log2)
                if abs(math.log2(max(ppd_now, 1e-9)) - math.log2(max(target, 1e-9))) > 1e-6:
                    self._scale_to_ppd(target, bump=False)

        # 2) 平移吸附
        if allow_translate_snap:
            tile_world = float(self.loader.tile_px) * float(ds_list[base_level])
            if tile_world <= 0:
                return
            vr = self.viewport().rect()
            if vr.isEmpty():
                return
            sr = self.mapToScene(vr).boundingRect()
            if sr.width() <= 0 or sr.height() <= 0:
                return

            left_target = round(sr.left() / tile_world) * tile_world
            top_target = round(sr.top() / tile_world) * tile_world
            dx = left_target - sr.left()
            dy = top_target - sr.top()
            if abs(dx) > (tile_world * 0.5):
                dx -= math.copysign(tile_world, dx)
            if abs(dy) > (tile_world * 0.5):
                dy -= math.copysign(tile_world, dy)

            if abs(dx) > translate_eps or abs(dy) > translate_eps:
                c = sr.center()
                self.centerOn(QPointF(c.x() + dx, c.y() + dy))

    def load_reader(self, reader):
        self.unload()
        self._zoom_locked = True

        self.reader = reader
        workers = int(self.app_cfg.get("viewer", {}).get("tile_workers", 4))
        self.loader = TileLoader(
            reader,
            tile_px=int(self.app_cfg["viewer"].get("tile_px", 256)),
            max_workers=max(1, workers),
            max_cache=int(self.app_cfg["viewer"].get("max_cache_tiles", 512)),
            parent=self,
        )
        self.item = TiledItem(self, reader, self.loader)
        self._scene.addItem(self.item)
        try:
            self._scene.setSceneRect(self.item.boundingRect())
        except Exception:
            pass

        self.loader.tileReady.connect(self.item.on_tile_ready, Qt.QueuedConnection)
        self.loader.tileReady.connect(self._on_loader_tile_ready, Qt.QueuedConnection)
        self.loader.tileFailed.connect(self._on_loader_tile_failed, Qt.QueuedConnection)

        self._set_loading_state(True, "Loading WSI...")

        # 后备底图：立即生成（若可），确保加载期无露白
        try:
            self._cover_backdrop_pix = self._make_backdrop_pix(reader, max_side=2048)
        except Exception:
            self._cover_backdrop_pix = None

        warm_tiles_cfg = int(self.app_cfg.get("viewer", {}).get("coarse_warmup_tiles", -1))
        warm_list_ready = False
        try:
            max_warm = None if warm_tiles_cfg <= 0 else max(512, warm_tiles_cfg)
            self.item.build_coarsest_warm_list(max_tiles=max_warm)
            warm_list_ready = bool(getattr(self.item, "_warm_coords", None))
        except Exception:
            warm_list_ready = False

        self.fitInView(self.item.boundingRect(), Qt.KeepAspectRatio)
        self._home_ppd = self._current_ppd()
        if self._home_ppd is not None and self._home_ppd > 0:
            self.max_ppd = max(self.max_ppd, self._home_ppd * 1.2)
        try:
            snapped = snap_ppd_to_near_ds(self._home_ppd, reader.level_downsamples, 0.08)
        except Exception:
            snapped = self._home_ppd
        if snapped and self._home_ppd and abs(log2(snapped) - log2(self._home_ppd)) > 1e-6:
            self._scale_to_ppd(snapped)
            self._home_ppd = snapped

        # 在加载动画期间，先做一次“缩放+平移”的网格吸附，确保解锁后不露底色
        try:
            self._snap_view_to_grid(
                base_level=self.item.current_level(),
                allow_scale_snap=True,
                allow_translate_snap=True,
                scale_eps_log2=0.02,
                translate_eps=0.5,
            )
        except Exception:
            pass

        # 吸附后重新获取可视范围
        try:
            vr = self.viewport().rect()
            vis = self.mapToScene(vr).boundingRect().intersected(self.item.boundingRect())
        except Exception:
            vis = QRectF()

        self._emit_metrics()

        if not vis.isEmpty():
            try:
                margin = max(0, int(getattr(self, "_coarse_prefetch_margin", 3)))
                budget = max(32, int(getattr(self, "_coarse_prefetch_zoom", 180)))
                self.item.prefetch_coarsest_for_rect(vis, margin=margin, only_center_n=budget)
            except Exception:
                pass
            try:
                self._begin_startup_prefetch(vis)
            except Exception:
                self._set_loading_state(False)
        else:
            self._set_loading_state(False)

        if warm_list_ready:
            try:
                self._warm_timer.start()
            except Exception:
                pass

        self._start_enter_anim(160)

    def unload(self):
        self._set_loading_state(False)
        if self.loader:
            try:
                self.loader.blockSignals(True)
                try:
                    self.loader.tileReady.disconnect()
                except Exception:
                    pass
                try:
                    self.loader.tileFailed.disconnect()
                except Exception:
                    pass
            except Exception:
                pass
            self.loader.stop_and_wait(10000)
            self.loader.deleteLater()
            self.loader = None

        if self.item:
            try:
                self._scene.removeItem(self.item)
            except Exception:
                pass
            self.item = None
            try:
                self._scene.clear()
            except Exception:
                pass

        # 清理帧级配额队列与状态
        try:
            self._frame_req_defer_q.clear()
        except Exception:
            pass
        self._frame_req_budget_left = getattr(self, "_frame_request_budget", 96)
        self._frame_req_posted = False


        self.reader = None
        self._home_ppd = None
        self._freeze_pix = None
        self._cover_backdrop_pix = None

        try:
            self._zoom_timer.stop()
        except Exception:
            pass
        try:
            self._warm_timer.stop()
        except Exception:
            pass
        try:
            self._kinetic_stop()
        except Exception:
            pass

        self._zoom_locked = False

    # ----- Loading timers helpers -----
    def _start_loading_timers(self):
        try:
            self._loading_soft_timer.start(self._loading_soft_ms)
            self._loading_hard_timer.start(self._loading_hard_ms)
        except Exception:
            pass

    def _cancel_loading_timers(self):
        for t in (self._loading_soft_timer, self._loading_hard_timer, self._loading_min_timer, self._loading_watchdog_timer):
            try:
                t.stop()
            except Exception:
                pass

    def _on_loading_soft_unlock(self):
        pass

    def _on_loading_hard_unlock(self):
        if not self._loading_active:
            return
        if len(self._startup_required_tiles) == 0:
            self._request_finish_loading(force=True)
        else:
            try:
                self._loading_hard_timer.start(self._loading_hard_ms)
            except Exception:
                pass
        self.viewport().update()

    def _on_loading_min_elapsed(self):
        self._loading_min_elapsed = True
        if self._loading_finish_pending and self._loading_active:
            self._request_finish_loading(force=True)

    def _on_tile_ready_rect(self, rect: QRectF):
        # 首块可见瓦片到达 → 解锁交互与缩放
        if getattr(self, "_zoom_locked", False) or getattr(self, "_interaction_locked", False):
            try:
                vr = self.viewport().rect()
                if not vr.isEmpty():
                    visible = self.mapToScene(vr).boundingRect()
                    if rect.intersects(visible):
                        self._zoom_locked = False
                        self._interaction_locked = False
            except Exception:
                self._zoom_locked = False
                self._interaction_locked = False

        # 只有当覆盖达标时才撤掉冻结遮罩（这里也参考当前层）
        if self._coarse_covered_visible(min_ratio=0.98):
            self._freeze_pix = None
            self.viewport().update()
        # ★ 新增：主层覆盖即时达标 → 立刻标记缓存并强制刷新
        try:
            vr = self.viewport().rect()
            if not vr.isEmpty() and self.item:
                vis = self.mapToScene(vr).boundingRect().intersected(self.item.boundingRect())
                lvl = self.item.current_level()
                ok, _ = self.item.has_level_coverage_for_rect(lvl, vis, min_ratio=0.98)
                if ok:
                    self._main_covered_cached = True
                    self._last_main_cover_check_t = 0.0
                    self._last_cover_check_t = 0.0
                    self.viewport().update()
        except Exception:
            pass

    def _set_loading_state(self, active: bool, text: Optional[str] = None):
        if active:
            self._loading_active = True
            self._interaction_locked = True

            self._startup_pending_tiles.clear()
            self._startup_required_tiles.clear()
            self._startup_required_total = 0
            self._startup_optional_total = 0
            self._startup_total_tiles = 0

            self._loading_progress = 0.0
            self._loading_text = text or "Loading WSI..."

            if self._locked_drag_mode is None:
                self._locked_drag_mode = self.dragMode()
            try:
                self.setDragMode(QGraphicsView.NoDrag)
            except Exception:
                pass

            self._zoom_locked = True

            try:
                self._complete_on_required = getattr(self, "_complete_on_required", True)
                self._loading_started_at = _time.perf_counter()
                self._last_progress_at = self._loading_started_at
                self._loading_min_elapsed = False
                self._loading_finish_pending = False
                self._loading_min_timer.start(self._loading_min_ms)
            except Exception:
                self._loading_min_elapsed = True
                self._loading_finish_pending = False

            self._start_loading_timers()
            try:
                self._loading_watchdog_timer.start()
            except Exception:
                pass
        else:
            self._loading_active = False
            self._interaction_locked = False

            self._loading_progress = 1.0
            self._loading_text = ""

            self._startup_pending_tiles.clear()
            self._startup_required_tiles.clear()
            self._startup_required_total = 0
            self._startup_optional_total = 0
            self._startup_total_tiles = 0

            self._loading_min_elapsed = False
            self._loading_finish_pending = False

            if self._locked_drag_mode is not None:
                try:
                    self.setDragMode(self._locked_drag_mode)
                except Exception:
                    pass
            self._locked_drag_mode = None

            self._zoom_locked = False
            self._cancel_loading_timers()
            self.viewport().update()

    def _notify_tile_ready(self, key: TileKey):
        if not self._loading_active or not self._startup_pending_tiles:
            return
        t = (int(key.level), int(key.tx), int(key.ty))
        if t not in self._startup_pending_tiles:
            return

        self._startup_pending_tiles.discard(t)
        if t in self._startup_required_tiles:
            self._startup_required_tiles.discard(t)

        req_left = len(self._startup_required_tiles)
        opt_left = len(self._startup_pending_tiles) - req_left
        req_total = max(1, self._startup_required_total)
        opt_total = max(1, self._startup_optional_total)
        req_done = self._startup_required_total - req_left
        opt_done = self._startup_optional_total - opt_left

        if self._startup_optional_total > 0:
            weight_opt = 0.25
            progress_req = req_done / req_total
            progress_opt = opt_done / opt_total
            self._loading_progress = max(0.0, min(1.0, (1.0 - weight_opt) * progress_req + weight_opt * progress_opt))
        else:
            self._loading_progress = max(0.0, min(1.0, req_done / req_total))

        try:
            self._last_progress_at = _time.perf_counter()
        except Exception:
            pass

        if not self._loading_min_elapsed:
            self._loading_progress = min(self._loading_progress, 0.96)

        if getattr(self, "_freeze_pix", None) is not None and not self._cover_mode_active():
            self._maybe_release_freeze(False)

        if getattr(self, "_complete_on_required", True) and req_left == 0:
            self._request_finish_loading(force=False)
            return

        if not self._startup_pending_tiles:
            self._request_finish_loading(force=False)
            return

        self.viewport().update()

    def _on_loader_tile_ready(self, generation: int, key: TileKey, _img: QImage):
        if not self.loader or generation != self.loader.generation:
            return
        self._notify_tile_ready(key)

    def _on_loader_tile_failed(self, generation: int, key: TileKey, _errmsg: str):
        if not self.loader or generation != self.loader.generation:
            return
        self._notify_tile_ready(key)

    def _begin_startup_prefetch(self, rect: QRectF):
        if not self.loader or not self.item or rect.isEmpty():
            self._set_loading_state(False)
            return False

        padded = QRectF(rect)
        pad = max(rect.width(), rect.height()) * 0.35
        padded.adjust(-pad, -pad, pad, pad)
        padded = padded.intersected(self.item.boundingRect())
        if padded.isEmpty():
            self._set_loading_state(False)
            return False

        ds_list = getattr(self.reader, "level_downsamples", [])
        base_level = self.item.current_level() if ds_list else 0
        base_level = max(0, min(base_level, len(ds_list) - 1))
        last_level = max(0, len(ds_list) - 1)

        # 必需：当前层 + 最粗层可见瓦片
        required_coords_base = self.item._iter_tile_coords(
            base_level, rect, margin=0, center_first=False, only_center_n=None
        )
        required_set: set[tuple[int, int, int]] = {
            (int(base_level), int(tx), int(ty)) for tx, ty in required_coords_base
        }

        required_coords_coarse = self.item._iter_tile_coords(
            last_level, rect, margin=0, center_first=False, only_center_n=None
        )
        required_set_coarse = {(int(last_level), int(tx), int(ty)) for tx, ty in required_coords_coarse}
        required_set.update(required_set_coarse)

        # 优先级：邻粗层 + 最粗层中心若干
        coarse_priority: set[tuple[int, int, int]] = set()
        if base_level + 1 < len(ds_list):
            coords_plus = self.item._iter_tile_coords(
                base_level + 1, rect, margin=0, center_first=False, only_center_n=None
            )
            coarse_priority.update((int(base_level + 1), int(tx), int(ty)) for tx, ty in coords_plus)

        if ds_list:
            coords_last = self.item._iter_tile_coords(
                last_level, padded, margin=0, center_first=True, only_center_n=96
            )
            coarse_priority.update((int(last_level), int(tx), int(ty)) for tx, ty in coords_last)

        coarse_priority -= required_set

        level_plan: list[tuple[int, int, Optional[int]]] = [(base_level, 1, 220)]
        if base_level > 0:
            level_plan.append((base_level - 1, 0, 160))
        if base_level + 1 < len(ds_list):
            level_plan.append((base_level + 1, 0, 200))
        if ds_list:
            level_plan.append((last_level, 0, None))

        pending: set[tuple[int, int, int]] = set(required_set)
        pending.update(coarse_priority)

        for lvl, margin, cap in level_plan:
            coords = self.item._iter_tile_coords(lvl, padded, margin=margin, center_first=True, only_center_n=cap)
            for tx, ty in coords:
                key = (int(lvl), int(tx), int(ty))
                pending.add(key)

        if not pending:
            self._set_loading_state(False)
            return False

        self._startup_pending_tiles = pending
        self._startup_required_tiles = required_set
        self._startup_total_tiles = len(pending)
        self._startup_required_total = len(required_set)
        self._startup_optional_total = self._startup_total_tiles - self._startup_required_total
        self._loading_progress = 0.0

        for (lvl, tx, ty) in required_set:
            try:
                self.loader.request(lvl, tx, ty)
            except Exception:
                pass
        for (lvl, tx, ty) in coarse_priority:
            try:
                self.loader.request(lvl, tx, ty)
            except Exception:
                pass
        for (lvl, tx, ty) in pending:
            if (lvl, tx, ty) in required_set:
                continue
            try:
                self.loader.request(lvl, tx, ty)
            except Exception:
                pass
        return True

    def _request_finish_loading(self, force: bool = False):
        if not self._loading_active:
            return

        if not force and not self._coarse_covered_visible(min_ratio=0.98):
            self._loading_finish_pending = True
            return

        if not force and not self._loading_min_elapsed:
            self._loading_finish_pending = True
            return

        self._loading_progress = 1.0
        self._set_loading_state(False)
        self._loading_finish_pending = False
        self._maybe_release_freeze(force=True)

    def _maybe_release_freeze(self, force: bool = False):
        pix = getattr(self, "_freeze_pix", None)
        if pix is None:
            return
        if not force:
            now = _time.perf_counter()
            if now < getattr(self, "_freeze_deadline_t", 0.0):
                return
            if self._cover_mode_active():
                return
            if getattr(self, "_loading_active", False):
                return
        self._freeze_pix = None
        try:
            self.viewport().update()
        except Exception:
            pass

    def _draw_loading_overlay(self, painter: QPainter):
        vp_rect = self.viewport().rect()
        if vp_rect.isEmpty():
            return

        painter.save()
        try:
            painter.resetTransform()
        except Exception:
            painter.setWorldMatrixEnabled(False)
        try:
            painter.setClipRect(vp_rect)
        except Exception:
            pass
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
        except Exception:
            pass

        painter.fillRect(vp_rect, QColor(0, 0, 0, 140))

        box_w = min(vp_rect.width() * 0.6, 420.0)
        box_h = 96.0
        center = vp_rect.center()
        box_rect = QRectF(center.x() - box_w * 0.5, center.y() - box_h * 0.5, box_w, box_h)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 235))
        painter.drawRoundedRect(box_rect, 10.0, 10.0)

        padding = 18.0
        fm = painter.fontMetrics()
        text_h = max(fm.height(), 18)
        text_rect = QRectF(
            box_rect.left() + padding,
            box_rect.top() + padding,
            box_rect.width() - padding * 2,
            float(text_h),
        )
        painter.setPen(QColor(32, 32, 32))
        painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, self._loading_text or "Loading...")

        progress = max(0.0, min(1.0, float(self._loading_progress)))
        bar_h = 14.0
        bar_rect = QRectF(
            box_rect.left() + padding,
            box_rect.bottom() - padding - bar_h,
            box_rect.width() - padding * 2,
            bar_h,
        )
        radius = bar_h * 0.5
        painter.setBrush(QColor(225, 225, 225))
        painter.drawRoundedRect(bar_rect, radius, radius)

        fill_w = bar_rect.width() * progress
        if fill_w > 1.0:
            fill_rect = QRectF(bar_rect.left(), bar_rect.top(), fill_w, bar_rect.height())
            painter.setBrush(QColor(66, 133, 244))
            painter.drawRoundedRect(fill_rect, radius, radius)

        painter.setPen(QColor(45, 45, 45))
        painter.drawText(bar_rect, Qt.AlignCenter, f"{round(progress * 100):d}%")
        painter.restore()

    def wheelEvent(self, e):
        if getattr(self, "_loading_active", False) or getattr(self, "_interaction_locked", False) or getattr(self, "_zoom_locked", False):
            e.ignore()
            return
        if not self.reader or not self.item:
            return super().wheelEvent(e)

        self._kinetic_stop()
        self._panning = False

        delta_units = self._extract_wheel_delta(e)
        if delta_units == 0.0:
            e.ignore()
            return

        steps = delta_units / 120.0 or delta_units / 240.0
        factor = pow(self._zoom_factor, steps)
        if not math.isfinite(factor) or abs(factor - 1.0) < 1e-4:
            e.ignore()
            return

        zooming_in = factor > 1.0
        ppd_now = self._current_ppd()
        target_ppd = ppd_now / factor
        lo, hi = self.min_ppd, self.max_ppd
        if self._home_ppd is not None:
            hi = max(hi, self._home_ppd * 1.2)
        if target_ppd < lo:
            if ppd_now <= lo + 1e-9:
                e.ignore()
                return
            target_ppd = lo
            factor = max(1e-9, ppd_now / target_ppd)
            zooming_in = factor > 1.0
        elif target_ppd > hi:
            if ppd_now >= hi - 1e-9:
                e.ignore()
                return
            target_ppd = hi
            factor = max(1e-9, ppd_now / target_ppd)
            zooming_in = factor > 1.0

        prev_mode = None
        update_locked = False
        try:
            prev_mode = self.viewportUpdateMode()
            self._prev_update_mode = prev_mode
            self.setViewportUpdateMode(QGraphicsView.NoViewportUpdate)
            update_locked = True
        except Exception:
            prev_mode = None

        scheduled_timer = False
        try:
            capture_freeze = False
            if self._freeze_capture_enabled:
                rel = factor if zooming_in else (1.0 / max(1e-6, factor))
                if rel >= self._freeze_capture_min_factor:
                    now_cap = _time.perf_counter()
                    if (now_cap - self._freeze_last_capture_t) >= self._freeze_capture_interval:
                        capture_freeze = True
                        self._freeze_last_capture_t = now_cap
            if capture_freeze:
                try:
                    self._freeze_pix = self.viewport().grab()
                except Exception:
                    self._freeze_pix = None
            else:
                self._freeze_pix = None

            try:
                pos = e.position() if hasattr(e, "position") else e.pos()
                if hasattr(pos, "toPointF"):
                    pos = pos.toPointF()
                if hasattr(pos, "toPoint"):
                    pos = pos.toPoint()
                self._zoom_focus_scene = self.mapToScene(pos)
            except Exception:
                self._zoom_focus_scene = None

            self._last_scale_dir = 1 if zooming_in else -1
            self._last_scale_t = _time.perf_counter()
            super().scale(factor, factor)

            now_perf = _time.perf_counter()
            self._cover_until_t = max(getattr(self, "_cover_until_t", 0.0), now_perf + self._cover_hold_ms / 1000.0)
            if capture_freeze and self._freeze_pix is not None:
                self._freeze_deadline_t = max(getattr(self, "_freeze_deadline_t", 0.0), now_perf + self._freeze_hold_ms / 1000.0)
            else:
                self._freeze_deadline_t = 0.0

            try:
                ppd_pred = max(1e-6, target_ppd)
                overs = 2.2 if zooming_in else 1.4
                lvl_pred = select_level_smart(self.reader.level_downsamples, ppd_pred, oversample_limit=overs)
            except Exception:
                lvl_pred = 0

            try:
                now_t = _time.perf_counter()
                if (now_t - self._last_prefetch_t) > 0.08:
                    vr = self.viewport().rect()
                    vis_after = self.mapToScene(vr).boundingRect().intersected(self.item.boundingRect())
                    if not vis_after.isEmpty():
                        coarse_margin = max(0, int(getattr(self, "_coarse_prefetch_margin", 3)))
                        coarse_target = self._coarse_prefetch_zoom if zooming_in else self._coarse_prefetch_idle
                        coarse_count = max(32, int(coarse_target))
                        try:
                            self.item.prefetch_coarsest_for_rect(
                                vis_after, margin=coarse_margin, only_center_n=coarse_count,
                            )
                        except Exception:
                            pass

                        last = len(self.reader.level_downsamples) - 1
                        if last >= 0 and last != lvl_pred:
                            lvl_coarse = min(lvl_pred + 1, last)
                            if lvl_coarse != lvl_pred:
                                try:
                                    neighbor_target = self._coarse_prefetch_idle if zooming_in else max(32, self._coarse_prefetch_idle // 2)
                                    neighbor_budget = max(32, int(neighbor_target))
                                    self.item.prefetch_level_for_rect(
                                        lvl_coarse, vis_after, margin=1, center_first=True, only_center_n=neighbor_budget,
                                    )
                                except Exception:
                                    pass

                        main_budget = self._request_budget_zoom if zooming_in else self._request_budget_idle
                        try:
                            main_budget = int(main_budget)
                        except Exception:
                            main_budget = 0
                        main_budget = max(0, main_budget)
                        only_center_main = max(32, main_budget) if zooming_in or main_budget > 0 else None
                        self.item.prefetch_level_for_rect(
                            lvl_pred, vis_after, margin=1, center_first=True, only_center_n=only_center_main,
                        )
                        if zooming_in and lvl_pred > 0:
                            self.item.prefetch_level_for_rect(
                                lvl_pred - 1, vis_after, margin=0, center_first=True, only_center_n=48,
                            )
                    self._last_prefetch_t = now_t
            except Exception:
                pass

            if self.item:
                self.item.prefetch_after_wheel(self._last_scale_dir)

            self._zoom_timer.start()
            scheduled_timer = True

            # ★ 新增：缩放事件末尾就安排一次“紧急全量补齐 + 扫尾计时器”
            try:
                self._schedule_fallback_prefetch(urgency=1)
                self._post_zoom_sweep_deadline = _time.perf_counter() + 1.2  # 延长到 1.2s
                self._post_zoom_sweep_timer.start()
            except Exception:
                pass

            self._emit_metrics()
            self.viewport().update()
            e.accept()
        finally:
            if update_locked and not scheduled_timer and prev_mode is not None:
                try:
                    self.setViewportUpdateMode(prev_mode)
                except Exception:
                    pass

    def drawForeground(self, painter: QPainter, rect):
        # ---- 修复：冻结帧按屏幕坐标绘制，避免“菱形/错位/持久化” ----
        if self._freeze_pix is not None:
            if (_time.perf_counter() <= getattr(self, "_freeze_deadline_t", 0.0)) or self._cover_mode_active():
                vp_rect = self.viewport().rect()
                painter.save()
                try:
                    painter.resetTransform()  # 关键：清除场景变换，回到视口坐标
                except Exception:
                    painter.setWorldMatrixEnabled(False)
                try:
                    painter.setClipRect(vp_rect)  # 只画在视口范围
                except Exception:
                    pass
                painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
                painter.drawPixmap(vp_rect, self._freeze_pix, self._freeze_pix.rect())
                painter.restore()
            else:
                self._freeze_pix = None
            super().drawForeground(painter, rect)
        else:
            super().drawForeground(painter, rect)

        if self._enter_anim_t > 0.0:
            alpha = int(255 * (1.0 - min(1.0, self._enter_anim_t)))
            # 进入动效用屏幕坐标绘制也更稳定
            painter.save()
            try:
                painter.resetTransform()
            except Exception:
                painter.setWorldMatrixEnabled(False)
            painter.fillRect(self.viewport().rect(), QColor(255, 255, 255, alpha))
            painter.restore()

        if self._loading_active:
            self._draw_loading_overlay(painter)

    def drawBackground(self, painter: QPainter, rect):
        painter.fillRect(rect, self.backgroundBrush())

    def _apply_deferred_zoom(self):
        try:
            if not self.reader:
                return
            ppd_now = self._current_ppd()
            lo, hi = self.min_ppd, self.max_ppd
            if self._home_ppd is not None:
                hi = max(hi, self._home_ppd * 1.2)
            target = max(lo, min(hi, ppd_now))
            target = snap_ppd_to_near_ds(target, getattr(self.reader, "level_downsamples", []), 0.08)
            self._scale_to_ppd(target, bump=False)
            if self.loader:
                self.loader.bump_generation()

            # 网格吸附（仅平移吸附）
            try:
                self._snap_view_to_grid(
                    base_level=self.item.current_level() if self.item else None,
                    allow_scale_snap=False,
                    allow_translate_snap=True,
                    translate_eps=0.25,
                )
            except Exception:
                pass

            # 唯一保留的“整块补齐”+ 覆盖检测（边缘=1）
            try:
                self._prefetch_visible_rect(full_cover=True, extra_margin=1, min_ratio=0.99)
            except Exception:
                pass

            # 扫尾时间窗口 0.8 s（其余场景只靠扫尾，不再做全量补齐）
            try:
                self._post_zoom_sweep_deadline = _time.perf_counter() + 0.8
                self._post_zoom_sweep_timer.start()
            except Exception:
                pass

        finally:
            # 加载期间做一次平移吸附
            try:
                if self._loading_active:
                    self._snap_view_to_grid(
                        base_level=self.item.current_level() if self.item else None,
                        allow_scale_snap=False,
                        allow_translate_snap=True,
                        translate_eps=0.5,
                    )
            except Exception:
                pass

            try:
                self.setViewportUpdateMode(self._prev_update_mode)
            except Exception:
                pass
            self.viewport().update()
            self._zoom_focus_scene = None

    # def mouseDoubleClickEvent(self, e):
    #     if getattr(self, "_loading_active", False) or getattr(self, "_interaction_locked", False):
    #         e.ignore()
    #         return
    #     if e.button() == Qt.LeftButton and getattr(self, "_allow_doubleclick_home", True):
    #         self.zoom_to_fit()
    #         e.accept()
    #         return
    #     super().mouseDoubleClickEvent(e)

    def mousePressEvent(self, e):
        if getattr(self, "_interaction_locked", False):
            e.ignore()
            return
        if e.button() == Qt.LeftButton:
            self._kinetic_stop()
            self._drag_samples.clear()
            self._add_drag_sample(e)
            self._panning = True
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if getattr(self, "_interaction_locked", False):
            e.ignore()
            return

        # 只有在按下左键拖动时，才记录拖动轨迹并标记为“正在平移”
        buttons = getattr(e, "buttons", lambda: 0)()
        if buttons & Qt.LeftButton:
            self._add_drag_sample(e)
            self._panning = True

        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if getattr(self, "_interaction_locked", False):
            e.ignore()
            return
        super().mouseReleaseEvent(e)
        if e.button() == Qt.LeftButton:
            vx, vy = self._calc_release_velocity()
            self._kinetic_start(vx, vy)
            if not getattr(self, "_kinetic_active", False):
                try:
                    self._post_zoom_sweep_deadline = _time.perf_counter() + 0.8
                    self._post_zoom_sweep_timer.start()
                except Exception:
                    pass
            self._panning = False

    def rotate_left(self):
        if getattr(self, "_loading_active", False):
            return
        self.rotate(-90)
        if self.loader:
            self.loader.bump_generation()
        if self.item:
            self.item.clear_pix()

    def rotate_right(self):
        if getattr(self, "_loading_active", False):
            return
        self.rotate(90)
        if self.loader:
            self.loader.bump_generation()
        if self.item:
            self.item.clear_pix()

    def begin_pan_transition(self, *, capture: bool = True, cover_ms: int = 220, freeze_ms: int = 120):
        """Provide a lightweight transition when external widgets (e.g., minimap) recenter the view."""
        now = _time.perf_counter()
        if capture:
            try:
                self._freeze_pix = self.viewport().grab()
                if self._freeze_pix is not None:
                    self._freeze_deadline_t = now + max(0.0, float(freeze_ms)) / 1000.0
            except Exception:
                self._freeze_pix = None
        cover_deadline = now + max(0.0, float(cover_ms)) / 1000.0
        self._cover_until_t = max(getattr(self, "_cover_until_t", 0.0), cover_deadline)
        try:
            self.viewport().update()
        except Exception:
            pass

    def _emit_metrics(self):
        if not self.reader:
            return
        vr = self.viewport().rect()
        if vr.isEmpty():
            return
        sr = self.mapToScene(vr).boundingRect()
        if sr.width() <= 0:
            return
        ppd = sr.width() / max(1.0, vr.width())
        ds_list = [float(d) for d in self.reader.level_downsamples]
        lvl = self.item.current_level() if self.item else 0
        ds = ds_list[lvl] if 0 <= lvl < len(ds_list) else 1.0
        mag = None
        try:
            obj0 = self.reader.objective_power()
            if obj0 and ppd > 0:
                mag = obj0 / ppd
        except Exception:
            mag = None
        self.metricsChanged.emit(mag, ppd, lvl, ds)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._emit_metrics()

    def _start_enter_anim(self, duration_ms: int = 160):
        self._enter_anim_t = 1e-6
        self._enter_anim_deadline = _time.perf_counter() + duration_ms / 1000.0
        try:
            self._enter_anim_timer.start()
        except Exception:
            pass

    def _on_enter_anim_tick(self):
        if self._enter_anim_t <= 0.0:
            self._enter_anim_timer.stop()
            return
        now = _time.perf_counter()
        if now >= getattr(self, "_enter_anim_deadline", 0.0):
            self._enter_anim_t = 0.0
            self._enter_anim_timer.stop()
        else:
            total = max(1e-3, self._enter_anim_deadline - (self._enter_anim_deadline - 0.16))
            rest = self._enter_anim_deadline - now
            self._enter_anim_t = max(0.0, rest / total)
            self.viewport().update()

    def _add_drag_sample(self, e):
        try:
            p = e.position() if hasattr(e, "position") else e.pos()
            if hasattr(p, "toPointF"):
                p = p.toPointF()
            t = _time.perf_counter()
            self._drag_samples.append((t, float(p.x()), float(p.y())))
        except Exception:
            pass

    def _calc_release_velocity(self):
        if len(self._drag_samples) < 2:
            return 0.0, 0.0
        t1, x1, y1 = self._drag_samples[-1]
        idx = None
        for i in range(len(self._drag_samples) - 2, -1, -1):
            t0, _, _ = self._drag_samples[i]
            if (t1 - t0) >= 0.04:
                idx = i
                break
        if idx is None:
            t0, x0, y0 = self._drag_samples[0]
        else:
            t0, x0, y0 = self._drag_samples[idx]
        dt = max(1e-4, t1 - t0)
        vx = (x1 - x0) / dt
        vy = (y1 - y0) / dt
        speed = math.hypot(vx, vy)
        if speed > self._kinetic_max_speed > 0:
            scale = self._kinetic_max_speed / speed
            vx *= scale
            vy *= scale
        return vx, vy

    def _kinetic_start(self, vx, vy):
        if not self._kinetic_enabled:
            return
        if math.hypot(vx, vy) < self._kinetic_min_speed:
            return
        self._kinetic_vx = vx
        self._kinetic_vy = vy
        self._kinetic_last_t = _time.perf_counter()
        self._kinetic_active = True
        self._panning = True
        try:
            self._kinetic_timer.start()
        except Exception:
            pass

    def _kinetic_stop(self):
        self._kinetic_active = False
        try:
            self._kinetic_timer.stop()
        except Exception:
            pass
        # 惯性停止 → 做一次全量补齐与扫尾
        try:
            # self._prefetch_visible_rect(full_cover=True, extra_margin=2, min_ratio=0.985)
            self._post_zoom_sweep_deadline = _time.perf_counter() + 0.8
            self._post_zoom_sweep_timer.start()
        except Exception:
            pass
        try:
            self._panning_decay_timer.start()  # 延迟清除“拖动态”，避免末尾突然失去像素对齐
        except Exception:
            self._panning = False

    def _on_kinetic_tick(self):
        if not self._kinetic_active:
            return
        now = _time.perf_counter()
        dt = now - self._kinetic_last_t
        self._kinetic_last_t = now
        if dt <= 1e-5:
            return
        try:
            decay = math.exp(-float(self._kinetic_decay) * dt)
        except Exception:
            decay = 0.9
        self._kinetic_vx *= decay
        self._kinetic_vy *= decay
        if math.hypot(self._kinetic_vx, self._kinetic_vy) < self._kinetic_min_speed:
            self._kinetic_stop()
            return

        dx = -self._kinetic_vx * dt
        dy = -self._kinetic_vy * dt

        hbar = self.horizontalScrollBar()
        vbar = self.verticalScrollBar()
        moved = False
        try:
            if hbar:
                v = int(hbar.value() + dx)
                if v != hbar.value():
                    hbar.setValue(v)
                    moved = True
            if vbar:
                v = int(vbar.value() + dy)
                if v != vbar.value():
                    vbar.setValue(v)
                    moved = True
        except Exception:
            moved = False

        if not moved:
            self._kinetic_stop()

    def _do_pan_prefetch(self):
        if not (self.item and self.reader):
            return
        vr = self.viewport().rect()
        vis = self.mapToScene(vr).boundingRect().intersected(self.item.boundingRect())
        if vis.isEmpty():
            return
        try:
            lvl = self.item.current_level()
            center_coarse = max(32, int(getattr(self, "_coarse_prefetch_pan", 260)))
            center_main = max(32, int(getattr(self, "_request_budget_idle", 220)))
            center_fine = max(24, int(center_main // 2))
            coarse_margin = max(0, int(getattr(self, "_coarse_prefetch_margin", 3)))

            padded = QRectF(vis)
            pad = max(vis.width(), vis.height()) * 0.25
            if pad > 0:
                padded.adjust(-pad, -pad, pad, pad)
            padded = padded.intersected(self.item.boundingRect())

            target_rect = padded if padded.width() > 0 and padded.height() > 0 else vis

            self.item.prefetch_coarsest_for_rect(
                target_rect, margin=coarse_margin, only_center_n=center_coarse,
            )
            last = self.item.last_level()
            if last > lvl:
                self.item.prefetch_level_for_rect(
                    min(lvl + 1, last), target_rect, margin=1, center_first=True, only_center_n=120,
                )
            self.item.prefetch_level_for_rect(
                lvl, target_rect, margin=1, center_first=True, only_center_n=center_main,
            )
            if lvl > 0:
                self.item.prefetch_level_for_rect(
                    lvl - 1, target_rect, margin=0, center_first=True, only_center_n=center_fine,
                )
        except Exception:
            pass

    def _flush_pan_prefetch_queue(self):
        self._pan_prefetch_pending = False
        try:
            self._do_pan_prefetch()
        except Exception:
            pass

    def scrollContentsBy(self, dx, dy):
        need_immediate = abs(int(dx)) > 2 or abs(int(dy)) > 2
        if int(dx) != 0 or int(dy) != 0:
            self._panning = True
            try:
                self._panning_decay_timer.start()
            except Exception:
                pass
        super().scrollContentsBy(dx, dy)
        if need_immediate and not self._pan_prefetch_pending:
            self._pan_prefetch_pending = True
            try:
                QTimer.singleShot(0, self._flush_pan_prefetch_queue)
            except Exception:
                self._pan_prefetch_pending = False
            try:
                self._pan_prefetch_timer.start()
            except Exception:
                pass

    def _on_loading_watchdog_tick(self):
        if not self._loading_active:
            try:
                self._loading_watchdog_timer.stop()
            except Exception:
                pass
            return
        now = _time.perf_counter()
        req_left = len(self._startup_required_tiles)
        if req_left == 0 and self._coarse_covered_visible(min_ratio=0.98):
            self._request_finish_loading(force=True)
            return

        total_to = float(self.app_cfg["viewer"].get("loading_soft_timeout_s", 5.0))
        idle_to = float(self.app_cfg["viewer"].get("loading_idle_timeout_s", 3.0))
        if (now - self._loading_started_at > total_to) or (now - self._last_progress_at > idle_to):
            self._startup_required_tiles.clear()
            if self._coarse_covered_visible(min_ratio=0.98):
                self._request_finish_loading(force=True)

    def _prefetch_visible_rect(self, *, full_cover: bool = False, extra_margin: int = 1,
                               min_ratio: float | None = None):
        """对当前层的可视区域做一次预取：
         - full_cover=True：不做中心限额，整块补齐（用于缩放/平移结束后的最终扫尾）
         - full_cover=False：中心优先，轻量补齐（用于一般 idle 触发）
         - min_ratio：只在 full_cover=False 时生效，用于“够用就别再下”的轻量优化
        """
        if not (self.item and self.reader):
            return
        vr = self.viewport().rect()
        if vr.isEmpty():
            return
        vis = self.mapToScene(vr).boundingRect().intersected(self.item.boundingRect())
        if vis.isEmpty():
            return

        lvl = self.item.current_level()

        # ---- 覆盖检测：只在非 full_cover（轻量补齐）时启用 ----
        if (not full_cover) and isinstance(min_ratio, (int, float)) and min_ratio > 0:
            try:
                ok, _ratio = self.item.has_level_coverage_for_rect(lvl, vis, min_ratio=float(min_ratio))
            except Exception:
                ok = False
            if ok:
                # 主层覆盖已经够了：刷新一下缓存状态，顺便解除冻结，然后直接返回
                try:
                    self._main_covered_cached = True
                    self._last_main_cover_check_t = 0.0
                    self._last_cover_check_t = 0.0
                except Exception:
                    pass
                try:
                    self._coarse_covered_visible(min_ratio=float(min_ratio))
                except Exception:
                    pass
                self._freeze_pix = None
                try:
                    self.viewport().update()
                except Exception:
                    pass
                return

        # ---- 预取逻辑 ----
        if full_cover:
            # 整块补齐：不做限额、不中心优先，确保整个可视区域的主层都被下完
            self.item.prefetch_level_for_rect(
                lvl,
                vis,
                margin=max(0, int(extra_margin)),
                center_first=False,
                only_center_n=None,
            )
        else:
            # 轻量：中心优先 + 限额（不阻塞绘制）
            center_main = max(32, int(getattr(self, "_request_budget_idle", 220)))
            self.item.prefetch_level_for_rect(
                lvl,
                vis,
                margin=1,
                center_first=True,
                only_center_n=center_main,
            )

    def _schedule_fallback_prefetch(self, *, urgency: int = 0):
        """
        当画面里出现 fallback（粗层马赛克）时触发：
          - 静止状态：先按“平移预取”逻辑做一轮 _do_pan_prefetch（相当于虚拟拖动一下）
          - 然后再用 _prefetch_visible_rect 做一次兜底（必要时整块补齐）
        """
        now = _time.perf_counter()
        interval = 0.035 if urgency > 0 else 0.12
        last = getattr(self, "_last_fallback_prefetch_t", 0.0)
        if (now - last) < interval:
            return

        try:
            # 1）如果当前并没有真正平移/惯性滑动，就模拟一次“平移预取”
            if not getattr(self, "_panning", False) and not getattr(self, "_kinetic_active", False):
                try:
                    self._do_pan_prefetch()  # 复用已有的平移预取策略（带 padding）
                except Exception:
                    pass

            # 2）再用原来的可视区域预取做一次兜底
            self._prefetch_visible_rect(
                full_cover=(urgency > 0),
                extra_margin=(2 if urgency > 0 else 1),
                min_ratio=(0.985 if urgency > 0 else None),
            )

            self._last_fallback_prefetch_t = now
        except Exception:
            pass

    def _budgeted_request(self, level: int, tx: int, ty: int, priority: int | None = None):
        """
        统一的“带配额”请求入口：
          - 若本轮还有配额，直接下发 loader.request 并递减
          - 若配额用尽，入队，安排下一轮事件循环冲刷
          - 为了确保“每轮都能重置配额”，在本轮第一次消耗配额时也安排一个 singleShot(0, ...)
        """
        if not self.loader:
            return
        # 第一次消耗配额时安排一次 flush，保证“下一轮”会重置配额
        if self._frame_req_budget_left == self._frame_request_budget and not self._frame_req_posted:
            self._frame_req_posted = True
            QTimer.singleShot(0, self._flush_budgeted_requests)

        if self._frame_req_budget_left > 0:
            self._frame_req_budget_left -= 1
            try:
                self.loader.request(level, tx, ty, priority=priority if priority is not None else 0)
            except Exception:
                pass
        else:
            self._frame_req_defer_q.append((int(level), int(tx), int(ty), int(priority or 0)))
            if not self._frame_req_posted:
                self._frame_req_posted = True
                QTimer.singleShot(0, self._flush_budgeted_requests)

    def _flush_budgeted_requests(self):
        """
        下一轮事件循环中被调用：
         - 重置配额
         - 把队列里积压的请求按配额发出去
         - 若还没清空，则再安排下一轮 singleShot(0, ...)
        """
        self._frame_req_posted = False
        if not self.loader:
            self._frame_req_defer_q.clear()
            self._frame_req_budget_left = self._frame_request_budget
            return

        # 每“帧”重置配额
        self._frame_req_budget_left = self._frame_request_budget

        while self._frame_req_defer_q and self._frame_req_budget_left > 0:
            lvl, tx, ty, prio = self._frame_req_defer_q.popleft()
            try:
                self.loader.request(lvl, tx, ty, priority=prio)
            except Exception:
                pass
            self._frame_req_budget_left -= 1

        if self._frame_req_defer_q:
            # 还有积压 → 下一轮继续
            self._frame_req_posted = True
            QTimer.singleShot(0, self._flush_budgeted_requests)


    def _post_zoom_sweep_tick(self):
        """在一小段时间内，反复检查主层覆盖度，不足就继续补齐请求，直到≥99.5%或到时限。"""
        if not (self.item and self.reader):
            self._post_zoom_sweep_timer.stop()
            return
        vr = self.viewport().rect()
        if vr.isEmpty():
            self._post_zoom_sweep_timer.stop()
            return
        vis = self.mapToScene(vr).boundingRect().intersected(self.item.boundingRect())
        if vis.isEmpty():
            self._post_zoom_sweep_timer.stop()
            return

        # ★ 新增：扫尾阶段也安排一次统一重绘，避免只靠单个 tile 的局部 update
        try:
            if hasattr(self, "_tile_repaint_timer") and not getattr(self, "_tile_repaint_pending", False):
                self._tile_repaint_pending = True
                self._tile_repaint_timer.start()
        except Exception:
            pass

        # 统一复用带覆盖检测的预取函数
        self._prefetch_visible_rect(full_cover=True, extra_margin=2, min_ratio=0.99)

        # 达标或过时 → 停止扫尾
        lvl = self.item.current_level()
        ok, _ = self.item.has_level_coverage_for_rect(lvl, vis, min_ratio=0.995)
        if ok or (_time.perf_counter() > getattr(self, "_post_zoom_sweep_deadline", 0.0)):
            self._post_zoom_sweep_timer.stop()
            return


    def _on_tile_repaint_tick(self):
        self._tile_repaint_pending = False
        # 有新瓦片时强制刷新一次视口
        try:
            self.viewport().update()
        except Exception:
            pass
