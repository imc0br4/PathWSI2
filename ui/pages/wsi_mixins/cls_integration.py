# ui/pages/wsi_mixins/cls_integration.py
from __future__ import annotations
import os, json
import numpy as np
from PySide6.QtWidgets import QMessageBox
from ui.dialogs.cls_dialog import ClsRunDialog


class ClsIntegrationMixin:
    def _on_cls_clicked(self):
        # 1) 当前 WSI 路径
        slide_path = None
        try:
            slide_path = getattr(self.reader, "path", None)
        except Exception:
            slide_path = None
        if not slide_path or not os.path.isfile(slide_path):
            QMessageBox.warning(self, "提示", "请先打开一张 WSI。")
            return

        # 2) palette（优先自身，其次 MainWindow）
        palette = getattr(self, "palette", None)
        if palette is None and hasattr(self.window(), "palette"):
            palette = getattr(self.window(), "palette")
        if palette is None:
            palette = {}

        # 3) models_cfg：优先从 app_cfg['models_json'] 读取“原始”文件
        models_cfg = None
        base_dir = None
        try:
            if isinstance(self.app_cfg, dict):
                models_json = self.app_cfg.get("models_json")
                if models_json and os.path.isfile(models_json):
                    with open(models_json, "r", encoding="utf-8") as f:
                        models_cfg = json.load(f)  # 保持原有结构（list/dict 均可）
                    base_dir = os.path.dirname(models_json)
        except Exception:
            pass

        # 兜底：用 self.models_cfg 或 MainWindow.models_cfg
        if models_cfg is None:
            models_cfg = getattr(self, "models_cfg", None)
            if models_cfg is None and hasattr(self.window(), "models_cfg"):
                models_cfg = getattr(self.window(), "models_cfg")
            # 兜底基准目录
            if base_dir is None:
                try:
                    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                    base_dir = os.path.join(root, "configs")
                except Exception:
                    base_dir = os.getcwd()

        if models_cfg is None:
            QMessageBox.warning(self, "提示", "未找到模型配置（configs/models.json）。")
            return

        # 4) 过滤掉检测模型，只保留分类模型
        #
        # 约定：
        #   - 在 configs/models.json 里，把检测模型标记为:
        #         "task": "det"   或  "type": "det" / "yolo" / "detection"
        #   - 分类模型可以不写 task/type，或者写 "cls"
        #
        # 这样分类对话框就不会把 yolov11.pt 也当成候选模型了。
        models_cfg = self._filter_cls_models(models_cfg)
        if models_cfg is None:
            QMessageBox.warning(self, "提示", "模型配置中未找到可用的分类模型（请检查 task/type 字段）。")
            return

        # 5) 绝对化权重路径（支持 list / {"models": {...}} / 纯 dict）
        def _absify_inplace(cfg, base):
            """
            把模型里的 weight/path 变成绝对路径：
             - 优先用 app_cfg['project_root'] 及其父目录（推到项目根）
             - 同时尝试常见子目录：models / models/weights / weights
             - 最后才退回到 base_dir（例如 ui/configs），防止拼错到 ui\\configs\\models\\weights
            """
            # 1) 构造搜索根目录列表
            search_dirs = []

            # (a) 当前 base_dir 本身
            if base:
                try:
                    search_dirs.append(os.path.abspath(base))
                except Exception:
                    pass

            # (b) 从 app_cfg 里推项目根及其父目录
            try:
                if isinstance(self.app_cfg, dict):
                    pr = self.app_cfg.get("project_root")
                    if isinstance(pr, str) and pr:
                        pr = os.path.abspath(os.path.expanduser(os.path.expandvars(pr)))
                        # 例如：C:\\a\\PathWSI-Seg\\ui
                        search_dirs.append(pr)
                        # 父目录：C:\\a\\PathWSI-Seg（通常是真正的项目根）
                        parent_pr = os.path.dirname(pr)
                        if parent_pr:
                            search_dirs.append(parent_pr)
                            # 常见模型存放位置
                            search_dirs.append(os.path.join(parent_pr, "models"))
                            search_dirs.append(os.path.join(parent_pr, "models", "weights"))
                            search_dirs.append(os.path.join(parent_pr, "weights"))
            except Exception:
                pass

            # (c) 从 base_dir 往上再爬两级，兜底到项目根
            if base:
                try:
                    b0 = os.path.abspath(base)
                    b1 = os.path.dirname(b0)
                    b2 = os.path.dirname(b1)
                    for d in (b1, b2):
                        if d:
                            search_dirs.append(d)
                except Exception:
                    pass

            # (d) 最后兜底：当前工作目录
            if not search_dirs:
                try:
                    search_dirs.append(os.getcwd())
                except Exception:
                    pass

            # 去重
            uniq = []
            for d in search_dirs:
                if d and d not in uniq:
                    uniq.append(d)
            search_dirs = uniq

            def _resolve_one(p_raw: str) -> str:
                """把单个路径变成绝对路径并尽量确保文件存在。"""
                p = os.path.expanduser(os.path.expandvars(p_raw))
                # 已经是绝对路径：只做规范化
                if os.path.isabs(p):
                    return os.path.abspath(p)

                # 依次在 search_dirs 下尝试拼接
                for root_dir in search_dirs:
                    cand = os.path.abspath(os.path.join(root_dir, p))
                    if os.path.isfile(cand):
                        return cand

                # 所有尝试都失败，至少给出一个合理的绝对路径（便于错误提示）
                try:
                    return os.path.abspath(os.path.join(search_dirs[0], p))
                except Exception:
                    return p

            def fix_one(d: dict):
                for k in ("weight", "path", "weights", "checkpoint"):
                    v = d.get(k)
                    if isinstance(v, str) and v.strip():
                        resolved = _resolve_one(v.strip())
                        d[k] = resolved
                        # 同时把 path/weight 对齐，避免对话框只读其中一个键
                        d.setdefault("path", resolved)
                        d.setdefault("weight", resolved)

            # 按原来的结构遍历 cfg
            if isinstance(cfg, list):
                for it in cfg:
                    if isinstance(it, dict):
                        fix_one(it)
            elif isinstance(cfg, dict):
                if "models" in cfg and isinstance(cfg["models"], dict):
                    for _, it in cfg["models"].items():
                        if isinstance(it, dict):
                            fix_one(it)
                else:
                    # 可能就是 {name: {...}} 的映射
                    for _, it in cfg.items():
                        if isinstance(it, dict):
                            fix_one(it)

        try:
            _absify_inplace(models_cfg, base_dir or os.getcwd())
        except Exception:
            pass

        # 6) 构造弹窗：严格按其签名传参
        try:
            dlg = ClsRunDialog(
                app_cfg=self.app_cfg,
                palette=palette,
                models_cfg=models_cfg,   # 允许是 list 或 dict；我们已绝对化
                logger=self.log,
                slide_path=slide_path,
                parent=self,
            )
        except TypeError as e:
            QMessageBox.critical(self, "错误", f"分类弹窗参数不兼容：{e}")
            return

        if hasattr(dlg, "sigOverlayReady"):
            dlg.sigOverlayReady.connect(self._on_cls_overlay_ready)
        dlg.exec()

    def _on_cls_overlay_ready(self, rgba, meta):
        """把分类结果直接铺到当前视图。"""
        try:
            if rgba is None:
                return
            if self.overlay_item is None or self.overlay_item.scene() is None:
                self._create_overlay_item()
            self.overlay_item.set_rgba(rgba)
            # 对齐参数（如果 meta 里有）
            ds = float(meta.get("downsample", getattr(self, "_overlay_ds", 1.0)))
            x0, y0, w0, h0 = 0, 0, rgba.shape[1], rgba.shape[0]
            if isinstance(meta.get("bbox_level0"), (list, tuple)) and len(meta["bbox_level0"]) >= 4:
                x0, y0, w0, h0 = [int(v) for v in meta["bbox_level0"][:4]]
            # 1) 取正确的 ds（若 meta 没带 downsample，用 reader 的 level ds）
            lvl = int(meta.get("level", 0))
            if "downsample" in meta:
                ds = float(meta["downsample"])
            else:
                ds_list = [float(d) for d in self.reader.level_downsamples]
                ds = ds_list[lvl] if 0 <= lvl < len(ds_list) else 1.0

            # 2) 显示安全降采样 + 同步 ds / grid
            rgba, ds, _, meta = self._downsample_overlay_for_display(rgba, ds, meta)

            # 3) 再去 set_rgba / setScale / setPos ...
            self.overlay_item.setPos(int(x0), int(y0))
            self.overlay_item.setScale(ds)
            self.overlay_item.setOpacity(self._overlay_opacity)
            self.overlay_item.setVisible(True)

            # 记住状态，后续编辑 / 二阶段检测 用
            self._overlay_rgba = rgba
            self._overlay_ds = ds
            self._overlay_pos = (int(x0), int(y0))
            self._overlay_meta = meta if isinstance(meta, dict) else {}
            # 自动推断格子大小
            auto_tile = self._infer_tile_from_meta(self._overlay_meta, default=self._grid_tile_px)
            if auto_tile and auto_tile > 0:
                self._grid_tile_px = int(auto_tile)
            if hasattr(self, "btn_edit"):
                self.btn_edit.setEnabled(True)
        except Exception:
            self.log.exception("Apply overlay failed")

    def _apply_overlay_from_worker(self, rgba: np.ndarray, meta: dict):
        """直接把分类得到的 overlay RGBA + meta 应用到场景，并记录对齐。"""
        try:
            if rgba is None or rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
                QMessageBox.warning(self, "警告", "收到的 overlay 数组无效。")
                return

            if rgba.shape[2] == 3:
                # 补 alpha 以便叠加
                a = np.full((rgba.shape[0], rgba.shape[1], 1), 180, dtype=np.uint8)
                rgba = np.concatenate([rgba, a], axis=2)

            # 解析 meta（与你保存的 _meta.json 对齐字段一致）
            lvl = int(meta.get("level", 0))
            ds  = float(meta.get("downsample", 1.0))
            roi = meta.get("bbox_level0") or meta.get("roi_level0") or [0, 0, 0, 0]
            x0 = int(roi[0]) if len(roi) >= 2 else 0
            y0 = int(roi[1]) if len(roi) >= 2 else 0

            if self.overlay_item is None or self.overlay_item.scene() is None:
                self._create_overlay_item()
            self.overlay_item.set_rgba(rgba)
            self.overlay_item.setPos(x0, y0)
            self.overlay_item.setScale(ds)
            self.overlay_item.setVisible(True)
            self.overlay_item.setOpacity(self._overlay_opacity)

            # 记录状态，后续编辑 / 导出 / 二阶段检测 可用
            self._overlay_rgba = rgba
            self._overlay_ds   = ds
            self._overlay_pos  = (x0, y0)
            self._overlay_meta = meta

            # 自动推断网格大小（供擦除/添加对齐）
            auto_tile = self._infer_tile_from_meta(self._overlay_meta, default=self._grid_tile_px)
            if auto_tile and auto_tile > 0:
                self._grid_tile_px = int(auto_tile)

            # 启用“编辑”按钮
            if hasattr(self, "btn_edit"):
                self.btn_edit.setEnabled(True)

            QMessageBox.information(self, "分类完成", "Overlay 已加载到视图。")
        except Exception as e:
            self.log.error("应用 overlay 失败", exc_info=True)
            QMessageBox.critical(self, "错误", f"应用 overlay 失败：{e}")

    # ------------------------------------------------------------------
    #  只保留“分类模型”的小工具
    # ------------------------------------------------------------------
    def _filter_cls_models(self, models_cfg):
        """
        根据 task/type 字段过滤掉检测模型，只保留分类模型。

        约定：
          - 检测模型：  task/type in {"det", "detect", "detection", "yolo"}
          - 分类模型：  task/type == "cls" 或 未设置（None / ""）
        """
        if models_cfg is None:
            return None

        def is_det_model(d: dict) -> bool:
            if not isinstance(d, dict):
                return False
            key = (d.get("task") or d.get("type") or "").lower()
            return key in {"det", "detect", "detection", "yolo"}

        # list 结构：[ {..}, {..}, ... ]
        if isinstance(models_cfg, list):
            filtered = [m for m in models_cfg if not (isinstance(m, dict) and is_det_model(m))]
            return filtered or None

        # dict 结构
        if isinstance(models_cfg, dict):
            # { "models": { name: {...}, ... }, ... }
            if "models" in models_cfg and isinstance(models_cfg["models"], dict):
                new_models = {}
                for name, spec in models_cfg["models"].items():
                    if isinstance(spec, dict) and is_det_model(spec):
                        continue
                    new_models[name] = spec
                if not new_models:
                    return None
                new_cfg = dict(models_cfg)
                new_cfg["models"] = new_models
                return new_cfg
            else:
                # 普通映射：{ name: {...}, "some_global": xxx, ... }
                new_cfg = {}
                for k, v in models_cfg.items():
                    if isinstance(v, dict) and is_det_model(v):
                        # 丢弃检测模型
                        continue
                    new_cfg[k] = v
                return new_cfg or None

        # 其它未知结构：原样返回
        return models_cfg

    # ------------------------------------------------------------------
    #  二阶段检测使用的接口
    # ------------------------------------------------------------------
    def get_cls_overlay_for_det(self):
        """
        提供给 det_integration 调用，返回用于二阶段检测的 overlay 信息。

        返回:
            None  -> 当前没有 overlay
            dict  -> {
                "rgba": np.ndarray(H, W, 4),
                "meta": dict,                # 分类阶段传下来的原始 meta
                "downsample": float,         # 显示使用的缩放（相对 level-0）
                "pos_level0": (x0, y0),      # overlay 左上角在 level-0 坐标系下的位置
            }
        """
        rgba = getattr(self, "_overlay_rgba", None)
        meta = getattr(self, "_overlay_meta", None)
        ds   = getattr(self, "_overlay_ds", None)
        pos  = getattr(self, "_overlay_pos", None)
        if rgba is None or meta is None:
            return None
        return {
            "rgba": rgba,
            "meta": meta,
            "downsample": float(ds) if ds is not None else 1.0,
            "pos_level0": pos or (0, 0),
        }
