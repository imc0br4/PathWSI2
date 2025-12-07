# ui/pages/wsi_mixins/models_cfg.py
from __future__ import annotations
import os, json

class ModelsCfgMixin:
    def _normalize_models_cfg(self, cfg) -> dict:
        def list_to_map(lst):
            m = {}
            for i, it in enumerate(lst or []):
                if not isinstance(it, dict):
                    continue
                name = it.get("name")
                if not name:
                    p = it.get("path") or it.get("weight") or it.get("weights") or it.get("checkpoint") or f"model_{i}.pth"
                    name = os.path.splitext(os.path.basename(str(p)))[0]
                # NEW: 拷贝并规范 backend 小写
                item = dict(it)
                b = item.get("backend")
                if isinstance(b, str):
                    item["backend"] = b.strip().lower()
                m[name] = item
            return m

        if cfg is None:
            return {"models": {}}

        if isinstance(cfg, list):
            return {"models": list_to_map(cfg)}

        if isinstance(cfg, dict):
            out = dict(cfg)
            models = out.get("models")
            if isinstance(models, list):
                out["models"] = list_to_map(models)
            elif isinstance(models, dict):
                # CHG: 逐项 backend 统一
                out["models"] = {
                    k: {**v, **({"backend": (v.get("backend") or "").strip().lower()} if isinstance(v.get("backend"), str) else {})}
                    for k, v in models.items()
                }
            else:
                if len(cfg) > 0 and all(isinstance(v, dict) for v in cfg.values()):
                    has_pathish = sum(1 for v in cfg.values()
                                      if any(k in v for k in ("path", "weight", "weights", "checkpoint")))
                    if has_pathish >= max(1, len(cfg)//2):
                        out = {"models": {
                            k: {**v, **({"backend": (v.get("backend") or "").strip().lower()} if isinstance(v.get("backend"), str) else {})}
                            for k, v in cfg.items()
                        }}
                    else:
                        out = {"models": {}}
            return out

        return {"models": {}}

    def _load_models_cfg_default(self) -> dict:
        cfg = None
        mw = None
        try:
            mw = self.window()
        except Exception:
            pass

        # ① 优先从 MainWindow 取：models_cfg 或 models（两者都兼容）
        if mw is not None:
            if hasattr(mw, "models_cfg"):
                cfg = getattr(mw, "models_cfg")
            elif hasattr(mw, "models"):
                cfg = getattr(mw, "models")

        models_json = None

        # ② 磁盘兜底
        if cfg is None:
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            models_json = os.path.join(root, "configs", "models.json")
            try:
                with open(models_json, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:
                cfg = None

        cfg = self._normalize_models_cfg(cfg)

        # ③ 写入定位信息
        try:
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            if isinstance(self.app_cfg, dict):
                self.app_cfg.setdefault("project_root", root)
                if models_json and os.path.isfile(models_json):
                    self.app_cfg.setdefault("models_json", models_json)
                    self.app_cfg.setdefault("models_base_dir", os.path.dirname(models_json))
                else:
                    self.app_cfg.setdefault("models_base_dir", os.path.join(root, "configs"))
        except Exception:
            pass

        # ④ 解析绝对路径（补齐 path/weight + 默认字段）
        cfg = self._postprocess_and_resolve_paths(cfg)
        return cfg

    # ---------- helpers ----------
    def _postprocess_and_resolve_paths(self, cfg: dict) -> dict:
        if not isinstance(cfg, dict):
            return {"models": {}}
        models = cfg.get("models", {}) or {}
        if not isinstance(models, dict):
            cfg["models"] = {}
            return cfg

        base_dirs = []
        try:
            if isinstance(self.app_cfg, dict):
                d = self.app_cfg.get("models_base_dir")
                if isinstance(d, str) and d:
                    base_dirs.append(os.path.abspath(os.path.expanduser(os.path.expandvars(d))))
                mj = self.app_cfg.get("models_json")
                if isinstance(mj, str) and mj:
                    base_dirs.append(os.path.abspath(os.path.dirname(mj)))
                pr = self.app_cfg.get("project_root")
                if isinstance(pr, str) and pr:
                    pr = os.path.abspath(pr)
                    base_dirs.append(pr)
                    # NEW: 常见模型路径追加
                    base_dirs.append(os.path.join(pr, "models"))
                    base_dirs.append(os.path.join(pr, "models", "weights"))
                    base_dirs.append(os.path.join(pr, "checkpoints"))
                    base_dirs.append(os.path.join(pr, "weights"))
        except Exception:
            pass
        # 代码所在位置上推两级（项目根兜底）
        base_dirs.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

        # 去重保持顺序
        uniq = []
        for d in base_dirs:
            if d and d not in uniq:
                uniq.append(d)
        base_dirs = uniq

        for name, item in list(models.items()):
            if not isinstance(item, dict):
                continue

            # 统一名称
            item.setdefault("name", name)

            # 采集候选路径键
            candidates = []
            for k in ("path", "weight", "weights", "checkpoint", "file"):
                v = item.get(k)
                if isinstance(v, str) and v.strip():
                    candidates.append(v.strip())

            resolved = None
            for c in candidates:
                resolved = self._resolve_candidate_path(c, base_dirs)
                if resolved:
                    break

            if not resolved:
                # 只有文件名时，限深递归搜索
                for c in candidates:
                    fname = os.path.basename(c)
                    if fname and fname == c:
                        resolved = self._search_filename_in_dirs(fname, base_dirs, max_depth=4)
                        if resolved:
                            break

            if resolved:
                # 统一回写（有些弹窗只看 weight）
                item["path"] = resolved
                item.setdefault("weight", resolved)
                item["_dir"] = os.path.dirname(resolved)  # NEW: 该模型文件所在目录，便于相对资源解析
            else:
                item.setdefault("_path_unresolved", True)

            # NEW: 字段补全与规范化
            # backend 默认 torch；统一小写
            b = item.get("backend")
            if not b:
                item["backend"] = "torch"
            elif isinstance(b, str):
                item["backend"] = b.strip().lower()

            # num_classes 自动从 classes 推断
            if "classes" in item and isinstance(item["classes"], list) and not item.get("num_classes"):
                try:
                    item["num_classes"] = int(len(item["classes"]))
                except Exception:
                    pass

            # 统一 path/weight 字段（避免上层读取时二选一判断）
            w = item.get("path") or item.get("weight") or item.get("checkpoint")
            if w:
                item["path"] = w
                item["weight"] = w

            models[name] = item

        cfg["models"] = models
        return cfg

    def _resolve_candidate_path(self, p: str, base_dirs: list[str]) -> str | None:
        if not p:
            return None
        p1 = os.path.expanduser(os.path.expandvars(p))
        if os.path.isabs(p1) and os.path.isfile(p1):
            return os.path.abspath(p1)
        for d in base_dirs:
            full = os.path.abspath(os.path.join(d, p1))
            if os.path.isfile(full):
                return full
        return None

    def _search_filename_in_dirs(self, filename: str, base_dirs: list[str], max_depth: int = 4) -> str | None:
        try:
            for root in base_dirs:
                if not os.path.isdir(root):
                    continue
                stack = [(root, 0)]
                seen = set()
                while stack:
                    cur, dep = stack.pop()
                    if cur in seen:
                        continue
                    seen.add(cur)
                    try:
                        with os.scandir(cur) as it:
                            for ent in it:
                                if ent.is_file() and ent.name == filename:
                                    return os.path.abspath(ent.path)
                                if ent.is_dir() and dep < max_depth:
                                    if ent.name.lower() in (".git", "__pycache__", "build", "dist"):
                                        continue
                                    stack.append((ent.path, dep + 1))
                    except Exception:
                        pass
        except Exception:
            pass
        return None

    # 提供一个“列表版”，很多弹窗更喜欢 list
    def _models_as_list(self) -> list[dict]:
        out = []
        cfg = getattr(self, "models_cfg", None) or {}
        models = cfg.get("models", {}) if isinstance(cfg, dict) else {}
        for name, it in models.items():
            if not isinstance(it, dict):
                continue
            d = dict(it)
            d["name"] = d.get("name", name)
            w = d.get("weight") or d.get("path")
            if w:
                d["weight"] = w
                d["path"] = w
            out.append(d)
        return out
