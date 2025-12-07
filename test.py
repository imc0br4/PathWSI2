#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import fnmatch
import os
import sys
from pathlib import Path
from typing import Iterable, List, Set

BOX_MID = "├─ "
BOX_END = "└─ "
BOX_BAR = "│  "
BOX_PAD = "   "

DEFAULT_IGNORES = {
    "__pycache__", "*.pyc", "*.pyo", "*.pyd",
    ".DS_Store", "Thumbs.db",
}

def should_ignore(name: str, ignore_patterns: Set[str], show_hidden: bool) -> bool:
    if not show_hidden and name.startswith("."):
        return True
    for pat in ignore_patterns:
        if fnmatch.fnmatch(name, pat):
            return True
    return False

def list_entries(path: Path, sort_dirs_first: bool, ignore_patterns: Set[str],
                 show_hidden: bool, include_files: bool, include_dirs: bool,
                 follow_symlinks: bool) -> List[Path]:
    try:
        entries = [Path(entry.path) for entry in os.scandir(path)]
    except PermissionError:
        return []

    def is_dir_like(p: Path) -> bool:
        if p.is_dir():
            return True
        if follow_symlinks and p.is_symlink():
            try:
                return p.resolve().is_dir()
            except Exception:
                return False
        return False

    def keep(p: Path) -> bool:
        name = p.name
        if should_ignore(name, ignore_patterns, show_hidden):
            return False
        if is_dir_like(p):
            return include_dirs
        else:
            return include_files

    entries = [e for e in entries if keep(e)]

    def sort_key(p: Path):
        dfirst = 0 if (sort_dirs_first and (p.is_dir())) else 1
        return (dfirst, p.name.lower())

    entries.sort(key=sort_key)
    return entries

def tree(path: Path,
         prefix: str,
         max_depth: int,
         ignore_patterns: Set[str],
         show_hidden: bool,
         sort_dirs_first: bool,
         include_files: bool,
         include_dirs: bool,
         follow_symlinks: bool,
         fullpath: bool,
         out_lines: List[str]) -> None:
    if max_depth == 0:
        return

    entries = list_entries(path, sort_dirs_first, ignore_patterns, show_hidden,
                           include_files, include_dirs, follow_symlinks)

    count = len(entries)
    for idx, e in enumerate(entries):
        connector = BOX_END if idx == count - 1 else BOX_MID
        is_dir = e.is_dir()
        name = str(e if fullpath else e.name)
        out_lines.append(prefix + f"{connector}{name}{'/' if is_dir else ''}")

        if is_dir:
            extension = BOX_PAD if connector == BOX_END else BOX_BAR
            try:
                tree(e, prefix + extension, max_depth - 1, ignore_patterns,
                     show_hidden, sort_dirs_first, include_files, include_dirs,
                     follow_symlinks, fullpath, out_lines)
            except (PermissionError, OSError):
                continue

def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Print directory tree with box-drawing characters."
    )
    p.add_argument("path", nargs="?", default=".", help="Target directory (default: current dir)")
    p.add_argument("-d", "--max-depth", type=int, default=9999, help="Max depth to traverse")
    p.add_argument("-I", "--ignore", action="append", default=[],
                   help="Glob pattern to ignore (repeatable). e.g. -I '*.log' -I 'build' -I 'dist'")
    p.add_argument("--no-default-ignore", action="store_true",
                   help="Do not apply default ignores like __pycache__ and *.pyc")
    p.add_argument("--show-hidden", action="store_true", help="Include dotfiles")
    p.add_argument("--dirs-first", action="store_true", help="List directories before files")
    p.add_argument("--files-only", action="store_true", help="Only list files")
    p.add_argument("--dirs-only", action="store_true", help="Only list directories")
    p.add_argument("-L", "--follow-symlinks", action="store_true", help="Follow symlinks to directories")
    p.add_argument("--fullpath", action="store_true", help="Show full path for entries")
    p.add_argument("-o", "--output", help="Write result to a file instead of stdout")
    return p.parse_args(argv)

def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    # Resolve target path
    root = Path(args.path).resolve()
    if not root.exists():
        print(f"Error: path not found: {root}", file=sys.stderr)
        return 1

    # Build ignore set (default + user)
    ignore_patterns: Set[str] = set(args.ignore or [])
    if not args.no_default_ignore:
        ignore_patterns |= DEFAULT_IGNORES

    include_files = not args.dirs_only
    include_dirs = not args.files_only

    lines: List[str] = []

    # ----- Root title: use user's input if possible -----
    # If user passed ".", show current folder name; else show the original arg's basename
    user_arg = args.path
    if user_arg == ".":
        root_title = f"{Path.cwd().name}/"
    else:
        root_title = f"{Path(user_arg).name}/"
    lines.append(root_title)

    # If root itself是文件，直接打印后返回
    if root.is_file():
        lines[-1] = str(root)  # show file path
    else:
        tree(root, "", args.max_depth, ignore_patterns, args.show_hidden,
             args.dirs_first, include_files, include_dirs, args.follow_symlinks,
             args.fullpath, lines)

    out_text = "\n".join(lines)
    if args.output:
        Path(args.output).write_text(out_text, encoding="utf-8")
    else:
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
        print(out_text)
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
