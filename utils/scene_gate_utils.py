"""Scene Prior Gate 分析用的轻量工具函数。"""

import csv  # 中文注释：使用标准库写 CSV，避免引入新的依赖。
import os  # 中文注释：用于路径拼接、相对路径和目录创建。


def image_rel_path_from_lr(lr_path, lr_root):
    """返回 LR 图像相对于 LR 根目录的路径，并统一为 POSIX 风格斜杠。"""
    # 中文注释：先转绝对路径，避免当前工作目录变化导致 relpath 不稳定。
    rel_path = os.path.relpath(os.path.abspath(lr_path), os.path.abspath(lr_root))
    # 中文注释：Windows 下 relpath 会生成反斜杠，这里统一替换成 / 方便跨平台合并 CSV。
    return rel_path.replace(os.sep, "/")


def scene_from_rel_path(image_rel_path, folder=""):
    """按项目约定从相对路径和当前 folder 解析场景名。"""
    # 中文注释：去掉首尾斜杠，避免 split 后出现空字段。
    clean_rel_path = str(image_rel_path).strip("/")
    # 中文注释：只要 LR 相对路径中有目录，第一层目录就是场景名。
    parts = [part for part in clean_rel_path.split("/") if part]
    # 中文注释：LR/airport/xxx.png 这种结构会进入这里，scene=airport。
    if len(parts) >= 2:
        return parts[0]
    # 中文注释：如果相对路径只有文件名，则优先使用当前验证 folder。
    if folder:
        return str(folder).strip("/\\") or "unknown"
    # 中文注释：既没有目录也没有 folder 时，显式标记 unknown。
    return "unknown"


def scene_and_rel_path(lr_path, lr_root, folder=""):
    """一次性返回 scene 和 image_rel_path，保证不同脚本使用同一规则。"""
    # 中文注释：先计算统一风格的 LR 相对路径。
    rel_path = image_rel_path_from_lr(lr_path, lr_root)
    # 中文注释：再基于相对路径和 folder 推断场景。
    scene = scene_from_rel_path(rel_path, folder=folder)
    # 中文注释：返回值顺序贴近 CSV 字段顺序，调用侧更直观。
    return scene, rel_path


def write_csv_rows(path, fieldnames, rows, append=False):
    """写入 CSV 行；append=True 时自动避免重复表头。"""
    # 中文注释：没有有效行时直接返回，避免生成空 CSV 干扰后续分析。
    if not rows:
        return
    # 中文注释：确保父目录存在，兼容全局 CSV 和局部 CSV 的不同位置。
    # 中文注释：允许用户把 CSV 直接写到当前目录，此时 dirname 为空不需要建目录。
    parent_dir = os.path.dirname(path)
    # 中文注释：只有父目录非空时才创建目录。
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    # 中文注释：追加模式下只有文件不存在或为空时才写表头。
    need_header = (not append) or (not os.path.isfile(path)) or os.path.getsize(path) == 0
    # 中文注释：按 append 决定打开模式；newline="" 避免 Windows 下空行。
    mode = "a" if append else "w"
    # 中文注释：UTF-8 便于中文路径或场景名安全落盘。
    with open(path, mode, newline="", encoding="utf-8") as f:
        # 中文注释：DictWriter 保证字段顺序稳定，便于后续脚本读取。
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # 中文注释：同一次运行多 folder 追加时只在首个 folder 写表头。
        if need_header:
            writer.writeheader()
        # 中文注释：批量写入逐图记录。
        writer.writerows(rows)
