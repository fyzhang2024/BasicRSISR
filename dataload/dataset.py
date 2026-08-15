"""
SCDR_RADR dataset module.
保留两个类:
- DatasetFromFolder:      训练集, 输出 (lr, hr) tuple, 支持随机 crop + 翻转/旋转。
- DatasetFromFolderEval:  验证/测试集, 输出 (lr, hr, gt_path) tuple, 不做 crop, 不做增强。

去掉的内容: UED teacher cache / 场景标签 / 域 id / hard patch sampler 等所有创新分支用的钩子。
Directory convention: GT/ and LR/ are side by side under the same root, with paired file names.
"""
import csv
import os
import random
from os.path import join

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image, ImageOps

from utils.scene_gate_utils import image_rel_path_from_lr


IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def is_image_file(filename):
    # 中文注释: 只接受常见图像扩展, 忽略 README / .DS_Store / 缩略图等噪声文件。
    return filename.lower().endswith(IMG_EXTENSIONS)


def load_img(filepath):
    # 中文注释: 统一转 RGB, 避免单通道/带 alpha 图引发后续 to_tensor 报错。
    return Image.open(filepath).convert("RGB")


def _image_paths(folder):
    """递归扫描目录下所有图像, 排序稳定。"""
    paths = []
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        for name in sorted(files):
            if is_image_file(name):
                paths.append(join(root, name))
    return sorted(paths)


def resolve_lr_path_for_hr(hr_path, hr_dir, lr_dir):
    """根据 HR 路径推断对应 LR 路径。
    优先按相对目录镜像找, 再退回 basename / GT->LR 替换的常见命名约定。
    """
    candidates = []
    try:
        rel_path = os.path.relpath(os.path.abspath(hr_path), os.path.abspath(hr_dir))
        if not rel_path.startswith(".."):
            candidates.append(join(lr_dir, rel_path))
    except (ValueError, OSError):
        pass
    basename = os.path.basename(hr_path)
    candidates.append(join(lr_dir, basename))
    candidates.append(hr_path.replace("{}GT{}".format(os.sep, os.sep), "{}LR{}".format(os.sep, os.sep)))
    candidates.append(hr_path.replace("/GT/", "/LR/").replace("\\GT\\", "\\LR\\"))
    candidates.append(join(os.path.dirname(os.path.dirname(hr_path)), "LR", basename))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError("LR pair not found for {}".format(hr_path))


# ---------------------------------------------------------------------
# 增强 / crop 工具
# ---------------------------------------------------------------------


def _sample_augment_ops(flip_h=True, rot=True):
    """随机抽取 vflip / hflip / rotate180 的子集, 与 SCDR_RADR/SwinIR 风格一致。"""
    ops = []
    if random.random() < 0.5 and flip_h:
        ops.append("vflip")
    if rot:
        if random.random() < 0.5:
            ops.append("hflip")
        if random.random() < 0.5:
            ops.append("rotate180")
    return tuple(ops)


def _apply_augment_ops_pil(img, ops):
    for op in ops:
        if op == "vflip":
            img = ImageOps.flip(img)
        elif op == "hflip":
            img = ImageOps.mirror(img)
        elif op == "rotate180":
            img = img.rotate(180)
        else:
            raise ValueError("Unsupported augment op: {}".format(op))
    return img


def _apply_augment_ops_array(arr, ops):
    """把与 GT 对齐的 HxW 或 CxHxW 数组做与图像相同的翻转/旋转。"""
    out = np.asarray(arr)
    for op in ops:
        if op == "vflip":
            out = np.flip(out, axis=-2)
        elif op == "hflip":
            out = np.flip(out, axis=-1)
        elif op == "rotate180":
            out = np.flip(np.flip(out, axis=-2), axis=-1)
        else:
            raise ValueError("Unsupported augment op: {}".format(op))
    return np.ascontiguousarray(out)


def get_patch(img_in, img_tar, patch_size, scale):
    """LR 上随机取 patch_size 大小的方块, HR 上同步取 patch_size*scale 大小的方块。"""
    lr_w, lr_h = img_in.size
    lr_patch = max(1, int(patch_size))
    hr_patch = lr_patch * int(scale)
    if lr_w < lr_patch or lr_h < lr_patch:
        raise ValueError("LR image {} is smaller than patch size {}".format(img_in.size, lr_patch))
    ix = random.randrange(0, lr_w - lr_patch + 1)
    iy = random.randrange(0, lr_h - lr_patch + 1)
    tx = ix * int(scale)
    ty = iy * int(scale)
    img_in = img_in.crop((ix, iy, ix + lr_patch, iy + lr_patch))
    img_tar = img_tar.crop((tx, ty, tx + hr_patch, ty + hr_patch))
    return img_in, img_tar


def get_patch_coords(img_in, patch_size, scale):
    """采样与 get_patch 相同的随机 crop 坐标，供标签图同步裁剪。"""
    lr_w, lr_h = img_in.size
    lr_patch = max(1, int(patch_size))
    hr_patch = lr_patch * int(scale)
    if lr_w < lr_patch or lr_h < lr_patch:
        raise ValueError("LR image {} is smaller than patch size {}".format(img_in.size, lr_patch))
    ix = random.randrange(0, lr_w - lr_patch + 1)
    iy = random.randrange(0, lr_h - lr_patch + 1)
    tx = ix * int(scale)
    ty = iy * int(scale)
    return {
        "lr_x": ix,
        "lr_y": iy,
        "lr_patch": lr_patch,
        "hr_x": tx,
        "hr_y": ty,
        "hr_patch": hr_patch,
    }


def crop_pair_by_coords(img_in, img_tar, coords):
    """按共享坐标裁剪 LR/HR。"""
    img_in = img_in.crop((
        coords["lr_x"],
        coords["lr_y"],
        coords["lr_x"] + coords["lr_patch"],
        coords["lr_y"] + coords["lr_patch"],
    ))
    img_tar = img_tar.crop((
        coords["hr_x"],
        coords["hr_y"],
        coords["hr_x"] + coords["hr_patch"],
        coords["hr_y"] + coords["hr_patch"],
    ))
    return img_in, img_tar


def crop_array_hr(arr, coords):
    """按 HR 尺度坐标裁剪标签或 valid mask。"""
    return np.ascontiguousarray(
        np.asarray(arr)[
            ...,
            coords["hr_y"]:coords["hr_y"] + coords["hr_patch"],
            coords["hr_x"]:coords["hr_x"] + coords["hr_patch"],
        ]
    )


def _resolve_index_path(path_value, index_dir):
    if not path_value:
        return ""
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(index_dir, path_value))


def load_radr_label_index(index_path):
    """读取 CSV，建立 image_rel_path -> label/valid 路径映射。"""
    if not index_path:
        return {}
    if not os.path.isfile(index_path):
        raise FileNotFoundError("RADR label index not found: {}".format(index_path))
    mapping = {}
    index_dir = os.path.dirname(os.path.abspath(index_path))
    with open(index_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image_rel_path", "label_path", "valid_path"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("RADR label index missing required fields {}: {}".format(sorted(missing), index_path))
        for row in reader:
            rel_path = str(row.get("image_rel_path", "")).replace("\\", "/").strip("/")
            if not rel_path:
                raise ValueError("RADR label index contains empty image_rel_path: {}".format(index_path))
            label_path = _resolve_index_path(str(row.get("label_path", "")).strip(), index_dir)
            valid_path = _resolve_index_path(str(row.get("valid_path", "")).strip(), index_dir)
            if not label_path or not os.path.isfile(label_path):
                raise FileNotFoundError("RADR label file not found for {}: {}".format(rel_path, label_path))
            if not valid_path or not os.path.isfile(valid_path):
                raise FileNotFoundError("RADR valid file not found for {}: {}".format(rel_path, valid_path))
            mapping[rel_path] = {
                "label_path": label_path,
                "valid_path": valid_path,
            }
    if not mapping:
        raise RuntimeError("RADR label index has no valid rows: {}".format(index_path))
    return mapping


# ---------------------------------------------------------------------
# 训练 / 验证 dataset
# ---------------------------------------------------------------------


class DatasetFromFolder(data.Dataset):
    """训练集: HR/LR 成对; 随机 crop + 可选翻转/旋转; 通过 transform 转 tensor。"""

    def __init__(self, HR_dir, LR_dir, patch_size, upscale_factor, data_augmentation,
                 transform=None, use_radr_label=False, radr_label_index=""):
        super().__init__()
        if not os.path.isdir(HR_dir):
            raise FileNotFoundError("HR directory not found: {}".format(HR_dir))
        if not os.path.isdir(LR_dir):
            raise FileNotFoundError("LR directory not found: {}".format(LR_dir))
        self.hr_image_filenames = _image_paths(HR_dir)
        if not self.hr_image_filenames:
            raise RuntimeError("No HR images found in {}".format(HR_dir))
        self.patch_size = int(patch_size)
        self.upscale_factor = int(upscale_factor)
        self.data_augmentation = bool(data_augmentation)
        self.transform = transform
        self.hr_root = os.path.abspath(HR_dir)
        self.lr_root = os.path.abspath(LR_dir)
        self.use_radr_label = bool(use_radr_label)
        self.radr_label_index = str(radr_label_index or "")
        self.radr_label_map = load_radr_label_index(self.radr_label_index) if self.use_radr_label else {}

    def __len__(self):
        return len(self.hr_image_filenames)

    def __getitem__(self, index):
        target_path = self.hr_image_filenames[index]
        input_path = resolve_lr_path_for_hr(target_path, self.hr_root, self.lr_root)
        target = load_img(target_path)
        input_img = load_img(input_path)
        rel_path = image_rel_path_from_lr(input_path, self.lr_root)
        radr_u_gt = None
        radr_valid = None
        if self.use_radr_label:
            label_item = self.radr_label_map.get(rel_path, None)
            if label_item is None:
                raise KeyError("RADR label missing for training sample {}".format(rel_path))
            radr_u_gt = np.load(label_item["label_path"])
            radr_valid = np.load(label_item["valid_path"])
            if radr_u_gt.ndim == 2:
                radr_u_gt = radr_u_gt[None, ...]
            if radr_valid.ndim == 2:
                radr_valid = radr_valid[None, ...]
            if radr_u_gt.shape != radr_valid.shape:
                raise RuntimeError("RADR label/valid shape mismatch for {}: {} vs {}".format(
                    rel_path, tuple(radr_u_gt.shape), tuple(radr_valid.shape)
                ))
            hr_h, hr_w = target.size[1], target.size[0]
            if tuple(radr_u_gt.shape[-2:]) != (hr_h, hr_w):
                raise RuntimeError("RADR label shape mismatch for {}: label {} vs GT {}".format(
                    rel_path, tuple(radr_u_gt.shape[-2:]), (hr_h, hr_w)
                ))
        coords = get_patch_coords(input_img, self.patch_size, self.upscale_factor)
        input_img, target = crop_pair_by_coords(input_img, target, coords)
        if self.use_radr_label:
            radr_u_gt = crop_array_hr(radr_u_gt, coords)
            radr_valid = crop_array_hr(radr_valid, coords)
        if self.data_augmentation:
            ops = _sample_augment_ops()
            if ops:
                input_img = _apply_augment_ops_pil(input_img, ops)
                target = _apply_augment_ops_pil(target, ops)
                if self.use_radr_label:
                    radr_u_gt = _apply_augment_ops_array(radr_u_gt, ops)
                    radr_valid = _apply_augment_ops_array(radr_valid, ops)
        if self.transform is not None:
            input_img = self.transform(input_img)
            target = self.transform(target)
        if not self.use_radr_label:
            return input_img, target
        radr_u_gt = torch.from_numpy(np.ascontiguousarray(radr_u_gt)).float()
        radr_valid = torch.from_numpy(np.ascontiguousarray(radr_valid)).float()
        return input_img, target, input_path, radr_u_gt, radr_valid


class DatasetFromFolderEval(data.Dataset):
    """验证/测试集: 不做 crop, 不做增强; 返回 (lr, hr, gt_path) 方便保存结果。"""

    def __init__(self, HR_dir, LR_dir, upscale_factor, transform=None):
        super().__init__()
        self.hr_image_filenames = _image_paths(HR_dir)
        self.lr_image_filenames = _image_paths(LR_dir)
        self.upscale_factor = int(upscale_factor)
        self.transform = transform

    def __len__(self):
        return len(self.hr_image_filenames)

    def __getitem__(self, index):
        target_path = self.hr_image_filenames[index]
        basename = os.path.basename(target_path)
        # 中文注释: 与训练集保持一致, 优先按 basename 找对应 LR; 找不到时再退回 GT/LR 字符串替换。
        input_path = join(os.path.dirname(self.lr_image_filenames[0]), basename) if self.lr_image_filenames else ""
        if not os.path.isfile(input_path):
            input_path = target_path.replace("/GT/", "/LR/").replace("\\GT\\", "\\LR\\")
        target = load_img(target_path)
        input_img = load_img(input_path)
        if self.transform is not None:
            input_img = self.transform(input_img)
            target = self.transform(target)
        return input_img, target, target_path
