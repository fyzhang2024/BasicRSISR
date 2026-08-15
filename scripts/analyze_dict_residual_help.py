"""Analyze whether ATD dictionary residual help is decoupled from LR structure."""

import argparse
import contextlib
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.checkpoint_utils import load_torch_file, strip_module_prefix  # noqa: E402
from utils.image_utils import IMAGE_EXTS, is_image_file, load_rgb_tensor  # noqa: E402
from utils.basicsr_compat import ensure_minimal_basicsr_compat  # noqa: E402
from utils.model_utils import DEFAULT_SCDR_RADR_CONFIG as DEFAULT_ATD_CONFIG  # noqa: E402
from utils.scene_gate_utils import scene_and_rel_path, write_csv_rows  # noqa: E402

ensure_minimal_basicsr_compat()
from model_archs.ATD_arch import ATD  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze ATD dictionary residual help against LR structure")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="dataset root, expected LR/ and GT/ subfolders")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="ATD polished baseline checkpoint")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--patch_size_hr", type=int, default=32,
                        help="non-overlapping patch size on HR/SR space")
    parser.add_argument("--crop_border", type=int, default=None,
                        help="crop border before help analysis; default equals scale")
    parser.add_argument("--max_images", type=int, default=0,
                        help="0 means use all images")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--save_patch_csv", type=int, default=1)
    parser.add_argument("--save_per_image_csv", type=int, default=1)
    parser.add_argument("--save_help_maps", type=int, default=0,
                        help="1: save per-image help map visualizations and arrays")
    parser.add_argument("--save_help_max_images", type=int, default=50,
                        help="maximum number of images to save help map visualizations; 0 means save all")
    parser.add_argument("--help_vis_percentile", type=float, default=99.0,
                        help="percentile for symmetric help-map visualization normalization")
    parser.add_argument("--analyze_help_spatial", type=int, default=1,
                        help="1: compute help sign spatial continuity metrics")
    parser.add_argument("--help_neg_threshold", type=float, default=0.0,
                        help="threshold for negative help; help < threshold is treated as harmful")
    parser.add_argument("--component_min_area", type=int, default=16,
                        help="minimum pixel area counted as a large negative-help component")
    return parser.parse_args()


def fmt(value):
    if value is None:
        return ""
    return "{:.6f}".format(float(value))


def corrcoef_np(x, y, eps=1e-8):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape[0] != y.shape[0] or x.shape[0] < 2:
        return None
    x = x - np.mean(x)
    y = y - np.mean(y)
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std < eps or y_std < eps:
        return None
    return float(np.mean(x * y) / (x_std * y_std + eps))


def percentile_np(values, q):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


def discover_lr_images(lr_root):
    rows = []
    for root, _, names in os.walk(lr_root):
        for name in sorted(names):
            path = os.path.join(root, name)
            if os.path.isfile(path) and is_image_file(path):
                rows.append(path)
    return sorted(rows)


def find_gt_for_lr(lr_path, lr_root, gt_root):
    rel_path = os.path.relpath(os.path.abspath(lr_path), os.path.abspath(lr_root))
    direct = os.path.join(gt_root, rel_path)
    if os.path.isfile(direct):
        return direct
    rel_dir = os.path.dirname(rel_path)
    stem = os.path.splitext(os.path.basename(rel_path))[0]
    for ext in IMAGE_EXTS:
        candidate = os.path.join(gt_root, rel_dir, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def _tuple_arg(value):
    if isinstance(value, (tuple, list)):
        return tuple(int(v) for v in value)
    if isinstance(value, str):
        return tuple(int(v.strip()) for v in value.split(",") if v.strip())
    return value


def build_baseline_model(scale, config=None):
    cfg = dict(DEFAULT_ATD_CONFIG)
    for key, value in (config or {}).items():
        if key in cfg:
            cfg[key] = value
    depths = _tuple_arg(cfg["atd_depths"])
    num_heads = _tuple_arg(cfg["atd_num_heads"])
    return ATD(
        img_size=cfg["atd_img_size"],
        patch_size=cfg["atd_patch_size"],
        in_chans=cfg["atd_in_chans"],
        embed_dim=cfg["atd_embed_dim"],
        depths=depths,
        num_heads=num_heads,
        window_size=cfg["atd_window_size"],
        dim_ffn_td=cfg["atd_dim_ffn_td"],
        category_size=cfg["atd_category_size"],
        num_tokens=cfg["atd_num_tokens"],
        reducted_dim=cfg["atd_reducted_dim"],
        convffn_kernel_size=cfg["atd_convffn_kernel_size"],
        mlp_ratio=cfg["atd_mlp_ratio"],
        qkv_bias=True,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        upscale=scale,
        img_range=cfg["atd_img_range"],
        upsampler=cfg["atd_upsampler"],
        resi_connection=cfg["atd_resi_connection"],
    )


def load_baseline_checkpoint(model, checkpoint_path):
    checkpoint = load_torch_file(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    state = strip_module_prefix(state)
    model.load_state_dict(state, strict=True)


def set_zero_dict_residual(model, enabled):
    """设置模型 zero_dict_residual 分析开关，兼容 DataParallel。"""
    net = model.module if hasattr(model, "module") else model
    setattr(net, "zero_dict_residual", bool(enabled))


def autocast_context(device, enabled):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=enabled)
    return contextlib.nullcontext()


def compute_lr_feature_maps(lr):
    """
    中文注释：只从 LR 图像计算 probe 可见的局部特征图。
    返回 gray / edge / lap / hf / structure_lr。
    """
    if lr.shape[1] == 3:
        gray = 0.299 * lr[:, 0:1] + 0.587 * lr[:, 1:2] + 0.114 * lr[:, 2:3]
    else:
        gray = lr.mean(dim=1, keepdim=True)
    dtype = gray.dtype
    device = gray.device
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    lap_kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    edge_x = F.conv2d(gray, sobel_x, padding=1)
    edge_y = F.conv2d(gray, sobel_y, padding=1)
    edge = torch.sqrt(edge_x * edge_x + edge_y * edge_y + 1e-12)
    lap = F.conv2d(gray, lap_kernel, padding=1).abs()
    blur = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
    hf = (gray - blur).abs()
    structure_lr = (torch.log1p(edge) + torch.log1p(lap) + torch.log1p(hf)) / 3.0
    return {
        "lr_gray": gray,
        "lr_edge": edge,
        "lr_lap": lap,
        "lr_hf": hf,
        "structure": structure_lr,
    }


def compute_structure_map(lr):
    # 中文注释：结构强度只从 LR 计算，避免使用 HR/SR 泄漏。
    return compute_lr_feature_maps(lr)["structure"]


def normalize_symmetric_map(arr, percentile=99.0):
    # 中文注释：用正负对称范围归一化 help_map，避免极端值支配颜色。
    arr = np.asarray(arr, dtype=np.float32)
    vmax = np.percentile(np.abs(arr), percentile)
    vmax = max(float(vmax), 1e-8)
    norm = np.clip(arr / vmax, -1.0, 1.0)
    return norm, vmax


def help_to_rgb(help_arr, percentile=99.0):
    # 中文注释：负 help 蓝色，正 help 红色，接近 0 白色。
    norm, vmax = normalize_symmetric_map(help_arr, percentile)
    rgb = np.ones((norm.shape[0], norm.shape[1], 3), dtype=np.float32)
    pos = norm > 0
    neg = norm < 0
    # 中文注释：正 help 越大越红。
    rgb[pos, 1] = 1.0 - norm[pos]
    rgb[pos, 2] = 1.0 - norm[pos]
    # 中文注释：负 help 越大越蓝。
    rgb[neg, 0] = 1.0 + norm[neg]
    rgb[neg, 1] = 1.0 + norm[neg]
    rgb = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    return rgb, vmax


def gray_to_uint8(arr):
    # 中文注释：把任意单通道图归一化为 uint8 灰度图。
    arr = np.asarray(arr, dtype=np.float32)
    lo = float(np.percentile(arr, 1.0))
    hi = float(np.percentile(arr, 99.0))
    if hi <= lo + 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def safe_vis_name(rel_path):
    safe = str(rel_path).replace("\\", "/").strip("/")
    safe = safe.replace("/", "__").replace(":", "_")
    stem, _ = os.path.splitext(safe)
    return stem or "image"


def connected_component_areas(mask):
    mask = np.asarray(mask).astype(bool)
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    areas = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            areas.append(area)
    return areas


def neighbor_agreement_4(values, valid_mask=None):
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("neighbor_agreement_4 expects H,W input")
    if valid_mask is None:
        valid_mask = np.ones_like(values, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask).astype(bool)
    agreements = []
    if values.shape[0] > 1:
        valid_v = valid_mask[:-1, :] & valid_mask[1:, :]
        if np.any(valid_v):
            agreements.append((values[:-1, :][valid_v] == values[1:, :][valid_v]).astype(np.float32))
    if values.shape[1] > 1:
        valid_h = valid_mask[:, :-1] & valid_mask[:, 1:]
        if np.any(valid_h):
            agreements.append((values[:, :-1][valid_h] == values[:, 1:][valid_h]).astype(np.float32))
    if not agreements:
        return None
    return float(np.mean(np.concatenate(agreements)))


def analyze_help_spatial_metrics(help_arr, structure_arr, neg_threshold=0.0, component_min_area=16):
    help_arr = np.asarray(help_arr, dtype=np.float32)
    structure_arr = np.asarray(structure_arr, dtype=np.float32)
    if help_arr.shape != structure_arr.shape:
        raise RuntimeError("help/structure shape mismatch: {} vs {}".format(help_arr.shape, structure_arr.shape))

    sign = np.sign(help_arr)
    help_valid = sign != 0
    help_neighbor_agreement = neighbor_agreement_4(sign, valid_mask=help_valid)
    neg_mask = help_arr < float(neg_threshold)
    neg_neighbor_agreement = neighbor_agreement_4(neg_mask.astype(np.uint8))

    areas = connected_component_areas(neg_mask)
    if areas:
        areas_np = np.asarray(areas, dtype=np.float64)
        neg_pixels = float(np.sum(areas_np))
        large_pixels = float(np.sum(areas_np[areas_np >= int(component_min_area)]))
        area_mean = float(np.mean(areas_np))
        area_p50 = float(np.percentile(areas_np, 50))
        area_p90 = float(np.percentile(areas_np, 90))
        area_max = float(np.max(areas_np))
        large_ratio = large_pixels / max(neg_pixels, 1.0)
    else:
        area_mean = area_p50 = area_p90 = area_max = large_ratio = 0.0

    q2_neg_ratio = q4_neg_ratio = None
    flat_structure = structure_arr.reshape(-1)
    flat_neg = neg_mask.reshape(-1)
    if flat_structure.size >= 4 and float(np.std(flat_structure)) > 1e-8:
        q25, q50, q75 = np.percentile(flat_structure, [25, 50, 75])
        q2_mask = (flat_structure >= q25) & (flat_structure < q50)
        q4_mask = flat_structure >= q75
        if np.any(q2_mask):
            q2_neg_ratio = float(np.mean(flat_neg[q2_mask]))
        if np.any(q4_mask):
            q4_neg_ratio = float(np.mean(flat_neg[q4_mask]))

    return {
        "help_neighbor_agreement_4": help_neighbor_agreement,
        "neg_neighbor_agreement_4": neg_neighbor_agreement,
        "neg_component_count": int(len(areas)),
        "neg_component_area_mean": area_mean,
        "neg_component_area_p50": area_p50,
        "neg_component_area_p90": area_p90,
        "neg_component_area_max": area_max,
        "neg_component_large_ratio": large_ratio,
        "q2_neg_ratio": q2_neg_ratio,
        "q4_neg_ratio": q4_neg_ratio,
    }


def save_help_map_artifacts(save_dir, rel_path, help_arr, structure_arr, base_gray_arr,
                            neg_threshold=0.0, percentile=99.0):
    out_dir = os.path.join(save_dir, "help_maps")
    os.makedirs(out_dir, exist_ok=True)
    safe_name = safe_vis_name(rel_path)
    help_arr = np.asarray(help_arr, dtype=np.float32)
    structure_arr = np.asarray(structure_arr, dtype=np.float32)
    np.save(os.path.join(out_dir, "{}_help.npy".format(safe_name)), help_arr)
    np.save(os.path.join(out_dir, "{}_structure.npy".format(safe_name)), structure_arr)

    help_rgb, _ = help_to_rgb(help_arr, percentile=percentile)
    Image.fromarray(help_rgb).save(os.path.join(out_dir, "{}_help_vis.png".format(safe_name)))

    neg_mask = (help_arr < float(neg_threshold)).astype(np.uint8) * 255
    Image.fromarray(neg_mask, mode="L").save(os.path.join(out_dir, "{}_neg_mask.png".format(safe_name)))

    structure_gray = gray_to_uint8(structure_arr)
    Image.fromarray(structure_gray, mode="L").save(os.path.join(out_dir, "{}_structure_vis.png".format(safe_name)))

    if base_gray_arr is not None:
        base_gray = gray_to_uint8(base_gray_arr)
        overlay = np.stack([base_gray, base_gray, base_gray], axis=-1).astype(np.float32)
        neg = neg_mask > 0
        overlay[neg, 0] = overlay[neg, 0] * 0.35
        overlay[neg, 1] = overlay[neg, 1] * 0.55
        overlay[neg, 2] = 255.0
        Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(
            os.path.join(out_dir, "{}_overlay.png".format(safe_name))
        )


def crop_border_tensor(tensor, border):
    if border <= 0:
        return tensor
    if tensor.shape[-2] <= 2 * border or tensor.shape[-1] <= 2 * border:
        return tensor[..., 0:0, 0:0]
    return tensor[..., border:-border, border:-border]


def ensure_same_hw(sr, gt, rel_path):
    if sr.shape[-2:] != gt.shape[-2:]:
        raise RuntimeError(
            "SR/GT size mismatch for {}: SR {} vs GT {}".format(
                rel_path, tuple(sr.shape[-2:]), tuple(gt.shape[-2:])
            )
        )


def patch_records_for_image(dataset, scene, rel_path, help_map, structure_hr,
                            err_full, err_nodict, patch_size, lr_feature_maps_hr=None):
    _, _, h, w = help_map.shape
    h_eff = (h // patch_size) * patch_size
    w_eff = (w // patch_size) * patch_size
    if h_eff <= 0 or w_eff <= 0:
        return []
    help_map = help_map[..., :h_eff, :w_eff]
    structure_hr = structure_hr[..., :h_eff, :w_eff]
    err_full = err_full[..., :h_eff, :w_eff]
    err_nodict = err_nodict[..., :h_eff, :w_eff]
    lr_feature_maps_hr = lr_feature_maps_hr or {}
    lr_feature_maps_hr = {
        key: value[..., :h_eff, :w_eff]
        for key, value in lr_feature_maps_hr.items()
        if value is not None
    }
    rows = []

    def patch_np(name, y0, x0):
        tensor = lr_feature_maps_hr.get(name, None)
        if tensor is None:
            return None
        return tensor[..., y0:y0 + patch_size, x0:x0 + patch_size].reshape(-1).detach().float().cpu().numpy()

    for y in range(0, h_eff, patch_size):
        for x in range(0, w_eff, patch_size):
            hp = help_map[..., y:y + patch_size, x:x + patch_size].reshape(-1).detach().float().cpu().numpy()
            sp = structure_hr[..., y:y + patch_size, x:x + patch_size].reshape(-1).detach().float().cpu().numpy()
            ef = err_full[..., y:y + patch_size, x:x + patch_size].reshape(-1).detach().float().cpu().numpy()
            en = err_nodict[..., y:y + patch_size, x:x + patch_size].reshape(-1).detach().float().cpu().numpy()
            gray = patch_np("lr_gray", y, x)
            edge = patch_np("lr_edge", y, x)
            lap = patch_np("lr_lap", y, x)
            hf = patch_np("lr_hf", y, x)
            row = {
                "dataset": dataset,
                "scene": scene,
                "image_rel_path": rel_path,
                "patch_y": int(y),
                "patch_x": int(x),
                "structure": float(np.mean(sp)),
                "help_mean": float(np.mean(hp)),
                "help_std": float(np.std(hp)),
                "help_p10": float(np.percentile(hp, 10)),
                "help_median": float(np.percentile(hp, 50)),
                "neg_ratio": float(np.mean(hp < 0.0)),
                "abs_full_mean": float(np.mean(ef)),
                "abs_nodict_mean": float(np.mean(en)),
                "patch_y_norm": float(y / max(1, h_eff)),
                "patch_x_norm": float(x / max(1, w_eff)),
                "structure_std": float(np.std(sp)),
                "structure_p90": float(np.percentile(sp, 90)),
                "structure_max": float(np.max(sp)),
            }
            if gray is not None:
                row.update({
                    "lr_mean": float(np.mean(gray)),
                    "lr_std": float(np.std(gray)),
                    "lr_min": float(np.min(gray)),
                    "lr_max": float(np.max(gray)),
                    "lr_gray_mean": float(np.mean(gray)),
                    "lr_gray_std": float(np.std(gray)),
                    "lr_gray_p10": float(np.percentile(gray, 10)),
                    "lr_gray_p90": float(np.percentile(gray, 90)),
                })
            if edge is not None:
                row.update({
                    "lr_edge_mean": float(np.mean(edge)),
                    "lr_edge_std": float(np.std(edge)),
                    "lr_edge_p90": float(np.percentile(edge, 90)),
                })
            if lap is not None:
                row.update({
                    "lr_lap_mean": float(np.mean(lap)),
                    "lr_lap_std": float(np.std(lap)),
                    "lr_lap_p90": float(np.percentile(lap, 90)),
                })
            if hf is not None:
                row.update({
                    "lr_hf_mean": float(np.mean(hf)),
                    "lr_hf_std": float(np.std(hf)),
                    "lr_hf_p90": float(np.percentile(hf, 90)),
                })
            rows.append(row)
    return rows


def format_rows(rows, fields):
    formatted = []
    for row in rows:
        item = {}
        for key in fields:
            value = row.get(key, "")
            if isinstance(value, float):
                item[key] = fmt(value)
            else:
                item[key] = value
        formatted.append(item)
    return formatted


def mean_optional(rows, key):
    values = [row.get(key, None) for row in rows]
    values = [float(value) for value in values if value is not None and value != ""]
    if not values:
        return None
    return float(np.mean(values))


def summarize_bins(patch_rows):
    scores = np.asarray([row["structure"] for row in patch_rows], dtype=np.float64)
    names = ["Q1_smooth_or_natural", "Q2_medium_low", "Q3_medium_high", "Q4_high_structure"]
    order = np.argsort(scores)
    labels = [""] * len(patch_rows)
    for rank, idx in enumerate(order):
        labels[int(idx)] = names[min(3, int(rank * 4 / max(1, len(patch_rows))))]
    out = []
    for name in names:
        rows = [patch_rows[idx] for idx, label in enumerate(labels) if label == name]
        if not rows:
            continue
        helps = np.asarray([row["help_mean"] for row in rows], dtype=np.float64)
        out.append({
            "bin": name,
            "N": len(rows),
            "structure_mean": float(np.mean([row["structure"] for row in rows])),
            "help_mean": float(np.mean(helps)),
            "help_std": float(np.std(helps)),
            "help_p10": float(np.percentile(helps, 10)),
            "help_median": float(np.percentile(helps, 50)),
            "neg_patch_ratio": float(np.mean(helps < 0.0)),
            "pixel_neg_ratio_mean": float(np.mean([row["neg_ratio"] for row in rows])),
            "abs_full_mean": float(np.mean([row["abs_full_mean"] for row in rows])),
            "abs_nodict_mean": float(np.mean([row["abs_nodict_mean"] for row in rows])),
        })
    return out


def make_decision(q4_neg_patch_ratio, q4_pixel_neg_ratio_mean, corr_non_smooth):
    if corr_non_smooth is None:
        return "MIXED"
    if (q4_neg_patch_ratio >= 0.15 or q4_pixel_neg_ratio_mean >= 0.15) and corr_non_smooth < 0.3:
        return "GO"
    if q4_neg_patch_ratio < 0.05 and corr_non_smooth > 0.6:
        return "NO-GO"
    return "MIXED"


def main():
    args = parse_args()
    lr_root = os.path.join(args.dataset_root, "LR")
    gt_root = os.path.join(args.dataset_root, "GT")
    if not os.path.isdir(lr_root) or not os.path.isdir(gt_root):
        raise FileNotFoundError("Expected LR/ and GT/ under {}".format(args.dataset_root))

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")
    device = requested_device
    amp_enabled = bool(args.amp) and device.type == "cuda"
    border = int(args.scale if args.crop_border is None else args.crop_border)
    patch_size = int(args.patch_size_hr)
    if patch_size <= 0:
        raise ValueError("--patch_size_hr must be positive")
    component_min_area = max(1, int(args.component_min_area))

    model = build_baseline_model(args.scale)
    load_baseline_checkpoint(model, args.checkpoint)
    model = model.to(device)
    model.eval()

    lr_paths = discover_lr_images(lr_root)
    if args.max_images and args.max_images > 0:
        lr_paths = lr_paths[:int(args.max_images)]
    if not lr_paths:
        raise RuntimeError("No LR images found under {}".format(lr_root))

    patch_rows = []
    per_image_rows = []
    saved_help_count = 0
    save_help_limit = int(args.save_help_max_images)
    for idx, lr_path in enumerate(lr_paths, start=1):
        gt_path = find_gt_for_lr(lr_path, lr_root, gt_root)
        if gt_path is None:
            print("[Warning] Missing GT for {}".format(lr_path))
            continue
        scene, rel_path = scene_and_rel_path(lr_path, lr_root, folder="")
        lr = load_rgb_tensor(lr_path).unsqueeze(0).to(device)
        gt = load_rgb_tensor(gt_path).unsqueeze(0).to(device)

        with torch.no_grad():
            with autocast_context(device, amp_enabled):
                set_zero_dict_residual(model, False)
                sr_full = model(lr)
                set_zero_dict_residual(model, True)
                sr_nodict = model(lr)
                set_zero_dict_residual(model, False)
            sr_full = sr_full.detach().float().clamp(0.0, 1.0)
            sr_nodict = sr_nodict.detach().float().clamp(0.0, 1.0)
            gt = gt.detach().float().clamp(0.0, 1.0)
            ensure_same_hw(sr_full, gt, rel_path)
            ensure_same_hw(sr_nodict, gt, rel_path)
            err_full = torch.mean(torch.abs(sr_full - gt), dim=1, keepdim=True)
            err_nodict = torch.mean(torch.abs(sr_nodict - gt), dim=1, keepdim=True)
            help_map = err_nodict - err_full
            lr_feature_maps = compute_lr_feature_maps(lr.detach().float().clamp(0.0, 1.0))
            lr_feature_maps_hr = {
                key: F.interpolate(
                    value,
                    size=help_map.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                for key, value in lr_feature_maps.items()
            }
            structure_hr = lr_feature_maps_hr["structure"]
            help_map = crop_border_tensor(help_map, border)
            lr_feature_maps_hr = {
                key: crop_border_tensor(value, border)
                for key, value in lr_feature_maps_hr.items()
            }
            structure_hr = lr_feature_maps_hr["structure"]
            err_full = crop_border_tensor(err_full, border)
            err_nodict = crop_border_tensor(err_nodict, border)
            sr_full_gray = torch.mean(sr_full, dim=1, keepdim=True)
            sr_full_gray = crop_border_tensor(sr_full_gray, border)
            if help_map.shape[-2] < patch_size or help_map.shape[-1] < patch_size:
                print("[Warning] Skip too-small image after crop: {}".format(rel_path))
                continue

            help_arr = help_map.squeeze(0).squeeze(0).detach().float().cpu().numpy().astype(np.float32)
            structure_arr = structure_hr.squeeze(0).squeeze(0).detach().float().cpu().numpy().astype(np.float32)
            base_gray_arr = sr_full_gray.squeeze(0).squeeze(0).detach().float().cpu().numpy().astype(np.float32)
            spatial_metrics = {}
            if bool(args.analyze_help_spatial):
                spatial_metrics = analyze_help_spatial_metrics(
                    help_arr,
                    structure_arr,
                    neg_threshold=float(args.help_neg_threshold),
                    component_min_area=component_min_area,
                )
            if bool(args.save_help_maps) and (save_help_limit == 0 or saved_help_count < save_help_limit):
                save_help_map_artifacts(
                    args.save_dir,
                    rel_path,
                    help_arr,
                    structure_arr,
                    base_gray_arr,
                    neg_threshold=float(args.help_neg_threshold),
                    percentile=float(args.help_vis_percentile),
                )
                saved_help_count += 1

            image_patch_rows = patch_records_for_image(
                args.dataset_name,
                scene,
                rel_path,
                help_map,
                structure_hr,
                err_full,
                err_nodict,
                patch_size,
                lr_feature_maps_hr=lr_feature_maps_hr,
            )
            patch_rows.extend(image_patch_rows)
            image_row = {
                "dataset": args.dataset_name,
                "scene": scene,
                "image_rel_path": rel_path,
                "num_patches": len(image_patch_rows),
                "help_mean": float(help_map.mean().detach().cpu()),
                "neg_ratio": float((help_map < float(args.help_neg_threshold)).float().mean().detach().cpu()),
                "abs_full_mean": float(err_full.mean().detach().cpu()),
                "abs_nodict_mean": float(err_nodict.mean().detach().cpu()),
            }
            image_row.update(spatial_metrics)
            per_image_rows.append(image_row)
        if idx % 20 == 0 or idx == len(lr_paths):
            print("Processed {}/{} images".format(idx, len(lr_paths)))

    set_zero_dict_residual(model, False)
    if not patch_rows:
        raise RuntimeError("No valid patches produced.")

    scores = np.asarray([row["structure"] for row in patch_rows], dtype=np.float64)
    helps = np.asarray([row["help_mean"] for row in patch_rows], dtype=np.float64)
    pixel_neg = np.asarray([row["neg_ratio"] for row in patch_rows], dtype=np.float64)
    corr_all = corrcoef_np(scores, helps)
    thr_q1 = float(np.percentile(scores, 25))
    non_smooth = scores > thr_q1
    corr_non_smooth = corrcoef_np(scores[non_smooth], helps[non_smooth])
    thr_q4 = float(np.percentile(scores, 75))
    q4 = scores >= thr_q4
    corr_q4_only = corrcoef_np(scores[q4], helps[q4])
    q4_help_mean = float(np.mean(helps[q4]))
    q4_help_p10 = float(np.percentile(helps[q4], 10))
    q4_neg_patch_ratio = float(np.mean(helps[q4] < 0.0))
    q4_pixel_neg_ratio_mean = float(np.mean(pixel_neg[q4]))
    decision = make_decision(q4_neg_patch_ratio, q4_pixel_neg_ratio_mean, corr_non_smooth)

    bin_rows = summarize_bins(patch_rows)
    summary_row = {
        "dataset": args.dataset_name,
        "num_images": len(per_image_rows),
        "num_patches": len(patch_rows),
        "patch_size_hr": patch_size,
        "help_mean": float(np.mean(helps)),
        "help_std": float(np.std(helps)),
        "help_p10": float(np.percentile(helps, 10)),
        "help_median": float(np.percentile(helps, 50)),
        "neg_patch_ratio": float(np.mean(helps < 0.0)),
        "pixel_neg_ratio_mean": float(np.mean(pixel_neg)),
        "corr_all": corr_all,
        "corr_non_smooth": corr_non_smooth,
        "corr_q4_only": corr_q4_only,
        "q4_help_mean": q4_help_mean,
        "q4_help_p10": q4_help_p10,
        "q4_neg_patch_ratio": q4_neg_patch_ratio,
        "q4_pixel_neg_ratio_mean": q4_pixel_neg_ratio_mean,
        "decision": decision,
    }
    spatial_summary = {
        "help_neighbor_agreement_4_mean": mean_optional(per_image_rows, "help_neighbor_agreement_4"),
        "neg_neighbor_agreement_4_mean": mean_optional(per_image_rows, "neg_neighbor_agreement_4"),
        "neg_component_count_mean": mean_optional(per_image_rows, "neg_component_count"),
        "neg_component_area_mean": mean_optional(per_image_rows, "neg_component_area_mean"),
        "neg_component_area_p90_mean": mean_optional(per_image_rows, "neg_component_area_p90"),
        "neg_component_area_max_mean": mean_optional(per_image_rows, "neg_component_area_max"),
        "neg_component_large_ratio_mean": mean_optional(per_image_rows, "neg_component_large_ratio"),
        "q2_neg_ratio_mean": mean_optional(per_image_rows, "q2_neg_ratio"),
        "q4_neg_ratio_mean": mean_optional(per_image_rows, "q4_neg_ratio"),
    }
    summary_row.update(spatial_summary)

    os.makedirs(args.save_dir, exist_ok=True)
    patch_fields = [
        "dataset", "scene", "image_rel_path", "patch_y", "patch_x", "structure",
        "help_mean", "help_std", "help_p10", "help_median", "neg_ratio",
        "abs_full_mean", "abs_nodict_mean",
        "lr_mean", "lr_std", "lr_min", "lr_max",
        "lr_gray_mean", "lr_gray_std", "lr_gray_p10", "lr_gray_p90",
        "lr_edge_mean", "lr_edge_std", "lr_edge_p90",
        "lr_lap_mean", "lr_lap_std", "lr_lap_p90",
        "lr_hf_mean", "lr_hf_std", "lr_hf_p90",
        "structure_std", "structure_p90", "structure_max",
        "patch_y_norm", "patch_x_norm",
    ]
    per_image_fields = [
        "dataset", "scene", "image_rel_path", "num_patches", "help_mean",
        "neg_ratio", "abs_full_mean", "abs_nodict_mean",
        "help_neighbor_agreement_4", "neg_neighbor_agreement_4",
        "neg_component_count", "neg_component_area_mean", "neg_component_area_p50",
        "neg_component_area_p90", "neg_component_area_max", "neg_component_large_ratio",
        "q2_neg_ratio", "q4_neg_ratio",
    ]
    bin_fields = [
        "bin", "N", "structure_mean", "help_mean", "help_std", "help_p10",
        "help_median", "neg_patch_ratio", "pixel_neg_ratio_mean",
        "abs_full_mean", "abs_nodict_mean",
    ]
    summary_fields = [
        "dataset", "num_images", "num_patches", "patch_size_hr", "help_mean",
        "help_std", "help_p10", "help_median", "neg_patch_ratio",
        "pixel_neg_ratio_mean", "corr_all", "corr_non_smooth", "corr_q4_only",
        "q4_help_mean", "q4_help_p10", "q4_neg_patch_ratio",
        "q4_pixel_neg_ratio_mean", "decision",
        "help_neighbor_agreement_4_mean", "neg_neighbor_agreement_4_mean",
        "neg_component_count_mean", "neg_component_area_mean",
        "neg_component_area_p90_mean", "neg_component_area_max_mean",
        "neg_component_large_ratio_mean", "q2_neg_ratio_mean", "q4_neg_ratio_mean",
    ]

    patch_csv = os.path.join(args.save_dir, "dict_help_patch_rows.csv")
    per_image_csv = os.path.join(args.save_dir, "dict_help_per_image.csv")
    bin_csv = os.path.join(args.save_dir, "dict_help_bin_summary.csv")
    summary_csv = os.path.join(args.save_dir, "dict_help_dataset_summary.csv")
    if bool(args.save_patch_csv):
        write_csv_rows(patch_csv, patch_fields, format_rows(patch_rows, patch_fields), append=False)
    if bool(args.save_per_image_csv):
        write_csv_rows(per_image_csv, per_image_fields, format_rows(per_image_rows, per_image_fields), append=False)
    write_csv_rows(bin_csv, bin_fields, format_rows(bin_rows, bin_fields), append=False)
    write_csv_rows(summary_csv, summary_fields, format_rows([summary_row], summary_fields), append=False)

    decision_txt = os.path.join(args.save_dir, "decision.txt")
    with open(decision_txt, "w", encoding="utf-8") as f:
        f.write("dataset: {}\n".format(args.dataset_name))
        f.write("num_images: {}\n".format(len(per_image_rows)))
        f.write("num_patches: {}\n".format(len(patch_rows)))
        f.write("patch_size_hr: {}\n".format(patch_size))
        f.write("help_mean: {}\n".format(fmt(summary_row["help_mean"])))
        f.write("neg_patch_ratio: {}\n".format(fmt(summary_row["neg_patch_ratio"])))
        f.write("pixel_neg_ratio_mean: {}\n".format(fmt(summary_row["pixel_neg_ratio_mean"])))
        f.write("corr_all: {}\n".format(fmt(corr_all)))
        f.write("corr_non_smooth: {}\n".format(fmt(corr_non_smooth)))
        f.write("corr_q4_only: {}\n".format(fmt(corr_q4_only)))
        f.write("q4_help_mean: {}\n".format(fmt(q4_help_mean)))
        f.write("q4_help_p10: {}\n".format(fmt(q4_help_p10)))
        f.write("q4_neg_patch_ratio: {}\n".format(fmt(q4_neg_patch_ratio)))
        f.write("q4_pixel_neg_ratio_mean: {}\n".format(fmt(q4_pixel_neg_ratio_mean)))
        f.write("help_neighbor_agreement_4_mean: {}\n".format(fmt(summary_row["help_neighbor_agreement_4_mean"])))
        f.write("neg_neighbor_agreement_4_mean: {}\n".format(fmt(summary_row["neg_neighbor_agreement_4_mean"])))
        f.write("neg_component_area_mean: {}\n".format(fmt(summary_row["neg_component_area_mean"])))
        f.write("neg_component_area_p90_mean: {}\n".format(fmt(summary_row["neg_component_area_p90_mean"])))
        f.write("neg_component_large_ratio_mean: {}\n".format(fmt(summary_row["neg_component_large_ratio_mean"])))
        f.write("q2_neg_ratio_mean: {}\n".format(fmt(summary_row["q2_neg_ratio_mean"])))
        f.write("q4_neg_ratio_mean: {}\n".format(fmt(summary_row["q4_neg_ratio_mean"])))
        f.write("decision: {}\n".format(decision))
        f.write("\n")
        f.write("GO if Q4 neg_patch_ratio >= 0.15 or Q4 pixel_neg_ratio_mean >= 0.15 and corr_non_smooth < 0.3\n")
        f.write("NO-GO if Q4 neg_patch_ratio < 0.05 and corr_non_smooth > 0.6\n")
        f.write("MIXED otherwise\n")
        f.write("This rule is a diagnostic aid, not an absolute conclusion.\n")

    print("Dataset: {}".format(args.dataset_name))
    print("Images: {}".format(len(per_image_rows)))
    print("Patches: {}".format(len(patch_rows)))
    print("Help mean: {}".format(fmt(summary_row["help_mean"])))
    print("Neg patch ratio: {}".format(fmt(summary_row["neg_patch_ratio"])))
    print("corr_all: {}".format(fmt(corr_all)))
    print("corr_non_smooth: {}".format(fmt(corr_non_smooth)))
    print("corr_q4_only: {}".format(fmt(corr_q4_only)))
    print("Q4 help mean: {}".format(fmt(q4_help_mean)))
    print("Q4 help p10: {}".format(fmt(q4_help_p10)))
    print("Q4 neg patch ratio: {}".format(fmt(q4_neg_patch_ratio)))
    print("Q4 pixel neg ratio mean: {}".format(fmt(q4_pixel_neg_ratio_mean)))
    print("Help sign neighbor agreement: {}".format(fmt(summary_row["help_neighbor_agreement_4_mean"])))
    print("Negative mask neighbor agreement: {}".format(fmt(summary_row["neg_neighbor_agreement_4_mean"])))
    print("Negative component area mean: {}".format(fmt(summary_row["neg_component_area_mean"])))
    print("Negative component area p90 mean: {}".format(fmt(summary_row["neg_component_area_p90_mean"])))
    print("Negative component large ratio: {}".format(fmt(summary_row["neg_component_large_ratio_mean"])))
    print("Q2 pixel neg ratio: {}".format(fmt(summary_row["q2_neg_ratio_mean"])))
    print("Q4 pixel neg ratio: {}".format(fmt(summary_row["q4_neg_ratio_mean"])))
    print("Decision: {}".format(decision))
    if bool(args.save_patch_csv):
        print("Saved patch CSV: {}".format(patch_csv))
    if bool(args.save_per_image_csv):
        print("Saved per-image CSV: {}".format(per_image_csv))
    print("Saved bin summary: {}".format(bin_csv))
    print("Saved dataset summary: {}".format(summary_csv))
    print("Saved decision TXT: {}".format(decision_txt))
    if bool(args.save_help_maps):
        print("Saved help maps: {}".format(os.path.join(args.save_dir, "help_maps")))


if __name__ == "__main__":
    main()
