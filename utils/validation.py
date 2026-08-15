import os
import time

import torch

from utils.image_utils import find_matching_gt, is_image_file, list_images, load_rgb_tensor, save_tensor_image
from utils.logger import build_progress_bar
from utils.metrics import calc_psnr, calc_ssim
from utils.scene_gate_utils import scene_and_rel_path, write_csv_rows  # 中文注释：复用 Scene Gate 的路径解析和 CSV 写入。


AID_CLASS_NAMES = [
    "Airport", "BareLand", "BaseballField", "Beach", "Bridge", "Center", "Church",
    "Commercial", "DenseResidential", "Desert", "Farmland", "Forest", "Industrial", "Meadow",
    "MediumResidential", "Mountain", "Park", "Parking", "Playground", "Pond", "Port",
    "RailwayStation", "Resort", "River", "School", "SparseResidential", "Square", "Stadium",
    "StorageTanks", "Viaduct",
]


def compute_structure_csv_values_from_lr(lr_tensor):
    """从当前 LR 图像计算 CSV 用固定结构统计，供 DECB-only 验证使用。"""
    # 中文注释：预置空字段，失败时保持兼容旧 CSV。
    values = {
        "struct_edge_density": "",
        "struct_edge_top25_mean": "",
        "struct_lap_var": "",
        "struct_grad_coherence": "",
        "struct_hf_energy": "",
    }
    # 中文注释：只取当前 batch 第 0 张，validation 通常单图 forward。
    lr = lr_tensor.detach().float()[0:1].clamp(0.0, 1.0)
    # 中文注释：按 RGB 权重转灰度；非 RGB 时退化为均值。
    if lr.shape[1] == 3:
        gray = 0.299 * lr[:, 0:1, :, :] + 0.587 * lr[:, 1:2, :, :] + 0.114 * lr[:, 2:3, :, :]
    else:
        gray = lr.mean(dim=1, keepdim=True)
    # 中文注释：构造固定 Sobel x 卷积核。
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=gray.device, dtype=gray.dtype).view(1, 1, 3, 3)
    # 中文注释：构造固定 Sobel y 卷积核。
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=gray.device, dtype=gray.dtype).view(1, 1, 3, 3)
    # 中文注释：构造固定 Laplacian 卷积核。
    lap_kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=gray.device, dtype=gray.dtype).view(1, 1, 3, 3)
    # 中文注释：计算 x/y 梯度。
    gx = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
    gy = torch.nn.functional.conv2d(gray, sobel_y, padding=1)
    # 中文注释：计算梯度幅值。
    mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
    # 中文注释：edge_density 使用梯度幅值均值。
    edge_density = mag.mean()
    # 中文注释：top-25% 梯度均值刻画强边缘结构。
    mag_flat = mag.flatten(1)
    # 中文注释：至少取 1 个像素，避免极小图像 topk 为 0。
    topk = max(1, int(0.25 * mag_flat.shape[1]))
    # 中文注释：沿像素维取 top-25% 梯度。
    edge_top25_mean = mag_flat.topk(topk, dim=1).values.mean()
    # 中文注释：计算 Laplacian 响应。
    lap = torch.nn.functional.conv2d(gray, lap_kernel, padding=1)
    # 中文注释：Laplacian 方差。
    lap_var = lap.flatten(1).var(dim=1, unbiased=False).mean()
    # 中文注释：结构张量 Jxx。
    jxx = (gx * gx).mean()
    # 中文注释：结构张量 Jyy。
    jyy = (gy * gy).mean()
    # 中文注释：结构张量 Jxy。
    jxy = (gx * gy).mean()
    # 中文注释：方向一致性。
    grad_coherence = torch.sqrt((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy + 1e-6) / (jxx + jyy + 1e-6)
    # 中文注释：3x3 平均池化估计低频背景。
    blur = torch.nn.functional.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
    # 中文注释：高频能量。
    hf_energy = (gray - blur).abs().mean()
    # 中文注释：格式化为 CSV 字符串。
    values["struct_edge_density"] = "{:.6f}".format(float(edge_density.cpu()))
    values["struct_edge_top25_mean"] = "{:.6f}".format(float(edge_top25_mean.cpu()))
    values["struct_lap_var"] = "{:.6f}".format(float(lap_var.cpu()))
    values["struct_grad_coherence"] = "{:.6f}".format(float(grad_coherence.cpu()))
    values["struct_hf_energy"] = "{:.6f}".format(float(hf_energy.cpu()))
    # 中文注释：返回可并入 CSV 的结构统计字段。
    return values


def autocast_context(opt, device):
    enabled = bool(opt.amp and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast(device_type=device.type, enabled=enabled)
        except TypeError:
            return torch.amp.autocast(device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def unwrap_prediction(output):
    if isinstance(output, dict):
        return output.get("sr", output.get("final_sr", output))
    if isinstance(output, (list, tuple)):
        return output[0]
    return output


def extract_struct_csv_values(model):
    """从模型 struct_debug_cache 提取当前图像的 SCDRC 调试字段。"""
    # 中文注释：兼容 DataParallel，拿到真实模型。
    net = model.module if hasattr(model, "module") else model
    # 中文注释：读取最近一次 forward 的 SCDRC cache；关闭时通常为空。
    cache = getattr(net, "struct_debug_cache", {})
    # 中文注释：预置空字段，保证 CSV 表头兼容开启/关闭两种模式。
    values = {
        "struct_delta": "",
        "structure_score": "",
        "struct_edge_density": "",
        "struct_edge_top25_mean": "",
        "struct_lap_var": "",
        "struct_grad_coherence": "",
        "struct_hf_energy": "",
        "struct_delta_spatial_std": "",
        "struct_delta_spatial_min": "",
        "struct_delta_spatial_max": "",
        "struct_res_abs_mean": "",
        "struct_res_abs_std": "",
        "struct_res_to_feat_ratio": "",
    }
    # 中文注释：没有 cache 时返回空字段。
    if not cache:
        return values
    # 中文注释：安全读取 delta/score/vec。
    delta = cache.get("delta", None)
    score = cache.get("structure_score", None)
    struct_vec = cache.get("struct_vec", None)
    # 中文注释：SCDRC-RR 额外缓存 residual reliability stats，普通 SCDRC 时通常为 None。
    residual_vec = cache.get("residual_vec", None)
    # 中文注释：SCDRC-Local 额外缓存 local delta map 的空间统计。
    delta_std_spatial = cache.get("delta_std_spatial", None)
    # 中文注释：SCDRC-Local 额外缓存 local delta map 的空间最小值。
    delta_min_spatial = cache.get("delta_min_spatial", None)
    # 中文注释：SCDRC-Local 额外缓存 local delta map 的空间最大值。
    delta_max_spatial = cache.get("delta_max_spatial", None)
    # 中文注释：validation 通常单图 forward，取 batch 第 0 张。
    if delta is not None and delta.numel() > 0:
        values["struct_delta"] = "{:.6f}".format(float(delta.detach().float().view(-1)[0].cpu()))
    # 中文注释：structure_score 同样取第 0 张。
    if score is not None and score.numel() > 0:
        values["structure_score"] = "{:.6f}".format(float(score.detach().float().view(-1)[0].cpu()))
    # 中文注释：struct_vec 字段顺序固定：edge_density, edge_top25_mean, lap_var, grad_coherence, hf_energy。
    if struct_vec is not None and struct_vec.numel() >= 5:
        vec = struct_vec.detach().float().view(struct_vec.shape[0], -1)[0].cpu()
        values["struct_edge_density"] = "{:.6f}".format(float(vec[0]))
        values["struct_edge_top25_mean"] = "{:.6f}".format(float(vec[1]))
        values["struct_lap_var"] = "{:.6f}".format(float(vec[2]))
        values["struct_grad_coherence"] = "{:.6f}".format(float(vec[3]))
        values["struct_hf_energy"] = "{:.6f}".format(float(vec[4]))
    # 中文注释：local delta map 空间标准差，validation 通常单图 forward，取第 0 张。
    if delta_std_spatial is not None and delta_std_spatial.numel() > 0:
        values["struct_delta_spatial_std"] = "{:.6f}".format(float(delta_std_spatial.detach().float().view(-1)[0].cpu()))
    # 中文注释：local delta map 空间最小值，validation 通常单图 forward，取第 0 张。
    if delta_min_spatial is not None and delta_min_spatial.numel() > 0:
        values["struct_delta_spatial_min"] = "{:.6f}".format(float(delta_min_spatial.detach().float().view(-1)[0].cpu()))
    # 中文注释：local delta map 空间最大值，validation 通常单图 forward，取第 0 张。
    if delta_max_spatial is not None and delta_max_spatial.numel() > 0:
        values["struct_delta_spatial_max"] = "{:.6f}".format(float(delta_max_spatial.detach().float().view(-1)[0].cpu()))
    # 中文注释：residual_vec 字段顺序固定：res_abs_mean, res_abs_std, res_to_feat_ratio。
    if residual_vec is not None and residual_vec.numel() >= 3:
        res_vec = residual_vec.detach().float().view(residual_vec.shape[0], -1)[0].cpu()
        values["struct_res_abs_mean"] = "{:.6f}".format(float(res_vec[0]))
        values["struct_res_abs_std"] = "{:.6f}".format(float(res_vec[1]))
        values["struct_res_to_feat_ratio"] = "{:.6f}".format(float(res_vec[2]))
    # 中文注释：返回可直接并入 per-image CSV 的字段。
    return values


def extract_decb_csv_values(model):
    """从模型 decb_debug_cache 提取当前图像的 DECB 调试字段。"""
    # 中文注释：兼容 DataParallel，拿到真实模型。
    net = model.module if hasattr(model, "module") else model
    # 中文注释：读取最近一次 forward 的 DECB cache；关闭时通常为空。
    cache = getattr(net, "decb_debug_cache", {})
    # 中文注释：预置空字段，保证 CSV 表头兼容开启/关闭两种模式。
    values = {
        "decb_gate_mean": "",
        "decb_gate_std": "",
        "decb_gate_min": "",
        "decb_gate_max": "",
        "decb_err_abs_mean": "",
        "decb_attn_entropy": "",
        "decb_residual_abs_mean": "",
    }
    # 中文注释：没有 cache 时返回空字段。
    if not cache:
        return values
    # 中文注释：CSV 字段到 cache key 的映射。
    field_to_key = {
        "decb_gate_mean": "gate_mean",
        "decb_gate_std": "gate_std",
        "decb_gate_min": "gate_min",
        "decb_gate_max": "gate_max",
        "decb_err_abs_mean": "err_abs_mean",
        "decb_attn_entropy": "attn_entropy",
        "decb_residual_abs_mean": "residual_abs_mean",
    }
    # 中文注释：validation 通常单图 forward，取 batch 第 0 张。
    for field, key in field_to_key.items():
        tensor = cache.get(key, None)
        if tensor is not None and tensor.numel() > 0:
            values[field] = "{:.6f}".format(float(tensor.detach().float().view(-1)[0].cpu()))
    # 中文注释：返回可直接并入 per-image CSV 的字段。
    return values


def extract_radr_csv_values(model):
    """从模型 radr_debug_cache 提取当前图像的 RADR 调试字段。"""
    net = model.module if hasattr(model, "module") else model
    cache = getattr(net, "radr_debug_cache", {})
    values = {
        "radr_u_mean": "",
        "radr_u_std": "",
        "radr_u_min": "",
        "radr_u_max": "",
        "radr_u_eff_mean": "",
        "radr_u_eff_std": "",
        "radr_u_eff_min": "",
        "radr_u_eff_max": "",
        "radr_tau": "",
        "radr_lambda": "",
        "radr_suppression_mean": "",
        "radr_corr_enabled": "",
        "radr_corr_lambda": "",
        "radr_corr_abs_mean": "",
        "radr_corr_abs_std": "",
        "radr_corr_gated_abs_mean": "",
        "radr_corr_gated_abs_std": "",
        "radr_corr_gate_mode_id": "",
        "radr_corr_gate_mean": "",
        "radr_corr_gate_std": "",
        "radr_corr_gate_min": "",
        "radr_corr_gate_max": "",
        "radr_corr_feature_mode_id": "",
        "radr_corr_context_abs_mean": "",
        "radr_corr_context_abs_std": "",
        "radr_ccd_delta_pred_abs": "",
        "radr_ccd_num_delta": "",
        "radr_lch_u_mean": "",
        "radr_lch_gate_mean": "",
        "radr_lch_corr_abs_mean": "",
        "radr_lch_gated_corr_abs_mean": "",
        "radr_lch_lambda": "",
        "scdr_alpha": "",
        "scdr_route_a_abs_mean": "",
        "scdr_route_b_abs_mean": "",
        "scdr_route_diff_abs_mean": "",
    }
    if cache:
        field_to_key = {
            "radr_u_mean": "u_mean",
            "radr_u_std": "u_std",
            "radr_u_min": "u_min",
            "radr_u_max": "u_max",
            "radr_u_eff_mean": "u_eff_mean",
            "radr_u_eff_std": "u_eff_std",
            "radr_u_eff_min": "u_eff_min",
            "radr_u_eff_max": "u_eff_max",
            "radr_tau": "tau",
            "radr_lambda": "lambda",
            "radr_suppression_mean": "suppression_mean",
            "radr_corr_enabled": "corr_enabled",
            "radr_corr_lambda": "corr_lambda",
            "radr_corr_abs_mean": "corr_abs_mean",
            "radr_corr_abs_std": "corr_abs_std",
            "radr_corr_gated_abs_mean": "corr_gated_abs_mean",
            "radr_corr_gated_abs_std": "corr_gated_abs_std",
            "radr_corr_gate_mode_id": "corr_gate_mode_id",
            "radr_corr_gate_mean": "corr_gate_mean",
            "radr_corr_gate_std": "corr_gate_std",
            "radr_corr_gate_min": "corr_gate_min",
            "radr_corr_gate_max": "corr_gate_max",
            "radr_corr_feature_mode_id": "corr_feature_mode_id",
            "radr_corr_context_abs_mean": "corr_context_abs_mean",
            "radr_corr_context_abs_std": "corr_context_abs_std",
        }
        for field, key in field_to_key.items():
            tensor = cache.get(key, None)
            if tensor is not None and hasattr(tensor, "numel") and tensor.numel() > 0:
                values[field] = "{:.6f}".format(float(tensor.detach().float().view(-1)[0].cpu()))
    ccd_cache = getattr(net, "radr_ccd_cache", {})
    ccd_field_to_key = {
        "radr_ccd_delta_pred_abs": "delta_pred_abs_mean",
        "radr_ccd_num_delta": "num_delta",
    }
    for field, key in ccd_field_to_key.items():
        tensor = ccd_cache.get(key, None) if ccd_cache else None
        if tensor is not None and hasattr(tensor, "numel") and tensor.numel() > 0:
            values[field] = "{:.6f}".format(float(tensor.detach().float().view(-1)[0].cpu()))
    lch_cache = getattr(net, "radr_lch_debug_cache", {})
    lch_field_to_key = {
        "radr_lch_u_mean": "u_mean",
        "radr_lch_gate_mean": "gate_mean",
        "radr_lch_corr_abs_mean": "late_corr_abs_mean",
        "radr_lch_gated_corr_abs_mean": "gated_corr_abs_mean",
        "radr_lch_lambda": "lambda",
    }
    for field, key in lch_field_to_key.items():
        tensor = lch_cache.get(key, None) if lch_cache else None
        if tensor is not None and hasattr(tensor, "numel") and tensor.numel() > 0:
            values[field] = "{:.6f}".format(float(tensor.detach().float().view(-1)[0].cpu()))
    scdr_cache = getattr(net, "scdr_debug_cache", {})
    scdr_field_to_key = {
        "scdr_alpha": "alpha",
        "scdr_route_a_abs_mean": "route_a_abs_mean",
        "scdr_route_b_abs_mean": "route_b_abs_mean",
        "scdr_route_diff_abs_mean": "route_diff_abs_mean",
    }
    for field, key in scdr_field_to_key.items():
        tensor = scdr_cache.get(key, None) if scdr_cache else None
        if tensor is not None and hasattr(tensor, "numel") and tensor.numel() > 0:
            values[field] = "{:.6f}".format(float(tensor.detach().float().view(-1)[0].cpu()))
    return values


def discover_datasets(root):
    if os.path.isdir(os.path.join(root, "LR")) and os.path.isdir(os.path.join(root, "GT")):
        return [(os.path.basename(os.path.abspath(root)) or "val", root)]
    datasets = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(os.path.join(path, "LR")) and os.path.isdir(os.path.join(path, "GT")):
            datasets.append((name, path))
    return datasets


def has_direct_images(root):
    """只检查当前目录直属文件中是否存在图像，不递归子目录。"""
    # 中文注释：目录不存在时直接返回 False。
    if not os.path.isdir(root):
        return False
    # 中文注释：只遍历当前目录的直属文件，避免把 LR/Airport/xxx.png 算作根目录图像。
    for name in os.listdir(root):
        # 中文注释：拼出当前直属条目的完整路径。
        path = os.path.join(root, name)
        # 中文注释：复用项目图像后缀判断规则，确保和 list_images 识别范围一致。
        if os.path.isfile(path) and is_image_file(name):
            return True
    # 中文注释：当前目录没有直属图像文件。
    return False


def discover_folders(dataset_root):
    lr_root = os.path.join(dataset_root, "LR")
    gt_root = os.path.join(dataset_root, "GT")
    folders = []
    if has_direct_images(lr_root):
        folders.append("")
    for name in AID_CLASS_NAMES:
        if list_images(os.path.join(lr_root, name)) and os.path.isdir(os.path.join(gt_root, name)):
            folders.append(name)
    if folders:
        return folders
    for name in sorted(os.listdir(lr_root)):
        lr_dir = os.path.join(lr_root, name)
        gt_dir = os.path.join(gt_root, name)
        if os.path.isdir(lr_dir) and os.path.isdir(gt_dir) and list_images(lr_dir):
            folders.append(name)
    return folders or [""]


def evaluate_folder(model, dataset_name, dataset_root, folder, device, opt, epoch=0, logger=print, writer=None, progress=None):
    # 中文注释：LR 根目录用于生成跨平台稳定的 image_rel_path。
    lr_root = os.path.join(dataset_root, "LR")
    lr_dir = os.path.join(dataset_root, "LR", folder)
    gt_dir = os.path.join(dataset_root, "GT", folder)
    lr_images = list_images(lr_dir)
    if not lr_images:
        logger("[Warning] No LR images in {}".format(lr_dir))
        return None

    save_dir = os.path.join(opt.save_folder, "val_results", "epoch_{:04d}".format(epoch), dataset_name, folder)
    psnr_sum = 0.0
    ssim_sum = 0.0
    count = 0
    start = time.time()
    model.eval()
    # 中文注释：只有显式开启 --save_per_image_csv 时才收集逐图指标，默认不影响原验证流程。
    save_per_image_csv = bool(getattr(opt, "save_per_image_csv", False))
    # 中文注释：逐图记录先缓存在内存中，folder 评估完成后一次性写入 CSV。
    per_image_rows = []
    # 中文注释：CSV 字段固定，方便后续 analyze_scene_prior_gate.py 读取。
    per_image_fields = [
        "dataset", "folder", "scene", "image_rel_path", "psnr_scdr_radr", "ssim_scdr_radr",
        "struct_delta", "structure_score", "struct_edge_density", "struct_edge_top25_mean",
        "struct_lap_var", "struct_grad_coherence", "struct_hf_energy",
        "struct_delta_spatial_std", "struct_delta_spatial_min", "struct_delta_spatial_max",
        "struct_res_abs_mean", "struct_res_abs_std", "struct_res_to_feat_ratio",
        "radr_u_mean", "radr_u_std", "radr_u_min", "radr_u_max",
        "radr_u_eff_mean", "radr_u_eff_std", "radr_u_eff_min", "radr_u_eff_max",
        "radr_tau", "radr_lambda", "radr_suppression_mean",
        "radr_corr_enabled", "radr_corr_lambda", "radr_corr_abs_mean", "radr_corr_abs_std",
        "radr_corr_gated_abs_mean", "radr_corr_gated_abs_std",
        "radr_corr_gate_mode_id", "radr_corr_gate_mean", "radr_corr_gate_std",
        "radr_corr_gate_min", "radr_corr_gate_max",
        "radr_corr_feature_mode_id", "radr_corr_context_abs_mean", "radr_corr_context_abs_std",
        "radr_ccd_delta_pred_abs", "radr_ccd_num_delta",
        "radr_lch_u_mean", "radr_lch_gate_mean", "radr_lch_corr_abs_mean",
        "radr_lch_gated_corr_abs_mean", "radr_lch_lambda",
        "scdr_alpha", "scdr_route_a_abs_mean", "scdr_route_b_abs_mean",
        "scdr_route_diff_abs_mean",
        "decb_gate_mean", "decb_gate_std", "decb_gate_min", "decb_gate_max",
        "decb_err_abs_mean", "decb_attn_entropy", "decb_residual_abs_mean",
    ]

    # 中文注释：优先复用数据集级进度条；没有传入时才创建当前文件夹进度条。
    local_progress = None
    iterator = enumerate(lr_images, 1)
    if progress is None:
        local_progress = build_progress_bar(
            iterator,
            enable=bool(opt.val_progress),
            total=len(lr_images),
            desc="Val {} {}".format(dataset_name, folder or "root"),
            dynamic_ncols=True,
            leave=False,
        )
        iterator = local_progress

    for idx, lr_path in iterator:
        # 中文注释：数据集级进度条按已处理 LR 图像前进，即使后续 GT 缺失也不会卡住。
        if progress is not None:
            progress.update(1)

        lr = load_rgb_tensor(lr_path).unsqueeze(0).to(device, non_blocking=True)
        with torch.no_grad():
            with autocast_context(opt, device):
                sr = unwrap_prediction(model(lr))
            sr = sr.float().clamp_(0.0, 1.0)

        if opt.val_save_images:
            save_name = os.path.splitext(os.path.basename(lr_path))[0] + ".png"
            save_tensor_image(sr, os.path.join(save_dir, save_name))

        gt_path = find_matching_gt(gt_dir, lr_path)
        if gt_path is None:
            logger("[Warning] Missing GT for {}".format(lr_path))
            continue
        gt = load_rgb_tensor(gt_path).unsqueeze(0).to(device, non_blocking=True)
        if gt.shape != sr.shape:
            logger("[Warning] Shape mismatch pred {} gt {} for {}".format(tuple(sr.shape), tuple(gt.shape), lr_path))
            continue

        psnr_value = float(calc_psnr(sr, gt, border=opt.upscale_factor).item())
        ssim_value = float(calc_ssim(sr, gt, border=opt.upscale_factor).item())
        psnr_sum += psnr_value
        ssim_sum += ssim_value
        count += 1
        if save_per_image_csv:
            # 中文注释：scene 和 image_rel_path 与 bicubic 脚本共享同一解析规则。
            scene, image_rel_path = scene_and_rel_path(lr_path, lr_root, folder=folder)
            # 中文注释：数值使用固定小数输出，既可读也避免二进制浮点表现差异。
            row = {
                "dataset": dataset_name,
                "folder": folder or "root",
                "scene": scene,
                "image_rel_path": image_rel_path,
                "psnr_scdr_radr": "{:.6f}".format(psnr_value),
                "ssim_scdr_radr": "{:.6f}".format(ssim_value),
            }
            # 中文注释：如果模型开启 SCDRC，则追加当前图像的 delta 和结构统计；关闭时为空字段。
            struct_values = extract_struct_csv_values(model)
            # 中文注释：DECB-only 不开启 SCDRC 时也需要结构指标做 Q1-Q4 分箱，因此从 LR 即时补齐。
            fallback_struct_values = compute_structure_csv_values_from_lr(lr)
            # 中文注释：只填充 SCDRC cache 为空的结构指标，不覆盖已有 SCDRC 统计。
            for key, value in fallback_struct_values.items():
                if not struct_values.get(key, ""):
                    struct_values[key] = value
            # 中文注释：合并 SCDRC/结构统计字段。
            row.update(struct_values)
            # 中文注释：如果模型开启 RADR，则追加当前图像的 reliability predictor 诊断；关闭时为空字段。
            row.update(extract_radr_csv_values(model))
            # 中文注释：如果模型开启 DECB，则追加当前图像的 gate/error 诊断；关闭时为空字段。
            row.update(extract_decb_csv_values(model))
            # 中文注释：缓存当前图像的完整逐图记录。
            per_image_rows.append(row)

        if local_progress is not None:
            local_progress.set_postfix(
                psnr="{:.4f}".format(psnr_sum / max(1, count)),
                ssim="{:.4f}".format(ssim_sum / max(1, count)),
                refresh=False,
            )

        if opt.val_progress and progress is None and local_progress is not None and not getattr(local_progress, "using_tqdm", False) and (idx == len(lr_images) or idx % 20 == 0):
            elapsed = max(1e-6, time.time() - start)
            logger("Validation {} {}: {}/{} ({:.2f} img/s)".format(dataset_name, folder or "root", idx, len(lr_images), idx / elapsed))

    if count == 0:
        return None
    if save_per_image_csv:
        # 中文注释：epoch 目录与已有 val_results 保存规则保持一致。
        epoch_dir = os.path.join(opt.save_folder, "val_results", "epoch_{:04d}".format(epoch))
        # 中文注释：局部 CSV 放在当前 folder 的结果目录下，便于按场景单独检查。
        local_csv = os.path.join(save_dir, "per_image.csv")
        # 中文注释：全局 CSV 放在 epoch 根目录，跨 folder 追加为一个 dataset 级文件。
        global_csv = os.path.join(epoch_dir, "{}_per_image_all.csv".format(dataset_name))
        # 中文注释：局部 CSV 每个 folder 独立覆盖写入，不影响其他 folder。
        write_csv_rows(local_csv, per_image_fields, per_image_rows, append=False)
        # 中文注释：全局 CSV 追加写入；write_csv_rows 会自动避免重复表头。
        write_csv_rows(global_csv, per_image_fields, per_image_rows, append=True)
    metrics = {"psnr": psnr_sum / count, "ssim": ssim_sum / count, "count": count}
    if getattr(opt, "val_folder_verbose", False):
        logger("====> Folder [{}] PSNR {:.4f} dB | SSIM {:.4f} | Images {}".format(
            folder or "Root Directory", metrics["psnr"], metrics["ssim"], count
        ))
    return metrics


def is_aid_dataset(dataset_name, folders):
    # 中文注释：只对 AID/AID900 这类按场景组织的数据集打印场景级结果。
    name = str(dataset_name).lower()
    if "aid" in name:
        return True
    aid_folder_set = set(AID_CLASS_NAMES)
    return any(folder in aid_folder_set for folder in folders)


def run_validation(model, val_root, device, opt, epoch=0, logger=print, writer=None, label=""):
    if not val_root or not os.path.isdir(val_root):
        logger("[Warning] Validation directory not found: {}".format(val_root))
        return None, None, 0, {}

    os.makedirs(opt.save_folder, exist_ok=True)
    metrics_path = os.path.join(opt.save_folder, opt.val_metrics_file)
    datasets = discover_datasets(val_root)
    if not datasets:
        logger("[Warning] No datasets with LR/GT found under {}".format(val_root))
        return None, None, 0, {}

    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write("\n{}\nEpoch {} Validation {}\n{}\n".format("=" * 80, epoch, label, "=" * 80))

    global_psnr_sum = 0.0
    global_ssim_sum = 0.0
    global_count = 0
    dataset_avg_psnr_sum = 0.0
    dataset_avg_ssim_sum = 0.0
    dataset_avg_count = 0
    dataset_metrics = {}

    for dataset_name, dataset_root in datasets:
        if getattr(opt, "val_dataset_verbose", False):
            logger("\nStarting Evaluation for Dataset: {}".format(dataset_name))
        # 中文注释：如果开启逐图 CSV 导出，则每次 run_validation 开始时清理当前 epoch 的旧全局 CSV，避免重复 eval 追加重复行。
        if getattr(opt, "save_per_image_csv", False):
            epoch_dir = os.path.join(opt.save_folder, "val_results", "epoch_{:04d}".format(epoch))
            global_csv = os.path.join(epoch_dir, "{}_per_image_all.csv".format(dataset_name))
            if os.path.isfile(global_csv):
                os.remove(global_csv)
        dataset_psnr_sum = 0.0
        dataset_ssim_sum = 0.0
        dataset_count = 0
        folders = discover_folders(dataset_root)
        show_aid_scene_metrics = bool(getattr(opt, "val_aid_scene_verbose", True) and is_aid_dataset(dataset_name, folders))
        # 中文注释：按数据集创建一条 tqdm 进度条，而不是按 AID 类别反复创建进度条。
        dataset_total_images = 0
        for folder in folders:
            dataset_total_images += len(list_images(os.path.join(dataset_root, "LR", folder)))
        dataset_progress = build_progress_bar(
            range(dataset_total_images),
            enable=bool(opt.val_progress and dataset_total_images > 0),
            total=dataset_total_images,
            desc="Val {}".format(dataset_name),
            dynamic_ncols=True,
            leave=False,
        )

        for folder in folders:
            metrics = evaluate_folder(
                model, dataset_name, dataset_root, folder, device, opt, epoch, logger, writer,
                progress=dataset_progress,
            )
            if metrics is None:
                continue
            if show_aid_scene_metrics and folder:
                logger("AID Scene [{}] PSNR {:.4f} dB | SSIM {:.4f} | Images {}".format(
                    folder, metrics["psnr"], metrics["ssim"], metrics["count"]
                ))
            dataset_psnr_sum += metrics["psnr"] * metrics["count"]
            dataset_ssim_sum += metrics["ssim"] * metrics["count"]
            dataset_count += metrics["count"]
            if dataset_count > 0:
                dataset_progress.set_postfix(
                    psnr="{:.4f}".format(dataset_psnr_sum / max(1, dataset_count)),
                    ssim="{:.4f}".format(dataset_ssim_sum / max(1, dataset_count)),
                    refresh=False,
                )

        dataset_progress.close()

        if dataset_count > 0:
            avg_psnr = dataset_psnr_sum / dataset_count
            avg_ssim = dataset_ssim_sum / dataset_count
            dataset_metrics[dataset_name] = {"psnr": avg_psnr, "ssim": avg_ssim, "count": dataset_count}
            if getattr(opt, "val_dataset_verbose", False):
                logger("VAL DATASET {:<12} | PSNR {:>7.4f} dB | SSIM {:>7.4f} | N {:>5}".format(
                    dataset_name, avg_psnr, avg_ssim, dataset_count
                ))
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write("{}: PSNR={:.4f} dB, SSIM={:.4f}, N={}\n".format(dataset_name, avg_psnr, avg_ssim, dataset_count))
            global_psnr_sum += dataset_psnr_sum
            global_ssim_sum += dataset_ssim_sum
            global_count += dataset_count
            # 中文注释：额外累计数据集级均值，用于计算 AID/DIOR/DOTA 等权总平均。
            dataset_avg_psnr_sum += avg_psnr
            dataset_avg_ssim_sum += avg_ssim
            dataset_avg_count += 1

    if global_count == 0:
        logger("Validation finished, but no valid image pairs were found.")
        return None, None, 0, dataset_metrics

    global_psnr = global_psnr_sum / global_count
    global_ssim = global_ssim_sum / global_count
    # 中文注释：Dataset Average 是当前主指标；Image-weighted 只写入文件，避免控制台刷屏。
    if dataset_avg_count > 0:
        dataset_mean_psnr = dataset_avg_psnr_sum / dataset_avg_count
        dataset_mean_ssim = dataset_avg_ssim_sum / dataset_avg_count
        logger("=" * 72)
        logger("VAL SUMMARY epoch {}{}".format(epoch, " [{}]".format(label) if label else ""))
        logger(">>> Dataset Average PSNR: {:.4f} dB".format(dataset_mean_psnr))
        logger(">>> Dataset Average SSIM: {:.4f}".format(dataset_mean_ssim))
        logger("=" * 72)
    else:
        dataset_mean_psnr = None
        dataset_mean_ssim = None
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write("Image-weighted Overall Average PSNR: {:.4f} dB\n".format(global_psnr))
        f.write("Image-weighted Overall Average SSIM: {:.4f}\n".format(global_ssim))
        if dataset_mean_psnr is not None:
            f.write("Dataset Average PSNR: {:.4f} dB\n".format(dataset_mean_psnr))
            f.write("Dataset Average SSIM: {:.4f}\n".format(dataset_mean_ssim))
        f.write("Total Evaluated Images: {}\n".format(global_count))
    if writer is not None:
        writer.add_scalar("Val/Overall_PSNR", global_psnr, epoch)
        writer.add_scalar("Val/Overall_SSIM", global_ssim, epoch)
        if dataset_mean_psnr is not None:
            writer.add_scalar("Val/Dataset_Average_PSNR", dataset_mean_psnr, epoch)
            writer.add_scalar("Val/Dataset_Average_SSIM", dataset_mean_ssim, epoch)
    # 中文注释：返回 Dataset Average 作为训练选择 best checkpoint 的主指标。
    return dataset_mean_psnr, dataset_mean_ssim, global_count, dataset_metrics
