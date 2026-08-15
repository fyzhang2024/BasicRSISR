"""Checkpoint helpers for SCDR_RADR training."""
import os

import torch


def _log(logger, message):
    if logger is None:
        print(message)
    else:
        logger(message)


def load_torch_file(path, map_location="cpu"):
    # 中文注释：优先使用 weights_only=True；旧版 torch 不支持时回退。
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def strip_module_prefix(state_dict):
    # 中文注释：去掉 DataParallel 保存时的 module. 前缀。
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def _is_decb_missing_key(key):
    """判断 missing key 是否属于 DECB 新增模块。"""
    # 中文注释：DECB 通常注册在 block 内，因此 key 中会包含 ".decb_branch."。
    if ".decb_branch." in key:
        return True
    # 中文注释：兼容顶层直接注册 decb_branch 的情况。
    if key.startswith("decb_branch."):
        return True
    # 中文注释：其他 key 不能按 DECB 规则放行。
    return False


def _is_radr_missing_key(key):
    """判断 missing key 是否属于 RADR 新增模块。"""
    return "radr_predictor" in str(key)


def _is_radr_correction_missing_key(key):
    """判断 missing key 是否属于 RC2-RADR correction branch。"""
    return "radr_correction" in str(key)


def _is_radr_ccd_missing_key(key):
    """判断 missing key 是否属于 CCD-RADR auxiliary head。"""
    return "radr_ccd_aux_head" in str(key)


def _is_radr_lch_missing_key(key):
    """判断 missing key 是否属于 LCH-RADR late correction head。"""
    return "radr_lch" in str(key)


def _is_scdr_adapter_missing_key(key):
    """判断 missing key 是否属于 SCDR-v2 route-specific adapter。"""
    return "scdr_adapter" in str(key)


def _is_existing_allowed_missing_key(key):
    """判断 missing key 是否属于已有可 warm-start 的新增模块。"""
    # 中文注释：route_bias_alpha 是方案 B 新增的 FDG 可选参数（只有在
    # use_fdg + fdg_use_acmsa_route_bias 同时打开时才会注册）。老 checkpoint
    # 自然不会有，缺失时使用构造时的 route_bias_init（默认 0，等价于关闭）。
    allowed_missing_tokens = (
        "freq_grouping",
        "freq_map_builder",
        "route_bias_alpha",
        "struct_extractor",
        "struct_controller",
        "local_struct_extractor",
        "local_struct_controller",
    )
    # 中文注释：只保留已有新增模块 allowlist，不允许静默忽略全部 missing key。
    return any(token in key for token in allowed_missing_tokens)


def _load_model_state_compatible(target_model, state, logger=None, label="model"):
    # 中文注释：strict=False 允许从 polished baseline checkpoint 恢复到带 FDG 的模型。
    missing, unexpected = target_model.load_state_dict(state, strict=False)
    missing = list(missing)
    unexpected = list(unexpected)
    # 中文注释：兼容 DataParallel，读取真实模型上的新增模块开关。
    net = target_model.module if hasattr(target_model, "module") else target_model
    net_ref = net.module if hasattr(net, "module") else net
    # 中文注释：只有当前模型确实开启 DECB，才允许 DECB 新增参数缺失。
    allow_decb_missing = bool(getattr(net_ref, "use_decb", False))
    # 中文注释：只有当前模型开启 RADR，才允许 reliability predictor 缺失。
    allow_radr_missing = bool(getattr(net_ref, "use_radr", False))
    # 中文注释：只有当前模型开启 RC2-RADR correction，才允许 correction branch 缺失。
    allow_radr_correction_missing = bool(
        getattr(net_ref, "use_radr", False)
        and getattr(net_ref, "radr_use_correction", False)
    )
    # 中文注释：只有当前模型开启 CCD-RADR，才允许新增 auxiliary head 缺失。
    allow_radr_ccd_missing = bool(
        getattr(net_ref, "use_radr", False)
        and getattr(net_ref, "radr_use_correction", False)
        and getattr(net_ref, "use_radr_ccd", False)
    )
    # 中文注释：只有当前模型开启 LCH-RADR，才允许模型级 late head 缺失。
    allow_radr_lch_missing = bool(
        getattr(net_ref, "use_radr", False)
        and getattr(net_ref, "use_radr_lch", False)
    )
    # 中文注释：只有当前模型开启 SCDR-v2 adapter，才允许 route adapter 缺失。
    allow_scdr_adapter_missing = bool(
        getattr(net_ref, "use_scdr", False)
        and getattr(net_ref, "use_scdr_adapter", False)
    )
    # 中文注释：单独记录 DECB missing key，用于打印明确 warm-start 提示。
    decb_missing = []
    # 中文注释：单独记录 RADR missing key，用于打印明确 warm-start 提示。
    radr_missing = []
    # 中文注释：单独记录 RC2-RADR correction missing key。
    radr_correction_missing = []
    # 中文注释：单独记录 CCD-RADR auxiliary head missing key。
    radr_ccd_missing = []
    # 中文注释：单独记录 LCH-RADR late head missing key。
    radr_lch_missing = []
    # 中文注释：单独记录 SCDR-v2 route adapter missing key。
    scdr_adapter_missing = []
    # 中文注释：其他非 allowlist missing key 仍然视为结构不兼容。
    bad_missing = []
    # 中文注释：逐项检查 missing key，避免放开所有缺失参数。
    for key in missing:
        # 中文注释：DECB 是新 error compensation branch，老 SCDR_RADR baseline 没有这些参数是预期的。
        if allow_decb_missing and _is_decb_missing_key(key):
            decb_missing.append(key)
            continue
        # 中文注释：RADR 是新 reliability predictor，老 SCDR_RADR baseline 没有这些参数是预期的。
        if allow_radr_missing and _is_radr_missing_key(key):
            radr_missing.append(key)
            continue
        # 中文注释：RC2-RADR correction 是 v1.2 新增分支，旧 RADR/SCDR_RADR checkpoint 没有是预期的。
        if allow_radr_correction_missing and _is_radr_correction_missing_key(key):
            radr_correction_missing.append(key)
            continue
        # 中文注释：CCD-RADR auxiliary head 是训练辅助分支，旧 RC2/RADR checkpoint 没有是预期的。
        if allow_radr_ccd_missing and _is_radr_ccd_missing_key(key):
            radr_ccd_missing.append(key)
            continue
        # 中文注释：LCH-RADR 是模型级 late correction head，旧 RC2/RADR checkpoint 没有是预期的。
        if allow_radr_lch_missing and _is_radr_lch_missing_key(key):
            radr_lch_missing.append(key)
            continue
        # 中文注释：SCDR-v2 adapter 是 route-specific 新增模块，旧 SCDR-v1 checkpoint 没有是预期的。
        if allow_scdr_adapter_missing and _is_scdr_adapter_missing_key(key):
            scdr_adapter_missing.append(key)
            continue
        # 中文注释：保留原有 FDG / SCDRC / Local SCDRC 的 warm-start allowlist。
        if _is_existing_allowed_missing_key(key):
            continue
        # 中文注释：非 DECB 且非已有 allowlist 的 key 不能放行。
        bad_missing.append(key)
    if bad_missing:
        raise RuntimeError(
            "{} checkpoint missing non-allowed keys: {}".format(label, bad_missing[:20])
        )
    bad_unexpected = list(unexpected)
    if bad_unexpected:
        raise RuntimeError(
            "{} checkpoint has unexpected keys: {}".format(label, bad_unexpected[:20])
        )
    if decb_missing:
        _log(logger, "DECB enabled: missing error compensation keys are expected when warm-starting from SCDR_RADR baseline.")
    if radr_missing:
        _log(logger, "RADR enabled: missing reliability predictor keys are expected when warm-starting from SCDR_RADR baseline.")
    if radr_correction_missing:
        _log(logger, "RADR correction enabled: missing correction keys are expected when warm-starting from RADR-v1.1 or SCDR_RADR baseline.")
    if radr_ccd_missing:
        _log(logger, "RADR CCD enabled: missing CCD aux head keys are expected when warm-starting from RC2-RADR/RADR checkpoint.")
    if radr_lch_missing:
        _log(logger, "RADR LCH enabled: missing late correction head keys are expected when warm-starting from RC2-RADR/RADR checkpoint.")
    if scdr_adapter_missing:
        _log(logger, "SCDR adapter enabled: missing route adapter keys are expected when warm-starting from SCDR-v1/RADR checkpoint.")
    _log(
        logger,
        "{} load_state_dict strict=False | missing {} | unexpected {}".format(
            label, len(missing), len(unexpected)
        ),
    )
    if missing:
        _log(logger, "{} missing keys sample: {}".format(label, missing[:20]))
    if unexpected:
        _log(logger, "{} unexpected keys sample: {}".format(label, unexpected[:20]))
    return missing, unexpected


def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=0,
                    best_psnr=0.0, best_ssim=0.0, best_epoch=0, ema_model=None, extra=None):
    # 中文注释：统一保存训练进度、模型、优化器、调度器和可选 EMA。
    os.makedirs(os.path.dirname(path), exist_ok=True)
    target_model = model.module if hasattr(model, "module") else model
    payload = {
        "epoch": int(epoch),
        "best_psnr": float(best_psnr),
        "best_ssim": float(best_ssim),
        "best_epoch": int(best_epoch),
        "model": target_model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if ema_model is not None:
        payload["ema_model"] = ema_model.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def resume_checkpoint(model, optimizer, scheduler, ckpt_path, logger=None, ema_model=None):
    # 中文注释：完整恢复 model/optimizer/scheduler/EMA，并返回历史 best 信息。
    checkpoint = load_torch_file(ckpt_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    state = strip_module_prefix(state)
    target_model = model.module if hasattr(model, "module") else model
    _load_model_state_compatible(target_model, state, logger=logger, label="model")
    if optimizer is not None and "optimizer" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except ValueError as exc:
            _log(logger, "Skip optimizer state due to incompatible parameter groups: {}".format(exc))
    if scheduler is not None and "scheduler" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler"])
        except Exception as exc:
            _log(logger, "Skip scheduler state due to incompatible checkpoint state: {}".format(exc))
    if ema_model is not None and "ema_model" in checkpoint:
        ema_state = strip_module_prefix(checkpoint["ema_model"])
        target_ema = ema_model.module if hasattr(ema_model, "module") else ema_model
        _load_model_state_compatible(target_ema, ema_state, logger=logger, label="ema_model")
    epoch = int(checkpoint.get("epoch", 0))
    best_psnr = float(checkpoint.get("best_psnr", 0.0))
    best_ssim = float(checkpoint.get("best_ssim", 0.0))
    best_epoch = int(checkpoint.get("best_epoch", 0))
    _log(
        logger,
        "Resumed checkpoint {} at epoch {}, best_psnr {:.4f}, best_ssim {:.4f}, best_epoch {}".format(
            ckpt_path, epoch, best_psnr, best_ssim, best_epoch
        ),
    )
    return epoch, best_psnr, best_ssim, best_epoch


def resume_model_only_checkpoint(model, ckpt_path, logger=None, ema_model=None):
    # 中文注释：只恢复模型/EMA 权重和历史 best，不恢复 optimizer/scheduler。
    checkpoint = load_torch_file(ckpt_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    state = strip_module_prefix(state)
    target_model = model.module if hasattr(model, "module") else model
    _load_model_state_compatible(target_model, state, logger=logger, label="model")

    ema_state = None
    if isinstance(checkpoint, dict):
        ema_state = checkpoint.get("ema_model", checkpoint.get("ema", None))
    if ema_model is not None and ema_state is not None:
        ema_state = strip_module_prefix(ema_state)
        target_ema = ema_model.module if hasattr(ema_model, "module") else ema_model
        _load_model_state_compatible(target_ema, ema_state, logger=logger, label="ema_model")

    epoch = int(checkpoint.get("epoch", 0))
    best_psnr = float(checkpoint.get("best_psnr", 0.0))
    best_ssim = float(checkpoint.get("best_ssim", 0.0))
    best_epoch = int(checkpoint.get("best_epoch", 0))
    _log(logger, "Resume model only; optimizer and scheduler are reset.")
    _log(
        logger,
        "Resumed model weights {} at epoch {}, best_psnr {:.4f}, best_ssim {:.4f}, best_epoch {}".format(
            ckpt_path, epoch, best_psnr, best_ssim, best_epoch
        ),
    )
    return epoch, best_psnr, best_ssim, best_epoch
