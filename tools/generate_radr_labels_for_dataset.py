"""Generate HR-derived RADR labels for a dataset split.

This script reuses the existing RADR label formula from
scripts/build_radr_labels.py:

    help_map = err_nodict - err_full
    u_gt, valid = build_label_from_help(help_map, tau_neg=tau, tau_pos=tau)

Example:
    python tools/generate_radr_labels_for_dataset.py \
        --data_root datasets/test \
        --dataset AID900 \
        --atd_ckpt saved_models/atd_best.pth \
        --save_root saved_models/radr_labels/test_AID900_tau1e4 \
        --device cuda \
        --patch_size 32 \
        --tau 1e-4
"""

import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.analyze_dict_residual_help import (  # noqa: E402
    build_baseline_model,
    discover_lr_images,
    find_gt_for_lr,
    load_baseline_checkpoint,
    safe_vis_name,
    set_zero_dict_residual,
)
from scripts.build_radr_labels import (  # noqa: E402
    build_label_from_help,
    gt_hw_from_path,
    infer_full_help_map,
    normalize_rel_path,
    recompute_help_map,
)
from utils.checkpoint_utils import _load_model_state_compatible, load_torch_file, strip_module_prefix  # noqa: E402
from utils.image_utils import load_rgb_tensor  # noqa: E402
from utils.scene_gate_utils import image_rel_path_from_lr  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate RADR labels for a dataset split")
    parser.add_argument("--data_root", type=str, required=True,
                        help="root that either contains DATASET/{LR,GT} or directly contains LR/GT")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--atd_ckpt", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--patch_size", type=int, default=32,
                        help="kept for experiment naming/metadata; labels are generated pixel-wise")
    parser.add_argument("--tau", type=float, default=1e-4)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--crop_border", type=int, default=None,
                        help="default equals --scale, matching scripts/build_radr_labels.py")
    parser.add_argument("--config", type=str, default="",
                        help="optional training args/config JSON/YAML for non-default ATD checkpoints")
    parser.add_argument("--max_images", type=int, default=0,
                        help="debug/smoke-test limit; 0 means all images")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--amp", type=int, default=1)
    return parser.parse_args(argv)


def log(message):
    print("[radr-labels] {}".format(message), flush=True)


def resolve_dataset_root(data_root, dataset):
    data_root = os.path.abspath(data_root)
    if os.path.isdir(os.path.join(data_root, "LR")) and os.path.isdir(os.path.join(data_root, "GT")):
        return data_root
    candidate = os.path.join(data_root, dataset)
    if os.path.isdir(os.path.join(candidate, "LR")) and os.path.isdir(os.path.join(candidate, "GT")):
        return os.path.abspath(candidate)
    raise FileNotFoundError("Expected LR/ and GT/ under {} or {}".format(data_root, candidate))


def load_config_dict(path):
    if not path:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError("config not found: {}".format(path))
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
            data = yaml.safe_load(text)
        except Exception:
            data = parse_simple_yaml(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("config must decode to a dict: {}".format(path))
    return flatten_config(data)


def parse_simple_yaml(text):
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and value:
            out[key] = parse_scalar(value)
    return out


def parse_scalar(value):
    low = str(value).lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    try:
        if any(ch in str(value) for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def flatten_config(data, prefix=""):
    flat = {}
    for key, value in data.items():
        name = str(key)
        full = "{}_{}".format(prefix, name) if prefix else name
        if isinstance(value, dict):
            flat.update(flatten_config(value, prefix=full))
            flat.update(flatten_config(value, prefix=""))
        else:
            flat[full] = value
    return flat


def build_model_from_config(config_path, scale, device):
    config = load_config_dict(config_path)
    for tuple_key in ("atd_depths", "atd_num_heads"):
        if isinstance(config.get(tuple_key, None), (list, tuple)):
            config[tuple_key] = ",".join(str(int(v)) for v in config[tuple_key])
    return build_baseline_model(scale, config=config).to(device)


def load_checkpoint_compatible(model, ckpt_path):
    checkpoint = load_torch_file(ckpt_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    state = strip_module_prefix(state)
    target = model.module if hasattr(model, "module") else model
    _load_model_state_compatible(target, state, logger=log, label="atd")


def build_and_load_model(args, device):
    if args.config:
        model = build_model_from_config(args.config, args.scale, device)
        load_checkpoint_compatible(model, args.atd_ckpt)
    else:
        # model = build_baseline_model(args.scale)
        model = build_baseline_model(
            args.scale,
            config={
                "atd_depths": "6,6,6,6,6,6"
            }
        )

        load_baseline_checkpoint(model, args.atd_ckpt)
        model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def write_index_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_rel_path", "label_path", "valid_path"])
        writer.writeheader()
        writer.writerows(rows)


def stats(arr):
    arr = np.asarray(arr)
    return {
        "shape": list(arr.shape),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def verify_random_records(rows, dataset_root, seed=123, k=5):
    lr_root = os.path.join(dataset_root, "LR")
    gt_root = os.path.join(dataset_root, "GT")
    if not rows:
        raise RuntimeError("No rows to verify")
    rng = random.Random(int(seed))
    selected = rng.sample(rows, min(int(k), len(rows)))
    checks = []
    for row in selected:
        rel_path = normalize_rel_path(row["image_rel_path"])
        lr_path = os.path.join(lr_root, rel_path.replace("/", os.sep))
        gt_path = find_gt_for_lr(lr_path, lr_root, gt_root)
        label_path = row["label_path"]
        valid_path = row["valid_path"]
        item = {
            "image_rel_path": rel_path,
            "lr_exists": os.path.isfile(lr_path),
            "gt_exists": bool(gt_path and os.path.isfile(gt_path)),
            "label_exists": os.path.isfile(label_path),
            "valid_exists": os.path.isfile(valid_path),
        }
        if not all(item[key] for key in ("lr_exists", "gt_exists", "label_exists", "valid_exists")):
            raise RuntimeError("Verification failed path existence check: {}".format(item))
        u_gt = np.load(label_path)
        valid = np.load(valid_path)
        if u_gt.shape != valid.shape:
            raise RuntimeError("u_gt/valid shape mismatch for {}: {} vs {}".format(rel_path, u_gt.shape, valid.shape))
        if not np.isfinite(u_gt).all() or not np.isfinite(valid).all():
            raise RuntimeError("Non-finite label values for {}".format(rel_path))
        if float(np.min(u_gt)) < 0.0 or float(np.max(u_gt)) > 1.0:
            raise RuntimeError("u_gt outside [0,1] for {}".format(rel_path))
        if float(np.min(valid)) < 0.0 or float(np.max(valid)) > 1.0:
            raise RuntimeError("valid outside [0,1] for {}".format(rel_path))
        item["u_gt"] = stats(u_gt)
        item["valid"] = stats(valid)
        checks.append(item)
    return checks


def main(argv=None):
    args = parse_args(argv)
    dataset_root = resolve_dataset_root(args.data_root, args.dataset)
    lr_root = os.path.join(dataset_root, "LR")
    gt_root = os.path.join(dataset_root, "GT")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        log("CUDA requested but unavailable; falling back to CPU")
        device = torch.device("cpu")
    amp_enabled = bool(int(args.amp)) and device.type == "cuda"
    crop_border = int(args.scale if args.crop_border is None else args.crop_border)
    save_label_dir = os.path.abspath(os.path.join(args.save_root, "labels"))
    os.makedirs(save_label_dir, exist_ok=True)

    model = build_and_load_model(args, device)
    lr_paths = discover_lr_images(lr_root)
    if args.max_images and int(args.max_images) > 0:
        lr_paths = lr_paths[:int(args.max_images)]
    if not lr_paths:
        raise RuntimeError("No LR images found under {}".format(lr_root))

    rows = []
    summary = {
        "dataset": args.dataset,
        "dataset_root": dataset_root,
        "atd_ckpt": args.atd_ckpt,
        "config": args.config,
        "tau": float(args.tau),
        "patch_size": int(args.patch_size),
        "crop_border": crop_border,
        "num_images": 0,
        "u_gt_mean": [],
        "valid_mean": [],
    }

    for idx, lr_path in enumerate(lr_paths, start=1):
        gt_path = find_gt_for_lr(lr_path, lr_root, gt_root)
        if gt_path is None:
            raise FileNotFoundError("Unable to find GT pair for {}".format(lr_path))
        rel_path = normalize_rel_path(image_rel_path_from_lr(lr_path, lr_root))
        safe_name = safe_vis_name(rel_path)
        full_hw = gt_hw_from_path(gt_path)
        lr = load_rgb_tensor(lr_path).unsqueeze(0).to(device)
        gt = load_rgb_tensor(gt_path).unsqueeze(0).to(device)

        set_zero_dict_residual(model, False)
        help_arr = recompute_help_map(model, lr, gt, device, amp_enabled, rel_path)

        help_full, region_valid = infer_full_help_map(help_arr, full_hw, crop_border, rel_path)
        u_gt, valid = build_label_from_help(help_full, region_valid, args.tau, args.tau)
        label_path = os.path.abspath(os.path.join(save_label_dir, "{}_u_gt.npy".format(safe_name)))
        valid_path = os.path.abspath(os.path.join(save_label_dir, "{}_valid.npy".format(safe_name)))
        np.save(label_path, u_gt.astype(np.float32))
        np.save(valid_path, valid.astype(np.float32))
        rows.append({
            "image_rel_path": rel_path,
            "label_path": label_path,
            "valid_path": valid_path,
        })
        summary["num_images"] += 1
        summary["u_gt_mean"].append(float(np.mean(u_gt)))
        summary["valid_mean"].append(float(np.mean(valid)))
        if idx % 20 == 0 or idx == len(lr_paths):
            log("processed {}/{}".format(idx, len(lr_paths)))

    set_zero_dict_residual(model, False)
    index_csv = os.path.abspath(os.path.join(args.save_root, "radr_label_index_{}.csv".format(args.dataset)))
    write_index_csv(index_csv, rows)
    checks = verify_random_records(rows, dataset_root, seed=args.seed, k=5)
    summary_out = dict(summary)
    for key in ("u_gt_mean", "valid_mean"):
        values = summary[key]
        summary_out[key] = float(np.mean(values)) if values else 0.0
    summary_out["index_csv"] = index_csv
    summary_out["label_dir"] = save_label_dir
    summary_out["random_checks"] = checks
    summary_json = os.path.abspath(os.path.join(args.save_root, "radr_label_summary_{}.json".format(args.dataset)))
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2, ensure_ascii=False)

    log("saved labels: {}".format(save_label_dir))
    log("saved index csv: {}".format(index_csv))
    log("saved summary: {}".format(summary_json))
    log("random verification records:")
    for item in checks:
        log(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()