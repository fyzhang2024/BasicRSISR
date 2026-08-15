import os

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def is_image_file(path):
    return path.lower().endswith(IMAGE_EXTS)


def load_rgb_tensor(path):
    return to_tensor(Image.open(path).convert("RGB"))


def save_tensor_image(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = tensor.detach().float().clamp(0.0, 1.0).squeeze(0)
    img = img.mul(255.0).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(img).save(path)


def list_images(path):
    if not os.path.isdir(path):
        return []
    return sorted(
        os.path.join(path, name)
        for name in os.listdir(path)
        if os.path.isfile(os.path.join(path, name)) and is_image_file(name)
    )


def find_matching_gt(gt_dir, lr_path):
    base = os.path.basename(lr_path)
    direct = os.path.join(gt_dir, base)
    if os.path.isfile(direct):
        return direct
    stem = os.path.splitext(base)[0]
    for ext in IMAGE_EXTS:
        candidate = os.path.join(gt_dir, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    return None
